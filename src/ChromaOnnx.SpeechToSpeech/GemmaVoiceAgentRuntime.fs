namespace ChromaOnnx.SpeechToSpeech

open System
open System.Collections.Concurrent
open System.Collections.Generic
open System.Diagnostics
open System.Globalization
open System.IO
open System.Text.Json
open System.Threading
open System.Threading.Tasks
open ChromaOnnx

type private VoiceTurn =
    { TurnIndex: int
      RequestId: string
      Transcript: string
      FinalText: string
      ToolCalls: AgentToolCallInfo array
      ToolResults: AgentToolResultInfo array
      WorkDir: string }

type private VoiceSession =
    { Id: string
      SystemPrompt: string
      Mode: string
      WorkDir: string
      CreatedUtc: DateTimeOffset
      SyncRoot: obj
      Turns: ResizeArray<VoiceTurn>
      mutable NextTurnIndex: int }

type private VoiceSessionStore(workDir: string) =
    let sessions = ConcurrentDictionary<string, VoiceSession>(StringComparer.Ordinal)

    let toInfo (session: VoiceSession) =
        { Id = session.Id
          ServiceName = "GemmaVoiceAgent"
          Mode = session.Mode
          SystemPrompt = session.SystemPrompt
          WebsocketUrl = $"/ws/agent/{session.Id}"
          CreatedUtc = session.CreatedUtc }

    member _.Create(systemPrompt: string, mode: string) =
        let id = Guid.NewGuid().ToString("N")
        let sessionWorkDir = Path.Combine(workDir, id)
        Directory.CreateDirectory(sessionWorkDir) |> ignore
        let session =
            { Id = id
              SystemPrompt = systemPrompt
              Mode = mode
              WorkDir = sessionWorkDir
              CreatedUtc = DateTimeOffset.UtcNow
              SyncRoot = obj()
              Turns = ResizeArray<VoiceTurn>()
              NextTurnIndex = 1 }
        sessions[id] <- session
        session

    member _.TryGet(id: string) =
        match sessions.TryGetValue id with
        | true, session -> Some session
        | false, _ -> None

    member this.TryGetInfo(id: string) =
        this.TryGet id |> Option.map toInfo

    member _.ToInfo(session: VoiceSession) = toInfo session

type private GemmaSttRuntime(gemma: IGemmaRuntime, maxTokens: int, maxAudioSeconds: float) =
    let resampleLinear (samples: float32 array) sourceRate targetRate =
        if sourceRate = targetRate || samples.Length = 0 then
            Array.copy samples
        else
            let length = max 1 (int (Math.Round(float samples.Length * float targetRate / float sourceRate)))
            let output = Array.zeroCreate<float32> length
            let scale = float sourceRate / float targetRate
            for i in 0 .. length - 1 do
                let source = float i * scale
                let left = int (Math.Floor source)
                let right = min (left + 1) (samples.Length - 1)
                let mix = source - float left
                output[i] <- samples[left] * float32 (1.0 - mix) + samples[right] * float32 mix
            output

    let prompt =
        "Transcribe the following speech segment in its original language. Follow these specific instructions for formatting the answer:\n* Only output the transcription, with no newlines.\n* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three.\n\n<|audio|>"

    let cleanTranscript (text: string) =
        (if Object.ReferenceEquals(text, null) then "" else text)
            .Replace("\r", " ")
            .Replace("\n", " ")
            .Trim()

    interface ISttRuntime with
        member _.Status() =
            let status = gemma.Status()
            { Ready = status.Ready
              Runtime = "gemma4-audio"
              InputSampleRate = 24000
              OutputLanguage = "auto"
              Message = if status.Ready then "Gemma audio transcription is ready." else status.Message }

        member _.TranscribeAsync(samples24k, _outputDirectory, cancellationToken) =
            task {
                let stopwatch = Stopwatch.StartNew()
                let truncated =
                    let maxSamples = int (Math.Ceiling(maxAudioSeconds * 24000.0))
                    if samples24k.Length > maxSamples then samples24k[0 .. maxSamples - 1] else samples24k
                let userAudio16k = resampleLinear truncated 24000 16000
                let! asr =
                    gemma.GenerateAsync(
                        { Messages = [| GemmaChatMessage.user prompt |]
                          Tools = Array.empty
                          Audio16k = Some userAudio16k
                          AddGenerationPrompt = true
                          MaxNewTokens = maxTokens
                          Temperature = 0.0
                          TopP = 1.0
                          TopK = 0 },
                        cancellationToken)
                stopwatch.Stop()
                return
                    { Transcript = cleanTranscript asr.Text
                      InputSampleRate = 24000
                      InputSamples = truncated.Length
                      DurationMs = stopwatch.Elapsed.TotalMilliseconds
                      Message = $"Gemma ASR stop reason: {asr.StopReason}" }
            }

type GemmaVoiceAgentRuntime(options: VoiceAgentOptions, ?gemmaRuntime: IGemmaRuntime, ?sttRuntime: ISttRuntime, ?ttsRuntime: ITtsRuntime, ?workDir: string) =
    let pathBase =
        S2sRuntimePaths.resolveBaseFromCandidates
            [| Directory.GetCurrentDirectory(); AppContext.BaseDirectory |]
            [| options.Gemma.ModelDir; options.Tts.ModelDir |]
    let fullPath path = S2sRuntimePaths.resolveAgainst pathBase path
    let resolvedWorkDir = defaultArg workDir (fullPath options.WorkDir)
    let gemmaModelDir = fullPath options.Gemma.ModelDir
    let maxTurnAudioSeconds = Math.Max(0.1, options.MaxTurnAudioSeconds)
    let maxTurnAudioSamples24k = int (Math.Ceiling(maxTurnAudioSeconds * 24000.0))
    let maxGemmaAudioSeconds = Math.Max(0.1, options.Gemma.MaxAudioSeconds)
    let maxHistoryTurns = max 0 options.MaxHistoryTurns
    let asrMaxNewTokens = max 1 options.Gemma.AsrMaxNewTokens
    let reasoningMaxNewTokens = max 1 options.Gemma.ReasoningMaxNewTokens
    let toolMaxRounds = max 0 options.Gemma.ToolMaxRounds
    let jsonOptions = JsonSerializerOptions(PropertyNamingPolicy = JsonNamingPolicy.CamelCase, WriteIndented = true)

    let gemmaRuntimeKind =
        if String.IsNullOrWhiteSpace options.Gemma.Runtime then "ort-genai"
        else options.Gemma.Runtime.Trim().ToLowerInvariant()
    let ownedGemma =
        match gemmaRuntime with
        | Some _ -> None
        | None ->
            match gemmaRuntimeKind with
            | "ort-genai" | "ortgenai" ->
                Some(new GemmaOrtGenAiRunner(gemmaModelDir, options.Gemma.Variant, options.Gemma.ExecutionProvider, maxGemmaAudioSeconds) :> IGemmaRuntime)
            | "raw-ort" | "onnx" | "raw-onnx" ->
                Some(new GemmaOnnxRunner(gemmaModelDir, options.Gemma.Variant, options.Gemma.ExecutionProvider, maxGemmaAudioSeconds) :> IGemmaRuntime)
            | other -> invalidArg (nameof options.Gemma.Runtime) $"Unsupported Gemma runtime '{other}'. Use ort-genai or raw-ort."
    let gemma = gemmaRuntime |> Option.defaultWith (fun () -> ownedGemma.Value)
    let processor =
        match gemma with
        | :? GemmaOrtGenAiRunner as runner -> runner.Processor
        | :? GemmaOnnxRunner as runner -> runner.Processor
        | _ -> GemmaProcessor(gemmaModelDir, maxGemmaAudioSeconds)
    let ownedStt =
        match sttRuntime with
        | Some _ -> None
        | None -> Some(new GemmaSttRuntime(gemma, asrMaxNewTokens, maxGemmaAudioSeconds) :> ISttRuntime)
    let stt = sttRuntime |> Option.defaultWith (fun () -> ownedStt.Value)
    let ownedTts =
        match ttsRuntime with
        | Some _ -> None
        | None -> Some(TtsRuntimeFactory.create options.Tts pathBase)
    let tts = ttsRuntime |> Option.defaultWith (fun () -> ownedTts.Value)
    let store = VoiceSessionStore(resolvedWorkDir)

    do Directory.CreateDirectory(resolvedWorkDir) |> ignore

    let safeId (value: string) =
        not (String.IsNullOrWhiteSpace value)
        && value |> Seq.forall (fun ch -> Char.IsLetterOrDigit(ch) || ch = '_' || ch = '-')

    let compactJson payload =
        JsonSerializer.Serialize(payload, JsonSerializerOptions(PropertyNamingPolicy = JsonNamingPolicy.CamelCase))

    let jsonElement payload =
        let json = JsonSerializer.Serialize(payload, jsonOptions)
        use doc = JsonDocument.Parse(json)
        doc.RootElement.Clone()

    let writeDetails (path: string) payload =
        let json = JsonSerializer.Serialize(payload, jsonOptions)
        match Path.GetDirectoryName(path) with
        | null | "" -> ()
        | dir -> Directory.CreateDirectory(dir) |> ignore
        File.WriteAllText(path, json)
        jsonElement payload

    let turnDirectory (session: VoiceSession) (turnIndex: int) =
        Path.Combine(session.WorkDir, "turns", turnIndex.ToString("0000", CultureInfo.InvariantCulture))

    let reserveTurnIndex (session: VoiceSession) =
        lock session.SyncRoot (fun () ->
            let turnIndex = session.NextTurnIndex
            session.NextTurnIndex <- turnIndex + 1
            turnIndex)

    let addCompletedTurn (session: VoiceSession) turn =
        lock session.SyncRoot (fun () -> session.Turns.Add turn)

    let completedTurns (session: VoiceSession) =
        lock session.SyncRoot (fun () -> session.Turns.ToArray())

    let personaPlexCompatibilityStatus =
        { Ready = false
          CodecReady = false
          SpeechToSpeechReady = false
          SupportsStreaming = false
          SupportsDuplex = false
          Runtime = "removed"
          ModelDir = ""
          ExecutionProvider = ""
          VoicePreset = ""
          MissingFiles = Array.empty
          Message = "PersonaPlex is not used in this STT -> Gemma -> TTS voice-agent path." }

    let tools =
        [| { Name = "get_current_time"
             Description = "Return the current local and UTC time."
             Parameters = Array.empty }
           { Name = "get_tts_status"
             Description = "Return a compact status snapshot for the TTS runtime."
             Parameters = Array.empty }
           { Name = "get_agent_status"
             Description = "Return a compact status snapshot for the voice agent runtime."
             Parameters = Array.empty } |]

    let executeTool name (_arguments: Map<string, string>) =
        match name with
        | "get_current_time" ->
            let now = DateTimeOffset.Now
            true,
            compactJson
                {| local = now.ToString("O", CultureInfo.InvariantCulture)
                   utc = now.ToUniversalTime().ToString("O", CultureInfo.InvariantCulture)
                   timeZone = TimeZoneInfo.Local.Id |},
            None
        | "get_tts_status" ->
            let status = tts.Status()
            true,
            compactJson
                {| ready = status.Ready
                   runtime = status.Runtime
                   modelDir = status.ModelDir
                   supportsVoiceCloning = status.SupportsVoiceCloning
                   supportsStreaming = status.SupportsStreaming
                   outputSampleRate = status.OutputSampleRate
                   message = status.Message |},
            None
        | "get_agent_status" ->
            let gemmaStatus = gemma.Status()
            let sttStatus = stt.Status()
            let ttsStatus = tts.Status()
            true,
            compactJson
                {| gemmaReady = gemmaStatus.Ready
                   gemmaModelDir = gemmaStatus.ModelDir
                   sttReady = sttStatus.Ready
                   ttsReady = ttsStatus.Ready
                   ttsRuntime = ttsStatus.Runtime
                   ttsVoiceCloning = ttsStatus.SupportsVoiceCloning |},
            None
        | other ->
            false, "", Some $"Tool '{other}' is not whitelisted."

    let cleanText (text: string) =
        (if Object.ReferenceEquals(text, null) then "" else text).Trim()

    let stripToolCall (text: string) =
        match processor.TryParseToolCall text with
        | Some call -> text.Replace(call.RawText, "").Trim()
        | None -> cleanText text

    let reasoningSystemPrompt (session: VoiceSession) =
        let basePrompt =
            if String.IsNullOrWhiteSpace session.SystemPrompt then
                "You are a concise voice assistant. Use tools when useful, then answer naturally."
            else
                session.SystemPrompt.Trim()
        basePrompt
        + "\n\nYou own reasoning, tool calls, filler text, and final answer text. The service will speak your filler and final answer through a voice-cloning TTS runtime. Keep every spoken answer concise, natural, and easy to synthesize."

    let reasoningMessages (session: VoiceSession) transcript toolMessages =
        let messages = ResizeArray<GemmaChatMessage>()
        messages.Add(GemmaChatMessage.system (reasoningSystemPrompt session))
        completedTurns session
        |> Array.sortBy _.TurnIndex
        |> Array.rev
        |> Array.truncate maxHistoryTurns
        |> Array.rev
        |> Array.iter (fun turn ->
            messages.Add(GemmaChatMessage.user turn.Transcript)
            messages.Add(GemmaChatMessage.model turn.FinalText))
        messages.Add(GemmaChatMessage.user transcript)
        for message in toolMessages do
            messages.Add message
        messages.ToArray()

    let generateFillerText (session: VoiceSession) transcript (call: GemmaToolCall) cancellationToken =
        task {
            let! result =
                gemma.GenerateAsync(
                    { Messages =
                        [| GemmaChatMessage.system "Generate exactly one short spoken filler phrase for a voice assistant while a tool is being called. No quotes. No mention of internal systems. Maximum 8 words."
                           GemmaChatMessage.user $"User said: {transcript}\nTool being called: {call.Name}" |]
                      Tools = Array.empty
                      Audio16k = None
                      AddGenerationPrompt = true
                      MaxNewTokens = 24
                      Temperature = 0.0
                      TopP = 1.0
                      TopK = 0 },
                    cancellationToken)
            let text = cleanText result.Text
            if String.IsNullOrWhiteSpace text then return "Let me check that."
            else return text.Replace("\r", " ").Replace("\n", " ")
        }

    let synthesize
        (session: VoiceSession)
        requestId
        turnIndex
        turnDir
        phase
        outputFileName
        text
        (emit: VoiceAgentStreamingEvent -> Task)
        cancellationToken =
        task {
            let status = tts.Status()
            if not status.Ready then
                do! emit (TtsUnavailable(session.Id, requestId, turnIndex, phase, status.Message))
                return None
            else
                do! emit (TtsSynthesisStarted(session.Id, requestId, turnIndex, phase, text))
                try
                    let! result =
                        tts.SynthesizeAsync(
                            { Phase = phase
                              Text = text
                              OutputDirectory = turnDir
                              OutputFileName = outputFileName
                              VoiceSamplePath = if String.IsNullOrWhiteSpace status.VoiceSamplePath then None else Some status.VoiceSamplePath
                              VoiceSampleTranscript = if String.IsNullOrWhiteSpace options.Tts.VoiceSampleTranscript then None else Some options.Tts.VoiceSampleTranscript },
                            (fun samples -> emit (TtsAudioChunk(session.Id, requestId, turnIndex, phase, status.OutputSampleRate, samples))),
                            cancellationToken)
                    do! emit (TtsSynthesisDone(session.Id, requestId, turnIndex, result))
                    return Some result
                with
                | :? OperationCanceledException ->
                    do! emit (TtsSynthesisCanceled(session.Id, requestId, turnIndex, phase))
                    return raise (OperationCanceledException(cancellationToken))
                | ex ->
                    do! emit (TtsUnavailable(session.Id, requestId, turnIndex, phase, ex.Message))
                    return None
        }

    let normalizeMode mode =
        if String.IsNullOrWhiteSpace mode then
            "gemma-tts"
        else
            match mode.Trim().ToLowerInvariant() with
            | "gemma-tts" | "gemma_tts" | "gemma" | "agent" | "gemma-personaplex" | "gemma_personaplex" -> "gemma-tts"
            | other -> invalidArg (nameof mode) $"Unsupported voice-agent mode '{other}'. Use gemma-tts."

    interface IVoiceAgentRuntime with
        member _.MaxTurnAudioSamples24k = maxTurnAudioSamples24k

        member _.Status() =
            let gemmaStatus = gemma.Status()
            let sttStatus = stt.Status()
            let ttsStatus = tts.Status()
            { Ready = gemmaStatus.Ready && sttStatus.Ready && ttsStatus.Ready
              ServiceName = "GemmaVoiceAgent"
              Mode = "gemma-tts"
              WorkDir = resolvedWorkDir
              MaxHistoryTurns = maxHistoryTurns
              MaxTurnAudioSeconds = maxTurnAudioSeconds
              MaxTurnAudioSamples24k = maxTurnAudioSamples24k
              Gemma = gemmaStatus
              Stt = sttStatus
              Tts = ttsStatus
              PersonaPlex = personaPlexCompatibilityStatus
              Message =
                if gemmaStatus.Ready && sttStatus.Ready && ttsStatus.Ready then
                    $"Gemma voice agent is ready. TTS runtime: {ttsStatus.Runtime}."
                else
                    $"{gemmaStatus.Message} {sttStatus.Message} {ttsStatus.Message}" }

        member _.CreateSession(request: VoiceAgentSessionRequest) =
            let systemPrompt =
                if String.IsNullOrWhiteSpace request.SystemPrompt then
                    "You are a concise voice assistant. Use tools when useful, then answer naturally."
                else
                    request.SystemPrompt.Trim()
            let session = store.Create(systemPrompt, normalizeMode request.Mode)
            store.ToInfo session

        member _.TryGetSession(id: string) =
            if safeId id then store.TryGetInfo id else None

        member _.RunTurnAsync(request: VoiceAgentTurnRequest, emit: VoiceAgentStreamingEvent -> Task, cancellationToken: CancellationToken) =
            task {
                if request.UserAudio24k.Length = 0 then invalidArg "userAudio24k" "User turn audio is required."
                if request.UserAudio24k.Length > maxTurnAudioSamples24k then
                    invalidArg "userAudio24k" $"User turn audio is too large. The configured maximum is {maxTurnAudioSamples24k} Float32 samples at 24 kHz."
                match store.TryGet request.SessionId with
                | None -> return invalidArg "sessionId" "Voice agent session was not found."
                | Some session ->
                    let requestId =
                        request.RequestId
                        |> Option.filter (String.IsNullOrWhiteSpace >> not)
                        |> Option.defaultWith (fun () -> $"{session.Id}_{Guid.NewGuid():N}")
                    let turnIndex = reserveTurnIndex session
                    let turnDir = turnDirectory session turnIndex
                    Directory.CreateDirectory(turnDir) |> ignore
                    File.WriteAllBytes(Path.Combine(turnDir, "user_audio_24k.f32"), AudioChunk.float32ToLittleEndianBytes request.UserAudio24k)

                    try
                        let! transcription = stt.TranscribeAsync(request.UserAudio24k, turnDir, cancellationToken)
                        let transcript = transcription.Transcript
                        File.WriteAllText(Path.Combine(turnDir, "transcript.txt"), transcript)
                        do! emit (VoiceAgentTranscription(session.Id, requestId, turnIndex, transcript))

                        let toolCalls = ResizeArray<AgentToolCallInfo>()
                        let toolResults = ResizeArray<AgentToolResultInfo>()
                        let toolMessages = ResizeArray<GemmaChatMessage>()
                        let fillerResults = ResizeArray<TtsSynthesisResult>()
                        let mutable finalText = ""
                        let mutable round = 0
                        let mutable doneReasoning = false
                        while not doneReasoning do
                            cancellationToken.ThrowIfCancellationRequested()
                            let! reasoning =
                                gemma.GenerateAsync(
                                    { Messages = reasoningMessages session transcript (toolMessages.ToArray())
                                      Tools = tools
                                      Audio16k = None
                                      AddGenerationPrompt = true
                                      MaxNewTokens = reasoningMaxNewTokens
                                      Temperature = 0.0
                                      TopP = 1.0
                                      TopK = 0 },
                                    cancellationToken)
                            match processor.TryParseToolCall reasoning.Text with
                            | Some call when round < toolMaxRounds ->
                                round <- round + 1
                                let callInfo =
                                    { Round = round
                                      Name = call.Name
                                      Arguments = call.Arguments
                                      RawText = call.RawText }
                                toolCalls.Add callInfo
                                do! emit (VoiceAgentToolCall(session.Id, requestId, turnIndex, callInfo))

                                let! fillerText = generateFillerText session transcript call cancellationToken
                                do! emit (VoiceAgentFillerText(session.Id, requestId, turnIndex, fillerText))
                                let toolTask =
                                    Task.Run(
                                        Func<_>(fun () -> executeTool call.Name call.Arguments),
                                        cancellationToken)
                                let! filler =
                                    synthesize session requestId turnIndex turnDir "filler" $"filler_{round}.wav" fillerText emit cancellationToken
                                filler |> Option.iter fillerResults.Add
                                let! success, result, error = toolTask
                                let resultInfo =
                                    { Round = round
                                      Name = call.Name
                                      Success = success
                                      Result = if success then result else ""
                                      Error = error }
                                toolResults.Add resultInfo
                                do! emit (VoiceAgentToolResult(session.Id, requestId, turnIndex, resultInfo))
                                toolMessages.Add(GemmaChatMessage.model call.RawText)
                                toolMessages.Add(GemmaChatMessage.tool call.Name (if success then result else error |> Option.defaultValue "Tool failed."))
                            | Some call ->
                                finalText <- stripToolCall reasoning.Text
                                if String.IsNullOrWhiteSpace finalText then
                                    finalText <- $"I could not complete the requested tool call '{call.Name}' within the configured tool round limit."
                                doneReasoning <- true
                            | None ->
                                finalText <- reasoning.Text.Trim()
                                doneReasoning <- true

                        if String.IsNullOrWhiteSpace finalText then
                            finalText <- "I could not produce a final answer."
                        File.WriteAllText(Path.Combine(turnDir, "final_text.txt"), finalText)
                        do! emit (VoiceAgentFinalText(session.Id, requestId, turnIndex, finalText))

                        let! finalTts = synthesize session requestId turnIndex turnDir "final" "audio.wav" finalText emit cancellationToken
                        let audioPath = finalTts |> Option.bind _.OutputPath |> Option.filter File.Exists
                        let audioUrl = audioPath |> Option.map (fun _ -> $"/api/agent/sessions/{session.Id}/turns/{turnIndex}/audio.wav")
                        let detailsUrl = $"/api/agent/sessions/{session.Id}/turns/{turnIndex}/details.json"
                        let details =
                            {| id = session.Id
                               requestId = requestId
                               turnIndex = turnIndex
                               mode = session.Mode
                               transcript = transcript
                               finalText = finalText
                               toolCalls = toolCalls.ToArray()
                               toolResults = toolResults.ToArray()
                               fillerTts = fillerResults.ToArray()
                               finalTts = finalTts
                               audioUrl = audioUrl
                               stt = transcription
                               gemmaStatus = gemma.Status()
                               ttsStatus = tts.Status() |}
                        let detailsElement = writeDetails (Path.Combine(turnDir, "details.json")) details
                        let result =
                            { Id = session.Id
                              RequestId = requestId
                              TurnIndex = turnIndex
                              Transcript = transcript
                              FinalText = finalText
                              ToolCalls = toolCalls.ToArray()
                              ToolResults = toolResults.ToArray()
                              AudioUrl = audioUrl
                              DetailsUrl = detailsUrl
                              Details = detailsElement }
                        addCompletedTurn
                            session
                            { TurnIndex = turnIndex
                              RequestId = requestId
                              Transcript = transcript
                              FinalText = finalText
                              ToolCalls = toolCalls.ToArray()
                              ToolResults = toolResults.ToArray()
                              WorkDir = turnDir }
                        do! emit (VoiceAgentDone result)
                        return result
                    with
                    | :? OperationCanceledException as ex ->
                        do! emit (VoiceAgentCanceled(session.Id, Some requestId))
                        return raise ex
            }

        member _.TryGetTurnArtifact(sessionId: string, turnIndex: int, fileName: string) =
            if not (safeId sessionId) || turnIndex < 1 then
                None
            else
                match store.TryGet sessionId with
                | None -> None
                | Some session ->
                    let path = Path.Combine(turnDirectory session turnIndex, fileName)
                    if File.Exists path then
                        let contentType =
                            match fileName with
                            | "details.json" -> "application/json; charset=utf-8"
                            | "transcript.txt" | "final_text.txt" -> "text/plain; charset=utf-8"
                            | file when file.EndsWith(".wav", StringComparison.OrdinalIgnoreCase) -> "audio/wav"
                            | _ -> "application/octet-stream"
                        Some { Path = path; ContentType = contentType }
                    else
                        None

    interface IDisposable with
        member _.Dispose() =
            ownedGemma
            |> Option.iter (fun runtime ->
                match runtime with
                | :? IDisposable as disposable -> disposable.Dispose()
                | _ -> ())
            ownedStt
            |> Option.iter (fun runtime ->
                match runtime with
                | :? IDisposable as disposable -> disposable.Dispose()
                | _ -> ())
            ownedTts
            |> Option.iter (fun runtime ->
                match runtime with
                | :? IDisposable as disposable -> disposable.Dispose()
                | _ -> ())
