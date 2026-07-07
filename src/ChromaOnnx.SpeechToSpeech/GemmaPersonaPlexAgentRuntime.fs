namespace ChromaOnnx.SpeechToSpeech

open System
open System.Collections.Concurrent
open System.Collections.Generic
open System.Globalization
open System.IO
open System.Text.Json
open System.Threading
open System.Threading.Tasks
open ChromaOnnx

type private VoiceAgentSession =
    { Id: string
      SystemPrompt: string
      Mode: string
      WorkDir: string
      CreatedUtc: DateTimeOffset
      SyncRoot: obj
      Turns: ResizeArray<VoiceAgentTurn>
      mutable NextTurnIndex: int }

and private VoiceAgentTurn =
    { TurnIndex: int
      RequestId: string
      Transcript: string
      FinalText: string
      ToolCalls: AgentToolCallInfo array
      ToolResults: AgentToolResultInfo array
      WorkDir: string }

type private VoiceAgentSessionStore(workDir: string) =
    let sessions = ConcurrentDictionary<string, VoiceAgentSession>(StringComparer.Ordinal)

    let toInfo (session: VoiceAgentSession) =
        { Id = session.Id
          ServiceName = "GemmaPersonaPlexAgent"
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
              Turns = ResizeArray<VoiceAgentTurn>()
              NextTurnIndex = 1 }
        sessions[id] <- session
        session

    member _.TryGet(id: string) =
        match sessions.TryGetValue id with
        | true, session -> Some session
        | false, _ -> None

    member this.TryGetInfo(id: string) =
        this.TryGet id |> Option.map toInfo

    member _.ToInfo(session: VoiceAgentSession) = toInfo session

type GemmaPersonaPlexAgentRuntime(options: VoiceAgentOptions, ?gemmaRuntime: IGemmaRuntime, ?personaPlexRuntime: IPersonaPlexRuntime, ?workDir: string) =
    let pathBase =
        S2sRuntimePaths.resolveBaseFromCandidates
            [| Directory.GetCurrentDirectory(); AppContext.BaseDirectory |]
            [| options.Gemma.ModelDir; options.PersonaPlex.ModelDir |]
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
        if String.IsNullOrWhiteSpace options.Gemma.Runtime then
            "ort-genai"
        else
            options.Gemma.Runtime.Trim().ToLowerInvariant()
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
    let gemma =
        match gemmaRuntime with
        | Some runtime -> runtime
        | None -> ownedGemma.Value
    let processor =
        match gemma with
        | :? GemmaOrtGenAiRunner as runner -> runner.Processor
        | :? GemmaOnnxRunner as runner -> runner.Processor
        | _ -> GemmaProcessor(gemmaModelDir, maxGemmaAudioSeconds)
    let ownedPersonaPlex =
        match personaPlexRuntime with
        | Some _ -> None
        | None ->
            let runtime =
                if String.IsNullOrWhiteSpace options.PersonaPlex.Runtime then
                    "full-onnx"
                else
                    options.PersonaPlex.Runtime.Trim().ToLowerInvariant()
            match runtime with
            | "full-onnx" | "full_onnx" | "onnx" -> Some(new PersonaPlexFullOnnxRuntime(options.PersonaPlex, pathBase) :> IPersonaPlexRuntime)
            | "elbruno-codec" | "elbruno_codec" | "codec" -> Some(new ElBrunoPersonaPlexRuntime(options.PersonaPlex, pathBase) :> IPersonaPlexRuntime)
            | other -> invalidArg (nameof options.PersonaPlex.Runtime) $"Unsupported PersonaPlex runtime '{other}'. Use full-onnx or elbruno-codec."
    let personaPlex =
        match personaPlexRuntime with
        | Some runtime -> runtime
        | None -> ownedPersonaPlex.Value
    let store = VoiceAgentSessionStore(resolvedWorkDir)

    do Directory.CreateDirectory(resolvedWorkDir) |> ignore

    let safeId (value: string) =
        not (String.IsNullOrWhiteSpace value)
        && value |> Seq.forall (fun ch -> Char.IsLetterOrDigit(ch) || ch = '_' || ch = '-')

    let turnDirectory (session: VoiceAgentSession) (turnIndex: int) =
        Path.Combine(session.WorkDir, "turns", turnIndex.ToString("0000", CultureInfo.InvariantCulture))

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

    let completedTurns (session: VoiceAgentSession) =
        lock session.SyncRoot (fun () -> session.Turns.ToArray())

    let reserveTurnIndex (session: VoiceAgentSession) =
        lock session.SyncRoot (fun () ->
            let turnIndex = session.NextTurnIndex
            session.NextTurnIndex <- turnIndex + 1
            turnIndex)

    let addCompletedTurn (session: VoiceAgentSession) turn =
        lock session.SyncRoot (fun () -> session.Turns.Add turn)

    let compactJson payload =
        JsonSerializer.Serialize(payload, JsonSerializerOptions(PropertyNamingPolicy = JsonNamingPolicy.CamelCase))

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

    let tools =
        [| { Name = "get_current_time"
             Description = "Return the current local and UTC time."
             Parameters = Array.empty }
           { Name = "get_personaplex_status"
             Description = "Return a compact status snapshot for the PersonaPlex runtime."
             Parameters = Array.empty }
           { Name = "get_agent_status"
             Description = "Return a compact status snapshot for the Gemma plus PersonaPlex agent runtime."
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
        | "get_personaplex_status" ->
            let status = personaPlex.Status()
            true,
            compactJson
                {| ready = status.Ready
                   codecReady = status.CodecReady
                   speechToSpeechReady = status.SpeechToSpeechReady
                   supportsStreaming = status.SupportsStreaming
                   supportsDuplex = status.SupportsDuplex
                   modelDir = status.ModelDir
                   message = status.Message |},
            None
        | "get_agent_status" ->
            let gemmaStatus = gemma.Status()
            let personaStatus = personaPlex.Status()
            true,
            compactJson
                {| gemmaReady = gemmaStatus.Ready
                   gemmaModelDir = gemmaStatus.ModelDir
                   gemmaLoadedSessions = gemmaStatus.LoadedSessions
                   personaPlexReady = personaStatus.Ready
                   personaPlexRuntime = personaStatus.Runtime
                   personaPlexSpeechToSpeechReady = personaStatus.SpeechToSpeechReady |},
            None
        | other ->
            false, "", Some $"Tool '{other}' is not whitelisted."

    let asrPrompt =
        "Transcribe the following speech segment in its original language. Follow these specific instructions for formatting the answer:\n* Only output the transcription, with no newlines.\n* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three.\n\n<|audio|>"

    let cleanTranscript (text: string) =
        (if Object.ReferenceEquals(text, null) then "" else text)
            .Replace("\r", " ")
            .Replace("\n", " ")
            .Trim()

    let stripToolCall (text: string) =
        match processor.TryParseToolCall text with
        | Some call -> text.Replace(call.RawText, "").Trim()
        | None -> (if Object.ReferenceEquals(text, null) then "" else text).Trim()

    let reasoningSystemPrompt (session: VoiceAgentSession) =
        let basePrompt =
            if String.IsNullOrWhiteSpace session.SystemPrompt then
                "You are a concise voice assistant. Use tools when useful, then answer naturally."
            else
                session.SystemPrompt.Trim()
        basePrompt
        + "\n\nGemma owns transcription, reasoning, tool calls, and final text. PersonaPlex is the speech layer and may only expose codec round-trip behavior in this build. Keep final answers concise and speakable."

    let reasoningMessages (session: VoiceAgentSession) transcript toolMessages =
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

    let normalizeSessionMode (mode: string) =
        if String.IsNullOrWhiteSpace mode then
            "gemma-personaplex"
        else
            match mode.Trim().ToLowerInvariant() with
            | "personaplex-full" | "personaplex_full" | "personaplex" | "full-onnx" -> "personaplex-full"
            | "gemma-personaplex" | "gemma_personaplex" | "gemma" | "agent" -> "gemma-personaplex"
            | other -> invalidArg (nameof mode) $"Unsupported voice-agent mode '{other}'. Use gemma-personaplex or personaplex-full."

    let runPersonaPlexFullTurn
        (session: VoiceAgentSession)
        (request: VoiceAgentTurnRequest)
        requestId
        turnIndex
        turnDir
        (emit: VoiceAgentStreamingEvent -> Task)
        cancellationToken =
        task {
            do! emit (PersonaPlexGenerationStarted(session.Id, requestId, turnIndex))
            let! generation = personaPlex.RunSpeechToSpeechAsync(request.UserAudio24k, turnDir, cancellationToken)
            do! emit (PersonaPlexGenerationDone(session.Id, requestId, turnIndex, generation))
            match generation.OutputPath with
            | Some path when File.Exists path -> ()
            | _ -> do! emit (PersonaPlexUnavailable(session.Id, requestId, turnIndex, generation.Message))

            let personaStatus = personaPlex.Status()
            if not personaStatus.SpeechToSpeechReady then
                do! emit (PersonaPlexUnavailable(session.Id, requestId, turnIndex, personaStatus.Message))

            let transcript = ""
            let finalText = generation.Message
            File.WriteAllText(Path.Combine(turnDir, "transcript.txt"), transcript)
            File.WriteAllText(Path.Combine(turnDir, "final_text.txt"), finalText)
            let audioUrl =
                generation.OutputPath
                |> Option.filter File.Exists
                |> Option.map (fun _ -> $"/api/agent/sessions/{session.Id}/turns/{turnIndex}/audio.wav")
            let detailsUrl = $"/api/agent/sessions/{session.Id}/turns/{turnIndex}/details.json"
            let details =
                {| id = session.Id
                   requestId = requestId
                   turnIndex = turnIndex
                   mode = session.Mode
                   transcript = transcript
                   finalText = finalText
                   toolCalls = Array.empty<AgentToolCallInfo>
                   toolResults = Array.empty<AgentToolResultInfo>
                   audioUrl = audioUrl
                   personaPlexGeneration = generation
                   personaPlexStatus = personaStatus |}
            let detailsElement = writeDetails (Path.Combine(turnDir, "details.json")) details
            let result =
                { Id = session.Id
                  RequestId = requestId
                  TurnIndex = turnIndex
                  Transcript = transcript
                  FinalText = finalText
                  ToolCalls = Array.empty
                  ToolResults = Array.empty
                  AudioUrl = audioUrl
                  DetailsUrl = detailsUrl
                  Details = detailsElement }
            addCompletedTurn
                session
                { TurnIndex = turnIndex
                  RequestId = requestId
                  Transcript = transcript
                  FinalText = finalText
                  ToolCalls = Array.empty
                  ToolResults = Array.empty
                  WorkDir = turnDir }
            do! emit (VoiceAgentDone result)
            return result
        }

    interface IVoiceAgentRuntime with
        member _.MaxTurnAudioSamples24k = maxTurnAudioSamples24k

        member _.Status() =
            let gemmaStatus = gemma.Status()
            let personaStatus = personaPlex.Status()
            { Ready = (gemmaStatus.Ready && personaStatus.CodecReady) || personaStatus.Ready
              ServiceName = "GemmaPersonaPlexAgent"
              Mode = "gemma-personaplex/personaplex-full"
              WorkDir = resolvedWorkDir
              MaxHistoryTurns = maxHistoryTurns
              MaxTurnAudioSeconds = maxTurnAudioSeconds
              MaxTurnAudioSamples24k = maxTurnAudioSamples24k
              Gemma = gemmaStatus
              PersonaPlex = personaStatus
              Message =
                if gemmaStatus.Ready && personaStatus.CodecReady then
                    $"Gemma plus PersonaPlex agent is ready. PersonaPlex runtime: {personaStatus.Runtime}. {personaStatus.Message}"
                elif personaStatus.Ready then
                    $"PersonaPlex runtime is ready for personaplex-full diagnostics. {personaStatus.Message}"
                else
                    $"{gemmaStatus.Message} {personaStatus.Message}" }

        member _.CreateSession(request: VoiceAgentSessionRequest) =
            let systemPrompt =
                if String.IsNullOrWhiteSpace request.SystemPrompt then
                    "You are a concise voice assistant. Use tools when useful, then answer naturally."
                else
                    request.SystemPrompt.Trim()
            let mode = normalizeSessionMode request.Mode
            let session = store.Create(systemPrompt, mode)
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

                    if session.Mode = "personaplex-full" then
                        return! runPersonaPlexFullTurn session request requestId turnIndex turnDir emit cancellationToken
                    else
                    try
                        let userAudio16k = resampleLinear request.UserAudio24k 24000 16000
                        let! asr =
                            gemma.GenerateAsync(
                                { Messages = [| GemmaChatMessage.user asrPrompt |]
                                  Tools = Array.empty
                                  Audio16k = Some userAudio16k
                                  AddGenerationPrompt = true
                                  MaxNewTokens = asrMaxNewTokens
                                  Temperature = 0.0
                                  TopP = 1.0
                                  TopK = 0 },
                                cancellationToken
                            )
                        let transcript = cleanTranscript asr.Text
                        do! emit (VoiceAgentTranscription(session.Id, requestId, turnIndex, transcript))

                        let toolCalls = ResizeArray<AgentToolCallInfo>()
                        let toolResults = ResizeArray<AgentToolResultInfo>()
                        let toolMessages = ResizeArray<GemmaChatMessage>()
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
                                    cancellationToken
                                )
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
                                let success, result, error = executeTool call.Name call.Arguments
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
                        File.WriteAllText(Path.Combine(turnDir, "transcript.txt"), transcript)
                        File.WriteAllText(Path.Combine(turnDir, "final_text.txt"), finalText)
                        do! emit (VoiceAgentFinalText(session.Id, requestId, turnIndex, finalText))

                        let personaStatusBefore = personaPlex.Status()
                        let mutable codecResult: PersonaPlexCodecResult option = None
                        if personaStatusBefore.CodecReady then
                            do! emit (PersonaPlexCodecStarted(session.Id, requestId, turnIndex))
                            try
                                let! codec = personaPlex.RunCodecRoundTripAsync(request.UserAudio24k, turnDir, cancellationToken)
                                codecResult <- Some codec
                                do! emit (PersonaPlexCodecDone(session.Id, requestId, turnIndex, codec))
                            with ex ->
                                do! emit (PersonaPlexUnavailable(session.Id, requestId, turnIndex, ex.Message))
                        else
                            do! emit (PersonaPlexUnavailable(session.Id, requestId, turnIndex, personaStatusBefore.Message))
                        if not personaStatusBefore.SpeechToSpeechReady then
                            do! emit (PersonaPlexUnavailable(session.Id, requestId, turnIndex, personaStatusBefore.Message))

                        let audioPath =
                            codecResult
                            |> Option.bind _.OutputPath
                            |> Option.filter File.Exists
                        let audioUrl =
                            audioPath
                            |> Option.map (fun _ -> $"/api/agent/sessions/{session.Id}/turns/{turnIndex}/audio.wav")
                        let detailsUrl = $"/api/agent/sessions/{session.Id}/turns/{turnIndex}/details.json"
                        let details =
                            {| id = session.Id
                               requestId = requestId
                               turnIndex = turnIndex
                               transcript = transcript
                               finalText = finalText
                               toolCalls = toolCalls.ToArray()
                               toolResults = toolResults.ToArray()
                               audioUrl = audioUrl
                               personaPlexCodec = codecResult
                               personaPlexStatus = personaPlex.Status()
                               asr = {| stopReason = asr.StopReason; inputTokenCount = asr.InputTokenCount; timingsMs = asr.TimingsMs |}
                               gemmaStatus = gemma.Status() |}
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
                            | "audio.wav" -> "audio/wav"
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
            ownedPersonaPlex
            |> Option.iter (fun runtime ->
                match runtime with
                | :? IDisposable as disposable -> disposable.Dispose()
                | _ -> ())
