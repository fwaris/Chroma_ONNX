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

type private AgentRuntimeSession =
    { Id: string
      PromptText: string
      SystemPrompt: string
      PromptAudio24k: float32 array
      MaxNewFrames: int
      CreatedUtc: DateTimeOffset
      WorkDir: string
      SyncRoot: obj
      Turns: ResizeArray<AgentRuntimeTurn>
      mutable NextTurnIndex: int }

and private AgentRuntimeTurn =
    { TurnIndex: int
      RequestId: string
      UserAudio16k: float32 array
      Transcript: string
      FinalText: string
      ToolCalls: AgentToolCallInfo array
      ToolResults: AgentToolResultInfo array
      WorkDir: string }

type private AgentSessionStore(workDir: string) =
    let sessions = ConcurrentDictionary<string, AgentRuntimeSession>(StringComparer.Ordinal)

    let newId () =
        let timestamp = DateTimeOffset.UtcNow.ToString("yyyyMMddHHmmssfff", CultureInfo.InvariantCulture)
        let suffix = Guid.NewGuid().ToString("N").Substring(0, 8)
        $"{timestamp}_{suffix}"

    let toInfo (session: AgentRuntimeSession) =
        { Id = session.Id
          ServiceName = "GemmaChromaAgent"
          Mode = "gemma_chroma_agent"
          PromptText = session.PromptText
          SystemPrompt = session.SystemPrompt
          MaxNewFrames = session.MaxNewFrames
          PromptAudioSamples = session.PromptAudio24k.Length
          PromptSampleRate = 24000
          WebsocketUrl = $"/ws/agent/{session.Id}"
          CreatedUtc = session.CreatedUtc }

    member _.Create(promptText: string, systemPrompt: string, promptAudio24k: float32 array, maxNewFrames: int) =
        let id = newId ()
        let sessionDir = Path.Combine(workDir, "agent", id)
        Directory.CreateDirectory(sessionDir) |> ignore
        let session =
            { Id = id
              PromptText = promptText
              SystemPrompt = systemPrompt
              PromptAudio24k = promptAudio24k
              MaxNewFrames = maxNewFrames
              CreatedUtc = DateTimeOffset.UtcNow
              WorkDir = sessionDir
              SyncRoot = obj ()
              Turns = ResizeArray<AgentRuntimeTurn>()
              NextTurnIndex = 1 }
        sessions[id] <- session
        session

    member _.TryGet(id: string) =
        match sessions.TryGetValue(id) with
        | true, session -> Some session
        | false, _ -> None

    member _.TryGetInfo(id: string) =
        match sessions.TryGetValue(id) with
        | true, session -> Some(toInfo session)
        | false, _ -> None

    member _.ToInfo(session: AgentRuntimeSession) = toInfo session

type GemmaChromaAgentRuntime(options: GemmaRuntimeOptions, s2s: IS2sRuntime, ?gemmaRuntime: IGemmaRuntime, ?workDir: string) =
    let pathBase =
        S2sRuntimePaths.resolveBaseFromCandidates
            [| Directory.GetCurrentDirectory(); AppContext.BaseDirectory |]
            [| options.ModelDir |]
    let fullPath path = S2sRuntimePaths.resolveAgainst pathBase path
    let modelDir = fullPath options.ModelDir
    let workDir = defaultArg workDir (fullPath "served_runs")
    let maxAudioSeconds = Math.Max(0.1, options.MaxAudioSeconds)
    let maxTurnAudioSamples = int (Math.Ceiling(maxAudioSeconds * 16000.0))
    let asrMaxNewTokens = max 1 options.AsrMaxNewTokens
    let reasoningMaxNewTokens = max 1 options.ReasoningMaxNewTokens
    let toolMaxRounds = max 0 options.ToolMaxRounds
    let maxHistoryTurns = max 0 options.MaxHistoryTurns
    let jsonOptions = JsonSerializerOptions(PropertyNamingPolicy = JsonNamingPolicy.CamelCase, WriteIndented = true)
    let runtimeKind =
        if String.IsNullOrWhiteSpace options.Runtime then
            "ort-genai"
        else
            options.Runtime.Trim().ToLowerInvariant()
    let ownedGemma =
        match gemmaRuntime with
        | Some runtime -> None
        | None ->
            match runtimeKind with
            | "ort-genai" | "ortgenai" ->
                Some(new GemmaOrtGenAiRunner(modelDir, options.Variant, options.ExecutionProvider, maxAudioSeconds) :> IGemmaRuntime)
            | "raw-ort" | "onnx" | "raw-onnx" ->
                Some(new GemmaOnnxRunner(modelDir, options.Variant, options.ExecutionProvider, maxAudioSeconds) :> IGemmaRuntime)
            | other -> invalidArg (nameof options.Runtime) $"Unsupported Gemma runtime '{other}'. Use ort-genai or raw-ort."
    let gemma =
        match gemmaRuntime with
        | Some runtime -> runtime
        | None -> ownedGemma.Value
    let processor =
        match gemma with
        | :? GemmaOrtGenAiRunner as runner -> runner.Processor
        | :? GemmaOnnxRunner as runner -> runner.Processor
        | _ -> GemmaProcessor(modelDir, maxAudioSeconds)
    let store = AgentSessionStore(workDir)

    do Directory.CreateDirectory(workDir) |> ignore

    let safeId (value: string) =
        not (String.IsNullOrWhiteSpace value)
        && value |> Seq.forall (fun ch -> Char.IsLetterOrDigit(ch) || ch = '_' || ch = '-')

    let turnDirectory (session: AgentRuntimeSession) (turnIndex: int) =
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

    let completedTurns (session: AgentRuntimeSession) =
        lock session.SyncRoot (fun () -> session.Turns.ToArray())

    let reserveTurnIndex (session: AgentRuntimeSession) =
        lock session.SyncRoot (fun () ->
            let turnIndex = session.NextTurnIndex
            session.NextTurnIndex <- turnIndex + 1
            turnIndex)

    let addCompletedTurn (session: AgentRuntimeSession) turn =
        lock session.SyncRoot (fun () -> session.Turns.Add(turn))

    let tools =
        [| { Name = "get_current_time"
             Description = "Return the current local and UTC time."
             Parameters = Array.empty }
           { Name = "get_chroma_status"
             Description = "Return a compact status snapshot for the Chroma vocalization runtime."
             Parameters = Array.empty }
           { Name = "get_agent_status"
             Description = "Return a compact status snapshot for the Gemma plus Chroma agent runtime."
             Parameters = Array.empty } |]

    let compactJson payload =
        JsonSerializer.Serialize(payload, JsonSerializerOptions(PropertyNamingPolicy = JsonNamingPolicy.CamelCase))

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
        | "get_chroma_status" ->
            let status = s2s.Status()
            true,
            compactJson
                {| ready = status.Ready
                   executionProvider = status.ExecutionProvider
                   memoryMode = status.MemoryMode
                   generationMode = status.GenerationMode
                   samplingAlgorithm = status.SamplingAlgorithm
                   queueLength = status.QueueLength
                   message = status.Message |},
            None
        | "get_agent_status" ->
            let status = gemma.Status()
            true,
            compactJson
                {| gemmaReady = status.Ready
                   modelDir = status.ModelDir
                   variant = status.Variant
                   executionProvider = status.ExecutionProvider
                   loadedSessions = status.LoadedSessions
                   message = status.Message |},
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

    let reasoningSystemPrompt (session: AgentRuntimeSession) =
        let basePrompt =
            if String.IsNullOrWhiteSpace session.SystemPrompt then
                "You are a concise voice assistant. Use tools when useful, then answer naturally."
            else
                session.SystemPrompt.Trim()
        basePrompt
        + "\n\nYou own transcription, reasoning, and tool calling. Chroma will only vocalize your final answer. Keep final answers concise and speakable."

    let reasoningMessages (session: AgentRuntimeSession) transcript toolMessages =
        let messages = ResizeArray<GemmaChatMessage>()
        messages.Add(GemmaChatMessage.system (reasoningSystemPrompt session))
        let history =
            completedTurns session
            |> Array.sortBy _.TurnIndex
            |> Array.rev
            |> Array.truncate maxHistoryTurns
            |> Array.rev
        for turn in history do
            messages.Add(GemmaChatMessage.user turn.Transcript)
            messages.Add(GemmaChatMessage.model turn.FinalText)
        messages.Add(GemmaChatMessage.user transcript)
        for message in toolMessages do
            messages.Add(message)
        messages.ToArray()

    let vocalizationSystemPrompt (finalText: string) =
        "You are Chroma, an expressive voice model. Vocalize the assistant answer naturally and clearly. The answer to speak is:\n\n\"\"\"\n"
        + finalText.Trim()
        + "\n\"\"\"\n\nDo not add extra facts or tool details."

    let silence16k =
        Array.zeroCreate<float32> 8000

    interface IAgentRuntime with
        member _.MaxTurnAudioSamples = maxTurnAudioSamples

        member _.Status() =
            let gemmaStatus = gemma.Status()
            let chromaStatus = s2s.Status()
            { Ready = gemmaStatus.Ready && chromaStatus.Ready
              ServiceName = "GemmaChromaAgent"
              Mode = "gemma_chroma_agent"
              ModelDir = modelDir
              Variant = options.Variant
              ExecutionProvider = options.ExecutionProvider
              MaxAudioSeconds = maxAudioSeconds
              AsrMaxNewTokens = asrMaxNewTokens
              ReasoningMaxNewTokens = reasoningMaxNewTokens
              ToolMaxRounds = toolMaxRounds
              MaxHistoryTurns = maxHistoryTurns
              Gemma = gemmaStatus
              ChromaReady = chromaStatus.Ready
              Message =
                if gemmaStatus.Ready && chromaStatus.Ready then
                    "Gemma plus Chroma agent is ready."
                elif not gemmaStatus.Ready then
                    gemmaStatus.Message
                else
                    chromaStatus.Message }

        member _.CreateSession(request: AgentSessionRequest) =
            if String.IsNullOrWhiteSpace request.PromptText then
                invalidArg "promptText" "promptText is required."
            if request.PromptAudio24k.Length = 0 then
                invalidArg "promptPcm24k" "promptPcm24k is required."
            let maxNewFrames = max 1 request.MaxNewFrames
            let session = store.Create(request.PromptText, request.SystemPrompt, request.PromptAudio24k, maxNewFrames)
            File.WriteAllBytes(Path.Combine(session.WorkDir, "prompt_audio_24k.f32"), AudioChunk.float32ToLittleEndianBytes request.PromptAudio24k)
            store.ToInfo session

        member _.TryGetSession(id: string) =
            if safeId id then store.TryGetInfo id else None

        member _.RunTurnAsync(request: AgentTurnRequest, emit: AgentStreamingEvent -> Task, cancellationToken: CancellationToken) =
            task {
                if request.UserAudio16k.Length = 0 then invalidArg "userAudio16k" "User turn audio is required."
                if request.UserAudio16k.Length > maxTurnAudioSamples then
                    invalidArg "userAudio16k" $"User turn audio is too large. The configured maximum is {maxTurnAudioSamples} Float32 samples."
                match store.TryGet request.SessionId with
                | None -> return invalidArg "sessionId" "Agent session was not found."
                | Some session ->
                    let requestId =
                        request.RequestId
                        |> Option.filter (String.IsNullOrWhiteSpace >> not)
                        |> Option.defaultWith (fun () -> $"{session.Id}_{Guid.NewGuid():N}")
                    let turnIndex = reserveTurnIndex session
                    let turnDir = turnDirectory session turnIndex
                    Directory.CreateDirectory(turnDir) |> ignore
                    File.WriteAllBytes(Path.Combine(turnDir, "user_audio_16k.f32"), AudioChunk.float32ToLittleEndianBytes request.UserAudio16k)

                    try
                        let! asr =
                            gemma.GenerateAsync(
                                { Messages = [| GemmaChatMessage.user asrPrompt |]
                                  Tools = Array.empty
                                  Audio16k = Some request.UserAudio16k
                                  AddGenerationPrompt = true
                                  MaxNewTokens = asrMaxNewTokens
                                  Temperature = 0.0
                                  TopP = 1.0
                                  TopK = 0 },
                                cancellationToken
                            )
                        let transcript = cleanTranscript asr.Text
                        do! emit (AgentTranscription(session.Id, requestId, turnIndex, transcript))

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
                                do! emit (AgentToolCall(session.Id, requestId, turnIndex, callInfo))
                                let success, result, error = executeTool call.Name call.Arguments
                                let resultInfo =
                                    { Round = round
                                      Name = call.Name
                                      Success = success
                                      Result = if success then result else ""
                                      Error = error }
                                toolResults.Add resultInfo
                                do! emit (AgentToolResult(session.Id, requestId, turnIndex, resultInfo))
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
                        do! emit (AgentFinalText(session.Id, requestId, turnIndex, finalText))

                        let chromaSession =
                            s2s.CreateSession
                                { PromptText = session.PromptText
                                  SystemPrompt = vocalizationSystemPrompt finalText
                                  Backend = "fsharp_onnx"
                                  PromptAudio24k = session.PromptAudio24k
                                  MaxNewFrames = session.MaxNewFrames }
                        do! emit (AgentVocalizationStarted(session.Id, requestId, turnIndex, chromaSession.Id))
                        let! chromaResult =
                            s2s.RunTurnAsync(
                                { SessionId = chromaSession.Id
                                  UserAudio16k = silence16k
                                  RequestId = Some $"{requestId}_chroma" },
                                (fun event -> emit (AgentChromaEvent event)),
                                cancellationToken
                            )

                        let detailsUrl = $"/api/agent/sessions/{session.Id}/turns/{turnIndex}/details.json"
                        let details =
                            {| id = session.Id
                               requestId = requestId
                               turnIndex = turnIndex
                               transcript = transcript
                               finalText = finalText
                               toolCalls = toolCalls.ToArray()
                               toolResults = toolResults.ToArray()
                               chromaSessionId = chromaSession.Id
                               chromaTurnIndex = chromaResult.TurnIndex
                               audioUrl = chromaResult.AudioUrl
                               chromaDetailsUrl = chromaResult.DetailsUrl
                               asr = {| stopReason = asr.StopReason; inputTokenCount = asr.InputTokenCount; timingsMs = asr.TimingsMs |}
                               gemmaStatus = gemma.Status()
                               chromaStatus = s2s.Status() |}
                        let detailsElement = writeDetails (Path.Combine(turnDir, "details.json")) details
                        let result =
                            { Id = session.Id
                              RequestId = requestId
                              TurnIndex = turnIndex
                              Transcript = transcript
                              FinalText = finalText
                              ToolCalls = toolCalls.ToArray()
                              ToolResults = toolResults.ToArray()
                              ChromaSessionId = Some chromaSession.Id
                              ChromaTurnIndex = Some chromaResult.TurnIndex
                              AudioUrl = Some chromaResult.AudioUrl
                              DetailsUrl = detailsUrl
                              Details = detailsElement }
                        addCompletedTurn
                            session
                            { TurnIndex = turnIndex
                              RequestId = requestId
                              UserAudio16k = Array.copy request.UserAudio16k
                              Transcript = transcript
                              FinalText = finalText
                              ToolCalls = toolCalls.ToArray()
                              ToolResults = toolResults.ToArray()
                              WorkDir = turnDir }
                        do! emit (AgentDone result)
                        return result
                    with
                    | :? OperationCanceledException as ex ->
                        do! emit (AgentCanceled(session.Id, Some requestId))
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
