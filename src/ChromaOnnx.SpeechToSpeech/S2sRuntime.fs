namespace ChromaOnnx.SpeechToSpeech

open System
open System.Collections.Concurrent
open System.Collections.Generic
open System.Globalization
open System.IO
open System.Linq
open System.Text.Json
open System.Threading
open System.Threading.Tasks
open ChromaOnnx
open Microsoft.ML.OnnxRuntime.Tensors

type private RuntimeSession =
    { Id: string
      PromptText: string
      SystemPrompt: string
      Backend: string
      PromptAudio24k: float32 array
      MaxNewFrames: int
      CreatedUtc: DateTimeOffset
      WorkDir: string
      mutable LastDetails: JsonElement option }

type private RuntimeSessionStore(workDir: string, promptSampleRate: int) =
    let sessions = ConcurrentDictionary<string, RuntimeSession>(StringComparer.Ordinal)

    let newId () =
        let timestamp = DateTimeOffset.UtcNow.ToString("yyyyMMddHHmmssfff", CultureInfo.InvariantCulture)
        let suffix = Guid.NewGuid().ToString("N").Substring(0, 8)
        $"{timestamp}_{suffix}"

    let toInfo (session: RuntimeSession) =
        { Id = session.Id
          ServiceName = "ChromaS2SONNX"
          Mode = "s2s_greedy_streaming"
          Backend = session.Backend
          PromptText = session.PromptText
          SystemPrompt = session.SystemPrompt
          MaxNewFrames = session.MaxNewFrames
          MaxResponseSeconds = Math.Round(float session.MaxNewFrames * 0.08, 2)
          PromptAudioSamples = session.PromptAudio24k.Length
          PromptSampleRate = promptSampleRate
          WebsocketUrl = $"/ws/s2s/{session.Id}"
          CreatedUtc = session.CreatedUtc }

    member _.Create(promptText: string, systemPrompt: string, backend: string, promptAudio24k: float32 array, maxNewFrames: int) =
        let id = newId ()
        let sessionDir = Path.Combine(workDir, "s2s", id)
        Directory.CreateDirectory(sessionDir) |> ignore
        let session =
            { Id = id
              PromptText = promptText
              SystemPrompt = systemPrompt
              Backend = backend
              PromptAudio24k = promptAudio24k
              MaxNewFrames = maxNewFrames
              CreatedUtc = DateTimeOffset.UtcNow
              WorkDir = sessionDir
              LastDetails = None }

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

    member _.ToInfo(session: RuntimeSession) = toInfo session

type ChromaS2sRuntime(options: S2sRuntimeOptions) =
    let jsonOptions =
        JsonSerializerOptions(PropertyNamingPolicy = JsonNamingPolicy.CamelCase, WriteIndented = true)

    let pathBase = S2sRuntimePaths.resolveBaseForOptions options
    let fullPath path = S2sRuntimePaths.resolveAgainst pathBase path
    let modelDir = fullPath options.ModelDir
    let bundleDir = fullPath options.BundleDir
    let workDir = fullPath options.WorkDir
    let streamDecodeFrames = max 1 options.StreamDecodeFrames
    let streamMinFreeVramMb = max 0 options.StreamMinFreeVramMb
    let codecStallGuardFrames = max 0 options.CodecStallGuardFrames
    let maxQueueLength = max 0 options.MaxQueueLength
    let maxPromptAudioSeconds = Math.Max(0.1, options.MaxPromptAudioSeconds)
    let maxTurnAudioSeconds = Math.Max(0.1, options.MaxTurnAudioSeconds)
    let optimizedModelCacheDir =
        if String.IsNullOrWhiteSpace options.OptimizedModelCacheDir then
            None
        else
            Some(fullPath options.OptimizedModelCacheDir)
    let cudaGpuMemLimitMb =
        if options.CudaGpuMemLimitMb.HasValue then
            Some options.CudaGpuMemLimitMb.Value
        else
            None
    let tuningOptions =
        { MemoryProfile = options.OrtMemoryProfile
          OptimizedModelCacheDir = optimizedModelCacheDir
          OptimizedModelCacheFormat = options.OptimizedModelCacheFormat
          CudaGpuMemLimitMb = cudaGpuMemLimitMb }
    let processor = ChromaNativeProcessor(modelDir, options.ThinkerActiveFrames)
    let runner = new ChromaS2sOnnxRunner(modelDir, bundleDir, options.ExecutionProvider, options.MemoryMode, tuningOptions)
    let store = RuntimeSessionStore(workDir, processor.PromptSampleRate)
    let workQueue = StreamingWorkQueue(maxQueueLength)
    let maxPromptAudioSamples = int (Math.Ceiling(maxPromptAudioSeconds * float processor.PromptSampleRate))
    let maxTurnAudioSamples = int (Math.Ceiling(maxTurnAudioSeconds * float processor.ThinkerSampleRate))

    do Directory.CreateDirectory(workDir) |> ignore

    let jsonElement payload =
        let json = JsonSerializer.Serialize(payload, jsonOptions)
        use doc = JsonDocument.Parse(json)
        doc.RootElement.Clone()

    let dimsOf (tensor: DenseTensor<'T>) =
        tensor.Dimensions.ToArray()

    let normalizeBackend (value: string | null) =
        let text =
            match Option.ofObj value with
            | Some item -> item
            | None -> ""
        match text.Trim().ToLowerInvariant() with
        | "" | "fsharp" | "fsharp_onnx" | "onnx" -> "fsharp_onnx"
        | "python" | "python_chroma" | "both" | "compare" ->
            invalidArg "backend" "ChromaOnnx.Service is F#/ONNX-only. Use backend 'fsharp_onnx'."
        | other -> invalidArg "backend" $"Unsupported backend '{other}'. Use fsharp_onnx."

    let backendLabel backend =
        match backend with
        | "fsharp_onnx" -> "F#/ONNX"
        | _ -> backend

    let safeId (value: string) =
        not (String.IsNullOrWhiteSpace value)
        && value |> Seq.forall (fun ch -> Char.IsLetterOrDigit(ch) || ch = '_' || ch = '-')

    let writeDetails path payload =
        let json = JsonSerializer.Serialize(payload, jsonOptions)
        File.WriteAllText(path, json)
        jsonElement payload

    let runFsharpBackend
        (session: RuntimeSession)
        (prepared: NativeS2sPrepared)
        (onFrame: S2sGeneratedFrame -> unit)
        (onAudioChunk: S2sAudioChunk -> unit)
        (shouldDecodeChunk: int -> bool)
        (cancellationToken: CancellationToken)
        =
        let backend = "fsharp_onnx"
        let backendDir = Path.Combine(session.WorkDir, backend)
        Directory.CreateDirectory(backendDir) |> ignore
        let memoryBefore = RuntimeMemory.current()
        use result =
            runner.GenerateStreaming(
                prepared,
                session.MaxNewFrames,
                streamDecodeFrames,
                onFrame,
                onAudioChunk,
                cancellationToken,
                ?shouldDecodeChunk = Some shouldDecodeChunk,
                ?codecStallGuardFrames = Some codecStallGuardFrames
            )
        let memoryAfter = RuntimeMemory.current()
        let codesPath = Path.Combine(backendDir, "audio_codes.i64")
        let rawAudioPath = Path.Combine(backendDir, "audio_values.f32")
        let wavPath = Path.Combine(backendDir, "audio.wav")
        TensorIO.writeInt64s codesPath result.AudioCodes
        TensorIO.writeSingles rawAudioPath result.AudioValues
        let wavStats = Wave.writeMono16 wavPath 24000 result.AudioValues
        let generationFrameRateFps =
            match result.Timings.TryGetValue "generateMs" with
            | true, generateMs when generateMs > 0.0 -> Nullable(Math.Round(float result.FrameCount / (generateMs / 1000.0), 3))
            | _ -> Nullable<float>()
        let truncatedByMaxFrames = result.StopReason = "max_frames"
        let detailsUrl = $"/api/s2s/sessions/{session.Id}/{backend}/details.json"
        let audioUrl = $"/api/s2s/sessions/{session.Id}/{backend}/audio.wav"
        let details =
            {| id = session.Id
               backend = backend
               label = backendLabel backend
               mode = "s2s_greedy"
               audioUrl = audioUrl
               detailsUrl = detailsUrl
               executionProvider = runner.Status.ExecutionProvider
               memoryMode = runner.MemoryMode
               ortMemoryProfile = runner.OrtMemoryProfile
               cudaGpuMemLimitMb = runner.CudaGpuMemLimitMb |> Option.toNullable
               optimizedModelCacheEnabled = runner.OptimizedModelCacheEnabled
               optimizedModelCacheDir = runner.OptimizedModelCacheDir
               optimizedModelCacheFormat = runner.OptimizedModelCacheFormat
               loadedOrtSessions = runner.LoadedSessionNames
               warmOrtSessions = runner.WarmSessionNames
               activePagedOrtSessions = runner.ActivePagedSessionNames
               mappedSafetensorShards = runner.MappedShardCount
               initializerCount = runner.InitializerCount
               uniqueInitializerSources = runner.UniqueSourceTensorCount
               uniqueOrtValues = runner.UniqueOrtValueCount
               sharedPrepackedWeights = runner.SharedPrepackedWeights
               promptText = session.PromptText
               systemPrompt = session.SystemPrompt
               requestedFrames = session.MaxNewFrames
               frameCount = result.FrameCount
               stopReason = result.StopReason
               truncatedByMaxFrames = truncatedByMaxFrames
               stepKinds = result.StepKinds
               promptAudioSamples = prepared.PromptAudioSamples
               userAudioSamples = prepared.UserAudioSamples
               effectiveUserAudioSamples = min prepared.UserAudioSamples processor.ThinkerTraceSamples
               thinkerFeatureMode = processor.ThinkerFeatureMode
               thinkerConfiguredActiveFrames = processor.ConfiguredThinkerActiveFrames
               thinkerTraceFeatureFrames = processor.ThinkerTraceFeatureFrames
               audioCodesShape = dimsOf result.AudioCodes
               audioValuesShape = dimsOf result.AudioValues
               timingsMs = result.Timings
               generationFrameRateFps = generationFrameRateFps
               memoryBefore = memoryBefore
               memoryAfter = memoryAfter
               peakPrivateGb = runner.PeakPrivateGb
               peakWorkingSetGb = runner.PeakWorkingSetGb
               wav = wavStats
               pythonInRequestPath = false |}
        let detailsElement = writeDetails (Path.Combine(backendDir, "details.json")) details
        { Backend = backend
          AudioUrl = audioUrl
          DetailsUrl = detailsUrl
          Details = detailsElement }

    let tryArtifactPath (session: RuntimeSession) (backend: string option) fileName =
        if not (safeId session.Id) then
            None
        else
            let path =
                match backend with
                | Some value when value <> "" ->
                    let normalized = normalizeBackend value
                    Path.Combine(session.WorkDir, normalized, fileName)
                | _ -> Path.Combine(session.WorkDir, fileName)
            if File.Exists path then Some path else None

    interface IS2sRuntime with
        member _.MaxPromptAudioSamples = maxPromptAudioSamples
        member _.MaxTurnAudioSamples = maxTurnAudioSamples
        member _.MaxQueueLength = workQueue.MaxQueueLength

        member _.Status() =
            let status = runner.Status
            { Ready = status.Ready
              ServiceName = "ChromaS2SONNX"
              Mode = "s2s_greedy_streaming"
              PythonInRequestPath = false
              ModelDir = modelDir
              BundleDir = bundleDir
              ExecutionProvider = status.ExecutionProvider
              MemoryMode = runner.MemoryMode
              OrtMemoryProfile = runner.OrtMemoryProfile
              CudaGpuMemLimitMb = runner.CudaGpuMemLimitMb |> Option.toNullable
              OptimizedModelCacheEnabled = runner.OptimizedModelCacheEnabled
              OptimizedModelCacheDir = runner.OptimizedModelCacheDir
              OptimizedModelCacheFormat = runner.OptimizedModelCacheFormat
              Memory = RuntimeMemory.current()
              LoadedOrtSessions = runner.LoadedSessionNames
              WarmOrtSessions = runner.WarmSessionNames
              ActivePagedOrtSessions = runner.ActivePagedSessionNames
              QueueLength = workQueue.QueueLength
              RunningRequestId = workQueue.RunningId
              MaxQueueLength = workQueue.MaxQueueLength
              StreamDecodeFrames = streamDecodeFrames
              StreamMinFreeVramMb = streamMinFreeVramMb
              CodecStallGuardFrames = codecStallGuardFrames
              MaxPromptAudioSeconds = maxPromptAudioSeconds
              MaxTurnAudioSeconds = maxTurnAudioSeconds
              GlobalGpuMemory = RuntimeMemory.tryGlobalGpuMemory()
              PeakPrivateGb = runner.PeakPrivateGb
              PeakWorkingSetGb = runner.PeakWorkingSetGb
              MappedSafetensorShards = runner.MappedShardCount
              InitializerCount = runner.InitializerCount
              UniqueInitializerSources = runner.UniqueSourceTensorCount
              UniqueOrtValues = runner.UniqueOrtValueCount
              SharedPrepackedWeights = runner.SharedPrepackedWeights
              Message = status.Message
              MissingGraphs = status.MissingGraphs
              AvailableGraphs = status.AvailableGraphs
              PromptSampleRate = processor.PromptSampleRate
              ThinkerSampleRate = processor.ThinkerSampleRate
              ThinkerFeatureMode = processor.ThinkerFeatureMode
              ThinkerConfiguredActiveFrames = processor.ConfiguredThinkerActiveFrames
              ThinkerTraceFeatureFrames = processor.ThinkerTraceFeatureFrames
              ThinkerTraceSamples = processor.ThinkerTraceSamples }

        member _.CreateSession(request: S2sSessionRequest) =
            if String.IsNullOrWhiteSpace request.PromptText then
                invalidArg "promptText" "promptText is required."
            if request.PromptAudio24k.Length = 0 then
                invalidArg "promptPcm24k" "promptPcm24k is required."
            if request.PromptAudio24k.Length > maxPromptAudioSamples then
                invalidArg
                    "promptPcm24k"
                    $"promptPcm24k is too large. The configured maximum is {maxPromptAudioSamples} Float32 samples."

            let backend = normalizeBackend request.Backend
            let maxNewFrames = max 1 (min 300 request.MaxNewFrames)
            let session = store.Create(request.PromptText, request.SystemPrompt, backend, request.PromptAudio24k, maxNewFrames)
            File.WriteAllBytes(Path.Combine(session.WorkDir, "prompt_audio_24k.f32"), ChromaOnnx.AudioChunk.float32ToLittleEndianBytes request.PromptAudio24k)
            store.ToInfo session

        member _.TryGetSession(id: string) =
            if safeId id then store.TryGetInfo id else None

        member _.RunTurnAsync(request: S2sTurnRequest, emit: S2sStreamingEvent -> Task, cancellationToken: CancellationToken) =
            task {
                if request.UserAudio16k.Length = 0 then
                    invalidArg "userAudio16k" "User turn audio is required."
                if request.UserAudio16k.Length > maxTurnAudioSamples then
                    invalidArg
                        "userAudio16k"
                        $"User turn audio is too large. The configured maximum is {maxTurnAudioSamples} Float32 samples."

                match store.TryGet request.SessionId with
                | None -> return invalidArg "sessionId" "S2S session was not found."
                | Some session ->
                    let requestId =
                        request.RequestId
                        |> Option.filter (String.IsNullOrWhiteSpace >> not)
                        |> Option.defaultWith (fun () -> $"{session.Id}_{Guid.NewGuid():N}")
                    let resultSource = TaskCompletionSource<S2sTurnResult>(TaskCreationOptions.RunContinuationsAsynchronously)
                    let emitBlocking event =
                        emit event |> fun work -> work.GetAwaiter().GetResult()
                    let notify snapshot =
                        try emitBlocking (QueueUpdated snapshot) with _ -> ()
                    let work (jobCancellationToken: CancellationToken) : Task =
                        task {
                            try
                                jobCancellationToken.ThrowIfCancellationRequested()
                                do! emit (QueueStarted(requestId, workQueue.QueueLength))
                                let turnAudioBytes = ChromaOnnx.AudioChunk.float32ToLittleEndianBytes request.UserAudio16k
                                File.WriteAllBytes(Path.Combine(session.WorkDir, "user_audio_16k.f32"), turnAudioBytes)
                                do!
                                    emit (
                                        GenerationStarted(
                                            requestId,
                                            session.MaxNewFrames,
                                            streamDecodeFrames,
                                            streamMinFreeVramMb,
                                            codecStallGuardFrames
                                        )
                                    )

                                use prepared = processor.Prepare(session.PromptText, session.SystemPrompt, session.PromptAudio24k, request.UserAudio16k)
                                File.WriteAllText(Path.Combine(session.WorkDir, "conversation.txt"), prepared.ConversationText)
                                let onFrame frame = emitBlocking (GenerationFrame(requestId, frame))
                                let onAudioChunk chunk = emitBlocking (AudioChunk(requestId, chunk))
                                let mutable lastDeferredFrame = -streamDecodeFrames
                                let shouldDecodeChunk currentFrameCount =
                                    if streamMinFreeVramMb <= 0
                                       || not (runner.Status.ExecutionProvider.Equals("cuda", StringComparison.OrdinalIgnoreCase)) then
                                        true
                                    else
                                        match RuntimeMemory.tryGlobalGpuMemory() with
                                        | None -> true
                                        | Some gpu when gpu.FreeMb >= streamMinFreeVramMb -> true
                                        | Some gpu ->
                                            if currentFrameCount - lastDeferredFrame >= streamDecodeFrames then
                                                lastDeferredFrame <- currentFrameCount
                                                emitBlocking (
                                                    AudioDeferred(
                                                        requestId,
                                                        { FrameCount = currentFrameCount
                                                          FreeVramMb = gpu.FreeMb
                                                          UsedVramMb = gpu.UsedMb
                                                          TotalVramMb = gpu.TotalMb
                                                          MinFreeVramMb = streamMinFreeVramMb
                                                          Message = "Deferred partial audio decode to keep CUDA memory below the configured headroom." }
                                                    )
                                                )
                                            false

                                let backendResult =
                                    runFsharpBackend
                                        session
                                        prepared
                                        onFrame
                                        onAudioChunk
                                        shouldDecodeChunk
                                        jobCancellationToken

                                let backendDir = Path.Combine(session.WorkDir, backendResult.Backend)
                                let backendAudioPath = Path.Combine(backendDir, "audio.wav")
                                if File.Exists backendAudioPath then
                                    File.Copy(backendAudioPath, Path.Combine(session.WorkDir, "audio.wav"), true)

                                let details =
                                    {| id = session.Id
                                       requestId = requestId
                                       mode = "s2s_greedy_streaming"
                                       backend = session.Backend
                                       maxNewFrames = session.MaxNewFrames
                                       streamDecodeFrames = streamDecodeFrames
                                       streamMinFreeVramMb = streamMinFreeVramMb
                                       codecStallGuardFrames = codecStallGuardFrames
                                       pythonInRequestPath = false
                                       results = [| backendResult.Details |] |}
                                let detailsJson = JsonSerializer.Serialize(details, jsonOptions)
                                File.WriteAllText(Path.Combine(session.WorkDir, "details.json"), detailsJson)
                                use doc = JsonDocument.Parse(detailsJson)
                                session.LastDetails <- Some(doc.RootElement.Clone())
                                let turnResult =
                                    { Id = session.Id
                                      RequestId = requestId
                                      Backend = session.Backend
                                      AudioUrl = backendResult.AudioUrl
                                      DetailsUrl = backendResult.DetailsUrl
                                      Results = [| backendResult |] }
                                resultSource.TrySetResult(turnResult) |> ignore
                                do! emit (GenerationDone turnResult)
                            with
                            | :? OperationCanceledException as ex ->
                                resultSource.TrySetCanceled(ex.CancellationToken) |> ignore
                                do! emit (GenerationCanceled(session.Id, Some requestId))
                                raise ex
                            | ex ->
                                resultSource.TrySetException(ex) |> ignore
                                raise ex
                        }
                        :> Task

                    match workQueue.TryEnqueue(requestId, work, notify, cancellationToken) with
                    | QueueFull maxQueueLength ->
                        return invalidOp $"Generation queue is full. Try again after a current request finishes. Max queue length: {maxQueueLength}."
                    | Enqueued(handle, snapshot) ->
                        do! emit (QueueEnqueued(requestId, snapshot))
                        try
                            do! handle.Completion
                            return! resultSource.Task
                        with
                        | :? OperationCanceledException as ex ->
                            resultSource.TrySetCanceled(ex.CancellationToken) |> ignore
                            do! emit (GenerationCanceled(session.Id, Some requestId))
                            return raise ex
                        | ex ->
                            resultSource.TrySetException(ex) |> ignore
                            return raise ex
            }

        member _.TryGetArtifact(sessionId: string, backend: string option, fileName: string) =
            if not (safeId sessionId) then
                None
            else
                match store.TryGet sessionId with
                | None -> None
                | Some session ->
                    let contentType =
                        match fileName with
                        | "details.json" -> Some "application/json; charset=utf-8"
                        | "audio.wav" -> Some "audio/wav"
                        | _ -> None
                    match contentType, tryArtifactPath session backend fileName with
                    | Some mediaType, Some path -> Some { Path = path; ContentType = mediaType }
                    | _ -> None

    interface IDisposable with
        member _.Dispose() =
            (runner :> IDisposable).Dispose()
