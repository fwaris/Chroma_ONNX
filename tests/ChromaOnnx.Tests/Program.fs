namespace ChromaOnnx.Tests

open System
open System.Collections.Concurrent
open System.Collections.Generic
open System.Diagnostics
open System.IO
open System.Net
open System.Net.Http
open System.Net.WebSockets
open System.Text
open System.Text.Json
open System.Threading
open System.Threading.Tasks
open Microsoft.AspNetCore.Builder
open Microsoft.AspNetCore.Hosting
open Microsoft.AspNetCore.TestHost
open Microsoft.ML.OnnxRuntime.Tensors
open Microsoft.Extensions.Configuration
open Microsoft.Extensions.Logging
open ChromaOnnx
open ChromaOnnx.Service
open ChromaOnnx.SpeechToSpeech

module Program =
    let private fail name message =
        failwith $"{name}: {message}"

    let private assertEqual name expected actual =
        if not (Object.Equals(expected, actual)) then
            fail name $"expected {expected}, got {actual}"

    let private assertArrayEqual name (expected: 'T array) (actual: 'T array) =
        if expected.Length <> actual.Length then
            fail name $"expected length {expected.Length}, got {actual.Length}"
        for index in 0 .. expected.Length - 1 do
            if not (Object.Equals(expected[index], actual[index])) then
                fail name $"at {index}: expected {expected[index]}, got {actual[index]}"

    let private waitUntil name (predicate: unit -> bool) =
        let stopwatch = Stopwatch.StartNew()
        while not (predicate ()) && stopwatch.Elapsed < TimeSpan.FromSeconds(5.0) do
            Thread.Sleep 10
        if not (predicate ()) then
            fail name "condition was not reached before timeout"

    let private expectEnqueued name result =
        match result with
        | Enqueued(handle, snapshot) -> handle, snapshot
        | QueueFull maxQueueLength -> fail name $"queue was full at {maxQueueLength}"

    let private waitTask (task: Task) =
        task.GetAwaiter().GetResult()

    let private waitTaskResult (task: Task<'T>) =
        task.GetAwaiter().GetResult()

    let private jsonElement payload =
        let json = JsonSerializer.Serialize(payload, JsonSerializerOptions(PropertyNamingPolicy = JsonNamingPolicy.CamelCase))
        use doc = JsonDocument.Parse(json)
        doc.RootElement.Clone()

    let private fifoOrdering () =
        let queue = StreamingWorkQueue(8)
        let order = ConcurrentQueue<string>()
        let work id (_: CancellationToken) : Task =
            task {
                order.Enqueue id
            }
            :> Task

        let first, _ = queue.TryEnqueue("first", work "first", ignore, CancellationToken.None) |> expectEnqueued "fifo first"
        let second, _ = queue.TryEnqueue("second", work "second", ignore, CancellationToken.None) |> expectEnqueued "fifo second"
        let third, _ = queue.TryEnqueue("third", work "third", ignore, CancellationToken.None) |> expectEnqueued "fifo third"
        Task.WaitAll([| first.Completion; second.Completion; third.Completion |])
        assertArrayEqual "fifo ordering" [| "first"; "second"; "third" |] (order.ToArray())

    let private queueFull () =
        let queue = StreamingWorkQueue(1)
        let blocker = TaskCompletionSource<unit>(TaskCreationOptions.RunContinuationsAsynchronously)
        let running, _ =
            queue.TryEnqueue(
                "running",
                (fun _ -> blocker.Task :> Task),
                ignore,
                CancellationToken.None
            )
            |> expectEnqueued "queue full running"
        waitUntil "queue full running id" (fun () -> queue.RunningId = Some "running")
        let waiting, waitingSnapshot =
            queue.TryEnqueue(
                "waiting",
                (fun _ -> Task.CompletedTask),
                ignore,
                CancellationToken.None
            )
            |> expectEnqueued "queue full waiting"
        assertEqual "queue full waiting position" 1 waitingSnapshot.Position
        match queue.TryEnqueue("overflow", (fun _ -> Task.CompletedTask), ignore, CancellationToken.None) with
        | QueueFull maxQueueLength -> assertEqual "queue full max" 1 maxQueueLength
        | Enqueued _ -> fail "queue full" "overflow request was accepted"
        blocker.SetResult(())
        Task.WaitAll([| running.Completion; waiting.Completion |])

    let private queuePositionUpdates () =
        let queue = StreamingWorkQueue(4)
        let blocker = TaskCompletionSource<unit>(TaskCreationOptions.RunContinuationsAsynchronously)
        let updates = ConcurrentQueue<WorkQueuePosition>()
        let notify snapshot = updates.Enqueue snapshot
        let running, _ =
            queue.TryEnqueue("running", (fun _ -> blocker.Task :> Task), ignore, CancellationToken.None)
            |> expectEnqueued "position running"
        waitUntil "position running id" (fun () -> queue.RunningId = Some "running")
        let waitingOne, _ =
            queue.TryEnqueue("waiting-one", (fun _ -> Task.CompletedTask), notify, CancellationToken.None)
            |> expectEnqueued "position waiting one"
        let waitingTwo, _ =
            queue.TryEnqueue("waiting-two", (fun _ -> Task.CompletedTask), notify, CancellationToken.None)
            |> expectEnqueued "position waiting two"
        waitUntil "position update" (fun () ->
            updates
            |> Seq.exists (fun update -> update.Id = "waiting-two" && update.Position = 2 && update.QueueLength = 2))
        blocker.SetResult(())
        Task.WaitAll([| running.Completion; waitingOne.Completion; waitingTwo.Completion |])

    let private queuedCancellation () =
        let queue = StreamingWorkQueue(4)
        let blocker = TaskCompletionSource<unit>(TaskCreationOptions.RunContinuationsAsynchronously)
        let running, _ =
            queue.TryEnqueue("running", (fun _ -> blocker.Task :> Task), ignore, CancellationToken.None)
            |> expectEnqueued "cancel running"
        waitUntil "cancel running id" (fun () -> queue.RunningId = Some "running")
        use cancellation = new CancellationTokenSource()
        let waiting, _ =
            queue.TryEnqueue("waiting", (fun _ -> Task.CompletedTask), ignore, cancellation.Token)
            |> expectEnqueued "cancel waiting"
        cancellation.Cancel()
        waitUntil "cancel queue length" (fun () -> queue.QueueLength = 0)
        try
            waitTask waiting.Completion
            fail "queued cancellation" "completion was not canceled"
        with
        | :? TaskCanceledException -> ()
        blocker.SetResult(())
        waitTask running.Completion

    let private float32ChunkRoundtrip () =
        let samples = [| 0.0f; -1.0f; 0.25f; 1.0f / 3.0f |]
        let bytes = AudioChunk.float32ToLittleEndianBytes samples
        assertEqual "chunk byte length" (samples.Length * sizeof<float32>) bytes.Length
        let decoded = AudioChunk.float32FromLittleEndianBytes bytes
        assertArrayEqual "chunk roundtrip" samples decoded

    type private FixedRandom(values: float array) =
        inherit Random()
        let mutable index = 0
        override _.Sample() =
            let value = values[min index (values.Length - 1)]
            index <- index + 1
            value

    let private sampleOne rng temperature topP topK (values: float32 array) =
        let logits = DenseTensor<float32>(values, [| 1; 1; values.Length |])
        TensorMath.sampleLast rng temperature topP topK logits
        |> Array.exactlyOne

    let private sampleChromaOne rng temperature topK (values: float32 array) =
        let logits = DenseTensor<float32>(values, [| 1; 1; values.Length |])
        TensorMath.sampleChromaTopKLast rng temperature topK logits
        |> Array.exactlyOne

    let private samplingLogits () =
        assertEqual
            "sampling top-k one"
            0L
            (sampleOne (FixedRandom [| 0.999 |]) 1.0 1.0 1 [| 10.0f; 9.0f; 8.0f |])

        assertEqual
            "sampling top-k excludes lower logits"
            1L
            (sampleOne (FixedRandom [| 0.999 |]) 1.0 1.0 2 [| 10.0f; 9.0f; 8.0f |])

        assertEqual
            "sampling top-p keeps nucleus only"
            1L
            (sampleOne (FixedRandom [| 0.999 |]) 1.0 0.8 4 [| 0.0f; -0.1f; -4.0f; -5.0f |])

        assertEqual
            "sampling zero top-p disables nucleus"
            1L
            (sampleOne (FixedRandom [| 0.999 |]) 1.0 0.0 2 [| 10.0f; 9.0f; 8.0f |])

        assertEqual
            "sampling zero temperature greedy"
            0L
            (sampleOne (FixedRandom [| 0.999 |]) 0.0 1.0 0 [| 2.0f; 0.0f |])

        assertEqual
            "sampling low temperature sharpens"
            0L
            (sampleOne (FixedRandom [| 0.8 |]) 0.5 1.0 0 [| 2.0f; 0.0f |])

        assertEqual
            "sampling high temperature broadens"
            1L
            (sampleOne (FixedRandom [| 0.8 |]) 2.0 1.0 0 [| 2.0f; 0.0f |])

        assertEqual
            "sampling ignores nan logits"
            1L
            (sampleOne (FixedRandom [| 0.0 |]) 1.0 1.0 0 [| Single.NaN; 0.0f |])

        assertEqual
            "chroma sampling top-k exponential race"
            1L
            (sampleChromaOne (FixedRandom [| 0.9; 0.1 |]) 1.0 2 [| 10.0f; 9.0f; 8.0f |])

        assertEqual
            "chroma sampling keeps top-k cutoff ties"
            2L
            (sampleChromaOne (FixedRandom [| 0.9; 0.9; 0.1 |]) 1.0 2 [| 10.0f; 9.0f; 9.0f; 0.0f |])

        assertEqual
            "chroma sampling ignores top-p by design"
            2L
            (sampleChromaOne (FixedRandom [| 0.9; 0.9; 0.000001; 0.9 |]) 1.0 4 [| 0.0f; -0.1f; -4.0f; -5.0f |])

    type private FakeRuntime(workDir: string) =
        let sessions = ConcurrentDictionary<string, S2sSessionInfo>(StringComparer.Ordinal)
        let turnIndexes = ConcurrentDictionary<string, int>(StringComparer.Ordinal)

        let sessionDir id =
            Path.Combine(workDir, "s2s", id)

        let turnDir id (turnIndex: int) =
            Path.Combine(sessionDir id, "turns", turnIndex.ToString("0000", Globalization.CultureInfo.InvariantCulture))

        let ensureArtifacts id turnIndex =
            let dir = sessionDir id
            let turnRoot = turnDir id turnIndex
            let backendDir = Path.Combine(turnRoot, "fsharp_onnx")
            let legacyBackendDir = Path.Combine(dir, "fsharp_onnx")
            Directory.CreateDirectory(backendDir) |> ignore
            Directory.CreateDirectory(legacyBackendDir) |> ignore
            let details = $"""{{"backend":"fsharp_onnx","mode":"fake","turnIndex":{turnIndex}}}"""
            File.WriteAllText(Path.Combine(turnRoot, "details.json"), details)
            File.WriteAllText(Path.Combine(dir, "details.json"), details)
            File.WriteAllText(Path.Combine(backendDir, "details.json"), details)
            File.WriteAllText(Path.Combine(legacyBackendDir, "details.json"), details)
            File.WriteAllBytes(Path.Combine(dir, "audio.wav"), [| byte 'R'; byte 'I'; byte 'F'; byte 'F' |])
            File.WriteAllBytes(Path.Combine(backendDir, "audio.wav"), [| byte 'R'; byte 'I'; byte 'F'; byte 'F' |])
            File.WriteAllBytes(Path.Combine(legacyBackendDir, "audio.wav"), [| byte 'R'; byte 'I'; byte 'F'; byte 'F' |])

        interface IS2sRuntime with
            member _.MaxPromptAudioSamples = 24000
            member _.MaxTurnAudioSamples = 16000
            member _.MaxQueueLength = 4

            member _.Status() =
                { Ready = true
                  ServiceName = "ChromaS2SONNX"
                  Mode = "s2s_greedy_streaming"
                  PythonInRequestPath = false
                  ModelDir = "models/test"
                  BundleDir = "onnx/test"
                  ExecutionProvider = "cpu"
                  MemoryMode = "resident-merged"
                  OrtMemoryProfile = "quality-safe"
                  CudaGpuMemLimitMb = Nullable<int>()
                  OptimizedModelCacheEnabled = false
                  OptimizedModelCacheDir = null
                  OptimizedModelCacheFormat = "onnx"
                  Memory = RuntimeMemory.current()
                  LoadedOrtSessions = [| "fake" |]
                  WarmOrtSessions = Array.empty
                  ActivePagedOrtSessions = Array.empty
                  QueueLength = 0
                  RunningRequestId = None
                  MaxQueueLength = 4
                  StreamDecodeFrames = 4
                  StreamMinFreeVramMb = 0
                  CodecStallGuardFrames = 0
                  GenerationMode = "greedy"
                  SamplingAlgorithm = "top-k-top-p"
                  SamplingTemperature = 1.0
                  SamplingTopP = 1.0
                  SamplingTopK = 0
                  MaxNewFrames = 900
                  MaxPromptAudioSeconds = 60.0
                  MaxTurnAudioSeconds = 60.0
                  GlobalGpuMemory = None
                  PeakPrivateGb = 0.0
                  PeakWorkingSetGb = 0.0
                  MappedSafetensorShards = 0
                  InitializerCount = 0
                  UniqueInitializerSources = 0
                  UniqueOrtValues = 0
                  SharedPrepackedWeights = false
                  Message = "fake ready"
                  MissingGraphs = Array.empty
                  AvailableGraphs = [| "fake" |]
                  PromptSampleRate = 24000
                  ThinkerSampleRate = 16000
                  BundleGraphMode = "one-shot"
                  BundleThinkerFeatureMode = "dynamic_batch1_full_length"
                  ThinkerMaxAudioItems = 1
                  ThinkerFeatureMode = "fake"
                  ThinkerConfiguredActiveFrames = 0
                  ThinkerTraceFeatureFrames = 0
                  ThinkerTraceSamples = 0 }

            member _.CreateSession(request: S2sSessionRequest) =
                match request.Backend.Trim().ToLowerInvariant() with
                | "" | "fsharp" | "fsharp_onnx" | "onnx" -> ()
                | _ -> invalidArg "backend" "ChromaOnnx.Service is F#/ONNX-only. Use backend 'fsharp_onnx'."

                let id = $"test_{Guid.NewGuid():N}"
                let info =
                    { Id = id
                      ServiceName = "ChromaS2SONNX"
                      Mode = "s2s_greedy_streaming"
                      Backend = "fsharp_onnx"
                      PromptText = request.PromptText
                      SystemPrompt = request.SystemPrompt
                      MaxNewFrames = request.MaxNewFrames
                      MaxResponseSeconds = Math.Round(float request.MaxNewFrames * 0.08, 2)
                      PromptAudioSamples = request.PromptAudio24k.Length
                      PromptSampleRate = 24000
                      WebsocketUrl = $"/ws/s2s/{id}"
                      CreatedUtc = DateTimeOffset.UtcNow }
                sessions[id] <- info
                turnIndexes[id] <- 0
                Directory.CreateDirectory(sessionDir id) |> ignore
                info

            member _.TryGetSession(id: string) =
                match sessions.TryGetValue(id) with
                | true, session -> Some session
                | false, _ -> None

            member _.RunTurnAsync(request: S2sTurnRequest, emit: S2sStreamingEvent -> Task, cancellationToken: CancellationToken) =
                task {
                    cancellationToken.ThrowIfCancellationRequested()
                    match sessions.TryGetValue(request.SessionId) with
                    | false, _ -> return invalidArg "sessionId" "S2S session was not found."
                    | true, session ->
                        let requestId = request.RequestId |> Option.defaultValue $"{session.Id}_{Guid.NewGuid():N}"
                        let turnIndex = turnIndexes.AddOrUpdate(session.Id, 1, fun _ previous -> previous + 1)
                        if turnIndex <> 1 then
                            invalidOp "Chroma S2S sessions are single-turn. Create a new session for another Chroma generation."
                        let context =
                            { TurnIndex = turnIndex }
                        let runningSnapshot =
                            { Id = requestId
                              Position = 0
                              QueueLength = 0
                              RunningId = Some requestId
                              IsRunning = true }
                        do! emit (QueueEnqueued(requestId, runningSnapshot))
                        do! emit (QueueStarted(requestId, 0))
                        do! emit (GenerationStarted(requestId, session.MaxNewFrames, 4, 0, 0, context))
                        do! emit (GenerationFrame(requestId, { FrameIndex = 0; StepKind = "frame"; IsEos = false; Codes = [| 1L; 2L |] }))
                        do! emit (AudioChunk(requestId, { ChunkIndex = 0; StartFrame = 0; FrameCount = 1; StartSample = 0; SampleRate = 24000; Samples = [| 0.0f; 0.1f |] }))
                        ensureArtifacts session.Id turnIndex
                        let details = jsonElement {| backend = "fsharp_onnx"; mode = "fake"; turnIndex = turnIndex |}
                        let backendResult =
                            { Backend = "fsharp_onnx"
                              AudioUrl = $"/api/s2s/sessions/{session.Id}/turns/{turnIndex}/fsharp_onnx/audio.wav"
                              DetailsUrl = $"/api/s2s/sessions/{session.Id}/turns/{turnIndex}/fsharp_onnx/details.json"
                              Details = details }
                        let result =
                            { Id = session.Id
                              RequestId = requestId
                              TurnIndex = turnIndex
                              Backend = "fsharp_onnx"
                              AudioUrl = backendResult.AudioUrl
                              DetailsUrl = backendResult.DetailsUrl
                              Results = [| backendResult |] }
                        do! emit (GenerationDone result)
                        return result
                }

            member _.TryGetArtifact(sessionId: string, backend: string option, fileName: string) =
                let dir =
                    match backend with
                    | Some value when not (String.IsNullOrWhiteSpace value) -> Path.Combine(sessionDir sessionId, value)
                    | _ -> sessionDir sessionId
                let path = Path.Combine(dir, fileName)
                if File.Exists path then
                    let contentType =
                        if fileName.EndsWith(".json", StringComparison.OrdinalIgnoreCase) then
                            "application/json; charset=utf-8"
                        else
                            "audio/wav"
                    Some { Path = path; ContentType = contentType }
                else
                    None

            member _.TryGetTurnArtifact(sessionId: string, turnIndex: int, backend: string option, fileName: string) =
                let dir =
                    match backend with
                    | Some value when not (String.IsNullOrWhiteSpace value) -> Path.Combine(turnDir sessionId turnIndex, value)
                    | _ -> turnDir sessionId turnIndex
                let path = Path.Combine(dir, fileName)
                if File.Exists path then
                    let contentType =
                        if fileName.EndsWith(".json", StringComparison.OrdinalIgnoreCase) then
                            "application/json; charset=utf-8"
                        else
                            "audio/wav"
                    Some { Path = path; ContentType = contentType }
                else
                    None

    type private FakeGemmaRuntime() =
        interface IGemmaRuntime with
            member _.Status() =
                { Ready = true
                  ModelDir = "models/fake-gemma"
                  Variant = "fake"
                  ExecutionProvider = "cpu"
                  MissingFiles = Array.empty
                  LoadedSessions = [| "fake" |]
                  Message = "fake gemma ready" }

            member _.Prepare(request: GemmaGenerationRequest) =
                { Prompt = "fake"
                  InputIds = DenseTensor<int64>([| 1L |], [| 1; 1 |])
                  AttentionMask = DenseTensor<int64>([| 1L |], [| 1; 1 |])
                  AudioFeatures = None }

            member _.GenerateAsync(request: GemmaGenerationRequest, cancellationToken: CancellationToken) =
                task {
                    cancellationToken.ThrowIfCancellationRequested()
                    let text =
                        match request.Audio16k with
                        | Some _ -> "what time is it"
                        | None ->
                            let hasToolResult =
                                request.Messages
                                |> Array.exists (fun message -> message.Role = GemmaChatRole.Tool)
                            if hasToolResult then
                                "It is time to test the agent."
                            else
                                """<|tool_call>call:get_current_time{}<tool_call|>"""
                    return
                        { Text = text
                          Prompt = "fake"
                          InputTokenCount = 1
                          OutputTokenIds = [| 1L |]
                          StopReason = "fake"
                          TimingsMs = Map.empty }
                }

    let private withTestApp (test: WebApplication -> HttpClient -> Task) =
        let workDir = Path.Combine(Path.GetTempPath(), $"chroma-onnx-tests-{Guid.NewGuid():N}")
        Directory.CreateDirectory(workDir) |> ignore
        let builder = WebApplication.CreateBuilder([||])
        builder.Logging.ClearProviders() |> ignore
        builder.WebHost.UseTestServer() |> ignore
        let app = builder.Build()
        let runtime = FakeRuntime(workDir) :> IS2sRuntime
        S2sWebApp.map app runtime |> ignore
        try
            waitTask (app.StartAsync())
            use client = app.GetTestClient()
            waitTask (test app client)
        finally
            waitTask (app.StopAsync())
            (app :> IDisposable).Dispose()
            if Directory.Exists workDir then
                Directory.Delete(workDir, true)

    let private withAgentTestApp (test: WebApplication -> HttpClient -> Task) =
        let workDir = Path.Combine(Path.GetTempPath(), $"chroma-onnx-agent-tests-{Guid.NewGuid():N}")
        Directory.CreateDirectory(workDir) |> ignore
        let builder = WebApplication.CreateBuilder([||])
        builder.Logging.ClearProviders() |> ignore
        builder.WebHost.UseTestServer() |> ignore
        let app = builder.Build()
        let runtime = FakeRuntime(workDir) :> IS2sRuntime
        let gemma = FakeGemmaRuntime() :> IGemmaRuntime
        let gemmaOptions = GemmaRuntimeOptions()
        let agent =
            new GemmaChromaAgentRuntime(gemmaOptions, runtime, gemmaRuntime = gemma, workDir = workDir)
            :> IAgentRuntime
        S2sWebApp.mapWithAgent app runtime agent |> ignore
        try
            waitTask (app.StartAsync())
            use client = app.GetTestClient()
            waitTask (test app client)
        finally
            waitTask (app.StopAsync())
            (agent :?> IDisposable).Dispose()
            (app :> IDisposable).Dispose()
            if Directory.Exists workDir then
                Directory.Delete(workDir, true)

    let private multipartSessionContent backend =
        let content = new MultipartFormDataContent()
        content.Add(new StringContent("You are Chroma, an advanced virtual human created by the FlashLabs. You possess the ability to understand auditory inputs and generate both text and speech."), "systemPrompt")
        content.Add(new StringContent("Reference text."), "promptText")
        content.Add(new StringContent(backend), "backend")
        content.Add(new StringContent("4"), "maxNewFrames")
        let promptBytes = AudioChunk.float32ToLittleEndianBytes [| 0.0f; 0.1f; -0.1f |]
        content.Add(new ByteArrayContent(promptBytes), "promptPcm24k", "prompt.f32")
        content

    let private serviceRejectsPythonBackend () =
        withTestApp (fun _ client ->
            task {
                use content = multipartSessionContent "python"
                let! response = client.PostAsync("/api/s2s/sessions", content)
                assertEqual "python backend rejection" HttpStatusCode.BadRequest response.StatusCode
                let! body = response.Content.ReadAsStringAsync()
                if not (body.Contains("F#/ONNX-only", StringComparison.OrdinalIgnoreCase)) then
                    fail "python backend rejection" $"unexpected body: {body}"
            })

    let private serviceWebSocketAndArtifacts () =
        let receiveMessage (socket: WebSocket) =
            task {
                let buffer = Array.zeroCreate<byte> 4096
                use stream = new MemoryStream()
                let mutable messageType = WebSocketMessageType.Close
                let mutable complete = false
                while not complete do
                    let! result = socket.ReceiveAsync(ArraySegment<byte>(buffer), CancellationToken.None)
                    messageType <- result.MessageType
                    if result.Count > 0 then
                        stream.Write(buffer, 0, result.Count)
                    complete <- result.EndOfMessage || result.MessageType = WebSocketMessageType.Close
                return messageType, stream.ToArray()
            }

        let sendText (socket: WebSocket) (text: string) =
            let bytes = Encoding.UTF8.GetBytes(text)
            socket.SendAsync(ArraySegment<byte>(bytes), WebSocketMessageType.Text, true, CancellationToken.None)

        let sendBinary (socket: WebSocket) bytes =
            socket.SendAsync(ArraySegment<byte>(bytes), WebSocketMessageType.Binary, true, CancellationToken.None)

        withTestApp (fun app client ->
            task {
                use content = multipartSessionContent "fsharp_onnx"
                let! response = client.PostAsync("/api/s2s/sessions", content)
                assertEqual "session create" HttpStatusCode.OK response.StatusCode
                let! sessionJson = response.Content.ReadAsStringAsync()
                use sessionDoc = JsonDocument.Parse(sessionJson)
                let sessionId = sessionDoc.RootElement.GetProperty("id").GetString()
                let websocketUrl = sessionDoc.RootElement.GetProperty("websocketUrl").GetString()

                let wsClient = app.GetTestServer().CreateWebSocketClient()
                use! socket = wsClient.ConnectAsync(Uri($"ws://localhost{websocketUrl}"), CancellationToken.None)
                let seen = ResizeArray<string>()
                let mutable binaryMessages = 0
                let receiveTextPayload () =
                    task {
                        let! messageType, payload = receiveMessage socket
                        match messageType with
                        | WebSocketMessageType.Text ->
                            let text = Encoding.UTF8.GetString(payload)
                            use doc = JsonDocument.Parse(text)
                            let eventType =
                                doc.RootElement.GetProperty("type").GetString()
                                |> Option.ofObj
                                |> Option.defaultValue ""
                            seen.Add(eventType)
                            return eventType, Some(doc.RootElement.Clone())
                        | WebSocketMessageType.Binary ->
                            binaryMessages <- binaryMessages + 1
                            return "binary", None
                        | _ -> return "close", None
                    }

                let receiveTextType () =
                    task {
                        let! eventType, _ = receiveTextPayload ()
                        return eventType
                    }

                let sendTurn samples =
                    task {
                        do! sendText socket """{"type":"turn.start"}"""
                        let! accepted = receiveTextType ()
                        assertEqual "turn accepted" "turn.accepted" accepted
                        do! sendBinary socket (AudioChunk.float32ToLittleEndianBytes samples)
                        let! chunkAck = receiveTextType ()
                        assertEqual "turn chunk" "turn.chunk" chunkAck
                        do! sendText socket """{"type":"turn.end"}"""

                        let mutable donePayload: JsonElement option = None
                        let stopwatch = Stopwatch.StartNew()
                        while donePayload.IsNone && stopwatch.Elapsed < TimeSpan.FromSeconds(5.0) do
                            let! eventType, payload = receiveTextPayload ()
                            if eventType = "generation.done" then
                                donePayload <- payload

                        match donePayload with
                        | None ->
                            let seenEvents = String.Join(",", seen)
                            return fail "websocket generation" $"generation.done was not observed. Events: {seenEvents}"
                        | Some payload -> return payload.GetProperty("turnIndex").GetInt32()
                    }

                let! first = receiveTextType ()
                assertEqual "socket ready" "session.ready" first
                let! firstTurn = sendTurn [| 0.0f; 0.2f |]
                assertEqual "first turn index" 1 firstTurn
                if binaryMessages < 1 then
                    fail "websocket generation" "expected streamed audio binary payload for the turn"

                let! detailsResponse = client.GetAsync($"/api/s2s/sessions/{sessionId}/details.json")
                assertEqual "details artifact" HttpStatusCode.OK detailsResponse.StatusCode
                let! audioResponse = client.GetAsync($"/api/s2s/sessions/{sessionId}/fsharp_onnx/audio.wav")
                assertEqual "audio artifact" HttpStatusCode.OK audioResponse.StatusCode
                let! turnDetailsResponse = client.GetAsync($"/api/s2s/sessions/{sessionId}/turns/1/fsharp_onnx/details.json")
                assertEqual "turn details artifact" HttpStatusCode.OK turnDetailsResponse.StatusCode
                let! turnAudioResponse = client.GetAsync($"/api/s2s/sessions/{sessionId}/turns/1/fsharp_onnx/audio.wav")
                assertEqual "turn audio artifact" HttpStatusCode.OK turnAudioResponse.StatusCode
            })

    let private gemmaRenderingAndParsing () =
        let processor = GemmaProcessor(Path.Combine(Path.GetTempPath(), $"missing-gemma-{Guid.NewGuid():N}"))
        assertEqual "gemma 1s audio tokens" 25 (processor.ComputeAudioTokenCount 16000)
        assertEqual "gemma 10s audio tokens" 250 (processor.ComputeAudioTokenCount 160000)
        assertEqual "gemma 30s audio tokens" 750 (processor.ComputeAudioTokenCount 480000)

        let prompt =
            processor.RenderChat(
                [| GemmaChatMessage.system "system"
                   GemmaChatMessage.user "Please transcribe <|audio|>." |],
                [| { Name = "get_current_time"
                     Description = "Return the current time."
                     Parameters = Array.empty } |],
                true,
                [| for _ in 1 .. 16000 -> 0.0f |]
            )
        if not (prompt.StartsWith("<bos><|turn>system", StringComparison.Ordinal)) then
            fail "gemma prompt" $"unexpected prompt start: {prompt.Substring(0, min prompt.Length 80)}"
        if not (prompt.Contains("<|turn>user", StringComparison.Ordinal)) then
            fail "gemma prompt" "user turn was not rendered"
        if not (prompt.Contains("<|turn>model\n", StringComparison.Ordinal)) then
            fail "gemma prompt" "generation prompt was not rendered"
        if not (prompt.Contains(processor.AudioToken, StringComparison.Ordinal)) then
            fail "gemma prompt" "audio token expansion was not rendered"

        match processor.TryParseToolCall("""<|tool_call>call:get_current_time{}<tool_call|>""") with
        | Some call -> assertEqual "gemma tool name" "get_current_time" call.Name
        | None -> fail "gemma tool parse" "tool call was not parsed"

        match processor.TryParseToolCall("""call:get_current_time{}""") with
        | Some call -> assertEqual "gemma bare tool name" "get_current_time" call.Name
        | None -> fail "gemma bare tool parse" "bare tool call was not parsed"

        match processor.TryParseToolCall("""<|tool_call>call:echo{text:<|"|>hi<|"|>}<tool_call|>""") with
        | Some call -> assertEqual "gemma tool argument" "hi" call.Arguments["text"]
        | None -> fail "gemma tool parse args" "tool call with args was not parsed"

    let private agentWebSocketAndDetails () =
        let receiveMessage (socket: WebSocket) =
            task {
                let buffer = Array.zeroCreate<byte> 4096
                use stream = new MemoryStream()
                let mutable messageType = WebSocketMessageType.Close
                let mutable complete = false
                while not complete do
                    let! result = socket.ReceiveAsync(ArraySegment<byte>(buffer), CancellationToken.None)
                    messageType <- result.MessageType
                    if result.Count > 0 then
                        stream.Write(buffer, 0, result.Count)
                    complete <- result.EndOfMessage || result.MessageType = WebSocketMessageType.Close
                return messageType, stream.ToArray()
            }

        let sendText (socket: WebSocket) (text: string) =
            let bytes = Encoding.UTF8.GetBytes(text)
            socket.SendAsync(ArraySegment<byte>(bytes), WebSocketMessageType.Text, true, CancellationToken.None)

        let sendBinary (socket: WebSocket) bytes =
            socket.SendAsync(ArraySegment<byte>(bytes), WebSocketMessageType.Binary, true, CancellationToken.None)

        withAgentTestApp (fun app client ->
            task {
                use content = multipartSessionContent "fsharp_onnx"
                let! response = client.PostAsync("/api/agent/sessions", content)
                assertEqual "agent session create" HttpStatusCode.OK response.StatusCode
                let! sessionJson = response.Content.ReadAsStringAsync()
                use sessionDoc = JsonDocument.Parse(sessionJson)
                let sessionId = sessionDoc.RootElement.GetProperty("id").GetString()
                let websocketUrl = sessionDoc.RootElement.GetProperty("websocketUrl").GetString()

                let wsClient = app.GetTestServer().CreateWebSocketClient()
                use! socket = wsClient.ConnectAsync(Uri($"ws://localhost{websocketUrl}"), CancellationToken.None)
                let seen = ResizeArray<string>()
                let mutable binaryMessages = 0
                let receivePayload () =
                    task {
                        let! messageType, payload = receiveMessage socket
                        match messageType with
                        | WebSocketMessageType.Text ->
                            let text = Encoding.UTF8.GetString(payload)
                            use doc = JsonDocument.Parse(text)
                            let eventType =
                                doc.RootElement.GetProperty("type").GetString()
                                |> Option.ofObj
                                |> Option.defaultValue ""
                            seen.Add(eventType)
                            return eventType, Some(doc.RootElement.Clone())
                        | WebSocketMessageType.Binary ->
                            binaryMessages <- binaryMessages + 1
                            return "binary", None
                        | _ -> return "close", None
                    }

                let! ready, _ = receivePayload ()
                assertEqual "agent socket ready" "session.ready" ready

                let sendAgentTurn samples expectedTurnIndex =
                    task {
                        let startSeenCount = seen.Count
                        do! sendText socket """{"type":"turn.start"}"""
                        let! accepted, _ = receivePayload ()
                        assertEqual $"agent turn {expectedTurnIndex} accepted" "turn.accepted" accepted
                        do! sendBinary socket (AudioChunk.float32ToLittleEndianBytes samples)
                        let! chunkAck, _ = receivePayload ()
                        assertEqual $"agent turn {expectedTurnIndex} chunk" "turn.chunk" chunkAck
                        do! sendText socket """{"type":"turn.end"}"""

                        let mutable donePayload: JsonElement option = None
                        let stopwatch = Stopwatch.StartNew()
                        while donePayload.IsNone && stopwatch.Elapsed < TimeSpan.FromSeconds(5.0) do
                            let! eventType, payload = receivePayload ()
                            if eventType = "agent.done" then
                                donePayload <- payload

                        match donePayload with
                        | None ->
                            let seenText = String.Join(",", seen)
                            return fail $"agent websocket turn {expectedTurnIndex}" $"agent.done was not observed. Events: {seenText}"
                        | Some payload ->
                            assertEqual $"agent done turn {expectedTurnIndex}" expectedTurnIndex (payload.GetProperty("turnIndex").GetInt32())
                            let turnEvents =
                                seen
                                |> Seq.skip startSeenCount
                                |> Seq.toArray
                            if not (turnEvents |> Array.contains "agent.transcription") then fail $"agent websocket turn {expectedTurnIndex}" "transcription event missing"
                            if not (turnEvents |> Array.contains "agent.tool_call") then fail $"agent websocket turn {expectedTurnIndex}" "tool call event missing"
                            if not (turnEvents |> Array.contains "agent.tool_result") then fail $"agent websocket turn {expectedTurnIndex}" "tool result event missing"
                            if not (turnEvents |> Array.contains "agent.final_text") then fail $"agent websocket turn {expectedTurnIndex}" "final text event missing"
                            return payload
                    }

                let! firstDone = sendAgentTurn [| 0.0f; 0.2f; 0.1f |] 1
                let! secondDone = sendAgentTurn [| 0.1f; 0.0f; 0.3f |] 2
                if binaryMessages < 2 then fail "agent websocket" "expected streamed Chroma audio for both turns"

                for turnIndex in [| 1; 2 |] do
                    let! detailsResponse = client.GetAsync($"/api/agent/sessions/{sessionId}/turns/{turnIndex}/details.json")
                    assertEqual $"agent details artifact turn {turnIndex}" HttpStatusCode.OK detailsResponse.StatusCode
                    let! detailsBody = detailsResponse.Content.ReadAsStringAsync()
                    if not (detailsBody.Contains("It is time to test the agent.", StringComparison.Ordinal)) then
                        fail $"agent details artifact turn {turnIndex}" $"unexpected details: {detailsBody}"
                    if not (detailsBody.Contains("get_current_time", StringComparison.Ordinal)) then
                        fail $"agent tool artifact turn {turnIndex}" $"tool call was not recorded: {detailsBody}"

                ignore firstDone
                ignore secondDone
            })

    let private realChromaAgentWebSocketSmokeIfRequested () =
        let enabled =
            match Environment.GetEnvironmentVariable("CHROMA_REAL_CHROMA_SMOKE") with
            | null | "" -> false
            | value ->
                value.Equals("1", StringComparison.OrdinalIgnoreCase)
                || value.Equals("true", StringComparison.OrdinalIgnoreCase)
                || value.Equals("yes", StringComparison.OrdinalIgnoreCase)

        if enabled then
            printfn "Running opt-in real Chroma two-turn agent smoke."

            let receiveMessage (socket: WebSocket) =
                task {
                    let buffer = Array.zeroCreate<byte> 4096
                    use stream = new MemoryStream()
                    let mutable messageType = WebSocketMessageType.Close
                    let mutable complete = false
                    while not complete do
                        let! result = socket.ReceiveAsync(ArraySegment<byte>(buffer), CancellationToken.None)
                        messageType <- result.MessageType
                        if result.Count > 0 then
                            stream.Write(buffer, 0, result.Count)
                        complete <- result.EndOfMessage || result.MessageType = WebSocketMessageType.Close
                    return messageType, stream.ToArray()
                }

            let sendText (socket: WebSocket) (text: string) =
                let bytes = Encoding.UTF8.GetBytes(text)
                socket.SendAsync(ArraySegment<byte>(bytes), WebSocketMessageType.Text, true, CancellationToken.None)

            let sendBinary (socket: WebSocket) bytes =
                socket.SendAsync(ArraySegment<byte>(bytes), WebSocketMessageType.Binary, true, CancellationToken.None)

            let runSmoke (app: WebApplication) (client: HttpClient) =
                task {
                    use content = multipartSessionContent "fsharp_onnx"
                    let! response = client.PostAsync("/api/agent/sessions", content)
                    assertEqual "real agent session create" HttpStatusCode.OK response.StatusCode
                    let! sessionJson = response.Content.ReadAsStringAsync()
                    use sessionDoc = JsonDocument.Parse(sessionJson)
                    let sessionId = sessionDoc.RootElement.GetProperty("id").GetString()
                    let websocketUrl = sessionDoc.RootElement.GetProperty("websocketUrl").GetString()

                    let wsClient = app.GetTestServer().CreateWebSocketClient()
                    use! socket = wsClient.ConnectAsync(Uri($"ws://localhost{websocketUrl}"), CancellationToken.None)
                    let seen = ResizeArray<string>()
                    let mutable binaryMessages = 0
                    let receivePayload () =
                        task {
                            let! messageType, payload = receiveMessage socket
                            match messageType with
                            | WebSocketMessageType.Text ->
                                let text = Encoding.UTF8.GetString(payload)
                                use doc = JsonDocument.Parse(text)
                                let eventType =
                                    doc.RootElement.GetProperty("type").GetString()
                                    |> Option.ofObj
                                    |> Option.defaultValue ""
                                seen.Add(eventType)
                                return eventType, Some(doc.RootElement.Clone())
                            | WebSocketMessageType.Binary ->
                                binaryMessages <- binaryMessages + 1
                                return "binary", None
                            | _ -> return "close", None
                        }

                    let! ready, _ = receivePayload ()
                    assertEqual "real agent socket ready" "session.ready" ready

                    let sendAgentTurn samples expectedTurnIndex =
                        task {
                            let startSeenCount = seen.Count
                            do! sendText socket """{"type":"turn.start"}"""
                            let! accepted, _ = receivePayload ()
                            assertEqual $"real agent turn {expectedTurnIndex} accepted" "turn.accepted" accepted
                            do! sendBinary socket (AudioChunk.float32ToLittleEndianBytes samples)
                            let! chunkAck, _ = receivePayload ()
                            assertEqual $"real agent turn {expectedTurnIndex} chunk" "turn.chunk" chunkAck
                            do! sendText socket """{"type":"turn.end"}"""

                            let mutable donePayload: JsonElement option = None
                            let stopwatch = Stopwatch.StartNew()
                            while donePayload.IsNone && stopwatch.Elapsed < TimeSpan.FromMinutes(20.0) do
                                let! eventType, payload = receivePayload ()
                                if eventType = "agent.done" then
                                    donePayload <- payload

                            match donePayload with
                            | None ->
                                let seenText = String.Join(",", seen)
                                return fail $"real agent websocket turn {expectedTurnIndex}" $"agent.done was not observed. Events: {seenText}"
                            | Some payload ->
                                assertEqual $"real agent done turn {expectedTurnIndex}" expectedTurnIndex (payload.GetProperty("turnIndex").GetInt32())
                                let turnEvents =
                                    seen
                                    |> Seq.skip startSeenCount
                                    |> Seq.toArray
                                if not (turnEvents |> Array.contains "agent.transcription") then fail $"real agent websocket turn {expectedTurnIndex}" "transcription event missing"
                                if not (turnEvents |> Array.contains "agent.tool_call") then fail $"real agent websocket turn {expectedTurnIndex}" "tool call event missing"
                                if not (turnEvents |> Array.contains "agent.tool_result") then fail $"real agent websocket turn {expectedTurnIndex}" "tool result event missing"
                                if not (turnEvents |> Array.contains "agent.final_text") then fail $"real agent websocket turn {expectedTurnIndex}" "final text event missing"
                                printfn "Real Chroma agent turn %d completed." expectedTurnIndex
                                return payload
                        }

                    let! firstDone = sendAgentTurn [| 0.0f; 0.2f; 0.1f |] 1
                    let! secondDone = sendAgentTurn [| 0.1f; 0.0f; 0.3f |] 2
                    if binaryMessages < 2 then fail "real agent websocket" "expected streamed Chroma audio for both turns"

                    for turnIndex in [| 1; 2 |] do
                        let! detailsResponse = client.GetAsync($"/api/agent/sessions/{sessionId}/turns/{turnIndex}/details.json")
                        assertEqual $"real agent details artifact turn {turnIndex}" HttpStatusCode.OK detailsResponse.StatusCode
                        let! detailsBody = detailsResponse.Content.ReadAsStringAsync()
                        if not (detailsBody.Contains("It is time to test the agent.", StringComparison.Ordinal)) then
                            fail $"real agent details artifact turn {turnIndex}" $"unexpected details: {detailsBody}"
                        if not (detailsBody.Contains("get_current_time", StringComparison.Ordinal)) then
                            fail $"real agent tool artifact turn {turnIndex}" $"tool call was not recorded: {detailsBody}"

                    ignore firstDone
                    ignore secondDone
                }

            let workDir = Path.Combine(Path.GetTempPath(), $"chroma-onnx-real-agent-smoke-{Guid.NewGuid():N}")
            Directory.CreateDirectory(workDir) |> ignore
            let builder = WebApplication.CreateBuilder([||])
            builder.Logging.ClearProviders() |> ignore
            builder.WebHost.UseTestServer() |> ignore
            let s2sOptions = S2sRuntimeOptions()
            s2sOptions.WorkDir <- workDir
            s2sOptions.MaxNewFrames <- 2
            s2sOptions.MaxPromptAudioSeconds <- 1.0
            s2sOptions.MaxTurnAudioSeconds <- 1.0
            s2sOptions.StreamDecodeFrames <- 1
            s2sOptions.StreamMinFreeVramMb <- 0
            let runtime = new ChromaS2sRuntime(s2sOptions) :> IS2sRuntime
            let gemma = FakeGemmaRuntime() :> IGemmaRuntime
            let gemmaOptions = GemmaRuntimeOptions()
            gemmaOptions.MaxAudioSeconds <- 1.0
            let agent =
                new GemmaChromaAgentRuntime(gemmaOptions, runtime, gemmaRuntime = gemma, workDir = workDir)
                :> IAgentRuntime
            let app = builder.Build()
            S2sWebApp.mapWithAgent app runtime agent |> ignore
            try
                waitTask (app.StartAsync())
                use client = app.GetTestClient()
                waitTask (runSmoke app client)
            finally
                waitTask (app.StopAsync())
                (agent :?> IDisposable).Dispose()
                (runtime :?> IDisposable).Dispose()
                (app :> IDisposable).Dispose()
                if Directory.Exists workDir then
                    Directory.Delete(workDir, true)

    let private realGemmaToolSmokeIfRequested () =
        let enabled =
            match Environment.GetEnvironmentVariable("CHROMA_REAL_GEMMA_SMOKE") with
            | null | "" -> false
            | value ->
                value.Equals("1", StringComparison.OrdinalIgnoreCase)
                || value.Equals("true", StringComparison.OrdinalIgnoreCase)
                || value.Equals("yes", StringComparison.OrdinalIgnoreCase)

        if enabled then
            let modelDir =
                match Environment.GetEnvironmentVariable("CHROMA_GEMMA_MODEL_DIR") with
                | null | "" -> "models/gemma-4-e2b-it-onnx-mobius/Q4_K_M/cuda"
                | value -> value
            let variant =
                match Environment.GetEnvironmentVariable("CHROMA_GEMMA_VARIANT") with
                | null | "" -> "Q4_K_M/cuda"
                | value -> value
            let provider =
                match Environment.GetEnvironmentVariable("CHROMA_GEMMA_PROVIDER") with
                | null | "" -> "cuda"
                | value -> value
            let runtimeKind =
                match Environment.GetEnvironmentVariable("CHROMA_GEMMA_RUNTIME") with
                | null | "" -> "ort-genai"
                | value -> value
            printfn "Running opt-in real Gemma tool-call smoke: %s (%s, %s, %s)." modelDir variant provider runtimeKind
            let gemma =
                match runtimeKind.Trim().ToLowerInvariant() with
                | "raw-ort" | "onnx" | "raw-onnx" -> new GemmaOnnxRunner(modelDir, variant, provider, 1.0) :> IGemmaRuntime
                | _ -> new GemmaOrtGenAiRunner(modelDir, variant, provider, 1.0) :> IGemmaRuntime
            use disposable = gemma :?> IDisposable
            let runtime = gemma
            let status = runtime.Status()
            if not status.Ready then
                let missing = String.Join(", ", status.MissingFiles)
                fail "real gemma status" $"Gemma model is not ready. Missing: {missing}"
            use cts = new CancellationTokenSource(TimeSpan.FromMinutes(10.0))
            let result =
                waitTaskResult (
                    runtime.GenerateAsync(
                        { Messages =
                            [| GemmaChatMessage.system "You must call get_current_time before answering."
                               GemmaChatMessage.user "What time is it? Call get_current_time now." |]
                          Tools =
                            [| { Name = "get_current_time"
                                 Description = "Return the current local and UTC time."
                                 Parameters = Array.empty } |]
                          Audio16k = None
                          AddGenerationPrompt = true
                          MaxNewTokens = 64
                          Temperature = 0.0
                          TopP = 1.0
                          TopK = 0 },
                        cts.Token
                    )
                )
            printfn "Real Gemma tool smoke output: %s" result.Text
            if String.IsNullOrWhiteSpace result.Text then
                fail "real gemma tool smoke" "Gemma returned empty text."
            let processor = GemmaProcessor(modelDir, 1.0)
            match processor.TryParseToolCall result.Text with
            | Some call ->
                assertEqual "real gemma tool call name" "get_current_time" call.Name
            | None ->
                fail "real gemma tool smoke" $"Gemma did not emit a parseable tool call. Output: {result.Text}"

    let private realGemmaAudioSmokeIfRequested () =
        let enabled =
            match Environment.GetEnvironmentVariable("CHROMA_REAL_GEMMA_AUDIO_SMOKE") with
            | null | "" -> false
            | value ->
                value.Equals("1", StringComparison.OrdinalIgnoreCase)
                || value.Equals("true", StringComparison.OrdinalIgnoreCase)
                || value.Equals("yes", StringComparison.OrdinalIgnoreCase)

        if enabled then
            let modelDir =
                match Environment.GetEnvironmentVariable("CHROMA_GEMMA_MODEL_DIR") with
                | null | "" -> "models/gemma-4-e2b-it-onnx-mobius/Q4_K_M/cuda"
                | value -> value
            let variant =
                match Environment.GetEnvironmentVariable("CHROMA_GEMMA_VARIANT") with
                | null | "" -> "Q4_K_M/cuda"
                | value -> value
            let provider =
                match Environment.GetEnvironmentVariable("CHROMA_GEMMA_PROVIDER") with
                | null | "" -> "cuda"
                | value -> value
            let runtimeKind =
                match Environment.GetEnvironmentVariable("CHROMA_GEMMA_RUNTIME") with
                | null | "" -> "ort-genai"
                | value -> value
            printfn "Running opt-in real Gemma audio smoke: %s (%s, %s, %s)." modelDir variant provider runtimeKind
            let gemma =
                match runtimeKind.Trim().ToLowerInvariant() with
                | "raw-ort" | "onnx" | "raw-onnx" -> new GemmaOnnxRunner(modelDir, variant, provider, 1.0) :> IGemmaRuntime
                | _ -> new GemmaOrtGenAiRunner(modelDir, variant, provider, 1.0) :> IGemmaRuntime
            use disposable = gemma :?> IDisposable
            let runtime = gemma
            let status = runtime.Status()
            if not status.Ready then
                let missing = String.Join(", ", status.MissingFiles)
                fail "real gemma audio status" $"Gemma model is not ready. Missing: {missing}"
            let audio =
                Array.init 16000 (fun index ->
                    let phase = 2.0 * Math.PI * 220.0 * float index / 16000.0
                    float32 (Math.Sin phase * 0.02))
            use cts = new CancellationTokenSource(TimeSpan.FromMinutes(10.0))
            let result =
                waitTaskResult (
                    runtime.GenerateAsync(
                        { Messages = [| GemmaChatMessage.user "Transcribe the following speech segment in its original language. Only output the transcription, with no newlines.\n\n<|audio|>" |]
                          Tools = Array.empty
                          Audio16k = Some audio
                          AddGenerationPrompt = true
                          MaxNewTokens = 8
                          Temperature = 0.0
                          TopP = 1.0
                          TopK = 0 },
                        cts.Token
                    )
                )
            printfn "Real Gemma audio smoke output: %s" result.Text
            if result.InputTokenCount <= 1 then
                fail "real gemma audio smoke" "Gemma audio prompt did not produce a usable input sequence."

    let private realGemmaChromaAgentSmokeIfRequested () =
        let enabled =
            match Environment.GetEnvironmentVariable("CHROMA_REAL_AGENT_SMOKE") with
            | null | "" -> false
            | value ->
                value.Equals("1", StringComparison.OrdinalIgnoreCase)
                || value.Equals("true", StringComparison.OrdinalIgnoreCase)
                || value.Equals("yes", StringComparison.OrdinalIgnoreCase)

        if enabled then
            printfn "Running opt-in real Gemma plus real Chroma two-turn agent smoke."
            let receiveMessage (socket: WebSocket) =
                task {
                    let buffer = Array.zeroCreate<byte> 4096
                    use stream = new MemoryStream()
                    let mutable messageType = WebSocketMessageType.Close
                    let mutable complete = false
                    while not complete do
                        let! result = socket.ReceiveAsync(ArraySegment<byte>(buffer), CancellationToken.None)
                        messageType <- result.MessageType
                        if result.Count > 0 then
                            stream.Write(buffer, 0, result.Count)
                        complete <- result.EndOfMessage || result.MessageType = WebSocketMessageType.Close
                    return messageType, stream.ToArray()
                }

            let sendText (socket: WebSocket) (text: string) =
                let bytes = Encoding.UTF8.GetBytes(text)
                socket.SendAsync(ArraySegment<byte>(bytes), WebSocketMessageType.Text, true, CancellationToken.None)

            let sendBinary (socket: WebSocket) bytes =
                socket.SendAsync(ArraySegment<byte>(bytes), WebSocketMessageType.Binary, true, CancellationToken.None)

            let createAgentContent () =
                let content = new MultipartFormDataContent()
                content.Add(
                    new StringContent(
                        "You are in an integration smoke test. If the current reasoning prompt has no tool result yet, output exactly call:get_current_time{}. After a tool result is present, answer in one short sentence."
                    ),
                    "systemPrompt"
                )
                content.Add(new StringContent("Reference text."), "promptText")
                content.Add(new StringContent("fsharp_onnx"), "backend")
                content.Add(new StringContent("2"), "maxNewFrames")
                let promptBytes = AudioChunk.float32ToLittleEndianBytes [| 0.0f; 0.1f; -0.1f |]
                content.Add(new ByteArrayContent(promptBytes), "promptPcm24k", "prompt.f32")
                content

            let runSmoke (app: WebApplication) (client: HttpClient) =
                task {
                    use content = createAgentContent ()
                    let! response = client.PostAsync("/api/agent/sessions", content)
                    assertEqual "real gemma agent session create" HttpStatusCode.OK response.StatusCode
                    let! sessionJson = response.Content.ReadAsStringAsync()
                    use sessionDoc = JsonDocument.Parse(sessionJson)
                    let sessionId = sessionDoc.RootElement.GetProperty("id").GetString()
                    let websocketUrl = sessionDoc.RootElement.GetProperty("websocketUrl").GetString()

                    let wsClient = app.GetTestServer().CreateWebSocketClient()
                    use! socket = wsClient.ConnectAsync(Uri($"ws://localhost{websocketUrl}"), CancellationToken.None)
                    let seen = ResizeArray<string>()
                    let mutable binaryMessages = 0
                    let receivePayload () =
                        task {
                            let! messageType, payload = receiveMessage socket
                            match messageType with
                            | WebSocketMessageType.Text ->
                                let text = Encoding.UTF8.GetString(payload)
                                use doc = JsonDocument.Parse(text)
                                let eventType =
                                    doc.RootElement.GetProperty("type").GetString()
                                    |> Option.ofObj
                                    |> Option.defaultValue ""
                                seen.Add(eventType)
                                return eventType, Some(doc.RootElement.Clone())
                            | WebSocketMessageType.Binary ->
                                binaryMessages <- binaryMessages + 1
                                return "binary", None
                            | _ -> return "close", None
                        }

                    let! ready, readyPayload = receivePayload ()
                    assertEqual "real gemma agent socket ready" "session.ready" ready
                    match readyPayload with
                    | Some payload when payload.TryGetProperty("gemmaReady") |> fst ->
                        if not (payload.GetProperty("gemmaReady").GetBoolean()) then
                            fail "real gemma agent readiness" (payload.GetProperty("gemmaMessage").GetString())
                    | _ -> ()

                    let sendAgentTurn samples expectedTurnIndex =
                        task {
                            let startSeenCount = seen.Count
                            do! sendText socket """{"type":"turn.start"}"""
                            let! accepted, _ = receivePayload ()
                            assertEqual $"real gemma agent turn {expectedTurnIndex} accepted" "turn.accepted" accepted
                            do! sendBinary socket (AudioChunk.float32ToLittleEndianBytes samples)
                            let! chunkAck, _ = receivePayload ()
                            assertEqual $"real gemma agent turn {expectedTurnIndex} chunk" "turn.chunk" chunkAck
                            do! sendText socket """{"type":"turn.end"}"""

                            let mutable donePayload: JsonElement option = None
                            let stopwatch = Stopwatch.StartNew()
                            while donePayload.IsNone && stopwatch.Elapsed < TimeSpan.FromMinutes(30.0) do
                                let! eventType, payload = receivePayload ()
                                if eventType = "agent.done" then
                                    donePayload <- payload

                            match donePayload with
                            | None ->
                                let seenText = String.Join(",", seen)
                                return fail $"real gemma agent websocket turn {expectedTurnIndex}" $"agent.done was not observed. Events: {seenText}"
                            | Some payload ->
                                assertEqual $"real gemma agent done turn {expectedTurnIndex}" expectedTurnIndex (payload.GetProperty("turnIndex").GetInt32())
                                let turnEvents =
                                    seen
                                    |> Seq.skip startSeenCount
                                    |> Seq.toArray
                                if not (turnEvents |> Array.contains "agent.transcription") then fail $"real gemma agent websocket turn {expectedTurnIndex}" "transcription event missing"
                                if not (turnEvents |> Array.contains "agent.tool_call") then fail $"real gemma agent websocket turn {expectedTurnIndex}" "tool call event missing"
                                if not (turnEvents |> Array.contains "agent.tool_result") then fail $"real gemma agent websocket turn {expectedTurnIndex}" "tool result event missing"
                                if not (turnEvents |> Array.contains "agent.final_text") then fail $"real gemma agent websocket turn {expectedTurnIndex}" "final text event missing"
                                printfn "Real Gemma+Chroma agent turn %d completed." expectedTurnIndex
                                return payload
                        }

                    let firstAudio = Array.init 16000 (fun index -> if index % 97 = 0 then 0.02f else 0.0f)
                    let secondAudio = Array.init 16000 (fun index -> if index % 113 = 0 then -0.02f else 0.0f)
                    let! firstDone = sendAgentTurn firstAudio 1
                    let! secondDone = sendAgentTurn secondAudio 2
                    if binaryMessages < 2 then fail "real gemma agent websocket" "expected streamed Chroma audio for both turns"

                    for turnIndex in [| 1; 2 |] do
                        let! detailsResponse = client.GetAsync($"/api/agent/sessions/{sessionId}/turns/{turnIndex}/details.json")
                        assertEqual $"real gemma agent details artifact turn {turnIndex}" HttpStatusCode.OK detailsResponse.StatusCode
                        let! detailsBody = detailsResponse.Content.ReadAsStringAsync()
                        if not (detailsBody.Contains("get_current_time", StringComparison.Ordinal)) then
                            fail $"real gemma agent tool artifact turn {turnIndex}" $"tool call was not recorded: {detailsBody}"

                    ignore firstDone
                    ignore secondDone
                }

            let workDir = Path.Combine(Path.GetTempPath(), $"chroma-onnx-real-gemma-agent-smoke-{Guid.NewGuid():N}")
            Directory.CreateDirectory(workDir) |> ignore
            let builder = WebApplication.CreateBuilder([||])
            builder.Logging.ClearProviders() |> ignore
            builder.WebHost.UseTestServer() |> ignore
            let s2sOptions = S2sRuntimeOptions()
            s2sOptions.WorkDir <- workDir
            s2sOptions.MaxNewFrames <- 2
            s2sOptions.MaxPromptAudioSeconds <- 1.0
            s2sOptions.MaxTurnAudioSeconds <- 1.0
            s2sOptions.StreamDecodeFrames <- 1
            s2sOptions.StreamMinFreeVramMb <- 0
            let runtime = new ChromaS2sRuntime(s2sOptions) :> IS2sRuntime
            let gemmaOptions = GemmaRuntimeOptions()
            gemmaOptions.ModelDir <-
                match Environment.GetEnvironmentVariable("CHROMA_GEMMA_MODEL_DIR") with
                | null | "" -> "models/gemma-4-e2b-it-onnx-mobius/Q4_K_M/cuda"
                | value -> value
            gemmaOptions.Variant <-
                match Environment.GetEnvironmentVariable("CHROMA_GEMMA_VARIANT") with
                | null | "" -> "Q4_K_M/cuda"
                | value -> value
            gemmaOptions.ExecutionProvider <-
                match Environment.GetEnvironmentVariable("CHROMA_GEMMA_PROVIDER") with
                | null | "" -> "cuda"
                | value -> value
            gemmaOptions.Runtime <-
                match Environment.GetEnvironmentVariable("CHROMA_GEMMA_RUNTIME") with
                | null | "" -> "ort-genai"
                | value -> value
            gemmaOptions.MaxAudioSeconds <- 1.0
            gemmaOptions.AsrMaxNewTokens <- 32
            gemmaOptions.ReasoningMaxNewTokens <- 64
            gemmaOptions.ToolMaxRounds <- 3
            gemmaOptions.MaxHistoryTurns <- 2
            let agent = new GemmaChromaAgentRuntime(gemmaOptions, runtime, workDir = workDir) :> IAgentRuntime
            let app = builder.Build()
            S2sWebApp.mapWithAgent app runtime agent |> ignore
            try
                waitTask (app.StartAsync())
                use client = app.GetTestClient()
                waitTask (runSmoke app client)
            finally
                waitTask (app.StopAsync())
                (agent :?> IDisposable).Dispose()
                (runtime :?> IDisposable).Dispose()
                (app :> IDisposable).Dispose()
                if Directory.Exists workDir then
                    Directory.Delete(workDir, true)

    let private configDefaultsAndBinding () =
        let defaults = S2sRuntimeOptions()
        assertEqual "default model dir" "models/chroma-4b" defaults.ModelDir
        assertEqual "default bundle dir" "onnx_deploy/chroma-s2s-full-v2" defaults.BundleDir
        assertEqual "default optimized cache dir" "onnx/chroma-s2s-full-v2/ort-cache-ort-local-external" defaults.OptimizedModelCacheDir
        assertEqual "default cuda memory cap enabled" true defaults.CudaGpuMemLimitMb.HasValue
        assertEqual "default cuda memory cap" 15360 defaults.CudaGpuMemLimitMb.Value
        assertEqual "default stream frames" 4 defaults.StreamDecodeFrames
        assertEqual "default generation mode" "sample" defaults.GenerationMode
        assertEqual "default sampling algorithm" "top-k-top-p" defaults.SamplingAlgorithm
        assertEqual "default sampling temperature" 0.7 defaults.SamplingTemperature
        assertEqual "default sampling top p" 0.9 defaults.SamplingTopP
        assertEqual "default sampling top k" 50 defaults.SamplingTopK
        assertEqual "default max queue" 32 defaults.MaxQueueLength
        let gemmaDefaults = GemmaRuntimeOptions()
        assertEqual "default gemma model dir" "models/gemma-4-e2b-it-onnx-mobius/Q4_K_M/cuda" gemmaDefaults.ModelDir
        assertEqual "default gemma variant" "Q4_K_M/cuda" gemmaDefaults.Variant
        assertEqual "default gemma runtime" "ort-genai" gemmaDefaults.Runtime
        assertEqual "default gemma provider" "cuda" gemmaDefaults.ExecutionProvider
        assertEqual "default gemma max audio" 30.0 gemmaDefaults.MaxAudioSeconds
        assertEqual "default gemma asr tokens" 128 gemmaDefaults.AsrMaxNewTokens
        assertEqual "default gemma reasoning tokens" 512 gemmaDefaults.ReasoningMaxNewTokens
        assertEqual "default gemma tool rounds" 3 gemmaDefaults.ToolMaxRounds
        assertEqual "default gemma history turns" 8 gemmaDefaults.MaxHistoryTurns

        let values = Dictionary<string, string>()
        values["ChromaOnnx:S2s:ModelDir"] <- "models/custom"
        values["ChromaOnnx:S2s:ExecutionProvider"] <- "cpu"
        values["ChromaOnnx:S2s:MaxQueueLength"] <- "7"
        values["ChromaOnnx:S2s:GenerationMode"] <- "greedy"
        values["ChromaOnnx:S2s:SamplingAlgorithm"] <- "chroma"
        values["ChromaOnnx:S2s:SamplingTemperature"] <- "0.7"
        values["ChromaOnnx:S2s:SamplingTopP"] <- "0.9"
        values["ChromaOnnx:S2s:SamplingTopK"] <- "25"
        values["ChromaOnnx:Gemma:ModelDir"] <- "models/gemma-custom"
        values["ChromaOnnx:Gemma:Variant"] <- "fp16"
        values["ChromaOnnx:Gemma:Runtime"] <- "raw-ort"
        values["ChromaOnnx:Gemma:ExecutionProvider"] <- "cpu"
        values["ChromaOnnx:Gemma:ToolMaxRounds"] <- "5"
        values["ChromaOnnx:Gemma:MaxHistoryTurns"] <- "11"
        let configuration =
            ConfigurationBuilder()
                .AddInMemoryCollection(values)
                .Build()
        let bound = S2sWebApp.bindOptions configuration
        assertEqual "bound model dir" "models/custom" bound.ModelDir
        assertEqual "bound execution provider" "cpu" bound.ExecutionProvider
        assertEqual "bound max queue" 7 bound.MaxQueueLength
        assertEqual "bound generation mode" "greedy" bound.GenerationMode
        assertEqual "bound sampling algorithm" "chroma" bound.SamplingAlgorithm
        assertEqual "bound sampling temperature" 0.7 bound.SamplingTemperature
        assertEqual "bound sampling top p" 0.9 bound.SamplingTopP
        assertEqual "bound sampling top k" 25 bound.SamplingTopK
        let boundGemma = S2sWebApp.bindGemmaOptions configuration
        assertEqual "bound gemma model dir" "models/gemma-custom" boundGemma.ModelDir
        assertEqual "bound gemma variant" "fp16" boundGemma.Variant
        assertEqual "bound gemma runtime" "raw-ort" boundGemma.Runtime
        assertEqual "bound gemma provider" "cpu" boundGemma.ExecutionProvider
        assertEqual "bound gemma tool rounds" 5 boundGemma.ToolMaxRounds
        assertEqual "bound gemma history turns" 11 boundGemma.MaxHistoryTurns

    let private runtimePathResolution () =
        let normalize path = Path.TrimEndingDirectorySeparator(Path.GetFullPath(path))
        let root = Path.Combine(Path.GetTempPath(), $"chroma-onnx-paths-{Guid.NewGuid():N}")
        let outputDir = Path.Combine(root, "src", "ChromaOnnx.Service", "bin", "Debug", "net10.0")
        try
            Directory.CreateDirectory(Path.Combine(root, "models", "chroma-4b")) |> ignore
            Directory.CreateDirectory(Path.Combine(root, "onnx_deploy", "chroma-s2s-full-v2")) |> ignore
            Directory.CreateDirectory(outputDir) |> ignore
            File.WriteAllText(Path.Combine(root, "Chroma_ONNX.slnx"), "<Solution />")

            let defaults = S2sRuntimeOptions()
            let baseDir =
                S2sRuntimePaths.resolveBaseFromCandidates
                    [| outputDir |]
                    [| defaults.ModelDir; defaults.BundleDir |]

            assertEqual "path base from output dir" (normalize root) baseDir
            assertEqual
                "relative model path from output dir"
                (Path.GetFullPath(Path.Combine(root, defaults.ModelDir)))
                (S2sRuntimePaths.resolveAgainst baseDir defaults.ModelDir)

            let absoluteWorkDir = Path.Combine(root, "served-runs-absolute")
            assertEqual
                "absolute path unchanged"
                (Path.GetFullPath absoluteWorkDir)
                (S2sRuntimePaths.resolveAgainst baseDir absoluteWorkDir)
        finally
            if Directory.Exists root then
                Directory.Delete(root, true)

    let private localExternalDataLinks () =
        let root = Path.Combine(Path.GetTempPath(), $"chroma-onnx-links-{Guid.NewGuid():N}")
        try
            let modelDir = Path.Combine(root, "models", "chroma-4b")
            let bundleDir = Path.Combine(root, "onnx_deploy", "chroma-s2s-full-v2")
            Directory.CreateDirectory(modelDir) |> ignore
            Directory.CreateDirectory(bundleDir) |> ignore

            let shardName = "model-00001-of-00003.safetensors"
            let sourcePath = Path.Combine(modelDir, shardName)
            let targetPath = Path.Combine(bundleDir, shardName)
            let graphPath = Path.Combine(bundleDir, "chroma_s2s_merged.weights_free.onnx")
            File.WriteAllBytes(sourcePath, [| 1uy; 2uy; 3uy; 4uy |])
            File.WriteAllText(graphPath, "placeholder")

            let entries =
                [| { Graph = "s2s_merged"
                     OnnxInitializer = "initializer"
                     SourceShard = shardName
                     SourceTensor = "tensor"
                     Dtype = "F32"
                     Shape = [| 1L |]
                     ByteLength = 4L
                     Transform = None } |]

            SharedWeights.ensureLocalExternalDataLinks modelDir graphPath entries

            assertEqual "external-data link exists" true (File.Exists targetPath)
            assertEqual "external-data link size" (FileInfo(sourcePath).Length) (FileInfo(targetPath).Length)
        finally
            if Directory.Exists root then
                Directory.Delete(root, true)

    let private sharedManifestCapabilities () =
        let root = Path.Combine(Path.GetTempPath(), $"chroma-onnx-manifest-{Guid.NewGuid():N}")
        let writeManifest bundleDir capabilitiesJson =
            Directory.CreateDirectory(bundleDir) |> ignore
            let capabilitiesBlock =
                if String.IsNullOrWhiteSpace capabilitiesJson then
                    ""
                else
                    $""","capabilities":{capabilitiesJson}"""
            let json =
                $$"""{
  "hidden_size": 2048,
  "audio_num_codebooks": 8,
  "graphs": {
    "s2s_merged": {
      "path": "graph.onnx",
      "inputs": ["input_ids"],
      "outputs": ["logits"]
    }
  }{{capabilitiesBlock}},
  "initializers": []
}
"""
            File.WriteAllText(Path.Combine(bundleDir, "shared_weights_manifest.json"), json)

        try
            let oldBundle = Path.Combine(root, "old")
            writeManifest oldBundle ""
            let oldManifest = SharedWeights.loadManifest oldBundle
            assertEqual "old manifest graph mode" None oldManifest.Capabilities.GraphMode
            assertEqual "old manifest thinker mode" None oldManifest.Capabilities.ThinkerFeatureMode
            assertEqual "old manifest max audio items" None oldManifest.Capabilities.ThinkerMaxAudioItems

            let s2sBundle = Path.Combine(root, "s2s")
            writeManifest
                s2sBundle
                """{"s2s_graph_mode":"one-shot","thinker_feature_mode":"dynamic_batch1_multi_audio_full_length","thinker_max_audio_items":1}"""
            let s2sManifest = SharedWeights.loadManifest s2sBundle
            assertEqual "s2s manifest graph mode" (Some "one-shot") s2sManifest.Capabilities.GraphMode
            assertEqual
                "s2s manifest thinker mode"
                (Some "dynamic_batch1_multi_audio_full_length")
                s2sManifest.Capabilities.ThinkerFeatureMode
            assertEqual "s2s manifest max audio items" (Some 1) s2sManifest.Capabilities.ThinkerMaxAudioItems
        finally
            if Directory.Exists root then
                Directory.Delete(root, true)

    [<EntryPoint>]
    let main _ =
        fifoOrdering ()
        queueFull ()
        queuePositionUpdates ()
        queuedCancellation ()
        float32ChunkRoundtrip ()
        samplingLogits ()
        configDefaultsAndBinding ()
        runtimePathResolution ()
        localExternalDataLinks ()
        sharedManifestCapabilities ()
        gemmaRenderingAndParsing ()
        serviceRejectsPythonBackend ()
        serviceWebSocketAndArtifacts ()
        agentWebSocketAndDetails ()
        realChromaAgentWebSocketSmokeIfRequested ()
        realGemmaToolSmokeIfRequested ()
        realGemmaAudioSmokeIfRequested ()
        realGemmaChromaAgentSmokeIfRequested ()
        printfn "All Chroma ONNX tests passed."
        0
