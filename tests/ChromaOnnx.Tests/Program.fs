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
                  SamplingTemperature = 1.0
                  SamplingTopP = 1.0
                  SamplingTopK = 0
                  MaxPromptAudioSeconds = 60.0
                  MaxTurnAudioSeconds = 60.0
                  MaxHistoryTurns = 2
                  MaxHistoryAudioSeconds = 180.0
                  IncludeAssistantAudioInHistory = true
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
                        let context =
                            { TurnIndex = turnIndex
                              HistoryTurnsUsed = max 0 (turnIndex - 1)
                              HistoryAudioSeconds = float request.UserAudio16k.Length / 16000.0
                              HistoryDropped = 0 }
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
                              HistoryTurnsUsed = context.HistoryTurnsUsed
                              HistoryAudioSeconds = context.HistoryAudioSeconds
                              HistoryDropped = context.HistoryDropped
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

    let private multipartSessionContent backend =
        let content = new MultipartFormDataContent()
        content.Add(new StringContent("You are a helpful assistant."), "systemPrompt")
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
                let! secondTurn = sendTurn [| 0.1f; 0.3f |]
                assertEqual "first turn index" 1 firstTurn
                assertEqual "second turn index" 2 secondTurn
                if binaryMessages < 2 then
                    fail "websocket generation" "expected streamed audio binary payloads for both turns"

                let! detailsResponse = client.GetAsync($"/api/s2s/sessions/{sessionId}/details.json")
                assertEqual "details artifact" HttpStatusCode.OK detailsResponse.StatusCode
                let! audioResponse = client.GetAsync($"/api/s2s/sessions/{sessionId}/fsharp_onnx/audio.wav")
                assertEqual "audio artifact" HttpStatusCode.OK audioResponse.StatusCode
                let! turnDetailsResponse = client.GetAsync($"/api/s2s/sessions/{sessionId}/turns/2/fsharp_onnx/details.json")
                assertEqual "turn details artifact" HttpStatusCode.OK turnDetailsResponse.StatusCode
                let! turnAudioResponse = client.GetAsync($"/api/s2s/sessions/{sessionId}/turns/2/fsharp_onnx/audio.wav")
                assertEqual "turn audio artifact" HttpStatusCode.OK turnAudioResponse.StatusCode
            })

    let private configDefaultsAndBinding () =
        let defaults = S2sRuntimeOptions()
        assertEqual "default model dir" "models/chroma-4b" defaults.ModelDir
        assertEqual "default bundle dir" "onnx_deploy/chroma-s2s-full-v2" defaults.BundleDir
        assertEqual "default optimized cache dir" "onnx/chroma-s2s-full-v2/ort-cache-ort-local-external" defaults.OptimizedModelCacheDir
        assertEqual "default cuda memory cap enabled" true defaults.CudaGpuMemLimitMb.HasValue
        assertEqual "default cuda memory cap" 15360 defaults.CudaGpuMemLimitMb.Value
        assertEqual "default stream frames" 4 defaults.StreamDecodeFrames
        assertEqual "default generation mode" "sample" defaults.GenerationMode
        assertEqual "default sampling temperature" 0.8 defaults.SamplingTemperature
        assertEqual "default sampling top p" 0.95 defaults.SamplingTopP
        assertEqual "default sampling top k" 50 defaults.SamplingTopK
        assertEqual "default max queue" 32 defaults.MaxQueueLength
        assertEqual "default max history turns" 2 defaults.MaxHistoryTurns
        assertEqual "default max history audio seconds" 180.0 defaults.MaxHistoryAudioSeconds
        assertEqual "default include assistant history" true defaults.IncludeAssistantAudioInHistory

        let values = Dictionary<string, string>()
        values["ChromaOnnx:S2s:ModelDir"] <- "models/custom"
        values["ChromaOnnx:S2s:ExecutionProvider"] <- "cpu"
        values["ChromaOnnx:S2s:MaxQueueLength"] <- "7"
        values["ChromaOnnx:S2s:GenerationMode"] <- "greedy"
        values["ChromaOnnx:S2s:SamplingTemperature"] <- "0.7"
        values["ChromaOnnx:S2s:SamplingTopP"] <- "0.9"
        values["ChromaOnnx:S2s:SamplingTopK"] <- "25"
        values["ChromaOnnx:S2s:MaxHistoryTurns"] <- "3"
        values["ChromaOnnx:S2s:MaxHistoryAudioSeconds"] <- "120"
        values["ChromaOnnx:S2s:IncludeAssistantAudioInHistory"] <- "false"
        let configuration =
            ConfigurationBuilder()
                .AddInMemoryCollection(values)
                .Build()
        let bound = S2sWebApp.bindOptions configuration
        assertEqual "bound model dir" "models/custom" bound.ModelDir
        assertEqual "bound execution provider" "cpu" bound.ExecutionProvider
        assertEqual "bound max queue" 7 bound.MaxQueueLength
        assertEqual "bound generation mode" "greedy" bound.GenerationMode
        assertEqual "bound sampling temperature" 0.7 bound.SamplingTemperature
        assertEqual "bound sampling top p" 0.9 bound.SamplingTopP
        assertEqual "bound sampling top k" 25 bound.SamplingTopK
        assertEqual "bound max history turns" 3 bound.MaxHistoryTurns
        assertEqual "bound max history audio seconds" 120.0 bound.MaxHistoryAudioSeconds
        assertEqual "bound include assistant history" false bound.IncludeAssistantAudioInHistory

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

    [<EntryPoint>]
    let main _ =
        fifoOrdering ()
        queueFull ()
        queuePositionUpdates ()
        queuedCancellation ()
        float32ChunkRoundtrip ()
        configDefaultsAndBinding ()
        runtimePathResolution ()
        localExternalDataLinks ()
        serviceRejectsPythonBackend ()
        serviceWebSocketAndArtifacts ()
        printfn "All Chroma ONNX tests passed."
        0
