namespace ChromaOnnx.Tests

open System
open System.Collections.Concurrent
open System.Diagnostics
open System.Threading
open System.Threading.Tasks
open ChromaOnnx

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

    [<EntryPoint>]
    let main _ =
        fifoOrdering ()
        queueFull ()
        queuePositionUpdates ()
        queuedCancellation ()
        float32ChunkRoundtrip ()
        printfn "All streaming primitive tests passed."
        0
