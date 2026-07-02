namespace ChromaOnnx

open System
open System.Collections.Generic
open System.Threading
open System.Threading.Tasks

type WorkQueuePosition =
    { Id: string
      Position: int
      QueueLength: int
      RunningId: string option
      IsRunning: bool }

type QueuedWorkHandle internal (id: string, completion: Task, cancel: unit -> unit) =
    member _.Id = id
    member _.Completion = completion
    member _.Cancel() = cancel ()

type WorkQueueEnqueueResult =
    | Enqueued of QueuedWorkHandle * WorkQueuePosition
    | QueueFull of maxQueueLength: int

type private QueuedWorkItem =
    { Id: string
      Work: CancellationToken -> Task
      Notify: WorkQueuePosition -> unit
      Completion: TaskCompletionSource<unit>
      Cancellation: CancellationTokenSource
      mutable Node: LinkedListNode<QueuedWorkItem> option
      mutable Registration: CancellationTokenRegistration option }

type StreamingWorkQueue(maxQueueLength: int) =
    let maxQueueLength = max 0 maxQueueLength
    let gate = obj ()
    let pending = LinkedList<QueuedWorkItem>()
    let mutable running: QueuedWorkItem option = None

    let safeNotify (item: QueuedWorkItem) (snapshot: WorkQueuePosition) =
        try
            item.Notify snapshot
        with _ ->
            ()

    let snapshotsUnsafe () =
        let queueLength = pending.Count
        let runningId = running |> Option.map _.Id
        let snapshots = ResizeArray<QueuedWorkItem * WorkQueuePosition>()
        let mutable position = 1
        for item in pending do
            snapshots.Add(
                item,
                { Id = item.Id
                  Position = position
                  QueueLength = queueLength
                  RunningId = runningId
                  IsRunning = false }
            )
            position <- position + 1

        match running with
        | Some item ->
            snapshots.Add(
                item,
                { Id = item.Id
                  Position = 0
                  QueueLength = queueLength
                  RunningId = runningId
                  IsRunning = true }
            )
        | None -> ()

        snapshots.ToArray()

    let notifySnapshots snapshots =
        for item, snapshot in snapshots do
            safeNotify item snapshot

    let notifyAll () =
        let snapshots = lock gate snapshotsUnsafe
        notifySnapshots snapshots

    let removePendingUnsafe (item: QueuedWorkItem) =
        match item.Node with
        | Some node ->
            pending.Remove node
            item.Node <- None
            true
        | None -> false

    let rec tryStartNext () =
        let next =
            lock gate (fun () ->
                if running.IsNone && pending.Count > 0 then
                    match Option.ofObj pending.First with
                    | Some node ->
                        let item = node.Value
                        pending.Remove node
                        item.Node <- None
                        running <- Some item
                        Some item
                    | None -> None
                else
                    None)

        match next with
        | None -> ()
        | Some item ->
            notifyAll ()
            Task.Run(Func<Task>(fun () ->
                task {
                    try
                        try
                            item.Cancellation.Token.ThrowIfCancellationRequested()
                            do! item.Work item.Cancellation.Token
                            item.Completion.TrySetResult(()) |> ignore
                        with
                        | :? OperationCanceledException as ex ->
                            item.Completion.TrySetCanceled(ex.CancellationToken) |> ignore
                        | ex ->
                            item.Completion.TrySetException(ex) |> ignore
                    finally
                        item.Registration |> Option.iter (fun registration -> registration.Dispose())
                        item.Cancellation.Dispose()
                        lock gate (fun () ->
                            match running with
                            | Some current when obj.ReferenceEquals(current, item) -> running <- None
                            | _ -> ())
                        notifyAll ()
                        tryStartNext ()
                }))
            |> ignore

    let cancelItem item =
        item.Cancellation.Cancel()
        let removed =
            lock gate (fun () ->
                if removePendingUnsafe item then
                    true
                else
                    false)
        if removed then
            item.Registration |> Option.iter (fun registration -> registration.Dispose())
            item.Completion.TrySetCanceled(item.Cancellation.Token) |> ignore
            item.Cancellation.Dispose()
            notifyAll ()
            tryStartNext ()

    member _.MaxQueueLength = maxQueueLength

    member _.QueueLength =
        lock gate (fun () -> pending.Count)

    member _.RunningId =
        lock gate (fun () -> running |> Option.map _.Id)

    member _.TryEnqueue(id: string, work: CancellationToken -> Task, notify: WorkQueuePosition -> unit, cancellationToken: CancellationToken) =
        if String.IsNullOrWhiteSpace id then
            invalidArg (nameof id) "Queued work id is required."

        let completion =
            TaskCompletionSource<unit>(TaskCreationOptions.RunContinuationsAsynchronously)

        let linkedCancellation =
            CancellationTokenSource.CreateLinkedTokenSource(cancellationToken)

        let item =
            { Id = id
              Work = work
              Notify = notify
              Completion = completion
              Cancellation = linkedCancellation
              Node = None
              Registration = None }

        let enqueueResult =
            lock gate (fun () ->
                if pending.Count >= maxQueueLength then
                    QueueFull maxQueueLength
                else
                    let node = pending.AddLast item
                    item.Node <- Some node
                    let position = pending.Count
                    let snapshot =
                        { Id = id
                          Position = position
                          QueueLength = pending.Count
                          RunningId = running |> Option.map _.Id
                          IsRunning = false }
                    let handle =
                        QueuedWorkHandle(
                            id,
                            completion.Task,
                            fun () -> cancelItem item
                        )
                    Enqueued(handle, snapshot))

        match enqueueResult with
        | QueueFull _ ->
            linkedCancellation.Dispose()
            enqueueResult
        | Enqueued _ ->
            item.Registration <- Some(linkedCancellation.Token.Register(fun _ -> cancelItem item))
            notifyAll ()
            tryStartNext ()
            enqueueResult

module AudioChunk =
    let float32ToLittleEndianBytes (samples: float32 array) =
        let bytes = Array.zeroCreate<byte> (samples.Length * sizeof<float32>)
        Buffer.BlockCopy(samples, 0, bytes, 0, bytes.Length)
        bytes

    let float32FromLittleEndianBytes (bytes: byte array) =
        if bytes.Length % sizeof<float32> <> 0 then
            invalidArg (nameof bytes) $"Float32 audio payload length must be divisible by 4, got {bytes.Length} bytes."

        let samples = Array.zeroCreate<float32> (bytes.Length / sizeof<float32>)
        Buffer.BlockCopy(bytes, 0, samples, 0, bytes.Length)
        samples
