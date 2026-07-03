namespace ChromaOnnx

open System
open System.Collections.Concurrent
open System.Collections.Generic
open System.Diagnostics
open System.IO
open System.Runtime.InteropServices
open System.Text.Json
open System.Threading
open Microsoft.ML.OnnxRuntime
open Microsoft.ML.OnnxRuntime.Tensors

type private OwnedDecoderOutput(logits: DenseTensor<float32>, cache: Dictionary<string, DenseTensor<float32>>, owners: IDisposable array) =
    let mutable disposed = false

    member _.Logits = logits
    member _.Cache = cache

    interface IDisposable with
        member _.Dispose() =
            if not disposed then
                disposed <- true
                for owner in owners do
                    owner.Dispose()

type ChromaS2sOnnxRunner(modelDir: string, bundleDir: string, executionProvider: string, memoryMode: string, tuningOptions: S2sOrtTuningOptions) =
    let manifestPath = Path.Combine(bundleDir, "shared_weights_manifest.json")
    let mergedGraphName = "s2s_merged"
    let requiredGraphs =
        [| "generate_prefill"
           "backbone_frame_step"
           "backbone_thinker_step"
           "decoder"
           "decoder_prefill"
           "decoder_step"
           "codec_decode" |]

    let log message =
        printfn "[%s] ChromaS2SONNX: %s" (DateTimeOffset.Now.ToString("HH:mm:ss")) message
        Console.Out.Flush()

    let elapsedSeconds (stopwatch: Stopwatch) =
        $"{stopwatch.Elapsed.TotalSeconds:N1}s"

    let manifest =
        if File.Exists manifestPath then
            log $"Loading S2S bundle manifest: {manifestPath}"
            Some(SharedWeights.loadManifest bundleDir)
        else
            None

    let normalizedMemoryMode =
        match memoryMode.Trim().ToLowerInvariant() with
        | "" | "python-footprint" | "python_footprint" | "paged" | "memory" -> "python-footprint"
        | "balanced" -> "balanced"
        | "warm" -> "warm"
        | "resident-merged" | "resident_merged" | "merged" | "python-like" | "python_like" -> "resident-merged"
        | value -> invalidArg (nameof memoryMode) $"Unsupported memory mode '{value}'. Use python-footprint, balanced, warm, or resident-merged."

    let useMergedSession = normalizedMemoryMode = "resident-merged"

    let normalizedOrtMemoryProfile =
        match tuningOptions.MemoryProfile.Trim().ToLowerInvariant() with
        | "" | "quality-safe" | "quality_safe" | "safe" | "reduced-memory" | "reduced_memory" -> "quality-safe"
        | "ort-default" | "ort_default" | "default" -> "ort-default"
        | value -> invalidArg (nameof tuningOptions.MemoryProfile) $"Unsupported ORT memory profile '{value}'. Use quality-safe or ort-default."

    let useQualitySafeOrtMemoryProfile = normalizedOrtMemoryProfile = "quality-safe"

    let optimizedModelCacheDir =
        tuningOptions.OptimizedModelCacheDir
        |> Option.map (fun path -> Path.GetFullPath(path))

    let normalizedOptimizedModelCacheFormat =
        match tuningOptions.OptimizedModelCacheFormat.Trim().ToLowerInvariant() with
        | "" | "onnx" -> "onnx"
        | "onnx-external" | "onnx_external" | "external-onnx" | "external_onnx" -> "onnx-external"
        | "ort" | "ort-mmap" | "ort_mmap" -> "ort"
        | value -> invalidArg (nameof tuningOptions.OptimizedModelCacheFormat) $"Unsupported optimized model cache format '{value}'. Use onnx, onnx-external, or ort."

    let graphOptimizationLevel graphName =
        if useMergedSession && graphName = mergedGraphName then
            GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        else
            GraphOptimizationLevel.ORT_DISABLE_ALL

    let cacheKeyForGraph graphName =
        let provider = executionProvider.Trim().ToLowerInvariant()
        let extension =
            if normalizedOptimizedModelCacheFormat = "ort" then
                "optimized.ort"
            else
                "optimized.onnx"
        $"{graphName}.{provider}.{normalizedOrtMemoryProfile}.{extension}"

    let optimizedModelCachePath graphName =
        optimizedModelCacheDir
        |> Option.map (fun dir -> Path.Combine(dir, cacheKeyForGraph graphName))

    let existingOptimizedModelCachePath graphName =
        optimizedModelCachePath graphName
        |> Option.bind (fun path ->
            if File.Exists path then
                let info = FileInfo(path)
                if info.Length > 0L then
                    Some path
                else
                    File.Delete(path)
                    None
            else
                None)

    let requireConfiguredOptimizedCache graphName =
        match optimizedModelCachePath graphName with
        | Some path when (existingOptimizedModelCachePath graphName).IsNone ->
            let cacheDir = optimizedModelCacheDir |> Option.defaultValue "<unset>"
            let provider = executionProvider.Trim().ToLowerInvariant()
            invalidOp (
                $"Optimized model cache is configured, but the expected cache file is missing or empty for graph '{graphName}': {path}. "
                + $"Build the machine-local cache before starting the service: .venv\\Scripts\\python.exe scripts\\rebuild_chroma_local_external_cache.py --model-dir \"{modelDir}\" --bundle-dir \"{bundleDir}\" --cache-dir \"{cacheDir}\" --provider {provider} --memory-profile {normalizedOrtMemoryProfile}. "
                + "Keep cache files under ignored onnx/ so large safetensor links are not created under committed onnx_deploy/."
            )
        | _ -> ()

    let useExistingOrtCacheWithoutRegistry =
        useMergedSession
        && normalizedOptimizedModelCacheFormat = "ort"
        && (existingOptimizedModelCachePath mergedGraphName).IsSome

    let store =
        if useExistingOrtCacheWithoutRegistry then
            None
        else
            manifest
            |> Option.map (fun value ->
                log $"Mapping safetensor shards from {modelDir}"
                let stopwatch = Stopwatch.StartNew()
                let weightStore = new SafetensorWeightStore(modelDir, value.Initializers)
                stopwatch.Stop()
                log $"Mapped {weightStore.MappedShardCount} safetensor shard(s) in {elapsedSeconds stopwatch}"
                weightStore)

    let registry =
        match manifest, store with
        | Some value, Some weightStore ->
            log $"Preparing shared ORT initializer views for {value.Initializers.Length} manifest entries"
            let stopwatch = Stopwatch.StartNew()
            let registry = new SharedInitializerRegistry(weightStore, value.Initializers)
            stopwatch.Stop()
            log $"Prepared {registry.InitializerCount} initializer binding(s), {registry.UniqueOrtValueCount} unique ORT value(s), in {elapsedSeconds stopwatch}"
            Some registry
        | _ -> None

    let prepackedWeights = new PrePackedWeightsContainer()

    let graphNames =
        match manifest with
        | Some value -> value.Graphs.Keys |> Seq.toArray |> Array.sort
        | None -> Array.empty

    let missingGraphs =
        if useMergedSession && not (graphNames |> Array.exists ((=) mergedGraphName)) then
            [| mergedGraphName |]
        elif useMergedSession then
            Array.empty
        else
            requiredGraphs
            |> Array.filter (fun name -> not (graphNames |> Array.exists ((=) name)))

    let createOptions graphName graphPath =
        let options = new SessionOptions()
        let cachePath = optimizedModelCachePath graphName
        let existingCachePath = existingOptimizedModelCachePath graphName
        let cacheExists = existingCachePath.IsSome
        options.GraphOptimizationLevel <-
            if cacheExists then GraphOptimizationLevel.ORT_DISABLE_ALL
            else graphOptimizationLevel graphName
        options.LogSeverityLevel <- OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR
        if useQualitySafeOrtMemoryProfile then
            options.EnableCpuMemArena <- false
            options.EnableMemoryPattern <- false
        cachePath
        |> Option.iter (fun path ->
            if not cacheExists then
                match Path.GetDirectoryName(path) with
                | null | "" -> ()
                | dir -> Directory.CreateDirectory(dir) |> ignore
                options.OptimizedModelFilePath <- path)
        if normalizedOptimizedModelCacheFormat = "ort" then
            if cacheExists then
                options.AddSessionConfigEntry("session.load_model_format", "ORT")
                options.AddSessionConfigEntry("session.use_memory_mapped_ort_model", "1")
                options.AddSessionConfigEntry("session.use_ort_model_bytes_for_initializers", "1")
            else
                options.AddSessionConfigEntry("session.save_model_format", "ORT")
        elif normalizedOptimizedModelCacheFormat = "onnx-external" && not cacheExists then
            cachePath
            |> Option.iter (fun path ->
                let externalDataFileName = $"{Path.GetFileName(path)}.data"
                options.AddSessionConfigEntry("session.optimized_model_external_initializers_file_name", externalDataFileName)
                options.AddSessionConfigEntry("session.optimized_model_external_initializers_min_size_in_bytes", "1024"))
        // Keep supplied safetensor-backed initializers on CPU memory so CUDA sessions can accept
        // the shared OrtValue views instead of rejecting them as device-memory mismatches.
        if registry.IsSome then
            options.AddSessionConfigEntry("session.use_device_allocator_for_initializers", "0")
        match executionProvider.Trim().ToLowerInvariant() with
        | "cuda" ->
            if useQualitySafeOrtMemoryProfile then
                let cudaOptions = new OrtCUDAProviderOptions()
                let cudaOptionValues =
                    Dictionary<string, string>(
                        dict [ "device_id", "0"
                               "arena_extend_strategy", "kSameAsRequested"
                               "use_tf32", "0" ]
                    )
                tuningOptions.CudaGpuMemLimitMb
                |> Option.iter (fun limitMb ->
                    if limitMb > 0 then
                        cudaOptionValues["gpu_mem_limit"] <- string (int64 limitMb * 1024L * 1024L))
                cudaOptions.UpdateOptions(
                    cudaOptionValues
                )
                options.AppendExecutionProvider_CUDA(cudaOptions)
                cudaOptions.Dispose()
            else
                options.AppendExecutionProvider_CUDA(0)
        | "cpu" -> options.AppendExecutionProvider_CPU(if useQualitySafeOrtMemoryProfile then 0 else 1)
        | value -> invalidArg (nameof executionProvider) $"Unsupported execution provider '{value}'. Use cuda or cpu."
        options, (if cacheExists then existingCachePath.Value else graphPath)

    let shouldKeepWarm graphName =
        match normalizedMemoryMode with
        | "resident-merged" -> false
        | "warm" -> true
        | "balanced" ->
            graphName = "decoder_prefill"
            || graphName = "decoder_step"
            || graphName = "codec_decode"
        | _ -> false

    let createSession graphName =
        match manifest with
        | None -> invalidOp $"S2S bundle manifest was not found: {manifestPath}"
        | Some value ->
            match value.Graphs.TryGetValue(graphName) with
            | false, _ -> invalidArg (nameof graphName) $"S2S bundle does not contain graph {graphName}."
            | true, graph ->
                if not (File.Exists graph.Path) then
                    invalidArg graphName $"S2S ONNX graph was not found: {graph.Path}"

                requireConfiguredOptimizedCache graphName
                let options, sessionPath = createOptions graphName graph.Path
                registry
                |> Option.iter (fun registryValue ->
                    if graphName = mergedGraphName then
                        let hasMergedInitializerEntries =
                            value.Initializers
                            |> Array.exists (fun entry -> entry.Graph = mergedGraphName)
                        if hasMergedInitializerEntries then
                            registryValue.AddInitializers(options, graphName)
                        else
                            registryValue.AddAllInitializers(options)
                    else
                        registryValue.AddInitializers(options, graphName))
                let graphEntries =
                    value.Initializers
                    |> Array.filter (fun entry -> entry.Graph = graphName)

                let shardCount =
                    graphEntries
                    |> Seq.map (fun entry -> entry.SourceShard)
                    |> Seq.distinct
                    |> Seq.length

                if shardCount > 0 then
                    log $"Ensuring {shardCount} local safetensor external-data link(s) for graph '{graphName}'"
                    SharedWeights.ensureLocalExternalDataLinks modelDir sessionPath graphEntries

                log $"Creating ONNX Runtime session for graph '{graphName}' from {sessionPath}"
                let stopwatch = Stopwatch.StartNew()
                let session = new InferenceSession(sessionPath, options, prepackedWeights)
                stopwatch.Stop()
                log $"ONNX Runtime session for graph '{graphName}' is ready in {elapsedSeconds stopwatch}"
                session, options

    let mergedSession = lazy (createSession mergedGraphName)
    let warmSessions = ConcurrentDictionary<string, Lazy<InferenceSession * SessionOptions>>(StringComparer.Ordinal)
    let activePagedSessions = ConcurrentDictionary<string, byte>(StringComparer.Ordinal)
    let memoryLock = obj()
    let mutable peakPrivateGb = 0.0
    let mutable peakWorkingSetGb = 0.0

    let observeProcessMemory () =
        let proc = Process.GetCurrentProcess()
        proc.Refresh()
        let privateGb = Math.Round(float proc.PrivateMemorySize64 / 1024.0 / 1024.0 / 1024.0, 3)
        let workingSetGb = Math.Round(float proc.WorkingSet64 / 1024.0 / 1024.0 / 1024.0, 3)
        lock memoryLock (fun () ->
            peakPrivateGb <- max peakPrivateGb privateGb
            peakWorkingSetGb <- max peakWorkingSetGb workingSetGb)

    do
        observeProcessMemory()
        if useMergedSession && missingGraphs.Length = 0 then
            log "Resident merged mode selected; loading merged S2S ONNX session before service startup completes"
            let stopwatch = Stopwatch.StartNew()
            let _ = mergedSession.Value
            stopwatch.Stop()
            observeProcessMemory()
            log $"Resident merged S2S session loaded in {elapsedSeconds stopwatch}"

    let getWarmSession graphName =
        warmSessions.GetOrAdd(graphName, fun name -> lazy (createSession name)).Value

    let disposeSessionPair (session: InferenceSession, options: SessionOptions) =
        session.Dispose()
        options.Dispose()

    let withSession graphName (action: InferenceSession -> 'T) =
        if shouldKeepWarm graphName then
            let session, _ = getWarmSession graphName
            observeProcessMemory()
            let result = action session
            observeProcessMemory()
            result
        else
            activePagedSessions[graphName] <- 0uy
            let session, options = createSession graphName
            observeProcessMemory()
            try
                let result = action session
                observeProcessMemory()
                result
            finally
                disposeSessionPair (session, options)
                observeProcessMemory()
                let mutable ignored = 0uy
                activePagedSessions.TryRemove(graphName, &ignored) |> ignore

    let audioNumCodebooks =
        match manifest with
        | Some value -> value.AudioNumCodebooks |> Option.defaultValue 8
        | None -> 8

    let graphInfo graphName =
        match manifest with
        | Some value ->
            match value.Graphs.TryGetValue(graphName) with
            | true, graph -> graph
            | false, _ -> invalidArg (nameof graphName) $"S2S bundle does not contain graph {graphName}."
        | None -> invalidOp $"S2S bundle manifest was not found: {manifestPath}"

    let graphInputNames graphName =
        let graph = graphInfo graphName
        if graph.Inputs.Length > 0 then
            graph.Inputs
        else
            invalidOp $"S2S graph {graphName} is missing input metadata in shared_weights_manifest.json."

    let graphOutputNames graphName =
        let graph = graphInfo graphName
        if graph.Outputs.Length > 0 then
            graph.Outputs
        else
            invalidOp $"S2S graph {graphName} is missing output metadata in shared_weights_manifest.json."

    let qualifiedName (graphName: string) (name: string) =
        if useMergedSession then $"{graphName}__{name}" else name

    let originalName (graphName: string) (name: string) =
        if useMergedSession then
            let prefix = $"{graphName}__"
            if name.StartsWith(prefix, StringComparison.Ordinal) then
                name.Substring(prefix.Length)
            else
                name
        else
            name

    let createInput graphName name (tensor: Tensor<'T>) =
        NamedOnnxValue.CreateFromTensor(qualifiedName graphName name, tensor)

    let tensorElementCount (dimensions: int array) =
        dimensions |> Array.fold (fun total dim -> total * max 1 dim) 1

    let dummyFromMetadata (metadata: NodeMetadata) =
        let dimensions =
            metadata.Dimensions
            |> Array.map (fun dim -> if dim > 0 then dim else 1)
        let elementCount = tensorElementCount dimensions
        if metadata.ElementType = typeof<float32> then
            DummyFloat(DenseTensor<float32>(Array.zeroCreate<float32> elementCount, dimensions))
        elif metadata.ElementType = typeof<int64> then
            DummyInt64(DenseTensor<int64>(Array.zeroCreate<int64> elementCount, dimensions))
        else
            invalidOp $"Merged S2S dummy feed does not support input element type {metadata.ElementType}."

    let dummyToInput name dummy =
        match dummy with
        | DummyFloat tensor -> NamedOnnxValue.CreateFromTensor(name, tensor)
        | DummyInt64 tensor -> NamedOnnxValue.CreateFromTensor(name, tensor)

    let mergedDummyInputs =
        lazy
            let session, _ = mergedSession.Value
            session.InputMetadata
            |> Seq.map (fun item -> item.Key, dummyFromMetadata item.Value)
            |> dict

    let createMergedFeeds (inputs: List<NamedOnnxValue>) =
        let actualNames = HashSet<string>(StringComparer.Ordinal)
        let feeds = List<NamedOnnxValue>()
        for input in inputs do
            actualNames.Add(input.Name) |> ignore
            feeds.Add(input)
        for item in mergedDummyInputs.Value do
            if actualNames.Add(item.Key) then
                feeds.Add(dummyToInput item.Key item.Value)
        feeds

    let mergedModeForGraph graphName =
        requiredGraphs
        |> Array.tryFindIndex ((=) graphName)
        |> Option.defaultWith (fun () -> invalidArg (nameof graphName) $"S2S graph {graphName} has no resident-merged mode id.")

    let runGraph graphName (inputs: List<NamedOnnxValue>) (consumer: IDisposableReadOnlyCollection<DisposableNamedOnnxValue> -> 'T) =
        if useMergedSession then
            let session, _ = mergedSession.Value
            let outputs = graphOutputNames graphName |> Array.map (qualifiedName graphName)
            let modeTensor = DenseTensor<int64>([| int64 (mergedModeForGraph graphName) |], Array.empty<int>)
            inputs.Add(NamedOnnxValue.CreateFromTensor("s2s_mode", modeTensor))
            let feeds = createMergedFeeds inputs
            observeProcessMemory()
            use runOptions = new RunOptions()
            runOptions.AddRunConfigEntry("only_execute_path_to_fetches", "1")
            use results = session.Run(feeds, outputs, runOptions)
            let value = consumer results
            observeProcessMemory()
            value
        else
            withSession graphName (fun session ->
                use results = session.Run(inputs)
                consumer results)

    let cloneFloatByName graphName (results: IDisposableReadOnlyCollection<DisposableNamedOnnxValue>) name =
        let expectedName = qualifiedName graphName name
        results
        |> Seq.find (fun value -> value.Name = expectedName)
        |> fun value -> value.AsTensor<float32>()
        |> TensorIO.cloneFloatTensor

    let cloneFloatByNameOwned graphName (results: IDisposableReadOnlyCollection<DisposableNamedOnnxValue>) name =
        let expectedName = qualifiedName graphName name
        results
        |> Seq.find (fun value -> value.Name = expectedName)
        |> fun value -> value.AsTensor<float32>()
        |> TensorIO.cloneFloatTensorOwned

    let cloneInt64ByName graphName (results: IDisposableReadOnlyCollection<DisposableNamedOnnxValue>) name =
        let expectedName = qualifiedName graphName name
        results
        |> Seq.find (fun value -> value.Name = expectedName)
        |> fun value -> value.AsTensor<int64>()
        |> TensorIO.cloneInt64Tensor

    let cloneInt64ByNameOwned graphName (results: IDisposableReadOnlyCollection<DisposableNamedOnnxValue>) name =
        let expectedName = qualifiedName graphName name
        results
        |> Seq.find (fun value -> value.Name = expectedName)
        |> fun value -> value.AsTensor<int64>()
        |> TensorIO.cloneInt64TensorOwned

    let collectFloatOutputs graphName (results: IDisposableReadOnlyCollection<DisposableNamedOnnxValue>) prefix =
        let values = Dictionary<string, DenseTensor<float32>>(StringComparer.Ordinal)
        for result in results do
            let name = originalName graphName result.Name
            if name.StartsWith(prefix, StringComparison.Ordinal) then
                values[name] <- result.AsTensor<float32>() |> TensorIO.cloneFloatTensor
        values

    let collectFloatOutputsOwned graphName (results: IDisposableReadOnlyCollection<DisposableNamedOnnxValue>) prefix =
        let values = Dictionary<string, DenseTensor<float32>>(StringComparer.Ordinal)
        let owners = ResizeArray<IDisposable>()
        try
            for result in results do
                let name = originalName graphName result.Name
                if name.StartsWith(prefix, StringComparison.Ordinal) then
                    let tensor, owner = result.AsTensor<float32>() |> TensorIO.cloneFloatTensorOwned
                    values[name] <- tensor
                    owners.Add(owner)
            values, owners.ToArray()
        with
        | _ ->
            for owner in owners do
                owner.Dispose()
            reraise()

    let addOwnedTensor (owners: ResizeArray<IDisposable>) (tensor, owner: IDisposable) =
        owners.Add(owner)
        tensor

    let disposeOwners (owners: ResizeArray<IDisposable>) =
        for owner in owners do
            owner.Dispose()

    let addBackboneCacheInputs graphName (inputs: List<NamedOnnxValue>) (cache: Dictionary<string, DenseTensor<float32>>) =
        for name in graphInputNames graphName do
            if name.StartsWith("backbone_past_", StringComparison.Ordinal) then
                let presentName = name.Replace("_past_", "_present_")
                match cache.TryGetValue(presentName) with
                | true, tensor -> inputs.Add(createInput graphName name tensor)
                | false, _ -> invalidOp $"Backbone cache tensor {presentName} is missing for input {name}."

    let addThinkerCacheInputs graphName (inputs: List<NamedOnnxValue>) (cache: Dictionary<string, DenseTensor<float32>>) =
        for name in graphInputNames graphName do
            if name.StartsWith("thinker_past_", StringComparison.Ordinal) then
                let presentName = name.Replace("_past_", "_present_")
                match cache.TryGetValue(presentName) with
                | true, tensor -> inputs.Add(createInput graphName name tensor)
                | false, _ -> invalidOp $"Thinker cache tensor {presentName} is missing for input {name}."

    let addDecoderCacheInputs graphName (inputs: List<NamedOnnxValue>) (cache: Dictionary<string, DenseTensor<float32>>) =
        for name in graphInputNames graphName do
            if name.StartsWith("decoder_past_", StringComparison.Ordinal) then
                let presentName = name.Replace("_past_", "_present_")
                match cache.TryGetValue(presentName) with
                | true, tensor -> inputs.Add(createInput graphName name tensor)
                | false, _ -> invalidOp $"Decoder cache tensor {presentName} is missing for input {name}."

    let disposeWarmSessions () =
        for session in warmSessions.Values do
            if session.IsValueCreated then
                disposeSessionPair session.Value
        warmSessions.Clear()

    let warmSessionNames () =
        warmSessions
        |> Seq.choose (fun item -> if item.Value.IsValueCreated then Some item.Key else None)
        |> Seq.sort
        |> Seq.toArray

    let activePagedSessionNames () =
        activePagedSessions.Keys
        |> Seq.sort
        |> Seq.toArray

    let residentMergedSessionNames () =
        if mergedSession.IsValueCreated then [| mergedGraphName |] else Array.empty

    let runDecoder (inputIds: DenseTensor<int64>) (backboneLastHiddenState: DenseTensor<float32>) =
        let graphName = "decoder"
        let inputs = List<NamedOnnxValue>()
        inputs.Add(createInput graphName "input_ids" inputIds)
        inputs.Add(createInput graphName "backbone_last_hidden_state" backboneLastHiddenState)
        runGraph graphName inputs (fun results -> cloneFloatByName graphName results "logits")

    let decoderCacheSequenceLength (cache: Dictionary<string, DenseTensor<float32>>) =
        cache.Values
        |> Seq.tryHead
        |> Option.map (fun tensor -> tensor.Dimensions[2])
        |> Option.defaultWith (fun () -> invalidOp "Decoder cache is empty.")

    let rentedDecoderAttentionMask (batch: int) (sequenceLength: int) =
        let buffer = RentedTensorBuffer.rent<int64> (batch * sequenceLength) false
        try
            buffer.Memory.Span.Fill(1L)
            DenseTensor<int64>(buffer.Memory, ReadOnlySpan<int>([| batch; sequenceLength |]), false), buffer :> IDisposable
        with
        | _ ->
            (buffer :> IDisposable).Dispose()
            reraise()

    let rentedDecoderCachePosition (start: int) (length: int) =
        let buffer = RentedTensorBuffer.rent<int64> length false
        try
            let values = buffer.Memory.Span
            for index in 0 .. length - 1 do
                values[index] <- int64 (start + index)
            DenseTensor<int64>(buffer.Memory, ReadOnlySpan<int>([| length |]), false), buffer :> IDisposable
        with
        | _ ->
            (buffer :> IDisposable).Dispose()
            reraise()

    let runDecoderPrefill (inputIds: DenseTensor<int64>) (backboneLastHiddenState: DenseTensor<float32>) =
        let graphName = "decoder_prefill"
        let batch = inputIds.Dimensions[0]
        let sequenceLength = inputIds.Dimensions[1]
        let attentionMask, attentionMaskBuffer = rentedDecoderAttentionMask batch (sequenceLength + 1)
        use _attentionMaskOwner = attentionMaskBuffer
        let cachePosition, cachePositionBuffer = rentedDecoderCachePosition 0 sequenceLength
        use _cachePositionOwner = cachePositionBuffer
        let inputs = List<NamedOnnxValue>()
        inputs.Add(createInput graphName "input_ids" inputIds)
        inputs.Add(createInput graphName "backbone_last_hidden_state" backboneLastHiddenState)
        inputs.Add(createInput graphName "attention_mask" attentionMask)
        inputs.Add(createInput graphName "cache_position" cachePosition)
        runGraph graphName inputs (fun results ->
            let logits, logitsOwner = cloneFloatByNameOwned graphName results "logits"
            try
                let cache, cacheOwners = collectFloatOutputsOwned graphName results "decoder_present_"
                new OwnedDecoderOutput(logits, cache, Array.append [| logitsOwner |] cacheOwners)
            with
            | _ ->
                logitsOwner.Dispose()
                reraise())

    let runDecoderStep (inputIds: DenseTensor<int64>) (cache: Dictionary<string, DenseTensor<float32>>) =
        let graphName = "decoder_step"
        let batch = inputIds.Dimensions[0]
        let sequenceLength = inputIds.Dimensions[1]
        let pastSequenceLength = decoderCacheSequenceLength cache
        let attentionMask, attentionMaskBuffer = rentedDecoderAttentionMask batch (pastSequenceLength + sequenceLength)
        use _attentionMaskOwner = attentionMaskBuffer
        let cachePosition, cachePositionBuffer = rentedDecoderCachePosition (pastSequenceLength - sequenceLength) sequenceLength
        use _cachePositionOwner = cachePositionBuffer
        let inputs = List<NamedOnnxValue>()
        inputs.Add(createInput graphName "input_ids" inputIds)
        inputs.Add(createInput graphName "attention_mask" attentionMask)
        inputs.Add(createInput graphName "cache_position" cachePosition)
        addDecoderCacheInputs graphName inputs cache
        runGraph graphName inputs (fun results ->
            let logits, logitsOwner = cloneFloatByNameOwned graphName results "logits"
            try
                let nextCache, cacheOwners = collectFloatOutputsOwned graphName results "decoder_present_"
                new OwnedDecoderOutput(logits, nextCache, Array.append [| logitsOwner |] cacheOwners)
            with
            | _ ->
                logitsOwner.Dispose()
                reraise())

    let audioFrame (samplingOptions: S2sSamplingOptions) (logits: DenseTensor<float32>) (hiddenStates: DenseTensor<float32>) =
        let selectNextIds (logits: DenseTensor<float32>) =
            if samplingOptions.Enabled then
                TensorMath.sampleLast Random.Shared samplingOptions.Temperature samplingOptions.TopP samplingOptions.TopK logits
            else
                TensorMath.argmaxLast logits

        let firstIds = selectNextIds logits
        let hidden = TensorMath.lastHidden hiddenStates
        let mutable ids = DenseTensor<int64>(firstIds, [| firstIds.Length; 1 |])
        let mutable decoderOutput = runDecoderPrefill ids hidden
        try
            let mutable nextIds = selectNextIds decoderOutput.Logits
            ids <- TensorMath.appendColumn ids nextIds

            for _ in 3 .. audioNumCodebooks do
                let stepInput = DenseTensor<int64>(nextIds, [| nextIds.Length; 1 |])
                let nextDecoderOutput = runDecoderStep stepInput decoderOutput.Cache
                let previousDecoderOutput = decoderOutput
                decoderOutput <- nextDecoderOutput
                (previousDecoderOutput :> IDisposable).Dispose()
                nextIds <- selectNextIds decoderOutput.Logits
                ids <- TensorMath.appendColumn ids nextIds
        finally
            (decoderOutput :> IDisposable).Dispose()

        ids

    let greedyAudioFrame (logits: DenseTensor<float32>) (hiddenStates: DenseTensor<float32>) =
        audioFrame S2sSamplingOptions.Greedy logits hiddenStates

    let validateAudioFrame (frame: DenseTensor<int64>) =
        let dims = frame.Dimensions
        if dims.Length <> 2 || dims[0] <> 1 || dims[1] <> audioNumCodebooks then
            invalidArg (nameof frame) $"Expected audio frame shape [1, {audioNumCodebooks}]."

    let copyFrameToSpan (frame: DenseTensor<int64>) (destination: Span<int64>) =
        validateAudioFrame frame
        if destination.Length < audioNumCodebooks then
            invalidArg (nameof destination) $"Destination must have room for {audioNumCodebooks} codebooks."

        let source = frame.Buffer.Span
        let strides = frame.Strides

        if strides[1] = 1 then
            source.Slice(0, audioNumCodebooks).CopyTo(destination)
        else
            for codebookIndex in 0 .. audioNumCodebooks - 1 do
                destination[codebookIndex] <- source[codebookIndex * strides[1]]

    let frameToInputIds (frame: DenseTensor<int64>) =
        validateAudioFrame frame
        let strides = frame.Strides
        let dimensions = [| 1; 1; audioNumCodebooks |]

        if strides[0] = audioNumCodebooks && strides[1] = 1 then
            DenseTensor<int64>(frame.Buffer, ReadOnlySpan<int>(dimensions), false)
        else
            let values = Array.zeroCreate<int64> audioNumCodebooks
            copyFrameToSpan frame (values.AsSpan())
            DenseTensor<int64>(values, dimensions)

    let appendFrame (frames: ResizeArray<int64>) (frame: DenseTensor<int64>) =
        let frameStart = frames.Count
        for _ in 1 .. audioNumCodebooks do
            frames.Add(0L)
        copyFrameToSpan frame (CollectionsMarshal.AsSpan(frames).Slice(frameStart, audioNumCodebooks))

    let frameIsEos (frame: DenseTensor<int64>) =
        validateAudioFrame frame
        let values = frame.Buffer.Span
        let strides = frame.Strides
        let checkedCount = max 1 (audioNumCodebooks - 1)
        let mutable isEos = true
        let mutable codebookIndex = 0

        while isEos && codebookIndex < checkedCount do
            if values[codebookIndex * strides[1]] <> 0L then
                isEos <- false
            codebookIndex <- codebookIndex + 1

        isEos

    let frameHasAnyNonZero (frame: DenseTensor<int64>) =
        validateAudioFrame frame
        let values = frame.Buffer.Span
        let strides = frame.Strides
        let mutable found = false
        let mutable codebookIndex = 0

        while not found && codebookIndex < audioNumCodebooks do
            if values[codebookIndex * strides[1]] <> 0L then
                found <- true
            codebookIndex <- codebookIndex + 1

        found

    let frameArraysEqual (left: int64 array) (right: int64 array) =
        if left.Length <> right.Length then
            false
        else
            let mutable equal = true
            let mutable index = 0

            while equal && index < left.Length do
                if left[index] <> right[index] then
                    equal <- false
                index <- index + 1

            equal

    let tensorContainsInt64 value (tensor: DenseTensor<int64>) =
        let values = tensor.Buffer.Span
        let mutable found = false
        let mutable index = 0

        while not found && index < values.Length do
            if values[index] = value then
                found <- true
            index <- index + 1

        found

    let frameCount (frames: ResizeArray<int64>) =
        frames.Count / audioNumCodebooks

    let framesToCodecTensorWithCount (frames: ResizeArray<int64>) requestedFrameCount =
        let availableFrameCount = frameCount frames
        let frameCount = max 0 (min requestedFrameCount availableFrameCount)
        let values = Array.zeroCreate<int64> (audioNumCodebooks * frameCount)
        for frameIndex in 0 .. frameCount - 1 do
            for codebookIndex in 0 .. audioNumCodebooks - 1 do
                values[codebookIndex * frameCount + frameIndex] <- frames[(frameIndex * audioNumCodebooks) + codebookIndex]
        DenseTensor<int64>(values, [| 1; audioNumCodebooks; frameCount |])

    let framesToCodecTensor (frames: ResizeArray<int64>) =
        framesToCodecTensorWithCount frames (frameCount frames)

    let frameToArray (frame: DenseTensor<int64>) =
        let values = Array.zeroCreate<int64> audioNumCodebooks
        copyFrameToSpan frame (values.AsSpan())
        values

    let audioSamplesFrom (startSample: int) (audio: DenseTensor<float32>) =
        let totalSamples = tensorElementCount (audio.Dimensions.ToArray())
        let startSample = max 0 (min startSample totalSamples)
        let count = totalSamples - startSample
        let samples = Array.zeroCreate<float32> count
        if count > 0 then
            audio.Buffer.Span.Slice(startSample, count).CopyTo(samples.AsSpan())
        samples

    let audioPrefix sampleCount (audio: DenseTensor<float32>) =
        let totalSamples = tensorElementCount (audio.Dimensions.ToArray())
        let sampleCount = max 0 (min sampleCount totalSamples)
        let samples = Array.zeroCreate<float32> sampleCount
        if sampleCount > 0 then
            audio.Buffer.Span.Slice(0, sampleCount).CopyTo(samples.AsSpan())
        DenseTensor<float32>(samples, [| 1; 1; sampleCount |])

    let runCodecDecode (audioCodes: DenseTensor<int64>) =
        let graphName = "codec_decode"
        let inputs = List<NamedOnnxValue>()
        inputs.Add(createInput graphName "audio_codes" audioCodes)
        runGraph graphName inputs (fun results -> cloneFloatByNameOwned graphName results "audio_values")

    let runGeneratePrefill (prepared: NativeS2sPrepared) =
        let graphName = "generate_prefill"
        let inputs = List<NamedOnnxValue>()
        inputs.Add(createInput graphName "input_ids" prepared.InputIds)
        inputs.Add(createInput graphName "attention_mask" prepared.AttentionMask)
        inputs.Add(createInput graphName "input_values" prepared.InputValues)
        inputs.Add(createInput graphName "input_values_cutoffs" prepared.InputValuesCutoffs)
        inputs.Add(createInput graphName "thinker_input_ids" prepared.ThinkerInputIds)
        inputs.Add(createInput graphName "thinker_attention_mask" prepared.ThinkerAttentionMask)
        inputs.Add(createInput graphName "thinker_input_features" prepared.ThinkerInputFeatures)
        inputs.Add(createInput graphName "thinker_feature_attention_mask" prepared.ThinkerFeatureAttentionMask)
        runGraph graphName inputs (fun results ->
            let stepOwners = ResizeArray<IDisposable>()
            let backboneOwners = ResizeArray<IDisposable>()
            let thinkerOwners = ResizeArray<IDisposable>()
            try
                let logits = cloneFloatByNameOwned graphName results "logits" |> addOwnedTensor stepOwners
                let hiddenStates = cloneFloatByNameOwned graphName results "hidden_states" |> addOwnedTensor stepOwners
                let attentionMask = cloneFloatByNameOwned graphName results "next_attention_mask" |> addOwnedTensor backboneOwners
                let thinkerInputIds = cloneInt64ByNameOwned graphName results "next_thinker_input_ids" |> addOwnedTensor thinkerOwners
                let thinkerAttentionMask = cloneInt64ByNameOwned graphName results "next_thinker_attention_mask" |> addOwnedTensor thinkerOwners
                let thinkerCachePosition = cloneInt64ByNameOwned graphName results "next_thinker_cache_position" |> addOwnedTensor thinkerOwners
                let thinkerEos = cloneInt64ByNameOwned graphName results "next_thinker_eos" |> addOwnedTensor thinkerOwners
                let backboneCache, backboneCacheOwners = collectFloatOutputsOwned graphName results "backbone_present_"
                let thinkerCache, thinkerCacheOwners = collectFloatOutputsOwned graphName results "thinker_present_"
                backboneOwners.AddRange(backboneCacheOwners)
                thinkerOwners.AddRange(thinkerCacheOwners)
                { Logits = logits
                  HiddenStates = hiddenStates
                  State =
                    { AttentionMask = attentionMask
                      ThinkerInputIds = thinkerInputIds
                      ThinkerAttentionMask = thinkerAttentionMask
                      ThinkerCachePosition = thinkerCachePosition
                      ThinkerEos = thinkerEos
                      BackboneCache = backboneCache
                      ThinkerCache = thinkerCache
                      BackboneOwnedBuffers = OwnedBufferGroup.ofArray (backboneOwners.ToArray())
                      ThinkerOwnedBuffers = OwnedBufferGroup.ofArray (thinkerOwners.ToArray()) }
                  OwnedBuffers = stepOwners.ToArray() }
            with
            | _ ->
                disposeOwners stepOwners
                disposeOwners backboneOwners
                disposeOwners thinkerOwners
                reraise())

    let runBackboneFrameStep transferThinkerOwnership (frameCodes: DenseTensor<int64>) (state: S2sGraphState) =
        let graphName = "backbone_frame_step"
        let inputs = List<NamedOnnxValue>()
        inputs.Add(createInput graphName "frame_codes" frameCodes)
        inputs.Add(createInput graphName "attention_mask" state.AttentionMask)
        addBackboneCacheInputs graphName inputs state.BackboneCache
        runGraph graphName inputs (fun results ->
            let stepOwners = ResizeArray<IDisposable>()
            let backboneOwners = ResizeArray<IDisposable>()
            let mutable transferredThinkerOwners: OwnedBufferGroup option = None
            try
                let logits = cloneFloatByNameOwned graphName results "logits" |> addOwnedTensor stepOwners
                let hiddenStates = cloneFloatByNameOwned graphName results "hidden_states" |> addOwnedTensor stepOwners
                let attentionMask = cloneFloatByNameOwned graphName results "next_attention_mask" |> addOwnedTensor backboneOwners
                let backboneCache, backboneCacheOwners = collectFloatOutputsOwned graphName results "backbone_present_"
                backboneOwners.AddRange(backboneCacheOwners)
                let thinkerOwnedBuffers =
                    if transferThinkerOwnership then
                        let transferred = state.ThinkerOwnedBuffers.Transfer()
                        transferredThinkerOwners <- Some transferred
                        transferred
                    else
                        OwnedBufferGroup.empty ()
                { Logits = logits
                  HiddenStates = hiddenStates
                  State =
                    { AttentionMask = attentionMask
                      ThinkerInputIds = state.ThinkerInputIds
                      ThinkerAttentionMask = state.ThinkerAttentionMask
                      ThinkerCachePosition = state.ThinkerCachePosition
                      ThinkerEos = state.ThinkerEos
                      BackboneCache = backboneCache
                      ThinkerCache = state.ThinkerCache
                      BackboneOwnedBuffers = OwnedBufferGroup.ofArray (backboneOwners.ToArray())
                      ThinkerOwnedBuffers = thinkerOwnedBuffers }
                  OwnedBuffers = stepOwners.ToArray() }
            with
            | _ ->
                disposeOwners stepOwners
                disposeOwners backboneOwners
                transferredThinkerOwners
                |> Option.iter (fun owners -> (owners :> IDisposable).Dispose())
                reraise())

    let runBackboneThinkerStep (frameCodes: DenseTensor<int64>) (state: S2sGraphState) =
        let graphName = "backbone_thinker_step"
        let inputs = List<NamedOnnxValue>()
        inputs.Add(createInput graphName "frame_codes" frameCodes)
        inputs.Add(createInput graphName "attention_mask" state.AttentionMask)
        inputs.Add(createInput graphName "thinker_input_ids" state.ThinkerInputIds)
        inputs.Add(createInput graphName "thinker_attention_mask" state.ThinkerAttentionMask)
        inputs.Add(createInput graphName "thinker_cache_position" state.ThinkerCachePosition)
        inputs.Add(createInput graphName "thinker_eos" state.ThinkerEos)
        addBackboneCacheInputs graphName inputs state.BackboneCache
        addThinkerCacheInputs graphName inputs state.ThinkerCache
        runGraph graphName inputs (fun results ->
            let stepOwners = ResizeArray<IDisposable>()
            let backboneOwners = ResizeArray<IDisposable>()
            let thinkerOwners = ResizeArray<IDisposable>()
            try
                let logits = cloneFloatByNameOwned graphName results "logits" |> addOwnedTensor stepOwners
                let hiddenStates = cloneFloatByNameOwned graphName results "hidden_states" |> addOwnedTensor stepOwners
                let attentionMask = cloneFloatByNameOwned graphName results "next_attention_mask" |> addOwnedTensor backboneOwners
                let thinkerInputIds = cloneInt64ByNameOwned graphName results "next_thinker_input_ids" |> addOwnedTensor thinkerOwners
                let thinkerAttentionMask = cloneInt64ByNameOwned graphName results "next_thinker_attention_mask" |> addOwnedTensor thinkerOwners
                let thinkerCachePosition = cloneInt64ByNameOwned graphName results "next_thinker_cache_position" |> addOwnedTensor thinkerOwners
                let thinkerEos = cloneInt64ByNameOwned graphName results "next_thinker_eos" |> addOwnedTensor thinkerOwners
                let backboneCache, backboneCacheOwners = collectFloatOutputsOwned graphName results "backbone_present_"
                let thinkerCache, thinkerCacheOwners = collectFloatOutputsOwned graphName results "thinker_present_"
                backboneOwners.AddRange(backboneCacheOwners)
                thinkerOwners.AddRange(thinkerCacheOwners)
                { Logits = logits
                  HiddenStates = hiddenStates
                  State =
                    { AttentionMask = attentionMask
                      ThinkerInputIds = thinkerInputIds
                      ThinkerAttentionMask = thinkerAttentionMask
                      ThinkerCachePosition = thinkerCachePosition
                      ThinkerEos = thinkerEos
                      BackboneCache = backboneCache
                      ThinkerCache = thinkerCache
                      BackboneOwnedBuffers = OwnedBufferGroup.ofArray (backboneOwners.ToArray())
                      ThinkerOwnedBuffers = OwnedBufferGroup.ofArray (thinkerOwners.ToArray()) }
                  OwnedBuffers = stepOwners.ToArray() }
            with
            | _ ->
                disposeOwners stepOwners
                disposeOwners backboneOwners
                disposeOwners thinkerOwners
                reraise())

    let debugJsonOptions =
        JsonSerializerOptions(PropertyNamingPolicy = JsonNamingPolicy.CamelCase, WriteIndented = true)

    let writeDebugFloat (outputDir: string) (infos: Dictionary<string, DebugTensorInfo>) name (tensor: DenseTensor<float32>) =
        let file = $"{name}.f32"
        TensorIO.writeSingles (Path.Combine(outputDir, file)) tensor
        infos[name] <- { File = file; Dtype = "f32"; Shape = tensor.Dimensions.ToArray() }

    let writeDebugInt64 (outputDir: string) (infos: Dictionary<string, DebugTensorInfo>) name (tensor: DenseTensor<int64>) =
        let file = $"{name}.i64"
        TensorIO.writeInt64s (Path.Combine(outputDir, file)) tensor
        infos[name] <- { File = file; Dtype = "i64"; Shape = tensor.Dimensions.ToArray() }

    let writeDebugInt64Array (outputDir: string) (infos: Dictionary<string, DebugTensorInfo>) name (values: int64 array) (shape: int array) =
        writeDebugInt64 outputDir infos name (DenseTensor<int64>(values, shape))

    new(modelDir: string, bundleDir: string, executionProvider: string, memoryMode: string) =
        new ChromaS2sOnnxRunner(modelDir, bundleDir, executionProvider, memoryMode, S2sOrtTuningOptions.Default)

    new(modelDir: string, bundleDir: string, executionProvider: string) =
        new ChromaS2sOnnxRunner(modelDir, bundleDir, executionProvider, "python-footprint", S2sOrtTuningOptions.Default)

    member _.Status =
        let bundle =
            match manifest with
            | Some value -> value.BundleDir
            | None -> bundleDir

        let message =
            if not (File.Exists manifestPath) then
                $"S2S bundle manifest was not found: {manifestPath}"
            elif missingGraphs.Length > 0 then
                let missing = String.Join(", ", missingGraphs)
                $"Bundle is not S2S-capable yet. Missing graphs: {missing}"
            else
                "S2S bundle is ready."

        { Ready = File.Exists manifestPath && missingGraphs.Length = 0
          Bundle = bundle
          MissingGraphs = missingGraphs
          AvailableGraphs = graphNames
          ExecutionProvider = executionProvider
          Message = message }

    member _.MemoryMode = normalizedMemoryMode

    member _.OrtMemoryProfile = normalizedOrtMemoryProfile

    member _.OptimizedModelCacheDir =
        optimizedModelCacheDir |> Option.toObj

    member _.OptimizedModelCacheEnabled = optimizedModelCacheDir.IsSome

    member _.OptimizedModelCacheFormat = normalizedOptimizedModelCacheFormat

    member _.CudaGpuMemLimitMb = tuningOptions.CudaGpuMemLimitMb

    member _.MappedShardCount =
        store
        |> Option.map (fun value -> value.MappedShardCount)
        |> Option.defaultValue 0

    member _.InitializerCount =
        registry
        |> Option.map (fun value -> value.InitializerCount)
        |> Option.defaultValue 0

    member _.UniqueSourceTensorCount =
        registry
        |> Option.map (fun value -> value.UniqueSourceTensorCount)
        |> Option.defaultValue 0

    member _.UniqueOrtValueCount =
        registry
        |> Option.map (fun value -> value.UniqueOrtValueCount)
        |> Option.defaultValue 0

    member _.SharedPrepackedWeights = true

    member _.WarmSessionNames = warmSessionNames ()

    member _.ActivePagedSessionNames = activePagedSessionNames ()

    member _.LoadedSessionNames =
        Seq.concat [ residentMergedSessionNames (); warmSessionNames (); activePagedSessionNames () ]
        |> Seq.distinct
        |> Seq.sort
        |> Seq.toArray

    member _.PeakPrivateGb =
        lock memoryLock (fun () -> peakPrivateGb)

    member _.PeakWorkingSetGb =
        lock memoryLock (fun () -> peakWorkingSetGb)

    member this.EnsureReady() =
        let status = this.Status
        if not status.Ready then
            invalidOp status.Message

    member _.CreateSessionOptions() =
        requireConfiguredOptimizedCache mergedGraphName
        fst (createOptions mergedGraphName "")

    member this.WriteDebug(prepared: NativeS2sPrepared, outputDir: string, ?maxFrames: int) =
        this.EnsureReady()
        Directory.CreateDirectory(outputDir) |> ignore
        let infos = Dictionary<string, DebugTensorInfo>(StringComparer.Ordinal)
        let maxDebugFrames = defaultArg maxFrames 8

        let prefill = runGeneratePrefill prepared
        let mutable traceState = prefill.State
        use _prefillOwner = prefill :> IDisposable
        use _traceStateOwner =
            { new IDisposable with
                member _.Dispose() = (traceState :> IDisposable).Dispose() }

        writeDebugFloat outputDir infos "prefill_logits" prefill.Logits
        writeDebugFloat outputDir infos "prefill_hidden_states" prefill.HiddenStates
        writeDebugFloat outputDir infos "prefill_next_attention_mask" prefill.State.AttentionMask
        writeDebugInt64 outputDir infos "prefill_next_thinker_input_ids" prefill.State.ThinkerInputIds
        writeDebugInt64 outputDir infos "prefill_next_thinker_attention_mask" prefill.State.ThinkerAttentionMask
        writeDebugInt64 outputDir infos "prefill_next_thinker_cache_position" prefill.State.ThinkerCachePosition
        writeDebugInt64 outputDir infos "prefill_next_thinker_eos" prefill.State.ThinkerEos

        let firstIds = TensorMath.argmaxLast prefill.Logits
        writeDebugInt64Array outputDir infos "codebook0_ids" firstIds [| firstIds.Length; 1 |]

        let hidden = TensorMath.lastHidden prefill.HiddenStates
        writeDebugFloat outputDir infos "decoder_backbone_last_hidden_state" hidden

        let mutable ids = DenseTensor<int64>(firstIds, [| firstIds.Length; 1 |])
        writeDebugInt64 outputDir infos "decoder_cache_step_1_input_ids" ids
        let mutable decoderOutput = runDecoderPrefill ids hidden
        try
            writeDebugFloat outputDir infos "decoder_cache_step_1_logits" decoderOutput.Logits
            let mutable nextIds = TensorMath.argmaxLast decoderOutput.Logits
            writeDebugInt64Array outputDir infos "decoder_cache_step_1_next_ids" nextIds [| nextIds.Length; 1 |]
            ids <- TensorMath.appendColumn ids nextIds

            for stepIndex in 2 .. audioNumCodebooks - 1 do
                let stepInput = DenseTensor<int64>(nextIds, [| nextIds.Length; 1 |])
                writeDebugInt64 outputDir infos $"decoder_cache_step_{stepIndex}_input_ids" stepInput
                let nextDecoderOutput = runDecoderStep stepInput decoderOutput.Cache
                let previousDecoderOutput = decoderOutput
                decoderOutput <- nextDecoderOutput
                (previousDecoderOutput :> IDisposable).Dispose()
                writeDebugFloat outputDir infos $"decoder_cache_step_{stepIndex}_logits" decoderOutput.Logits
                nextIds <- TensorMath.argmaxLast decoderOutput.Logits
                writeDebugInt64Array outputDir infos $"decoder_cache_step_{stepIndex}_next_ids" nextIds [| nextIds.Length; 1 |]
                ids <- TensorMath.appendColumn ids nextIds
        finally
            (decoderOutput :> IDisposable).Dispose()

        writeDebugInt64 outputDir infos "decoder_cache_frame_ids" ids
        writeDebugInt64 outputDir infos "decoder_generate_frame_ids" ids
        writeDebugInt64 outputDir infos "decoder_loop_frame_ids" ids

        let frameInput = frameToInputIds ids
        let frameStep = runBackboneFrameStep false frameInput prefill.State
        try
            writeDebugInt64 outputDir infos "backbone_frame_step_input_ids" frameInput
            writeDebugFloat outputDir infos "backbone_frame_step_logits" frameStep.Logits
            writeDebugFloat outputDir infos "backbone_frame_step_hidden_states" frameStep.HiddenStates
            let nextCodebook0 = TensorMath.argmaxLast frameStep.Logits
            writeDebugInt64Array outputDir infos "backbone_frame_step_codebook0_ids" nextCodebook0 [| nextCodebook0.Length; 1 |]
        finally
            (frameStep.State :> IDisposable).Dispose()
            (frameStep :> IDisposable).Dispose()

        let mutable traceCurrent = ids
        let mutable traceInput = frameToInputIds traceCurrent
        let mutable traceUseThinker = false
        writeDebugInt64 outputDir infos "trace_frame_0_ids" traceCurrent
        writeDebugFloat outputDir infos "trace_frame_0_logits" prefill.Logits
        writeDebugFloat outputDir infos "trace_frame_0_hidden_states" prefill.HiddenStates
        writeDebugFloat outputDir infos "trace_state_0_attention_mask" traceState.AttentionMask
        writeDebugInt64 outputDir infos "trace_state_0_thinker_input_ids" traceState.ThinkerInputIds
        writeDebugInt64 outputDir infos "trace_state_0_thinker_attention_mask" traceState.ThinkerAttentionMask
        writeDebugInt64 outputDir infos "trace_state_0_thinker_cache_position" traceState.ThinkerCachePosition
        writeDebugInt64 outputDir infos "trace_state_0_thinker_eos" traceState.ThinkerEos

        for frameIndex in 1 .. max 0 (maxDebugFrames - 1) do
            let thinkerActive = tensorContainsInt64 0L traceState.ThinkerEos
            let stepKind, traceStep =
                if traceUseThinker && thinkerActive then
                    traceUseThinker <- false
                    "thinker", runBackboneThinkerStep traceInput traceState
                else
                    traceUseThinker <- thinkerActive
                    "frame", runBackboneFrameStep true traceInput traceState

            let previousTraceState = traceState
            traceState <- traceStep.State
            (previousTraceState :> IDisposable).Dispose()
            try
                writeDebugInt64Array outputDir infos $"trace_step_{frameIndex}_kind" [| if stepKind = "thinker" then 1L else 0L |] [| 1 |]
                writeDebugFloat outputDir infos $"trace_step_{frameIndex}_logits" traceStep.Logits
                writeDebugFloat outputDir infos $"trace_step_{frameIndex}_hidden_states" traceStep.HiddenStates
                traceCurrent <- greedyAudioFrame traceStep.Logits traceStep.HiddenStates
                traceInput <- frameToInputIds traceCurrent
                writeDebugInt64 outputDir infos $"trace_frame_{frameIndex}_ids" traceCurrent
                writeDebugFloat outputDir infos $"trace_state_{frameIndex}_attention_mask" traceState.AttentionMask
                writeDebugInt64 outputDir infos $"trace_state_{frameIndex}_thinker_input_ids" traceState.ThinkerInputIds
                writeDebugInt64 outputDir infos $"trace_state_{frameIndex}_thinker_attention_mask" traceState.ThinkerAttentionMask
                writeDebugInt64 outputDir infos $"trace_state_{frameIndex}_thinker_cache_position" traceState.ThinkerCachePosition
                writeDebugInt64 outputDir infos $"trace_state_{frameIndex}_thinker_eos" traceState.ThinkerEos
            finally
                (traceStep :> IDisposable).Dispose()

        let manifestJson =
            JsonSerializer.Serialize(
                {| runtime = "fsharp_onnx"
                   executionProvider = this.Status.ExecutionProvider
                   tensors = infos |},
                debugJsonOptions
            )
        File.WriteAllText(Path.Combine(outputDir, "debug_manifest.json"), manifestJson)

    member this.Generate(prepared: NativeS2sPrepared, maxNewFrames: int) =
        this.GenerateStreaming(
            prepared,
            maxNewFrames,
            4,
            ignore,
            ignore,
            CancellationToken.None
        )

    member this.GenerateStreaming
        (
            prepared: NativeS2sPrepared,
            maxNewFrames: int,
            streamDecodeFrames: int,
            onFrame: S2sGeneratedFrame -> unit,
            onAudioChunk: S2sAudioChunk -> unit,
            cancellationToken: CancellationToken,
            ?shouldDecodeChunk: int -> bool,
            ?codecStallGuardFrames: int,
            ?samplingOptions: S2sSamplingOptions
        ) =
        this.EnsureReady()
        if maxNewFrames < 1 then
            invalidArg (nameof maxNewFrames) "maxNewFrames must be positive."

        let streamDecodeFrames = max 1 streamDecodeFrames
        let codecStallGuardFrames = defaultArg codecStallGuardFrames 16 |> max 0
        let samplingOptions = defaultArg samplingOptions S2sSamplingOptions.Greedy
        let outputSampleRate = 24000
        let outputSamplesPerFrame = 1920
        let decodedSilenceRmsThreshold = 0.001
        let decodedSilenceGuardSeconds = 1.0
        let decodedSilenceGuardSamples = int (float outputSampleRate * decodedSilenceGuardSeconds)
        let decodedSilenceGuardEnabled = codecStallGuardFrames > 0
        let timings = Dictionary<string, float>(StringComparer.Ordinal)
        timings["samplingEnabled"] <- if samplingOptions.Enabled then 1.0 else 0.0
        timings["samplingTemperature"] <- samplingOptions.Temperature
        timings["samplingTopP"] <- samplingOptions.TopP
        timings["samplingTopK"] <- float samplingOptions.TopK
        let stopwatch = Stopwatch.StartNew()
        let shouldDecodeChunk = defaultArg shouldDecodeChunk (fun _ -> true)
        cancellationToken.ThrowIfCancellationRequested()
        let prefill = runGeneratePrefill prepared
        timings["prefillMs"] <- stopwatch.Elapsed.TotalMilliseconds

        let frames = ResizeArray<int64>(maxNewFrames * audioNumCodebooks)
        let stepKinds = ResizeArray<string>()
        let mutable emittedFrameStart = 0
        let mutable nextStreamDecodeFrame = streamDecodeFrames
        let mutable emittedSampleStart = 0
        let mutable analyzedSampleStart = 0
        let mutable chunkIndex = 0
        let mutable decodeMs = 0.0
        let mutable streamDecodeSkips = 0
        let mutable hasAudibleDecodedSamples = false
        let mutable decodedSilenceRunSamples = 0
        let mutable decodedSilenceRunStartSample = -1
        let mutable maxDecodedSilenceRunSamples = 0
        let mutable decodedSilenceStallStartSample = -1
        let mutable decodedSilenceStallSampleCount = 0
        let mutable state = prefill.State
        let mutable stateIsLive = true
        let disposeCurrentState () =
            if stateIsLive then
                stateIsLive <- false
                (state :> IDisposable).Dispose()

        let sampleRms (samples: float32 array) start count =
            if count <= 0 then
                0.0
            else
                let mutable sumSquares = 0.0
                let stop = start + count - 1
                for index in start .. stop do
                    let value = float samples[index]
                    sumSquares <- sumSquares + (value * value)
                Math.Sqrt(sumSquares / float count)

        let noteDecodedSilenceStall startSample sampleCount =
            if decodedSilenceStallStartSample < 0 then
                decodedSilenceStallStartSample <- startSample
                decodedSilenceStallSampleCount <- sampleCount
            true

        let updateDecodedSilenceGuard startSample (samples: float32 array) =
            if not decodedSilenceGuardEnabled || samples.Length = 0 || decodedSilenceStallStartSample >= 0 then
                false
            else
                let rms = sampleRms samples 0 samples.Length
                if rms < decodedSilenceRmsThreshold then
                    if hasAudibleDecodedSamples then
                        if decodedSilenceRunSamples = 0 then
                            decodedSilenceRunStartSample <- startSample
                        decodedSilenceRunSamples <- decodedSilenceRunSamples + samples.Length
                        maxDecodedSilenceRunSamples <- max maxDecodedSilenceRunSamples decodedSilenceRunSamples
                        if decodedSilenceRunSamples >= decodedSilenceGuardSamples then
                            noteDecodedSilenceStall decodedSilenceRunStartSample decodedSilenceRunSamples
                        else
                            false
                    else
                        false
                else
                    hasAudibleDecodedSamples <- true
                    decodedSilenceRunSamples <- 0
                    decodedSilenceRunStartSample <- -1
                    false

        let findDecodedSilenceStall (samples: float32 array) =
            if not decodedSilenceGuardEnabled || samples.Length = 0 then
                None
            else
                let mutable found: (int * int) option = None
                let mutable localHasAudible = false
                let mutable localRunSamples = 0
                let mutable localRunStart = -1
                let mutable index = 0
                while Option.isNone found && index < samples.Length do
                    let count = min outputSamplesPerFrame (samples.Length - index)
                    let rms = sampleRms samples index count
                    if rms < decodedSilenceRmsThreshold then
                        if localHasAudible then
                            if localRunSamples = 0 then
                                localRunStart <- index
                            localRunSamples <- localRunSamples + count
                            if localRunSamples >= decodedSilenceGuardSamples then
                                found <- Some(localRunStart, localRunSamples)
                    else
                        localHasAudible <- true
                        localRunSamples <- 0
                        localRunStart <- -1
                    index <- index + count
                found

        let emitFrame stepKind (frame: DenseTensor<int64>) =
            onFrame
                { FrameIndex = frameCount frames - 1
                  StepKind = stepKind
                  IsEos = frameIsEos frame
                  Codes = frameToArray frame }

        let emitDecodedChunk force =
            cancellationToken.ThrowIfCancellationRequested()
            let currentFrameCount = frameCount frames
            let chunkIsDue = currentFrameCount > 0 && (force || currentFrameCount >= nextStreamDecodeFrame)
            let mutable silenceStalled = false
            if chunkIsDue then
                let shouldEmitChunk = force || shouldDecodeChunk currentFrameCount
                if not shouldEmitChunk then
                    streamDecodeSkips <- streamDecodeSkips + 1

                if shouldEmitChunk || decodedSilenceGuardEnabled then
                    let codes = framesToCodecTensor frames
                    let decodeWatch = Stopwatch.StartNew()
                    let audio, audioOwner = runCodecDecode codes
                    decodeWatch.Stop()
                    decodeMs <- decodeMs + decodeWatch.Elapsed.TotalMilliseconds
                    try
                        let guardSamples = audioSamplesFrom analyzedSampleStart audio
                        if guardSamples.Length > 0 then
                            silenceStalled <- updateDecodedSilenceGuard analyzedSampleStart guardSamples
                            analyzedSampleStart <- analyzedSampleStart + guardSamples.Length

                        if shouldEmitChunk then
                            let samples = audioSamplesFrom emittedSampleStart audio
                            if samples.Length > 0 then
                                onAudioChunk
                                    { ChunkIndex = chunkIndex
                                      StartFrame = emittedFrameStart
                                      FrameCount = currentFrameCount - emittedFrameStart
                                      StartSample = emittedSampleStart
                                      SampleRate = outputSampleRate
                                      Samples = samples }
                                chunkIndex <- chunkIndex + 1
                                emittedSampleStart <- emittedSampleStart + samples.Length
                            emittedFrameStart <- currentFrameCount
                    finally
                        audioOwner.Dispose()
                nextStreamDecodeFrame <- currentFrameCount + streamDecodeFrames
            silenceStalled

        try
            let mutable current =
                try
                    audioFrame samplingOptions prefill.Logits prefill.HiddenStates
                finally
                    (prefill :> IDisposable).Dispose()

            appendFrame frames current
            stepKinds.Add("prefill")
            emitFrame "prefill" current
            emitDecodedChunk false |> ignore
            let mutable currentInput = frameToInputIds current
            let mutable stopReason = if frameIsEos current then "eos" else "max_frames"
            let mutable useThinker = false

            while frameCount frames < maxNewFrames && stopReason = "max_frames" do
                cancellationToken.ThrowIfCancellationRequested()
                let mutable stepKind = "frame"
                let step =
                    if useThinker && tensorContainsInt64 0L state.ThinkerEos then
                        useThinker <- false
                        stepKind <- "thinker"
                        runBackboneThinkerStep currentInput state
                    else
                        useThinker <- tensorContainsInt64 0L state.ThinkerEos
                        runBackboneFrameStep true currentInput state

                let previousState = state
                state <- step.State
                stateIsLive <- true
                try
                    (previousState :> IDisposable).Dispose()
                    current <- audioFrame samplingOptions step.Logits step.HiddenStates
                finally
                    (step :> IDisposable).Dispose()

                if frameHasAnyNonZero current then
                    appendFrame frames current
                    stepKinds.Add(stepKind)
                    emitFrame stepKind current
                    if emitDecodedChunk false then
                        stopReason <- "codec_stall"
                currentInput <- frameToInputIds current
                if frameIsEos current then
                    stopReason <- "eos"

            timings["generateMs"] <- stopwatch.Elapsed.TotalMilliseconds - timings["prefillMs"]
            let codes = framesToCodecTensor frames
            disposeCurrentState ()
            cancellationToken.ThrowIfCancellationRequested()
            let decodeWatch = Stopwatch.StartNew()
            let audio, audioOwner = runCodecDecode codes
            decodeWatch.Stop()
            try
                decodeMs <- decodeMs + decodeWatch.Elapsed.TotalMilliseconds
                let mutable resultCodes = codes
                let mutable resultAudio = audio
                let mutable resultAudioOwners: IDisposable array = [| audioOwner |]
                let mutable resultFrameCount = frameCount frames
                let totalDecodedSamples = tensorElementCount (audio.Dimensions.ToArray())

                if stopReason <> "eos" && decodedSilenceGuardEnabled then
                    let silenceStall =
                        if decodedSilenceStallStartSample >= 0 then
                            Some(decodedSilenceStallStartSample, max decodedSilenceStallSampleCount decodedSilenceGuardSamples)
                        else
                            audioSamplesFrom 0 audio
                            |> findDecodedSilenceStall

                    match silenceStall with
                    | Some(startSample, sampleCount) ->
                        noteDecodedSilenceStall startSample sampleCount |> ignore
                        stopReason <- "codec_stall"
                        let keepSamples = min totalDecodedSamples (startSample + decodedSilenceGuardSamples)
                        let keepFrames =
                            max 1 (min resultFrameCount ((keepSamples + outputSamplesPerFrame - 1) / outputSamplesPerFrame))
                        let keepSamples = min totalDecodedSamples (keepFrames * outputSamplesPerFrame)
                        if keepFrames < resultFrameCount || keepSamples < totalDecodedSamples then
                            resultCodes <- framesToCodecTensorWithCount frames keepFrames
                            resultAudio <- audioPrefix keepSamples audio
                            resultFrameCount <- keepFrames
                            resultAudioOwners <- Array.empty
                            audioOwner.Dispose()
                            emittedSampleStart <- min emittedSampleStart keepSamples
                            emittedFrameStart <- min emittedFrameStart keepFrames
                            timings["decodedSilenceTrimmedFrames"] <- float (frameCount frames - keepFrames)
                            timings["decodedSilenceTrimmedSeconds"] <- float (totalDecodedSamples - keepSamples) / float outputSampleRate
                    | None -> ()

                let finalSamples = audioSamplesFrom emittedSampleStart resultAudio
                if finalSamples.Length > 0 then
                    onAudioChunk
                        { ChunkIndex = chunkIndex
                          StartFrame = emittedFrameStart
                          FrameCount = resultFrameCount - emittedFrameStart
                          StartSample = emittedSampleStart
                          SampleRate = outputSampleRate
                          Samples = finalSamples }
                    emittedSampleStart <- emittedSampleStart + finalSamples.Length
                    emittedFrameStart <- resultFrameCount
                timings["decodeMs"] <- decodeMs
                timings["streamDecodeSkips"] <- float streamDecodeSkips
                timings["decodedSilenceRmsThreshold"] <- decodedSilenceRmsThreshold
                timings["decodedSilenceGuardSeconds"] <- decodedSilenceGuardSeconds
                timings["decodedSilenceGuardSamples"] <- float decodedSilenceGuardSamples
                timings["maxDecodedSilenceSeconds"] <- float maxDecodedSilenceRunSamples / float outputSampleRate
                if decodedSilenceStallStartSample >= 0 then
                    timings["decodedSilenceStallStartSample"] <- float decodedSilenceStallStartSample
                    timings["decodedSilenceStallStartFrame"] <- float decodedSilenceStallStartSample / float outputSamplesPerFrame
                    timings["decodedSilenceStallSeconds"] <- float decodedSilenceStallSampleCount / float outputSampleRate
                timings["totalMs"] <- stopwatch.Elapsed.TotalMilliseconds
                let resultStepKindCount = min resultFrameCount stepKinds.Count
                let resultStepKinds = Array.init resultStepKindCount (fun index -> stepKinds[index])

                { AudioCodes = resultCodes
                  AudioValues = resultAudio
                  FrameCount = resultFrameCount
                  StopReason = stopReason
                  StepKinds = resultStepKinds
                  Timings = timings
                  OwnedBuffers = resultAudioOwners }
            with
            | _ ->
                audioOwner.Dispose()
                reraise()
        with
        | _ ->
            disposeCurrentState ()
            reraise()

    interface IDisposable with
        member _.Dispose() =
            if mergedSession.IsValueCreated then
                disposeSessionPair mergedSession.Value
            disposeWarmSessions()
            prepackedWeights.Dispose()
            registry |> Option.iter (fun value -> (value :> IDisposable).Dispose())
            store |> Option.iter (fun value -> (value :> IDisposable).Dispose())

