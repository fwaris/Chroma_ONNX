namespace ChromaOnnx

open System
open System.Collections.Concurrent
open System.Collections.Generic
open System.Diagnostics
open System.IO
open System.Linq
open System.Text.Json
open Microsoft.ML.OnnxRuntime
open Microsoft.ML.OnnxRuntime.Tensors

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

    let manifest =
        if File.Exists manifestPath then
            Some(SharedWeights.loadManifest bundleDir)
        else
            None

    let normalizedExecutionProvider = OrtExecutionProvider.normalize executionProvider

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

    let tensorRtEngineCacheDir =
        if normalizedExecutionProvider = "tensorrt" then
            Some(OrtExecutionProvider.tensorRtEngineCacheDir optimizedModelCacheDir)
        else
            None

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
        let extension =
            if normalizedOptimizedModelCacheFormat = "ort" then
                "optimized.ort"
            else
                "optimized.onnx"
        $"{graphName}.{normalizedExecutionProvider}.{normalizedOrtMemoryProfile}.{extension}"

    let optimizedModelCachePath graphName =
        if normalizedExecutionProvider = "tensorrt" then
            None
        else
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

    let useExistingOrtCacheWithoutRegistry =
        useMergedSession
        && normalizedOptimizedModelCacheFormat = "ort"
        && (existingOptimizedModelCachePath mergedGraphName).IsSome

    let store =
        if useExistingOrtCacheWithoutRegistry then
            None
        else
            manifest
            |> Option.map (fun value -> new SafetensorWeightStore(modelDir, value.Initializers))

    let registry =
        match manifest, store with
        | Some value, Some weightStore -> Some(new SharedInitializerRegistry(weightStore, value.Initializers))
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
        OrtExecutionProvider.appendToSessionOptions options normalizedExecutionProvider optimizedModelCacheDir useQualitySafeOrtMemoryProfile
        |> ignore
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

    let resolveGraphPath (path: string) =
        if File.Exists path then
            path
        else
            match Path.GetFileName(path) with
            | null -> path
            | fileName when String.IsNullOrWhiteSpace fileName -> path
            | fileName ->
                let localPath = Path.Combine(bundleDir, fileName)
                if File.Exists localPath then localPath else path

    let sourceGraphPath graphName path =
        let resolvedPath = resolveGraphPath path
        let localExternalName =
            if graphName = mergedGraphName then
                "chroma_s2s_merged.local_external.onnx"
            else
                $"{Path.GetFileNameWithoutExtension(resolvedPath)}.local_external.onnx"
        optimizedModelCacheDir
        |> Option.map (fun dir -> Path.Combine(dir, localExternalName))
        |> Option.filter File.Exists
        |> Option.defaultValue resolvedPath

    let createSession graphName =
        match manifest with
        | None -> invalidOp $"S2S bundle manifest was not found: {manifestPath}"
        | Some value ->
            match value.Graphs.TryGetValue(graphName) with
            | false, _ -> invalidArg (nameof graphName) $"S2S bundle does not contain graph {graphName}."
            | true, graph ->
                let graphPath = sourceGraphPath graphName graph.Path
                if not (File.Exists graphPath) then
                    invalidArg graphName $"S2S ONNX graph was not found: {graph.Path}"

                let options, sessionPath = createOptions graphName graphPath
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
                new InferenceSession(sessionPath, options, prepackedWeights), options

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
            let _ = mergedSession.Value
            observeProcessMemory()

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

    let cloneInt64ByName graphName (results: IDisposableReadOnlyCollection<DisposableNamedOnnxValue>) name =
        let expectedName = qualifiedName graphName name
        results
        |> Seq.find (fun value -> value.Name = expectedName)
        |> fun value -> value.AsTensor<int64>()
        |> TensorIO.cloneInt64Tensor

    let collectFloatOutputs graphName (results: IDisposableReadOnlyCollection<DisposableNamedOnnxValue>) prefix =
        let values = Dictionary<string, DenseTensor<float32>>(StringComparer.Ordinal)
        for result in results do
            let name = originalName graphName result.Name
            if name.StartsWith(prefix, StringComparison.Ordinal) then
                values[name] <- result.AsTensor<float32>() |> TensorIO.cloneFloatTensor
        values

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

    let decoderAttentionMask (batch: int) (sequenceLength: int) =
        DenseTensor<int64>(Array.create (batch * sequenceLength) 1L, [| batch; sequenceLength |])

    let decoderCachePosition (start: int) (length: int) =
        DenseTensor<int64>(Array.init length (fun index -> int64 (start + index)), [| length |])

    let runDecoderPrefill (inputIds: DenseTensor<int64>) (backboneLastHiddenState: DenseTensor<float32>) =
        let graphName = "decoder_prefill"
        let batch = inputIds.Dimensions[0]
        let sequenceLength = inputIds.Dimensions[1]
        let attentionMask = decoderAttentionMask batch (sequenceLength + 1)
        let cachePosition = decoderCachePosition 0 sequenceLength
        let inputs = List<NamedOnnxValue>()
        inputs.Add(createInput graphName "input_ids" inputIds)
        inputs.Add(createInput graphName "backbone_last_hidden_state" backboneLastHiddenState)
        inputs.Add(createInput graphName "attention_mask" attentionMask)
        inputs.Add(createInput graphName "cache_position" cachePosition)
        runGraph graphName inputs (fun results ->
            cloneFloatByName graphName results "logits", collectFloatOutputs graphName results "decoder_present_")

    let runDecoderStep (inputIds: DenseTensor<int64>) (cache: Dictionary<string, DenseTensor<float32>>) =
        let graphName = "decoder_step"
        let batch = inputIds.Dimensions[0]
        let sequenceLength = inputIds.Dimensions[1]
        let pastSequenceLength = decoderCacheSequenceLength cache
        let attentionMask = decoderAttentionMask batch (pastSequenceLength + sequenceLength)
        let cachePosition = decoderCachePosition (pastSequenceLength - sequenceLength) sequenceLength
        let inputs = List<NamedOnnxValue>()
        inputs.Add(createInput graphName "input_ids" inputIds)
        inputs.Add(createInput graphName "attention_mask" attentionMask)
        inputs.Add(createInput graphName "cache_position" cachePosition)
        addDecoderCacheInputs graphName inputs cache
        runGraph graphName inputs (fun results ->
            cloneFloatByName graphName results "logits", collectFloatOutputs graphName results "decoder_present_")

    let greedyAudioFrame (logits: DenseTensor<float32>) (hiddenStates: DenseTensor<float32>) =
        let firstIds = TensorMath.argmaxLast logits
        let hidden = TensorMath.lastHidden hiddenStates
        let mutable ids = DenseTensor<int64>(firstIds, [| firstIds.Length; 1 |])
        let mutable decoderLogits, decoderCache = runDecoderPrefill ids hidden
        let mutable nextIds = TensorMath.argmaxLast decoderLogits
        ids <- TensorMath.appendColumn ids nextIds

        for _ in 3 .. audioNumCodebooks do
            let stepInput = DenseTensor<int64>(nextIds, [| nextIds.Length; 1 |])
            let stepLogits, stepCache = runDecoderStep stepInput decoderCache
            decoderLogits <- stepLogits
            decoderCache <- stepCache
            nextIds <- TensorMath.argmaxLast decoderLogits
            ids <- TensorMath.appendColumn ids nextIds

        ids

    let frameToInputIds (frame: DenseTensor<int64>) =
        let data = Enumerable.ToArray(frame)
        DenseTensor<int64>(data, [| 1; 1; audioNumCodebooks |])

    let appendFrame (frames: ResizeArray<int64 array>) (frame: DenseTensor<int64>) =
        frames.Add(Enumerable.ToArray(frame))

    let frameIsEos (frame: DenseTensor<int64>) =
        let values = Enumerable.ToArray(frame)
        let checkedCount = max 1 (audioNumCodebooks - 1)
        values.Length >= checkedCount
        && values |> Seq.take checkedCount |> Seq.forall ((=) 0L)

    let framesToCodecTensor (frames: ResizeArray<int64 array>) =
        let frameCount = frames.Count
        let values = Array.zeroCreate<int64> (audioNumCodebooks * frameCount)
        for frameIndex in 0 .. frameCount - 1 do
            let frame = frames[frameIndex]
            for codebookIndex in 0 .. audioNumCodebooks - 1 do
                values[codebookIndex * frameCount + frameIndex] <- frame[codebookIndex]
        DenseTensor<int64>(values, [| 1; audioNumCodebooks; frameCount |])

    let runCodecDecode (audioCodes: DenseTensor<int64>) =
        let graphName = "codec_decode"
        let inputs = List<NamedOnnxValue>()
        inputs.Add(createInput graphName "audio_codes" audioCodes)
        runGraph graphName inputs (fun results -> cloneFloatByName graphName results "audio_values")

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

            { Logits = cloneFloatByName graphName results "logits"
              HiddenStates = cloneFloatByName graphName results "hidden_states"
              State =
                { AttentionMask = cloneFloatByName graphName results "next_attention_mask"
                  ThinkerInputIds = cloneInt64ByName graphName results "next_thinker_input_ids"
                  ThinkerAttentionMask = cloneInt64ByName graphName results "next_thinker_attention_mask"
                  ThinkerCachePosition = cloneInt64ByName graphName results "next_thinker_cache_position"
                  ThinkerEos = cloneInt64ByName graphName results "next_thinker_eos"
                  BackboneCache = collectFloatOutputs graphName results "backbone_present_"
                  ThinkerCache = collectFloatOutputs graphName results "thinker_present_" } })

    let runBackboneFrameStep (frameCodes: DenseTensor<int64>) (state: S2sGraphState) =
        let graphName = "backbone_frame_step"
        let inputs = List<NamedOnnxValue>()
        inputs.Add(createInput graphName "frame_codes" frameCodes)
        inputs.Add(createInput graphName "attention_mask" state.AttentionMask)
        addBackboneCacheInputs graphName inputs state.BackboneCache
        runGraph graphName inputs (fun results ->

            { Logits = cloneFloatByName graphName results "logits"
              HiddenStates = cloneFloatByName graphName results "hidden_states"
              State =
                { state with
                    AttentionMask = cloneFloatByName graphName results "next_attention_mask"
                    BackboneCache = collectFloatOutputs graphName results "backbone_present_" } })

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

            { Logits = cloneFloatByName graphName results "logits"
              HiddenStates = cloneFloatByName graphName results "hidden_states"
              State =
                { AttentionMask = cloneFloatByName graphName results "next_attention_mask"
                  ThinkerInputIds = cloneInt64ByName graphName results "next_thinker_input_ids"
                  ThinkerAttentionMask = cloneInt64ByName graphName results "next_thinker_attention_mask"
                  ThinkerCachePosition = cloneInt64ByName graphName results "next_thinker_cache_position"
                  ThinkerEos = cloneInt64ByName graphName results "next_thinker_eos"
                  BackboneCache = collectFloatOutputs graphName results "backbone_present_"
                  ThinkerCache = collectFloatOutputs graphName results "thinker_present_" } })

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
          ExecutionProvider = normalizedExecutionProvider
          Message = message }

    member _.MemoryMode = normalizedMemoryMode

    member _.OrtMemoryProfile = normalizedOrtMemoryProfile

    member _.OptimizedModelCacheDir =
        optimizedModelCacheDir |> Option.toObj

    member _.OptimizedModelCacheEnabled = optimizedModelCacheDir.IsSome

    member _.OptimizedModelCacheFormat = normalizedOptimizedModelCacheFormat

    member _.TensorRtEngineCacheDir =
        tensorRtEngineCacheDir |> Option.toObj

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

    member _.CreateSessionOptions() = fst (createOptions mergedGraphName "")

    member this.WriteDebug(prepared: NativeS2sPrepared, outputDir: string, ?maxFrames: int) =
        this.EnsureReady()
        Directory.CreateDirectory(outputDir) |> ignore
        let infos = Dictionary<string, DebugTensorInfo>(StringComparer.Ordinal)
        let maxDebugFrames = defaultArg maxFrames 8

        let prefill = runGeneratePrefill prepared
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
        let prefillDecoderLogits, prefillDecoderCache = runDecoderPrefill ids hidden
        writeDebugFloat outputDir infos "decoder_cache_step_1_logits" prefillDecoderLogits
        let mutable nextIds = TensorMath.argmaxLast prefillDecoderLogits
        writeDebugInt64Array outputDir infos "decoder_cache_step_1_next_ids" nextIds [| nextIds.Length; 1 |]
        ids <- TensorMath.appendColumn ids nextIds
        let mutable decoderCache = prefillDecoderCache

        for stepIndex in 2 .. audioNumCodebooks - 1 do
            let stepInput = DenseTensor<int64>(nextIds, [| nextIds.Length; 1 |])
            writeDebugInt64 outputDir infos $"decoder_cache_step_{stepIndex}_input_ids" stepInput
            let decoderLogits, nextCache = runDecoderStep stepInput decoderCache
            writeDebugFloat outputDir infos $"decoder_cache_step_{stepIndex}_logits" decoderLogits
            nextIds <- TensorMath.argmaxLast decoderLogits
            writeDebugInt64Array outputDir infos $"decoder_cache_step_{stepIndex}_next_ids" nextIds [| nextIds.Length; 1 |]
            ids <- TensorMath.appendColumn ids nextIds
            decoderCache <- nextCache

        writeDebugInt64 outputDir infos "decoder_cache_frame_ids" ids
        writeDebugInt64 outputDir infos "decoder_generate_frame_ids" ids
        writeDebugInt64 outputDir infos "decoder_loop_frame_ids" ids

        let frameInput = frameToInputIds ids
        let frameStep = runBackboneFrameStep frameInput prefill.State
        writeDebugInt64 outputDir infos "backbone_frame_step_input_ids" frameInput
        writeDebugFloat outputDir infos "backbone_frame_step_logits" frameStep.Logits
        writeDebugFloat outputDir infos "backbone_frame_step_hidden_states" frameStep.HiddenStates
        let nextCodebook0 = TensorMath.argmaxLast frameStep.Logits
        writeDebugInt64Array outputDir infos "backbone_frame_step_codebook0_ids" nextCodebook0 [| nextCodebook0.Length; 1 |]

        let mutable traceCurrent = ids
        let mutable traceState = prefill.State
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
            let thinkerActive =
                traceState.ThinkerEos
                |> Enumerable.ToArray
                |> Array.exists ((=) 0L)
            let stepKind, traceStep =
                if traceUseThinker && thinkerActive then
                    traceUseThinker <- false
                    "thinker", runBackboneThinkerStep traceInput traceState
                else
                    traceUseThinker <- thinkerActive
                    "frame", runBackboneFrameStep traceInput traceState

            writeDebugInt64Array outputDir infos $"trace_step_{frameIndex}_kind" [| if stepKind = "thinker" then 1L else 0L |] [| 1 |]
            writeDebugFloat outputDir infos $"trace_step_{frameIndex}_logits" traceStep.Logits
            writeDebugFloat outputDir infos $"trace_step_{frameIndex}_hidden_states" traceStep.HiddenStates
            traceState <- traceStep.State
            traceCurrent <- greedyAudioFrame traceStep.Logits traceStep.HiddenStates
            traceInput <- frameToInputIds traceCurrent
            writeDebugInt64 outputDir infos $"trace_frame_{frameIndex}_ids" traceCurrent
            writeDebugFloat outputDir infos $"trace_state_{frameIndex}_attention_mask" traceState.AttentionMask
            writeDebugInt64 outputDir infos $"trace_state_{frameIndex}_thinker_input_ids" traceState.ThinkerInputIds
            writeDebugInt64 outputDir infos $"trace_state_{frameIndex}_thinker_attention_mask" traceState.ThinkerAttentionMask
            writeDebugInt64 outputDir infos $"trace_state_{frameIndex}_thinker_cache_position" traceState.ThinkerCachePosition
            writeDebugInt64 outputDir infos $"trace_state_{frameIndex}_thinker_eos" traceState.ThinkerEos

        let manifestJson =
            JsonSerializer.Serialize(
                {| runtime = "fsharp_onnx"
                   executionProvider = this.Status.ExecutionProvider
                   tensors = infos |},
                debugJsonOptions
            )
        File.WriteAllText(Path.Combine(outputDir, "debug_manifest.json"), manifestJson)

    member this.Generate(prepared: NativeS2sPrepared, maxNewFrames: int) =
        this.EnsureReady()
        if maxNewFrames < 1 then
            invalidArg (nameof maxNewFrames) "maxNewFrames must be positive."

        let timings = Dictionary<string, float>(StringComparer.Ordinal)
        let stopwatch = Stopwatch.StartNew()
        let prefill = runGeneratePrefill prepared
        timings["prefillMs"] <- stopwatch.Elapsed.TotalMilliseconds

        let frames = ResizeArray<int64 array>()
        let stepKinds = ResizeArray<string>()
        let mutable current = greedyAudioFrame prefill.Logits prefill.HiddenStates
        appendFrame frames current
        stepKinds.Add("prefill")
        let mutable state = prefill.State
        let mutable currentInput = frameToInputIds current
        let mutable stopReason = if frameIsEos current then "eos" else "max_frames"
        let mutable useThinker = false

        while frames.Count < maxNewFrames && stopReason <> "eos" do
            let mutable stepKind = "frame"
            let step =
                if useThinker && state.ThinkerEos |> Enumerable.ToArray |> Array.exists ((=) 0L) then
                    useThinker <- false
                    stepKind <- "thinker"
                    runBackboneThinkerStep currentInput state
                else
                    useThinker <- state.ThinkerEos |> Enumerable.ToArray |> Array.exists ((=) 0L)
                    runBackboneFrameStep currentInput state

            state <- step.State
            current <- greedyAudioFrame step.Logits step.HiddenStates
            if not (current |> Enumerable.ToArray |> Array.forall ((=) 0L)) then
                appendFrame frames current
                stepKinds.Add(stepKind)
            currentInput <- frameToInputIds current
            if frameIsEos current then
                stopReason <- "eos"

        timings["generateMs"] <- stopwatch.Elapsed.TotalMilliseconds - timings["prefillMs"]
        let codes = framesToCodecTensor frames
        let decodeWatch = Stopwatch.StartNew()
        let audio = runCodecDecode codes
        timings["decodeMs"] <- decodeWatch.Elapsed.TotalMilliseconds
        timings["totalMs"] <- stopwatch.Elapsed.TotalMilliseconds

        { AudioCodes = codes
          AudioValues = audio
          FrameCount = frames.Count
          StopReason = stopReason
          StepKinds = stepKinds.ToArray()
          Timings = timings }

    interface IDisposable with
        member _.Dispose() =
            if mergedSession.IsValueCreated then
                disposeSessionPair mergedSession.Value
            disposeWarmSessions()
            prepackedWeights.Dispose()
            registry |> Option.iter (fun value -> (value :> IDisposable).Dispose())
            store |> Option.iter (fun value -> (value :> IDisposable).Dispose())

