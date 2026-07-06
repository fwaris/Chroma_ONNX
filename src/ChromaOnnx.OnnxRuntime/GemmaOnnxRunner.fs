namespace ChromaOnnx

open System
open System.Collections.Generic
open System.Diagnostics
open System.IO
open System.Linq
open System.Threading
open System.Threading.Tasks
open Microsoft.ML.OnnxRuntime
open Microsoft.ML.OnnxRuntime.Tensors

type private GemmaModelLayout =
    | TransformersJs
    | MobiusGenAi

type private GemmaLoadedSessions =
    { EmbedTokens: InferenceSession
      AudioEncoder: InferenceSession option
      Decoder: InferenceSession
      Options: SessionOptions array
      Layout: GemmaModelLayout }
    interface IDisposable with
        member this.Dispose() =
            this.EmbedTokens.Dispose()
            this.AudioEncoder |> Option.iter _.Dispose()
            this.Decoder.Dispose()
            this.Options |> Array.iter _.Dispose()

type private GemmaOrtTensor =
    | TensorFloat of DenseTensor<float32>
    | TensorHalf of DenseTensor<Half>
    | TensorOrtFloat16 of DenseTensor<Float16>

type GemmaOnnxRunner(modelDir: string, variant: string, executionProvider: string, ?maxAudioSeconds: float) =
    let normalizedVariant =
        if String.IsNullOrWhiteSpace variant then "Q4_K_M/cuda" else variant.Trim()
    let variantParts =
        normalizedVariant.Split([| '/'; '\\' |], StringSplitOptions.RemoveEmptyEntries)
    let looksLikeMobiusVariant =
        variantParts.Length >= 2
        || normalizedVariant.Equals("mobius", StringComparison.OrdinalIgnoreCase)
        || normalizedVariant.Contains("Q4", StringComparison.OrdinalIgnoreCase)
        || normalizedVariant.Contains("NF4", StringComparison.OrdinalIgnoreCase)
    let combineParts (root: string) (parts: string array) =
        parts |> Array.fold (fun current part -> Path.Combine(current, part)) root
    let directMobiusDir =
        File.Exists(Path.Combine(modelDir, "genai_config.json"))
        || Directory.Exists(Path.Combine(modelDir, "embedding"))
        || Directory.Exists(Path.Combine(modelDir, "decoder"))
    let variantModelDir =
        if looksLikeMobiusVariant && variantParts.Length > 0 then
            combineParts modelDir variantParts
        else
            modelDir
    let effectiveModelDir =
        if directMobiusDir then
            modelDir
        elif looksLikeMobiusVariant then
            variantModelDir
        else
            modelDir
    let layout =
        if directMobiusDir
           || looksLikeMobiusVariant
           || File.Exists(Path.Combine(effectiveModelDir, "genai_config.json")) then
            MobiusGenAi
        else
            TransformersJs
    let processor = GemmaProcessor(effectiveModelDir, ?maxAudioSeconds = maxAudioSeconds)
    let syncRoot = obj ()
    let mutable loaded: GemmaLoadedSessions option = None

    let graphPath name =
        match layout with
        | MobiusGenAi ->
            match name with
            | "embed_tokens" -> Path.Combine(effectiveModelDir, "embedding", "model.onnx")
            | "audio_encoder" -> Path.Combine(effectiveModelDir, "audio_encoder", "model.onnx")
            | "decoder_model_merged" -> Path.Combine(effectiveModelDir, "decoder", "model.onnx")
            | _ -> invalidArg (nameof name) $"Unsupported Mobius Gemma graph '{name}'."
        | TransformersJs ->
            Path.Combine(effectiveModelDir, "onnx", $"{name}_{normalizedVariant}.onnx")

    let requiredFiles =
        match layout with
        | MobiusGenAi ->
            [| Path.Combine(effectiveModelDir, "genai_config.json")
               Path.Combine(effectiveModelDir, "tokenizer.json")
               Path.Combine(effectiveModelDir, "tokenizer_config.json")
               Path.Combine(effectiveModelDir, "chat_template.jinja")
               Path.Combine(effectiveModelDir, "audio_feature_extraction.json")
               graphPath "embed_tokens"
               graphPath "audio_encoder"
               graphPath "decoder_model_merged" |]
        | TransformersJs ->
            [| Path.Combine(effectiveModelDir, "config.json")
               Path.Combine(effectiveModelDir, "generation_config.json")
               Path.Combine(effectiveModelDir, "tokenizer.json")
               Path.Combine(effectiveModelDir, "tokenizer_config.json")
               Path.Combine(effectiveModelDir, "processor_config.json")
               Path.Combine(effectiveModelDir, "chat_template.jinja")
               graphPath "embed_tokens"
               graphPath "audio_encoder"
               graphPath "decoder_model_merged" |]

    let optionalAudioFile = graphPath "audio_encoder"

    let missingFiles () =
        requiredFiles |> Array.filter (File.Exists >> not)

    let createOptions () =
        let options = new SessionOptions()
        options.GraphOptimizationLevel <- GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        options.LogSeverityLevel <- OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR
        match executionProvider.Trim().ToLowerInvariant() with
        | "" | "cuda" ->
            let cudaOptions = new OrtCUDAProviderOptions()
            try
                cudaOptions.UpdateOptions(
                    Dictionary<string, string>(
                        dict [ "device_id", "0"
                               "enable_skip_layer_norm_strict_mode", "1"
                               "enable_cuda_graph", "0" ]
                    )
                )
                options.AppendExecutionProvider_CUDA(cudaOptions)
            finally
                cudaOptions.Dispose()
        | "cpu" -> options.AppendExecutionProvider_CPU(0)
        | value -> invalidArg (nameof executionProvider) $"Unsupported Gemma execution provider '{value}'. Use cuda or cpu."
        options

    let loadSessions () =
        match loaded with
        | Some sessions -> sessions
        | None ->
            lock syncRoot (fun () ->
                match loaded with
                | Some sessions -> sessions
                | None ->
                    let missing = missingFiles ()
                    if missing.Length > 0 then
                        let missingText = String.Join(", ", missing)
                        invalidOp $"Gemma ONNX model is not ready. Missing files: {missingText}"
                    let embedOptions = createOptions ()
                    let decoderOptions = createOptions ()
                    let audioOptions =
                        if File.Exists optionalAudioFile then Some(createOptions ()) else None
                    try
                        let sessions =
                            { EmbedTokens = new InferenceSession(graphPath "embed_tokens", embedOptions)
                              AudioEncoder =
                                audioOptions
                                |> Option.map (fun options -> new InferenceSession(optionalAudioFile, options))
                              Decoder = new InferenceSession(graphPath "decoder_model_merged", decoderOptions)
                              Options =
                                [| yield embedOptions
                                   yield decoderOptions
                                   match audioOptions with
                                   | Some options -> yield options
                                   | None -> () |]
                              Layout = layout }
                        loaded <- Some sessions
                        sessions
                    with
                    | _ ->
                        embedOptions.Dispose()
                        decoderOptions.Dispose()
                        audioOptions |> Option.iter _.Dispose()
                        reraise())

    let inputNames (session: InferenceSession) =
        session.InputMetadata.Keys |> Seq.toArray

    let outputNameOrDefault (session: InferenceSession) preferred fallback =
        session.OutputMetadata.Keys
        |> Seq.tryFind (fun name -> name.Equals(preferred, StringComparison.Ordinal))
        |> Option.defaultWith (fun () ->
            session.OutputMetadata.Keys
            |> Seq.tryHead
            |> Option.defaultValue fallback)

    let createBoolTensor value =
        DenseTensor<bool>(Memory<bool>([| value |]), ReadOnlySpan<int>([| 1 |]), false)

    let createScalarInt64 value =
        DenseTensor<int64>(Memory<int64>([| value |]), ReadOnlySpan<int>(Array.empty<int>), false)

    let positionIds (attentionMask: int64 array) takeLast =
        let values = Array.zeroCreate<int64> attentionMask.Length
        let mutable sum = 0L
        for index in 0 .. attentionMask.Length - 1 do
            if attentionMask[index] = 0L then
                values[index] <- 1L
            else
                values[index] <- sum
                sum <- sum + attentionMask[index]
        let finalValues =
            if takeLast > 0 && takeLast < values.Length then
                values[values.Length - takeLast ..]
            else
                values
        DenseTensor<int64>(finalValues, [| 1; finalValues.Length |])

    let metadataDims (metadata: NodeMetadata) batchSize =
        metadata.Dimensions
        |> Seq.map (fun dim ->
            if dim > 0 then dim
            elif dim = -1 then 0
            else 0)
        |> Seq.toArray
        |> Array.mapi (fun index value -> if index = 0 && value = 0 then batchSize else value)

    let tensorCount dims =
        dims |> Array.fold (fun total dim -> total * dim) 1

    let toFloat16 (value: float32) : Float16 =
        Float16.op_Explicit value

    let zeroTensorForMetadata name (metadata: NodeMetadata) batchSize =
        let dims = metadataDims metadata batchSize
        let count = tensorCount dims
        if metadata.ElementType = typeof<float32> then
            NamedOnnxValue.CreateFromTensor(name, DenseTensor<float32>(Array.zeroCreate<float32> count, dims))
        elif metadata.ElementType = typeof<Half> then
            NamedOnnxValue.CreateFromTensor(name, DenseTensor<Half>(Memory<Half>(Array.zeroCreate<Half> count), ReadOnlySpan<int>(dims), false))
        elif metadata.ElementType = typeof<Float16> then
            NamedOnnxValue.CreateFromTensor(name, DenseTensor<Float16>(Memory<Float16>(Array.zeroCreate<Float16> count), ReadOnlySpan<int>(dims), false))
        elif metadata.ElementType = typeof<int64> then
            NamedOnnxValue.CreateFromTensor(name, DenseTensor<int64>(Array.zeroCreate<int64> count, dims))
        elif metadata.ElementType = typeof<bool> then
            NamedOnnxValue.CreateFromTensor(name, DenseTensor<bool>(Memory<bool>(Array.zeroCreate<bool> count), ReadOnlySpan<int>(dims), false))
        else
            let other = metadata.ElementType
            invalidOp $"Unsupported Gemma empty cache tensor type {other} for {name}."

    let createEmptyFeatureInput name (metadata: NodeMetadata) =
        let dims =
            let metadataDims = metadata.Dimensions |> Seq.toArray
            let hidden =
                metadataDims
                |> Array.rev
                |> Array.tryFind (fun dim -> dim > 0)
                |> Option.defaultValue 1536
            [| 0; hidden |]
        if metadata.ElementType = typeof<float32> then
            NamedOnnxValue.CreateFromTensor(name, DenseTensor<float32>(Array.empty<float32>, dims))
        elif metadata.ElementType = typeof<Half> then
            NamedOnnxValue.CreateFromTensor(name, DenseTensor<Half>(Memory<Half>(Array.empty<Half>), ReadOnlySpan<int>(dims), false))
        elif metadata.ElementType = typeof<Float16> then
            NamedOnnxValue.CreateFromTensor(name, DenseTensor<Float16>(Memory<Float16>(Array.empty<Float16>), ReadOnlySpan<int>(dims), false))
        else
            invalidOp $"Unsupported Gemma feature tensor type {metadata.ElementType} for {name}."

    let cloneFloatTensor (tensor: Tensor<float32>) =
        DenseTensor<float32>(tensor.ToArray(), tensor.Dimensions.ToArray())

    let cloneOrtTensor (value: DisposableNamedOnnxValue) =
        match value.Value with
        | :? Tensor<float32> as tensor -> TensorFloat(DenseTensor<float32>(tensor.ToArray(), tensor.Dimensions.ToArray()))
        | :? Tensor<Half> as tensor -> TensorHalf(DenseTensor<Half>(tensor.ToArray(), tensor.Dimensions.ToArray()))
        | :? Tensor<Float16> as tensor -> TensorOrtFloat16(DenseTensor<Float16>(Memory<Float16>(tensor.ToArray()), ReadOnlySpan<int>(tensor.Dimensions.ToArray()), false))
        | other -> invalidOp $"Gemma output {value.Name} has unsupported tensor value type {other.GetType().FullName}."

    let readFloatTensor (value: DisposableNamedOnnxValue) =
        match value.Value with
        | :? Tensor<float32> as tensor -> cloneFloatTensor tensor
        | :? Tensor<Half> as tensor ->
            let values = tensor |> Seq.map float32 |> Seq.toArray
            DenseTensor<float32>(values, tensor.Dimensions.ToArray())
        | :? Tensor<Float16> as tensor ->
            let values = tensor |> Seq.map (fun value -> float32 value) |> Seq.toArray
            DenseTensor<float32>(values, tensor.Dimensions.ToArray())
        | other -> invalidOp $"Gemma output {value.Name} has unsupported tensor value type {other.GetType().FullName}."

    let tensorDims tensor =
        match tensor with
        | TensorFloat value -> value.Dimensions.ToArray()
        | TensorHalf value -> value.Dimensions.ToArray()
        | TensorOrtFloat16 value -> value.Dimensions.ToArray()

    let tensorToInput name tensor =
        match tensor with
        | TensorFloat value -> NamedOnnxValue.CreateFromTensor(name, value)
        | TensorHalf value -> NamedOnnxValue.CreateFromTensor(name, value)
        | TensorOrtFloat16 value -> NamedOnnxValue.CreateFromTensor(name, value)

    let tensorToFloat32 tensor =
        match tensor with
        | TensorFloat value -> DenseTensor<float32>(value.ToArray(), value.Dimensions.ToArray())
        | TensorHalf value ->
            DenseTensor<float32>(value |> Seq.map float32 |> Seq.toArray, value.Dimensions.ToArray())
        | TensorOrtFloat16 value ->
            DenseTensor<float32>(value |> Seq.map (fun item -> float32 item) |> Seq.toArray, value.Dimensions.ToArray())

    let squeezeBatch tensor =
        match tensor with
        | TensorFloat value ->
            let dims = value.Dimensions.ToArray()
            if dims.Length = 3 && dims[0] = 1 then
                TensorFloat(DenseTensor<float32>(value.ToArray(), [| dims[1]; dims[2] |]))
            else
                tensor
        | TensorHalf value ->
            let dims = value.Dimensions.ToArray()
            if dims.Length = 3 && dims[0] = 1 then
                TensorHalf(DenseTensor<Half>(value.ToArray(), [| dims[1]; dims[2] |]))
            else
                tensor
        | TensorOrtFloat16 value ->
            let dims = value.Dimensions.ToArray()
            if dims.Length = 3 && dims[0] = 1 then
                TensorOrtFloat16(DenseTensor<Float16>(Memory<Float16>(value.ToArray()), ReadOnlySpan<int>([| dims[1]; dims[2] |]), false))
            else
                tensor

    let createAudioFeatureValue name (metadata: NodeMetadata) (features: GemmaAudioFeatures) =
        let dims = features.InputFeatures.Dimensions.ToArray()
        let values = features.InputFeatures.Buffer.Span.ToArray()
        if metadata.ElementType = typeof<float32> then
            NamedOnnxValue.CreateFromTensor(name, DenseTensor<float32>(values, dims))
        elif metadata.ElementType = typeof<Half> then
            let converted = values |> Array.map Half.op_Explicit
            NamedOnnxValue.CreateFromTensor(name, DenseTensor<Half>(Memory<Half>(converted), ReadOnlySpan<int>(dims), false))
        elif metadata.ElementType = typeof<Float16> then
            let converted = values |> Array.map toFloat16
            NamedOnnxValue.CreateFromTensor(name, DenseTensor<Float16>(Memory<Float16>(converted), ReadOnlySpan<int>(dims), false))
        else
            invalidOp $"Unsupported Gemma audio feature tensor type {metadata.ElementType} for {name}."

    let runAudioEncoder (sessions: GemmaLoadedSessions) (features: GemmaAudioFeatures) =
        match sessions.AudioEncoder with
        | None -> invalidOp "Gemma audio_encoder graph is required for audio turns but was not found."
        | Some session ->
            let inputSet = HashSet<string>(inputNames session, StringComparer.Ordinal)
            let featureName =
                if inputSet.Contains("input_features") then "input_features"
                elif inputSet.Contains("audio_embeds") then "audio_embeds"
                else invalidOp "Gemma audio_encoder has no supported feature input."
            let maskName =
                if inputSet.Contains("input_features_mask") then "input_features_mask"
                elif inputSet.Contains("attention_mask") then "attention_mask"
                else invalidOp "Gemma audio_encoder has no supported mask input."
            use results =
                [ createAudioFeatureValue featureName session.InputMetadata[featureName] features
                  NamedOnnxValue.CreateFromTensor(maskName, features.InputFeaturesMask) ]
                |> session.Run
            let outputName = outputNameOrDefault session "audio_features" "audio_features"
            results |> Seq.find (fun value -> value.Name = outputName) |> cloneOrtTensor

    let runEmbed (sessions: GemmaLoadedSessions) (inputIds: DenseTensor<int64>) (audioFeatures: GemmaOrtTensor option) =
        let feeds = ResizeArray<NamedOnnxValue>()
        feeds.Add(NamedOnnxValue.CreateFromTensor("input_ids", inputIds))
        let inputSet = HashSet<string>(inputNames sessions.EmbedTokens, StringComparer.Ordinal)
        if inputSet.Contains("image_features") then
            feeds.Add(createEmptyFeatureInput "image_features" sessions.EmbedTokens.InputMetadata["image_features"])
        if inputSet.Contains("audio_features") then
            match audioFeatures with
            | Some tensor -> feeds.Add(tensorToInput "audio_features" (squeezeBatch tensor))
            | None -> feeds.Add(createEmptyFeatureInput "audio_features" sessions.EmbedTokens.InputMetadata["audio_features"])
        use results = sessions.EmbedTokens.Run(feeds)
        let inputsEmbedsName = outputNameOrDefault sessions.EmbedTokens "inputs_embeds" "inputs_embeds"
        let perLayerName = outputNameOrDefault sessions.EmbedTokens "per_layer_inputs" "per_layer_inputs"
        let inputsEmbeds =
            results
            |> Seq.find (fun value -> value.Name = inputsEmbedsName)
            |> cloneOrtTensor
        let perLayerInputs =
            results
            |> Seq.tryFind (fun value -> value.Name = perLayerName)
            |> Option.map cloneOrtTensor
        inputsEmbeds, perLayerInputs

    let mergeAudioFeatures (inputIds: DenseTensor<int64>) (audioFeatures: DenseTensor<float32>) (embeds: DenseTensor<float32>) =
        let embedDims = embeds.Dimensions.ToArray()
        if embedDims.Length <> 3 then
            invalidOp "Gemma inputs_embeds must have shape [batch, sequence, hidden]."
        let featureDims = audioFeatures.Dimensions.ToArray()
        let hidden = embedDims[2]
        let featuresFlat =
            match featureDims with
            | dims when dims.Length = 3 && dims[0] = 1 ->
                let count = dims[1] * dims[2]
                let values = Array.zeroCreate<float32> count
                audioFeatures.Buffer.Span.Slice(0, count).CopyTo(values.AsSpan())
                values, dims[1], dims[2]
            | dims when dims.Length = 2 ->
                let count = dims[0] * dims[1]
                let values = Array.zeroCreate<float32> count
                audioFeatures.Buffer.Span.Slice(0, count).CopyTo(values.AsSpan())
                values, dims[0], dims[1]
            | _ ->
                let shapeText = String.Join("x", featureDims)
                invalidOp $"Unsupported Gemma audio_features shape: {shapeText}"
        let values, featureCount, featureHidden = featuresFlat
        if featureHidden <> hidden then
            invalidOp $"Gemma audio feature hidden size {featureHidden} does not match token embedding hidden size {hidden}."
        let mutable featureIndex = 0
        let embedValues = embeds.Buffer.Span
        let inputValues = inputIds.Buffer.Span
        for tokenIndex in 0 .. inputIds.Dimensions[1] - 1 do
            if inputValues[tokenIndex] = processor.AudioTokenId then
                if featureIndex >= featureCount then
                    invalidOp "Gemma prompt contains more audio tokens than audio encoder features."
                let targetOffset = tokenIndex * hidden
                let sourceOffset = featureIndex * hidden
                values.AsSpan(sourceOffset, hidden).CopyTo(embedValues.Slice(targetOffset, hidden))
                featureIndex <- featureIndex + 1
        if featureIndex <> featureCount then
            invalidOp $"Gemma audio token/features mismatch. Tokens: {featureIndex}; features: {featureCount}."

    let presentNameToPast (name: string) =
        if name.StartsWith("present", StringComparison.Ordinal) then
            "past_key_values" + name.Substring("present".Length)
        else
            name

    let updateCache (results: seq<DisposableNamedOnnxValue>) =
        let cache = Dictionary<string, GemmaOrtTensor>(StringComparer.Ordinal)
        for value in results do
            if value.Name.StartsWith("present", StringComparison.Ordinal) then
                cache[presentNameToPast value.Name] <- cloneOrtTensor value
        cache

    let runDecoder
        (sessions: GemmaLoadedSessions)
        (inputIds: DenseTensor<int64> option)
        (inputsEmbeds: GemmaOrtTensor)
        (perLayerInputs: GemmaOrtTensor option)
        (attentionMaskValues: int64 array)
        (hasCache: bool)
        (cache: Dictionary<string, GemmaOrtTensor> option)
        =
        let feeds = ResizeArray<NamedOnnxValue>()
        let decoderInputs = inputNames sessions.Decoder
        let decoderInputSet = HashSet<string>(decoderInputs, StringComparer.Ordinal)
        if decoderInputSet.Contains("input_ids") then
            match inputIds with
            | Some ids -> feeds.Add(NamedOnnxValue.CreateFromTensor("input_ids", ids))
            | None -> ()
        if decoderInputSet.Contains("inputs_embeds") then
            feeds.Add(tensorToInput "inputs_embeds" inputsEmbeds)
        match perLayerInputs with
        | Some tensor when decoderInputSet.Contains("per_layer_inputs") ->
            feeds.Add(tensorToInput "per_layer_inputs" tensor)
        | _ -> ()
        if decoderInputSet.Contains("attention_mask") then
            feeds.Add(NamedOnnxValue.CreateFromTensor("attention_mask", DenseTensor<int64>(attentionMaskValues, [| 1; attentionMaskValues.Length |])))
        if decoderInputSet.Contains("position_ids") then
            let dims = tensorDims inputsEmbeds
            feeds.Add(NamedOnnxValue.CreateFromTensor("position_ids", positionIds attentionMaskValues dims[1]))
        if decoderInputSet.Contains("use_cache_branch") then
            feeds.Add(NamedOnnxValue.CreateFromTensor("use_cache_branch", createBoolTensor hasCache))
        if decoderInputSet.Contains("num_logits_to_keep") then
            feeds.Add(NamedOnnxValue.CreateFromTensor("num_logits_to_keep", createScalarInt64 1L))
        match cache with
        | Some values ->
            for KeyValue(name, value) in values do
                if decoderInputSet.Contains name then
                    feeds.Add(tensorToInput name value)
        | None ->
            for KeyValue(name, metadata) in sessions.Decoder.InputMetadata do
                if name.StartsWith("past_key_values.", StringComparison.Ordinal) then
                    feeds.Add(zeroTensorForMetadata name metadata 1)
        sessions.Decoder.Run(feeds)

    let sampleNextToken rng temperature topP topK (logits: DenseTensor<float32>) =
        if temperature <= 0.0 then
            TensorMath.argmaxLast logits |> Array.head
        else
            TensorMath.sampleLast rng temperature topP topK logits |> Array.head

    let isEos tokenId =
        processor.EosTokenIds |> Array.exists ((=) tokenId)

    let layoutName =
        match layout with
        | MobiusGenAi -> "mobius"
        | TransformersJs -> "transformers-js"

    member _.Processor = processor

    interface IGemmaRuntime with
        member _.Status() =
            let missing = missingFiles ()
            let loadedNames =
                match loaded with
                | None -> Array.empty
                | Some sessions ->
                    [| match sessions.Layout with
                       | MobiusGenAi -> "embedding"
                       | TransformersJs -> "embed_tokens"
                       if sessions.AudioEncoder.IsSome then "audio_encoder"
                       match sessions.Layout with
                       | MobiusGenAi -> "decoder"
                       | TransformersJs -> "decoder_model_merged" |]
            { Ready = missing.Length = 0
              ModelDir = effectiveModelDir
              Variant = normalizedVariant
              ExecutionProvider = executionProvider
              MissingFiles = missing
              LoadedSessions = loadedNames
              Message =
                if missing.Length = 0 then
                    if loadedNames.Length = 0 then $"Gemma {layoutName} model files are present; sessions will load on first use."
                    else $"Gemma {layoutName} sessions loaded."
                else
                    $"Gemma {layoutName} model is missing {missing.Length} required file(s)." }

        member _.Prepare(request: GemmaGenerationRequest) =
            processor.Prepare(request)

        member _.GenerateAsync(request: GemmaGenerationRequest, cancellationToken: CancellationToken) =
            task {
                cancellationToken.ThrowIfCancellationRequested()
                let stopwatch = Stopwatch.StartNew()
                let sessions = loadSessions ()
                let prepared = processor.Prepare(request)
                let promptMs = stopwatch.Elapsed.TotalMilliseconds
                let inputIds = prepared.InputIds
                let mutable attentionMask = prepared.AttentionMask.Buffer.Span.ToArray()
                let embedInputs = HashSet<string>(inputNames sessions.EmbedTokens, StringComparer.Ordinal)
                let audioForEmbedding =
                    match prepared.AudioFeatures with
                    | Some features when embedInputs.Contains("audio_features") ->
                        Some(runAudioEncoder sessions features |> squeezeBatch)
                    | _ -> None
                let initialInputsEmbeds, initialPerLayerInputs = runEmbed sessions inputIds audioForEmbedding
                let mutable inputsEmbeds = initialInputsEmbeds
                let mutable perLayerInputs = initialPerLayerInputs
                match prepared.AudioFeatures, audioForEmbedding with
                | Some features, None ->
                    let encoded = runAudioEncoder sessions features |> tensorToFloat32
                    let embeds = tensorToFloat32 inputsEmbeds
                    mergeAudioFeatures inputIds encoded embeds
                    inputsEmbeds <- TensorFloat embeds
                | _ -> ()
                let rng = Random()
                let generated = ResizeArray<int64>()
                let mutable cache: Dictionary<string, GemmaOrtTensor> option = None
                let mutable stopReason = "max_tokens"
                let mutable step = 0
                let mutable keepGoing = true
                let maxNewTokens = max 1 request.MaxNewTokens
                let mutable nextInputId: int64 option = None

                while keepGoing && step < maxNewTokens do
                    cancellationToken.ThrowIfCancellationRequested()
                    let stepInputIds, stepEmbeds, stepPerLayer =
                        match nextInputId with
                        | None -> Some inputIds, inputsEmbeds, perLayerInputs
                        | Some id ->
                            let ids = DenseTensor<int64>([| id |], [| 1; 1 |])
                            let embeds, perLayer = runEmbed sessions ids None
                            Some ids, embeds, perLayer
                    use results =
                        runDecoder
                            sessions
                            stepInputIds
                            stepEmbeds
                            stepPerLayer
                            attentionMask
                            cache.IsSome
                            cache
                    let logitsName = outputNameOrDefault sessions.Decoder "logits" "logits"
                    let logits =
                        results
                        |> Seq.find (fun value -> value.Name = logitsName)
                        |> readFloatTensor
                    let token = sampleNextToken rng request.Temperature request.TopP request.TopK logits
                    if isEos token then
                        stopReason <- "eos"
                        keepGoing <- false
                    else
                        generated.Add token
                        nextInputId <- Some token
                        attentionMask <- Array.append attentionMask [| 1L |]
                    cache <- Some(updateCache results)
                    step <- step + 1

                stopwatch.Stop()
                let outputIds = generated.ToArray()
                return
                    { Text = processor.Decode outputIds
                      Prompt = prepared.Prompt
                      InputTokenCount = inputIds.Dimensions[1]
                      OutputTokenIds = outputIds
                      StopReason = stopReason
                      TimingsMs =
                        [ "prepareMs", promptMs
                          "totalMs", stopwatch.Elapsed.TotalMilliseconds ]
                        |> Map.ofList }
            }

    interface IDisposable with
        member _.Dispose() =
            lock syncRoot (fun () ->
                loaded |> Option.iter (fun sessions -> (sessions :> IDisposable).Dispose())
                loaded <- None)
