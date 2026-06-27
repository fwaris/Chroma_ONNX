namespace ChromaOnnx

open System
open System.Collections.Generic
open System.IO
open Microsoft.ML.OnnxRuntime
open Microsoft.ML.OnnxRuntime.Tensors

type ChromaOnnxRunner(paths: ChromaPaths) =
    let sessionOptions = new SessionOptions()
    let systemPrefill = lazy (paths.SystemPrefill |> Option.map (fun path -> new InferenceSession(path, sessionOptions)))
    let backbone = lazy (new InferenceSession(paths.Backbone, sessionOptions))
    let decoder = lazy (new InferenceSession(paths.Decoder, sessionOptions))
    let codecDecode = lazy (paths.CodecDecode |> Option.map (fun path -> new InferenceSession(path, sessionOptions)))

    member _.Inspect() =
        let printSession (path: string) =
            use session = new InferenceSession(path, sessionOptions)
            let name = Path.GetFileNameWithoutExtension(path)
            printfn "%s" name
            printfn "  inputs:"
            for item in session.InputMetadata do
                printfn "    %s : %A %A" item.Key item.Value.ElementType item.Value.Dimensions
            printfn "  outputs:"
            for item in session.OutputMetadata do
                printfn "    %s : %A %A" item.Key item.Value.ElementType item.Value.Dimensions

        if paths.GraphFiles.Length = 0 then
            invalidArg paths.OnnxDir "No .onnx files found in the ONNX directory."

        paths.GraphFiles |> Array.iter printSession

    member _.RunSystemPrefill(inputIds: DenseTensor<int64>, attentionMask: DenseTensor<int64>, inputValues: DenseTensor<float32>, inputValuesCutoffs: DenseTensor<int64>) =
        let session =
            systemPrefill.Value
            |> Option.defaultWith (fun () -> invalidArg paths.OnnxDir "chroma_system_prefill.onnx was not found in the ONNX directory.")

        let inputs = new List<NamedOnnxValue>()
        inputs.Add(NamedOnnxValue.CreateFromTensor("input_ids", inputIds))
        inputs.Add(NamedOnnxValue.CreateFromTensor("attention_mask", attentionMask))
        inputs.Add(NamedOnnxValue.CreateFromTensor("input_values", inputValues))
        inputs.Add(NamedOnnxValue.CreateFromTensor("input_values_cutoffs", inputValuesCutoffs))
        use results = session.Run(inputs)

        let logits =
            results
            |> Seq.find (fun value -> value.Name = "logits")
            |> fun value -> value.AsTensor<float32>()
            |> TensorIO.cloneFloatTensor

        let hiddenStates =
            results
            |> Seq.find (fun value -> value.Name = "hidden_states")
            |> fun value -> value.AsTensor<float32>()
            |> TensorIO.cloneFloatTensor

        let backboneAttentionMask =
            results
            |> Seq.find (fun value -> value.Name = "backbone_attention_mask")
            |> fun value -> value.AsTensor<float32>()
            |> TensorIO.cloneFloatTensor

        { Logits = logits
          HiddenStates = hiddenStates
          BackboneAttentionMask = backboneAttentionMask }

    member _.RunBackbone(inputEmbeddings: DenseTensor<float32>, attentionMask: DenseTensor<int64>) =
        let inputs = new List<NamedOnnxValue>()
        inputs.Add(NamedOnnxValue.CreateFromTensor("input_embeddings", inputEmbeddings))
        inputs.Add(NamedOnnxValue.CreateFromTensor("attention_mask", attentionMask))
        use results = backbone.Value.Run(inputs)

        let logits =
            results
            |> Seq.find (fun value -> value.Name = "logits")
            |> fun value -> value.AsTensor<float32>()
            |> TensorIO.cloneFloatTensor

        let hiddenStates =
            results
            |> Seq.find (fun value -> value.Name = "hidden_states")
            |> fun value -> value.AsTensor<float32>()
            |> TensorIO.cloneFloatTensor

        { Logits = logits; HiddenStates = hiddenStates }

    member _.RunDecoder(inputIds: DenseTensor<int64>, backboneLastHiddenState: DenseTensor<float32>) =
        let inputs = new List<NamedOnnxValue>()
        inputs.Add(NamedOnnxValue.CreateFromTensor("input_ids", inputIds))
        inputs.Add(NamedOnnxValue.CreateFromTensor("backbone_last_hidden_state", backboneLastHiddenState))
        use results = decoder.Value.Run(inputs)

        results
        |> Seq.find (fun value -> value.Name = "logits")
        |> fun value -> value.AsTensor<float32>()
        |> TensorIO.cloneFloatTensor

    member this.GreedyAudioFrame(backboneResult: BackboneResult, audioNumCodebooks: int) =
        if audioNumCodebooks < 1 then
            invalidArg (nameof audioNumCodebooks) "audioNumCodebooks must be positive."

        let firstIds = TensorMath.argmaxLast backboneResult.Logits
        let hidden = TensorMath.lastHidden backboneResult.HiddenStates
        let mutable ids = DenseTensor<int64>(firstIds, [| firstIds.Length; 1 |])

        for _ in 2 .. audioNumCodebooks do
            let decoderLogits = this.RunDecoder(ids, hidden)
            let nextIds = TensorMath.argmaxLast decoderLogits
            ids <- TensorMath.appendColumn ids nextIds

        ids

    member this.GreedyAudioFrame(systemPrefillResult: SystemPrefillResult, audioNumCodebooks: int) =
        this.GreedyAudioFrame({ Logits = systemPrefillResult.Logits; HiddenStates = systemPrefillResult.HiddenStates }, audioNumCodebooks)

    member _.RunCodecDecode(audioCodes: DenseTensor<int64>) =
        let session =
            codecDecode.Value
            |> Option.defaultWith (fun () -> invalidArg paths.OnnxDir "chroma_codec_decode.onnx was not found in the ONNX directory.")

        let inputs = new List<NamedOnnxValue>()
        inputs.Add(NamedOnnxValue.CreateFromTensor("audio_codes", audioCodes))
        use results = session.Run(inputs)

        results
        |> Seq.find (fun value -> value.Name = "audio_values")
        |> fun value -> value.AsTensor<float32>()
        |> TensorIO.cloneFloatTensor

    interface IDisposable with
        member _.Dispose() =
            if systemPrefill.IsValueCreated then systemPrefill.Value |> Option.iter (fun session -> session.Dispose())
            if backbone.IsValueCreated then backbone.Value.Dispose()
            if decoder.IsValueCreated then decoder.Value.Dispose()
            if codecDecode.IsValueCreated then codecDecode.Value |> Option.iter (fun session -> session.Dispose())
            sessionOptions.Dispose()

type ChromaSharedOnnxRunner(modelDir: string, bundleDir: string) =
    let manifest = SharedWeights.loadManifest bundleDir
    let store = new SafetensorWeightStore(modelDir, manifest.Initializers)
    let initializerCount = manifest.Initializers.Length
    let uniqueSourceTensorCount =
        manifest.Initializers
        |> Seq.map (fun entry -> $"{entry.SourceShard}::{entry.SourceTensor}")
        |> Seq.distinct
        |> Seq.length

    let createSession graphName =
        match manifest.Graphs.TryGetValue(graphName) with
        | false, _ -> invalidArg (nameof graphName) $"Shared bundle does not contain graph {graphName}."
        | true, graph ->
            if not (File.Exists graph.Path) then
                invalidArg graphName $"Shared ONNX graph was not found: {graph.Path}"

            let options = new SessionOptions()
            options.GraphOptimizationLevel <- GraphOptimizationLevel.ORT_DISABLE_ALL
            let session = new InferenceSession(graph.Path, options)
            session, options

    let systemPrefill = lazy (createSession "system_prefill")
    let decoder = lazy (createSession "decoder")
    let codecDecode = lazy (createSession "codec_decode")

    let disposeSession (session: Lazy<InferenceSession * SessionOptions>) =
        if session.IsValueCreated then
            let inferenceSession, options = session.Value
            inferenceSession.Dispose()
            options.Dispose()

    member _.AudioNumCodebooks = manifest.AudioNumCodebooks |> Option.defaultValue 8
    member _.MappedShardCount = store.MappedShardCount
    member _.InitializerCount = initializerCount
    member _.UniqueSourceTensorCount = uniqueSourceTensorCount

    member _.RunSystemPrefill(inputIds: DenseTensor<int64>, attentionMask: DenseTensor<int64>, inputValues: DenseTensor<float32>, inputValuesCutoffs: DenseTensor<int64>) =
        let session, _ = systemPrefill.Value
        let inputs = new List<NamedOnnxValue>()
        inputs.Add(NamedOnnxValue.CreateFromTensor("input_ids", inputIds))
        inputs.Add(NamedOnnxValue.CreateFromTensor("attention_mask", attentionMask))
        inputs.Add(NamedOnnxValue.CreateFromTensor("input_values", inputValues))
        inputs.Add(NamedOnnxValue.CreateFromTensor("input_values_cutoffs", inputValuesCutoffs))
        use results = session.Run(inputs)

        let logits =
            results
            |> Seq.find (fun value -> value.Name = "logits")
            |> fun value -> value.AsTensor<float32>()
            |> TensorIO.cloneFloatTensor

        let hiddenStates =
            results
            |> Seq.find (fun value -> value.Name = "hidden_states")
            |> fun value -> value.AsTensor<float32>()
            |> TensorIO.cloneFloatTensor

        let backboneAttentionMask =
            results
            |> Seq.find (fun value -> value.Name = "backbone_attention_mask")
            |> fun value -> value.AsTensor<float32>()
            |> TensorIO.cloneFloatTensor

        { Logits = logits
          HiddenStates = hiddenStates
          BackboneAttentionMask = backboneAttentionMask }

    member _.RunDecoder(inputIds: DenseTensor<int64>, backboneLastHiddenState: DenseTensor<float32>) =
        let session, _ = decoder.Value
        let inputs = new List<NamedOnnxValue>()
        inputs.Add(NamedOnnxValue.CreateFromTensor("input_ids", inputIds))
        inputs.Add(NamedOnnxValue.CreateFromTensor("backbone_last_hidden_state", backboneLastHiddenState))
        use results = session.Run(inputs)

        results
        |> Seq.find (fun value -> value.Name = "logits")
        |> fun value -> value.AsTensor<float32>()
        |> TensorIO.cloneFloatTensor

    member this.GreedyAudioFrame(backboneResult: BackboneResult, audioNumCodebooks: int) =
        if audioNumCodebooks < 1 then
            invalidArg (nameof audioNumCodebooks) "audioNumCodebooks must be positive."

        let firstIds = TensorMath.argmaxLast backboneResult.Logits
        let hidden = TensorMath.lastHidden backboneResult.HiddenStates
        let mutable ids = DenseTensor<int64>(firstIds, [| firstIds.Length; 1 |])

        for _ in 2 .. audioNumCodebooks do
            let decoderLogits = this.RunDecoder(ids, hidden)
            let nextIds = TensorMath.argmaxLast decoderLogits
            ids <- TensorMath.appendColumn ids nextIds

        ids

    member this.GreedyAudioFrame(systemPrefillResult: SystemPrefillResult, audioNumCodebooks: int) =
        this.GreedyAudioFrame({ Logits = systemPrefillResult.Logits; HiddenStates = systemPrefillResult.HiddenStates }, audioNumCodebooks)

    member _.RunCodecDecode(audioCodes: DenseTensor<int64>) =
        let session, _ = codecDecode.Value
        let inputs = new List<NamedOnnxValue>()
        inputs.Add(NamedOnnxValue.CreateFromTensor("audio_codes", audioCodes))
        use results = session.Run(inputs)

        results
        |> Seq.find (fun value -> value.Name = "audio_values")
        |> fun value -> value.AsTensor<float32>()
        |> TensorIO.cloneFloatTensor

    interface IDisposable with
        member _.Dispose() =
            disposeSession systemPrefill
            disposeSession decoder
            disposeSession codecDecode
            (store :> IDisposable).Dispose()

