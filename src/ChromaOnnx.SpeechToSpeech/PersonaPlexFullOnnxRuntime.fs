namespace ChromaOnnx.SpeechToSpeech

open System
open System.Collections.Generic
open System.Diagnostics
open System.IO
open System.Linq
open System.Text.Json
open System.Threading
open System.Threading.Tasks
open Microsoft.ML.OnnxRuntime
open Microsoft.ML.OnnxRuntime.Tensors
open ChromaOnnx

type private PersonaPlexFullSessions =
    { Encoder: InferenceSession
      Backbone: InferenceSession
      Decoder: InferenceSession
      MetadataPath: string
      SupportsAudioTokenGeneration: bool }

type PersonaPlexFullOnnxRuntime(options: PersonaPlexRuntimeOptions, ?pathBase: string) =
    let baseDir = defaultArg pathBase (Directory.GetCurrentDirectory())
    let fullPath path = S2sRuntimePaths.resolveAgainst baseDir path
    let modelDir = fullPath options.ModelDir
    let executionProvider =
        if String.IsNullOrWhiteSpace options.ExecutionProvider then
            "cuda"
        else
            options.ExecutionProvider.Trim().ToLowerInvariant()
    let voicePresetText =
        if String.IsNullOrWhiteSpace options.VoicePreset then
            "NATF2"
        else
            options.VoicePreset.Trim()
    let maxNewFrames = max 0 options.MaxNewFrames
    let warmupFrames = max 1 options.WarmupFrames
    let requiredFiles =
        [| "mimi_encoder.onnx"
           "mimi_decoder.onnx"
           "lm_backbone.onnx"
           "lm_backbone.onnx.data" |]
    let syncRoot = obj()
    let jsonOptions = JsonSerializerOptions(PropertyNamingPolicy = JsonNamingPolicy.CamelCase, WriteIndented = true)
    let mutable sessions: PersonaPlexFullSessions option = None
    let mutable loadError: string option = None

    let missingFiles () =
        requiredFiles
        |> Array.map (fun file -> Path.Combine(modelDir, file))
        |> Array.filter (File.Exists >> not)

    let createOptions () =
        let sessionOptions = new SessionOptions()
        sessionOptions.GraphOptimizationLevel <- GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        sessionOptions.LogSeverityLevel <- OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING
        match executionProvider with
        | "" | "cuda" ->
            let cudaOptions = new OrtCUDAProviderOptions()
            try
                cudaOptions.UpdateOptions(Dictionary<string, string>(dict [ "device_id", "0"; "enable_cuda_graph", "0" ]))
                sessionOptions.AppendExecutionProvider_CUDA(cudaOptions)
            finally
                cudaOptions.Dispose()
        | "cpu" -> sessionOptions.AppendExecutionProvider_CPU(0)
        | other -> invalidArg (nameof options.ExecutionProvider) $"Unsupported PersonaPlex execution provider '{other}'. Use cuda or cpu."
        sessionOptions

    let nodeMetadata (metadata: IReadOnlyDictionary<string, NodeMetadata>) =
        metadata
        |> Seq.map (fun pair ->
            {| name = pair.Key
               elementType = pair.Value.ElementType.ToString()
               dimensions = pair.Value.Dimensions |})
        |> Seq.toArray

    let graphMetadata (session: InferenceSession) =
        {| inputs = nodeMetadata session.InputMetadata
           outputs = nodeMetadata session.OutputMetadata |}

    let supportsAudioTokenGeneration (backbone: InferenceSession) =
        backbone.OutputMetadata.Keys
        |> Seq.exists (fun name ->
            let lower = name.ToLowerInvariant()
            lower.Contains("audio") || lower.Contains("codebook") || lower.Contains("codes") || lower.Contains("depformer"))

    let ensureSessions () =
        match sessions with
        | Some active -> active
        | None ->
            lock syncRoot (fun () ->
                match sessions with
                | Some active -> active
                | None ->
                    let missing = missingFiles ()
                    if missing.Length > 0 then
                        let message =
                            missing
                            |> Array.map (fun path -> Path.GetFileName(path) |> Option.ofObj |> Option.defaultValue path)
                            |> String.concat ", "
                            |> fun names -> $"PersonaPlex full ONNX files are missing under {modelDir}: {names}"
                        loadError <- Some message
                        invalidOp message

                    try
                        let encoderOptions = createOptions ()
                        let backboneOptions = createOptions ()
                        let decoderOptions = createOptions ()
                        let encoder = new InferenceSession(Path.Combine(modelDir, "mimi_encoder.onnx"), encoderOptions)
                        let backbone = new InferenceSession(Path.Combine(modelDir, "lm_backbone.onnx"), backboneOptions)
                        let decoder = new InferenceSession(Path.Combine(modelDir, "mimi_decoder.onnx"), decoderOptions)
                        let metadataPath = Path.Combine(modelDir, "personaplex_graph_metadata.json")
                        let audioHeadReady = supportsAudioTokenGeneration backbone
                        let metadata =
                            {| createdUtc = DateTimeOffset.UtcNow.ToString("O")
                               modelDir = modelDir
                               runtime = "full-onnx"
                               executionProvider = executionProvider
                               supportsAudioTokenGeneration = audioHeadReady
                               encoder = graphMetadata encoder
                               backbone = graphMetadata backbone
                               decoder = graphMetadata decoder |}
                        File.WriteAllText(metadataPath, JsonSerializer.Serialize(metadata, jsonOptions))

                        let active =
                            { Encoder = encoder
                              Backbone = backbone
                              Decoder = decoder
                              MetadataPath = metadataPath
                              SupportsAudioTokenGeneration = audioHeadReady }
                        sessions <- Some active
                        loadError <- None
                        active
                    with ex ->
                        loadError <- Some ex.Message
                        reraise())

    let tryEnsureSessions () =
        try
            Some(ensureSessions ())
        with ex ->
            loadError <- Some ex.Message
            None

    let cloneInt64 (tensor: Tensor<int64>) =
        DenseTensor<int64>(Enumerable.ToArray tensor, tensor.Dimensions.ToArray())

    let cloneFloat (tensor: Tensor<float32>) =
        DenseTensor<float32>(Enumerable.ToArray tensor, tensor.Dimensions.ToArray())

    let writeJson path payload =
        match Path.GetDirectoryName(Path.GetFullPath(path)) with
        | null | "" -> ()
        | dir -> Directory.CreateDirectory(dir) |> ignore
        File.WriteAllText(path, JsonSerializer.Serialize(payload, jsonOptions))

    let runEncoder (active: PersonaPlexFullSessions) (samples24k: float32 array) =
        let input: NamedOnnxValue = NamedOnnxValue.CreateFromTensor("audio", DenseTensor<float32>(samples24k, [| 1; 1; samples24k.Length |]))
        use results = active.Encoder.Run([| input |])
        let output = results |> Seq.head
        cloneInt64 (output.AsTensor<int64>())

    let runDecoder (active: PersonaPlexFullSessions) (codes: DenseTensor<int64>) =
        let input: NamedOnnxValue = NamedOnnxValue.CreateFromTensor("codes", codes)
        use results = active.Decoder.Run([| input |])
        let output = results |> Seq.head
        cloneFloat (output.AsTensor<float32>())

    let lmInputFromMimiCodes (mimiCodes: DenseTensor<int64>) =
        let dims = mimiCodes.Dimensions
        if dims.Length <> 3 || dims[0] <> 1 || dims[1] <> 8 then
            invalidArg (nameof mimiCodes) "Expected Mimi codes shape [1,8,frames]."
        let frames = dims[2]
        let values = Array.zeroCreate<int64> (17 * frames)
        let source = mimiCodes.Buffer.Span
        let sourceStrides = mimiCodes.Strides
        for codebook in 0 .. 7 do
            for frame in 0 .. frames - 1 do
                values[(codebook * frames) + frame] <- source[(codebook * sourceStrides[1]) + (frame * sourceStrides[2])]
        DenseTensor<int64>(values, [| 1; 17; frames |])

    let runBackboneProbe (active: PersonaPlexFullSessions) (codes: DenseTensor<int64>) =
        let lmCodes = lmInputFromMimiCodes codes
        let input: NamedOnnxValue = NamedOnnxValue.CreateFromTensor("codes", lmCodes)
        let stopwatch = Stopwatch.StartNew()
        use results = active.Backbone.Run([| input |])
        stopwatch.Stop()
        let outputs =
            results
            |> Seq.map (fun value ->
                let metadata = active.Backbone.OutputMetadata[value.Name]
                {| name = value.Name
                   elementType = metadata.ElementType.ToString()
                   dimensions = metadata.Dimensions |})
            |> Seq.toArray
        stopwatch.Elapsed.TotalMilliseconds, outputs

    let writeWav path (samples: DenseTensor<float32>) =
        Wave.writeMono16 path 24000 samples

    interface IPersonaPlexRuntime with
        member _.Status() =
            let missing = missingFiles ()
            let codecReady =
                File.Exists(Path.Combine(modelDir, "mimi_encoder.onnx"))
                && File.Exists(Path.Combine(modelDir, "mimi_decoder.onnx"))
            let active = if missing.Length = 0 then tryEnsureSessions () else None
            let loaded = active.IsSome
            let audioHeadReady =
                active
                |> Option.exists _.SupportsAudioTokenGeneration
            let message =
                match missing.Length, active, loadError with
                | count, _, _ when count > 0 -> $"PersonaPlex full ONNX files are missing under {modelDir}."
                | _, Some session, _ when session.SupportsAudioTokenGeneration ->
                    "PersonaPlex full ONNX graphs loaded and expose audio-token generation outputs."
                | _, Some _, _ ->
                    "PersonaPlex full ONNX graphs loaded, but lm_backbone.onnx exposes transformer_out/text_logits only; audio-token generation head is missing, so full STS is not ready."
                | _, None, Some error -> $"PersonaPlex full ONNX load failed: {error}"
                | _ -> "PersonaPlex full ONNX runtime is not loaded."
            { Ready = loaded
              CodecReady = codecReady
              SpeechToSpeechReady = loaded && audioHeadReady
              SupportsStreaming = false
              SupportsDuplex = false
              Runtime = "full-onnx"
              ModelDir = modelDir
              ExecutionProvider = executionProvider
              VoicePreset = voicePresetText
              MissingFiles = missing
              Message = message }

        member _.RunCodecRoundTripAsync(samples24k, outputDirectory, _cancellationToken) =
            task {
                let active = ensureSessions ()
                Directory.CreateDirectory(outputDirectory) |> ignore
                let stopwatch = Stopwatch.StartNew()
                let inputPath = Path.Combine(outputDirectory, "personaplex_input.wav")
                Wave.writeMono16 inputPath 24000 (DenseTensor<float32>(samples24k, [| samples24k.Length |])) |> ignore
                let codes = runEncoder active samples24k
                TensorIO.writeInt64s (Path.Combine(outputDirectory, "personaplex_codes.i64")) codes
                let decoded = runDecoder active codes
                let outputPath = Path.Combine(outputDirectory, "audio.wav")
                let stats = writeWav outputPath decoded
                stopwatch.Stop()
                writeJson
                    (Path.Combine(outputDirectory, "personaplex_codec_metadata.json"))
                    {| inputSamples = samples24k.Length
                       codeShape = codes.Dimensions.ToArray()
                       outputSamples = decoded.Dimensions.ToArray()
                       wav = stats |}
                return
                    { OutputPath = Some outputPath
                      DurationMs = stopwatch.Elapsed.TotalMilliseconds
                      InferenceTimeMs = stopwatch.Elapsed.TotalMilliseconds
                      SampleRate = 24000
                      Message = "PersonaPlex full ONNX codec round-trip completed using first-party ORT sessions." }
            }

        member _.RunSpeechToSpeechAsync(samples24k, outputDirectory, _cancellationToken) =
            task {
                let active = ensureSessions ()
                Directory.CreateDirectory(outputDirectory) |> ignore
                let stopwatch = Stopwatch.StartNew()
                let inputPath = Path.Combine(outputDirectory, "personaplex_input.wav")
                Wave.writeMono16 inputPath 24000 (DenseTensor<float32>(samples24k, [| samples24k.Length |])) |> ignore
                let codes = runEncoder active samples24k
                let inputFrames = codes.Dimensions[2]
                let lmMs, lmOutputs = runBackboneProbe active codes
                TensorIO.writeInt64s (Path.Combine(outputDirectory, "personaplex_input_codes.i64")) codes

                let outputPath = Path.Combine(outputDirectory, "audio.wav")
                let decoded = runDecoder active codes
                let stats = writeWav outputPath decoded
                stopwatch.Stop()

                let generatedFrames =
                    if active.SupportsAudioTokenGeneration then maxNewFrames else 0
                let message =
                    if active.SupportsAudioTokenGeneration then
                        "PersonaPlex LM graph exposes audio-token outputs, but generation loop has not yet been specialized for this graph contract."
                    else
                        "PersonaPlex lm_backbone.onnx ran successfully, but exposes transformer_out/text_logits only. No audio-token generation head is available, so audio.wav is codec passthrough for diagnostics."
                writeJson
                    (Path.Combine(outputDirectory, "personaplex_lm_metadata.json"))
                    {| inputSamples = samples24k.Length
                       inputCodeShape = codes.Dimensions.ToArray()
                       lmInputShape = [| 1; 17; inputFrames |]
                       lmOutputs = lmOutputs
                       lmInferenceMs = lmMs
                       supportsAudioTokenGeneration = active.SupportsAudioTokenGeneration
                       generatedFrames = generatedFrames
                       requestedMaxNewFrames = maxNewFrames
                       warmupFrames = warmupFrames
                       wav = stats
                       graphMetadataPath = active.MetadataPath
                       message = message |}
                return
                    { OutputPath = Some outputPath
                      InputFrames = inputFrames
                      GeneratedFrames = generatedFrames
                      SampleRate = 24000
                      DurationMs = stopwatch.Elapsed.TotalMilliseconds
                      InferenceTimeMs = stopwatch.Elapsed.TotalMilliseconds
                      Message = message }
            }

    interface IDisposable with
        member _.Dispose() =
            match sessions with
            | Some active ->
                active.Encoder.Dispose()
                active.Backbone.Dispose()
                active.Decoder.Dispose()
            | None -> ()
            sessions <- None
