namespace ChromaOnnx.SpeechToSpeech

open System
open System.Collections.Generic
open System.Diagnostics
open System.IO
open System.Linq
open System.Text
open System.Threading
open System.Threading.Tasks
open Microsoft.ML.OnnxRuntime
open Microsoft.ML.OnnxRuntime.Tensors
open Tokenizers.HuggingFace.Tokenizer
open ChromaOnnx

module private ChatterboxWave =
    let private invalidData message = raise (InvalidDataException(message))

    let readMono (path: string) =
        use stream = File.OpenRead(path)
        use reader = new BinaryReader(stream, Encoding.ASCII, leaveOpen = false)
        let ascii count = Encoding.ASCII.GetString(reader.ReadBytes count)
        if ascii 4 <> "RIFF" then invalidData "WAV file is missing RIFF header."
        reader.ReadUInt32() |> ignore
        if ascii 4 <> "WAVE" then invalidData "WAV file is missing WAVE header."

        let mutable audioFormat = 0us
        let mutable channels = 0us
        let mutable bitsPerSample = 0us
        let mutable sampleRate = 0
        let mutable foundData = false
        let mutable samples = Array.empty<float32>
        while stream.Position + 8L <= stream.Length && not foundData do
            let chunkId = ascii 4
            let chunkSize = reader.ReadUInt32()
            let chunkStart = stream.Position
            match chunkId with
            | "fmt " ->
                audioFormat <- reader.ReadUInt16()
                channels <- reader.ReadUInt16()
                sampleRate <- reader.ReadInt32()
                reader.ReadUInt32() |> ignore
                reader.ReadUInt16() |> ignore
                bitsPerSample <- reader.ReadUInt16()
                stream.Position <- chunkStart + int64 chunkSize
            | "data" ->
                if channels = 0us then invalidData "WAV fmt chunk must appear before data."
                let raw = reader.ReadBytes(int chunkSize)
                let frameCount = raw.Length / ((int bitsPerSample / 8) * int channels)
                samples <- Array.zeroCreate<float32> frameCount
                match audioFormat, bitsPerSample with
                | 1us, 16us ->
                    let values = Array.zeroCreate<int16> (raw.Length / sizeof<int16>)
                    Buffer.BlockCopy(raw, 0, values, 0, raw.Length)
                    for frame in 0 .. frameCount - 1 do
                        let mutable acc = 0.0f
                        for channel in 0 .. int channels - 1 do
                            acc <- acc + float32 values[frame * int channels + channel] / 32768.0f
                        samples[frame] <- acc / float32 channels
                | 3us, 32us ->
                    let values = Array.zeroCreate<float32> (raw.Length / sizeof<float32>)
                    Buffer.BlockCopy(raw, 0, values, 0, raw.Length)
                    for frame in 0 .. frameCount - 1 do
                        let mutable acc = 0.0f
                        for channel in 0 .. int channels - 1 do
                            acc <- acc + values[frame * int channels + channel]
                        samples[frame] <- acc / float32 channels
                | _ ->
                    invalidData $"Unsupported WAV format {audioFormat} with {bitsPerSample} bits per sample."
                foundData <- true
            | _ ->
                stream.Position <- chunkStart + int64 chunkSize
            if chunkSize % 2u = 1u && stream.Position < stream.Length then
                stream.Position <- stream.Position + 1L

        if not foundData then invalidData "WAV file has no data chunk."
        sampleRate, samples

    let resampleLinear targetRate sourceRate (samples: float32 array) =
        if sourceRate = targetRate then
            samples
        elif samples.Length = 0 then
            Array.empty
        else
            let outputLength = max 1 (int (Math.Round(float samples.Length * float targetRate / float sourceRate)))
            Array.init outputLength (fun index ->
                let position = float index * float sourceRate / float targetRate
                let left = int (Math.Floor position)
                let right = min (samples.Length - 1) (left + 1)
                let frac = float32 (position - float left)
                samples[left] * (1.0f - frac) + samples[right] * frac)

    let writeMono16 (path: string) sampleRate (samples: float32 array) =
        Wave.writeMono16 path sampleRate (DenseTensor<float32>(samples, [| samples.Length |])) |> ignore

    let formatStats name sampleRate (samples: float32 array) =
        if samples.Length = 0 then
            $"{name}=0.00s peak=0.0000 rms=0.0000 meanAbs=0.0000"
        else
            let mutable peak = 0.0
            let mutable sumAbs = 0.0
            let mutable sumSq = 0.0
            for sample in samples do
                let value = float sample
                let absValue = Math.Abs value
                peak <- max peak absValue
                sumAbs <- sumAbs + absValue
                sumSq <- sumSq + value * value
            let duration = float samples.Length / float sampleRate
            let rms = Math.Sqrt(sumSq / float samples.Length)
            let meanAbs = sumAbs / float samples.Length
            sprintf "%s=%0.2fs peak=%0.4f rms=%0.4f meanAbs=%0.4f" name duration peak rms meanAbs

type private ChatterboxOrtTensor =
    | TensorFloat of DenseTensor<float32>
    | TensorHalf of DenseTensor<Half>
    | TensorOrtFloat16 of DenseTensor<Float16>

type private ChatterboxReference =
    { AudioFeatures: DenseTensor<float32>
      AudioTokens: DenseTensor<int64>
      SpeakerEmbeddings: DenseTensor<float32>
      SpeakerFeatures: DenseTensor<float32>
      Key: string }

type private ChatterboxSessions =
    { SpeechEncoder: InferenceSession
      EmbedTokens: InferenceSession
      LanguageModel: InferenceSession
      Decoder: InferenceSession
      Tokenizer: Tokenizer
      Options: SessionOptions array }
    interface IDisposable with
        member this.Dispose() =
            this.SpeechEncoder.Dispose()
            this.EmbedTokens.Dispose()
            this.LanguageModel.Dispose()
            this.Decoder.Dispose()
            this.Options |> Array.iter _.Dispose()

type ChatterboxOnnxTtsRuntime(options: TtsRuntimeOptions, ?pathBase: string) =
    let startSpeechToken = 6561L
    let stopSpeechToken = 6562L
    let exaggerationToken = 6563L
    let textStartToken = 255L
    let textStopToken = 0L
    let outputSampleRate = 24000
    let pathBase = defaultArg pathBase (Directory.GetCurrentDirectory())
    let fullPath path = S2sRuntimePaths.resolveAgainst pathBase path
    let modelDir =
        if String.IsNullOrWhiteSpace options.ModelDir then ""
        else fullPath options.ModelDir
    let variant =
        if String.IsNullOrWhiteSpace options.Variant then "q4f16"
        else options.Variant.Trim().ToLowerInvariant()
    let executionProvider =
        if String.IsNullOrWhiteSpace options.ExecutionProvider then "cuda"
        else options.ExecutionProvider.Trim().ToLowerInvariant()
    let maxSteps = max 8 options.MaxSteps
    let exaggeration = options.Exaggeration |> max 0.0 |> min 2.0
    let repetitionPenalty =
        if options.RepetitionPenalty > 0.0 then options.RepetitionPenalty else 1.2
    let chunkSeconds = max 0.05 options.StreamingChunkSeconds
    let syncRoot = obj()
    let synthesisGate = SemaphoreSlim(1, 1)
    let mutable sessions: ChatterboxSessions option = None
    let mutable cachedReference: ChatterboxReference option = None

    let lmFileName () =
        match variant with
        | "" | "q4f16" -> "language_model_q4f16.onnx"
        | "q4" -> "language_model_q4.onnx"
        | "fp16" | "f16" -> "language_model_fp16.onnx"
        | "fp32" | "f32" | "onnx" -> "language_model.onnx"
        | other -> invalidArg (nameof options.Variant) $"Unsupported Chatterbox language model variant '{other}'. Use q4f16, q4, fp16, or fp32."

    let onnxPath fileName = Path.Combine(modelDir, "onnx", fileName)

    let requiredFiles () =
        let lm = lmFileName ()
        [| Path.Combine(modelDir, "tokenizer.json")
           Path.Combine(modelDir, "default_voice.wav")
           onnxPath "speech_encoder.onnx"
           onnxPath "speech_encoder.onnx_data"
           onnxPath "embed_tokens.onnx"
           onnxPath "embed_tokens.onnx_data"
           onnxPath "conditional_decoder.onnx"
           onnxPath "conditional_decoder.onnx_data"
           onnxPath lm
           onnxPath $"{lm}_data" |]

    let configuredVoiceSample () =
        if String.IsNullOrWhiteSpace options.VoiceSamplePath then
            Path.Combine(modelDir, "default_voice.wav")
        else
            fullPath options.VoiceSamplePath

    let requestVoiceSample (request: TtsSynthesisRequest) =
        request.VoiceSamplePath
        |> Option.filter (String.IsNullOrWhiteSpace >> not)
        |> Option.map fullPath
        |> Option.defaultValue (configuredVoiceSample ())

    let missingFiles () =
        let missing = ResizeArray<string>()
        if String.IsNullOrWhiteSpace modelDir || not (Directory.Exists modelDir) then
            missing.Add(if String.IsNullOrWhiteSpace modelDir then "VoiceAgent:Tts:ModelDir" else modelDir)
        else
            for path in requiredFiles () do
                if not (File.Exists path) then missing.Add(path)
        if not (String.IsNullOrWhiteSpace options.VoiceSamplePath) then
            let voicePath = fullPath options.VoiceSamplePath
            if not (File.Exists voicePath) then missing.Add(voicePath)
        missing.ToArray()

    let createOptions () =
        let sessionOptions = new SessionOptions()
        sessionOptions.GraphOptimizationLevel <- GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        sessionOptions.LogSeverityLevel <- OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR
        match executionProvider with
        | "" | "cuda" ->
            let cudaOptions = new OrtCUDAProviderOptions()
            try
                cudaOptions.UpdateOptions(
                    Dictionary<string, string>(
                        dict [ "device_id", (max 0 options.CudaDeviceId).ToString(Globalization.CultureInfo.InvariantCulture)
                               "enable_cuda_graph", "0" ]
                    )
                )
                sessionOptions.AppendExecutionProvider_CUDA(cudaOptions)
            finally
                cudaOptions.Dispose()
        | "cpu" -> sessionOptions.AppendExecutionProvider_CPU(0)
        | other -> invalidArg (nameof options.ExecutionProvider) $"Unsupported Chatterbox execution provider '{other}'. Use cuda or cpu."
        sessionOptions

    let loadSessions () =
        match sessions with
        | Some active -> active
        | None ->
            lock syncRoot (fun () ->
                match sessions with
                | Some active -> active
                | None ->
                    let missing = missingFiles ()
                    if missing.Length > 0 then
                        let missingText = String.Join(", ", missing)
                        invalidOp $"Chatterbox ONNX runtime is not ready. Missing: {missingText}"
                    let speechOptions = createOptions ()
                    let embedOptions = createOptions ()
                    let lmOptions = createOptions ()
                    let decoderOptions = createOptions ()
                    try
                        let active =
                            { SpeechEncoder = new InferenceSession(onnxPath "speech_encoder.onnx", speechOptions)
                              EmbedTokens = new InferenceSession(onnxPath "embed_tokens.onnx", embedOptions)
                              LanguageModel = new InferenceSession(onnxPath (lmFileName ()), lmOptions)
                              Decoder = new InferenceSession(onnxPath "conditional_decoder.onnx", decoderOptions)
                              Tokenizer = Tokenizer.FromFile(Path.Combine(modelDir, "tokenizer.json"))
                              Options = [| speechOptions; embedOptions; lmOptions; decoderOptions |] }
                        sessions <- Some active
                        active
                    with
                    | _ ->
                        speechOptions.Dispose()
                        embedOptions.Dispose()
                        lmOptions.Dispose()
                        decoderOptions.Dispose()
                        reraise())

    let encode (tokenizer: Tokenizer) (text: string) =
        let encoding =
            tokenizer.Encode(
                text,
                false,
                null,
                false,
                false,
                false,
                false,
                false,
                true,
                false
            )

        encoding
        |> Seq.head
        |> fun value -> value.Ids
        |> Seq.map int64
        |> Seq.toArray
        |> fun raw ->
            [| yield exaggerationToken
               yield textStartToken
               yield! raw
               yield textStopToken
               yield startSpeechToken
               yield startSpeechToken |]

    let cloneFloat (value: DisposableNamedOnnxValue) =
        match value.Value with
        | :? Tensor<float32> as tensor -> DenseTensor<float32>(Enumerable.ToArray tensor, tensor.Dimensions.ToArray())
        | :? Tensor<Half> as tensor ->
            DenseTensor<float32>(tensor |> Seq.map float32 |> Seq.toArray, tensor.Dimensions.ToArray())
        | :? Tensor<Float16> as tensor ->
            DenseTensor<float32>(tensor |> Seq.map (fun item -> float32 item) |> Seq.toArray, tensor.Dimensions.ToArray())
        | other -> invalidOp $"Chatterbox output {value.Name} has unsupported tensor type {other.GetType().FullName}."

    let cloneInt64 (value: DisposableNamedOnnxValue) =
        DenseTensor<int64>(value.AsTensor<int64>().ToArray(), value.AsTensor<int64>().Dimensions.ToArray())

    let cloneOrtTensor (value: DisposableNamedOnnxValue) =
        match value.Value with
        | :? Tensor<float32> as tensor -> TensorFloat(DenseTensor<float32>(Enumerable.ToArray tensor, tensor.Dimensions.ToArray()))
        | :? Tensor<Half> as tensor -> TensorHalf(DenseTensor<Half>(tensor.ToArray(), tensor.Dimensions.ToArray()))
        | :? Tensor<Float16> as tensor -> TensorOrtFloat16(DenseTensor<Float16>(Memory<Float16>(tensor.ToArray()), ReadOnlySpan<int>(tensor.Dimensions.ToArray()), false))
        | other -> invalidOp $"Chatterbox output {value.Name} has unsupported tensor type {other.GetType().FullName}."

    let tensorToInput name tensor =
        match tensor with
        | TensorFloat value -> NamedOnnxValue.CreateFromTensor(name, value)
        | TensorHalf value -> NamedOnnxValue.CreateFromTensor(name, value)
        | TensorOrtFloat16 value -> NamedOnnxValue.CreateFromTensor(name, value)

    let metadataDims (metadata: NodeMetadata) =
        metadata.Dimensions
        |> Seq.toArray
        |> Array.mapi (fun index dim ->
            if index = 0 then 1
            elif dim > 0 then dim
            else 0)

    let tensorCount dims =
        dims |> Array.fold (fun total dim -> total * dim) 1

    let zeroTensorForMetadata name (metadata: NodeMetadata) =
        let dims = metadataDims metadata
        let count = tensorCount dims
        if metadata.ElementType = typeof<float32> then
            NamedOnnxValue.CreateFromTensor(name, DenseTensor<float32>(Array.zeroCreate<float32> count, dims))
        elif metadata.ElementType = typeof<Half> then
            NamedOnnxValue.CreateFromTensor(name, DenseTensor<Half>(Memory<Half>(Array.zeroCreate<Half> count), ReadOnlySpan<int>(dims), false))
        elif metadata.ElementType = typeof<Float16> then
            NamedOnnxValue.CreateFromTensor(name, DenseTensor<Float16>(Memory<Float16>(Array.zeroCreate<Float16> count), ReadOnlySpan<int>(dims), false))
        else
            invalidOp $"Unsupported Chatterbox cache tensor type {metadata.ElementType} for {name}."

    let positionIds (ids: int64 array) =
        ids
        |> Array.mapi (fun index id ->
            if id >= startSpeechToken then 0L else int64 index - 1L)

    let findOutput name (results: seq<DisposableNamedOnnxValue>) =
        results |> Seq.find (fun value -> value.Name = name)

    let runSpeechEncoder (active: ChatterboxSessions) referencePath =
        let sampleRate, original = ChatterboxWave.readMono referencePath
        let samples = ChatterboxWave.resampleLinear outputSampleRate sampleRate original
        use results =
            active.SpeechEncoder.Run(
                [| NamedOnnxValue.CreateFromTensor("audio_values", DenseTensor<float32>(samples, [| 1; samples.Length |])) |]
            )
        { AudioFeatures = findOutput "audio_features" results |> cloneFloat
          AudioTokens = findOutput "audio_tokens" results |> cloneInt64
          SpeakerEmbeddings = findOutput "speaker_embeddings" results |> cloneFloat
          SpeakerFeatures = findOutput "speaker_features" results |> cloneFloat
          Key = $"{referencePath}|{File.GetLastWriteTimeUtc(referencePath).Ticks}|{samples.Length}" }

    let ensureReference active referencePath =
        let key =
            if File.Exists referencePath then
                $"{referencePath}|{File.GetLastWriteTimeUtc(referencePath).Ticks}"
            else
                referencePath
        match cachedReference with
        | Some reference when reference.Key.StartsWith(key, StringComparison.Ordinal) -> reference
        | _ ->
            let loaded = runSpeechEncoder active referencePath
            cachedReference <- Some loaded
            loaded

    let runEmbed (active: ChatterboxSessions) (inputIds: DenseTensor<int64>) (posIds: DenseTensor<int64>) =
        use results =
            active.EmbedTokens.Run(
                [| NamedOnnxValue.CreateFromTensor("input_ids", inputIds)
                   NamedOnnxValue.CreateFromTensor("position_ids", posIds)
                   NamedOnnxValue.CreateFromTensor("exaggeration", DenseTensor<float32>([| float32 exaggeration |], [| 1 |])) |]
            )
        findOutput "inputs_embeds" results |> cloneFloat

    let concatEmbeds (first: DenseTensor<float32>) (second: DenseTensor<float32>) =
        let firstDims = first.Dimensions.ToArray()
        let secondDims = second.Dimensions.ToArray()
        if firstDims.Length <> 3 || secondDims.Length <> 3 || firstDims[0] <> 1 || secondDims[0] <> 1 || firstDims[2] <> secondDims[2] then
            invalidOp "Chatterbox embeddings must have compatible [1, sequence, hidden] shapes."
        let firstCount = firstDims[1] * firstDims[2]
        let secondCount = secondDims[1] * secondDims[2]
        let values = Array.zeroCreate<float32> (firstCount + secondCount)
        first.Buffer.Span.Slice(0, firstCount).CopyTo(values.AsSpan(0, firstCount))
        second.Buffer.Span.Slice(0, secondCount).CopyTo(values.AsSpan(firstCount, secondCount))
        DenseTensor<float32>(values, [| 1; firstDims[1] + secondDims[1]; firstDims[2] |])

    let presentNameToPast (name: string) =
        if name.StartsWith("present.", StringComparison.Ordinal) then
            "past_key_values." + name.Substring("present.".Length)
        else
            name

    let updateCache (results: seq<DisposableNamedOnnxValue>) =
        let cache = Dictionary<string, ChatterboxOrtTensor>(StringComparer.Ordinal)
        for value in results do
            if value.Name.StartsWith("present.", StringComparison.Ordinal) then
                cache[presentNameToPast value.Name] <- cloneOrtTensor value
        cache

    let runLanguageModel (active: ChatterboxSessions) (inputsEmbeds: DenseTensor<float32>) (attentionMask: int64 array) (cache: Dictionary<string, ChatterboxOrtTensor> option) =
        let feeds = ResizeArray<NamedOnnxValue>()
        feeds.Add(NamedOnnxValue.CreateFromTensor("inputs_embeds", inputsEmbeds))
        feeds.Add(NamedOnnxValue.CreateFromTensor("attention_mask", DenseTensor<int64>(attentionMask, [| 1; attentionMask.Length |])))
        match cache with
        | Some current ->
            for KeyValue(name, tensor) in current do
                feeds.Add(tensorToInput name tensor)
        | None ->
            for KeyValue(name, metadata) in active.LanguageModel.InputMetadata do
                if name.StartsWith("past_key_values.", StringComparison.Ordinal) then
                    feeds.Add(zeroTensorForMetadata name metadata)

        use results = active.LanguageModel.Run(feeds)
        let logits = findOutput "logits" results |> cloneFloat
        let nextCache = updateCache results
        logits, nextCache

    let applyRepetitionPenaltyAndArgmax (generated: ResizeArray<int64>) (logits: DenseTensor<float32>) =
        let dims = logits.Dimensions.ToArray()
        let vocab = dims[dims.Length - 1]
        let vectorOffset =
            match dims.Length with
            | n when n >= 3 -> (dims[1] - 1) * vocab
            | 2 -> 0
            | _ -> 0
        let span = logits.Buffer.Span
        let scores = Array.zeroCreate<float32> vocab
        for index in 0 .. vocab - 1 do
            scores[index] <- span[vectorOffset + index]
        for token in generated do
            let index = int token
            if index >= 0 && index < scores.Length then
                let score = scores[index]
                scores[index] <-
                    if score < 0.0f then score * float32 repetitionPenalty
                    else score / float32 repetitionPenalty
        let mutable bestIndex = 0
        let mutable bestValue = Single.NegativeInfinity
        for index in 0 .. scores.Length - 1 do
            if scores[index] > bestValue then
                bestValue <- scores[index]
                bestIndex <- index
        int64 bestIndex

    let concatSpeechTokens (promptTokens: DenseTensor<int64>) (generated: int64 array) =
        let prompt = promptTokens.Buffer.Span.ToArray()
        let values = Array.zeroCreate<int64> (prompt.Length + generated.Length)
        Array.Copy(prompt, values, prompt.Length)
        Array.Copy(generated, 0, values, prompt.Length, generated.Length)
        DenseTensor<int64>(values, [| 1; values.Length |])

    let runDecoder (active: ChatterboxSessions) reference (speechTokens: DenseTensor<int64>) =
        use results =
            active.Decoder.Run(
                [| NamedOnnxValue.CreateFromTensor("speech_tokens", speechTokens)
                   NamedOnnxValue.CreateFromTensor("speaker_embeddings", reference.SpeakerEmbeddings)
                   NamedOnnxValue.CreateFromTensor("speaker_features", reference.SpeakerFeatures) |]
            )
        findOutput "waveform" results |> cloneFloat

    let flattenFloatTensor (tensor: DenseTensor<float32>) =
        let values = tensor.Buffer.Span.ToArray()
        for index in 0 .. values.Length - 1 do
            values[index] <- max -1.0f (min 1.0f values[index])
        values

    let synthesize (active: ChatterboxSessions) text referencePath (cancellationToken: CancellationToken) =
        let reference = ensureReference active referencePath
        let inputIds = encode active.Tokenizer text
        if inputIds.Length = 0 then invalidArg (nameof text) "Chatterbox text produced no tokenizer ids."
        let textInput = DenseTensor<int64>(inputIds, [| 1; inputIds.Length |])
        let textPositions = DenseTensor<int64>(positionIds inputIds, [| 1; inputIds.Length |])
        let textEmbeds = runEmbed active textInput textPositions
        let mutable inputEmbeds = concatEmbeds reference.AudioFeatures textEmbeds
        let mutable attentionMask = Array.create inputEmbeds.Dimensions[1] 1L
        let generatedWithStart = ResizeArray<int64>()
        generatedWithStart.Add startSpeechToken
        let generatedSpeech = ResizeArray<int64>()
        let mutable cache: Dictionary<string, ChatterboxOrtTensor> option = None
        let mutable stopped = false
        let mutable step = 0

        while step < maxSteps && not stopped do
            cancellationToken.ThrowIfCancellationRequested()
            let logits, nextCache = runLanguageModel active inputEmbeds attentionMask cache
            let nextToken = applyRepetitionPenaltyAndArgmax generatedWithStart logits
            generatedWithStart.Add nextToken
            if nextToken = stopSpeechToken then
                stopped <- true
            else
                generatedSpeech.Add nextToken
                let nextIds = DenseTensor<int64>([| nextToken |], [| 1; 1 |])
                let nextPositions = DenseTensor<int64>([| int64 (step + 1) |], [| 1; 1 |])
                inputEmbeds <- runEmbed active nextIds nextPositions
                attentionMask <- Array.append attentionMask [| 1L |]
                cache <- Some nextCache
            step <- step + 1

        let speechTokens = concatSpeechTokens reference.AudioTokens (generatedSpeech.ToArray())
        let waveform = runDecoder active reference speechTokens |> flattenFloatTensor
        waveform, generatedSpeech.Count, stopped

    interface ITtsRuntime with
        member _.Status() =
            let missing = missingFiles ()
            let voiceSample = configuredVoiceSample ()
            { Ready = missing.Length = 0
              SupportsVoiceCloning = true
              SupportsStreaming = false
              Runtime = "chatterbox-onnx"
              ModelDir = modelDir
              ExecutionProvider = executionProvider
              OutputSampleRate = outputSampleRate
              VoiceSamplePath = voiceSample
              MissingFiles = missing
              Message =
                if missing.Length = 0 then
                    $"Chatterbox ONNX TTS runtime is ready. variant={variant}; voiceSample={voiceSample}."
                else
                    let missingText = String.Join(", ", missing)
                    $"Chatterbox ONNX TTS runtime is not ready. Missing: {missingText}" }

        member this.SynthesizeAsync(request, emitChunk, cancellationToken) =
            task {
                let status = (this :> ITtsRuntime).Status()
                if not status.Ready then invalidOp status.Message
                let text =
                    if String.IsNullOrWhiteSpace request.Text then "."
                    else request.Text.Trim()
                Directory.CreateDirectory(request.OutputDirectory) |> ignore
                let outputPath = Path.Combine(request.OutputDirectory, request.OutputFileName)
                let referencePath = requestVoiceSample request

                do! synthesisGate.WaitAsync(cancellationToken)
                let stopwatch = Stopwatch.StartNew()
                let mutable tokenCount = 0
                let mutable stopped = false
                let mutable samples = Array.empty<float32>
                try
                    let active = loadSessions ()
                    let waveform, generatedTokens, stoppedOnStopToken = synthesize active text referencePath cancellationToken
                    samples <- waveform
                    tokenCount <- generatedTokens
                    stopped <- stoppedOnStopToken
                finally
                    stopwatch.Stop()
                    synthesisGate.Release() |> ignore

                ChatterboxWave.writeMono16 outputPath outputSampleRate samples
                let chunkSize = max 1 (int (float outputSampleRate * chunkSeconds))
                let mutable offset = 0
                while offset < samples.Length do
                    cancellationToken.ThrowIfCancellationRequested()
                    let length = min chunkSize (samples.Length - offset)
                    let chunk = Array.zeroCreate<float32> length
                    Array.Copy(samples, offset, chunk, 0, length)
                    do! emitChunk chunk
                    offset <- offset + length

                let stats = ChatterboxWave.formatStats "output" outputSampleRate samples
                let transcriptNote =
                    if request.VoiceSampleTranscript.IsSome then
                        " Voice sample transcript is configured but Chatterbox ONNX does not consume it."
                    else
                        ""
                let exaggerationText = exaggeration.ToString("0.###", Globalization.CultureInfo.InvariantCulture)
                let repetitionPenaltyText = repetitionPenalty.ToString("0.###", Globalization.CultureInfo.InvariantCulture)
                return
                    { Phase = request.Phase
                      Text = text
                      OutputPath = Some outputPath
                      SampleRate = outputSampleRate
                      Samples = samples.Length
                      DurationMs = float samples.Length / float outputSampleRate * 1000.0
                      InferenceTimeMs = stopwatch.Elapsed.TotalMilliseconds
                      Message = $"Chatterbox ONNX synthesis completed. variant={variant}; tokens={tokenCount}; stop={stopped}; exaggeration={exaggerationText}; repetitionPenalty={repetitionPenaltyText}.{transcriptNote} {stats}." }
            }

    interface IDisposable with
        member _.Dispose() =
            lock syncRoot (fun () ->
                sessions |> Option.iter (fun active -> (active :> IDisposable).Dispose())
                sessions <- None
                cachedReference <- None)
