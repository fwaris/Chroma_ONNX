namespace ChromaOnnx

open System
open System.Diagnostics
open System.IO
open System.Text
open System.Threading
open System.Threading.Tasks
open Microsoft.ML.OnnxRuntimeGenAI
open Microsoft.ML.OnnxRuntime.Tensors

type private GemmaOrtGenAiHandles =
    { Oga: OgaHandle
      Model: Model
      Tokenizer: Tokenizer
      Processor: MultiModalProcessor }

type GemmaOrtGenAiRunner(modelDir: string, variant: string, executionProvider: string, ?maxAudioSeconds: float) =
    let effectiveModelDir = Path.GetFullPath(modelDir)
    let processor = GemmaProcessor(effectiveModelDir, ?maxAudioSeconds = maxAudioSeconds)
    let provider = if String.IsNullOrWhiteSpace executionProvider then "cuda" else executionProvider.Trim().ToLowerInvariant()
    let maxAudioSeconds = defaultArg maxAudioSeconds 30.0 |> max 0.1
    let syncRoot = obj ()
    let mutable handles: GemmaOrtGenAiHandles option = None

    let requiredFiles =
        [| "genai_config.json"
           "tokenizer.json"
           "embedding/model.onnx"
           "embedding/model.onnx.data"
           "audio_encoder/model.onnx"
           "audio_encoder/model.onnx.data"
           "audio_feature_extraction.json"
           "decoder/model.onnx"
           "decoder/model.onnx.data" |]

    let missingFiles () =
        requiredFiles
        |> Array.map (fun relative -> relative, Path.Combine(effectiveModelDir, relative.Replace('/', Path.DirectorySeparatorChar)))
        |> Array.filter (fun (_, path) -> not (File.Exists path))
        |> Array.map fst

    let createModel () =
        match provider with
        | "cuda" -> new Model(effectiveModelDir)
        | "cpu" ->
            use config = new Config(effectiveModelDir)
            config.ClearProviders()
            config.AppendProvider("cpu")
            new Model(config)
        | other -> invalidArg (nameof executionProvider) $"Unsupported Gemma ORT GenAI execution provider '{other}'. Use cuda or cpu."

    let load () =
        match handles with
        | Some loaded -> loaded
        | None ->
            lock syncRoot (fun () ->
                match handles with
                | Some loaded -> loaded
                | None ->
                    let missing = missingFiles ()
                    if missing.Length > 0 then
                        let missingText = String.Join(", ", missing)
                        invalidOp $"Gemma ORT GenAI model is not ready. Missing files: {missingText}"

                    let oga = new OgaHandle()
                    try
                        let model = createModel ()
                        try
                            let tokenizer = new Tokenizer(model)
                            try
                                let multimodal = new MultiModalProcessor(model)
                                let loaded =
                                    { Oga = oga
                                      Model = model
                                      Tokenizer = tokenizer
                                      Processor = multimodal }
                                handles <- Some loaded
                                loaded
                            with
                            | ex ->
                                tokenizer.Dispose()
                                reraise ()
                        with
                        | ex ->
                            model.Dispose()
                            reraise ()
                    with
                    | ex ->
                        oga.Dispose()
                        reraise ())

    let clamp16 (sample: float32) =
        let value = float sample |> max -1.0 |> min 1.0
        int16 (Math.Round(value * 32767.0))

    let wavBytes16k (samples: float32 array) =
        use stream = new MemoryStream()
        use writer = new BinaryWriter(stream, Encoding.ASCII, leaveOpen = true)
        let dataBytes = samples.Length * sizeof<int16>
        let writeAscii (text: string) = writer.Write(Encoding.ASCII.GetBytes(text))
        writeAscii "RIFF"
        writer.Write(36 + dataBytes)
        writeAscii "WAVE"
        writeAscii "fmt "
        writer.Write(16)
        writer.Write(int16 1)
        writer.Write(int16 1)
        writer.Write(16000)
        writer.Write(16000 * sizeof<int16>)
        writer.Write(int16 sizeof<int16>)
        writer.Write(int16 16)
        writeAscii "data"
        writer.Write(dataBytes)
        for sample in samples do
            writer.Write(clamp16 sample)
        writer.Flush()
        stream.ToArray()

    let promptWithoutAudioExpansion (request: GemmaGenerationRequest) =
        processor.RenderChat(request.Messages, request.Tools, request.AddGenerationPrompt)

    let prepareWithoutAudioExtraction (request: GemmaGenerationRequest) =
        processor.Prepare({ request with Audio16k = None })

    let configureSearch (parameters: GeneratorParams) maxLength (request: GemmaGenerationRequest) =
        let topK = if request.TopK <= 0 then 1 else request.TopK
        let topP = request.TopP |> max 0.0 |> min 1.0
        let temperature = if request.Temperature <= 0.0 then 1.0 else request.Temperature
        let doSample = request.Temperature > 0.0 && topK > 1
        parameters.SetSearchOption("max_length", float maxLength)
        parameters.SetSearchOption("do_sample", doSample)
        parameters.SetSearchOption("temperature", temperature)
        parameters.SetSearchOption("top_p", topP)
        parameters.SetSearchOption("top_k", float topK)

    let generate (request: GemmaGenerationRequest) (cancellationToken: CancellationToken) =
        let stopwatch = Stopwatch.StartNew()
        let loaded = load ()
        let prompt = promptWithoutAudioExpansion request
        let prepared = prepareWithoutAudioExtraction request
        let promptTokenEstimate = prepared.InputIds.Dimensions[1]
        let audioTokenBudget =
            match request.Audio16k with
            | Some audio -> processor.ComputeAudioTokenCount audio.Length
            | None -> 0
        let maxNewTokens = max 1 request.MaxNewTokens
        let maxLength = max 1 (promptTokenEstimate + audioTokenBudget + maxNewTokens + 8)
        use parameters = new GeneratorParams(loaded.Model)
        configureSearch parameters maxLength request
        use generator = new Generator(loaded.Model, parameters)

        let inputTokenCount =
            match request.Audio16k with
            | Some audio ->
                let truncated =
                    let maxSamples = int (Math.Ceiling(maxAudioSeconds * 16000.0))
                    if audio.Length > maxSamples then audio[0 .. maxSamples - 1] else audio
                use audios = Audios.Load(wavBytes16k truncated)
                use inputs = loaded.Processor.ProcessAudios(prompt, audios)
                generator.SetInputs(inputs)
                int (generator.TokenCount())
            | None ->
                use sequences = loaded.Tokenizer.Encode(prompt)
                let count = sequences[0UL].Length
                generator.AppendTokenSequences(sequences)
                count

        let mutable generated = 0
        while not (generator.IsDone()) && generated < maxNewTokens do
            cancellationToken.ThrowIfCancellationRequested()
            generator.GenerateNextToken()
            generated <- generated + 1

        let sequence = generator.GetSequence(0UL).ToArray()
        let newTokens =
            if sequence.Length > inputTokenCount then
                sequence[inputTokenCount ..]
            else
                Array.empty
        let text =
            if newTokens.Length = 0 then
                ""
            else
                loaded.Tokenizer.Decode(ReadOnlySpan<int>(newTokens)).Trim()
        stopwatch.Stop()
        { Text = text
          Prompt = prompt
          InputTokenCount = inputTokenCount
          OutputTokenIds = newTokens |> Array.map int64
          StopReason = if generator.IsDone() then "eos" else "max_new_tokens"
          TimingsMs = Map.ofList [ "total", stopwatch.Elapsed.TotalMilliseconds ] }

    member _.Processor = processor

    interface IGemmaRuntime with
        member _.Status() =
            let missing = missingFiles ()
            let loaded =
                match handles with
                | Some _ -> [| "ort-genai" |]
                | None -> Array.empty
            { Ready = missing.Length = 0
              ModelDir = effectiveModelDir
              Variant = variant
              ExecutionProvider = provider
              MissingFiles = missing
              LoadedSessions = loaded
              Message =
                if missing.Length = 0 then
                    if loaded.Length = 0 then
                        "Gemma ORT GenAI model files are present; model will load on first use."
                    else
                        "Gemma ORT GenAI model is loaded."
                else
                    $"Gemma ORT GenAI model is missing {missing.Length} required file(s)." }

        member _.Prepare(request: GemmaGenerationRequest) =
            prepareWithoutAudioExtraction request

        member _.GenerateAsync(request: GemmaGenerationRequest, cancellationToken: CancellationToken) =
            Task.Run((fun () -> generate request cancellationToken), cancellationToken)

    interface IDisposable with
        member _.Dispose() =
            lock syncRoot (fun () ->
                match handles with
                | Some loaded ->
                    loaded.Processor.Dispose()
                    loaded.Tokenizer.Dispose()
                    loaded.Model.Dispose()
                    loaded.Oga.Dispose()
                    handles <- None
                | None -> ())
