namespace ChromaOnnx.SpeechToSpeech

open System
open System.Globalization
open System.IO
open System.Threading
open System.Threading.Tasks
open ElBruno.PersonaPlex
open ElBruno.PersonaPlex.Audio
open ElBruno.PersonaPlex.Pipeline

type ElBrunoPersonaPlexRuntime(options: PersonaPlexRuntimeOptions, ?pathBase: string) =
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
    let requiredFiles =
        [| "mimi_encoder.onnx"
           "mimi_decoder.onnx" |]
    let stsUnavailableMessage =
        "PersonaPlex speech-to-speech generation is unavailable in ElBruno.PersonaPlex 0.6.1; this package currently exposes Mimi codec encode/decode and ProcessAsync round-trip behavior."
    let syncRoot = obj()
    let mutable pipeline: PersonaPlexPipeline option = None

    let missingFiles () =
        requiredFiles
        |> Array.map (fun file -> Path.Combine(modelDir, file))
        |> Array.filter (File.Exists >> not)

    let parseVoicePreset () =
        match Enum.TryParse<VoicePreset>(voicePresetText, true) with
        | true, value -> value
        | false, _ -> VoicePreset.NATF2

    let parseExecutionProvider () =
        match executionProvider with
        | "cpu" -> ExecutionProvider.CPU
        | "directml" | "dml" -> ExecutionProvider.DirectML
        | _ -> ExecutionProvider.CUDA

    let sessionOptionsFactory () =
        match executionProvider with
        | "cpu" -> SessionOptionsHelper.CreateCpuOptions()
        | "directml" | "dml" -> SessionOptionsHelper.CreateDirectMlOptions()
        | _ -> SessionOptionsHelper.CreateCudaOptions()

    let ensurePipeline (cancellationToken: CancellationToken) =
        task {
            match pipeline with
            | Some value -> return value
            | None ->
                let missing = missingFiles ()
                if missing.Length > 0 then
                    let missingText =
                        missing
                        |> Array.map (fun path -> Path.GetFileName(path) |> Option.ofObj |> Option.defaultValue path)
                        |> String.concat ", "
                    invalidOp $"PersonaPlex codec files are not staged locally under {modelDir}. Missing: {missingText}"

                let! created =
                    PersonaPlexPipeline.CreateAsync(
                        modelDir,
                        PersonaPlexOptions(
                            ModelDirectory = modelDir,
                            VoicePreset = parseVoicePreset (),
                            TextPrompt = options.TextPrompt,
                            ExecutionProvider = parseExecutionProvider (),
                            HuggingFaceRepoId = options.HuggingFaceRepoId
                        ),
                        Func<Microsoft.ML.OnnxRuntime.SessionOptions>(sessionOptionsFactory),
                        null,
                        cancellationToken
                    )

                let active =
                    lock syncRoot (fun () ->
                        match pipeline with
                        | Some existing ->
                            created.Dispose()
                            existing
                        | None ->
                            pipeline <- Some created
                            created)
                return active
        }

    interface IPersonaPlexRuntime with
        member _.Status() =
            let missing = missingFiles ()
            let codecReady = missing.Length = 0
            { Ready = codecReady
              CodecReady = codecReady
              SpeechToSpeechReady = false
              SupportsStreaming = false
              SupportsDuplex = false
              Runtime = "elbruno-codec"
              ModelDir = modelDir
              ExecutionProvider = executionProvider
              VoicePreset = voicePresetText
              MissingFiles = missing
              Message =
                if codecReady then
                    stsUnavailableMessage
                else
                    $"PersonaPlex codec files are missing under {modelDir}." }

        member _.RunCodecRoundTripAsync(samples24k, outputDirectory, cancellationToken) =
            task {
                Directory.CreateDirectory(outputDirectory) |> ignore
                let! activePipeline = ensurePipeline cancellationToken
                let inputPath = Path.Combine(outputDirectory, "personaplex_input.wav")
                let outputPath = Path.Combine(outputDirectory, "audio.wav")
                WavWriter.Write(inputPath, samples24k, 24000)
                let! result =
                    activePipeline.ProcessAsync(
                        inputPath,
                        Nullable(parseVoicePreset ()),
                        options.TextPrompt,
                        outputPath,
                        Nullable<int>(),
                        cancellationToken
                    )
                return
                    { OutputPath = if File.Exists outputPath then Some outputPath else None
                      DurationMs = result.DurationMs
                      InferenceTimeMs = result.InferenceTimeMs
                      SampleRate = 24000
                      Message =
                        String.Format(
                            CultureInfo.InvariantCulture,
                            "PersonaPlex codec round-trip completed; full STS remains unavailable in ElBruno.PersonaPlex 0.6.1."
                        ) }
            }

        member _.RunSpeechToSpeechAsync(_samples24k, _outputDirectory, _cancellationToken) =
            task {
                return
                    { OutputPath = None
                      InputFrames = 0
                      GeneratedFrames = 0
                      SampleRate = 24000
                      DurationMs = 0.0
                      InferenceTimeMs = 0.0
                      Message = stsUnavailableMessage }
            }

    interface IDisposable with
        member _.Dispose() =
            match pipeline with
            | Some value -> value.Dispose()
            | None -> ()
            pipeline <- None
