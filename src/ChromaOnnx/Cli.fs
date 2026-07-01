namespace ChromaOnnx

open System
open System.Collections.Concurrent
open System.Collections.Generic
open System.Diagnostics
open System.Globalization
open System.IO
open System.Linq
open System.Net.WebSockets
open System.Text
open System.Text.Json
open System.Threading
open System.Threading.Tasks
open Microsoft.AspNetCore.Builder
open Microsoft.AspNetCore.Hosting
open Microsoft.AspNetCore.Http
open Microsoft.Extensions.Hosting
open Microsoft.ML.OnnxRuntime
open Microsoft.ML.OnnxRuntime.Tensors
open Tokenizers.HuggingFace.Tokenizer
open TorchSharp
open TorchSharp.Fun
module Cli =
    let private cliJsonOptions =
        JsonSerializerOptions(PropertyNamingPolicy = JsonNamingPolicy.CamelCase, WriteIndented = true)

    let rec private tryFind name args =
        match args with
        | current :: value :: _ when current = name -> Some value
        | _ :: tail -> tryFind name tail
        | [] -> None

    let private required name args =
        tryFind name args
        |> Option.defaultWith (fun () -> invalidArg name $"Missing required argument {name}.")

    let private optional defaultValue name args =
        tryFind name args |> Option.defaultValue defaultValue

    type private S2sBenchmarkIteration =
        { Iteration: int
          WallClockMs: float
          TimingsMs: Dictionary<string, float>
          FrameCount: int
          StopReason: string
          AudioCodesShape: int array
          AudioValuesShape: int array }

    let private summarizeMilliseconds (values: float array) =
        if values.Length = 0 then
            {| count = 0
               minMs = Nullable<float>()
               maxMs = Nullable<float>()
               meanMs = Nullable<float>()
               medianMs = Nullable<float>() |}
        else
            let sorted = Array.copy values
            Array.Sort(sorted)
            let median =
                if sorted.Length % 2 = 1 then
                    sorted[sorted.Length / 2]
                else
                    (sorted[sorted.Length / 2 - 1] + sorted[sorted.Length / 2]) / 2.0

            {| count = values.Length
               minMs = Nullable(sorted[0])
               maxMs = Nullable(sorted[sorted.Length - 1])
               meanMs = Nullable(Array.average values)
               medianMs = Nullable(median) |}

    let private summarizeTimingKey (key: string) (iterations: S2sBenchmarkIteration array) =
        iterations
        |> Array.choose (fun iteration ->
            match iteration.TimingsMs.TryGetValue key with
            | true, value -> Some value
            | _ -> None)
        |> summarizeMilliseconds

    let private paths args =
        let onnxDir = required "--onnx-dir" args
        { OnnxDir = onnxDir
          SystemPrefill =
            let path = Path.Combine(onnxDir, "chroma_system_prefill.onnx")
            if File.Exists path then Some path else None
          Backbone = Path.Combine(onnxDir, "chroma_backbone.onnx")
          Decoder = Path.Combine(onnxDir, "chroma_decoder.onnx")
          CodecDecode =
            let path = Path.Combine(onnxDir, "chroma_codec_decode.onnx")
            if File.Exists path then Some path else None
          GraphFiles =
            if Directory.Exists onnxDir then
                Directory.GetFiles(onnxDir, "*.onnx") |> Array.sort
            else
                Array.empty }

    let printUsage() =
        printfn "Usage:"
        printfn "  ChromaOnnx inspect --onnx-dir onnx/chroma"
        printfn "  ChromaOnnx backbone --onnx-dir onnx/chroma --input-embeddings input.f32 --attention-mask mask.i64 --batch 1 --seq 8 --hidden 2048"
        printfn "  ChromaOnnx e2e --onnx-dir onnx/chroma --input-ids ids.i64 --attention-mask mask.i64 --input-values audio.f32 --input-values-cutoffs cutoffs.i64 --batch 1 --text-seq 8 --audio-samples 24000 --output-codes codes.i64 [--output-audio audio.f32]"
        printfn "  ChromaOnnx shared-e2e --model-dir models/chroma-4b --bundle-dir onnx/chroma-shared --input-ids ids.i64 --attention-mask mask.i64 --input-values audio.f32 --input-values-cutoffs cutoffs.i64 --batch 1 --text-seq 8 --audio-samples 24000 --output-codes codes.i64 [--output-audio audio.f32]"
        printfn "  ChromaOnnx serve --model-dir models/chroma-4b --bundle-dir onnx/chroma-shared --work-dir served_runs --port 5055 --python .venv/Scripts/python.exe"
        printfn "  ChromaOnnx s2s-serve --model-dir models/chroma-4b --bundle-dir onnx/chroma-s2s --work-dir served_runs --port 5055 --execution-provider cuda --memory-mode resident-merged --ort-memory-profile quality-safe --thinker-active-frames 0 --optimized-model-cache-dir onnx/chroma-s2s/ort-cache --optimized-model-cache-format onnx --python .venv/Scripts/python.exe --python-device cuda"
        printfn "  ChromaOnnx s2s-offline --model-dir models/chroma-4b --bundle-dir onnx/chroma-s2s --prompt-text text --prompt-audio-f32 reference_24k.f32 --user-audio-f32 turn_16k.f32 --frames 8 --output-dir served_runs/offline/fsharp --execution-provider cuda --memory-mode resident-merged --ort-memory-profile quality-safe --thinker-active-frames 0"
        printfn "  ChromaOnnx s2s-debug-onnx --model-dir models/chroma-4b --bundle-dir onnx/chroma-s2s --prepared-dir served_runs/debug/prepared --output-dir served_runs/debug/onnx --execution-provider cuda --memory-mode resident-merged --ort-memory-profile quality-safe"
        printfn "  ChromaOnnx s2s-benchmark --model-dir models/chroma-4b --bundle-dir onnx/chroma-s2s --prompt-text text --prompt-audio-f32 reference_24k.f32 --user-audio-f32 turn_16k.f32 --frames 8 --warmup 1 --iterations 5 --output-dir served_runs/bench/fsharp --execution-provider cuda --memory-mode resident-merged --ort-memory-profile quality-safe --thinker-active-frames 0"
        printfn "  ChromaOnnx s2s-compare --model-dir models/chroma-4b --bundle-dir onnx/chroma-s2s --prompt-text text --prompt-audio reference.wav --user-audio turn.wav --output-dir served_runs/compare --python .venv/Scripts/python.exe --execution-provider cuda --memory-mode resident-merged --ort-memory-profile quality-safe [--thinker-active-frames 0]"
        printfn "  ChromaOnnx s2s-memory-report --model-dir models/chroma-4b --bundle-dir onnx/chroma-s2s --prompt-text text --prompt-audio-f32 reference_24k.f32 --user-audio-f32 turn_16k.f32 --frames 8 --output-dir served_runs/memory --python .venv/Scripts/python.exe --execution-provider cuda --memory-mode resident-merged --ort-memory-profile quality-safe --thinker-active-frames 0"

    let inspect args =
        use runner = new ChromaOnnxRunner(paths args)
        runner.Inspect()
        0

    let backbone args =
        let batch = required "--batch" args |> int
        let sequence = required "--seq" args |> int
        let hidden = optional "2048" "--hidden" args |> int
        let inputEmbeddingPath = required "--input-embeddings" args
        let attentionMaskPath = required "--attention-mask" args

        let embeddings =
            TensorIO.readSingles inputEmbeddingPath (batch * sequence * hidden)
            |> fun values -> TensorIO.denseFloat values [| batch; sequence; hidden |]

        let attentionMask =
            TensorIO.readInt64s attentionMaskPath (batch * sequence)
            |> fun values -> TensorIO.denseInt64 values [| batch; sequence |]

        use runner = new ChromaOnnxRunner(paths args)
        let result = runner.RunBackbone(embeddings, attentionMask)
        let ids = TensorMath.argmaxLast result.Logits
        printfn "Backbone logits shape: %A" (result.Logits.Dimensions.ToArray())
        printfn "Hidden states shape: %A" (result.HiddenStates.Dimensions.ToArray())
        printfn "Greedy codebook-0 ids: %s" (ids |> Array.map string |> String.concat ",")
        0

    let e2e args =
        let chromaPaths = paths args
        let manifest = Manifest.load chromaPaths.OnnxDir
        let batch = required "--batch" args |> int
        let textSequence = required "--text-seq" args |> int
        let audioSamples = required "--audio-samples" args |> int
        let audioNumCodebooks =
            optional (manifest.AudioNumCodebooks |> Option.map string |> Option.defaultValue "8") "--audio-codebooks" args
            |> int
        let inputIdsPath = required "--input-ids" args
        let attentionMaskPath = required "--attention-mask" args
        let inputValuesPath = required "--input-values" args
        let inputValuesCutoffsPath = required "--input-values-cutoffs" args
        let outputCodesPath = required "--output-codes" args
        let outputAudioPath = tryFind "--output-audio" args

        let inputIds =
            TensorIO.readInt64s inputIdsPath (batch * textSequence)
            |> fun values -> TensorIO.denseInt64 values [| batch; textSequence |]

        let attentionMask =
            TensorIO.readInt64s attentionMaskPath (batch * textSequence)
            |> fun values -> TensorIO.denseInt64 values [| batch; textSequence |]

        let inputValues =
            TensorIO.readSingles inputValuesPath (batch * audioSamples)
            |> fun values -> TensorIO.denseFloat values [| batch; 1; audioSamples |]

        let inputValuesCutoffs =
            TensorIO.readInt64s inputValuesCutoffsPath batch
            |> fun values -> TensorIO.denseInt64 values [| batch |]

        use runner = new ChromaOnnxRunner(chromaPaths)
        let prefill = runner.RunSystemPrefill(inputIds, attentionMask, inputValues, inputValuesCutoffs)
        let frameCodes = runner.GreedyAudioFrame(prefill, audioNumCodebooks)
        let frameCodeValues = Enumerable.ToArray(frameCodes)
        let codecCodes = DenseTensor<int64>(frameCodeValues, [| batch; audioNumCodebooks; 1 |])
        TensorIO.writeInt64s outputCodesPath codecCodes

        printfn "System prefill logits shape: %A" (prefill.Logits.Dimensions.ToArray())
        printfn "System prefill hidden states shape: %A" (prefill.HiddenStates.Dimensions.ToArray())
        printfn "Greedy audio codes shape: %A" (codecCodes.Dimensions.ToArray())
        printfn "Wrote audio codes: %s" outputCodesPath

        match outputAudioPath with
        | Some path ->
            let audio = runner.RunCodecDecode(codecCodes)
            TensorIO.writeSingles path audio
            printfn "Codec audio shape: %A" (audio.Dimensions.ToArray())
            printfn "Wrote decoded audio: %s" path
        | None -> ()

        0

    let sharedE2e args =
        let modelDir = required "--model-dir" args
        let bundleDir = required "--bundle-dir" args
        let batch = required "--batch" args |> int
        let textSequence = required "--text-seq" args |> int
        let audioSamples = required "--audio-samples" args |> int
        let inputIdsPath = required "--input-ids" args
        let attentionMaskPath = required "--attention-mask" args
        let inputValuesPath = required "--input-values" args
        let inputValuesCutoffsPath = required "--input-values-cutoffs" args
        let outputCodesPath = required "--output-codes" args
        let outputAudioPath = tryFind "--output-audio" args

        let inputIds =
            TensorIO.readInt64s inputIdsPath (batch * textSequence)
            |> fun values -> TensorIO.denseInt64 values [| batch; textSequence |]

        let attentionMask =
            TensorIO.readInt64s attentionMaskPath (batch * textSequence)
            |> fun values -> TensorIO.denseInt64 values [| batch; textSequence |]

        let inputValues =
            TensorIO.readSingles inputValuesPath (batch * audioSamples)
            |> fun values -> TensorIO.denseFloat values [| batch; 1; audioSamples |]

        let inputValuesCutoffs =
            TensorIO.readInt64s inputValuesCutoffsPath batch
            |> fun values -> TensorIO.denseInt64 values [| batch |]

        use runner = new ChromaSharedOnnxRunner(modelDir, bundleDir)
        let audioNumCodebooks = optional (string runner.AudioNumCodebooks) "--audio-codebooks" args |> int

        printfn "Mapped safetensors shards: %d" runner.MappedShardCount
        printfn "Safetensor-backed initializers: %d (%d unique source tensors)" runner.InitializerCount runner.UniqueSourceTensorCount

        let prefill = runner.RunSystemPrefill(inputIds, attentionMask, inputValues, inputValuesCutoffs)
        let frameCodes = runner.GreedyAudioFrame(prefill, audioNumCodebooks)
        let frameCodeValues = Enumerable.ToArray(frameCodes)
        let codecCodes = DenseTensor<int64>(frameCodeValues, [| batch; audioNumCodebooks; 1 |])
        TensorIO.writeInt64s outputCodesPath codecCodes

        printfn "System prefill logits shape: %A" (prefill.Logits.Dimensions.ToArray())
        printfn "System prefill hidden states shape: %A" (prefill.HiddenStates.Dimensions.ToArray())
        printfn "Greedy audio codes shape: %A" (codecCodes.Dimensions.ToArray())
        printfn "Wrote audio codes: %s" outputCodesPath

        match outputAudioPath with
        | Some path ->
            let audio = runner.RunCodecDecode(codecCodes)
            TensorIO.writeSingles path audio
            printfn "Codec audio shape: %A" (audio.Dimensions.ToArray())
            printfn "Wrote decoded audio: %s" path
        | None -> ()

        0

    let serve args =
        Serve.run args tryFind required optional

    let s2sServe args =
        S2sServe.run args required optional

    let private s2sTuningOptions args =
        let ortMemoryProfile = optional "quality-safe" "--ort-memory-profile" args
        let optimizedModelCacheDir =
            let value = optional "" "--optimized-model-cache-dir" args
            if String.IsNullOrWhiteSpace value then None else Some(Path.GetFullPath value)
        let optimizedModelCacheFormat = optional "onnx" "--optimized-model-cache-format" args
        { MemoryProfile = ortMemoryProfile
          OptimizedModelCacheDir = optimizedModelCacheDir
          OptimizedModelCacheFormat = optimizedModelCacheFormat }

    let private createStandaloneOrtOptions (executionProvider: string) (tuningOptions: S2sOrtTuningOptions) =
        let options = new SessionOptions()
        let qualitySafe =
            tuningOptions.MemoryProfile.Trim().Equals("quality-safe", StringComparison.OrdinalIgnoreCase)
        options.GraphOptimizationLevel <- GraphOptimizationLevel.ORT_DISABLE_ALL
        options.LogSeverityLevel <- OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR
        if qualitySafe then
            options.EnableCpuMemArena <- false
            options.EnableMemoryPattern <- false
        match executionProvider.Trim().ToLowerInvariant() with
        | "cuda" ->
            if qualitySafe then
                let cudaOptions = new OrtCUDAProviderOptions()
                cudaOptions.UpdateOptions(
                    Dictionary<string, string>(
                        dict [ "device_id", "0"
                               "arena_extend_strategy", "kSameAsRequested"
                               "use_tf32", "0" ]
                    )
                )
                options.AppendExecutionProvider_CUDA(cudaOptions)
                cudaOptions.Dispose()
            else
                options.AppendExecutionProvider_CUDA(0)
        | "cpu" -> options.AppendExecutionProvider_CPU(if qualitySafe then 0 else 1)
        | value -> invalidArg "--execution-provider" $"Unsupported execution provider '{value}'. Use cuda or cpu."
        options

    let codecEncodeOnnxDebug args =
        let onnxPath = required "--onnx" args
        let inputPath = required "--input-f32" args
        let outputDir = required "--output-dir" args
        let executionProvider = optional "cuda" "--execution-provider" args
        let tuningOptions = s2sTuningOptions args
        let sampleCount = int (FileInfo(inputPath).Length / int64 sizeof<float32>)
        let inputValues =
            TensorIO.readSingles inputPath sampleCount
            |> fun values -> TensorIO.denseFloat values [| 1; 1; sampleCount |]
        let infos = Dictionary<string, DebugTensorInfo>(StringComparer.Ordinal)
        let writeFloat name (tensor: Tensor<float32>) =
            let dense = TensorIO.cloneFloatTensor tensor
            let file = $"{name}.f32"
            TensorIO.writeSingles (Path.Combine(outputDir, file)) dense
            infos[name] <- { File = file; Dtype = "f32"; Shape = dense.Dimensions.ToArray() }
        let writeInt64 name (tensor: Tensor<int64>) =
            let dense = TensorIO.cloneInt64Tensor tensor
            let file = $"{name}.i64"
            TensorIO.writeInt64s (Path.Combine(outputDir, file)) dense
            infos[name] <- { File = file; Dtype = "i64"; Shape = dense.Dimensions.ToArray() }

        Directory.CreateDirectory(outputDir) |> ignore
        use options = createStandaloneOrtOptions executionProvider tuningOptions
        use session = new InferenceSession(onnxPath, options)
        let inputs = List<NamedOnnxValue>()
        inputs.Add(NamedOnnxValue.CreateFromTensor("input_values", inputValues))
        use results = session.Run(inputs)
        for result in results do
            match result.Value with
            | :? Tensor<float32> as tensor -> writeFloat result.Name tensor
            | :? Tensor<int64> as tensor -> writeInt64 result.Name tensor
            | value -> invalidOp $"Unsupported output tensor type for {result.Name}: {value.GetType().FullName}"
        let manifest =
            {| runtime = "fsharp_onnx"
               executionProvider = executionProvider
               ortMemoryProfile = tuningOptions.MemoryProfile
               onnx = Path.GetFullPath onnxPath
               input = Path.GetFullPath inputPath
               tensors = infos |}
        let json =
            JsonSerializer.Serialize(
                manifest,
                JsonSerializerOptions(PropertyNamingPolicy = JsonNamingPolicy.CamelCase, WriteIndented = true)
            )
        File.WriteAllText(Path.Combine(outputDir, "debug_manifest.json"), json)
        printfn "%s" json
        0

    let s2sOffline args =
        let modelDir = required "--model-dir" args
        let bundleDir = required "--bundle-dir" args
        let promptText = required "--prompt-text" args
        let systemPrompt = optional "You are a helpful assistant." "--system-prompt" args
        let promptAudioPath = required "--prompt-audio-f32" args
        let userAudioPath = required "--user-audio-f32" args
        let frames = optional "8" "--frames" args |> int
        let outputDir = required "--output-dir" args
        let executionProvider = optional "cuda" "--execution-provider" args
        let memoryMode = optional "python-footprint" "--memory-mode" args
        let tuningOptions = s2sTuningOptions args
        let thinkerActiveFrames = optional "0" "--thinker-active-frames" args |> int

        if frames < 1 then
            invalidArg "--frames" "Frame count must be positive."

        Directory.CreateDirectory(outputDir) |> ignore

        let processor = ChromaNativeProcessor(modelDir, thinkerActiveFrames)
        let promptAudio = processor.ReadFloat32Pcm(promptAudioPath)
        let userAudio = processor.ReadFloat32Pcm(userAudioPath)
        use prepared = processor.Prepare(promptText, systemPrompt, promptAudio, userAudio)

        File.WriteAllText(Path.Combine(outputDir, "conversation.txt"), prepared.ConversationText)
        File.Copy(promptAudioPath, Path.Combine(outputDir, "prompt_audio_24k.f32"), true)
        File.Copy(userAudioPath, Path.Combine(outputDir, "user_audio_16k.f32"), true)

        let stopwatch = Stopwatch.StartNew()
        let memoryBefore = RuntimeMemory.current()
        use runner = new ChromaS2sOnnxRunner(modelDir, bundleDir, executionProvider, memoryMode, tuningOptions)
        let memoryAfterLoad = RuntimeMemory.current()
        use result = runner.Generate(prepared, frames)
        let memoryAfterGenerate = RuntimeMemory.current()
        stopwatch.Stop()

        let codesPath = Path.Combine(outputDir, "audio_codes.i64")
        let rawAudioPath = Path.Combine(outputDir, "audio_values.f32")
        let wavPath = Path.Combine(outputDir, "audio.wav")
        TensorIO.writeInt64s codesPath result.AudioCodes
        TensorIO.writeSingles rawAudioPath result.AudioValues
        let wavStats = Wave.writeMono16 wavPath 24000 result.AudioValues
        let details =
            {| mode = "s2s_offline"
               executionProvider = runner.Status.ExecutionProvider
               memoryMode = runner.MemoryMode
               ortMemoryProfile = runner.OrtMemoryProfile
               optimizedModelCacheEnabled = runner.OptimizedModelCacheEnabled
               optimizedModelCacheDir = runner.OptimizedModelCacheDir
               optimizedModelCacheFormat = runner.OptimizedModelCacheFormat
               loadedOrtSessions = runner.LoadedSessionNames
               warmOrtSessions = runner.WarmSessionNames
               activePagedOrtSessions = runner.ActivePagedSessionNames
               peakPrivateGb = runner.PeakPrivateGb
               peakWorkingSetGb = runner.PeakWorkingSetGb
               mappedSafetensorShards = runner.MappedShardCount
               initializerCount = runner.InitializerCount
               uniqueInitializerSources = runner.UniqueSourceTensorCount
               uniqueOrtValues = runner.UniqueOrtValueCount
               sharedPrepackedWeights = runner.SharedPrepackedWeights
               memoryBefore = memoryBefore
               memoryAfterLoad = memoryAfterLoad
               memoryAfterGenerate = memoryAfterGenerate
               promptText = promptText
               systemPrompt = systemPrompt
               requestedFrames = frames
               frameCount = result.FrameCount
               stopReason = result.StopReason
               stepKinds = result.StepKinds
               promptAudioSamples = prepared.PromptAudioSamples
               userAudioSamples = prepared.UserAudioSamples
               effectiveUserAudioSamples = min prepared.UserAudioSamples processor.ThinkerTraceSamples
               thinkerFeatureMode = processor.ThinkerFeatureMode
               thinkerConfiguredActiveFrames = processor.ConfiguredThinkerActiveFrames
               thinkerTraceFeatureFrames = processor.ThinkerTraceFeatureFrames
               audioCodesShape = result.AudioCodes.Dimensions.ToArray()
               audioValuesShape = result.AudioValues.Dimensions.ToArray()
               timingsMs = result.Timings
               wallClockMs = stopwatch.Elapsed.TotalMilliseconds
               wav = wavStats |}
        let detailsJson =
            JsonSerializer.Serialize(
                details,
                JsonSerializerOptions(PropertyNamingPolicy = JsonNamingPolicy.CamelCase, WriteIndented = true)
            )
        File.WriteAllText(Path.Combine(outputDir, "details.json"), detailsJson)

        printfn "%s" detailsJson
        0

    let s2sBenchmark args =
        let modelDir = required "--model-dir" args
        let bundleDir = required "--bundle-dir" args
        let promptText = required "--prompt-text" args
        let systemPrompt = optional "You are a helpful assistant." "--system-prompt" args
        let promptAudioPath = required "--prompt-audio-f32" args
        let userAudioPath = required "--user-audio-f32" args
        let frames = optional "8" "--frames" args |> int
        let warmup = optional "1" "--warmup" args |> int
        let iterationCount = optional "5" "--iterations" args |> int
        let outputDir = required "--output-dir" args
        let executionProvider = optional "cuda" "--execution-provider" args
        let memoryMode = optional "resident-merged" "--memory-mode" args
        let tuningOptions = s2sTuningOptions args
        let thinkerActiveFrames = optional "0" "--thinker-active-frames" args |> int

        if frames < 1 then
            invalidArg "--frames" "Frame count must be positive."

        if warmup < 0 then
            invalidArg "--warmup" "Warmup count cannot be negative."

        if iterationCount < 1 then
            invalidArg "--iterations" "Iteration count must be positive."

        Directory.CreateDirectory(outputDir) |> ignore

        let processor = ChromaNativeProcessor(modelDir, thinkerActiveFrames)
        let promptAudio = processor.ReadFloat32Pcm(promptAudioPath)
        let userAudio = processor.ReadFloat32Pcm(userAudioPath)
        use prepared = processor.Prepare(promptText, systemPrompt, promptAudio, userAudio)

        File.WriteAllText(Path.Combine(outputDir, "conversation.txt"), prepared.ConversationText)
        File.Copy(promptAudioPath, Path.Combine(outputDir, "prompt_audio_24k.f32"), true)
        File.Copy(userAudioPath, Path.Combine(outputDir, "user_audio_16k.f32"), true)

        let memoryBeforeCreate = RuntimeMemory.current()
        use runner = new ChromaS2sOnnxRunner(modelDir, bundleDir, executionProvider, memoryMode, tuningOptions)
        let memoryAfterCreate = RuntimeMemory.current()

        for warmupIndex in 1 .. warmup do
            use warmupResult = runner.Generate(prepared, frames)
            let totalMs =
                match warmupResult.Timings.TryGetValue "totalMs" with
                | true, value -> value
                | _ -> Double.NaN

            printfn "Warmup %d/%d: %.3f ms, frames=%d, stop=%s" warmupIndex warmup totalMs warmupResult.FrameCount warmupResult.StopReason

        let memoryAfterWarmup = RuntimeMemory.current()
        let mutable lastResult: S2sGenerationResult option = None
        let disposeLastResult () =
            match lastResult with
            | Some result ->
                (result :> IDisposable).Dispose()
                lastResult <- None
            | None -> ()

        let benchmarkStopwatch = Stopwatch.StartNew()
        let iterations =
            try
                [| for iterationIndex in 1 .. iterationCount do
                       let stopwatch = Stopwatch.StartNew()
                       let result = runner.Generate(prepared, frames)
                       stopwatch.Stop()
                       let previousResult = lastResult
                       lastResult <- Some result
                       previousResult
                       |> Option.iter (fun previous -> (previous :> IDisposable).Dispose())
                       yield
                           { Iteration = iterationIndex
                             WallClockMs = stopwatch.Elapsed.TotalMilliseconds
                             TimingsMs = Dictionary<string, float>(result.Timings)
                             FrameCount = result.FrameCount
                             StopReason = result.StopReason
                             AudioCodesShape = result.AudioCodes.Dimensions.ToArray()
                             AudioValuesShape = result.AudioValues.Dimensions.ToArray() } |]
            with
            | _ ->
                disposeLastResult ()
                reraise()
        benchmarkStopwatch.Stop()
        let memoryAfterBenchmark = RuntimeMemory.current()

        try
            match lastResult with
            | Some result ->
                TensorIO.writeInt64s (Path.Combine(outputDir, "last_audio_codes.i64")) result.AudioCodes
                TensorIO.writeSingles (Path.Combine(outputDir, "last_audio_values.f32")) result.AudioValues
                Wave.writeMono16 (Path.Combine(outputDir, "last_audio.wav")) 24000 result.AudioValues |> ignore
            | None -> ()
        finally
            disposeLastResult ()

        let details =
            {| backend = "fsharp_onnx"
               mode = "s2s_benchmark"
               generationOnly = true
               loadTimeExcluded = true
               warmupExcluded = true
               executionProvider = runner.Status.ExecutionProvider
               memoryMode = runner.MemoryMode
               ortMemoryProfile = runner.OrtMemoryProfile
               optimizedModelCacheEnabled = runner.OptimizedModelCacheEnabled
               optimizedModelCacheDir = runner.OptimizedModelCacheDir
               optimizedModelCacheFormat = runner.OptimizedModelCacheFormat
               loadedOrtSessions = runner.LoadedSessionNames
               warmOrtSessions = runner.WarmSessionNames
               activePagedOrtSessions = runner.ActivePagedSessionNames
               mappedSafetensorShards = runner.MappedShardCount
               initializerCount = runner.InitializerCount
               uniqueInitializerSources = runner.UniqueSourceTensorCount
               uniqueOrtValues = runner.UniqueOrtValueCount
               sharedPrepackedWeights = runner.SharedPrepackedWeights
               promptText = promptText
               systemPrompt = systemPrompt
               requestedFrames = frames
               warmupIterations = warmup
               measuredIterations = iterationCount
               promptAudioSamples = prepared.PromptAudioSamples
               userAudioSamples = prepared.UserAudioSamples
               effectiveUserAudioSamples = min prepared.UserAudioSamples processor.ThinkerTraceSamples
               thinkerFeatureMode = processor.ThinkerFeatureMode
               thinkerConfiguredActiveFrames = processor.ConfiguredThinkerActiveFrames
               thinkerTraceFeatureFrames = processor.ThinkerTraceFeatureFrames
               summary =
                   {| wallClockMs = iterations |> Array.map (fun item -> item.WallClockMs) |> summarizeMilliseconds
                      totalMs = summarizeTimingKey "totalMs" iterations
                      prefillMs = summarizeTimingKey "prefillMs" iterations
                      generateMs = summarizeTimingKey "generateMs" iterations
                      decodeMs = summarizeTimingKey "decodeMs" iterations |}
               benchmarkWallClockMs = benchmarkStopwatch.Elapsed.TotalMilliseconds
               iterations =
                   iterations
                   |> Array.map (fun item ->
                       {| iteration = item.Iteration
                          wallClockMs = item.WallClockMs
                          timingsMs = item.TimingsMs
                          frameCount = item.FrameCount
                          stopReason = item.StopReason
                          audioCodesShape = item.AudioCodesShape
                          audioValuesShape = item.AudioValuesShape |})
               memory =
                   {| beforeCreate = memoryBeforeCreate
                      afterCreate = memoryAfterCreate
                      afterWarmup = memoryAfterWarmup
                      afterBenchmark = memoryAfterBenchmark
                      peakPrivateGb = runner.PeakPrivateGb
                      peakWorkingSetGb = runner.PeakWorkingSetGb |}
               artifacts =
                   {| lastAudioCodes = Path.Combine(outputDir, "last_audio_codes.i64")
                      lastAudioValues = Path.Combine(outputDir, "last_audio_values.f32")
                      lastAudioWav = Path.Combine(outputDir, "last_audio.wav") |} |}

        let detailsJson = JsonSerializer.Serialize(details, cliJsonOptions)
        File.WriteAllText(Path.Combine(outputDir, "benchmark.json"), detailsJson)
        printfn "%s" detailsJson
        0

    let private readPreparedTensors (preparedDir: string) =
        let manifestPath = Path.Combine(preparedDir, "manifest.json")
        if not (File.Exists manifestPath) then
            invalidArg "--prepared-dir" $"Prepared tensor manifest was not found: {manifestPath}"

        use doc = JsonDocument.Parse(File.ReadAllText(manifestPath))
        let tensors = doc.RootElement.GetProperty("tensors")

        let tensorInfo (name: string) =
            let element = tensors.GetProperty(name)
            let file =
                match element.GetProperty("file").GetString() with
                | null -> invalidArg name $"Missing file for prepared tensor {name}."
                | value -> value
            let shape =
                element.GetProperty("shape").EnumerateArray()
                |> Seq.map (fun item -> item.GetInt32())
                |> Seq.toArray
            file, shape

        let countOf (shape: int array) =
            shape |> Array.fold (fun total value -> total * value) 1

        let readI64 (name: string) =
            let file, shape = tensorInfo name
            TensorIO.readInt64s (Path.Combine(preparedDir, file)) (countOf shape)
            |> fun values -> TensorIO.denseInt64 values shape

        let readF32 (name: string) =
            let file, shape = tensorInfo name
            TensorIO.readSingles (Path.Combine(preparedDir, file)) (countOf shape)
            |> fun values -> TensorIO.denseFloat values shape

        let inputValues = readF32 "input_values"
        let inputValuesShape = inputValues.Dimensions.ToArray()
        let promptAudioSamples =
            if inputValuesShape.Length >= 3 then inputValuesShape[2] else 0

        new NativeS2sPrepared(
            readI64 "input_ids",
            readI64 "attention_mask",
            inputValues,
            readI64 "input_values_cutoffs",
            readI64 "thinker_input_ids",
            readI64 "thinker_attention_mask",
            readF32 "thinker_input_features",
            readI64 "thinker_feature_attention_mask",
            promptAudioSamples,
            0,
            ""
        )

    let private writeS2sOnnxDebug modelDir bundleDir preparedDir outputDir executionProvider memoryMode tuningOptions frames =
        use prepared = readPreparedTensors preparedDir

        use runner = new ChromaS2sOnnxRunner(modelDir, bundleDir, executionProvider, memoryMode, tuningOptions)
        runner.WriteDebug(prepared, outputDir, frames)

    let private writePreparedDebug outputDir (prepared: NativeS2sPrepared) =
        Directory.CreateDirectory(outputDir) |> ignore
        let infos = Dictionary<string, DebugTensorInfo>(StringComparer.Ordinal)
        let writeF32 name (tensor: DenseTensor<float32>) =
            let file = $"{name}.f32"
            TensorIO.writeSingles (Path.Combine(outputDir, file)) tensor
            infos[name] <- { File = file; Dtype = "f32"; Shape = tensor.Dimensions.ToArray() }
        let writeI64 name (tensor: DenseTensor<int64>) =
            let file = $"{name}.i64"
            TensorIO.writeInt64s (Path.Combine(outputDir, file)) tensor
            infos[name] <- { File = file; Dtype = "i64"; Shape = tensor.Dimensions.ToArray() }

        writeI64 "input_ids" prepared.InputIds
        writeI64 "attention_mask" prepared.AttentionMask
        writeF32 "input_values" prepared.InputValues
        writeI64 "input_values_cutoffs" prepared.InputValuesCutoffs
        writeI64 "thinker_input_ids" prepared.ThinkerInputIds
        writeI64 "thinker_attention_mask" prepared.ThinkerAttentionMask
        writeF32 "thinker_input_features" prepared.ThinkerInputFeatures
        writeI64 "thinker_feature_attention_mask" prepared.ThinkerFeatureAttentionMask

        let manifestJson =
            JsonSerializer.Serialize(
                {| runtime = "fsharp_native_prepared"
                   conversationText = prepared.ConversationText
                   tensors = infos |},
                JsonSerializerOptions(PropertyNamingPolicy = JsonNamingPolicy.CamelCase, WriteIndented = true)
            )
        File.WriteAllText(Path.Combine(outputDir, "debug_manifest.json"), manifestJson)

    let s2sDebugOnnx args =
        let modelDir = required "--model-dir" args
        let bundleDir = required "--bundle-dir" args
        let preparedDir = required "--prepared-dir" args
        let outputDir = required "--output-dir" args
        let executionProvider = optional "cuda" "--execution-provider" args
        let memoryMode = optional "python-footprint" "--memory-mode" args
        let frames = optional "8" "--frames" args |> int
        let tuningOptions = s2sTuningOptions args
        writeS2sOnnxDebug modelDir bundleDir preparedDir outputDir executionProvider memoryMode tuningOptions frames
        printfn "Wrote F#/ONNX debug tensors to %s" outputDir
        0

    let private runCheckedProcess fileName arguments =
        let result =
            ProcessRunner.run fileName arguments (Directory.GetCurrentDirectory())
            |> fun task -> task.GetAwaiter().GetResult()
        if not (String.IsNullOrWhiteSpace result.Stdout) then
            printfn "%s" result.Stdout
        if not (String.IsNullOrWhiteSpace result.Stderr) then
            eprintfn "%s" result.Stderr
        if result.ExitCode <> 0 then
            invalidOp $"{fileName} exited with code {result.ExitCode}."
        result

    let private readJsonElementOrObject path fallback =
        if File.Exists path then
            use doc = JsonDocument.Parse(File.ReadAllText(path))
            doc.RootElement.Clone()
        else
            JsonSerializer.SerializeToElement(fallback, cliJsonOptions)

    let private maxPrivateGb (snapshots: RuntimeMemory.Snapshot array) =
        snapshots
        |> Array.map (fun snapshot -> snapshot.PrivateGb)
        |> Array.max

    let private maxWorkingSetGb (snapshots: RuntimeMemory.Snapshot array) =
        snapshots
        |> Array.map (fun snapshot -> snapshot.WorkingSetGb)
        |> Array.max

    let private tryNestedFloat (root: JsonElement) (path: string array) =
        let mutable current = root
        let mutable found = true
        for name in path do
            if found then
                match current.TryGetProperty(name) with
                | true, value -> current <- value
                | false, _ -> found <- false

        if found && current.ValueKind = JsonValueKind.Number then
            match current.TryGetDouble() with
            | true, value -> Some value
            | false, _ -> None
        else
            None

    let private ratioOrNull numerator denominator =
        match numerator, denominator with
        | Some n, Some d when d > 0.0 -> Nullable(Math.Round(n / d, 3))
        | _ -> Nullable<float>()

    let s2sMemoryReport args =
        let modelDir = required "--model-dir" args
        let bundleDir = required "--bundle-dir" args
        let promptText = required "--prompt-text" args
        let systemPrompt = optional "You are a helpful assistant." "--system-prompt" args
        let promptAudioPath = required "--prompt-audio-f32" args
        let userAudioPath = required "--user-audio-f32" args
        let frames = optional "8" "--frames" args |> int
        let outputDir = required "--output-dir" args
        let python = optional ".venv/Scripts/python.exe" "--python" args
        let executionProvider = optional "cuda" "--execution-provider" args
        let memoryMode = optional "python-footprint" "--memory-mode" args
        let tuningOptions = s2sTuningOptions args
        let pythonDevice = optional (if executionProvider.Equals("cuda", StringComparison.OrdinalIgnoreCase) then "cuda" else "cpu") "--python-device" args
        let thinkerActiveFrames = optional "0" "--thinker-active-frames" args
        let thinkerActiveFrameCount = int thinkerActiveFrames
        let pythonDir = Path.Combine(outputDir, "python")
        let fsharpDir = Path.Combine(outputDir, "fsharp_onnx")
        let scriptPath = Path.Combine(Directory.GetCurrentDirectory(), "scripts", "chroma_memory_probe.py")

        if frames < 1 then
            invalidArg "--frames" "Frame count must be positive."

        Directory.CreateDirectory(outputDir) |> ignore
        Directory.CreateDirectory(pythonDir) |> ignore
        Directory.CreateDirectory(fsharpDir) |> ignore

        printfn "Running persistent Python Chroma memory baseline..."
        let pythonArgs =
            [ scriptPath
              "--model-dir"
              modelDir
              "--prompt-text"
              promptText
              "--system-prompt"
              systemPrompt
              "--prompt-audio-f32"
              promptAudioPath
              "--user-audio-f32"
              userAudioPath
              "--output-dir"
              pythonDir
              "--max-new-frames"
              string frames
              "--device"
              pythonDevice
              "--thinker-active-frames"
              thinkerActiveFrames ]
        let pythonResult =
            ProcessRunner.run python pythonArgs (Directory.GetCurrentDirectory())
            |> fun task -> task.GetAwaiter().GetResult()

        File.WriteAllText(Path.Combine(pythonDir, "stdout.json"), pythonResult.Stdout)
        File.WriteAllText(Path.Combine(pythonDir, "stderr.txt"), pythonResult.Stderr)
        if pythonResult.ExitCode <> 0 then
            invalidOp $"Python Chroma memory probe failed with exit code {pythonResult.ExitCode}: {pythonResult.Stderr}"

        let pythonReport =
            readJsonElementOrObject
                (Path.Combine(pythonDir, "memory_report.json"))
                {| stdout = pythonResult.Stdout; stderr = pythonResult.Stderr |}

        printfn "Running F#/ONNX memory probe (%s)..." memoryMode
        let processor = ChromaNativeProcessor(modelDir, thinkerActiveFrameCount)
        let promptAudio = processor.ReadFloat32Pcm(promptAudioPath)
        let userAudio = processor.ReadFloat32Pcm(userAudioPath)
        use prepared = processor.Prepare(promptText, systemPrompt, promptAudio, userAudio)
        File.Copy(promptAudioPath, Path.Combine(fsharpDir, "prompt_audio_24k.f32"), true)
        File.Copy(userAudioPath, Path.Combine(fsharpDir, "user_audio_16k.f32"), true)
        File.WriteAllText(Path.Combine(fsharpDir, "conversation.txt"), prepared.ConversationText)

        let beforeCreate = RuntimeMemory.current()
        let fsharpReport, afterDispose =
            let reportElement =
                use runner = new ChromaS2sOnnxRunner(modelDir, bundleDir, executionProvider, memoryMode, tuningOptions)
                let afterRunnerCreate = RuntimeMemory.current()
                let stopwatch = Stopwatch.StartNew()
                use result = runner.Generate(prepared, frames)
                stopwatch.Stop()
                let afterGenerate = RuntimeMemory.current()

                let codesPath = Path.Combine(fsharpDir, "audio_codes.i64")
                let rawAudioPath = Path.Combine(fsharpDir, "audio_values.f32")
                let wavPath = Path.Combine(fsharpDir, "audio.wav")
                TensorIO.writeInt64s codesPath result.AudioCodes
                TensorIO.writeSingles rawAudioPath result.AudioValues
                let wavStats = Wave.writeMono16 wavPath 24000 result.AudioValues
                let snapshots = [| beforeCreate; afterRunnerCreate; afterGenerate |]
                let details =
                    {| backend = "fsharp_onnx"
                       mode = "s2s_memory_probe"
                       executionProvider = runner.Status.ExecutionProvider
                       memoryMode = runner.MemoryMode
                       ortMemoryProfile = runner.OrtMemoryProfile
                       optimizedModelCacheEnabled = runner.OptimizedModelCacheEnabled
                       optimizedModelCacheDir = runner.OptimizedModelCacheDir
                       optimizedModelCacheFormat = runner.OptimizedModelCacheFormat
                       requestedFrames = frames
                       frameCount = result.FrameCount
                       stopReason = result.StopReason
                       stepKinds = result.StepKinds
                       loadedOrtSessions = runner.LoadedSessionNames
                       warmOrtSessions = runner.WarmSessionNames
                       activePagedOrtSessions = runner.ActivePagedSessionNames
                       peakPrivateGb = runner.PeakPrivateGb
                       peakWorkingSetGb = runner.PeakWorkingSetGb
                       mappedSafetensorShards = runner.MappedShardCount
                       initializerCount = runner.InitializerCount
                       uniqueInitializerSources = runner.UniqueSourceTensorCount
                       uniqueOrtValues = runner.UniqueOrtValueCount
                       sharedPrepackedWeights = runner.SharedPrepackedWeights
                       promptAudioSamples = prepared.PromptAudioSamples
                       userAudioSamples = prepared.UserAudioSamples
                       effectiveUserAudioSamples = min prepared.UserAudioSamples processor.ThinkerTraceSamples
                       thinkerFeatureMode = processor.ThinkerFeatureMode
                       thinkerConfiguredActiveFrames = processor.ConfiguredThinkerActiveFrames
                       thinkerTraceFeatureFrames = processor.ThinkerTraceFeatureFrames
                       audioCodesShape = result.AudioCodes.Dimensions.ToArray()
                       audioValuesShape = result.AudioValues.Dimensions.ToArray()
                       timingsMs = result.Timings
                       wallClockMs = stopwatch.Elapsed.TotalMilliseconds
                       memory =
                           {| beforeCreate = beforeCreate
                              afterRunnerCreate = afterRunnerCreate
                              afterGenerate = afterGenerate
                              peakPrivateGb = max runner.PeakPrivateGb (maxPrivateGb snapshots)
                              peakWorkingSetGb = max runner.PeakWorkingSetGb (maxWorkingSetGb snapshots) |}
                       wav = wavStats |}
                let detailsJson = JsonSerializer.Serialize(details, cliJsonOptions)
                File.WriteAllText(Path.Combine(fsharpDir, "memory_report.json"), detailsJson)
                JsonSerializer.SerializeToElement(details, cliJsonOptions)

            GC.Collect()
            GC.WaitForPendingFinalizers()
            GC.Collect()
            reportElement, RuntimeMemory.current()

        let pythonPeakPrivate = tryNestedFloat pythonReport [| "memory"; "peakPrivateGb" |]
        let pythonSteadyPrivate = tryNestedFloat pythonReport [| "memory"; "afterGenerate"; "privateGb" |]
        let fsharpPeakPrivate = tryNestedFloat fsharpReport [| "memory"; "peakPrivateGb" |]
        let fsharpSteadyPrivate = Some afterDispose.PrivateGb
        let peakRatio = ratioOrNull fsharpPeakPrivate pythonPeakPrivate
        let steadyRatio = ratioOrNull fsharpSteadyPrivate pythonSteadyPrivate
        let acceptance =
            {| peakPrivateRatio = peakRatio
               steadyPrivateRatio = steadyRatio
               peakWithin25Percent =
                   if peakRatio.HasValue then Nullable(peakRatio.Value <= 1.25) else Nullable<bool>()
               steadyWithin15Percent =
                   if steadyRatio.HasValue then Nullable(steadyRatio.Value <= 1.15) else Nullable<bool>()
               note = "Ratios compare sampled private memory. Python is kept loaded through generation; F#/ONNX after-dispose is measured after paging sessions out." |}

        let report =
            {| mode = "s2s_memory_report"
               modelDir = modelDir
               bundleDir = bundleDir
               frames = frames
               executionProvider = executionProvider
               memoryMode = memoryMode
               ortMemoryProfile = tuningOptions.MemoryProfile
               optimizedModelCacheDir = tuningOptions.OptimizedModelCacheDir
               pythonDevice = pythonDevice
               python = pythonReport
               fsharpOnnx = fsharpReport
               fsharpMemoryAfterDispose = afterDispose
               acceptance = acceptance |}
        let reportJson = JsonSerializer.Serialize(report, cliJsonOptions)
        File.WriteAllText(Path.Combine(outputDir, "memory_report.json"), reportJson)
        printfn "%s" reportJson
        0

    let s2sCompare args =
        let modelDir = required "--model-dir" args
        let bundleDir = required "--bundle-dir" args
        let promptText = required "--prompt-text" args
        let systemPrompt = optional "You are a helpful assistant." "--system-prompt" args
        let promptAudioPath = required "--prompt-audio" args
        let userAudioPath = required "--user-audio" args
        let outputDir = required "--output-dir" args
        let python = optional ".venv/Scripts/python.exe" "--python" args
        let executionProvider = optional "cuda" "--execution-provider" args
        let memoryMode = optional "python-footprint" "--memory-mode" args
        let tuningOptions = s2sTuningOptions args
        let frames = optional "8" "--frames" args |> int
        let thinkerActiveFrames = optional "0" "--thinker-active-frames" args
        let thinkerActiveFrameCount = int thinkerActiveFrames
        let pythonDevice = if executionProvider.Equals("cuda", StringComparison.OrdinalIgnoreCase) then "cuda" else "cpu"
        let pythonTraceDir = Path.Combine(outputDir, "python_trace")
        let onnxDebugDir = Path.Combine(outputDir, "onnx_debug")
        let scriptPath = Path.Combine(Directory.GetCurrentDirectory(), "scripts", "chroma_s2s_step_debug.py")

        Directory.CreateDirectory(outputDir) |> ignore

        printfn "Running Python Chroma oracle trace..."
        runCheckedProcess
            python
            [ scriptPath
              "--model-dir"
              modelDir
              "--prompt-text"
              promptText
              "--system-prompt"
              systemPrompt
              "--prompt-audio"
              promptAudioPath
              "--user-audio"
              userAudioPath
              "--output-dir"
              pythonTraceDir
              "--device"
              pythonDevice
              "--frames"
              string frames
              "--thinker-active-frames"
              thinkerActiveFrames ]
        |> ignore

        printfn "Comparing native F# preprocessing against Python AutoProcessor..."
        let rawInputDir = Path.Combine(pythonTraceDir, "raw_inputs")
        let nativeProcessor = ChromaNativeProcessor(modelDir, thinkerActiveFrameCount)
        let nativePromptAudio = nativeProcessor.ReadFloat32Pcm(Path.Combine(rawInputDir, "prompt_audio_24k.f32"))
        let nativeUserAudio = nativeProcessor.ReadFloat32Pcm(Path.Combine(rawInputDir, "user_audio_16k.f32"))
        use nativePrepared = nativeProcessor.Prepare(promptText, systemPrompt, nativePromptAudio, nativeUserAudio)
        let nativePreparedDir = Path.Combine(outputDir, "fsharp_prepared")
        writePreparedDebug nativePreparedDir nativePrepared
        runCheckedProcess
            python
            [ scriptPath
              "--compare-only"
              "--python-debug-dir"
              Path.Combine(pythonTraceDir, "prepared")
              "--python-manifest"
              "manifest.json"
              "--onnx-debug-dir"
              nativePreparedDir
              "--onnx-manifest"
              "debug_manifest.json" ]
        |> ignore

        printfn "Running F#/ONNX trace from the exact Python-prepared tensors..."
        writeS2sOnnxDebug modelDir bundleDir (Path.Combine(pythonTraceDir, "prepared")) onnxDebugDir executionProvider memoryMode tuningOptions frames

        printfn "Comparing step-by-step tensors..."
        runCheckedProcess
            python
            [ scriptPath
              "--compare-only"
              "--python-debug-dir"
              Path.Combine(pythonTraceDir, "python")
              "--onnx-debug-dir"
              onnxDebugDir ]
        |> ignore

        printfn "Wrote parity artifacts to %s" outputDir
        0

