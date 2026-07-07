namespace ChromaOnnx.SpeechToSpeech

open System
open System.Diagnostics
open System.Globalization
open System.IO
open System.Text
open System.Threading
open System.Threading.Tasks
open ChromaOnnx

module private TtsWav =
    let private invalidData message = raise (InvalidDataException(message))

    let readMono16 (path: string) =
        use stream = File.OpenRead(path)
        use reader = new BinaryReader(stream, Encoding.ASCII, leaveOpen = false)
        let ascii count = Encoding.ASCII.GetString(reader.ReadBytes count)
        if ascii 4 <> "RIFF" then invalidData "WAV file is missing RIFF header."
        reader.ReadUInt32() |> ignore
        if ascii 4 <> "WAVE" then invalidData "WAV file is missing WAVE header."

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
                let audioFormat = reader.ReadUInt16()
                channels <- reader.ReadUInt16()
                sampleRate <- reader.ReadInt32()
                reader.ReadUInt32() |> ignore
                reader.ReadUInt16() |> ignore
                bitsPerSample <- reader.ReadUInt16()
                if audioFormat <> 1us then invalidData "Only PCM WAV files are supported."
                stream.Position <- chunkStart + int64 chunkSize
            | "data" ->
                if channels = 0us || bitsPerSample <> 16us then
                    invalidData "Only 16-bit PCM WAV data is supported."
                let raw = reader.ReadBytes(int chunkSize)
                let values = Array.zeroCreate<int16> (raw.Length / sizeof<int16>)
                Buffer.BlockCopy(raw, 0, values, 0, raw.Length)
                let frameCount = values.Length / int channels
                samples <- Array.zeroCreate<float32> frameCount
                for frame in 0 .. frameCount - 1 do
                    let mutable acc = 0.0f
                    for channel in 0 .. int channels - 1 do
                        acc <- acc + float32 values[frame * int channels + channel] / 32768.0f
                    samples[frame] <- acc / float32 channels
                foundData <- true
            | _ ->
                stream.Position <- chunkStart + int64 chunkSize
            if chunkSize % 2u = 1u && stream.Position < stream.Length then
                stream.Position <- stream.Position + 1L

        if not foundData then invalidData "WAV file has no data chunk."
        sampleRate, samples

    let writeMono16 (path: string) (sampleRate: int) (samples: float32 array) =
        let tensor = Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float32>(samples, [| 1; 1; samples.Length |])
        Wave.writeMono16 path sampleRate tensor |> ignore

type FakeToneTtsRuntime(options: TtsRuntimeOptions, ?pathBase: string) =
    let outputSampleRate = max 8000 options.OutputSampleRate
    let chunkSeconds = max 0.05 options.StreamingChunkSeconds

    interface ITtsRuntime with
        member _.Status() =
            { Ready = true
              SupportsVoiceCloning = not (String.IsNullOrWhiteSpace options.VoiceSamplePath)
              SupportsStreaming = true
              Runtime = "fake-tone"
              ModelDir = options.ModelDir
              ExecutionProvider = "cpu"
              OutputSampleRate = outputSampleRate
              VoiceSamplePath = options.VoiceSamplePath
              MissingFiles = Array.empty
              Message = "Fake tone TTS runtime is ready for tests." }

        member _.SynthesizeAsync(request, emitChunk, cancellationToken) =
            task {
                cancellationToken.ThrowIfCancellationRequested()
                Directory.CreateDirectory(request.OutputDirectory) |> ignore
                let durationSeconds = max 0.25 (min 2.0 (float request.Text.Length / 40.0))
                let count = int (Math.Ceiling(durationSeconds * float outputSampleRate))
                let samples = Array.zeroCreate<float32> count
                let frequency =
                    if String.Equals(request.Phase, "filler", StringComparison.OrdinalIgnoreCase) then 330.0
                    else 440.0
                for index in 0 .. count - 1 do
                    samples[index] <- float32 (Math.Sin(2.0 * Math.PI * frequency * float index / float outputSampleRate) * 0.12)

                let outputPath = Path.Combine(request.OutputDirectory, request.OutputFileName)
                TtsWav.writeMono16 outputPath outputSampleRate samples
                let chunkSize = max 1 (int (float outputSampleRate * chunkSeconds))
                let stopwatch = Stopwatch.StartNew()
                let mutable offset = 0
                while offset < samples.Length do
                    cancellationToken.ThrowIfCancellationRequested()
                    let length = min chunkSize (samples.Length - offset)
                    let chunk = Array.zeroCreate<float32> length
                    Array.Copy(samples, offset, chunk, 0, length)
                    do! emitChunk chunk
                    offset <- offset + length
                stopwatch.Stop()
                return
                    { Phase = request.Phase
                      Text = request.Text
                      OutputPath = Some outputPath
                      SampleRate = outputSampleRate
                      Samples = samples.Length
                      DurationMs = float samples.Length / float outputSampleRate * 1000.0
                      InferenceTimeMs = stopwatch.Elapsed.TotalMilliseconds
                      Message = "Fake tone TTS synthesis completed." }
            }

type VoxCpm2CliTtsRuntime(options: TtsRuntimeOptions, ?pathBase: string) =
    let pathBase = defaultArg pathBase (Directory.GetCurrentDirectory())
    let fullPath path = S2sRuntimePaths.resolveAgainst pathBase path
    let runtimeName =
        if String.IsNullOrWhiteSpace options.Runtime then "voxcpm2-cli"
        else options.Runtime.Trim().ToLowerInvariant()
    let variant =
        if String.IsNullOrWhiteSpace options.Variant then "onnx"
        else options.Variant.Trim().ToLowerInvariant()
    let modelDir =
        if String.IsNullOrWhiteSpace options.ModelDir then ""
        else fullPath options.ModelDir
    let executable =
        if String.IsNullOrWhiteSpace options.ExecutablePath then "speech_voxcpm2_clone_onnx"
        else options.ExecutablePath.Trim()
    let resolvedExecutable =
        let hasDirectory = executable.Contains(Path.DirectorySeparatorChar) || executable.Contains(Path.AltDirectorySeparatorChar)
        if Path.IsPathRooted executable || hasDirectory then fullPath executable else executable
    let outputSampleRate = max 8000 options.OutputSampleRate
    let chunkSeconds = max 0.05 options.StreamingChunkSeconds
    let maxSteps = max 8 options.MaxSteps
    let seed = options.Seed
    let wantsCuda =
        String.Equals(options.ExecutionProvider, "cuda", StringComparison.OrdinalIgnoreCase)

    let searchPathExecutable name =
        if File.Exists name then Some(Path.GetFullPath name)
        elif name.Contains(Path.DirectorySeparatorChar) || name.Contains(Path.AltDirectorySeparatorChar) then None
        else
            let pathValue = Environment.GetEnvironmentVariable("PATH")
            let extensions =
                if OperatingSystem.IsWindows() then
                    let pathext = Environment.GetEnvironmentVariable("PATHEXT")
                    if String.IsNullOrWhiteSpace pathext then [| ".exe"; ".cmd"; ".bat"; "" |]
                    else pathext.Split(';', StringSplitOptions.RemoveEmptyEntries) |> Array.append [| "" |]
                else
                    [| "" |]
            if String.IsNullOrWhiteSpace pathValue then
                None
            else
                pathValue.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries)
                |> Array.tryPick (fun dir ->
                    extensions
                    |> Array.tryPick (fun ext ->
                        let candidate = Path.Combine(dir, name + ext)
                        if File.Exists candidate then Some candidate else None))

    let executableExists () =
        match searchPathExecutable resolvedExecutable with
        | Some _ -> true
        | None -> false

    let modelFiles () =
        if String.IsNullOrWhiteSpace modelDir then
            Array.empty<string>
        elif variant.Contains("litert") || resolvedExecutable.Contains("litert", StringComparison.OrdinalIgnoreCase) then
            [| "voxcpm2-text-prefill.tflite"
               "voxcpm2-token-step.tflite"
               "voxcpm2-audio-encoder.tflite"
               "voxcpm2-audio-decoder.tflite"
               "tokenizer.json" |]
        else
            [| "voxcpm2-decoder.onnx"
               "voxcpm2-decoder.onnx.data"
               "voxcpm2-audio-encoder.onnx"
               "voxcpm2-audio-decoder.onnx"
               "tokenizer.json" |]

    let missingFiles () =
        let missing = ResizeArray<string>()
        if not (executableExists ()) then missing.Add(resolvedExecutable)
        if not (String.IsNullOrWhiteSpace modelDir) then
            if not (Directory.Exists modelDir) then
                missing.Add(modelDir)
            else
                for fileName in modelFiles () do
                    let path = Path.Combine(modelDir, fileName)
                    if not (File.Exists path) then missing.Add(path)
        if not (String.IsNullOrWhiteSpace options.VoiceSamplePath) then
            let voicePath = fullPath options.VoiceSamplePath
            if not (File.Exists voicePath) then missing.Add(voicePath)
        missing.ToArray()

    let configuredVoiceSample () =
        if String.IsNullOrWhiteSpace options.VoiceSamplePath then None
        else Some(fullPath options.VoiceSamplePath)

    let configuredVoiceTranscript () =
        if String.IsNullOrWhiteSpace options.VoiceSampleTranscript then None
        else Some options.VoiceSampleTranscript

    let runProcess (arguments: string array) (cancellationToken: CancellationToken) =
        task {
            use proc = new Process()
            proc.StartInfo.FileName <- resolvedExecutable
            proc.StartInfo.UseShellExecute <- false
            proc.StartInfo.RedirectStandardOutput <- true
            proc.StartInfo.RedirectStandardError <- true
            proc.StartInfo.CreateNoWindow <- true
            if wantsCuda then
                proc.StartInfo.Environment["SPEECH_CORE_ORT_PROVIDER"] <- "cuda"
                proc.StartInfo.Environment["SPEECH_CORE_CUDA_DEVICE_ID"] <- (max 0 options.CudaDeviceId).ToString(CultureInfo.InvariantCulture)
                proc.StartInfo.Environment["SPEECH_CORE_DEVICE_INITIALIZERS"] <- "1"
                if options.RequireGpu then
                    proc.StartInfo.Environment["SPEECH_CUDA_REQUIRE_GPU"] <- "1"
                    proc.StartInfo.Environment["SPEECH_CORE_REQUIRE_GPU"] <- "1"
                if options.RequireFullGpu then
                    proc.StartInfo.Environment["SPEECH_CORE_REQUIRE_FULL_CUDA"] <- "1"
                if options.GpuMemoryLimitGb > 0.0 then
                    proc.StartInfo.Environment["SPEECH_CORE_GPU_MEM_LIMIT_GB"] <- options.GpuMemoryLimitGb.ToString(CultureInfo.InvariantCulture)
            for argument in arguments do
                proc.StartInfo.ArgumentList.Add argument
            let stdout = StringBuilder()
            let stderr = StringBuilder()
            proc.OutputDataReceived.Add(fun args -> if not (isNull args.Data) then stdout.AppendLine(args.Data) |> ignore)
            proc.ErrorDataReceived.Add(fun args -> if not (isNull args.Data) then stderr.AppendLine(args.Data) |> ignore)
            if not (proc.Start()) then invalidOp "Failed to start VoxCPM2 TTS process."
            proc.BeginOutputReadLine()
            proc.BeginErrorReadLine()
            use _registration =
                cancellationToken.Register(fun () ->
                    try
                        if not proc.HasExited then proc.Kill(entireProcessTree = true)
                    with _ -> ())
            do! proc.WaitForExitAsync(cancellationToken)
            if proc.ExitCode <> 0 then
                invalidOp $"VoxCPM2 TTS process failed with exit code {proc.ExitCode}. {stderr}"
            return stdout.ToString(), stderr.ToString()
        }

    interface ITtsRuntime with
        member _.Status() =
            let missing = missingFiles ()
            let voiceSample = configuredVoiceSample () |> Option.defaultValue ""
            { Ready = missing.Length = 0
              SupportsVoiceCloning = true
              SupportsStreaming = false
              Runtime = runtimeName
              ModelDir = modelDir
              ExecutionProvider = options.ExecutionProvider
              OutputSampleRate = outputSampleRate
              VoiceSamplePath = voiceSample
              MissingFiles = missing
              Message =
                if missing.Length = 0 then
                    "VoxCPM2 CLI TTS runtime is ready."
                else
                    let missingText = String.Join(", ", missing)
                    $"VoxCPM2 CLI TTS runtime is not ready. Missing: {missingText}" }

        member this.SynthesizeAsync(request, emitChunk, cancellationToken) =
            task {
                let status = (this :> ITtsRuntime).Status()
                if not status.Ready then invalidOp status.Message
                let text =
                    if String.IsNullOrWhiteSpace request.Text then "."
                    else request.Text.Trim()
                Directory.CreateDirectory(request.OutputDirectory) |> ignore
                let outputPath = Path.Combine(request.OutputDirectory, request.OutputFileName)
                let refPath =
                    request.VoiceSamplePath
                    |> Option.orElse (configuredVoiceSample ())
                    |> Option.defaultValue "none"
                let instruction =
                    if String.IsNullOrWhiteSpace options.Instruction then ""
                    else options.Instruction.Trim()
                let args = ResizeArray<string>()
                if not (String.IsNullOrWhiteSpace modelDir) then args.Add modelDir
                args.Add refPath
                args.Add text
                args.Add outputPath
                args.Add instruction
                args.Add(maxSteps.ToString(CultureInfo.InvariantCulture))
                if seed > 0 then args.Add(seed.ToString(CultureInfo.InvariantCulture))

                let stopwatch = Stopwatch.StartNew()
                let! stdout, stderr = runProcess (args.ToArray()) cancellationToken
                stopwatch.Stop()
                let cudaReported =
                    stderr.Contains("CUDA EP appended", StringComparison.OrdinalIgnoreCase)
                    || stdout.Contains("CUDA EP appended", StringComparison.OrdinalIgnoreCase)
                if wantsCuda && options.RequireGpu && not cudaReported then
                    invalidOp "VoxCPM2 CUDA was required but the CLI did not report CUDA EP usage."
                let sampleRate, samples = TtsWav.readMono16 outputPath
                let chunkSize = max 1 (int (float sampleRate * chunkSeconds))
                let mutable offset = 0
                while offset < samples.Length do
                    cancellationToken.ThrowIfCancellationRequested()
                    let length = min chunkSize (samples.Length - offset)
                    let chunk = Array.zeroCreate<float32> length
                    Array.Copy(samples, offset, chunk, 0, length)
                    do! emitChunk chunk
                    offset <- offset + length
                let transcriptNote =
                    match request.VoiceSampleTranscript |> Option.orElse (configuredVoiceTranscript ()) with
                    | Some _ -> " Voice sample transcript is configured but the CLI backend cannot pass it yet."
                    | None -> ""
                let cudaNote =
                    if wantsCuda then
                        if cudaReported then " CUDA EP was reported by the CLI."
                        else " CUDA EP was not reported by the CLI."
                    else
                        " CPU execution was requested."
                return
                    { Phase = request.Phase
                      Text = text
                      OutputPath = Some outputPath
                      SampleRate = sampleRate
                      Samples = samples.Length
                      DurationMs = float samples.Length / float sampleRate * 1000.0
                      InferenceTimeMs = stopwatch.Elapsed.TotalMilliseconds
                      Message = $"VoxCPM2 CLI TTS synthesis completed.{cudaNote}{transcriptNote}" }
            }

module TtsRuntimeFactory =
    let create (options: TtsRuntimeOptions) (pathBase: string) =
        let runtime =
            if String.IsNullOrWhiteSpace options.Runtime then "voxcpm2-cli"
            else options.Runtime.Trim().ToLowerInvariant()
        match runtime with
        | "fake" | "fake-tone" | "tone" -> new FakeToneTtsRuntime(options, pathBase) :> ITtsRuntime
        | "voxcpm2-cli" | "voxcpm2" | "external-voxcpm2" -> new VoxCpm2CliTtsRuntime(options, pathBase) :> ITtsRuntime
        | other -> invalidArg (nameof options.Runtime) $"Unsupported TTS runtime '{other}'. Use voxcpm2-cli or fake-tone."
