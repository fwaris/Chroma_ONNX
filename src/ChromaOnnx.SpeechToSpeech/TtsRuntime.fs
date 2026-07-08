namespace ChromaOnnx.SpeechToSpeech

open System
open System.Diagnostics
open System.IO
open System.Threading.Tasks
open ChromaOnnx

module private TtsWav =
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

module TtsRuntimeFactory =
    let create (options: TtsRuntimeOptions) (pathBase: string) =
        let runtime =
            if String.IsNullOrWhiteSpace options.Runtime then "chatterbox-onnx"
            else options.Runtime.Trim().ToLowerInvariant()

        match runtime with
        | "fake" | "fake-tone" | "tone" -> new FakeToneTtsRuntime(options, pathBase) :> ITtsRuntime
        | "chatterbox" | "chatterbox-onnx" | "chatterbox_onnx" -> new ChatterboxOnnxTtsRuntime(options, pathBase) :> ITtsRuntime
        | other -> invalidArg (nameof options.Runtime) $"Unsupported TTS runtime '{other}'. Use chatterbox-onnx or fake-tone."
