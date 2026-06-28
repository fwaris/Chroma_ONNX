namespace ChromaOnnx

open System
open System.Collections.Generic
open System.IO
open Microsoft.ML.OnnxRuntime
open Microsoft.ML.OnnxRuntime.Tensors

type ChromaPaths =
    { OnnxDir: string
      SystemPrefill: string option
      Backbone: string
      Decoder: string
      CodecDecode: string option
      GraphFiles: string array }

type ChromaManifest =
    { HiddenSize: int option
      AudioNumCodebooks: int option }

type BackboneResult =
    { Logits: DenseTensor<float32>
      HiddenStates: DenseTensor<float32> }

type SystemPrefillResult =
    { Logits: DenseTensor<float32>
      HiddenStates: DenseTensor<float32>
      BackboneAttentionMask: DenseTensor<float32> }

type NativeS2sPrepared =
    { InputIds: DenseTensor<int64>
      AttentionMask: DenseTensor<int64>
      InputValues: DenseTensor<float32>
      InputValuesCutoffs: DenseTensor<int64>
      ThinkerInputIds: DenseTensor<int64>
      ThinkerAttentionMask: DenseTensor<int64>
      ThinkerInputFeatures: DenseTensor<float32>
      ThinkerFeatureAttentionMask: DenseTensor<int64>
      PromptAudioSamples: int
      UserAudioSamples: int
      ConversationText: string }

type ChromaS2sBundleStatus =
    { Ready: bool
      Bundle: string
      MissingGraphs: string array
      AvailableGraphs: string array
      ExecutionProvider: string
      Message: string }

type S2sGraphState =
    { AttentionMask: DenseTensor<float32>
      ThinkerInputIds: DenseTensor<int64>
      ThinkerAttentionMask: DenseTensor<int64>
      ThinkerCachePosition: DenseTensor<int64>
      ThinkerEos: DenseTensor<int64>
      BackboneCache: Dictionary<string, DenseTensor<float32>>
      ThinkerCache: Dictionary<string, DenseTensor<float32>> }

type S2sBackboneStep =
    { Logits: DenseTensor<float32>
      HiddenStates: DenseTensor<float32>
      State: S2sGraphState }

type S2sGenerationResult =
    { AudioCodes: DenseTensor<int64>
      AudioValues: DenseTensor<float32>
      FrameCount: int
      StopReason: string
      StepKinds: string array
      Timings: Dictionary<string, float> }

type DebugTensorInfo =
    { File: string
      Dtype: string
      Shape: int array }

type S2sDummyInput =
    | DummyFloat of DenseTensor<float32>
    | DummyInt64 of DenseTensor<int64>

type S2sOrtTuningOptions =
    { MemoryProfile: string
      OptimizedModelCacheDir: string option
      OptimizedModelCacheFormat: string }

module S2sOrtTuningOptions =
    let Default =
        { MemoryProfile = "quality-safe"
          OptimizedModelCacheDir = None
          OptimizedModelCacheFormat = "onnx" }

module OrtExecutionProvider =
    let normalize (executionProvider: string) =
        match executionProvider.Trim().ToLowerInvariant() with
        | "cuda" -> "cuda"
        | "cpu" -> "cpu"
        | "tensorrt" | "trt" -> "tensorrt"
        | value -> invalidArg (nameof executionProvider) $"Unsupported execution provider '{value}'. Use cuda, cpu, or tensorrt."

    let usesCudaDevice executionProvider =
        match normalize executionProvider with
        | "cuda" | "tensorrt" -> true
        | _ -> false

    let pythonDeviceDefault executionProvider =
        if usesCudaDevice executionProvider then "cuda" else "cpu"

    let tensorRtEngineCacheDir optimizedModelCacheDir =
        let root =
            optimizedModelCacheDir
            |> Option.defaultWith (fun () ->
                let localAppData =
                    Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData)
                if String.IsNullOrWhiteSpace localAppData then
                    Path.Combine(Path.GetTempPath(), "ChromaOnnx")
                else
                    Path.Combine(localAppData, "ChromaOnnx"))
        Path.Combine(root, "tensorrt-engines")

    let appendCuda (options: SessionOptions) qualitySafe =
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

    let appendToSessionOptions (options: SessionOptions) executionProvider optimizedModelCacheDir qualitySafe =
        match normalize executionProvider with
        | "tensorrt" ->
            let engineCacheDir = tensorRtEngineCacheDir optimizedModelCacheDir
            Directory.CreateDirectory(engineCacheDir) |> ignore
            let trtOptions = new OrtTensorRTProviderOptions()
            trtOptions.UpdateOptions(
                Dictionary<string, string>(
                    dict [ "device_id", "0"
                           "trt_engine_cache_enable", "1"
                           "trt_engine_cache_path", engineCacheDir
                           "trt_dump_subgraphs", "0"
                           "trt_fp16_enable", "0" ]
                )
            )
            options.AppendExecutionProvider_Tensorrt(trtOptions)
            trtOptions.Dispose()
            appendCuda options qualitySafe
            Some engineCacheDir
        | "cuda" ->
            appendCuda options qualitySafe
            None
        | "cpu" ->
            options.AppendExecutionProvider_CPU(if qualitySafe then 0 else 1)
            None
        | value -> invalidArg (nameof executionProvider) $"Unsupported execution provider '{value}'. Use cuda, cpu, or tensorrt."
