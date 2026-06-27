namespace ChromaOnnx

open System.Collections.Generic
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
