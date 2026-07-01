namespace ChromaOnnx

open System
open System.Buffers
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

type RentedTensorBuffer<'T>(array: 'T array, count: int, clearOnReturn: bool) =
    let mutable returned = false

    member _.Memory = Memory<'T>(array, 0, count)

    interface IDisposable with
        member _.Dispose() =
            if not returned then
                returned <- true
                ArrayPool<'T>.Shared.Return(array, clearArray = clearOnReturn)

module RentedTensorBuffer =
    let rent<'T> count clearOnReturn =
        if count < 0 then
            invalidArg (nameof count) "Buffer length cannot be negative."

        new RentedTensorBuffer<'T>(ArrayPool<'T>.Shared.Rent(count), count, clearOnReturn)

type NativeS2sPrepared
    (
        inputIds: DenseTensor<int64>,
        attentionMask: DenseTensor<int64>,
        inputValues: DenseTensor<float32>,
        inputValuesCutoffs: DenseTensor<int64>,
        thinkerInputIds: DenseTensor<int64>,
        thinkerAttentionMask: DenseTensor<int64>,
        thinkerInputFeatures: DenseTensor<float32>,
        thinkerFeatureAttentionMask: DenseTensor<int64>,
        promptAudioSamples: int,
        userAudioSamples: int,
        conversationText: string,
        ?ownedBuffers: IDisposable array
    ) =
    let ownedBuffers = defaultArg ownedBuffers Array.empty
    let mutable disposed = false

    let ensureActive () =
        if disposed then
            invalidOp "NativeS2sPrepared has been disposed."

    member _.InputIds =
        ensureActive ()
        inputIds

    member _.AttentionMask =
        ensureActive ()
        attentionMask

    member _.InputValues =
        ensureActive ()
        inputValues

    member _.InputValuesCutoffs =
        ensureActive ()
        inputValuesCutoffs

    member _.ThinkerInputIds =
        ensureActive ()
        thinkerInputIds

    member _.ThinkerAttentionMask =
        ensureActive ()
        thinkerAttentionMask

    member _.ThinkerInputFeatures =
        ensureActive ()
        thinkerInputFeatures

    member _.ThinkerFeatureAttentionMask =
        ensureActive ()
        thinkerFeatureAttentionMask

    member _.PromptAudioSamples =
        ensureActive ()
        promptAudioSamples

    member _.UserAudioSamples =
        ensureActive ()
        userAudioSamples

    member _.ConversationText =
        ensureActive ()
        conversationText

    interface IDisposable with
        member _.Dispose() =
            if not disposed then
                disposed <- true
                for buffer in ownedBuffers do
                    buffer.Dispose()

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
