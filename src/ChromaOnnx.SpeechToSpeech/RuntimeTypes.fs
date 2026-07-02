namespace ChromaOnnx.SpeechToSpeech

open System
open System.Text.Json
open System.Threading
open System.Threading.Tasks
open ChromaOnnx

type S2sRuntimeOptions() =
    member val ModelDir = "models/chroma-4b" with get, set
    member val BundleDir = "onnx/chroma-s2s-full-v2" with get, set
    member val WorkDir = "served_runs" with get, set
    member val ExecutionProvider = "cuda" with get, set
    member val MemoryMode = "resident-merged" with get, set
    member val OrtMemoryProfile = "quality-safe" with get, set
    member val OptimizedModelCacheDir = "onnx/chroma-s2s-full-v2/ort-cache-ort-local-external" with get, set
    member val OptimizedModelCacheFormat = "onnx" with get, set
    member val CudaGpuMemLimitMb = Nullable<int>(15360) with get, set
    member val ThinkerActiveFrames = 0 with get, set
    member val StreamDecodeFrames = 4 with get, set
    member val StreamMinFreeVramMb = 2048 with get, set
    member val CodecStallGuardFrames = 16 with get, set
    member val MaxQueueLength = 32 with get, set
    member val MaxPromptAudioSeconds = 60.0 with get, set
    member val MaxTurnAudioSeconds = 60.0 with get, set

type S2sSessionRequest =
    { PromptText: string
      SystemPrompt: string
      Backend: string
      PromptAudio24k: float32 array
      MaxNewFrames: int }

type S2sSessionInfo =
    { Id: string
      ServiceName: string
      Mode: string
      Backend: string
      PromptText: string
      SystemPrompt: string
      MaxNewFrames: int
      MaxResponseSeconds: float
      PromptAudioSamples: int
      PromptSampleRate: int
      WebsocketUrl: string
      CreatedUtc: DateTimeOffset }

type S2sTurnRequest =
    { SessionId: string
      UserAudio16k: float32 array
      RequestId: string option }

type S2sBackendResult =
    { Backend: string
      AudioUrl: string
      DetailsUrl: string
      Details: JsonElement }

type S2sTurnResult =
    { Id: string
      RequestId: string
      Backend: string
      AudioUrl: string
      DetailsUrl: string
      Results: S2sBackendResult array }

type S2sAudioDeferredEvent =
    { FrameCount: int
      FreeVramMb: int
      UsedVramMb: int
      TotalVramMb: int
      MinFreeVramMb: int
      Message: string }

type S2sStreamingEvent =
    | QueueEnqueued of requestId: string * snapshot: WorkQueuePosition
    | QueueUpdated of snapshot: WorkQueuePosition
    | QueueStarted of requestId: string * queueLength: int
    | GenerationStarted of requestId: string * maxNewFrames: int * streamDecodeFrames: int * streamMinFreeVramMb: int * codecStallGuardFrames: int
    | GenerationFrame of requestId: string * frame: S2sGeneratedFrame
    | AudioChunk of requestId: string * chunk: S2sAudioChunk
    | AudioDeferred of requestId: string * deferred: S2sAudioDeferredEvent
    | GenerationDone of result: S2sTurnResult
    | GenerationCanceled of sessionId: string * requestId: string option

type S2sRuntimeStatus =
    { Ready: bool
      ServiceName: string
      Mode: string
      PythonInRequestPath: bool
      ModelDir: string
      BundleDir: string
      ExecutionProvider: string
      MemoryMode: string
      OrtMemoryProfile: string
      CudaGpuMemLimitMb: Nullable<int>
      OptimizedModelCacheEnabled: bool
      OptimizedModelCacheDir: string | null
      OptimizedModelCacheFormat: string
      Memory: RuntimeMemory.Snapshot
      LoadedOrtSessions: string array
      WarmOrtSessions: string array
      ActivePagedOrtSessions: string array
      QueueLength: int
      RunningRequestId: string option
      MaxQueueLength: int
      StreamDecodeFrames: int
      StreamMinFreeVramMb: int
      CodecStallGuardFrames: int
      MaxPromptAudioSeconds: float
      MaxTurnAudioSeconds: float
      GlobalGpuMemory: RuntimeMemory.GpuGlobalSnapshot option
      PeakPrivateGb: float
      PeakWorkingSetGb: float
      MappedSafetensorShards: int
      InitializerCount: int
      UniqueInitializerSources: int
      UniqueOrtValues: int
      SharedPrepackedWeights: bool
      Message: string
      MissingGraphs: string array
      AvailableGraphs: string array
      PromptSampleRate: int
      ThinkerSampleRate: int
      ThinkerFeatureMode: string
      ThinkerConfiguredActiveFrames: int
      ThinkerTraceFeatureFrames: int
      ThinkerTraceSamples: int }

type S2sArtifact =
    { Path: string
      ContentType: string }

type IS2sRuntime =
    abstract MaxPromptAudioSamples: int
    abstract MaxTurnAudioSamples: int
    abstract MaxQueueLength: int
    abstract Status: unit -> S2sRuntimeStatus
    abstract CreateSession: request: S2sSessionRequest -> S2sSessionInfo
    abstract TryGetSession: id: string -> S2sSessionInfo option
    abstract RunTurnAsync:
        request: S2sTurnRequest *
        emit: (S2sStreamingEvent -> Task) *
        cancellationToken: CancellationToken -> Task<S2sTurnResult>
    abstract TryGetArtifact: sessionId: string * backend: string option * fileName: string -> S2sArtifact option
