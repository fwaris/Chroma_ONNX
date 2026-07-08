namespace ChromaOnnx.SpeechToSpeech

open System
open System.Text.Json
open System.Threading
open System.Threading.Tasks
open ChromaOnnx

type PersonaPlexRuntimeOptions() =
    member val ModelDir = "models/personaplex-7b-v1-onnx" with get, set
    member val Runtime = "full-onnx" with get, set
    member val ExecutionProvider = "cuda" with get, set
    member val VoicePreset = "NATF2" with get, set
    member val TextPrompt = "You are a friendly assistant." with get, set
    member val HuggingFaceRepoId = "elbruno/personaplex-7b-v1-onnx" with get, set
    member val MaxNewFrames = 8 with get, set
    member val WarmupFrames = 1 with get, set

type VoiceAgentOptions() =
    member val WorkDir = "served_runs" with get, set
    member val MaxHistoryTurns = 8 with get, set
    member val MaxTurnAudioSeconds = 30.0 with get, set
    member val Gemma = GemmaRuntimeOptions() with get, set
    member val Tts = TtsRuntimeOptions() with get, set
    member val PersonaPlex = PersonaPlexRuntimeOptions() with get, set

and TtsRuntimeOptions() =
    member val ModelDir = "models/chatterbox-onnx" with get, set
    member val Runtime = "chatterbox-onnx" with get, set
    member val ExecutionProvider = "cuda" with get, set
    member val Variant = "q4f16" with get, set
    member val HuggingFaceRepoId = "onnx-community/chatterbox-ONNX" with get, set
    member val VoiceSamplePath = "" with get, set
    member val VoiceSampleTranscript = "" with get, set
    member val Instruction = "" with get, set
    member val OutputSampleRate = 24000 with get, set
    member val MaxSteps = 256 with get, set
    member val Seed = 12345 with get, set
    member val Exaggeration = 0.5 with get, set
    member val RepetitionPenalty = 1.2 with get, set
    member val StreamingChunkSeconds = 0.5 with get, set
    member val RequireGpu = true with get, set
    member val RequireFullGpu = false with get, set
    member val CudaDeviceId = 0 with get, set
    member val GpuMemoryLimitGb = 0.0 with get, set

type SttRuntimeStatus =
    { Ready: bool
      Runtime: string
      InputSampleRate: int
      OutputLanguage: string
      Message: string }

type SttTranscriptionResult =
    { Transcript: string
      InputSampleRate: int
      InputSamples: int
      DurationMs: float
      Message: string }

type ISttRuntime =
    abstract Status: unit -> SttRuntimeStatus
    abstract TranscribeAsync:
        samples24k: float32 array *
        outputDirectory: string *
        cancellationToken: CancellationToken -> Task<SttTranscriptionResult>

type TtsRuntimeStatus =
    { Ready: bool
      SupportsVoiceCloning: bool
      SupportsStreaming: bool
      Runtime: string
      ModelDir: string
      ExecutionProvider: string
      OutputSampleRate: int
      VoiceSamplePath: string
      MissingFiles: string array
      Message: string }

type TtsSynthesisRequest =
    { Phase: string
      Text: string
      OutputDirectory: string
      OutputFileName: string
      VoiceSamplePath: string option
      VoiceSampleTranscript: string option }

type TtsSynthesisResult =
    { Phase: string
      Text: string
      OutputPath: string option
      SampleRate: int
      Samples: int
      DurationMs: float
      InferenceTimeMs: float
      Message: string }

type ITtsRuntime =
    abstract Status: unit -> TtsRuntimeStatus
    abstract SynthesizeAsync:
        request: TtsSynthesisRequest *
        emitChunk: (float32 array -> Task) *
        cancellationToken: CancellationToken -> Task<TtsSynthesisResult>

type PersonaPlexRuntimeStatus =
    { Ready: bool
      CodecReady: bool
      SpeechToSpeechReady: bool
      SupportsStreaming: bool
      SupportsDuplex: bool
      Runtime: string
      ModelDir: string
      ExecutionProvider: string
      VoicePreset: string
      MissingFiles: string array
      Message: string }

type PersonaPlexCodecResult =
    { OutputPath: string option
      DurationMs: float
      InferenceTimeMs: float
      SampleRate: int
      Message: string }

type PersonaPlexGenerationResult =
    { OutputPath: string option
      InputFrames: int
      GeneratedFrames: int
      SampleRate: int
      DurationMs: float
      InferenceTimeMs: float
      Message: string }

type IPersonaPlexRuntime =
    abstract Status: unit -> PersonaPlexRuntimeStatus
    abstract RunCodecRoundTripAsync:
        samples24k: float32 array *
        outputDirectory: string *
        cancellationToken: CancellationToken -> Task<PersonaPlexCodecResult>
    abstract RunSpeechToSpeechAsync:
        samples24k: float32 array *
        outputDirectory: string *
        cancellationToken: CancellationToken -> Task<PersonaPlexGenerationResult>

type VoiceAgentSessionRequest =
    { SystemPrompt: string
      Mode: string }

type VoiceAgentSessionInfo =
    { Id: string
      ServiceName: string
      Mode: string
      SystemPrompt: string
      WebsocketUrl: string
      CreatedUtc: DateTimeOffset }

type VoiceAgentTurnRequest =
    { SessionId: string
      UserAudio24k: float32 array
      RequestId: string option }

type VoiceAgentTurnResult =
    { Id: string
      RequestId: string
      TurnIndex: int
      Transcript: string
      FinalText: string
      ToolCalls: AgentToolCallInfo array
      ToolResults: AgentToolResultInfo array
      AudioUrl: string option
      DetailsUrl: string
      Details: JsonElement }

type VoiceAgentStreamingEvent =
    | VoiceAgentTranscription of sessionId: string * requestId: string * turnIndex: int * transcript: string
    | VoiceAgentToolCall of sessionId: string * requestId: string * turnIndex: int * call: AgentToolCallInfo
    | VoiceAgentToolResult of sessionId: string * requestId: string * turnIndex: int * result: AgentToolResultInfo
    | VoiceAgentFillerText of sessionId: string * requestId: string * turnIndex: int * text: string
    | VoiceAgentFinalText of sessionId: string * requestId: string * turnIndex: int * text: string
    | TtsSynthesisStarted of sessionId: string * requestId: string * turnIndex: int * phase: string * text: string
    | TtsAudioChunk of sessionId: string * requestId: string * turnIndex: int * phase: string * sampleRate: int * samples: float32 array
    | TtsSynthesisDone of sessionId: string * requestId: string * turnIndex: int * result: TtsSynthesisResult
    | TtsSynthesisCanceled of sessionId: string * requestId: string * turnIndex: int * phase: string
    | TtsUnavailable of sessionId: string * requestId: string * turnIndex: int * phase: string * message: string
    | PersonaPlexCodecStarted of sessionId: string * requestId: string * turnIndex: int
    | PersonaPlexCodecDone of sessionId: string * requestId: string * turnIndex: int * result: PersonaPlexCodecResult
    | PersonaPlexGenerationStarted of sessionId: string * requestId: string * turnIndex: int
    | PersonaPlexGenerationDone of sessionId: string * requestId: string * turnIndex: int * result: PersonaPlexGenerationResult
    | PersonaPlexAudioChunk of sessionId: string * requestId: string * turnIndex: int * samples: float32 array
    | PersonaPlexUnavailable of sessionId: string * requestId: string * turnIndex: int * message: string
    | VoiceAgentDone of VoiceAgentTurnResult
    | VoiceAgentCanceled of sessionId: string * requestId: string option

type VoiceAgentRuntimeStatus =
    { Ready: bool
      ServiceName: string
      Mode: string
      WorkDir: string
      MaxHistoryTurns: int
      MaxTurnAudioSeconds: float
      MaxTurnAudioSamples24k: int
      Gemma: GemmaRuntimeStatus
      Stt: SttRuntimeStatus
      Tts: TtsRuntimeStatus
      PersonaPlex: PersonaPlexRuntimeStatus
      Message: string }

type IVoiceAgentRuntime =
    abstract MaxTurnAudioSamples24k: int
    abstract Status: unit -> VoiceAgentRuntimeStatus
    abstract CreateSession: request: VoiceAgentSessionRequest -> VoiceAgentSessionInfo
    abstract TryGetSession: id: string -> VoiceAgentSessionInfo option
    abstract RunTurnAsync:
        request: VoiceAgentTurnRequest *
        emit: (VoiceAgentStreamingEvent -> Task) *
        cancellationToken: CancellationToken -> Task<VoiceAgentTurnResult>
    abstract TryGetTurnArtifact: sessionId: string * turnIndex: int * fileName: string -> S2sArtifact option
