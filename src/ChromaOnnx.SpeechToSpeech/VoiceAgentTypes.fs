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
    member val PersonaPlex = PersonaPlexRuntimeOptions() with get, set

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
    | VoiceAgentFinalText of sessionId: string * requestId: string * turnIndex: int * text: string
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
