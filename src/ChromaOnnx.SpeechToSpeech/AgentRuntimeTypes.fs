namespace ChromaOnnx.SpeechToSpeech

open System
open System.Text.Json
open System.Threading
open System.Threading.Tasks
open ChromaOnnx

type GemmaRuntimeOptions() =
    member val ModelDir = "models/gemma-4-e2b-it-onnx-mobius/Q4_K_M/cuda" with get, set
    member val Variant = "Q4_K_M/cuda" with get, set
    member val Runtime = "ort-genai" with get, set
    member val ExecutionProvider = "cuda" with get, set
    member val MaxAudioSeconds = 30.0 with get, set
    member val AsrMaxNewTokens = 128 with get, set
    member val ReasoningMaxNewTokens = 512 with get, set
    member val ToolMaxRounds = 3 with get, set
    member val MaxHistoryTurns = 8 with get, set

type AgentSessionRequest =
    { SystemPrompt: string
      PromptText: string
      PromptAudio24k: float32 array
      MaxNewFrames: int }

type AgentSessionInfo =
    { Id: string
      ServiceName: string
      Mode: string
      PromptText: string
      SystemPrompt: string
      MaxNewFrames: int
      PromptAudioSamples: int
      PromptSampleRate: int
      WebsocketUrl: string
      CreatedUtc: DateTimeOffset }

type AgentTurnRequest =
    { SessionId: string
      UserAudio16k: float32 array
      RequestId: string option }

type AgentToolCallInfo =
    { Round: int
      Name: string
      Arguments: Map<string, string>
      RawText: string }

type AgentToolResultInfo =
    { Round: int
      Name: string
      Success: bool
      Result: string
      Error: string option }

type AgentTurnResult =
    { Id: string
      RequestId: string
      TurnIndex: int
      Transcript: string
      FinalText: string
      ToolCalls: AgentToolCallInfo array
      ToolResults: AgentToolResultInfo array
      ChromaSessionId: string option
      ChromaTurnIndex: int option
      AudioUrl: string option
      DetailsUrl: string
      Details: JsonElement }

type AgentStreamingEvent =
    | AgentTranscription of sessionId: string * requestId: string * turnIndex: int * transcript: string
    | AgentToolCall of sessionId: string * requestId: string * turnIndex: int * call: AgentToolCallInfo
    | AgentToolResult of sessionId: string * requestId: string * turnIndex: int * result: AgentToolResultInfo
    | AgentFinalText of sessionId: string * requestId: string * turnIndex: int * text: string
    | AgentVocalizationStarted of sessionId: string * requestId: string * turnIndex: int * chromaSessionId: string
    | AgentChromaEvent of S2sStreamingEvent
    | AgentDone of AgentTurnResult
    | AgentCanceled of sessionId: string * requestId: string option

type AgentRuntimeStatus =
    { Ready: bool
      ServiceName: string
      Mode: string
      ModelDir: string
      Variant: string
      ExecutionProvider: string
      MaxAudioSeconds: float
      AsrMaxNewTokens: int
      ReasoningMaxNewTokens: int
      ToolMaxRounds: int
      MaxHistoryTurns: int
      Gemma: GemmaRuntimeStatus
      ChromaReady: bool
      Message: string }

type IAgentRuntime =
    abstract MaxTurnAudioSamples: int
    abstract Status: unit -> AgentRuntimeStatus
    abstract CreateSession: request: AgentSessionRequest -> AgentSessionInfo
    abstract TryGetSession: id: string -> AgentSessionInfo option
    abstract RunTurnAsync:
        request: AgentTurnRequest *
        emit: (AgentStreamingEvent -> Task) *
        cancellationToken: CancellationToken -> Task<AgentTurnResult>
    abstract TryGetTurnArtifact: sessionId: string * turnIndex: int * fileName: string -> S2sArtifact option
