namespace ChromaOnnx

open System
open Microsoft.ML.OnnxRuntime.Tensors

type GemmaChatRole =
    | System
    | User
    | Model
    | Tool

type GemmaChatMessage =
    { Role: GemmaChatRole
      Content: string
      ToolName: string option }

module GemmaChatMessage =
    let system content =
        { Role = GemmaChatRole.System
          Content = content
          ToolName = None }

    let user content =
        { Role = GemmaChatRole.User
          Content = content
          ToolName = None }

    let model content =
        { Role = GemmaChatRole.Model
          Content = content
          ToolName = None }

    let tool name content =
        { Role = GemmaChatRole.Tool
          Content = content
          ToolName = Some name }

type GemmaToolParameter =
    { Name: string
      Description: string
      Type: string
      Required: bool }

type GemmaToolDeclaration =
    { Name: string
      Description: string
      Parameters: GemmaToolParameter array }

type GemmaToolCall =
    { Name: string
      Arguments: Map<string, string>
      RawText: string }

type GemmaAudioFeatures =
    { InputFeatures: DenseTensor<float32>
      InputFeaturesMask: DenseTensor<bool>
      AudioTokenCount: int
      FrameCount: int
      ValidFrameCount: int
      SampleCount: int }

type GemmaPreparedInputs =
    { Prompt: string
      InputIds: DenseTensor<int64>
      AttentionMask: DenseTensor<int64>
      AudioFeatures: GemmaAudioFeatures option }

type GemmaGenerationRequest =
    { Messages: GemmaChatMessage array
      Tools: GemmaToolDeclaration array
      Audio16k: float32 array option
      AddGenerationPrompt: bool
      MaxNewTokens: int
      Temperature: float
      TopP: float
      TopK: int }

type GemmaGenerationResult =
    { Text: string
      Prompt: string
      InputTokenCount: int
      OutputTokenIds: int64 array
      StopReason: string
      TimingsMs: Map<string, float> }

type GemmaRuntimeStatus =
    { Ready: bool
      ModelDir: string
      Variant: string
      ExecutionProvider: string
      MissingFiles: string array
      LoadedSessions: string array
      Message: string }

type IGemmaRuntime =
    abstract Status: unit -> GemmaRuntimeStatus
    abstract Prepare: request: GemmaGenerationRequest -> GemmaPreparedInputs
    abstract GenerateAsync:
        request: GemmaGenerationRequest *
        cancellationToken: System.Threading.CancellationToken -> System.Threading.Tasks.Task<GemmaGenerationResult>

