param(
    [string]$RuntimeRoot = "E:\s\temp\VoiceAgent_A100_runtime",
    [string]$AssetsRoot = "E:\s\temp\VoiceAgent_assets",
    [string]$Configuration = "Release",
    [string]$RuntimeIdentifier = "win-x64",
    [string]$TtsExecutablePath = "",
    [switch]$SkipPublish,
    [switch]$FrameworkDependent,
    [switch]$SkipAssetValidation
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptRoot
$RuntimeRoot = [System.IO.Path]::GetFullPath($RuntimeRoot)
$AssetsRoot = [System.IO.Path]::GetFullPath($AssetsRoot)
$serviceOut = Join-Path $RuntimeRoot "app\service"
$toolsOut = Join-Path $RuntimeRoot "tools"

function Assert-PathExists([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Label was not found: $Path"
    }
}

function Assert-ChildPath([string]$Path, [string]$Parent, [string]$Label) {
    $resolvedPath = [System.IO.Path]::GetFullPath($Path)
    $resolvedParent = [System.IO.Path]::GetFullPath($Parent).TrimEnd('\')
    if (-not ($resolvedPath.StartsWith($resolvedParent + "\", [System.StringComparison]::OrdinalIgnoreCase))) {
        throw "$Label is outside the expected root. Path: $resolvedPath Root: $resolvedParent"
    }
}

if (-not $SkipAssetValidation) {
    Assert-PathExists (Join-Path $AssetsRoot "models\gemma-4-e2b-it-onnx-mobius\Q4_K_M\cuda\genai_config.json") "Gemma ORT GenAI config"
    Assert-PathExists (Join-Path $AssetsRoot "models\gemma-4-e2b-it-onnx-mobius\Q4_K_M\cuda\embedding\model.onnx") "Gemma embedding graph"
    Assert-PathExists (Join-Path $AssetsRoot "models\gemma-4-e2b-it-onnx-mobius\Q4_K_M\cuda\audio_encoder\model.onnx") "Gemma audio encoder graph"
    Assert-PathExists (Join-Path $AssetsRoot "models\gemma-4-e2b-it-onnx-mobius\Q4_K_M\cuda\decoder\model.onnx") "Gemma decoder graph"
    Assert-PathExists (Join-Path $AssetsRoot "models\voxcpm2-onnx\voxcpm2-decoder.onnx") "VoxCPM2 decoder graph"
    Assert-PathExists (Join-Path $AssetsRoot "models\voxcpm2-onnx\voxcpm2-decoder.onnx.data") "VoxCPM2 decoder external data"
    Assert-PathExists (Join-Path $AssetsRoot "models\voxcpm2-onnx\voxcpm2-audio-encoder.onnx") "VoxCPM2 audio encoder graph"
    Assert-PathExists (Join-Path $AssetsRoot "models\voxcpm2-onnx\voxcpm2-audio-decoder.onnx") "VoxCPM2 audio decoder graph"
    Assert-PathExists (Join-Path $AssetsRoot "models\voxcpm2-onnx\tokenizer.json") "VoxCPM2 tokenizer"
}

if (-not $SkipPublish) {
    Assert-ChildPath $serviceOut $RuntimeRoot "Publish output"
    if (Test-Path -LiteralPath $serviceOut) {
        Remove-Item -LiteralPath $serviceOut -Recurse -Force
    }
}
New-Item -ItemType Directory -Force -Path $serviceOut | Out-Null
New-Item -ItemType Directory -Force -Path $toolsOut | Out-Null

if (-not $SkipPublish) {
    $selfContained = if ($FrameworkDependent) { "false" } else { "true" }
    dotnet publish (Join-Path $repoRoot "src\ChromaOnnx.Service\ChromaOnnx.Service.fsproj") `
        -c $Configuration `
        -r $RuntimeIdentifier `
        --self-contained $selfContained `
        -o $serviceOut
}

foreach ($file in @(
    "ChromaOnnx.Service.exe",
    "ChromaOnnx.Service.dll",
    "ChromaOnnx.Service.deps.json",
    "ChromaOnnx.Service.runtimeconfig.json",
    "ChromaOnnx.SpeechToSpeech.dll",
    "ChromaOnnx.OnnxRuntime.dll",
    "ChromaOnnx.Core.dll",
    "Microsoft.ML.OnnxRuntimeGenAI.dll",
    "onnxruntime.dll",
    "onnxruntime-genai.dll",
    "onnxruntime-genai-cuda.dll",
    "onnxruntime_providers_cuda.dll",
    "onnxruntime_providers_shared.dll"
)) {
    Assert-PathExists (Join-Path $serviceOut $file) "Published dependency $file"
}

if (-not [string]::IsNullOrWhiteSpace($TtsExecutablePath)) {
    $resolvedTtsExecutable = [System.IO.Path]::GetFullPath($TtsExecutablePath)
    Assert-PathExists $resolvedTtsExecutable "VoxCPM2 ONNX CLI executable"
    Copy-Item -LiteralPath $resolvedTtsExecutable -Destination (Join-Path $toolsOut "speech_voxcpm2_clone_onnx.exe") -Force
    foreach ($nativeDll in @(
        "onnxruntime.dll",
        "onnxruntime_providers_cuda.dll",
        "onnxruntime_providers_shared.dll"
    )) {
        $cliNative = Join-Path (Split-Path -Parent $resolvedTtsExecutable) $nativeDll
        $sourceNative =
            if (Test-Path -LiteralPath $cliNative) { $cliNative }
            else { Join-Path $serviceOut $nativeDll }
        Copy-Item -LiteralPath $sourceNative -Destination (Join-Path $toolsOut $nativeDll) -Force
    }
}

$runService = @'
param(
    [string]$AssetsRoot = "",
    [int]$Port = 5055,
    [int]$MaxHistoryTurns = 8,
    [double]$MaxTurnAudioSeconds = 30,
    [string]$GemmaModelDir = "",
    [string]$GemmaVariant = "Q4_K_M/cuda",
    [ValidateSet("ort-genai", "raw-ort")]
    [string]$GemmaRuntime = "ort-genai",
    [ValidateSet("cpu", "cuda")]
    [string]$GemmaExecutionProvider = "cuda",
    [ValidateSet("voxcpm2-cli", "fake-tone")]
    [string]$TtsRuntime = "voxcpm2-cli",
    [string]$TtsModelDir = "",
    [string]$TtsExecutablePath = "",
    [string]$VoiceSamplePath = "",
    [string]$VoiceSampleTranscript = "",
    [int]$TtsMaxSteps = 256,
    [bool]$RequireTtsGpu = $true,
    [switch]$RequireFullTtsGpu,
    [int]$TtsCudaDeviceId = 0,
    [double]$TtsGpuMemoryLimitGb = 0,
    [string]$CudaBin = "",
    [string]$CudnnBin = "",
    [switch]$LocalhostOnly
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $AssetsRoot) {
    $AssetsRoot = Join-Path (Split-Path -Parent $Root) "VoiceAgent_assets"
}
$AssetsRoot = [System.IO.Path]::GetFullPath($AssetsRoot)
if ($CudaBin) { $env:PATH = "$CudaBin;$env:PATH" }
if ($CudnnBin) { $env:PATH = "$CudnnBin;$env:PATH" }

if ($GemmaModelDir) {
    $selectedGemmaModelDir = [System.IO.Path]::GetFullPath($GemmaModelDir)
} else {
    $selectedGemmaModelDir = Join-Path $AssetsRoot "models\gemma-4-e2b-it-onnx-mobius\Q4_K_M\cuda"
}
if ($TtsModelDir) {
    $selectedTtsModelDir = [System.IO.Path]::GetFullPath($TtsModelDir)
} else {
    $selectedTtsModelDir = Join-Path $AssetsRoot "models\voxcpm2-onnx"
}
if ($TtsExecutablePath) {
    $selectedTtsExecutablePath = [System.IO.Path]::GetFullPath($TtsExecutablePath)
} else {
    $selectedTtsExecutablePath = Join-Path $Root "tools\speech_voxcpm2_clone_onnx.exe"
}

$serviceDir = Join-Path $Root "app\service"
$requiredPaths = @(
    (Join-Path $serviceDir "ChromaOnnx.Service.exe"),
    (Join-Path $serviceDir "onnxruntime.dll"),
    (Join-Path $serviceDir "onnxruntime_providers_cuda.dll"),
    (Join-Path $serviceDir "onnxruntime_providers_shared.dll"),
    (Join-Path $serviceDir "Microsoft.ML.OnnxRuntimeGenAI.dll"),
    (Join-Path $serviceDir "onnxruntime-genai.dll"),
    (Join-Path $serviceDir "onnxruntime-genai-cuda.dll"),
    (Join-Path $selectedGemmaModelDir "genai_config.json"),
    (Join-Path $selectedGemmaModelDir "tokenizer.json"),
    (Join-Path $selectedGemmaModelDir "embedding\model.onnx"),
    (Join-Path $selectedGemmaModelDir "audio_encoder\model.onnx"),
    (Join-Path $selectedGemmaModelDir "decoder\model.onnx")
)
if ($TtsRuntime -eq "voxcpm2-cli") {
    $requiredPaths += @(
        $selectedTtsExecutablePath,
        (Join-Path $selectedTtsModelDir "voxcpm2-decoder.onnx"),
        (Join-Path $selectedTtsModelDir "voxcpm2-decoder.onnx.data"),
        (Join-Path $selectedTtsModelDir "voxcpm2-audio-encoder.onnx"),
        (Join-Path $selectedTtsModelDir "voxcpm2-audio-decoder.onnx"),
        (Join-Path $selectedTtsModelDir "tokenizer.json")
    )
    if ($VoiceSamplePath) {
        $requiredPaths += [System.IO.Path]::GetFullPath($VoiceSamplePath)
    }
}
foreach ($path in $requiredPaths) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required path is missing: $path"
    }
}

$env:VoiceAgent__WorkDir = Join-Path $Root "served_runs"
$env:VoiceAgent__MaxHistoryTurns = [string]$MaxHistoryTurns
$env:VoiceAgent__MaxTurnAudioSeconds = [string]$MaxTurnAudioSeconds
$env:VoiceAgent__Gemma__ModelDir = $selectedGemmaModelDir
$env:VoiceAgent__Gemma__Variant = $GemmaVariant
$env:VoiceAgent__Gemma__Runtime = $GemmaRuntime
$env:VoiceAgent__Gemma__ExecutionProvider = $GemmaExecutionProvider
$env:VoiceAgent__Tts__ModelDir = $selectedTtsModelDir
$env:VoiceAgent__Tts__Runtime = $TtsRuntime
$env:VoiceAgent__Tts__ExecutionProvider = "cuda"
$env:VoiceAgent__Tts__ExecutablePath = $selectedTtsExecutablePath
$env:VoiceAgent__Tts__Variant = "onnx"
$env:VoiceAgent__Tts__VoiceSamplePath = $VoiceSamplePath
$env:VoiceAgent__Tts__VoiceSampleTranscript = $VoiceSampleTranscript
$env:VoiceAgent__Tts__OutputSampleRate = "48000"
$env:VoiceAgent__Tts__MaxSteps = [string]$TtsMaxSteps
$env:VoiceAgent__Tts__StreamingChunkSeconds = "0.5"
$env:VoiceAgent__Tts__RequireGpu = [string]$RequireTtsGpu
$env:VoiceAgent__Tts__RequireFullGpu = [string]([bool]$RequireFullTtsGpu)
$env:VoiceAgent__Tts__CudaDeviceId = [string]$TtsCudaDeviceId
$env:VoiceAgent__Tts__GpuMemoryLimitGb = [string]$TtsGpuMemoryLimitGb

$hostName = if ($LocalhostOnly) { "localhost" } else { "0.0.0.0" }
$url = "http://${hostName}:$Port"
Write-Host "Starting GemmaVoiceAgent on $url"
Write-Host "AssetsRoot: $AssetsRoot"
Write-Host "GemmaModelDir: $selectedGemmaModelDir"
Write-Host "TtsRuntime: $TtsRuntime"
Write-Host "TtsModelDir: $selectedTtsModelDir"
Write-Host "TtsExecutablePath: $selectedTtsExecutablePath"
Write-Host "VoiceSamplePath: $VoiceSamplePath"
& (Join-Path $serviceDir "ChromaOnnx.Service.exe") --urls $url
exit $LASTEXITCODE
'@

$readme = @'
# Gemma Voice Agent A100 Runtime

This package runs the text-first voice-agent path:

user audio -> Gemma 4 ASR -> Gemma 4 reasoning/tool calls/filler text -> TTS filler -> TTS final.

Default TTS runtime is `voxcpm2-cli`, which expects:

- `tools\speech_voxcpm2_clone_onnx.exe`
- `VoiceAgent_assets\models\voxcpm2-onnx\voxcpm2-decoder.onnx`
- `VoiceAgent_assets\models\voxcpm2-onnx\voxcpm2-decoder.onnx.data`
- `VoiceAgent_assets\models\voxcpm2-onnx\voxcpm2-audio-encoder.onnx`
- `VoiceAgent_assets\models\voxcpm2-onnx\voxcpm2-audio-decoder.onnx`
- `VoiceAgent_assets\models\voxcpm2-onnx\tokenizer.json`

The bundled VoxCPM2 CLI is built with an explicit CUDA EP hook. The service sets
`RequireTtsGpu = true` by default, so TTS fails instead of silently falling back
to CPU if CUDA cannot be appended. `-RequireFullTtsGpu` is a stricter diagnostic
that disables ORT CPU EP fallback; it may fail because ORT often keeps small
shape/control nodes on CPU even when the heavy model compute runs on CUDA.

Gemma assets are not included in this runtime package and must be staged under:

`VoiceAgent_assets\models\gemma-4-e2b-it-onnx-mobius\Q4_K_M\cuda`

Run service:

```powershell
.\run-voice-agent-a100.ps1 -AssetsRoot G:\Chroma\VoiceAgent_assets -VoiceSamplePath G:\Chroma\VoiceAgent_assets\voices\reference.wav
```

Smoke test with fake TTS plumbing:

```powershell
.\smoke-voice-agent-a100.ps1 -AssetsRoot G:\Chroma\VoiceAgent_assets -TtsRuntime fake-tone
```

Smoke test with real VoxCPM2:

```powershell
.\smoke-voice-agent-a100.ps1 -AssetsRoot G:\Chroma\VoiceAgent_assets -TtsRuntime voxcpm2-cli -VoiceSamplePath G:\Chroma\VoiceAgent_assets\voices\reference.wav -TtsMaxSteps 64
```
'@

Set-Content -LiteralPath (Join-Path $RuntimeRoot "run-voice-agent-a100.ps1") -Value $runService -Encoding UTF8
Copy-Item -LiteralPath (Join-Path $repoRoot "scripts\smoke_voice_agent_two_turn_tool.ps1") -Destination (Join-Path $RuntimeRoot "smoke-voice-agent-a100.ps1") -Force
Set-Content -LiteralPath (Join-Path $RuntimeRoot "README-voice-agent-a100.md") -Value $readme -Encoding UTF8

$manifest = @"
Gemma Voice Agent A100 Runtime
Generated: $(Get-Date -Format o)
RuntimeRoot: $RuntimeRoot
AssetsRoot: $AssetsRoot
Service: GemmaVoiceAgent
TTS runtime: voxcpm2-cli
Voice cloning: reference WAV via VoiceAgent:Tts:VoiceSamplePath
Models included: no
"@
Set-Content -LiteralPath (Join-Path $RuntimeRoot "VOICE_AGENT_MANIFEST.txt") -Value $manifest -Encoding UTF8

Write-Host "Gemma Voice Agent A100 runtime is ready at $RuntimeRoot"
Write-Host "ServiceDir: $serviceOut"
Write-Host "ToolsDir: $toolsOut"
Write-Host "AssetsRoot: $AssetsRoot"
