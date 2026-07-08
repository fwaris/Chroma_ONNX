param(
    [string]$RuntimeRoot = "E:\s\temp\VoiceAgent_A100_runtime",
    [string]$AssetsRoot = "E:\s\temp\VoiceAgent_assets",
    [string]$Configuration = "Release",
    [string]$RuntimeIdentifier = "win-x64",
    [string]$OrtGenAiRoot = "E:\s\repos\onnxruntime-genai",
    [string]$OrtGenAiBuildName = "WindowsNinjaCudaA100Sm80",
    [string]$OrtGenAiBuildDir = "",
    [string]$OrtGenAiManagedDir = "",
    [string]$OrtNativeDir = "",
    [string]$OrtNativePackageVersion = "1.27.0",
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
$OrtGenAiRoot = [System.IO.Path]::GetFullPath($OrtGenAiRoot)
if ([string]::IsNullOrWhiteSpace($OrtGenAiBuildDir)) {
    $OrtGenAiBuildDir = Join-Path $OrtGenAiRoot "build\$OrtGenAiBuildName\Release"
}
if ([string]::IsNullOrWhiteSpace($OrtGenAiManagedDir)) {
    $OrtGenAiManagedDir = Join-Path $OrtGenAiRoot "src\csharp\bin\$Configuration\net8.0"
}
if ([string]::IsNullOrWhiteSpace($OrtNativeDir)) {
    $nugetOrtNativeDir = Join-Path $env:USERPROFILE ".nuget\packages\microsoft.ml.onnxruntime.gpu.windows\$OrtNativePackageVersion\runtimes\win-x64\native"
    if (Test-Path -LiteralPath $nugetOrtNativeDir) {
        $OrtNativeDir = $nugetOrtNativeDir
    } else {
        $OrtNativeDir = Join-Path $OrtGenAiBuildDir "_deps\ortlib-src\runtimes\win-x64\native"
    }
}
$OrtGenAiBuildDir = [System.IO.Path]::GetFullPath($OrtGenAiBuildDir)
$OrtGenAiManagedDir = [System.IO.Path]::GetFullPath($OrtGenAiManagedDir)
$OrtNativeDir = [System.IO.Path]::GetFullPath($OrtNativeDir)

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

function Copy-RequiredFile([string]$Source, [string]$Destination, [string]$Label) {
    Assert-PathExists $Source $Label
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
    Copy-Item -LiteralPath $Source -Destination $Destination -Force
}

function Assert-BinaryContainsText([string]$Path, [string]$Needle, [string]$Label) {
    Assert-PathExists $Path $Label
    $bytes = [IO.File]::ReadAllBytes($Path)
    $text = [Text.Encoding]::ASCII.GetString($bytes)
    if (-not $text.Contains($Needle)) {
        throw "$Label does not contain '$Needle'. This binary is not suitable for A100/SM80: $Path"
    }
}

if (-not $SkipAssetValidation) {
    Assert-PathExists (Join-Path $AssetsRoot "models\gemma-4-e2b-it-onnx-mobius\Q4_K_M\cuda\genai_config.json") "Gemma ORT GenAI config"
    Assert-PathExists (Join-Path $AssetsRoot "models\gemma-4-e2b-it-onnx-mobius\Q4_K_M\cuda\embedding\model.onnx") "Gemma embedding graph"
    Assert-PathExists (Join-Path $AssetsRoot "models\gemma-4-e2b-it-onnx-mobius\Q4_K_M\cuda\audio_encoder\model.onnx") "Gemma audio encoder graph"
    Assert-PathExists (Join-Path $AssetsRoot "models\gemma-4-e2b-it-onnx-mobius\Q4_K_M\cuda\decoder\model.onnx") "Gemma decoder graph"
    Assert-PathExists (Join-Path $AssetsRoot "models\chatterbox-onnx\default_voice.wav") "Chatterbox default voice"
    Assert-PathExists (Join-Path $AssetsRoot "models\chatterbox-onnx\tokenizer.json") "Chatterbox tokenizer"
    Assert-PathExists (Join-Path $AssetsRoot "models\chatterbox-onnx\onnx\speech_encoder.onnx") "Chatterbox speech encoder"
    Assert-PathExists (Join-Path $AssetsRoot "models\chatterbox-onnx\onnx\speech_encoder.onnx_data") "Chatterbox speech encoder external data"
    Assert-PathExists (Join-Path $AssetsRoot "models\chatterbox-onnx\onnx\embed_tokens.onnx") "Chatterbox embed tokens"
    Assert-PathExists (Join-Path $AssetsRoot "models\chatterbox-onnx\onnx\embed_tokens.onnx_data") "Chatterbox embed tokens external data"
    Assert-PathExists (Join-Path $AssetsRoot "models\chatterbox-onnx\onnx\language_model_q4f16.onnx") "Chatterbox q4f16 language model"
    Assert-PathExists (Join-Path $AssetsRoot "models\chatterbox-onnx\onnx\language_model_q4f16.onnx_data") "Chatterbox q4f16 language model external data"
    Assert-PathExists (Join-Path $AssetsRoot "models\chatterbox-onnx\onnx\conditional_decoder.onnx") "Chatterbox conditional decoder"
    Assert-PathExists (Join-Path $AssetsRoot "models\chatterbox-onnx\onnx\conditional_decoder.onnx_data") "Chatterbox conditional decoder external data"
}

Assert-PathExists (Join-Path $OrtGenAiBuildDir "onnxruntime-genai.dll") "SM80 ORT GenAI native DLL"
Assert-PathExists (Join-Path $OrtGenAiBuildDir "onnxruntime-genai-cuda.dll") "SM80 ORT GenAI CUDA DLL"
Assert-PathExists (Join-Path $OrtGenAiManagedDir "Microsoft.ML.OnnxRuntimeGenAI.dll") "ORT GenAI managed DLL"
Assert-PathExists (Join-Path $OrtNativeDir "onnxruntime.dll") "SM80 ONNX Runtime DLL"
Assert-PathExists (Join-Path $OrtNativeDir "onnxruntime_providers_cuda.dll") "SM80 ONNX Runtime CUDA provider DLL"
Assert-PathExists (Join-Path $OrtNativeDir "onnxruntime_providers_shared.dll") "SM80 ONNX Runtime shared provider DLL"
Assert-BinaryContainsText (Join-Path $OrtGenAiBuildDir "onnxruntime-genai-cuda.dll") "sm_80" "SM80 ORT GenAI CUDA DLL"
Assert-BinaryContainsText (Join-Path $OrtNativeDir "onnxruntime_providers_cuda.dll") "sm_80" "SM80 ONNX Runtime CUDA provider DLL"

if (-not $SkipPublish) {
    Assert-ChildPath $serviceOut $RuntimeRoot "Publish output"
    if (Test-Path -LiteralPath $serviceOut) {
        Remove-Item -LiteralPath $serviceOut -Recurse -Force
    }
}
New-Item -ItemType Directory -Force -Path $serviceOut | Out-Null

if (-not $SkipPublish) {
    $selfContained = if ($FrameworkDependent) { "false" } else { "true" }
    dotnet publish (Join-Path $repoRoot "src\ChromaOnnx.Service\ChromaOnnx.Service.fsproj") `
        -c $Configuration `
        -r $RuntimeIdentifier `
        --self-contained $selfContained `
        -o $serviceOut `
        /p:OrtGenAiBuildDir=$OrtGenAiBuildDir `
        /p:OrtGenAiManagedDir=$OrtGenAiManagedDir `
        /p:OrtGenAiOrtNativeDir=$OrtNativeDir
}

Copy-RequiredFile (Join-Path $OrtGenAiManagedDir "Microsoft.ML.OnnxRuntimeGenAI.dll") (Join-Path $serviceOut "Microsoft.ML.OnnxRuntimeGenAI.dll") "ORT GenAI managed DLL"
Copy-RequiredFile (Join-Path $OrtGenAiBuildDir "onnxruntime-genai.dll") (Join-Path $serviceOut "onnxruntime-genai.dll") "SM80 ORT GenAI native DLL"
Copy-RequiredFile (Join-Path $OrtGenAiBuildDir "onnxruntime-genai-cuda.dll") (Join-Path $serviceOut "onnxruntime-genai-cuda.dll") "SM80 ORT GenAI CUDA DLL"
Copy-RequiredFile (Join-Path $OrtNativeDir "onnxruntime.dll") (Join-Path $serviceOut "onnxruntime.dll") "SM80 ONNX Runtime DLL"
Copy-RequiredFile (Join-Path $OrtNativeDir "onnxruntime_providers_cuda.dll") (Join-Path $serviceOut "onnxruntime_providers_cuda.dll") "SM80 ONNX Runtime CUDA provider DLL"
Copy-RequiredFile (Join-Path $OrtNativeDir "onnxruntime_providers_shared.dll") (Join-Path $serviceOut "onnxruntime_providers_shared.dll") "SM80 ONNX Runtime shared provider DLL"
if (Test-Path -LiteralPath (Join-Path $OrtNativeDir "onnxruntime_providers_tensorrt.dll")) {
    Copy-Item -LiteralPath (Join-Path $OrtNativeDir "onnxruntime_providers_tensorrt.dll") -Destination (Join-Path $serviceOut "onnxruntime_providers_tensorrt.dll") -Force
}

foreach ($file in @(
    "ChromaOnnx.Service.exe",
    "ChromaOnnx.Service.dll",
    "ChromaOnnx.Service.deps.json",
    "ChromaOnnx.Service.runtimeconfig.json",
    "ChromaOnnx.SpeechToSpeech.dll",
    "ChromaOnnx.OnnxRuntime.dll",
    "ChromaOnnx.Core.dll",
    "Microsoft.ML.OnnxRuntime.dll",
    "Microsoft.ML.OnnxRuntimeGenAI.dll",
    "Tokenizers.HuggingFace.dll",
    "onnxruntime.dll",
    "onnxruntime-genai.dll",
    "onnxruntime-genai-cuda.dll",
    "onnxruntime_providers_cuda.dll",
    "onnxruntime_providers_shared.dll"
)) {
    Assert-PathExists (Join-Path $serviceOut $file) "Published dependency $file"
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
    [ValidateSet("chatterbox-onnx", "fake-tone")]
    [string]$TtsRuntime = "chatterbox-onnx",
    [string]$TtsModelDir = "",
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

function Find-DllForProcess([string]$DllName, [string[]]$ExtraDirectories) {
    foreach ($dir in $ExtraDirectories) {
        if ($dir -and (Test-Path -LiteralPath (Join-Path $dir $DllName))) {
            return (Join-Path $dir $DllName)
        }
    }
    $pathValue = $env:PATH
    if (-not [string]::IsNullOrWhiteSpace($pathValue)) {
        foreach ($dir in $pathValue.Split([IO.Path]::PathSeparator, [StringSplitOptions]::RemoveEmptyEntries)) {
            if ($dir -and (Test-Path -LiteralPath (Join-Path $dir $DllName))) {
                return (Join-Path $dir $DllName)
            }
        }
    }
    return $null
}

function Assert-CudaRuntimeDlls([string]$ServiceDir) {
    $requiredDlls = @(
        "cudart64_12.dll",
        "cublas64_12.dll",
        "cublasLt64_12.dll",
        "cufft64_11.dll",
        "curand64_10.dll",
        "cudnn64_9.dll"
    )
    $missing = @()
    foreach ($dll in $requiredDlls) {
        if (-not (Find-DllForProcess $dll @($ServiceDir))) {
            $missing += $dll
        }
    }
    if ($missing.Count -gt 0) {
        throw "Missing CUDA/cuDNN runtime DLL(s) required by the packaged ONNX Runtime CUDA provider: $($missing -join ', '). Install/stage CUDA 12.x runtime plus cuDNN 9, copy the DLLs next to ChromaOnnx.Service.exe, or rerun this script with -CudaBin and -CudnnBin pointing at their bin folders."
    }
}

if ($GemmaModelDir) {
    $selectedGemmaModelDir = [System.IO.Path]::GetFullPath($GemmaModelDir)
} else {
    $selectedGemmaModelDir = Join-Path $AssetsRoot "models\gemma-4-e2b-it-onnx-mobius\Q4_K_M\cuda"
}
if ($TtsModelDir) {
    $selectedTtsModelDir = [System.IO.Path]::GetFullPath($TtsModelDir)
} else {
    $selectedTtsModelDir = Join-Path $AssetsRoot "models\chatterbox-onnx"
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
if ($TtsRuntime -eq "chatterbox-onnx") {
    $requiredPaths += @(
        (Join-Path $selectedTtsModelDir "default_voice.wav"),
        (Join-Path $selectedTtsModelDir "tokenizer.json"),
        (Join-Path $selectedTtsModelDir "onnx\speech_encoder.onnx"),
        (Join-Path $selectedTtsModelDir "onnx\speech_encoder.onnx_data"),
        (Join-Path $selectedTtsModelDir "onnx\embed_tokens.onnx"),
        (Join-Path $selectedTtsModelDir "onnx\embed_tokens.onnx_data"),
        (Join-Path $selectedTtsModelDir "onnx\language_model_q4f16.onnx"),
        (Join-Path $selectedTtsModelDir "onnx\language_model_q4f16.onnx_data"),
        (Join-Path $selectedTtsModelDir "onnx\conditional_decoder.onnx"),
        (Join-Path $selectedTtsModelDir "onnx\conditional_decoder.onnx_data")
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
Assert-CudaRuntimeDlls $serviceDir
if ($TtsRuntime -eq "chatterbox-onnx") {
    $speechDll = Join-Path $serviceDir "ChromaOnnx.SpeechToSpeech.dll"
    $speechDllText = [Text.Encoding]::Unicode.GetString([IO.File]::ReadAllBytes($speechDll))
    if (-not $speechDllText.Contains("chatterbox-onnx")) {
        throw "The deployed ChromaOnnx.SpeechToSpeech.dll does not contain the Chatterbox runtime. Re-apply the latest VoiceAgent A100 runtime diff."
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
$env:VoiceAgent__Tts__Variant = "q4f16"
$env:VoiceAgent__Tts__VoiceSamplePath = $VoiceSamplePath
$env:VoiceAgent__Tts__VoiceSampleTranscript = $VoiceSampleTranscript
$env:VoiceAgent__Tts__OutputSampleRate = "24000"
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
Write-Host "VoiceSamplePath: $VoiceSamplePath"
& (Join-Path $serviceDir "ChromaOnnx.Service.exe") --urls $url
exit $LASTEXITCODE
'@

$readme = @'
# Gemma Voice Agent A100 Runtime

This package runs the text-first voice-agent path:

user audio -> Gemma 4 ASR -> Gemma 4 reasoning/tool calls/filler text -> Chatterbox filler/final TTS.

Default TTS runtime is `chatterbox-onnx`, which loads the Chatterbox ONNX graphs
in-process and keeps sessions warm. It expects:

- `VoiceAgent_assets\models\chatterbox-onnx\default_voice.wav`
- `VoiceAgent_assets\models\chatterbox-onnx\tokenizer.json`
- `VoiceAgent_assets\models\chatterbox-onnx\onnx\speech_encoder.onnx`
- `VoiceAgent_assets\models\chatterbox-onnx\onnx\speech_encoder.onnx_data`
- `VoiceAgent_assets\models\chatterbox-onnx\onnx\embed_tokens.onnx`
- `VoiceAgent_assets\models\chatterbox-onnx\onnx\embed_tokens.onnx_data`
- `VoiceAgent_assets\models\chatterbox-onnx\onnx\language_model_q4f16.onnx`
- `VoiceAgent_assets\models\chatterbox-onnx\onnx\language_model_q4f16.onnx_data`
- `VoiceAgent_assets\models\chatterbox-onnx\onnx\conditional_decoder.onnx`
- `VoiceAgent_assets\models\chatterbox-onnx\onnx\conditional_decoder.onnx_data`

Chatterbox voice cloning uses `VoiceAgent:Tts:VoiceSamplePath` as a reference
WAV when provided; otherwise it uses `default_voice.wav`.

The ONNX model files are hardware-neutral. The packaged ORT GenAI CUDA DLLs are
from the SM80/A100 ORT GenAI build:

`$OrtGenAiBuildDir`

`onnxruntime.dll` and `onnxruntime_providers_cuda.dll` are copied from the
SM80-capable ONNX Runtime native folder:

`$OrtNativeDir`

The host must also expose CUDA 12.x runtime DLLs and cuDNN 9 DLLs on `PATH`, or
those DLLs must be copied next to `app\service\ChromaOnnx.Service.exe`. At
minimum the CUDA provider needs `cudart64_12.dll`, `cublas64_12.dll`,
`cublasLt64_12.dll`, `cufft64_11.dll`, `curand64_10.dll`, and `cudnn64_9.dll`.
Use `-CudaBin` and `-CudnnBin` when they are not already on `PATH`.

Gemma assets are not included in this runtime package and must be staged under:

`VoiceAgent_assets\models\gemma-4-e2b-it-onnx-mobius\Q4_K_M\cuda`

Run service. Use the default `TtsMaxSteps=256` for browser testing; `64` is only
for short smoke tests and can truncate or degrade long answers.

```powershell
.\run-voice-agent-a100.ps1 -AssetsRoot G:\Chroma\VoiceAgent_assets -TtsRuntime chatterbox-onnx
```

Run with a voice-clone reference WAV:

```powershell
.\run-voice-agent-a100.ps1 -AssetsRoot G:\Chroma\VoiceAgent_assets -TtsRuntime chatterbox-onnx -VoiceSamplePath G:\Chroma\VoiceAgent_assets\voices\reference.wav
```

Smoke test with fake TTS plumbing:

```powershell
.\smoke-voice-agent-a100.ps1 -AssetsRoot G:\Chroma\VoiceAgent_assets -TtsRuntime fake-tone
```

Smoke test with real Chatterbox:

```powershell
.\smoke-voice-agent-a100.ps1 -AssetsRoot G:\Chroma\VoiceAgent_assets -TtsRuntime chatterbox-onnx -TtsMaxSteps 96
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
TTS runtime: chatterbox-onnx
ORT GenAI build dir: $OrtGenAiBuildDir
ORT native dir: $OrtNativeDir
CUDA target: SM80/A100
ORT native package version: $OrtNativePackageVersion
Voice cloning: reference WAV via VoiceAgent:Tts:VoiceSamplePath
Models included: no
"@
Set-Content -LiteralPath (Join-Path $RuntimeRoot "VOICE_AGENT_MANIFEST.txt") -Value $manifest -Encoding UTF8

Write-Host "Gemma Voice Agent A100 runtime is ready at $RuntimeRoot"
Write-Host "ServiceDir: $serviceOut"
Write-Host "AssetsRoot: $AssetsRoot"
Write-Host "ORT GenAI build dir: $OrtGenAiBuildDir"
Write-Host "ORT native dir: $OrtNativeDir"
