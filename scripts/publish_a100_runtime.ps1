param(
    [string]$RuntimeRoot = "E:\s\temp\Chroma_ONNX_A100_runtime",
    [string]$AssetsRoot = "E:\s\temp\Chroma_ONNX_assets",
    [string]$Configuration = "Release",
    [string]$RuntimeIdentifier = "win-x64",
    [switch]$SkipPublish,
    [switch]$FrameworkDependent
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptRoot
$RuntimeRoot = [System.IO.Path]::GetFullPath($RuntimeRoot)
$AssetsRoot = [System.IO.Path]::GetFullPath($AssetsRoot)

$serviceOut = Join-Path $RuntimeRoot "app\service"
$cliOut = Join-Path $RuntimeRoot "app\cli"
$deployRoot = Join-Path $RuntimeRoot "onnx_deploy"
$fixtureOut = Join-Path $RuntimeRoot "served_runs\compare_inputs"
$oneShotBundleName = "chroma-s2s-full-v2"

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

Assert-PathExists (Join-Path $AssetsRoot "models\chroma-4b\model-00001-of-00003.safetensors") "Shared model shard"
Assert-PathExists (Join-Path $AssetsRoot "onnx\chroma-s2s-full-v2\ort-cache-ort-local-external\s2s_merged.cuda.quality-safe.optimized.onnx") "Shared optimized CUDA cache"

if (-not $SkipPublish) {
    foreach ($out in @($serviceOut, $cliOut)) {
        Assert-ChildPath $out $RuntimeRoot "Publish output"
        if (Test-Path -LiteralPath $out) {
            Remove-Item -LiteralPath $out -Recurse -Force
        }
    }
}

Assert-ChildPath $deployRoot $RuntimeRoot "Deploy output"
if (Test-Path -LiteralPath $deployRoot) {
    Remove-Item -LiteralPath $deployRoot -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $serviceOut, $cliOut, $deployRoot, $fixtureOut | Out-Null

if (-not $SkipPublish) {
    $selfContained = if ($FrameworkDependent) { "false" } else { "true" }

    dotnet publish (Join-Path $repoRoot "src\ChromaOnnx.Service\ChromaOnnx.Service.fsproj") `
        -c $Configuration `
        -r $RuntimeIdentifier `
        --self-contained $selfContained `
        -o $serviceOut

    dotnet publish (Join-Path $repoRoot "src\ChromaOnnx\ChromaOnnx.fsproj") `
        -c $Configuration `
        -r $RuntimeIdentifier `
        --self-contained $selfContained `
        -o $cliOut
}

function Copy-Bundle([string]$BundleName, [bool]$Required) {
    $sourceDir = Join-Path $repoRoot "onnx_deploy\$BundleName"
    $targetDir = Join-Path $deployRoot $BundleName
    $graphPath = Join-Path $sourceDir "chroma_s2s_merged.weights_free.onnx"
    $manifestPath = Join-Path $sourceDir "shared_weights_manifest.json"

    if (-not ((Test-Path -LiteralPath $graphPath) -and (Test-Path -LiteralPath $manifestPath))) {
        if ($Required) {
            throw "Required ONNX bundle was not found or is incomplete: $sourceDir"
        }
        Write-Host "Optional ONNX bundle not present, skipping: $BundleName"
        return
    }

    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
    Copy-Item -LiteralPath $graphPath -Destination $targetDir -Force
    Copy-Item -LiteralPath $manifestPath -Destination $targetDir -Force

    $bundleManifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    $graphMode = if ($bundleManifest.capabilities -and $bundleManifest.capabilities.s2s_graph_mode) { $bundleManifest.capabilities.s2s_graph_mode } else { "one-shot" }
    $featureMode = if ($bundleManifest.capabilities -and $bundleManifest.capabilities.thinker_feature_mode) { $bundleManifest.capabilities.thinker_feature_mode } else { "unknown" }
    $maxAudioItems = if ($bundleManifest.capabilities -and $bundleManifest.capabilities.thinker_max_audio_items) { [int]$bundleManifest.capabilities.thinker_max_audio_items } else { 1 }
    Write-Host "Copied bundle '$BundleName' ($graphMode): thinker feature mode $featureMode, max audio items $maxAudioItems"
    if ($maxAudioItems -gt 1) {
        Write-Warning "Chroma S2S is current-turn-only; extra thinker audio item capacity in '$BundleName' is ignored."
    }
}

Copy-Bundle $oneShotBundleName $true

$assetFixture = Join-Path $AssetsRoot "served_runs\compare_inputs"
if (Test-Path -LiteralPath $assetFixture) {
    Get-ChildItem -LiteralPath $assetFixture -File | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $fixtureOut -Force
    }
}

$runService = @'
param(
    [string]$AssetsRoot = "",
    [int]$Port = 5055,
    [int]$CudaGpuMemLimitMb = 22528,
    [int]$StreamMinFreeVramMb = 3072,
    [ValidateSet("greedy", "sample")]
    [string]$GenerationMode = "sample",
    [ValidateSet("top-k-top-p", "chroma")]
    [string]$SamplingAlgorithm = "top-k-top-p",
    [string]$BundleDir = "",
    [string]$OptimizedModelCacheDir = "",
    [string]$GemmaModelDir = "",
    [string]$GemmaVariant = "Q4_K_M/cuda",
    [ValidateSet("ort-genai", "raw-ort")]
    [string]$GemmaRuntime = "ort-genai",
    [ValidateSet("cpu", "cuda")]
    [string]$GemmaExecutionProvider = "cuda",
    [int]$MaxNewFrames = 900,
    [string]$CudaBin = "",
    [string]$CudnnBin = "",
    [switch]$LocalhostOnly
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $AssetsRoot) {
    $AssetsRoot = Join-Path (Split-Path -Parent $Root) "Chroma_ONNX_assets"
}
$AssetsRoot = [System.IO.Path]::GetFullPath($AssetsRoot)

if ($CudaBin) { $env:PATH = "$CudaBin;$env:PATH" }
if ($CudnnBin) { $env:PATH = "$CudnnBin;$env:PATH" }

$oneShotBundleName = "chroma-s2s-full-v2"
$bundleRoot = Join-Path $Root "onnx_deploy"
$oneShotBundleDir = Join-Path $bundleRoot $oneShotBundleName

if ($BundleDir) {
    $selectedBundleDir = [System.IO.Path]::GetFullPath($BundleDir)
    $selectedBundleMode = "custom"
} else {
    $selectedBundleDir = $oneShotBundleDir
    $selectedBundleMode = "one-shot"
}

if ($OptimizedModelCacheDir) {
    $selectedOptimizedModelCacheDir = [System.IO.Path]::GetFullPath($OptimizedModelCacheDir)
} else {
    $selectedBundleName = Split-Path -Leaf $selectedBundleDir
    $selectedOptimizedModelCacheDir = Join-Path $AssetsRoot "onnx\$selectedBundleName\ort-cache-ort-local-external"
}

if ($GemmaModelDir) {
    $selectedGemmaModelDir = [System.IO.Path]::GetFullPath($GemmaModelDir)
} else {
    $selectedGemmaModelDir = Join-Path $AssetsRoot "models\gemma-4-e2b-it-onnx-mobius\Q4_K_M\cuda"
}

$env:ChromaOnnx__S2s__ModelDir = Join-Path $AssetsRoot "models\chroma-4b"
$env:ChromaOnnx__S2s__BundleDir = $selectedBundleDir
$env:ChromaOnnx__S2s__WorkDir = Join-Path $Root "served_runs"
$env:ChromaOnnx__S2s__ExecutionProvider = "cuda"
$env:ChromaOnnx__S2s__MemoryMode = "resident-merged"
$env:ChromaOnnx__S2s__OrtMemoryProfile = "quality-safe"
$env:ChromaOnnx__S2s__OptimizedModelCacheDir = $selectedOptimizedModelCacheDir
$env:ChromaOnnx__S2s__OptimizedModelCacheFormat = "onnx"
$env:ChromaOnnx__S2s__CudaGpuMemLimitMb = [string]$CudaGpuMemLimitMb
$env:ChromaOnnx__S2s__StreamMinFreeVramMb = [string]$StreamMinFreeVramMb
$env:ChromaOnnx__S2s__GenerationMode = $GenerationMode
$env:ChromaOnnx__S2s__SamplingAlgorithm = $SamplingAlgorithm
$env:ChromaOnnx__S2s__MaxNewFrames = [string]$MaxNewFrames
$env:ChromaOnnx__Gemma__ModelDir = $selectedGemmaModelDir
$env:ChromaOnnx__Gemma__Variant = $GemmaVariant
$env:ChromaOnnx__Gemma__Runtime = $GemmaRuntime
$env:ChromaOnnx__Gemma__ExecutionProvider = $GemmaExecutionProvider

$serviceDir = Join-Path $Root "app\service"
$requiredPaths = @(
    (Join-Path $Root "app\service\ChromaOnnx.Service.exe"),
    (Join-Path $serviceDir "onnxruntime.dll"),
    (Join-Path $serviceDir "onnxruntime_providers_cuda.dll"),
    (Join-Path $serviceDir "onnxruntime_providers_shared.dll"),
    $env:ChromaOnnx__S2s__ModelDir,
    $env:ChromaOnnx__S2s__BundleDir,
    $env:ChromaOnnx__S2s__OptimizedModelCacheDir,
    $env:ChromaOnnx__Gemma__ModelDir,
    (Join-Path $env:ChromaOnnx__Gemma__ModelDir "genai_config.json"),
    (Join-Path $env:ChromaOnnx__Gemma__ModelDir "tokenizer.json"),
    (Join-Path $env:ChromaOnnx__Gemma__ModelDir "embedding\model.onnx"),
    (Join-Path $env:ChromaOnnx__Gemma__ModelDir "audio_encoder\model.onnx"),
    (Join-Path $env:ChromaOnnx__Gemma__ModelDir "decoder\model.onnx")
)
if ($GemmaRuntime -eq "ort-genai") {
    $requiredPaths += @(
        (Join-Path $serviceDir "Microsoft.ML.OnnxRuntimeGenAI.dll"),
        (Join-Path $serviceDir "onnxruntime-genai.dll"),
        (Join-Path $serviceDir "onnxruntime-genai-cuda.dll")
    )
}
foreach ($path in $requiredPaths) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required path is missing: $path"
    }
}

$bundleManifestPath = Join-Path $env:ChromaOnnx__S2s__BundleDir "shared_weights_manifest.json"
$bundleThinkerMaxAudioItems = 1
$bundleGraphMode = $selectedBundleMode
if (Test-Path -LiteralPath $bundleManifestPath) {
    $bundleManifest = Get-Content -LiteralPath $bundleManifestPath -Raw | ConvertFrom-Json
    if ($bundleManifest.capabilities -and $bundleManifest.capabilities.s2s_graph_mode) {
        $bundleGraphMode = [string]$bundleManifest.capabilities.s2s_graph_mode
    }
    if ($bundleManifest.capabilities -and $bundleManifest.capabilities.thinker_max_audio_items) {
        $bundleThinkerMaxAudioItems = [int]$bundleManifest.capabilities.thinker_max_audio_items
    }
}
if ($bundleThinkerMaxAudioItems -gt 1) {
    Write-Warning "Chroma S2S is current-turn-only; extra thinker audio item capacity in this bundle is ignored."
}

$hostName = if ($LocalhostOnly) { "localhost" } else { "0.0.0.0" }
$url = "http://${hostName}:$Port"
Write-Host "Starting ChromaS2SONNX on $url"
Write-Host "AssetsRoot: $AssetsRoot"
Write-Host "SelectedBundleMode: $selectedBundleMode"
Write-Host "BundleGraphMode: $bundleGraphMode"
Write-Host "BundleDir: $selectedBundleDir"
Write-Host "OptimizedModelCacheDir: $selectedOptimizedModelCacheDir"
Write-Host "GenerationMode: $GenerationMode"
Write-Host "SamplingAlgorithm: $SamplingAlgorithm"
Write-Host "CudaGpuMemLimitMb: $CudaGpuMemLimitMb"
Write-Host "StreamMinFreeVramMb: $StreamMinFreeVramMb"
Write-Host "MaxNewFrames: $MaxNewFrames"
Write-Host "GemmaModelDir: $selectedGemmaModelDir"
Write-Host "GemmaVariant: $GemmaVariant"
Write-Host "GemmaRuntime: $GemmaRuntime"
Write-Host "GemmaExecutionProvider: $GemmaExecutionProvider"
Write-Host "Bundle thinker audio items: $bundleThinkerMaxAudioItems"
& (Join-Path $Root "app\service\ChromaOnnx.Service.exe") --urls $url
exit $LASTEXITCODE
'@

$smokeTest = @'
param(
    [string]$AssetsRoot = "",
    [int]$Frames = 1,
    [int]$CudaGpuMemLimitMb = 22528,
    [string]$CudaBin = "",
    [string]$CudnnBin = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $AssetsRoot) {
    $AssetsRoot = Join-Path (Split-Path -Parent $Root) "Chroma_ONNX_assets"
}
$AssetsRoot = [System.IO.Path]::GetFullPath($AssetsRoot)

if ($CudaBin) { $env:PATH = "$CudaBin;$env:PATH" }
if ($CudnnBin) { $env:PATH = "$CudnnBin;$env:PATH" }

$outputDir = Join-Path $Root "served_runs\smoke\a100"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$args = @(
    "s2s-offline",
    "--model-dir", (Join-Path $AssetsRoot "models\chroma-4b"),
    "--bundle-dir", (Join-Path $Root "onnx_deploy\chroma-s2s-full-v2"),
    "--prompt-text", "War",
    "--prompt-audio-f32", (Join-Path $AssetsRoot "served_runs\compare_inputs\reference_audio_24k.f32"),
    "--user-audio-f32", (Join-Path $AssetsRoot "served_runs\compare_inputs\make_taco_16k.f32"),
    "--frames", ([string]$Frames),
    "--output-dir", $outputDir,
    "--execution-provider", "cuda",
    "--memory-mode", "resident-merged",
    "--ort-memory-profile", "quality-safe",
    "--thinker-active-frames", "0",
    "--optimized-model-cache-dir", (Join-Path $AssetsRoot "onnx\chroma-s2s-full-v2\ort-cache-ort-local-external"),
    "--optimized-model-cache-format", "onnx",
    "--cuda-gpu-mem-limit-mb", ([string]$CudaGpuMemLimitMb)
)

Write-Host "Running $Frames-frame CUDA smoke test..."
Write-Host "AssetsRoot: $AssetsRoot"
& (Join-Path $Root "app\cli\ChromaOnnx.exe") @args
exit $LASTEXITCODE
'@

$repairLinks = Get-Content -LiteralPath (Join-Path $scriptRoot "repair_cache_links.ps1") -Raw

$readme = @"
# Chroma ONNX A100 Runtime

This is the small, replaceable runtime folder. It intentionally does not contain the large model or optimized cache.

Expected sibling layout:

```text
Chroma_ONNX_A100_runtime\
Chroma_ONNX_assets\
  models\chroma-4b\
  models\gemma-4-e2b-it-onnx-mobius\Q4_K_M\cuda\
  onnx\chroma-s2s-full-v2\ort-cache-ort-local-external\
```

Run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\smoke-test-a100.ps1
.\run-service-a100.ps1
```

The runtime carries the current-turn Chroma S2S bundle:

```text
onnx_deploy\chroma-s2s-full-v2\
```

To test a custom bundle, pass it directly:

```powershell
.\run-service-a100.ps1 -BundleDir E:\path\to\onnx_deploy\chroma-s2s-full-v2
```

Chroma S2S is current-turn-only. Agent-level history should live outside Chroma and call Chroma once per vocalized turn.

Gemma defaults to the Mobius `Q4_K_M/cuda` export and runs through ORT GenAI with CUDA.

If the shared assets are somewhere else:

```powershell
.\smoke-test-a100.ps1 -AssetsRoot E:\path\to\Chroma_ONNX_assets
.\run-service-a100.ps1 -AssetsRoot E:\path\to\Chroma_ONNX_assets
.\repair-cache-links.ps1 -ModelDir E:\path\to\Chroma_ONNX_assets\models\chroma-4b
```

The runtime is self-contained for .NET. The host still needs the NVIDIA driver plus the CUDA/cuDNN DLLs expected by the packaged ONNX Runtime build on PATH. The current ORT GenAI build is ORT 1.28-based and expects CUDA 12.x runtime DLLs with cuDNN 9; pass `-CudaBin` and `-CudnnBin` if they are not already on PATH.
"@

Set-Content -LiteralPath (Join-Path $RuntimeRoot "run-service-a100.ps1") -Value $runService -Encoding ASCII
Set-Content -LiteralPath (Join-Path $RuntimeRoot "smoke-test-a100.ps1") -Value $smokeTest -Encoding ASCII
Set-Content -LiteralPath (Join-Path $RuntimeRoot "repair-cache-links.ps1") -Value $repairLinks -Encoding ASCII
Set-Content -LiteralPath (Join-Path $RuntimeRoot "README_RUNTIME_A100.md") -Value $readme -Encoding ASCII

$manifest = @"
Created: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
Source repo: $repoRoot
Runtime root: $RuntimeRoot
Assets root: $AssetsRoot
Service: app\service\ChromaOnnx.Service.exe
CLI: app\cli\ChromaOnnx.exe
Deploy bundles: onnx_deploy
Gemma runtime: ORT GenAI
ORT native DLLs: app\service\onnxruntime*.dll
"@

Set-Content -LiteralPath (Join-Path $RuntimeRoot "RUNTIME_MANIFEST.txt") -Value $manifest -Encoding ASCII

Write-Host "Runtime is ready at $RuntimeRoot"
Write-Host "Shared assets expected at $AssetsRoot"
