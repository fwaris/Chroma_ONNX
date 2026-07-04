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
$deployOut = Join-Path $RuntimeRoot "onnx_deploy\chroma-s2s-full-v2"
$fixtureOut = Join-Path $RuntimeRoot "served_runs\compare_inputs"

function Assert-PathExists([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Label was not found: $Path"
    }
}

Assert-PathExists (Join-Path $AssetsRoot "models\chroma-4b\model-00001-of-00003.safetensors") "Shared model shard"
Assert-PathExists (Join-Path $AssetsRoot "onnx\chroma-s2s-full-v2\ort-cache-ort-local-external\s2s_merged.cuda.quality-safe.optimized.onnx") "Shared optimized CUDA cache"

New-Item -ItemType Directory -Force -Path $serviceOut, $cliOut, $deployOut, $fixtureOut | Out-Null

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

Copy-Item -LiteralPath (Join-Path $repoRoot "onnx_deploy\chroma-s2s-full-v2\chroma_s2s_merged.weights_free.onnx") -Destination $deployOut -Force
Copy-Item -LiteralPath (Join-Path $repoRoot "onnx_deploy\chroma-s2s-full-v2\shared_weights_manifest.json") -Destination $deployOut -Force

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
    [int]$CudaGpuMemLimitMb = 32768,
    [int]$StreamMinFreeVramMb = 4096,
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

$env:ChromaOnnx__S2s__ModelDir = Join-Path $AssetsRoot "models\chroma-4b"
$env:ChromaOnnx__S2s__BundleDir = Join-Path $Root "onnx_deploy\chroma-s2s-full-v2"
$env:ChromaOnnx__S2s__WorkDir = Join-Path $Root "served_runs"
$env:ChromaOnnx__S2s__ExecutionProvider = "cuda"
$env:ChromaOnnx__S2s__MemoryMode = "resident-merged"
$env:ChromaOnnx__S2s__OrtMemoryProfile = "quality-safe"
$env:ChromaOnnx__S2s__OptimizedModelCacheDir = Join-Path $AssetsRoot "onnx\chroma-s2s-full-v2\ort-cache-ort-local-external"
$env:ChromaOnnx__S2s__OptimizedModelCacheFormat = "onnx"
$env:ChromaOnnx__S2s__CudaGpuMemLimitMb = [string]$CudaGpuMemLimitMb
$env:ChromaOnnx__S2s__StreamMinFreeVramMb = [string]$StreamMinFreeVramMb
$env:ChromaOnnx__S2s__MaxNewFrames = [string]$MaxNewFrames

foreach ($path in @(
    (Join-Path $Root "app\service\ChromaOnnx.Service.exe"),
    $env:ChromaOnnx__S2s__ModelDir,
    $env:ChromaOnnx__S2s__BundleDir,
    $env:ChromaOnnx__S2s__OptimizedModelCacheDir
)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required path is missing: $path"
    }
}

$hostName = if ($LocalhostOnly) { "localhost" } else { "0.0.0.0" }
$url = "http://${hostName}:$Port"
Write-Host "Starting ChromaS2SONNX on $url"
Write-Host "AssetsRoot: $AssetsRoot"
Write-Host "MaxNewFrames: $MaxNewFrames"
& (Join-Path $Root "app\service\ChromaOnnx.Service.exe") --urls $url
exit $LASTEXITCODE
'@

$smokeTest = @'
param(
    [string]$AssetsRoot = "",
    [int]$Frames = 1,
    [int]$CudaGpuMemLimitMb = 32768,
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
```

Run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\smoke-test-a100.ps1
.\run-service-a100.ps1
```

If the shared assets are somewhere else:

```powershell
.\smoke-test-a100.ps1 -AssetsRoot E:\path\to\Chroma_ONNX_assets
.\run-service-a100.ps1 -AssetsRoot E:\path\to\Chroma_ONNX_assets
.\repair-cache-links.ps1 -ModelDir E:\path\to\Chroma_ONNX_assets\models\chroma-4b
```

The runtime is self-contained for .NET. The host still needs NVIDIA driver, CUDA 13.x DLLs, and cuDNN 9 on PATH.
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
Deploy bundle: onnx_deploy\chroma-s2s-full-v2
"@

Set-Content -LiteralPath (Join-Path $RuntimeRoot "RUNTIME_MANIFEST.txt") -Value $manifest -Encoding ASCII

Write-Host "Runtime is ready at $RuntimeRoot"
Write-Host "Shared assets expected at $AssetsRoot"
