param(
    [string]$AssetsRoot = "E:\s\temp\Chroma_ONNX_assets",
    [int]$Frames = 1,
    [int]$CudaGpuMemLimitMb = 15360,
    [string]$CudaBin = "",
    [string]$CudnnBin = ""
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptRoot
$AssetsRoot = [System.IO.Path]::GetFullPath($AssetsRoot)

if ($CudaBin) {
    $env:PATH = "$CudaBin;$env:PATH"
}

if ($CudnnBin) {
    $env:PATH = "$CudnnBin;$env:PATH"
}

$outputDir = Join-Path $repoRoot "served_runs\smoke\local_shared_assets"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$args = @(
    "s2s-offline",
    "--model-dir", (Join-Path $AssetsRoot "models\chroma-4b"),
    "--bundle-dir", (Join-Path $repoRoot "onnx_deploy\chroma-s2s-full-v2"),
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

Write-Host "Running local repo smoke test with shared assets..."
Write-Host "AssetsRoot: $AssetsRoot"

Push-Location $repoRoot
try {
    dotnet run --project src\ChromaOnnx -- @args
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
