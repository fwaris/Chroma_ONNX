param(
    [string]$AssetsRoot = "E:\s\temp\Chroma_ONNX_assets",
    [int]$Port = 5055,
    [int]$CudaGpuMemLimitMb = 15360,
    [int]$StreamMinFreeVramMb = 2048,
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

$env:ChromaOnnx__S2s__ModelDir = Join-Path $AssetsRoot "models\chroma-4b"
$env:ChromaOnnx__S2s__BundleDir = Join-Path $repoRoot "onnx_deploy\chroma-s2s-full-v2"
$env:ChromaOnnx__S2s__WorkDir = Join-Path $repoRoot "served_runs"
$env:ChromaOnnx__S2s__ExecutionProvider = "cuda"
$env:ChromaOnnx__S2s__MemoryMode = "resident-merged"
$env:ChromaOnnx__S2s__OrtMemoryProfile = "quality-safe"
$env:ChromaOnnx__S2s__OptimizedModelCacheDir = Join-Path $AssetsRoot "onnx\chroma-s2s-full-v2\ort-cache-ort-local-external"
$env:ChromaOnnx__S2s__OptimizedModelCacheFormat = "onnx"
$env:ChromaOnnx__S2s__CudaGpuMemLimitMb = [string]$CudaGpuMemLimitMb
$env:ChromaOnnx__S2s__StreamMinFreeVramMb = [string]$StreamMinFreeVramMb

Write-Host "Running local service from repo code."
Write-Host "AssetsRoot: $AssetsRoot"
Write-Host "BundleDir: $env:ChromaOnnx__S2s__BundleDir"

Push-Location $repoRoot
try {
    dotnet run --project src\ChromaOnnx.Service --urls "http://localhost:$Port"
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
