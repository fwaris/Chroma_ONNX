param(
    [string]$AssetsRoot = "E:\s\temp\Chroma_ONNX_assets",
    [string]$ModelSource = "",
    [string]$CacheSource = "",
    [string]$FixtureSource = "",
    [switch]$ForceRelink,
    [switch]$CopyIfHardlinkFails
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptRoot

if (-not $ModelSource) {
    $ModelSource = Join-Path $repoRoot "models\chroma-4b"
}

if (-not $CacheSource) {
    $CacheSource = Join-Path $repoRoot "onnx\chroma-s2s-full-v2\ort-cache-ort-local-external"
}

if (-not $FixtureSource) {
    $FixtureSource = Join-Path $repoRoot "served_runs\compare_inputs"
}

$AssetsRoot = [System.IO.Path]::GetFullPath($AssetsRoot)
$modelTarget = Join-Path $AssetsRoot "models\chroma-4b"
$cacheTarget = Join-Path $AssetsRoot "onnx\chroma-s2s-full-v2\ort-cache-ort-local-external"
$fixtureTarget = Join-Path $AssetsRoot "served_runs\compare_inputs"

function Assert-PathExists([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Label was not found: $Path"
    }
}

function Link-OrCopyFile([string]$Source, [string]$Target) {
    if ((Test-Path -LiteralPath $Target) -and -not $ForceRelink) {
        return
    }

    if (Test-Path -LiteralPath $Target) {
        Remove-Item -LiteralPath $Target -Force
    }

    $targetDir = Split-Path -Parent $Target
    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null

    try {
        New-Item -ItemType HardLink -Path $Target -Target $Source | Out-Null
    }
    catch {
        if (-not $CopyIfHardlinkFails) {
            throw
        }

        Copy-Item -LiteralPath $Source -Destination $Target -Force
    }
}

Assert-PathExists $ModelSource "Model source"
Assert-PathExists $CacheSource "Optimized cache source"
Assert-PathExists $FixtureSource "Fixture source"

New-Item -ItemType Directory -Force -Path $modelTarget, $cacheTarget, $fixtureTarget | Out-Null

robocopy $ModelSource $modelTarget /E /XD .cache __pycache__ /XF *.safetensors | Out-Host
if ($LASTEXITCODE -gt 7) {
    throw "robocopy model metadata failed with exit code $LASTEXITCODE"
}

foreach ($name in @("model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors")) {
    $source = Join-Path $ModelSource $name
    $target = Join-Path $modelTarget $name
    Assert-PathExists $source "Model shard"
    Link-OrCopyFile $source $target
}

foreach ($name in @("chroma_s2s_merged.local_external.onnx", "s2s_merged.cuda.quality-safe.optimized.onnx", "local_external_cache_report.json")) {
    $source = Join-Path $CacheSource $name
    if (Test-Path -LiteralPath $source) {
        Copy-Item -LiteralPath $source -Destination $cacheTarget -Force
    }
}

foreach ($name in @("model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors")) {
    $source = Join-Path $modelTarget $name
    $target = Join-Path $cacheTarget $name
    Link-OrCopyFile $source $target
}

Get-ChildItem -LiteralPath $FixtureSource -File | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $fixtureTarget -Force
}

$manifest = @"
Created: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
Purpose: Shared large assets for Chroma_ONNX development and exported runtimes.
ModelDir: $modelTarget
OptimizedModelCacheDir: $cacheTarget
FixtureDir: $fixtureTarget

The runtime folder should carry only code, published binaries, launch scripts, and small ONNX deploy metadata.
"@

Set-Content -LiteralPath (Join-Path $AssetsRoot "ASSETS_MANIFEST.txt") -Value $manifest -Encoding ASCII

Write-Host "Large assets are ready at $AssetsRoot"
Write-Host "ModelDir: $modelTarget"
Write-Host "OptimizedModelCacheDir: $cacheTarget"
Write-Host "FixtureDir: $fixtureTarget"
