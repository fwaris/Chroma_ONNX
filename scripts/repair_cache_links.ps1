param(
    [string]$ModelDir = "",
    [string]$AssetsRoot = "",
    [string]$CacheDir = "",
    [string]$CacheSource = "",
    [string]$FixtureSource = "",
    [switch]$ForceRelink,
    [switch]$CopyIfHardlinkFails
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

function Get-FullPath([string]$Path) {
    [System.IO.Path]::GetFullPath($Path)
}

function Assert-PathExists([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Label was not found: $Path"
    }
}

function Find-RepoRoot {
    $current = Get-Item -LiteralPath $ScriptDir
    while ($null -ne $current) {
        if ((Test-Path -LiteralPath (Join-Path $current.FullName "Chroma_ONNX.slnx")) -or
            (Test-Path -LiteralPath (Join-Path $current.FullName "src\ChromaOnnx.Service"))) {
            return $current.FullName
        }
        $current = $current.Parent
    }
    return ""
}

function Infer-AssetsRootFromModelDir([string]$ResolvedModelDir) {
    $modelInfo = Get-Item -LiteralPath $ResolvedModelDir
    if ($modelInfo.Name -ieq "chroma-4b" -and
        $null -ne $modelInfo.Parent -and
        $modelInfo.Parent.Name -ieq "models" -and
        $null -ne $modelInfo.Parent.Parent) {
        return $modelInfo.Parent.Parent.FullName
    }
    return ""
}

function Resolve-ModelDir {
    $candidates = New-Object System.Collections.Generic.List[string]

    if ($ModelDir) {
        $candidate = Get-FullPath $ModelDir
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
        throw "Explicit ModelDir was not found: $candidate"
    }

    if ($AssetsRoot) {
        $candidate = Get-FullPath (Join-Path $AssetsRoot "models\chroma-4b")
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
        throw "AssetsRoot does not contain models\chroma-4b: $candidate"
    }

    $siblingAssetsRoot = Join-Path (Split-Path -Parent $ScriptDir) "Chroma_ONNX_assets"
    $candidates.Add((Join-Path $siblingAssetsRoot "models\chroma-4b"))
    $candidates.Add((Join-Path $ScriptDir "models\chroma-4b"))

    $repoRoot = Find-RepoRoot
    if ($repoRoot) {
        $candidates.Add((Join-Path $repoRoot "models\chroma-4b"))
    }

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {
            return Get-FullPath $candidate
        }
    }

    throw "Could not find models\chroma-4b. Pass -ModelDir E:\path\to\models\chroma-4b."
}

function Test-PathEquals([string]$Left, [string]$Right) {
    [string]::Equals((Get-FullPath $Left), (Get-FullPath $Right), [System.StringComparison]::OrdinalIgnoreCase)
}

function Link-OrCopyFile([string]$Source, [string]$Target) {
    $sourceInfo = Get-Item -LiteralPath $Source

    if (Test-Path -LiteralPath $Target) {
        $targetInfo = Get-Item -LiteralPath $Target
        if ($targetInfo.Length -ne $sourceInfo.Length) {
            Remove-Item -LiteralPath $Target -Force
        }
        elseif (-not $ForceRelink) {
            Write-Host "Exists: $Target"
            return
        }
        else {
            Remove-Item -LiteralPath $Target -Force
        }
    }

    $targetDir = Split-Path -Parent $Target
    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null

    try {
        New-Item -ItemType HardLink -Path $Target -Target $Source | Out-Null
        Write-Host "Hardlinked: $Target"
    }
    catch {
        if (-not $CopyIfHardlinkFails) {
            throw
        }

        Copy-Item -LiteralPath $Source -Destination $Target -Force
        Write-Host "Copied: $Target"
    }
}

function Copy-IfAvailable([string]$Name, [string]$SourceDir, [string]$TargetDir) {
    if (-not $SourceDir) {
        return
    }

    $source = Join-Path $SourceDir $Name
    if (Test-Path -LiteralPath $source) {
        New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
        Copy-Item -LiteralPath $source -Destination $TargetDir -Force
        Write-Host "Copied cache file: $(Join-Path $TargetDir $Name)"
    }
}

$resolvedModelDir = Resolve-ModelDir

if (-not $AssetsRoot) {
    $AssetsRoot = Infer-AssetsRootFromModelDir $resolvedModelDir
}

if (-not $AssetsRoot) {
    throw "Could not infer AssetsRoot from ModelDir '$resolvedModelDir'. Pass -AssetsRoot or -CacheDir."
}

$AssetsRoot = Get-FullPath $AssetsRoot

if (-not $CacheDir) {
    $CacheDir = Join-Path $AssetsRoot "onnx\chroma-s2s-full-v2\ort-cache-ort-local-external"
}

$CacheDir = Get-FullPath $CacheDir
New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null

Write-Host "ModelDir: $resolvedModelDir"
Write-Host "AssetsRoot: $AssetsRoot"
Write-Host "CacheDir: $CacheDir"

foreach ($name in @(
    "config.json",
    "tokenizer.json",
    "model.safetensors.index.json",
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors"
)) {
    Assert-PathExists (Join-Path $resolvedModelDir $name) "Required model file"
}

if (-not $CacheSource) {
    $repoRoot = Find-RepoRoot
    $sourceCandidates = @(
        (Join-Path $ScriptDir "onnx\chroma-s2s-full-v2\ort-cache-ort-local-external"),
        (Join-Path $AssetsRoot "onnx\chroma-s2s-full-v2\ort-cache-ort-local-external")
    )

    if ($repoRoot) {
        $sourceCandidates += (Join-Path $repoRoot "onnx\chroma-s2s-full-v2\ort-cache-ort-local-external")
    }

    foreach ($candidate in $sourceCandidates) {
        if ((Test-Path -LiteralPath $candidate) -and
            (Test-Path -LiteralPath (Join-Path $candidate "s2s_merged.cuda.quality-safe.optimized.onnx"))) {
            $CacheSource = $candidate
            break
        }
    }
}

foreach ($name in @(
    "chroma_s2s_merged.local_external.onnx",
    "s2s_merged.cuda.quality-safe.optimized.onnx",
    "local_external_cache_report.json"
)) {
    if (-not (Test-Path -LiteralPath (Join-Path $CacheDir $name))) {
        Copy-IfAvailable $name $CacheSource $CacheDir
    }
}

$optimizedCachePath = Join-Path $CacheDir "s2s_merged.cuda.quality-safe.optimized.onnx"
if (-not (Test-Path -LiteralPath $optimizedCachePath)) {
    $message =
        "Optimized CUDA cache is missing: $optimizedCachePath. " +
        "This repair script can recreate safetensor hardlinks from models\chroma-4b, " +
        "but it cannot rebuild the optimized ONNX cache without the export/build tooling. " +
        "Copy the optimized cache into CacheDir or rerun scripts\rebuild_chroma_local_external_cache.py."
    throw $message
}

foreach ($name in @("model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors")) {
    $source = Join-Path $resolvedModelDir $name
    $target = Join-Path $CacheDir $name
    Link-OrCopyFile $source $target
}

if (-not $FixtureSource) {
    $runtimeFixture = Join-Path $ScriptDir "served_runs\compare_inputs"
    $repoRoot = Find-RepoRoot
    if (Test-Path -LiteralPath $runtimeFixture) {
        $FixtureSource = $runtimeFixture
    }
    elseif ($repoRoot -and (Test-Path -LiteralPath (Join-Path $repoRoot "served_runs\compare_inputs"))) {
        $FixtureSource = Join-Path $repoRoot "served_runs\compare_inputs"
    }
}

$fixtureTarget = Join-Path $AssetsRoot "served_runs\compare_inputs"
if ($FixtureSource -and (Test-Path -LiteralPath $FixtureSource)) {
    New-Item -ItemType Directory -Force -Path $fixtureTarget | Out-Null
    Get-ChildItem -LiteralPath $FixtureSource -File | ForEach-Object {
        $target = Join-Path $fixtureTarget $_.Name
        if (-not (Test-PathEquals $_.FullName $target)) {
            Copy-Item -LiteralPath $_.FullName -Destination $target -Force
        }
    }
    Write-Host "Fixture files are ready: $fixtureTarget"
}
else {
    Write-Warning "Smoke-test fixture was not found. Runtime service can still start, but smoke-test-a100.ps1 needs served_runs\compare_inputs."
}

Write-Host "Cache links repaired."
