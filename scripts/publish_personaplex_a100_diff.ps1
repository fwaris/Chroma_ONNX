param(
    [string]$RepoRoot = "E:\s\repos\Chroma_ONNX",
    [string]$RuntimeRoot = "E:\s\temp\VoiceAgent_A100_runtime",
    [string]$AssetsRoot = "E:\s\temp\VoiceAgent_assets",
    [string]$DiffParent = "E:\s\temp",
    [string]$Configuration = "Release",
    [string]$RuntimeIdentifier = "win-x64",
    [switch]$SkipPublish,
    [switch]$SkipAssetValidation,
    [switch]$NoZip
)

$ErrorActionPreference = "Stop"

$RepoRoot = [System.IO.Path]::GetFullPath($RepoRoot)
$RuntimeRoot = [System.IO.Path]::GetFullPath($RuntimeRoot)
$AssetsRoot = [System.IO.Path]::GetFullPath($AssetsRoot)
$DiffParent = [System.IO.Path]::GetFullPath($DiffParent)

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

Assert-PathExists (Join-Path $RepoRoot "Chroma_ONNX.slnx") "Chroma_ONNX repo"
Assert-PathExists (Join-Path $RepoRoot "scripts\publish_personaplex_a100_runtime.ps1") "PersonaPlex A100 publish script"

if (-not $SkipPublish) {
    $serviceOut = Join-Path $RuntimeRoot "app\service"
    Assert-ChildPath $serviceOut $RuntimeRoot "Publish output"
    if (Test-Path -LiteralPath $serviceOut) {
        Remove-Item -LiteralPath $serviceOut -Recurse -Force
    }
}

$publishArgs = @(
    "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $RepoRoot "scripts\publish_personaplex_a100_runtime.ps1"),
    "-RuntimeRoot", $RuntimeRoot,
    "-AssetsRoot", $AssetsRoot,
    "-Configuration", $Configuration,
    "-RuntimeIdentifier", $RuntimeIdentifier,
    "-FrameworkDependent"
)
if ($SkipPublish) { $publishArgs += "-SkipPublish" }
if ($SkipAssetValidation) { $publishArgs += "-SkipAssetValidation" }

if ($SkipPublish) {
    Write-Host "Refreshing PersonaPlex A100 runtime package metadata..."
} else {
    Write-Host "Publishing framework-dependent PersonaPlex A100 runtime..."
}
& powershell @publishArgs
if ($LASTEXITCODE -ne 0) {
    throw "PersonaPlex A100 publish failed with exit code $LASTEXITCODE."
}

Assert-PathExists (Join-Path $RuntimeRoot "app\service\ChromaOnnx.Service.dll") "Published service"
Assert-PathExists (Join-Path $RuntimeRoot "run-service-a100.ps1") "A100 launcher"
Assert-PathExists (Join-Path $RuntimeRoot "smoke-test-a100.ps1") "A100 smoke script"

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$diffRoot = Join-Path $DiffParent "VoiceAgent_A100_diff_$stamp"
New-Item -ItemType Directory -Force -Path $diffRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $diffRoot "app") | Out-Null

Copy-Item -LiteralPath (Join-Path $RuntimeRoot "app\service") -Destination (Join-Path $diffRoot "app") -Recurse -Force
foreach ($file in @(
    "run-service-a100.ps1",
    "smoke-test-a100.ps1",
    "README_RUNTIME_A100.md",
    "RUNTIME_MANIFEST.txt"
)) {
    Copy-Item -LiteralPath (Join-Path $RuntimeRoot $file) -Destination $diffRoot -Force
}

$manifest = @"
Created: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
Source repo: $RepoRoot
Source runtime: $RuntimeRoot
Apply target: existing VoiceAgent_A100_runtime root
Framework dependent: true
Includes CLI: false
PersonaPlex runtime: first-party full-onnx
PersonaPlex fallback: ElBruno.PersonaPlex 0.6.1 codec diagnostics

Included paths:
  app\service\
  run-service-a100.ps1
  smoke-test-a100.ps1
  README_RUNTIME_A100.md
  RUNTIME_MANIFEST.txt

Excluded paths:
  served_runs\
  VoiceAgent_assets\
  models\
  repo bin/obj/.git internals

Apply:
  Expand-Archive <zip> -DestinationPath .\diff -Force
  Copy-Item .\diff\* <VoiceAgent_A100_runtime> -Recurse -Force
"@
Set-Content -LiteralPath (Join-Path $diffRoot "DIFF_MANIFEST.txt") -Value $manifest -Encoding ASCII

$folderBytes = (Get-ChildItem -LiteralPath $diffRoot -Recurse -File | Measure-Object Length -Sum).Sum
$zipPath = $null
$zipBytes = $null
if (-not $NoZip) {
    $zipPath = "$diffRoot.zip"
    if (Test-Path -LiteralPath $zipPath) {
        Remove-Item -LiteralPath $zipPath -Force
    }
    Compress-Archive -Path (Join-Path $diffRoot "*") -DestinationPath $zipPath -CompressionLevel Optimal -Force
    $zipBytes = (Get-Item -LiteralPath $zipPath).Length
}

[pscustomobject]@{
    DiffRoot = $diffRoot
    ZipPath = $zipPath
    FolderBytes = $folderBytes
    ZipBytes = $zipBytes
}
