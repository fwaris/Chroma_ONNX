param(
    [string]$SourceModelDir = "models\gemma-4-e2b-it-onnx-mobius",
    [string]$AssetsRoot = "E:\s\temp\Chroma_ONNX_assets",
    [string]$Variant = "Q4_K_M/cuda",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$source = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $SourceModelDir))
$assets = [System.IO.Path]::GetFullPath($AssetsRoot)
$target = Join-Path $assets "models\gemma-4-e2b-it-onnx-mobius"
$variantRelative = ($Variant -replace "/", [System.IO.Path]::DirectorySeparatorChar) -replace "\\", [System.IO.Path]::DirectorySeparatorChar

function Assert-PathExists([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Label was not found: $Path"
    }
}

function With-Variant([string]$RelativePath) {
    Join-Path $variantRelative $RelativePath
}

$required = @(
    "audio_feature_extraction.json",
    "chat_template.jinja",
    "genai_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "audio_encoder\model.onnx",
    "audio_encoder\model.onnx.data",
    "decoder\model.onnx",
    "decoder\model.onnx.data",
    "embedding\model.onnx",
    "embedding\model.onnx.data"
) | ForEach-Object { With-Variant $_ }

Assert-PathExists $source "Gemma source model directory"
foreach ($relativePath in $required) {
    Assert-PathExists (Join-Path $source $relativePath) "Gemma source asset"
}

New-Item -ItemType Directory -Force -Path $target | Out-Null
foreach ($item in Get-ChildItem -LiteralPath $source -Force | Where-Object { $_.Name -ne ".cache" }) {
    $destination = Join-Path $target $item.Name
    if ((Test-Path -LiteralPath $destination) -and -not $Force) {
        Write-Host "exists  $($item.Name)"
    } else {
        Write-Host "copy    $($item.Name)"
        Copy-Item -LiteralPath $item.FullName -Destination $destination -Recurse -Force
    }
}

foreach ($relativePath in $required) {
    Assert-PathExists (Join-Path $target $relativePath) "Gemma staged asset"
}

$bytes =
    Get-ChildItem -LiteralPath $target -Recurse -File |
    Measure-Object -Property Length -Sum |
    Select-Object -ExpandProperty Sum
$gb = [Math]::Round([double]$bytes / 1GB, 3)

$manifest = @"
Gemma model assets
Source: $source
Target: $target
Variant: $Variant
Bytes: $bytes
GiB: $gb
"@

Set-Content -LiteralPath (Join-Path $target "ASSET_MANIFEST.txt") -Value $manifest -Encoding ASCII
Write-Host "Gemma Mobius assets staged at $target ($gb GiB)."
