param(
    [string]$AssetsRoot = "E:\s\temp\VoiceAgent_assets",
    [string]$RepoId = "elbruno/personaplex-7b-v1-onnx",
    [string]$ModelSubdir = "models\personaplex-7b-v1-onnx",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$AssetsRoot = [System.IO.Path]::GetFullPath($AssetsRoot)
$modelDir = Join-Path $AssetsRoot $ModelSubdir
New-Item -ItemType Directory -Force -Path $modelDir | Out-Null

$files = @(
    "mimi_encoder.onnx",
    "mimi_decoder.onnx",
    "lm_backbone.onnx",
    "lm_backbone.onnx.data",
    "README.md"
)

foreach ($file in $files) {
    $target = Join-Path $modelDir $file
    if ((Test-Path -LiteralPath $target) -and -not $Force) {
        Write-Host "Already present: $target"
        continue
    }

    $encodedFile = [Uri]::EscapeDataString($file).Replace("%2F", "/")
    $url = "https://huggingface.co/$RepoId/resolve/main/$encodedFile"
    Write-Host "Downloading $file from $RepoId..."
    Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $target
}

$manifestPath = Join-Path $modelDir "PERSONAPLEX_ASSETS_MANIFEST.json"
$manifestFiles = @(
    $files | ForEach-Object {
        $path = Join-Path $modelDir $_
        [pscustomobject]@{
            file = $_
            bytes = if (Test-Path -LiteralPath $path) { (Get-Item -LiteralPath $path).Length } else { 0 }
        }
    }
)

$manifest = [pscustomobject]@{
    createdUtc = (Get-Date).ToUniversalTime().ToString("O")
    repoId = $RepoId
    modelDir = $modelDir
    assetSource = "Hugging Face"
    runtime = "first-party full-onnx"
    fallbackPackage = "ElBruno.PersonaPlex 0.6.1 codec diagnostics"
    capability = "Full ONNX asset staging for first-party encoder/backbone/decoder loading and audio-token generation diagnostics."
    files = $manifestFiles
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding ASCII

Write-Host "PersonaPlex assets staged at $modelDir"
Write-Host "Manifest: $manifestPath"
