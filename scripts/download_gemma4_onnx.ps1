param(
    [string]$ModelDir = "models/gemma-4-e2b-it-onnx-mobius",
    [string]$Variant = "Q4_K_M/cuda",
    [string]$Repository = "justinchuby/gemma-4-e2b-it-onnx",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$token = if ($env:HF_TOKEN) { $env:HF_TOKEN } elseif ($env:HUGGINGFACE_HUB_TOKEN) { $env:HUGGINGFACE_HUB_TOKEN } else { "" }
$headers = @{}
if ($token) {
    $headers["Authorization"] = "Bearer $token"
}

function Join-VariantPath {
    param([string]$RelativePath)
    $variantPrefix = ($Variant -replace "\\", "/").Trim("/")
    if ($variantPrefix) {
        "$variantPrefix/$RelativePath"
    } else {
        $RelativePath
    }
}

function Download-File {
    param([string]$RelativePath)

    $repoPath = Join-VariantPath $RelativePath
    $target = Join-Path $ModelDir $repoPath
    $parent = Split-Path -Parent $target
    if ($parent) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }

    if ((Test-Path -LiteralPath $target) -and -not $Force) {
        Write-Host "exists  $repoPath"
        return
    }

    $encoded = ($repoPath -replace "\\", "/").Split("/") | ForEach-Object {
        [System.Uri]::EscapeDataString($_)
    }
    $path = [string]::Join("/", $encoded)
    $url = "https://huggingface.co/$Repository/resolve/main/$path"
    Write-Host "fetch   $repoPath"
    if ($headers.Count -gt 0) {
        Invoke-WebRequest -Uri $url -OutFile $target -UseBasicParsing -Headers $headers
    } else {
        Invoke-WebRequest -Uri $url -OutFile $target -UseBasicParsing
    }
}

$files = @(
    "audio_feature_extraction.json",
    "chat_template.jinja",
    "genai_config.json",
    "image_processor.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "audio_encoder/model.onnx",
    "audio_encoder/model.onnx.data",
    "decoder/model.onnx",
    "decoder/model.onnx.data",
    "embedding/model.onnx",
    "embedding/model.onnx.data",
    "vision_encoder/model.onnx",
    "vision_encoder/model.onnx.data"
)

New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null

foreach ($file in $files) {
    Download-File $file
}

$variantDir = Join-Path $ModelDir (($Variant -replace "/", [System.IO.Path]::DirectorySeparatorChar) -replace "\\", [System.IO.Path]::DirectorySeparatorChar)
Write-Host "Gemma Mobius ONNX bundle is ready at $variantDir ($Variant)."
