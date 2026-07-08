param(
    [string]$AssetsRoot = "E:\s\temp\VoiceAgent_assets",
    [string]$RepoId = "onnx-community/chatterbox-ONNX",
    [string]$Revision = "main",
    [ValidateSet("q4f16", "q4", "fp16", "fp32")]
    [string]$LanguageModelVariant = "q4f16",
    [switch]$SkipExisting
)

$ErrorActionPreference = "Stop"

$AssetsRoot = [System.IO.Path]::GetFullPath($AssetsRoot)
$modelDir = Join-Path $AssetsRoot "models\chatterbox-onnx"
New-Item -ItemType Directory -Force -Path $modelDir | Out-Null

$languageModelFiles = switch ($LanguageModelVariant) {
    "q4f16" { @("onnx/language_model_q4f16.onnx", "onnx/language_model_q4f16.onnx_data") }
    "q4" { @("onnx/language_model_q4.onnx", "onnx/language_model_q4.onnx_data") }
    "fp16" { @("onnx/language_model_fp16.onnx", "onnx/language_model_fp16.onnx_data") }
    "fp32" { @("onnx/language_model.onnx", "onnx/language_model.onnx_data") }
}

$files = @(
    "README.md",
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "default_voice.wav",
    "onnx/conditional_decoder.onnx",
    "onnx/conditional_decoder.onnx_data",
    "onnx/embed_tokens.onnx",
    "onnx/embed_tokens.onnx_data",
    "onnx/speech_encoder.onnx",
    "onnx/speech_encoder.onnx_data"
) + $languageModelFiles

function Invoke-HuggingFaceDownload([string]$FileName, [string]$Destination) {
    $url = "https://huggingface.co/$RepoId/resolve/$Revision/$FileName"
    $tmp = "$Destination.partial"
    if ($SkipExisting -and (Test-Path -LiteralPath $Destination)) {
        Write-Host "Already present: $Destination"
        return
    }

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($curl) {
        & $curl.Source -L --fail --retry 5 --retry-delay 2 -C - -o $tmp $url
        if ($LASTEXITCODE -ne 0) {
            throw "curl failed with exit code $LASTEXITCODE while downloading $url"
        }
    } else {
        Invoke-WebRequest -Uri $url -OutFile $tmp
    }
    Move-Item -LiteralPath $tmp -Destination $Destination -Force
}

foreach ($fileName in $files) {
    $destination = Join-Path $modelDir ($fileName -replace '/', [System.IO.Path]::DirectorySeparatorChar)
    Write-Host "Downloading $RepoId/$fileName"
    Invoke-HuggingFaceDownload $fileName $destination
}

$manifestFiles =
    foreach ($fileName in $files) {
        $path = Join-Path $modelDir ($fileName -replace '/', [System.IO.Path]::DirectorySeparatorChar)
        $item = Get-Item -LiteralPath $path
        [pscustomobject]@{
            path = $fileName
            sizeBytes = $item.Length
            lastWriteTimeUtc = $item.LastWriteTimeUtc.ToString("o")
        }
    }

$manifest = [pscustomobject]@{
    repoId = $RepoId
    revision = $Revision
    modelDir = $modelDir
    generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
    languageModelVariant = $LanguageModelVariant
    files = $manifestFiles
}

$manifestPath = Join-Path $modelDir "voice_agent_chatterbox_onnx_manifest.json"
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

Write-Host "Chatterbox ONNX assets are ready."
Write-Host "ModelDir: $modelDir"
Write-Host "Variant: $LanguageModelVariant"
Write-Host "Manifest: $manifestPath"
