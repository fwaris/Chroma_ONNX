param(
    [string]$AssetsRoot = "E:\s\temp\VoiceAgent_assets",
    [string]$RepoId = "soniqo/VoxCPM2-ONNX",
    [string]$Revision = "main",
    [switch]$IncludeSplitGraphs,
    [switch]$SkipExisting
)

$ErrorActionPreference = "Stop"

$AssetsRoot = [System.IO.Path]::GetFullPath($AssetsRoot)
$modelDir = Join-Path $AssetsRoot "models\voxcpm2-onnx"
New-Item -ItemType Directory -Force -Path $modelDir | Out-Null

$files = @(
    "config.json",
    "special_tokens_map.json",
    "tokenization_voxcpm2.py",
    "tokenizer.json",
    "tokenizer_config.json",
    "voxcpm2-audio-decoder.onnx",
    "voxcpm2-audio-encoder.onnx",
    "voxcpm2-decoder.onnx",
    "voxcpm2-decoder.onnx.data"
)

if ($IncludeSplitGraphs) {
    $files += @(
        "voxcpm2-text-prefill.onnx",
        "voxcpm2-text-prefill.onnx.data",
        "voxcpm2-token-step.onnx",
        "voxcpm2-token-step.onnx.data"
    )
}

function Invoke-HuggingFaceDownload([string]$FileName, [string]$Destination) {
    $encodedRepo = $RepoId
    $url = "https://huggingface.co/$encodedRepo/resolve/$Revision/$FileName"
    $tmp = "$Destination.partial"
    if ($SkipExisting -and (Test-Path -LiteralPath $Destination)) {
        Write-Host "Already present: $Destination"
        return
    }

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
    $destination = Join-Path $modelDir $fileName
    Write-Host "Downloading $RepoId/$fileName"
    Invoke-HuggingFaceDownload $fileName $destination
}

$manifestFiles =
    foreach ($fileName in $files) {
        $path = Join-Path $modelDir $fileName
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
    includesSplitGraphs = [bool]$IncludeSplitGraphs
    files = $manifestFiles
}

$manifestPath = Join-Path $modelDir "voice_agent_voxcpm2_onnx_manifest.json"
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

Write-Host "VoxCPM2 ONNX assets are ready."
Write-Host "ModelDir: $modelDir"
Write-Host "Manifest: $manifestPath"
