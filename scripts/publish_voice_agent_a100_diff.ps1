param(
    [string]$RuntimeRoot = "E:\s\temp\VoiceAgent_A100_runtime",
    [string]$AssetsRoot = "E:\s\temp\VoiceAgent_assets",
    [string]$DiffParent = "E:\s\temp",
    [string]$ZipPath = "",
    [string]$Configuration = "Release",
    [string]$RuntimeIdentifier = "win-x64",
    [string]$TtsExecutablePath = "",
    [switch]$SkipPublish,
    [switch]$SkipAssetValidation
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RuntimeRoot = [System.IO.Path]::GetFullPath($RuntimeRoot)
$AssetsRoot = [System.IO.Path]::GetFullPath($AssetsRoot)
$DiffParent = [System.IO.Path]::GetFullPath($DiffParent)
New-Item -ItemType Directory -Force -Path $DiffParent | Out-Null

if (-not $SkipPublish) {
    $publishArgs = @(
        "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $scriptRoot "publish_voice_agent_a100_runtime.ps1"),
        "-RuntimeRoot", $RuntimeRoot,
        "-AssetsRoot", $AssetsRoot,
        "-Configuration", $Configuration,
        "-RuntimeIdentifier", $RuntimeIdentifier,
        "-FrameworkDependent"
    )
    if ($TtsExecutablePath) {
        $publishArgs += @("-TtsExecutablePath", $TtsExecutablePath)
    }
    if ($SkipAssetValidation) {
        $publishArgs += "-SkipAssetValidation"
    }
    & powershell @publishArgs
    if ($LASTEXITCODE -ne 0) {
        throw "publish_voice_agent_a100_runtime.ps1 failed with exit code $LASTEXITCODE"
    }
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$diffRoot = Join-Path $DiffParent "VoiceAgent_A100_diff_$timestamp"
$resolvedDiffRoot = [System.IO.Path]::GetFullPath($diffRoot)
$resolvedDiffParent = [System.IO.Path]::GetFullPath($DiffParent).TrimEnd('\')
if (-not ($resolvedDiffRoot.StartsWith($resolvedDiffParent + "\", [System.StringComparison]::OrdinalIgnoreCase))) {
    throw "DiffRoot is outside DiffParent. DiffRoot: $resolvedDiffRoot Parent: $resolvedDiffParent"
}
if (Test-Path -LiteralPath $resolvedDiffRoot) {
    Remove-Item -LiteralPath $resolvedDiffRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $resolvedDiffRoot | Out-Null

foreach ($relative in @("app", "tools")) {
    $source = Join-Path $RuntimeRoot $relative
    if (Test-Path -LiteralPath $source) {
        Copy-Item -LiteralPath $source -Destination $resolvedDiffRoot -Recurse -Force
    }
}

foreach ($fileName in @(
    "run-voice-agent-a100.ps1",
    "smoke-voice-agent-a100.ps1",
    "README-voice-agent-a100.md",
    "VOICE_AGENT_MANIFEST.txt"
)) {
    $source = Join-Path $RuntimeRoot $fileName
    if (Test-Path -LiteralPath $source) {
        Copy-Item -LiteralPath $source -Destination (Join-Path $resolvedDiffRoot $fileName) -Force
    }
}

$manifestPath = Join-Path $resolvedDiffRoot "DIFF_MANIFEST.txt"
$expandedBytes = (
    Get-ChildItem -LiteralPath $resolvedDiffRoot -Recurse -File |
    Measure-Object -Property Length -Sum
).Sum

@"
Gemma Voice Agent A100 Diff
Generated: $(Get-Date -Format o)
RuntimeRoot: $RuntimeRoot
AssetsRoot: $AssetsRoot
ModelsIncluded: no
ExpandedBytes: $expandedBytes
"@ | Set-Content -LiteralPath $manifestPath -Encoding UTF8

if (-not $ZipPath) {
    $ZipPath = Join-Path $DiffParent "VoiceAgent_A100_diff_$timestamp.zip"
}
$ZipPath = [System.IO.Path]::GetFullPath($ZipPath)
if (Test-Path -LiteralPath $ZipPath) {
    Remove-Item -LiteralPath $ZipPath -Force
}
Compress-Archive -Path (Join-Path $resolvedDiffRoot "*") -DestinationPath $ZipPath -Force

$expandedBytes = (
    Get-ChildItem -LiteralPath $resolvedDiffRoot -Recurse -File |
    Measure-Object -Property Length -Sum
).Sum
$zipBytes = (Get-Item -LiteralPath $ZipPath).Length

Write-Host "DiffRoot: $resolvedDiffRoot"
Write-Host "ZipPath: $ZipPath"
Write-Host ("ExpandedSizeMB: {0:N2}" -f ($expandedBytes / 1MB))
Write-Host ("ZipSizeMB: {0:N2}" -f ($zipBytes / 1MB))
