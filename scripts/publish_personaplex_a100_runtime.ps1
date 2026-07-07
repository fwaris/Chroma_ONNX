param(
    [string]$RuntimeRoot = "E:\s\temp\VoiceAgent_A100_runtime",
    [string]$AssetsRoot = "E:\s\temp\VoiceAgent_assets",
    [string]$Configuration = "Release",
    [string]$RuntimeIdentifier = "win-x64",
    [switch]$SkipPublish,
    [switch]$FrameworkDependent,
    [switch]$SkipAssetValidation
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptRoot
$RuntimeRoot = [System.IO.Path]::GetFullPath($RuntimeRoot)
$AssetsRoot = [System.IO.Path]::GetFullPath($AssetsRoot)
$serviceOut = Join-Path $RuntimeRoot "app\service"

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

if (-not $SkipAssetValidation) {
    Assert-PathExists (Join-Path $AssetsRoot "models\gemma-4-e2b-it-onnx-mobius\Q4_K_M\cuda\genai_config.json") "Gemma ORT GenAI config"
    Assert-PathExists (Join-Path $AssetsRoot "models\gemma-4-e2b-it-onnx-mobius\Q4_K_M\cuda\embedding\model.onnx") "Gemma embedding graph"
    Assert-PathExists (Join-Path $AssetsRoot "models\gemma-4-e2b-it-onnx-mobius\Q4_K_M\cuda\audio_encoder\model.onnx") "Gemma audio encoder graph"
    Assert-PathExists (Join-Path $AssetsRoot "models\gemma-4-e2b-it-onnx-mobius\Q4_K_M\cuda\decoder\model.onnx") "Gemma decoder graph"
    Assert-PathExists (Join-Path $AssetsRoot "models\personaplex-7b-v1-onnx\mimi_encoder.onnx") "PersonaPlex Mimi encoder"
    Assert-PathExists (Join-Path $AssetsRoot "models\personaplex-7b-v1-onnx\mimi_decoder.onnx") "PersonaPlex Mimi decoder"
    Assert-PathExists (Join-Path $AssetsRoot "models\personaplex-7b-v1-onnx\lm_backbone.onnx") "PersonaPlex LM backbone graph"
    Assert-PathExists (Join-Path $AssetsRoot "models\personaplex-7b-v1-onnx\lm_backbone.onnx.data") "PersonaPlex LM backbone external data"
}

if (-not $SkipPublish) {
    Assert-ChildPath $serviceOut $RuntimeRoot "Publish output"
    if (Test-Path -LiteralPath $serviceOut) {
        Remove-Item -LiteralPath $serviceOut -Recurse -Force
    }
}
New-Item -ItemType Directory -Force -Path $serviceOut | Out-Null

if (-not $SkipPublish) {
    $selfContained = if ($FrameworkDependent) { "false" } else { "true" }
    dotnet publish (Join-Path $repoRoot "src\ChromaOnnx.Service\ChromaOnnx.Service.fsproj") `
        -c $Configuration `
        -r $RuntimeIdentifier `
        --self-contained $selfContained `
        -o $serviceOut
}

foreach ($file in @(
    "ChromaOnnx.Service.exe",
    "ChromaOnnx.Service.dll",
    "ChromaOnnx.Service.deps.json",
    "ChromaOnnx.Service.runtimeconfig.json",
    "ChromaOnnx.SpeechToSpeech.dll",
    "ChromaOnnx.OnnxRuntime.dll",
    "ChromaOnnx.Core.dll",
    "ElBruno.PersonaPlex.dll",
    "Microsoft.ML.OnnxRuntimeGenAI.dll",
    "onnxruntime.dll",
    "onnxruntime-genai.dll",
    "onnxruntime-genai-cuda.dll",
    "onnxruntime_providers_cuda.dll",
    "onnxruntime_providers_shared.dll"
)) {
    Assert-PathExists (Join-Path $serviceOut $file) "Published dependency $file"
}

$runService = @'
param(
    [string]$AssetsRoot = "",
    [int]$Port = 5055,
    [int]$MaxHistoryTurns = 8,
    [double]$MaxTurnAudioSeconds = 30,
    [string]$GemmaModelDir = "",
    [string]$GemmaVariant = "Q4_K_M/cuda",
    [ValidateSet("ort-genai", "raw-ort")]
    [string]$GemmaRuntime = "ort-genai",
    [ValidateSet("cpu", "cuda")]
    [string]$GemmaExecutionProvider = "cuda",
    [string]$PersonaPlexModelDir = "",
    [ValidateSet("full-onnx", "elbruno-codec")]
    [string]$PersonaPlexRuntime = "full-onnx",
    [ValidateSet("cpu", "cuda", "directml")]
    [string]$PersonaPlexExecutionProvider = "cuda",
    [string]$PersonaPlexVoicePreset = "NATF2",
    [string]$CudaBin = "",
    [string]$CudnnBin = "",
    [switch]$LocalhostOnly
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $AssetsRoot) {
    $AssetsRoot = Join-Path (Split-Path -Parent $Root) "VoiceAgent_assets"
}
$AssetsRoot = [System.IO.Path]::GetFullPath($AssetsRoot)
if ($CudaBin) { $env:PATH = "$CudaBin;$env:PATH" }
if ($CudnnBin) { $env:PATH = "$CudnnBin;$env:PATH" }

if ($GemmaModelDir) {
    $selectedGemmaModelDir = [System.IO.Path]::GetFullPath($GemmaModelDir)
} else {
    $selectedGemmaModelDir = Join-Path $AssetsRoot "models\gemma-4-e2b-it-onnx-mobius\Q4_K_M\cuda"
}
if ($PersonaPlexModelDir) {
    $selectedPersonaPlexModelDir = [System.IO.Path]::GetFullPath($PersonaPlexModelDir)
} else {
    $selectedPersonaPlexModelDir = Join-Path $AssetsRoot "models\personaplex-7b-v1-onnx"
}

$serviceDir = Join-Path $Root "app\service"
$requiredPaths = @(
    (Join-Path $serviceDir "ChromaOnnx.Service.exe"),
    (Join-Path $serviceDir "ElBruno.PersonaPlex.dll"),
    (Join-Path $serviceDir "onnxruntime.dll"),
    (Join-Path $serviceDir "onnxruntime_providers_cuda.dll"),
    (Join-Path $serviceDir "onnxruntime_providers_shared.dll"),
    (Join-Path $serviceDir "Microsoft.ML.OnnxRuntimeGenAI.dll"),
    (Join-Path $serviceDir "onnxruntime-genai.dll"),
    (Join-Path $serviceDir "onnxruntime-genai-cuda.dll"),
    (Join-Path $selectedGemmaModelDir "genai_config.json"),
    (Join-Path $selectedGemmaModelDir "tokenizer.json"),
    (Join-Path $selectedGemmaModelDir "embedding\model.onnx"),
    (Join-Path $selectedGemmaModelDir "audio_encoder\model.onnx"),
    (Join-Path $selectedGemmaModelDir "decoder\model.onnx"),
    (Join-Path $selectedPersonaPlexModelDir "mimi_encoder.onnx"),
    (Join-Path $selectedPersonaPlexModelDir "mimi_decoder.onnx"),
    (Join-Path $selectedPersonaPlexModelDir "lm_backbone.onnx"),
    (Join-Path $selectedPersonaPlexModelDir "lm_backbone.onnx.data")
)
foreach ($path in $requiredPaths) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required path is missing: $path"
    }
}

$env:VoiceAgent__WorkDir = Join-Path $Root "served_runs"
$env:VoiceAgent__MaxHistoryTurns = [string]$MaxHistoryTurns
$env:VoiceAgent__MaxTurnAudioSeconds = [string]$MaxTurnAudioSeconds
$env:VoiceAgent__Gemma__ModelDir = $selectedGemmaModelDir
$env:VoiceAgent__Gemma__Variant = $GemmaVariant
$env:VoiceAgent__Gemma__Runtime = $GemmaRuntime
$env:VoiceAgent__Gemma__ExecutionProvider = $GemmaExecutionProvider
$env:VoiceAgent__PersonaPlex__ModelDir = $selectedPersonaPlexModelDir
$env:VoiceAgent__PersonaPlex__Runtime = $PersonaPlexRuntime
$env:VoiceAgent__PersonaPlex__ExecutionProvider = $PersonaPlexExecutionProvider
$env:VoiceAgent__PersonaPlex__VoicePreset = $PersonaPlexVoicePreset

$hostName = if ($LocalhostOnly) { "localhost" } else { "0.0.0.0" }
$url = "http://${hostName}:$Port"
Write-Host "Starting GemmaPersonaPlexAgent on $url"
Write-Host "AssetsRoot: $AssetsRoot"
Write-Host "GemmaModelDir: $selectedGemmaModelDir"
Write-Host "GemmaRuntime: $GemmaRuntime"
Write-Host "GemmaExecutionProvider: $GemmaExecutionProvider"
Write-Host "PersonaPlexModelDir: $selectedPersonaPlexModelDir"
Write-Host "PersonaPlexRuntime: $PersonaPlexRuntime"
Write-Host "PersonaPlexExecutionProvider: $PersonaPlexExecutionProvider"
Write-Host "PersonaPlexVoicePreset: $PersonaPlexVoicePreset"
& (Join-Path $serviceDir "ChromaOnnx.Service.exe") --urls $url
exit $LASTEXITCODE
'@

$smokeTest = @'
param(
    [string]$AssetsRoot = "",
    [int]$Turns = 2,
    [string]$RequireTool = "get_current_time",
    [int]$Port = 5065,
    [int]$TimeoutSeconds = 600,
    [string]$CudaBin = "",
    [string]$CudnnBin = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $AssetsRoot) {
    $AssetsRoot = Join-Path (Split-Path -Parent $Root) "VoiceAgent_assets"
}
$AssetsRoot = [System.IO.Path]::GetFullPath($AssetsRoot)
if ($CudaBin) { $env:PATH = "$CudaBin;$env:PATH" }
if ($CudnnBin) { $env:PATH = "$CudnnBin;$env:PATH" }

$env:VoiceAgent__WorkDir = Join-Path $Root "served_runs"
$env:VoiceAgent__MaxHistoryTurns = "8"
$env:VoiceAgent__MaxTurnAudioSeconds = "30"
$env:VoiceAgent__Gemma__ModelDir = Join-Path $AssetsRoot "models\gemma-4-e2b-it-onnx-mobius\Q4_K_M\cuda"
$env:VoiceAgent__Gemma__Variant = "Q4_K_M/cuda"
$env:VoiceAgent__Gemma__Runtime = "ort-genai"
$env:VoiceAgent__Gemma__ExecutionProvider = "cuda"
$env:VoiceAgent__PersonaPlex__ModelDir = Join-Path $AssetsRoot "models\personaplex-7b-v1-onnx"
$env:VoiceAgent__PersonaPlex__ExecutionProvider = "cuda"
$env:VoiceAgent__PersonaPlex__VoicePreset = "NATF2"

$logDir = Join-Path $Root "served_runs\smoke_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$outLog = Join-Path $logDir "voice_agent_a100.out.log"
$errLog = Join-Path $logDir "voice_agent_a100.err.log"
$url = "http://localhost:$Port"
$exe = Join-Path $Root "app\service\ChromaOnnx.Service.exe"

Write-Host "Starting service for smoke test on $url..."
$process = Start-Process -FilePath $exe -ArgumentList @("--urls", $url) -PassThru -WindowStyle Hidden -RedirectStandardOutput $outLog -RedirectStandardError $errLog
try {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $status = $null
    while ((Get-Date) -lt $deadline -and -not $status) {
        try {
            $status = Invoke-RestMethod -Uri "$url/api/status" -TimeoutSec 5
        } catch {
            Start-Sleep -Milliseconds 500
        }
        if ($process.HasExited) {
            throw "Service exited early with code $($process.ExitCode). See $outLog and $errLog"
        }
    }
    if (-not $status) { throw "Service did not become ready before timeout." }
    if (-not $status.gemma.ready) { throw "Gemma is not ready: $($status.gemma.message)" }
    if (-not $status.personaPlex.codecReady) { throw "PersonaPlex codec is not ready: $($status.personaPlex.message)" }
    Write-Host "Status ready. PersonaPlex STS ready: $($status.personaPlex.speechToSpeechReady)"

    Add-Type -AssemblyName System.Net.Http
    $http = [System.Net.Http.HttpClient]::new()
    $form = [System.Net.Http.MultipartFormDataContent]::new()
    $prompt = "You are in an A100 smoke test. Always call get_current_time once before final text, regardless of the transcript."
    $form.Add([System.Net.Http.StringContent]::new($prompt), "systemPrompt")
    $response = $http.PostAsync("$url/api/agent/sessions", $form).GetAwaiter().GetResult()
    if (-not $response.IsSuccessStatusCode) { throw "Session create failed: $($response.StatusCode) $($response.Content.ReadAsStringAsync().Result)" }
    $session = $response.Content.ReadAsStringAsync().Result | ConvertFrom-Json

    function Convert-FloatsToBytes([single[]]$Samples) {
        $bytes = New-Object byte[] ($Samples.Length * 4)
        [Buffer]::BlockCopy($Samples, 0, $bytes, 0, $bytes.Length)
        $bytes
    }

    function New-TestAudio {
        $samples = New-Object single[] 24000
        for ($i = 0; $i -lt $samples.Length; $i++) {
            $samples[$i] = [single](0.02 * [Math]::Sin(2.0 * [Math]::PI * 220.0 * $i / 24000.0))
        }
        Convert-FloatsToBytes $samples
    }

    function Send-Text($Socket, [string]$Text) {
        $bytes = [Text.Encoding]::UTF8.GetBytes($Text)
        $Socket.SendAsync([ArraySegment[byte]]::new($bytes), [Net.WebSockets.WebSocketMessageType]::Text, $true, [Threading.CancellationToken]::None).GetAwaiter().GetResult()
    }

    function Send-Binary($Socket, [byte[]]$Bytes) {
        $Socket.SendAsync([ArraySegment[byte]]::new($Bytes), [Net.WebSockets.WebSocketMessageType]::Binary, $true, [Threading.CancellationToken]::None).GetAwaiter().GetResult()
    }

    function Receive-Json($Socket) {
        $buffer = New-Object byte[] 65536
        $stream = [IO.MemoryStream]::new()
        do {
            $result = $Socket.ReceiveAsync([ArraySegment[byte]]::new($buffer), [Threading.CancellationToken]::None).GetAwaiter().GetResult()
            if ($result.Count -gt 0) { $stream.Write($buffer, 0, $result.Count) }
        } while (-not $result.EndOfMessage)
        if ($result.MessageType -ne [Net.WebSockets.WebSocketMessageType]::Text) { return $null }
        [Text.Encoding]::UTF8.GetString($stream.ToArray()) | ConvertFrom-Json
    }

    $ws = [Net.WebSockets.ClientWebSocket]::new()
    $ws.ConnectAsync([Uri]"ws://localhost:$Port$($session.websocketUrl)", [Threading.CancellationToken]::None).GetAwaiter().GetResult()
    $ready = Receive-Json $ws
    if ($ready.type -ne "session.ready") { throw "Unexpected first websocket event: $($ready.type)" }

    for ($turn = 1; $turn -le $Turns; $turn++) {
        Write-Host "Running smoke turn $turn..."
        Send-Text $ws '{"type":"turn.start"}'
        [void](Receive-Json $ws)
        Send-Binary $ws (New-TestAudio)
        [void](Receive-Json $ws)
        Send-Text $ws '{"type":"turn.end"}'
        $events = @()
        $done = $null
        while (-not $done -and (Get-Date) -lt $deadline) {
            $event = Receive-Json $ws
            if ($event) {
                $events += $event.type
                if ($event.type -eq "agent.done") { $done = $event }
            }
        }
        if (-not $done) { throw "Turn $turn did not complete. Events: $($events -join ',')" }
        if ($done.turnIndex -ne $turn) { throw "Expected turnIndex $turn, got $($done.turnIndex)" }
        if ($RequireTool -and -not ($events -contains "agent.tool_call")) {
            throw "Required tool call event was not observed. Events: $($events -join ',')"
        }
        Write-Host "Turn $turn done: $($done.finalText)"
    }
    $ws.Dispose()
    Write-Host "A100 voice-agent smoke test passed."
} finally {
    if ($process -and -not $process.HasExited) {
        Stop-Process -Id $process.Id -Force
    }
}
'@

$readme = @"
# Gemma + PersonaPlex A100 Runtime

This runtime contains the .NET service and native ONNX Runtime dependencies only. Large model files live in the sibling assets folder.

Expected sibling layout:

```text
VoiceAgent_A100_runtime\
VoiceAgent_assets\
  models\gemma-4-e2b-it-onnx-mobius\Q4_K_M\cuda\
  models\personaplex-7b-v1-onnx\
```

Run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run-service-a100.ps1
```

Smoke test:

```powershell
.\smoke-test-a100.ps1 -AssetsRoot G:\Chroma\VoiceAgent_assets -Turns 2 -RequireTool get_current_time
```

Full PersonaPlex ONNX diagnostic smoke:

```powershell
.\smoke-test-a100.ps1 -AssetsRoot G:\Chroma\VoiceAgent_assets -Mode PersonaPlexFull -Turns 2
```

The default smoke starts the service, creates one Gemma agent session, sends two turns over the same WebSocket, requires `get_current_time` on each turn, and verifies tool results and per-turn details artifacts. `-Mode PersonaPlexFull` creates a PersonaPlex-only session and verifies that the full ONNX asset set loads, the LM backbone runs, and an audio artifact is written. If `lm_backbone.onnx` exposes only `transformer_out`/`text_logits`, the runtime reports that the audio-token generation head is missing and writes codec-passthrough audio for diagnostics.

PersonaPlex full ONNX mode uses first-party ONNX Runtime sessions. `ElBruno.PersonaPlex` 0.6.1 remains packaged for the `elbruno-codec` fallback/debug mode only.

The runtime is framework-dependent; .NET must already be installed. The host also needs the NVIDIA driver plus the CUDA/cuDNN DLLs expected by the packaged ONNX Runtime builds on PATH. Use `-CudaBin` and `-CudnnBin` when needed.
"@

$repoSmokeTest = Join-Path $scriptRoot "smoke_personaplex_two_turn_tool.ps1"
Assert-PathExists $repoSmokeTest "Two-turn PersonaPlex tool smoke script"
$smokeTest = Get-Content -LiteralPath $repoSmokeTest -Raw

Set-Content -LiteralPath (Join-Path $RuntimeRoot "run-service-a100.ps1") -Value $runService -Encoding ASCII
Set-Content -LiteralPath (Join-Path $RuntimeRoot "smoke-test-a100.ps1") -Value $smokeTest -Encoding ASCII
Set-Content -LiteralPath (Join-Path $RuntimeRoot "README_RUNTIME_A100.md") -Value $readme -Encoding ASCII

$manifest = @"
Created: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
Source repo: $repoRoot
Runtime root: $RuntimeRoot
Assets root: $AssetsRoot
Service: app\service\ChromaOnnx.Service.exe
Agent: Gemma + PersonaPlex
PersonaPlex runtime: first-party full-onnx
PersonaPlex fallback: ElBruno.PersonaPlex 0.6.1 codec diagnostics
PersonaPlex capability: full ONNX asset validation plus first-party encoder/backbone/decoder diagnostics
Smoke: smoke-test-a100.ps1 runs two Gemma tool turns by default or PersonaPlex full diagnostics with -Mode PersonaPlexFull
"@
Set-Content -LiteralPath (Join-Path $RuntimeRoot "RUNTIME_MANIFEST.txt") -Value $manifest -Encoding ASCII

Write-Host "PersonaPlex A100 runtime is ready at $RuntimeRoot"
Write-Host "Shared assets expected at $AssetsRoot"
