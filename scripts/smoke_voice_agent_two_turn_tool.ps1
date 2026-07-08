param(
    [string]$AssetsRoot = "",
    [string]$RuntimeRoot = "",
    [ValidateSet("chatterbox-onnx", "fake-tone")]
    [string]$TtsRuntime = "fake-tone",
    [string]$TtsModelDir = "",
    [string]$VoiceSamplePath = "",
    [string]$VoiceSampleTranscript = "",
    [int]$TtsMaxSteps = 64,
    [bool]$RequireTtsGpu = $true,
    [switch]$RequireFullTtsGpu,
    [int]$TtsCudaDeviceId = 0,
    [double]$TtsGpuMemoryLimitGb = 0,
    [int]$Turns = 2,
    [string]$RequireTool = "get_current_time",
    [int]$Port = 5065,
    [int]$TimeoutSeconds = 900,
    [int]$EventTimeoutSeconds = 900,
    [string]$CudaBin = "",
    [string]$CudnnBin = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($RuntimeRoot) {
    $RuntimeRoot = [System.IO.Path]::GetFullPath($RuntimeRoot)
} elseif (Test-Path -LiteralPath (Join-Path $Root "app\service\ChromaOnnx.Service.exe")) {
    $RuntimeRoot = [System.IO.Path]::GetFullPath($Root)
} else {
    $RuntimeRoot = [System.IO.Path]::GetFullPath((Split-Path -Parent $Root))
}
if (-not $AssetsRoot) {
    $AssetsRoot = Join-Path (Split-Path -Parent $RuntimeRoot) "VoiceAgent_assets"
}
$AssetsRoot = [System.IO.Path]::GetFullPath($AssetsRoot)
if ($CudaBin) { $env:PATH = "$CudaBin;$env:PATH" }
if ($CudnnBin) { $env:PATH = "$CudnnBin;$env:PATH" }

function Find-DllForProcess([string]$DllName, [string[]]$ExtraDirectories) {
    foreach ($dir in $ExtraDirectories) {
        if ($dir -and (Test-Path -LiteralPath (Join-Path $dir $DllName))) {
            return (Join-Path $dir $DllName)
        }
    }
    $pathValue = $env:PATH
    if (-not [string]::IsNullOrWhiteSpace($pathValue)) {
        foreach ($dir in $pathValue.Split([IO.Path]::PathSeparator, [StringSplitOptions]::RemoveEmptyEntries)) {
            if ($dir -and (Test-Path -LiteralPath (Join-Path $dir $DllName))) {
                return (Join-Path $dir $DllName)
            }
        }
    }
    return $null
}

function Assert-CudaRuntimeDlls([string]$RuntimeRoot) {
    $serviceDir = Join-Path $RuntimeRoot "app\service"
    $extraDirs = @($serviceDir)
    $requiredDlls = @(
        "cudart64_12.dll",
        "cublas64_12.dll",
        "cublasLt64_12.dll",
        "cufft64_11.dll",
        "curand64_10.dll",
        "cudnn64_9.dll"
    )
    $missing = @()
    foreach ($dll in $requiredDlls) {
        if (-not (Find-DllForProcess $dll $extraDirs)) {
            $missing += $dll
        }
    }
    if ($missing.Count -gt 0) {
        throw "Missing CUDA/cuDNN runtime DLL(s) required by the packaged ONNX Runtime CUDA provider: $($missing -join ', '). Install/stage CUDA 12.x runtime plus cuDNN 9, or rerun this script with -CudaBin and -CudnnBin pointing at their bin folders."
    }
}

function Assert-PathExists([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Label was not found: $Path"
    }
}

if (-not $TtsModelDir) {
    $TtsModelDir = Join-Path $AssetsRoot "models\chatterbox-onnx"
}

$env:VoiceAgent__WorkDir = Join-Path $RuntimeRoot "served_runs"
$env:VoiceAgent__MaxHistoryTurns = "8"
$env:VoiceAgent__MaxTurnAudioSeconds = "30"
$env:VoiceAgent__Gemma__ModelDir = Join-Path $AssetsRoot "models\gemma-4-e2b-it-onnx-mobius\Q4_K_M\cuda"
$env:VoiceAgent__Gemma__Variant = "Q4_K_M/cuda"
$env:VoiceAgent__Gemma__Runtime = "ort-genai"
$env:VoiceAgent__Gemma__ExecutionProvider = "cuda"
$env:VoiceAgent__Tts__ModelDir = [System.IO.Path]::GetFullPath($TtsModelDir)
$env:VoiceAgent__Tts__Runtime = $TtsRuntime
$env:VoiceAgent__Tts__ExecutionProvider = "cuda"
$env:VoiceAgent__Tts__Variant = "q4f16"
$env:VoiceAgent__Tts__VoiceSamplePath = $VoiceSamplePath
$env:VoiceAgent__Tts__VoiceSampleTranscript = $VoiceSampleTranscript
$env:VoiceAgent__Tts__OutputSampleRate = "24000"
$env:VoiceAgent__Tts__MaxSteps = [string]$TtsMaxSteps
$env:VoiceAgent__Tts__StreamingChunkSeconds = "0.5"
$env:VoiceAgent__Tts__RequireGpu = [string]$RequireTtsGpu
$env:VoiceAgent__Tts__RequireFullGpu = [string]([bool]$RequireFullTtsGpu)
$env:VoiceAgent__Tts__CudaDeviceId = [string]$TtsCudaDeviceId
$env:VoiceAgent__Tts__GpuMemoryLimitGb = [string]$TtsGpuMemoryLimitGb

Assert-PathExists (Join-Path $RuntimeRoot "app\service\ChromaOnnx.Service.exe") "Voice agent service"
if ($TtsRuntime -eq "chatterbox-onnx") {
    $speechDll = Join-Path $RuntimeRoot "app\service\ChromaOnnx.SpeechToSpeech.dll"
    Assert-PathExists $speechDll "Speech-to-speech runtime DLL"
    $speechDllText = [Text.Encoding]::Unicode.GetString([IO.File]::ReadAllBytes($speechDll))
    if (-not $speechDllText.Contains("chatterbox-onnx")) {
        throw "The deployed ChromaOnnx.SpeechToSpeech.dll does not contain the Chatterbox runtime. Re-apply the latest VoiceAgent A100 runtime diff."
    }
}
Assert-PathExists (Join-Path $env:VoiceAgent__Gemma__ModelDir "genai_config.json") "Gemma ORT GenAI config"
Assert-PathExists (Join-Path $env:VoiceAgent__Gemma__ModelDir "tokenizer.json") "Gemma tokenizer"
Assert-PathExists (Join-Path $env:VoiceAgent__Gemma__ModelDir "embedding\model.onnx") "Gemma embedding graph"
Assert-PathExists (Join-Path $env:VoiceAgent__Gemma__ModelDir "audio_encoder\model.onnx") "Gemma audio encoder graph"
Assert-PathExists (Join-Path $env:VoiceAgent__Gemma__ModelDir "decoder\model.onnx") "Gemma decoder graph"

if ($TtsRuntime -eq "chatterbox-onnx") {
    Assert-PathExists (Join-Path $env:VoiceAgent__Tts__ModelDir "default_voice.wav") "Chatterbox default voice"
    Assert-PathExists (Join-Path $env:VoiceAgent__Tts__ModelDir "tokenizer.json") "Chatterbox tokenizer"
    Assert-PathExists (Join-Path $env:VoiceAgent__Tts__ModelDir "onnx\speech_encoder.onnx") "Chatterbox speech encoder"
    Assert-PathExists (Join-Path $env:VoiceAgent__Tts__ModelDir "onnx\speech_encoder.onnx_data") "Chatterbox speech encoder external data"
    Assert-PathExists (Join-Path $env:VoiceAgent__Tts__ModelDir "onnx\embed_tokens.onnx") "Chatterbox embed tokens"
    Assert-PathExists (Join-Path $env:VoiceAgent__Tts__ModelDir "onnx\embed_tokens.onnx_data") "Chatterbox embed tokens external data"
    Assert-PathExists (Join-Path $env:VoiceAgent__Tts__ModelDir "onnx\language_model_q4f16.onnx") "Chatterbox q4f16 language model"
    Assert-PathExists (Join-Path $env:VoiceAgent__Tts__ModelDir "onnx\language_model_q4f16.onnx_data") "Chatterbox q4f16 language model external data"
    Assert-PathExists (Join-Path $env:VoiceAgent__Tts__ModelDir "onnx\conditional_decoder.onnx") "Chatterbox conditional decoder"
    Assert-PathExists (Join-Path $env:VoiceAgent__Tts__ModelDir "onnx\conditional_decoder.onnx_data") "Chatterbox conditional decoder external data"
    if ($VoiceSamplePath) {
        Assert-PathExists ([System.IO.Path]::GetFullPath($VoiceSamplePath)) "Voice sample"
    }
}
Assert-CudaRuntimeDlls $RuntimeRoot

$logDir = Join-Path $RuntimeRoot "served_runs\smoke_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$outLog = Join-Path $logDir "voice_agent_two_turn_tool.out.log"
$errLog = Join-Path $logDir "voice_agent_two_turn_tool.err.log"
$url = "http://localhost:$Port"
$exe = Join-Path $RuntimeRoot "app\service\ChromaOnnx.Service.exe"

function Convert-FloatsToBytes([single[]]$Samples) {
    $bytes = New-Object byte[] ($Samples.Length * 4)
    [Buffer]::BlockCopy($Samples, 0, $bytes, 0, $bytes.Length)
    $bytes
}

function New-TestAudio([int]$TurnIndex) {
    $sampleCount = 24000
    $frequency = if (($TurnIndex % 2) -eq 0) { 330.0 } else { 220.0 }
    $samples = New-Object single[] $sampleCount
    for ($i = 0; $i -lt $samples.Length; $i++) {
        $samples[$i] = [single](0.02 * [Math]::Sin(2.0 * [Math]::PI * $frequency * $i / 24000.0))
    }
    Convert-FloatsToBytes $samples
}

function Send-Text($Socket, [string]$Text) {
    $bytes = [Text.Encoding]::UTF8.GetBytes($Text)
    [void]$Socket.SendAsync([ArraySegment[byte]]::new($bytes), [Net.WebSockets.WebSocketMessageType]::Text, $true, [Threading.CancellationToken]::None).GetAwaiter().GetResult()
}

function Send-Binary($Socket, [byte[]]$Bytes) {
    [void]$Socket.SendAsync([ArraySegment[byte]]::new($Bytes), [Net.WebSockets.WebSocketMessageType]::Binary, $true, [Threading.CancellationToken]::None).GetAwaiter().GetResult()
}

function Receive-Json($Socket, [int]$TimeoutSeconds) {
    $buffer = New-Object byte[] 65536
    $stream = [IO.MemoryStream]::new()
    $cts = [Threading.CancellationTokenSource]::new([TimeSpan]::FromSeconds($TimeoutSeconds))
    try {
        do {
            $result = $Socket.ReceiveAsync([ArraySegment[byte]]::new($buffer), $cts.Token).GetAwaiter().GetResult()
            if ($result.MessageType -eq [Net.WebSockets.WebSocketMessageType]::Close) {
                throw "WebSocket closed before the smoke test completed."
            }
            if ($result.Count -gt 0) {
                $stream.Write($buffer, 0, $result.Count)
            }
        } while (-not $result.EndOfMessage)
        if ($result.MessageType -ne [Net.WebSockets.WebSocketMessageType]::Text) {
            return $null
        }
        [Text.Encoding]::UTF8.GetString($stream.ToArray()) | ConvertFrom-Json
    } finally {
        $cts.Dispose()
        $stream.Dispose()
    }
}

function Require-Event([object[]]$Events, [string]$Type, [int]$TurnIndex) {
    if (-not ($Events | Where-Object { $_.type -eq $Type -and $_.turnIndex -eq $TurnIndex } | Select-Object -First 1)) {
        $seen = ($Events | ForEach-Object { $_.type }) -join ","
        throw "Turn $TurnIndex did not emit $Type. Events: $seen"
    }
}

function Require-ToolEvent([object[]]$Events, [string]$Type, [string]$ToolName, [int]$TurnIndex) {
    $matching = $Events | Where-Object { $_.type -eq $Type -and $_.turnIndex -eq $TurnIndex -and $_.name -eq $ToolName } | Select-Object -First 1
    if (-not $matching) {
        $seen = ($Events | ForEach-Object {
            if ($_.name) { "$($_.type):$($_.name)" } else { $_.type }
        }) -join ","
        throw "Turn $TurnIndex did not emit $Type for tool '$ToolName'. Events: $seen"
    }
    $matching
}

Write-Host "Starting GemmaVoiceAgent smoke test on $url with TTS runtime $TtsRuntime..."
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
    if (-not $status.tts.ready) { throw "TTS is not ready: $($status.tts.message)" }
    Write-Host "Status ready. TTS runtime: $($status.tts.runtime). Voice cloning: $($status.tts.supportsVoiceCloning)"

    Add-Type -AssemblyName System.Net.Http
    $http = [System.Net.Http.HttpClient]::new()
    try {
        $form = [System.Net.Http.MultipartFormDataContent]::new()
        try {
            $prompt = "You are in a two-turn A100 smoke test. For every user turn, first call the $RequireTool tool exactly once. After the tool result, return exactly: Done."
            $form.Add([System.Net.Http.StringContent]::new($prompt), "systemPrompt")
            $form.Add([System.Net.Http.StringContent]::new("gemma-tts"), "mode")
            $response = $http.PostAsync("$url/api/agent/sessions", $form).GetAwaiter().GetResult()
            if (-not $response.IsSuccessStatusCode) {
                throw "Session create failed: $($response.StatusCode) $($response.Content.ReadAsStringAsync().Result)"
            }
            $session = $response.Content.ReadAsStringAsync().Result | ConvertFrom-Json
        } finally {
            $form.Dispose()
        }

        $ws = [Net.WebSockets.ClientWebSocket]::new()
        try {
            [void]$ws.ConnectAsync([Uri]"ws://localhost:$Port$($session.websocketUrl)", [Threading.CancellationToken]::None).GetAwaiter().GetResult()
            $ready = Receive-Json $ws 30
            if ($ready.type -ne "session.ready") { throw "Unexpected first WebSocket event: $($ready.type)" }
            if (-not $ready.gemmaReady) { throw "Gemma was not ready on WebSocket session: $($ready.gemmaMessage)" }
            if (-not $ready.ttsReady) { throw "TTS was not ready on WebSocket session: $($ready.ttsMessage)" }

            for ($turn = 1; $turn -le $Turns; $turn++) {
                Write-Host "Running turn $turn..."
                Send-Text $ws '{"type":"turn.start"}'
                $accepted = Receive-Json $ws 30
                if ($accepted.type -ne "turn.accepted") { throw "Expected turn.accepted, got $($accepted.type)" }
                Send-Binary $ws (New-TestAudio $turn)
                $chunkAck = Receive-Json $ws 30
                if ($chunkAck.type -ne "turn.chunk") { throw "Expected turn.chunk, got $($chunkAck.type)" }
                Send-Text $ws '{"type":"turn.end"}'

                $events = @()
                $done = $null
                while (-not $done -and (Get-Date) -lt $deadline) {
                    $event = Receive-Json $ws $EventTimeoutSeconds
                    if ($event) {
                        $events += $event
                        if ($event.type -eq "error") {
                            throw "Service error during turn ${turn}: $($event.message)"
                        }
                        if ($event.type -eq "agent.done") {
                            $done = $event
                        }
                    }
                }
                if (-not $done) {
                    $seen = ($events | ForEach-Object { $_.type }) -join ","
                    throw "Turn $turn did not complete. Events: $seen"
                }
                if ($done.turnIndex -ne $turn) { throw "Expected turnIndex $turn, got $($done.turnIndex)" }

                Require-Event $events "agent.transcription" $turn
                Require-ToolEvent $events "agent.tool_call" $RequireTool $turn | Out-Null
                $toolResult = Require-ToolEvent $events "agent.tool_result" $RequireTool $turn
                if ($toolResult.success -ne $true) {
                    throw "Tool '$RequireTool' failed on turn ${turn}: $($toolResult.error)"
                }
                Require-Event $events "agent.filler_text" $turn
                Require-Event $events "tts.filler.started" $turn
                Require-Event $events "tts.filler.done" $turn
                Require-Event $events "agent.final_text" $turn
                Require-Event $events "tts.final.started" $turn
                Require-Event $events "tts.final.done" $turn

                $detailsResponse = $http.GetAsync("$url/api/agent/sessions/$($session.id)/turns/$turn/details.json").GetAwaiter().GetResult()
                if (-not $detailsResponse.IsSuccessStatusCode) {
                    throw "Details fetch failed for turn ${turn}: $($detailsResponse.StatusCode)"
                }
                $detailsText = $detailsResponse.Content.ReadAsStringAsync().GetAwaiter().GetResult()
                $details = $detailsText | ConvertFrom-Json
                if ($details.turnIndex -ne $turn) { throw "Details turnIndex mismatch. Expected $turn, got $($details.turnIndex)" }
                if (-not ($details.toolCalls | Where-Object { $_.name -eq $RequireTool } | Select-Object -First 1)) {
                    throw "Details for turn $turn did not record tool '$RequireTool'."
                }
                if (-not $details.ttsStatus) { throw "Details for turn $turn did not record TTS status." }
                if ($done.audioUrl) {
                    $audioResponse = $http.GetAsync("$url$($done.audioUrl)").GetAwaiter().GetResult()
                    if (-not $audioResponse.IsSuccessStatusCode) {
                        throw "Audio artifact fetch failed for turn ${turn}: $($audioResponse.StatusCode)"
                    }
                } else {
                    throw "Turn $turn did not return an audioUrl."
                }
                Write-Host "Turn $turn passed. Final text: $($done.finalText)"
            }
        } finally {
            if ($ws) { $ws.Dispose() }
        }
    } finally {
        $http.Dispose()
    }
    Write-Host "GemmaVoiceAgent smoke test passed."
} finally {
    if ($process -and -not $process.HasExited) {
        Stop-Process -Id $process.Id -Force
    }
}
