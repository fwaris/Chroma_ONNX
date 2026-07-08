namespace ChromaOnnx.Service

open System
open System.IO
open System.Net.WebSockets
open System.Text
open System.Text.Json
open System.Threading
open System.Threading.Tasks
open Microsoft.AspNetCore.Builder
open Microsoft.AspNetCore.Http
open Microsoft.Extensions.Configuration
open ChromaOnnx
open ChromaOnnx.SpeechToSpeech

module VoiceAgentWebApp =
    let private jsonOptions =
        JsonSerializerOptions(PropertyNamingPolicy = JsonNamingPolicy.CamelCase, WriteIndented = true)

    let bindOptions (configuration: IConfiguration) =
        let options = VoiceAgentOptions()
        configuration.GetSection("VoiceAgent").Bind(options)
        options

    let private indexHtml =
        """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Gemma + TTS</title>
  <style>
    body { margin: 0; font: 15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; color: #17212b; background: #fff; }
    header { padding: 14px 22px; border-bottom: 1px solid #d7dee6; display: flex; align-items: center; justify-content: space-between; gap: 16px; }
    main { display: grid; grid-template-columns: minmax(320px, 430px) 1fr; min-height: calc(100vh - 58px); }
    form { padding: 18px 22px; background: #f5f7f9; border-right: 1px solid #d7dee6; display: grid; gap: 13px; align-content: start; }
    label { display: grid; gap: 6px; font-weight: 650; }
    textarea, input, select, button { font: inherit; }
    textarea, input[type=file], select { width: 100%; border: 1px solid #cbd4dd; border-radius: 6px; padding: 9px 10px; background: white; }
    textarea { min-height: 100px; resize: vertical; }
    button { border: 0; border-radius: 6px; padding: 10px 13px; background: #0f766e; color: white; font-weight: 750; cursor: pointer; }
    button:disabled { opacity: .6; cursor: wait; }
    section { padding: 18px 24px; display: grid; gap: 14px; align-content: start; }
    .status { border: 1px solid #d7dee6; border-radius: 8px; padding: 12px; color: #5d6875; background: white; }
    .audioResults { display: grid; gap: 10px; }
    .audioResult { border: 1px solid #d7dee6; border-radius: 8px; padding: 12px; display: grid; gap: 8px; background: white; }
    .audioResult a { color: #0f766e; font-weight: 650; text-decoration: none; }
    audio { width: min(720px, 100%); }
    pre { margin: 0; max-height: 60vh; overflow: auto; background: #101923; color: #e6edf3; border-radius: 6px; padding: 12px; font-size: 12px; }
    @media (max-width: 840px) { main { grid-template-columns: 1fr; } form { border-right: 0; border-bottom: 1px solid #d7dee6; } }
  </style>
</head>
<body>
  <header><strong>Gemma + TTS</strong><span id="runtime">Checking runtime...</span></header>
  <main>
    <form id="form">
      <label>Mode<select id="mode"><option value="gemma-tts">Gemma + TTS</option></select></label>
      <label>System prompt<textarea id="systemPrompt">You are a concise voice assistant. Use tools when useful. Reply with one or two short spoken sentences unless the user explicitly asks for detail.</textarea></label>
      <label>Turn audio<input id="turnAudio" type="file" accept="audio/*,.f32" required></label>
      <button id="send" type="submit">Send</button>
    </form>
    <section>
      <div id="message" class="status">Idle</div>
      <div id="audioResults" class="audioResults"></div>
      <pre id="details">{}</pre>
    </section>
  </main>
  <script>
    const runtime = document.getElementById('runtime');
    const form = document.getElementById('form');
    const message = document.getElementById('message');
    const details = document.getElementById('details');
    const audioResults = document.getElementById('audioResults');
    const send = document.getElementById('send');
    let audioContext = null;
    let session = null;
    let socket = null;

    function asByteView(samples) { return new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength); }
    function mixToMono(buffer) {
      const mono = new Float32Array(buffer.length);
      for (let c = 0; c < buffer.numberOfChannels; c++) {
        const data = buffer.getChannelData(c);
        for (let i = 0; i < mono.length; i++) mono[i] += data[i] / buffer.numberOfChannels;
      }
      return mono;
    }
    function resampleLinear(samples, sourceRate, targetRate) {
      if (sourceRate === targetRate) return samples;
      const length = Math.max(1, Math.round(samples.length * targetRate / sourceRate));
      const output = new Float32Array(length);
      const scale = sourceRate / targetRate;
      for (let i = 0; i < length; i++) {
        const source = i * scale;
        const left = Math.floor(source);
        const right = Math.min(left + 1, samples.length - 1);
        const mix = source - left;
        output[i] = (samples[left] || 0) * (1 - mix) + (samples[right] || 0) * mix;
      }
      return output;
    }
    async function readAudioAs24k(file) {
      const bytes = await file.arrayBuffer();
      if (/\.f32$/i.test(file.name)) return new Uint8Array(bytes);
      audioContext = audioContext || new (window.AudioContext || window.webkitAudioContext)();
      const decoded = await audioContext.decodeAudioData(bytes.slice(0));
      return asByteView(resampleLinear(mixToMono(decoded), decoded.sampleRate, 24000));
    }
    function renderAudioResult(payload) {
      if (!payload.audioUrl) return;
      const url = new URL(payload.audioUrl, location.origin);
      url.searchParams.set('t', Date.now().toString());
      const container = document.createElement('div');
      container.className = 'audioResult';
      const label = document.createElement('a');
      label.href = url.toString();
      label.target = '_blank';
      label.rel = 'noopener';
      label.textContent = `Turn ${payload.turnIndex} audio`;
      const audio = document.createElement('audio');
      audio.controls = true;
      audio.preload = 'auto';
      audio.src = url.toString();
      container.append(label, audio);
      audioResults.prepend(container);
      audio.play()
        .then(() => { message.textContent = payload.finalText || 'Playing response audio.'; })
        .catch(() => { message.textContent = 'Audio is ready. Press play in the response audio control.'; });
    }
    async function ensureSession() {
      if (session) return session;
      const formData = new FormData();
      formData.set('systemPrompt', document.getElementById('systemPrompt').value);
      formData.set('mode', document.getElementById('mode').value);
      const response = await fetch('/api/agent/sessions', { method: 'POST', body: formData });
      if (!response.ok) throw new Error(await response.text());
      session = await response.json();
      socket = new WebSocket(`${location.origin.replace('http', 'ws')}/ws/agent/${session.id}`);
      socket.onmessage = event => {
        if (typeof event.data !== 'string') return;
        const payload = JSON.parse(event.data);
        details.textContent = JSON.stringify(payload, null, 2);
        if (payload.type === 'agent.final_text') message.textContent = payload.text;
        if (payload.type === 'agent.filler_text') message.textContent = payload.text;
        if (payload.type === 'agent.done') renderAudioResult(payload);
        if (payload.type === 'tts.unavailable') message.textContent = payload.message;
      };
      await new Promise((resolve, reject) => {
        socket.onopen = resolve;
        socket.onerror = reject;
      });
      return session;
    }
    async function sendTurn(bytes) {
      socket.send(JSON.stringify({ type: 'turn.start' }));
      socket.send(bytes);
      socket.send(JSON.stringify({ type: 'turn.end' }));
    }
    async function refreshStatus() {
      const response = await fetch('/api/status');
      const status = await response.json();
      runtime.textContent = status.message || 'Ready';
    }
    form.addEventListener('submit', async event => {
      event.preventDefault();
      send.disabled = true;
      try {
        await ensureSession();
        await sendTurn(await readAudioAs24k(document.getElementById('turnAudio').files[0]));
      } catch (error) {
        message.textContent = error.message;
      } finally {
        send.disabled = false;
      }
    });
    refreshStatus().catch(error => runtime.textContent = error.message);
  </script>
</body>
</html>
"""

    let private writeJson (ctx: HttpContext) status payload =
        task {
            ctx.Response.StatusCode <- status
            ctx.Response.ContentType <- "application/json; charset=utf-8"
            do! JsonSerializer.SerializeAsync(ctx.Response.Body, payload, jsonOptions)
        }

    let private writeText (ctx: HttpContext) contentType (text: string) =
        task {
            ctx.Response.ContentType <- contentType
            do! ctx.Response.WriteAsync text
        }

    let private error status message =
        {| error = {| code = status; message = message |} |}

    let private safeId (value: string) =
        not (String.IsNullOrWhiteSpace value)
        && value |> Seq.forall (fun ch -> Char.IsLetterOrDigit(ch) || ch = '_' || ch = '-')

    let private routeValue (ctx: HttpContext) name =
        match ctx.Request.RouteValues.TryGetValue name with
        | true, value when not (isNull value) -> value.ToString()
        | _ -> ""

    let private nullableString value =
        match value with
        | Some text -> text :> obj
        | None -> null

    let private statusPayload (agent: IVoiceAgentRuntime) =
        let status = agent.Status()
        {| ready = status.Ready
           serviceName = status.ServiceName
           mode = status.Mode
           workDir = status.WorkDir
           maxHistoryTurns = status.MaxHistoryTurns
           maxTurnAudioSeconds = status.MaxTurnAudioSeconds
           maxTurnAudioSamples24k = status.MaxTurnAudioSamples24k
           gemma =
            {| ready = status.Gemma.Ready
               modelDir = status.Gemma.ModelDir
               variant = status.Gemma.Variant
               executionProvider = status.Gemma.ExecutionProvider
               missingFiles = status.Gemma.MissingFiles
               loadedSessions = status.Gemma.LoadedSessions
               message = status.Gemma.Message |}
           stt =
            {| ready = status.Stt.Ready
               runtime = status.Stt.Runtime
               inputSampleRate = status.Stt.InputSampleRate
               outputLanguage = status.Stt.OutputLanguage
               message = status.Stt.Message |}
           tts =
            {| ready = status.Tts.Ready
               supportsVoiceCloning = status.Tts.SupportsVoiceCloning
               supportsStreaming = status.Tts.SupportsStreaming
               runtime = status.Tts.Runtime
               modelDir = status.Tts.ModelDir
               executionProvider = status.Tts.ExecutionProvider
               outputSampleRate = status.Tts.OutputSampleRate
               voiceSamplePath = status.Tts.VoiceSamplePath
               missingFiles = status.Tts.MissingFiles
               message = status.Tts.Message |}
           personaPlex =
            {| ready = status.PersonaPlex.Ready
               codecReady = status.PersonaPlex.CodecReady
               speechToSpeechReady = status.PersonaPlex.SpeechToSpeechReady
               supportsStreaming = status.PersonaPlex.SupportsStreaming
               supportsDuplex = status.PersonaPlex.SupportsDuplex
               runtime = status.PersonaPlex.Runtime
               modelDir = status.PersonaPlex.ModelDir
               executionProvider = status.PersonaPlex.ExecutionProvider
               voicePreset = status.PersonaPlex.VoicePreset
               missingFiles = status.PersonaPlex.MissingFiles
               message = status.PersonaPlex.Message |}
           message = status.Message |}

    let private receiveMessage (socket: WebSocket) =
        task {
            let buffer = Array.zeroCreate<byte> 65536
            use stream = new MemoryStream()
            let mutable complete = false
            let mutable messageType = WebSocketMessageType.Close
            while not complete do
                let! result = socket.ReceiveAsync(ArraySegment<byte>(buffer), CancellationToken.None)
                messageType <- result.MessageType
                if result.Count > 0 then
                    stream.Write(buffer, 0, result.Count)
                complete <- result.EndOfMessage || result.MessageType = WebSocketMessageType.Close
            return messageType, stream.ToArray()
        }

    let private sendJson (socket: WebSocket) payload =
        task {
            let bytes = JsonSerializer.SerializeToUtf8Bytes(payload, jsonOptions)
            do! socket.SendAsync(ArraySegment<byte>(bytes), WebSocketMessageType.Text, true, CancellationToken.None)
        }

    let private readSessionRequest (ctx: HttpContext) : Task<VoiceAgentSessionRequest> =
        task {
            if ctx.Request.HasFormContentType then
                let! form = ctx.Request.ReadFormAsync()
                let systemPrompt = form["systemPrompt"].ToString()
                let mode = form["mode"].ToString()
                return ({ SystemPrompt = systemPrompt; Mode = mode }: VoiceAgentSessionRequest)
            elif ctx.Request.ContentType <> null && ctx.Request.ContentType.Contains("application/json", StringComparison.OrdinalIgnoreCase) then
                use! doc = JsonDocument.ParseAsync(ctx.Request.Body)
                let systemPrompt =
                    match doc.RootElement.TryGetProperty("systemPrompt") with
                    | true, value -> value.GetString()
                    | _ -> ""
                let mode =
                    match doc.RootElement.TryGetProperty("mode") with
                    | true, value -> value.GetString()
                    | _ -> ""
                let systemPromptText = if isNull systemPrompt then "" else systemPrompt
                let modeText = if isNull mode then "" else mode
                return ({ SystemPrompt = systemPromptText; Mode = modeText }: VoiceAgentSessionRequest)
            else
                return ({ SystemPrompt = ""; Mode = "" }: VoiceAgentSessionRequest)
        }

    let private createAgentSession (agent: IVoiceAgentRuntime) (ctx: HttpContext) =
        task {
            try
                let! request = readSessionRequest ctx
                let session = agent.CreateSession request
                do! writeJson ctx 200 session
            with
            | :? ArgumentException as ex -> do! writeJson ctx 400 (error 400 ex.Message)
            | ex -> do! writeJson ctx 500 (error 500 ex.Message)
        }

    let private handleAgentSocket (agent: IVoiceAgentRuntime) (ctx: HttpContext) =
        task {
            let sessionId = routeValue ctx "id"
            if not ctx.WebSockets.IsWebSocketRequest then
                do! writeJson ctx 400 (error 400 "Expected WebSocket request.")
            elif not (safeId sessionId) then
                do! writeJson ctx 400 (error 400 "Invalid session id.")
            else
                match agent.TryGetSession sessionId with
                | None -> do! writeJson ctx 404 (error 404 "Agent session was not found.")
                | Some session ->
                    use! socket = ctx.WebSockets.AcceptWebSocketAsync()
                    use sendLock = new SemaphoreSlim(1, 1)
                    use socketCancellation = CancellationTokenSource.CreateLinkedTokenSource(ctx.RequestAborted)
                    let sendJsonLocked payload =
                        task {
                            do! sendLock.WaitAsync()
                            try
                                if socket.State = WebSocketState.Open then
                                    do! sendJson socket payload
                            finally
                                sendLock.Release() |> ignore
                        }
                    let trySendJsonLocked payload =
                        task {
                            try do! sendJsonLocked payload with _ -> ()
                        }

                    let status = agent.Status()
                    do!
                        sendJsonLocked
                            {| ``type`` = "session.ready"
                               id = session.Id
                               mode = session.Mode
                               maxTurnAudioSamples24k = agent.MaxTurnAudioSamples24k
                               gemmaReady = status.Gemma.Ready
                               gemmaMessage = status.Gemma.Message
                               sttReady = status.Stt.Ready
                               sttMessage = status.Stt.Message
                               ttsReady = status.Tts.Ready
                               ttsRuntime = status.Tts.Runtime
                               ttsVoiceCloning = status.Tts.SupportsVoiceCloning
                               ttsMessage = status.Tts.Message
                               personaPlexCodecReady = status.PersonaPlex.CodecReady
                               personaPlexSpeechToSpeechReady = status.PersonaPlex.SpeechToSpeechReady
                               personaPlexRuntime = status.PersonaPlex.Runtime
                               personaPlexMessage = status.PersonaPlex.Message
                               maxHistoryTurns = status.MaxHistoryTurns |}

                    use turnAudio = new MemoryStream()
                    let mutable activeTurnCancellation: CancellationTokenSource option = None
                    let mutable activeTurnTask: Task option = None
                    let cancelActiveTurn () =
                        match activeTurnCancellation with
                        | Some cts ->
                            try cts.Cancel() with _ -> ()
                        | None -> ()
                    let mutable running = true
                    while running && socket.State = WebSocketState.Open do
                        let! messageType, payload = receiveMessage socket
                        match messageType with
                        | WebSocketMessageType.Close ->
                            socketCancellation.Cancel()
                            running <- false
                        | WebSocketMessageType.Binary ->
                            let maxTurnBytes = int64 agent.MaxTurnAudioSamples24k * int64 sizeof<float32>
                            if turnAudio.Length + int64 payload.Length > maxTurnBytes then
                                do!
                                    sendJsonLocked
                                        {| ``type`` = "error"
                                           message = $"User turn audio is too large. The configured maximum is {agent.MaxTurnAudioSamples24k} Float32 samples at 24 kHz." |}
                            else
                                turnAudio.Write(payload, 0, payload.Length)
                                do! sendJsonLocked {| ``type`` = "turn.chunk"; bytes = payload.Length; totalBytes = turnAudio.Length |}
                        | WebSocketMessageType.Text ->
                            try
                                let text = Encoding.UTF8.GetString payload
                                use doc = JsonDocument.Parse text
                                let eventType =
                                    match doc.RootElement.TryGetProperty("type") with
                                    | true, value -> value.GetString()
                                    | _ -> null
                                match eventType with
                                | "turn.start" ->
                                    cancelActiveTurn ()
                                    turnAudio.SetLength 0L
                                    do! sendJsonLocked {| ``type`` = "turn.accepted"; id = session.Id |}
                                | "turn.cancel" ->
                                    cancelActiveTurn ()
                                    do! trySendJsonLocked {| ``type`` = "generation.canceled"; id = session.Id |}
                                | "turn.end" ->
                                    let turnAudioBytes = turnAudio.ToArray()
                                    if turnAudioBytes.Length = 0 then
                                        do! sendJsonLocked {| ``type`` = "error"; message = "User turn audio is required before turn.end." |}
                                    else
                                        let userAudio = AudioChunk.float32FromLittleEndianBytes turnAudioBytes
                                        let emit event =
                                            task {
                                                match event with
                                                | VoiceAgentTranscription(id, requestId, turnIndex, transcript) ->
                                                    do! sendJsonLocked {| ``type`` = "agent.transcription"; id = id; requestId = requestId; turnIndex = turnIndex; transcript = transcript |}
                                                | VoiceAgentToolCall(id, requestId, turnIndex, call) ->
                                                    do! sendJsonLocked {| ``type`` = "agent.tool_call"; id = id; requestId = requestId; turnIndex = turnIndex; round = call.Round; name = call.Name; arguments = call.Arguments; rawText = call.RawText |}
                                                | VoiceAgentToolResult(id, requestId, turnIndex, result) ->
                                                    do! sendJsonLocked {| ``type`` = "agent.tool_result"; id = id; requestId = requestId; turnIndex = turnIndex; round = result.Round; name = result.Name; success = result.Success; result = result.Result; error = nullableString result.Error |}
                                                | VoiceAgentFillerText(id, requestId, turnIndex, fillerText) ->
                                                    do! sendJsonLocked {| ``type`` = "agent.filler_text"; id = id; requestId = requestId; turnIndex = turnIndex; text = fillerText |}
                                                | VoiceAgentFinalText(id, requestId, turnIndex, finalText) ->
                                                    do! sendJsonLocked {| ``type`` = "agent.final_text"; id = id; requestId = requestId; turnIndex = turnIndex; text = finalText |}
                                                | TtsSynthesisStarted(id, requestId, turnIndex, phase, text) ->
                                                    do! sendJsonLocked {| ``type`` = $"tts.{phase}.started"; id = id; requestId = requestId; turnIndex = turnIndex; phase = phase; text = text |}
                                                | TtsAudioChunk(id, requestId, turnIndex, phase, sampleRate, samples) ->
                                                    let bytes = AudioChunk.float32ToLittleEndianBytes samples
                                                    do! sendJsonLocked {| ``type`` = $"tts.{phase}.chunk"; id = id; requestId = requestId; turnIndex = turnIndex; phase = phase; sampleRate = sampleRate; bytes = bytes.Length; samples = samples.Length |}
                                                    do! sendLock.WaitAsync()
                                                    try
                                                        if socket.State = WebSocketState.Open then
                                                            do! socket.SendAsync(ArraySegment<byte>(bytes), WebSocketMessageType.Binary, true, CancellationToken.None)
                                                    finally
                                                        sendLock.Release() |> ignore
                                                | TtsSynthesisDone(id, requestId, turnIndex, result) ->
                                                    do! sendJsonLocked {| ``type`` = $"tts.{result.Phase}.done"; id = id; requestId = requestId; turnIndex = turnIndex; phase = result.Phase; text = result.Text; outputPath = nullableString result.OutputPath; sampleRate = result.SampleRate; samples = result.Samples; durationMs = result.DurationMs; inferenceTimeMs = result.InferenceTimeMs; message = result.Message |}
                                                | TtsSynthesisCanceled(id, requestId, turnIndex, phase) ->
                                                    do! sendJsonLocked {| ``type`` = $"tts.{phase}.canceled"; id = id; requestId = requestId; turnIndex = turnIndex; phase = phase |}
                                                | TtsUnavailable(id, requestId, turnIndex, phase, message) ->
                                                    do! sendJsonLocked {| ``type`` = "tts.unavailable"; id = id; requestId = requestId; turnIndex = turnIndex; phase = phase; message = message |}
                                                | PersonaPlexCodecStarted(id, requestId, turnIndex) ->
                                                    do! sendJsonLocked {| ``type`` = "personaplex.codec.started"; id = id; requestId = requestId; turnIndex = turnIndex |}
                                                | PersonaPlexCodecDone(id, requestId, turnIndex, result) ->
                                                    do! sendJsonLocked {| ``type`` = "personaplex.codec.done"; id = id; requestId = requestId; turnIndex = turnIndex; outputPath = nullableString result.OutputPath; durationMs = result.DurationMs; inferenceTimeMs = result.InferenceTimeMs; sampleRate = result.SampleRate; message = result.Message |}
                                                | PersonaPlexGenerationStarted(id, requestId, turnIndex) ->
                                                    do! sendJsonLocked {| ``type`` = "personaplex.generation.started"; id = id; requestId = requestId; turnIndex = turnIndex |}
                                                | PersonaPlexGenerationDone(id, requestId, turnIndex, result) ->
                                                    do! sendJsonLocked {| ``type`` = "personaplex.generation.done"; id = id; requestId = requestId; turnIndex = turnIndex; outputPath = nullableString result.OutputPath; inputFrames = result.InputFrames; generatedFrames = result.GeneratedFrames; durationMs = result.DurationMs; inferenceTimeMs = result.InferenceTimeMs; sampleRate = result.SampleRate; message = result.Message |}
                                                | PersonaPlexAudioChunk(_id, _requestId, _turnIndex, samples) ->
                                                    let bytes = AudioChunk.float32ToLittleEndianBytes samples
                                                    do! sendLock.WaitAsync()
                                                    try
                                                        if socket.State = WebSocketState.Open then
                                                            do! socket.SendAsync(ArraySegment<byte>(bytes), WebSocketMessageType.Binary, true, CancellationToken.None)
                                                    finally
                                                        sendLock.Release() |> ignore
                                                | PersonaPlexUnavailable(id, requestId, turnIndex, message) ->
                                                    do! sendJsonLocked {| ``type`` = "personaplex.unavailable"; id = id; requestId = requestId; turnIndex = turnIndex; message = message |}
                                                | VoiceAgentDone result ->
                                                    do! sendJsonLocked {| ``type`` = "agent.done"; id = result.Id; requestId = result.RequestId; turnIndex = result.TurnIndex; transcript = result.Transcript; finalText = result.FinalText; audioUrl = nullableString result.AudioUrl; detailsUrl = result.DetailsUrl; toolCalls = result.ToolCalls; toolResults = result.ToolResults |}
                                                | VoiceAgentCanceled(id, requestId) ->
                                                    do! sendJsonLocked {| ``type`` = "generation.canceled"; id = id; requestId = nullableString requestId |}
                                            }
                                            :> Task
                                        cancelActiveTurn ()
                                        let turnCancellation = CancellationTokenSource.CreateLinkedTokenSource(ctx.RequestAborted)
                                        activeTurnCancellation <- Some turnCancellation
                                        turnAudio.SetLength 0L
                                        let runTurn =
                                            task {
                                                try
                                                    try
                                                        let! _ =
                                                            agent.RunTurnAsync(
                                                                { SessionId = session.Id
                                                                  UserAudio24k = userAudio
                                                                  RequestId = None },
                                                                emit,
                                                                turnCancellation.Token
                                                            )
                                                        ()
                                                    with
                                                    | :? OperationCanceledException ->
                                                        do! trySendJsonLocked {| ``type`` = "generation.canceled"; id = session.Id |}
                                                    | ex ->
                                                        do! trySendJsonLocked {| ``type`` = "error"; message = ex.Message; agent = statusPayload agent |}
                                                finally
                                                    match activeTurnCancellation with
                                                    | Some current when obj.ReferenceEquals(current, turnCancellation) ->
                                                        activeTurnCancellation <- None
                                                        activeTurnTask <- None
                                                    | _ -> ()
                                                    turnCancellation.Dispose()
                                            }
                                        activeTurnTask <- Some(runTurn :> Task)
                                | _ ->
                                    do! sendJsonLocked {| ``type`` = "error"; message = $"Unknown WebSocket event '{eventType}'." |}
                            with ex ->
                                do! sendJsonLocked {| ``type`` = "error"; message = ex.Message; agent = statusPayload agent |}
                        | _ ->
                            do! sendJsonLocked {| ``type`` = "error"; message = $"Unsupported WebSocket message type {messageType}." |}

                    if socket.State = WebSocketState.Open || socket.State = WebSocketState.CloseReceived then
                        cancelActiveTurn ()
                        do! socket.CloseAsync(WebSocketCloseStatus.NormalClosure, "done", CancellationToken.None)
        }

    let private serveAgentTurnArtifact (agent: IVoiceAgentRuntime) fileName (ctx: HttpContext) =
        task {
            let sessionId = routeValue ctx "id"
            let turnIndexText = routeValue ctx "turnIndex"
            match Int32.TryParse turnIndexText with
            | false, _ -> do! writeJson ctx 400 (error 400 "Invalid turn index.")
            | true, turnIndex ->
                if not (safeId sessionId) || turnIndex < 1 then
                    do! writeJson ctx 400 (error 400 "Invalid session id or turn index.")
                else
                    match agent.TryGetTurnArtifact(sessionId, turnIndex, fileName) with
                    | None -> do! writeJson ctx 404 (error 404 $"Agent turn artifact {fileName} was not found.")
                    | Some artifact ->
                        ctx.Response.ContentType <- artifact.ContentType
                        do! ctx.Response.SendFileAsync artifact.Path
        }

    let map (app: WebApplication) (agent: IVoiceAgentRuntime) =
        app.UseWebSockets() |> ignore
        app.MapGet("/", RequestDelegate(fun ctx -> task { do! writeText ctx "text/html; charset=utf-8" indexHtml })) |> ignore
        app.MapGet("/healthz", RequestDelegate(fun ctx -> writeJson ctx 200 {| ok = true |})) |> ignore
        app.MapGet("/api/status", RequestDelegate(fun ctx -> writeJson ctx 200 (statusPayload agent))) |> ignore
        app.MapPost("/api/agent/sessions", RequestDelegate(fun ctx -> createAgentSession agent ctx)) |> ignore
        app.MapGet("/ws/agent/{id}", RequestDelegate(fun ctx -> handleAgentSocket agent ctx)) |> ignore
        app.MapGet("/api/agent/sessions/{id}/turns/{turnIndex}/details.json", RequestDelegate(fun ctx -> serveAgentTurnArtifact agent "details.json" ctx)) |> ignore
        app.MapGet("/api/agent/sessions/{id}/turns/{turnIndex}/audio.wav", RequestDelegate(fun ctx -> serveAgentTurnArtifact agent "audio.wav" ctx)) |> ignore
        app
