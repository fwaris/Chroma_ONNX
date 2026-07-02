namespace ChromaOnnx

open System
open System.Collections.Concurrent
open System.Globalization
open System.IO
open System.Net.WebSockets
open System.Text
open System.Text.Json
open System.Threading
open System.Threading.Tasks
open Microsoft.AspNetCore.Builder
open Microsoft.AspNetCore.Hosting
open Microsoft.AspNetCore.Http
open Microsoft.Extensions.Hosting
open Microsoft.ML.OnnxRuntime.Tensors

type S2sSession =
    { Id: string
      PromptText: string
      SystemPrompt: string
      Backend: string
      PromptAudio24k: float32 array
      MaxNewFrames: int
      CreatedUtc: DateTimeOffset
      WorkDir: string
      mutable LastDetails: JsonElement option }

type S2sSessionStore(workDir: string) =
    let sessions = ConcurrentDictionary<string, S2sSession>(StringComparer.Ordinal)

    let newId () =
        let timestamp = DateTimeOffset.UtcNow.ToString("yyyyMMddHHmmssfff", CultureInfo.InvariantCulture)
        let suffix = Guid.NewGuid().ToString("N").Substring(0, 8)
        $"{timestamp}_{suffix}"

    member _.Create(promptText: string, systemPrompt: string, backend: string, promptAudio24k: float32 array, maxNewFrames: int) =
        let id = newId ()
        let sessionDir = Path.Combine(workDir, "s2s", id)
        Directory.CreateDirectory(sessionDir) |> ignore
        let session =
            { Id = id
              PromptText = promptText
              SystemPrompt = systemPrompt
              Backend = backend
              PromptAudio24k = promptAudio24k
              MaxNewFrames = maxNewFrames
              CreatedUtc = DateTimeOffset.UtcNow
              WorkDir = sessionDir
              LastDetails = None }

        sessions[id] <- session
        session

    member _.TryGet(id: string) =
        match sessions.TryGetValue(id) with
        | true, session -> Some session
        | false, _ -> None

module S2sServe =
    let private jsonOptions =
        JsonSerializerOptions(PropertyNamingPolicy = JsonNamingPolicy.CamelCase, WriteIndented = true)

    let private indexHtml =
        """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ChromaS2SONNX</title>
  <style>
    :root { color-scheme: light; --ink: #18212a; --muted: #5e6a75; --line: #d7dde3; --panel: #f5f7f9; --accent: #0f766e; --danger: #b42318; }
    * { box-sizing: border-box; }
    body { margin: 0; font: 15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--ink); background: white; }
    header { min-height: 58px; padding: 14px 22px; border-bottom: 1px solid var(--line); display: flex; align-items: center; justify-content: space-between; gap: 16px; }
    main { display: grid; grid-template-columns: minmax(330px, 450px) 1fr; min-height: calc(100vh - 58px); }
    form { padding: 18px 22px; background: var(--panel); border-right: 1px solid var(--line); display: grid; gap: 13px; align-content: start; }
    label { display: grid; gap: 6px; font-weight: 650; }
    input, textarea, button, select { font: inherit; }
    textarea, input[type=file], input[type=number], select { width: 100%; border: 1px solid #cbd4dd; border-radius: 6px; padding: 9px 10px; background: white; }
    textarea { min-height: 86px; resize: vertical; }
    button { border: 0; border-radius: 6px; padding: 10px 13px; background: var(--accent); color: white; font-weight: 750; cursor: pointer; }
    button:disabled { opacity: .6; cursor: wait; }
    button.secondary { background: #344054; }
    button.danger { background: var(--danger); }
    .actions { display: grid; grid-template-columns: 1fr auto; gap: 10px; }
    section { padding: 18px 24px; display: grid; gap: 14px; align-content: start; }
    .status { border: 1px solid var(--line); border-radius: 8px; padding: 12px; color: var(--muted); background: white; }
    .bad { color: var(--danger); }
    .metrics { display: grid; grid-template-columns: repeat(3, minmax(130px, 1fr)); gap: 10px; }
    .metric { border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; background: white; }
    .metric span { display: block; color: var(--muted); font-size: 12px; }
    .metric strong { display: block; margin-top: 2px; font-size: 18px; }
    pre { margin: 0; max-height: 46vh; overflow: auto; background: #101923; color: #e6edf3; border-radius: 6px; padding: 12px; font-size: 12px; }
    audio { width: min(720px, 100%); }
    .audioResults { display: grid; gap: 12px; }
    .audioResult { display: grid; gap: 6px; border-top: 1px solid var(--line); padding-top: 12px; }
    .audioResult strong { font-size: 13px; color: #52606d; }
    .timing { font-size: 13px; color: #52606d; }
    [hidden] { display: none !important; }
    @media (max-width: 840px) { main { grid-template-columns: 1fr; } form { border-right: 0; border-bottom: 1px solid var(--line); } .metrics { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <strong>ChromaS2SONNX</strong>
    <span id="runtime">Checking runtime...</span>
  </header>
  <main>
    <form id="sessionForm">
      <label>System prompt<textarea id="systemPrompt">You are a helpful assistant.</textarea></label>
      <label>Reference text<textarea id="promptText" required>Why don't skeletons fight each other. They don't have the guts.</textarea></label>
      <label>Reference audio<input id="promptPcm" type="file" accept="audio/*,.f32"></label>
      <label>Turn source<select id="inputMode">
        <option value="file" selected>Audio file</option>
        <option value="mic">Microphone</option>
      </select></label>
      <label id="turnFileLabel">Turn audio<input id="turnPcm" type="file" accept="audio/*,.f32"></label>
      <label id="micSecondsLabel" hidden>Mic seconds<input id="micSeconds" type="number" min="1" max="60" value="6"></label>
      <label>Backend<select id="backend">
        <option value="fsharp_onnx" selected>F#/ONNX</option>
        <option value="python">Python Chroma</option>
        <option value="both">Both</option>
      </select></label>
      <label>Max response seconds<input id="maxResponseSeconds" type="number" min="1" max="24" step="0.5" value="12"></label>
      <div class="actions">
        <button id="sendButton" type="submit">Send</button>
        <button id="cancelButton" class="danger" type="button" disabled>Cancel</button>
      </div>
    </form>
    <section>
      <div id="message" class="status">Idle</div>
      <div class="metrics">
        <div class="metric"><span>Queue</span><strong id="queueMetric">0</strong></div>
        <div class="metric"><span>Frames</span><strong id="frameMetric">0</strong></div>
        <div class="metric"><span>Streamed</span><strong id="streamMetric">0.00 s</strong></div>
      </div>
      <div id="audioResults" class="audioResults"></div>
      <pre id="details">{}</pre>
    </section>
  </main>
  <script>
    const runtime = document.getElementById('runtime');
    const form = document.getElementById('sessionForm');
    const button = document.getElementById('sendButton');
    const cancelButton = document.getElementById('cancelButton');
    const message = document.getElementById('message');
    const details = document.getElementById('details');
    const audioResults = document.getElementById('audioResults');
    const inputMode = document.getElementById('inputMode');
    const turnFileLabel = document.getElementById('turnFileLabel');
    const micSecondsLabel = document.getElementById('micSecondsLabel');
    const queueMetric = document.getElementById('queueMetric');
    const frameMetric = document.getElementById('frameMetric');
    const streamMetric = document.getElementById('streamMetric');

    let audioContext = null;
    let playbackContext = null;
    let playbackCursor = 0;
    let currentSocket = null;
    let activeMicStop = null;
    let pendingChunk = null;
    let streamedSamples = 0;
    let latestFrame = 0;
    const defaultPromptTextUrl = '/assets/southern_belle_prompt.txt';
    const defaultPromptAudioUrl = '/assets/southern_belle.mp3';

    function setBusy(isBusy) {
      button.disabled = isBusy;
      cancelButton.disabled = !isBusy;
      button.textContent = isBusy ? 'Running' : 'Send';
      button.setAttribute('aria-busy', isBusy ? 'true' : 'false');
    }

    function cacheBust(url) {
      if (!url) return '';
      return `${url}${url.includes('?') ? '&' : '?'}v=${Date.now()}`;
    }

    function formatDuration(ms) {
      if (!Number.isFinite(ms)) return '';
      return ms >= 1000 ? `${(ms / 1000).toFixed(ms >= 10000 ? 1 : 2)} s` : `${Math.round(ms)} ms`;
    }

    function timingText(result) {
      const timings = result.timingsMs || {};
      const parts = [];
      if (Number.isFinite(timings.totalMs)) parts.push(`end-to-end ${formatDuration(timings.totalMs)}`);
      if (Number.isFinite(timings.prefillMs)) parts.push(`prefill ${formatDuration(timings.prefillMs)}`);
      if (Number.isFinite(timings.generateMs)) parts.push(`generate ${formatDuration(timings.generateMs)}`);
      if (Number.isFinite(timings.decodeMs)) parts.push(`decode ${formatDuration(timings.decodeMs)}`);
      return parts.join(' | ');
    }

    function renderBackendResults(results) {
      audioResults.innerHTML = '';
      for (const result of results) {
        const block = document.createElement('div');
        block.className = 'audioResult';
        const label = document.createElement('strong');
        label.textContent = result.label || result.backend || 'Result';
        const timing = document.createElement('div');
        timing.className = 'timing';
        timing.textContent = timingText(result);
        const player = document.createElement('audio');
        player.controls = true;
        player.src = cacheBust(result.audioUrl);
        const links = document.createElement('div');
        if (result.audioUrl) {
          const download = document.createElement('a');
          download.href = result.audioUrl;
          download.download = `${result.backend || 'audio'}.wav`;
          download.textContent = 'Download WAV';
          links.append(download);
        }
        if (result.detailsUrl) {
          if (links.childNodes.length) links.append(' | ');
          const detailLink = document.createElement('a');
          detailLink.href = result.detailsUrl;
          detailLink.textContent = 'Details';
          links.append(detailLink);
        }
        block.append(label);
        if (timing.textContent) block.append(timing);
        block.append(player, links);
        audioResults.append(block);
      }
    }

    async function ensureAudioContext() {
      audioContext = audioContext || new (window.AudioContext || window.webkitAudioContext)();
      if (audioContext.state === 'suspended') await audioContext.resume();
      return audioContext;
    }

    function asByteView(samples) {
      return new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength);
    }

    function mixToMono(audioBuffer) {
      const frames = audioBuffer.length;
      const channels = audioBuffer.numberOfChannels;
      const mono = new Float32Array(frames);
      for (let channel = 0; channel < channels; channel++) {
        const data = audioBuffer.getChannelData(channel);
        for (let index = 0; index < frames; index++) mono[index] += data[index] / channels;
      }
      return mono;
    }

    function resampleLinear(samples, sourceRate, targetRate) {
      if (sourceRate === targetRate) return samples;
      const length = Math.max(1, Math.round(samples.length * targetRate / sourceRate));
      const output = new Float32Array(length);
      const scale = sourceRate / targetRate;
      for (let index = 0; index < length; index++) {
        const position = index * scale;
        const left = Math.floor(position);
        const fraction = position - left;
        const a = samples[Math.min(left, samples.length - 1)] || 0;
        const b = samples[Math.min(left + 1, samples.length - 1)] || a;
        output[index] = a + (b - a) * fraction;
      }
      return output;
    }

    async function readAudioBufferAsF32Bytes(bytes, fileName, targetRate) {
      if (/\.f32$/i.test(fileName)) return new Uint8Array(bytes);
      const ctx = await ensureAudioContext();
      const audioBuffer = await ctx.decodeAudioData(bytes.slice(0));
      const mono = mixToMono(audioBuffer);
      const resampled = resampleLinear(mono, audioBuffer.sampleRate, targetRate);
      return asByteView(resampled);
    }

    async function readAudioAsF32Bytes(input, targetRate) {
      if (!input.files.length) throw new Error(`${input.id} is required`);
      const file = input.files[0];
      return readAudioBufferAsF32Bytes(await file.arrayBuffer(), file.name, targetRate);
    }

    async function readReferenceAudioAsF32Bytes(targetRate) {
      const input = document.getElementById('promptPcm');
      if (input.files.length) return readAudioAsF32Bytes(input, targetRate);
      const response = await fetch(defaultPromptAudioUrl);
      if (!response.ok) throw new Error('Default reference audio was not found.');
      return readAudioBufferAsF32Bytes(await response.arrayBuffer(), 'southern_belle.mp3', targetRate);
    }

    async function sendBytesInChunks(ws, bytes, chunkBytes = 32768) {
      for (let offset = 0; offset < bytes.length; offset += chunkBytes) {
        if (ws.readyState !== WebSocket.OPEN) throw new Error('WebSocket closed while sending audio.');
        ws.send(bytes.slice(offset, Math.min(offset + chunkBytes, bytes.length)));
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }

    async function playFloat32Chunk(samples, sampleRate) {
      playbackContext = playbackContext || new (window.AudioContext || window.webkitAudioContext)();
      if (playbackContext.state === 'suspended') await playbackContext.resume();
      const buffer = playbackContext.createBuffer(1, samples.length, sampleRate);
      buffer.copyToChannel(samples, 0);
      const source = playbackContext.createBufferSource();
      source.buffer = buffer;
      source.connect(playbackContext.destination);
      const now = playbackContext.currentTime;
      if (playbackCursor < now + 0.04) playbackCursor = now + 0.04;
      source.start(playbackCursor);
      playbackCursor += buffer.duration;
    }

    async function handleBinaryChunk(arrayBuffer) {
      const chunk = pendingChunk;
      pendingChunk = null;
      if (!chunk) return;
      const samples = new Float32Array(arrayBuffer);
      streamedSamples += samples.length;
      streamMetric.textContent = `${(streamedSamples / (chunk.sampleRate || 24000)).toFixed(2)} s`;
      await playFloat32Chunk(samples, chunk.sampleRate || 24000);
    }

    function resetRunUi() {
      audioResults.innerHTML = '';
      details.textContent = '{}';
      queueMetric.textContent = '0';
      frameMetric.textContent = '0';
      streamMetric.textContent = '0.00 s';
      pendingChunk = null;
      streamedSamples = 0;
      latestFrame = 0;
      playbackCursor = 0;
    }

    function updateMode() {
      const mic = inputMode.value === 'mic';
      turnFileLabel.hidden = mic;
      micSecondsLabel.hidden = !mic;
    }

    async function streamMicrophone(ws, seconds) {
      const media = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true }, video: false });
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const source = ctx.createMediaStreamSource(media);
      const processor = ctx.createScriptProcessor(4096, 1, 1);
      const silent = ctx.createGain();
      silent.gain.value = 0;
      let finished = false;
      let timeoutId = 0;
      const stop = () => {
        if (finished) return;
        finished = true;
        clearTimeout(timeoutId);
        processor.disconnect();
        source.disconnect();
        silent.disconnect();
        media.getTracks().forEach(track => track.stop());
        ctx.close().catch(() => {});
        activeMicStop = null;
        if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'turn.end' }));
      };
      activeMicStop = stop;
      processor.onaudioprocess = event => {
        if (finished || ws.readyState !== WebSocket.OPEN) return;
        const input = event.inputBuffer.getChannelData(0);
        const copy = new Float32Array(input.length);
        copy.set(input);
        const resampled = resampleLinear(copy, ctx.sampleRate, 16000);
        ws.send(asByteView(resampled));
      };
      source.connect(processor);
      processor.connect(silent);
      silent.connect(ctx.destination);
      timeoutId = setTimeout(stop, Math.max(1, seconds) * 1000);
    }

    async function refreshStatus() {
      const response = await fetch('/api/status');
      const payload = await response.json();
      runtime.textContent = payload.ready ? `Ready: ${payload.executionProvider}` : 'Not ready';
      runtime.className = payload.ready ? '' : 'bad';
      queueMetric.textContent = String(payload.queueLength || 0);
      details.textContent = JSON.stringify(payload, null, 2);
    }

    function describeEvent(payload) {
      switch (payload.type) {
        case 'queue.enqueued':
        case 'queue.updated':
          if (payload.isRunning) return 'Running';
          return `Queued ${payload.position || 0}/${payload.queueLength || 0}`;
        case 'queue.started':
          return 'Queue started';
        case 'generation.started':
          return 'Generating';
        case 'generation.frame':
          latestFrame = Math.max(latestFrame, (payload.frameIndex || 0) + 1);
          frameMetric.textContent = String(latestFrame);
          return `Frame ${latestFrame}`;
        case 'audio.chunk':
          return `Streaming chunk ${(payload.chunkIndex || 0) + 1}`;
        case 'audio.deferred':
          return `Deferred audio decode, ${payload.freeVramMb || 0} MiB VRAM free`;
        case 'generation.done':
          return 'Done';
        case 'generation.canceled':
          return 'Canceled';
        case 'turn.chunk':
          return `Sent ${(payload.totalBytes / 1024).toFixed(1)} KiB`;
        default:
          return payload.type || 'Event';
      }
    }

    async function startSocket(session, turnBytes, mode) {
      const ws = new WebSocket(`${location.origin.replace('http', 'ws')}/ws/s2s/${session.id}`);
      currentSocket = ws;
      ws.binaryType = 'arraybuffer';
      let generationStartedAt = 0;
      ws.onmessage = event => {
        if (typeof event.data !== 'string') {
          handleBinaryChunk(event.data).catch(error => {
            message.textContent = error.message;
            message.className = 'status bad';
          });
          return;
        }
        const payload = JSON.parse(event.data);
        details.textContent = JSON.stringify(payload, null, 2);
        if (Number.isFinite(payload.queueLength)) queueMetric.textContent = String(payload.queueLength);
        if (payload.type === 'generation.started') generationStartedAt = performance.now();
        if (payload.type === 'audio.chunk') pendingChunk = payload;
        if (payload.type === 'generation.done' && Array.isArray(payload.results)) renderBackendResults(payload.results);
        if (payload.type === 'error') {
          message.textContent = payload.message;
          message.className = 'status bad';
          setBusy(false);
          if (ws.readyState === WebSocket.OPEN) ws.close(1000, 'error handled');
          return;
        }
        if (payload.type === 'generation.done' && generationStartedAt > 0) {
          const truncated = Array.isArray(payload.results) && payload.results.some(result => result.truncatedByMaxFrames);
          message.textContent = truncated
            ? `Stopped at max response length in ${formatDuration(performance.now() - generationStartedAt)}`
            : `Done in ${formatDuration(performance.now() - generationStartedAt)}`;
        } else {
          message.textContent = describeEvent(payload);
        }
        message.className = payload.type === 'generation.canceled' ? 'status bad' : 'status';
        if (payload.type === 'generation.done' || payload.type === 'generation.canceled') {
          setBusy(false);
          if (ws.readyState === WebSocket.OPEN) ws.close(1000, payload.type);
        }
      };
      ws.onopen = async () => {
        try {
          ws.send(JSON.stringify({ type: 'turn.start' }));
          if (mode === 'mic') {
            const seconds = Number(document.getElementById('micSeconds').value) || 6;
            await streamMicrophone(ws, seconds);
          } else {
            await sendBytesInChunks(ws, turnBytes);
            ws.send(JSON.stringify({ type: 'turn.end' }));
          }
        } catch (error) {
          message.textContent = error.message;
          message.className = 'status bad';
          setBusy(false);
          if (ws.readyState === WebSocket.OPEN) ws.close(1011, error.message);
        }
      };
      ws.onerror = () => {
        message.textContent = 'WebSocket error while generating.';
        message.className = 'status bad';
        setBusy(false);
      };
      ws.onclose = () => {
        if (currentSocket === ws) currentSocket = null;
        if (activeMicStop) activeMicStop();
        setBusy(false);
      };
    }

    form.addEventListener('submit', async event => {
      event.preventDefault();
      if (currentSocket && currentSocket.readyState === WebSocket.OPEN) currentSocket.close(1000, 'new request');
      setBusy(true);
      resetRunUi();
      message.textContent = 'Creating session';
      message.className = 'status';
      try {
        const mode = inputMode.value;
        const promptPcm = await readReferenceAudioAsF32Bytes(24000);
        const turnPcm = mode === 'file' ? await readAudioAsF32Bytes(document.getElementById('turnPcm'), 16000) : null;
        const formData = new FormData();
        const maxSeconds = Math.max(1, Math.min(24, Number(document.getElementById('maxResponseSeconds').value) || 12));
        const maxFrames = Math.max(1, Math.round(maxSeconds * 12.5));
        formData.set('promptText', document.getElementById('promptText').value);
        formData.set('systemPrompt', document.getElementById('systemPrompt').value);
        formData.set('backend', document.getElementById('backend').value);
        formData.set('maxNewFrames', String(maxFrames));
        formData.set('promptPcm24k', new Blob([promptPcm], { type: 'application/octet-stream' }), 'prompt.f32');
        const sessionResponse = await fetch('/api/s2s/sessions', { method: 'POST', body: formData });
        const session = await sessionResponse.json();
        details.textContent = JSON.stringify(session, null, 2);
        if (!sessionResponse.ok) throw new Error(session.error || sessionResponse.statusText);
        message.textContent = mode === 'mic' ? 'Opening microphone' : 'Sending turn audio';
        await startSocket(session, turnPcm, mode);
      } catch (error) {
        message.textContent = error.message;
        message.className = 'status bad';
        setBusy(false);
      }
    });

    cancelButton.addEventListener('click', () => {
      if (activeMicStop) activeMicStop();
      if (currentSocket && currentSocket.readyState === WebSocket.OPEN) {
        currentSocket.send(JSON.stringify({ type: 'turn.cancel' }));
        currentSocket.close(1000, 'canceled');
      }
      setBusy(false);
    });

    inputMode.addEventListener('change', updateMode);
    updateMode();
    fetch(defaultPromptTextUrl, { cache: 'no-store' })
      .then(response => response.ok ? response.text() : '')
      .then(text => {
        if (text.trim()) document.getElementById('promptText').value = text.trim();
      })
      .catch(() => {});
    refreshStatus().catch(error => {
      runtime.textContent = error.message;
      runtime.className = 'bad';
    });
  </script>
</body>
</html>
"""

    let private writeJson (ctx: HttpContext) (statusCode: int) (payload: 'T) =
        task {
            ctx.Response.StatusCode <- statusCode
            ctx.Response.ContentType <- "application/json; charset=utf-8"
            let json = JsonSerializer.Serialize(payload, jsonOptions)
            do! ctx.Response.WriteAsync(json)
        }

    let private writeText (ctx: HttpContext) (contentType: string) (text: string) =
        task {
            ctx.Response.ContentType <- contentType
            do! ctx.Response.WriteAsync(text)
        }

    let private assetPath fileName =
        [ Path.Combine(AppContext.BaseDirectory, "assets", fileName)
          Path.Combine(Directory.GetCurrentDirectory(), "assets", fileName) ]
        |> List.tryFind File.Exists

    let private serveAsset fileName contentType (ctx: HttpContext) =
        task {
            match assetPath fileName with
            | Some path ->
                ctx.Response.ContentType <- contentType
                do! ctx.Response.SendFileAsync(path)
            | None -> do! writeJson ctx 404 {| error = $"Asset {fileName} was not found."; status = 404 |}
        }

    let private error statusCode message =
        {| error = message; status = statusCode |}

    let private dimsOf (tensor: DenseTensor<'T>) =
        tensor.Dimensions.ToArray()

    let private normalizeBackend (value: string) =
        match value.Trim().ToLowerInvariant() with
        | "" | "fsharp" | "fsharp_onnx" | "onnx" -> "fsharp_onnx"
        | "python" | "python_chroma" -> "python"
        | "both" | "compare" -> "both"
        | other -> invalidArg "backend" $"Unsupported backend '{other}'. Expected fsharp_onnx, python, or both."

    let private backendLabel backend =
        match backend with
        | "fsharp_onnx" -> "F#/ONNX"
        | "python" -> "Python Chroma"
        | _ -> backend

    let private jsonElement payload =
        let json = JsonSerializer.Serialize(payload, jsonOptions)
        use doc = JsonDocument.Parse(json)
        doc.RootElement.Clone()

    let private routeBackend (ctx: HttpContext) =
        match ctx.Request.RouteValues.TryGetValue("backend") with
        | true, value when value <> null -> string value
        | _ -> ""

    let private nullableString (value: string option) : string | null =
        match value with
        | Some text -> text
        | None -> null

    let private sendJson (socket: WebSocket) (payload: 'T) =
        task {
            let bytes = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(payload, jsonOptions))
            do! socket.SendAsync(ArraySegment<byte>(bytes), WebSocketMessageType.Text, true, CancellationToken.None)
        }

    let private sendBinary (socket: WebSocket) (payload: byte array) =
        task {
            do! socket.SendAsync(ArraySegment<byte>(payload), WebSocketMessageType.Binary, true, CancellationToken.None)
        }

    let private readFormFileBytes (form: IFormCollection) (name: string) =
        match form.Files.GetFile(name) with
        | null -> invalidArg name $"Missing multipart file {name}."
        | file when file.Length = 0L -> invalidArg name $"Multipart file {name} is empty."
        | file ->
            use stream = new MemoryStream()
            file.CopyTo(stream)
            stream.ToArray()

    let private routeSessionId (ctx: HttpContext) =
        match ctx.Request.RouteValues.TryGetValue("id") with
        | true, value when value <> null -> string value
        | _ -> ""

    let private isSafeId (value: string) =
        value.Length > 0
        && value |> Seq.forall (fun ch -> Char.IsLetterOrDigit(ch) || ch = '_' || ch = '-')

    let private receiveMessage (socket: WebSocket) =
        task {
            let buffer = Array.zeroCreate<byte> 65536
            use stream = new MemoryStream()
            let mutable messageType = WebSocketMessageType.Close
            let mutable complete = false
            while not complete do
                let! result = socket.ReceiveAsync(ArraySegment<byte>(buffer), CancellationToken.None)
                messageType <- result.MessageType
                if result.Count > 0 then
                    stream.Write(buffer, 0, result.Count)
                complete <- result.EndOfMessage || result.MessageType = WebSocketMessageType.Close

            return messageType, stream.ToArray()
        }

    let private createSession (processor: ChromaNativeProcessor) (maxPromptAudioSamples: int) (store: S2sSessionStore) (ctx: HttpContext) =
        task {
            if not ctx.Request.HasFormContentType then
                do! writeJson ctx 400 (error 400 "Expected multipart/form-data.")
            else
                try
                    let! form = ctx.Request.ReadFormAsync()
                    let promptText = form["promptText"].ToString()
                    let systemPrompt = form["systemPrompt"].ToString()
                    let backend = normalizeBackend (form["backend"].ToString())
                    let maxNewFrames =
                        match Int32.TryParse(form["maxNewFrames"].ToString()) with
                        | true, value -> max 1 (min 300 value)
                        | false, _ -> 150

                    if String.IsNullOrWhiteSpace(promptText) then
                        do! writeJson ctx 400 (error 400 "promptText is required.")
                    else
                        let promptAudioBytes = readFormFileBytes form "promptPcm24k"
                        let maxPromptBytes = maxPromptAudioSamples * sizeof<float32>
                        if promptAudioBytes.Length > maxPromptBytes then
                            do!
                                writeJson
                                    ctx
                                    413
                                    (error
                                        413
                                        $"promptPcm24k is too large. The configured maximum is {maxPromptAudioSamples} Float32 samples.")
                        else
                            let promptAudio = processor.ReadFloat32PcmFromBytes promptAudioBytes
                            let session = store.Create(promptText, systemPrompt, backend, promptAudio, maxNewFrames)
                            File.WriteAllBytes(Path.Combine(session.WorkDir, "prompt_audio_24k.f32"), promptAudioBytes)
                            let payload =
                                {| id = session.Id
                                   serviceName = "ChromaS2SONNX"
                                   mode = "s2s_greedy_streaming"
                                   backend = session.Backend
                                   promptText = session.PromptText
                                   systemPrompt = session.SystemPrompt
                                   maxNewFrames = session.MaxNewFrames
                                   maxResponseSeconds = Math.Round(float session.MaxNewFrames * 0.08, 2)
                                   promptAudioSamples = session.PromptAudio24k.Length
                                   promptSampleRate = processor.PromptSampleRate
                                   websocketUrl = $"/ws/s2s/{session.Id}" |}
                            do! writeJson ctx 200 payload
                with ex ->
                    do! writeJson ctx 400 (error 400 ex.Message)
        }

    let private jsonIntArray (element: JsonElement) (name: string) =
        element.GetProperty(name).EnumerateArray()
        |> Seq.map (fun item -> item.GetInt32())
        |> Seq.toArray

    let private runFsharpBackend
        (processor: ChromaNativeProcessor)
        (runner: ChromaS2sOnnxRunner)
        streamDecodeFrames
        streamMinFreeVramMb
        codecStallGuardFrames
        (shouldDecodeChunk: int -> bool)
        (onFrame: S2sGeneratedFrame -> unit)
        (onAudioChunk: S2sAudioChunk -> unit)
        (cancellationToken: CancellationToken)
        (session: S2sSession)
        (userAudio: float32 array)
        (prepared: NativeS2sPrepared)
        =
        let backend = "fsharp_onnx"
        let backendDir = Path.Combine(session.WorkDir, backend)
        Directory.CreateDirectory(backendDir) |> ignore
        let memoryBefore = RuntimeMemory.current()
        use result =
            runner.GenerateStreaming(
                prepared,
                session.MaxNewFrames,
                streamDecodeFrames,
                onFrame,
                onAudioChunk,
                cancellationToken,
                ?shouldDecodeChunk = Some shouldDecodeChunk,
                ?codecStallGuardFrames = Some codecStallGuardFrames
            )
        let memoryAfter = RuntimeMemory.current()
        let codesPath = Path.Combine(backendDir, "audio_codes.i64")
        let rawAudioPath = Path.Combine(backendDir, "audio_values.f32")
        let wavPath = Path.Combine(backendDir, "audio.wav")
        TensorIO.writeInt64s codesPath result.AudioCodes
        TensorIO.writeSingles rawAudioPath result.AudioValues
        let wavStats = Wave.writeMono16 wavPath 24000 result.AudioValues
        let generationFrameRateFps =
            match result.Timings.TryGetValue "generateMs" with
            | true, generateMs when generateMs > 0.0 -> Nullable(Math.Round(float result.FrameCount / (generateMs / 1000.0), 3))
            | _ -> Nullable<float>()
        let truncatedByMaxFrames = result.StopReason = "max_frames"
        let detailsUrl = $"/api/s2s/sessions/{session.Id}/{backend}/details.json"
        let audioUrl = $"/api/s2s/sessions/{session.Id}/{backend}/audio.wav"
        let details =
            {| id = session.Id
               backend = backend
               label = backendLabel backend
               mode = "s2s_greedy"
               audioUrl = audioUrl
               detailsUrl = detailsUrl
               executionProvider = runner.Status.ExecutionProvider
               memoryMode = runner.MemoryMode
               ortMemoryProfile = runner.OrtMemoryProfile
               cudaGpuMemLimitMb = runner.CudaGpuMemLimitMb |> Option.toNullable
               optimizedModelCacheEnabled = runner.OptimizedModelCacheEnabled
               optimizedModelCacheDir = runner.OptimizedModelCacheDir
               optimizedModelCacheFormat = runner.OptimizedModelCacheFormat
               loadedOrtSessions = runner.LoadedSessionNames
               warmOrtSessions = runner.WarmSessionNames
               activePagedOrtSessions = runner.ActivePagedSessionNames
               mappedSafetensorShards = runner.MappedShardCount
               initializerCount = runner.InitializerCount
               uniqueInitializerSources = runner.UniqueSourceTensorCount
               uniqueOrtValues = runner.UniqueOrtValueCount
               sharedPrepackedWeights = runner.SharedPrepackedWeights
               memoryBefore = memoryBefore
               memoryAfter = memoryAfter
               promptAudioSamples = session.PromptAudio24k.Length
               userAudioSamples = userAudio.Length
               effectiveUserAudioSamples = min userAudio.Length processor.ThinkerTraceSamples
               warning = $"F#/ONNX uses native Whisper-style log-mel preprocessing with {processor.ThinkerFeatureMode}."
               frameCount = result.FrameCount
               maxResponseSeconds = Math.Round(float session.MaxNewFrames * 0.08, 2)
               generationFrameRateFps = generationFrameRateFps
               stopReason = result.StopReason
               truncatedByMaxFrames = truncatedByMaxFrames
               stalledByCodecPattern = result.StopReason = "codec_stall"
               truncationWarning =
                   if truncatedByMaxFrames then
                       $"Generation reached maxNewFrames ({session.MaxNewFrames}, about {Math.Round(float session.MaxNewFrames * 0.08, 2)} seconds) before EOS."
                   else
                       ""
               stallWarning =
                   if result.StopReason = "codec_stall" then
                       $"Generation stopped after {codecStallGuardFrames} consecutive repeated codec frames, which usually indicates repeated near-silence."
                   else
                       ""
               stepKinds = result.StepKinds
               streamDecodeFrames = streamDecodeFrames
               streamMinFreeVramMb = streamMinFreeVramMb
               codecStallGuardFrames = codecStallGuardFrames
               audioCodesShape = dimsOf result.AudioCodes
               audioValuesShape = dimsOf result.AudioValues
               timingsMs = result.Timings
               wav = wavStats
               pythonInRequestPath = false |}
        let detailsJson = JsonSerializer.Serialize(details, jsonOptions)
        File.WriteAllText(Path.Combine(backendDir, "details.json"), detailsJson)
        jsonElement details

    let private runPythonBackend
        modelDir
        python
        pythonDevice
        pythonThinkerActiveFrames
        (processor: ChromaNativeProcessor)
        (session: S2sSession)
        (userAudioPath: string)
        =
        task {
            let backend = "python"
            let backendDir = Path.Combine(session.WorkDir, backend)
            Directory.CreateDirectory(backendDir) |> ignore
            let scriptPath = Path.Combine(Directory.GetCurrentDirectory(), "scripts", "chroma_s2s_backend.py")
            let! processResult =
                ProcessRunner.run
                    python
                    [ scriptPath
                      "--model-dir"
                      modelDir
                      "--prompt-text"
                      session.PromptText
                      "--system-prompt"
                      session.SystemPrompt
                      "--prompt-audio-f32"
                      Path.Combine(session.WorkDir, "prompt_audio_24k.f32")
                      "--user-audio-f32"
                      userAudioPath
                      "--output-dir"
                      backendDir
                      "--max-new-frames"
                      string session.MaxNewFrames
                      "--device"
                      pythonDevice
                      "--thinker-active-frames"
                      pythonThinkerActiveFrames ]
                    (Directory.GetCurrentDirectory())
            File.WriteAllText(Path.Combine(backendDir, "stdout.json"), processResult.Stdout)
            File.WriteAllText(Path.Combine(backendDir, "stderr.txt"), processResult.Stderr)
            if processResult.ExitCode <> 0 then
                invalidOp $"Python Chroma backend failed with exit code {processResult.ExitCode}: {processResult.Stderr}"

            let pythonDetailsPath = Path.Combine(backendDir, "details.json")
            use doc = JsonDocument.Parse(File.ReadAllText(pythonDetailsPath))
            let root = doc.RootElement
            let pythonDetails = root.Clone()
            let detailsUrl = $"/api/s2s/sessions/{session.Id}/{backend}/details.json"
            let audioUrl = $"/api/s2s/sessions/{session.Id}/{backend}/audio.wav"
            let details =
                {| id = session.Id
                   backend = backend
                   label = backendLabel backend
                   mode = root.GetProperty("mode").GetString()
                   audioUrl = audioUrl
                   detailsUrl = detailsUrl
                   device = root.GetProperty("device").GetString()
                   promptAudioSamples = root.GetProperty("promptAudioSamples").GetInt32()
                   userAudioSamples = root.GetProperty("userAudioSamples").GetInt32()
                   effectiveUserAudioSamples = root.GetProperty("effectiveUserAudioSamples").GetInt32()
                   thinkerActiveFrames = root.GetProperty("thinkerActiveFrames").GetInt32()
                   frameCount = root.GetProperty("frameCount").GetInt32()
                   stopReason = root.GetProperty("stopReason").GetString()
                   audioCodesShape = jsonIntArray root "audioCodesShape"
                   audioValuesShape = jsonIntArray root "audioValuesShape"
                   timingsMs = root.GetProperty("timingsMs").Clone()
                   wav = root.GetProperty("wav").Clone()
                   pythonInRequestPath = true
                   pythonDetails = pythonDetails |}
            let detailsJson = JsonSerializer.Serialize(details, jsonOptions)
            File.WriteAllText(pythonDetailsPath, detailsJson)
            return jsonElement details
        }

    let private handleSocket
        (processor: ChromaNativeProcessor)
        (runner: ChromaS2sOnnxRunner)
        modelDir
        python
        pythonDevice
        pythonThinkerActiveFrames
        streamDecodeFrames
        streamMinFreeVramMb
        codecStallGuardFrames
        maxTurnAudioSamples
        (workQueue: StreamingWorkQueue)
        (store: S2sSessionStore)
        (ctx: HttpContext)
        =
        task {
            let sessionId = routeSessionId ctx
            if not ctx.WebSockets.IsWebSocketRequest then
                do! writeJson ctx 400 (error 400 "Expected WebSocket request.")
            elif not (isSafeId sessionId) then
                do! writeJson ctx 400 (error 400 "Invalid session id.")
            else
                match store.TryGet(sessionId) with
                | None -> do! writeJson ctx 404 (error 404 "S2S session was not found.")
                | Some session ->
                    use! socket = ctx.WebSockets.AcceptWebSocketAsync()
                    use sendLock = new SemaphoreSlim(1, 1)
                    use socketCancellation = CancellationTokenSource.CreateLinkedTokenSource(ctx.RequestAborted)
                    let sendJsonLocked payload =
                        task {
                            do! sendLock.WaitAsync()
                            try
                                do! sendJson socket payload
                            finally
                                sendLock.Release() |> ignore
                        }
                    let sendBinaryLocked payload =
                        task {
                            do! sendLock.WaitAsync()
                            try
                                do! sendBinary socket payload
                            finally
                                sendLock.Release() |> ignore
                        }
                    let trySendJsonLocked payload =
                        task {
                            try
                                if socket.State = WebSocketState.Open then
                                    do! sendJsonLocked payload
                            with _ ->
                                ()
                        }
                    let sendJsonBlocking payload =
                        sendJsonLocked payload |> fun work -> work.GetAwaiter().GetResult()
                    let sendBinaryBlocking payload =
                        sendBinaryLocked payload |> fun work -> work.GetAwaiter().GetResult()
                    let queueUpdate (snapshot: WorkQueuePosition) =
                        sendJsonBlocking
                            {| ``type`` = "queue.updated"
                               id = session.Id
                               requestId = snapshot.Id
                               position = snapshot.Position
                               queueLength = snapshot.QueueLength
                               runningId = nullableString snapshot.RunningId
                               isRunning = snapshot.IsRunning |}

                    do!
                        sendJsonLocked
                            {| ``type`` = "session.ready"
                               id = session.Id
                               maxNewFrames = session.MaxNewFrames
                               streamDecodeFrames = streamDecodeFrames
                               streamMinFreeVramMb = streamMinFreeVramMb
                               codecStallGuardFrames = codecStallGuardFrames
                               cudaGpuMemLimitMb = runner.CudaGpuMemLimitMb |> Option.toNullable
                               queueLength = workQueue.QueueLength
                               maxQueueLength = workQueue.MaxQueueLength |}
                    use turnAudio = new MemoryStream()
                    let mutable running = true
                    let mutable queuedHandle: QueuedWorkHandle option = None
                    while running && socket.State = WebSocketState.Open do
                        let! messageType, payload = receiveMessage socket
                        match messageType with
                        | WebSocketMessageType.Close ->
                            queuedHandle |> Option.iter (fun handle -> handle.Cancel())
                            socketCancellation.Cancel()
                            running <- false
                        | WebSocketMessageType.Binary ->
                            let maxTurnBytes = int64 maxTurnAudioSamples * int64 sizeof<float32>
                            if turnAudio.Length + int64 payload.Length > maxTurnBytes then
                                socketCancellation.Cancel()
                                running <- false
                                do!
                                    sendJsonLocked
                                        {| ``type`` = "error"
                                           message = $"User turn audio is too large. The configured maximum is {maxTurnAudioSamples} Float32 samples." |}
                            else
                                turnAudio.Write(payload, 0, payload.Length)
                                do! sendJsonLocked {| ``type`` = "turn.chunk"; bytes = payload.Length; totalBytes = turnAudio.Length |}
                        | WebSocketMessageType.Text ->
                            let text = Encoding.UTF8.GetString(payload)
                            use doc = JsonDocument.Parse(text)
                            let eventType =
                                match doc.RootElement.TryGetProperty("type") with
                                | true, value -> value.GetString()
                                | _ -> null

                            match eventType with
                            | "turn.start" ->
                                turnAudio.SetLength(0L)
                                do! sendJsonLocked {| ``type`` = "turn.accepted"; id = session.Id |}
                            | "turn.cancel" ->
                                queuedHandle |> Option.iter (fun handle -> handle.Cancel())
                                socketCancellation.Cancel()
                                running <- false
                                do! trySendJsonLocked {| ``type`` = "generation.canceled"; id = session.Id |}
                            | "turn.end" ->
                                try
                                    let turnAudioBytes = turnAudio.ToArray()
                                    if turnAudioBytes.Length = 0 then
                                        do! sendJsonLocked {| ``type`` = "error"; message = "User turn audio is required before turn.end." |}
                                    else
                                        let requestId = $"{session.Id}_{Guid.NewGuid():N}"
                                        let work (jobCancellationToken: CancellationToken) : Task =
                                            task {
                                                jobCancellationToken.ThrowIfCancellationRequested()
                                                do!
                                                    sendJsonLocked
                                                        {| ``type`` = "queue.started"
                                                           id = session.Id
                                                           requestId = requestId
                                                           queueLength = workQueue.QueueLength |}
                                                let userAudio = processor.ReadFloat32PcmFromBytes(turnAudioBytes)
                                                let requestedBackends =
                                                    match session.Backend with
                                                    | "both" -> [| "fsharp_onnx"; "python" |]
                                                    | backend -> [| backend |]
                                                File.WriteAllBytes(Path.Combine(session.WorkDir, "user_audio_16k.f32"), turnAudioBytes)
                                                do!
                                                    sendJsonLocked
                                                        {| ``type`` = "generation.started"
                                                           id = session.Id
                                                           requestId = requestId
                                                           backend = session.Backend
                                                           backends = requestedBackends
                                                           maxNewFrames = session.MaxNewFrames
                                                           streamDecodeFrames = streamDecodeFrames
                                                           streamMinFreeVramMb = streamMinFreeVramMb
                                                           codecStallGuardFrames = codecStallGuardFrames |}
                                                let results = ResizeArray<JsonElement>()

                                                if requestedBackends |> Array.contains "fsharp_onnx" then
                                                    use prepared = processor.Prepare(session.PromptText, session.SystemPrompt, session.PromptAudio24k, userAudio)
                                                    File.WriteAllText(Path.Combine(session.WorkDir, "conversation.txt"), prepared.ConversationText)
                                                    let onFrame (frame: S2sGeneratedFrame) =
                                                        sendJsonBlocking
                                                            {| ``type`` = "generation.frame"
                                                               id = session.Id
                                                               requestId = requestId
                                                               frameIndex = frame.FrameIndex
                                                               stepKind = frame.StepKind
                                                               isEos = frame.IsEos
                                                               codes = frame.Codes |}
                                                    let onAudioChunk (chunk: S2sAudioChunk) =
                                                        let bytes = AudioChunk.float32ToLittleEndianBytes chunk.Samples
                                                        sendJsonBlocking
                                                            {| ``type`` = "audio.chunk"
                                                               id = session.Id
                                                               requestId = requestId
                                                               chunkIndex = chunk.ChunkIndex
                                                               startFrame = chunk.StartFrame
                                                               frameCount = chunk.FrameCount
                                                               startSample = chunk.StartSample
                                                               sampleRate = chunk.SampleRate
                                                               sampleCount = chunk.Samples.Length
                                                               byteLength = bytes.Length
                                                               format = "f32le" |}
                                                        sendBinaryBlocking bytes
                                                    let mutable lastDeferredFrame = -streamDecodeFrames
                                                    let shouldDecodeChunk currentFrameCount =
                                                        if streamMinFreeVramMb <= 0 || not (runner.Status.ExecutionProvider.Equals("cuda", StringComparison.OrdinalIgnoreCase)) then
                                                            true
                                                        else
                                                            match RuntimeMemory.tryGlobalGpuMemory() with
                                                            | None -> true
                                                            | Some gpu when gpu.FreeMb >= streamMinFreeVramMb -> true
                                                            | Some gpu ->
                                                                if currentFrameCount - lastDeferredFrame >= streamDecodeFrames then
                                                                    lastDeferredFrame <- currentFrameCount
                                                                    sendJsonBlocking
                                                                        {| ``type`` = "audio.deferred"
                                                                           id = session.Id
                                                                           requestId = requestId
                                                                           frameCount = currentFrameCount
                                                                           freeVramMb = gpu.FreeMb
                                                                           usedVramMb = gpu.UsedMb
                                                                           totalVramMb = gpu.TotalMb
                                                                           minFreeVramMb = streamMinFreeVramMb
                                                                           message = "Deferred partial audio decode to keep CUDA memory below the configured headroom." |}
                                                                false
                                                    results.Add(
                                                        runFsharpBackend
                                                            processor
                                                            runner
                                                            streamDecodeFrames
                                                            streamMinFreeVramMb
                                                            codecStallGuardFrames
                                                            shouldDecodeChunk
                                                            onFrame
                                                            onAudioChunk
                                                            jobCancellationToken
                                                            session
                                                            userAudio
                                                            prepared
                                                    )

                                                if requestedBackends |> Array.contains "python" then
                                                    jobCancellationToken.ThrowIfCancellationRequested()
                                                    let! pythonResult =
                                                        runPythonBackend
                                                            modelDir
                                                            python
                                                            pythonDevice
                                                            pythonThinkerActiveFrames
                                                            processor
                                                            session
                                                            (Path.Combine(session.WorkDir, "user_audio_16k.f32"))
                                                    results.Add(pythonResult)

                                                let firstResult = results[0]
                                                let firstBackend = firstResult.GetProperty("backend").GetString() |> Option.ofObj |> Option.defaultValue "fsharp_onnx"
                                                let firstAudioUrl = firstResult.GetProperty("audioUrl").GetString() |> Option.ofObj |> Option.defaultValue ""
                                                let firstDetailsUrl = firstResult.GetProperty("detailsUrl").GetString() |> Option.ofObj |> Option.defaultValue ""
                                                let firstBackendDir = Path.Combine(session.WorkDir, firstBackend)
                                                let firstAudioPath = Path.Combine(firstBackendDir, "audio.wav")
                                                if File.Exists firstAudioPath then
                                                    File.Copy(firstAudioPath, Path.Combine(session.WorkDir, "audio.wav"), true)
                                                let details =
                                                    {| id = session.Id
                                                       requestId = requestId
                                                       mode = "s2s_greedy_streaming"
                                                       backend = session.Backend
                                                       maxNewFrames = session.MaxNewFrames
                                                       streamDecodeFrames = streamDecodeFrames
                                                       streamMinFreeVramMb = streamMinFreeVramMb
                                                       codecStallGuardFrames = codecStallGuardFrames
                                                       pythonInRequestPath = requestedBackends |> Array.contains "python"
                                                       results = results.ToArray() |}
                                                let detailsJson = JsonSerializer.Serialize(details, jsonOptions)
                                                File.WriteAllText(Path.Combine(session.WorkDir, "details.json"), detailsJson)
                                                do!
                                                    sendJsonLocked
                                                        {| ``type`` = "generation.done"
                                                           id = session.Id
                                                           requestId = requestId
                                                           backend = session.Backend
                                                           audioUrl = firstAudioUrl
                                                           detailsUrl = firstDetailsUrl
                                                           results = results.ToArray() |}
                                            }
                                            :> Task

                                        match workQueue.TryEnqueue(requestId, work, queueUpdate, socketCancellation.Token) with
                                        | QueueFull maxQueueLength ->
                                            do!
                                                sendJsonLocked
                                                    {| ``type`` = "error"
                                                       message = $"Generation queue is full. Try again after a current request finishes."
                                                       maxQueueLength = maxQueueLength |}
                                        | Enqueued(handle, snapshot) ->
                                            queuedHandle <- Some handle
                                            do!
                                                sendJsonLocked
                                                    {| ``type`` = "queue.enqueued"
                                                       id = session.Id
                                                       requestId = requestId
                                                       position = snapshot.Position
                                                       queueLength = snapshot.QueueLength
                                                       runningId = nullableString snapshot.RunningId
                                                       maxQueueLength = workQueue.MaxQueueLength |}
                                            try
                                                do! handle.Completion
                                                running <- false
                                            with
                                            | :? OperationCanceledException ->
                                                running <- false
                                                do! trySendJsonLocked {| ``type`` = "generation.canceled"; id = session.Id; requestId = requestId |}
                                            | ex ->
                                                running <- false
                                                do! trySendJsonLocked {| ``type`` = "error"; message = ex.Message; bundle = runner.Status |}
                                with ex ->
                                    do! sendJsonLocked {| ``type`` = "error"; message = ex.Message; bundle = runner.Status |}
                            | _ ->
                                do! sendJsonLocked {| ``type`` = "error"; message = $"Unknown WebSocket event '{eventType}'." |}
                        | _ ->
                            do! sendJsonLocked {| ``type`` = "error"; message = $"Unsupported WebSocket message type {messageType}." |}

                    if socket.State = WebSocketState.Open || socket.State = WebSocketState.CloseReceived then
                        do! socket.CloseAsync(WebSocketCloseStatus.NormalClosure, "done", CancellationToken.None)
        }

    let private serveSessionFile (store: S2sSessionStore) (fileName: string) (contentType: string) (ctx: HttpContext) =
        task {
            let sessionId = routeSessionId ctx
            match store.TryGet(sessionId) with
            | None -> do! writeJson ctx 404 (error 404 "S2S session was not found.")
            | Some session ->
                let path = Path.Combine(session.WorkDir, fileName)
                if File.Exists path then
                    ctx.Response.ContentType <- contentType
                    let! bytes = File.ReadAllBytesAsync(path)
                    do! ctx.Response.Body.WriteAsync(bytes)
                else
                    do! writeJson ctx 404 (error 404 $"Session artifact {fileName} was not found.")
        }

    let private serveBackendSessionFile (store: S2sSessionStore) (fileName: string) (contentType: string) (ctx: HttpContext) =
        task {
            let sessionId = routeSessionId ctx
            let backend =
                try normalizeBackend (routeBackend ctx)
                with _ -> ""
            match store.TryGet(sessionId) with
            | None -> do! writeJson ctx 404 (error 404 "S2S session was not found.")
            | Some session when backend = "" || backend = "both" ->
                do! writeJson ctx 400 (error 400 "Invalid backend artifact route.")
            | Some session ->
                let path = Path.Combine(session.WorkDir, backend, fileName)
                if File.Exists path then
                    ctx.Response.ContentType <- contentType
                    let! bytes = File.ReadAllBytesAsync(path)
                    do! ctx.Response.Body.WriteAsync(bytes)
                else
                    do! writeJson ctx 404 (error 404 $"Session artifact {backend}/{fileName} was not found.")
        }

    let run args (required: string -> string list -> string) (optional: string -> string -> string list -> string) =
        let modelDir = required "--model-dir" args |> Path.GetFullPath
        let bundleDir = required "--bundle-dir" args |> Path.GetFullPath
        let workDir = optional "served_runs" "--work-dir" args |> Path.GetFullPath
        let port = optional "5055" "--port" args |> int
        let executionProvider = optional "cuda" "--execution-provider" args
        let memoryMode = optional "python-footprint" "--memory-mode" args
        let ortMemoryProfile = optional "quality-safe" "--ort-memory-profile" args
        let optimizedModelCacheDir =
            let value = optional "" "--optimized-model-cache-dir" args
            if String.IsNullOrWhiteSpace value then None else Some(Path.GetFullPath value)
        let optimizedModelCacheFormat = optional "onnx" "--optimized-model-cache-format" args
        let thinkerActiveFramesArg = optional "0" "--thinker-active-frames" args
        let thinkerActiveFrames = int thinkerActiveFramesArg
        let python = optional ".venv\\Scripts\\python.exe" "--python" args
        let pythonDevice = optional (if executionProvider.Equals("cuda", StringComparison.OrdinalIgnoreCase) then "cuda" else "cpu") "--python-device" args
        let streamDecodeFrames = optional "8" "--stream-decode-frames" args |> int |> max 1
        let streamMinFreeVramMb = optional "1024" "--stream-min-free-vram-mb" args |> int |> max 0
        let codecStallGuardFrames = optional "16" "--codec-stall-guard-frames" args |> int |> max 0
        let maxQueueLength = optional "32" "--max-queue-length" args |> int |> max 0
        let maxPromptAudioSeconds =
            optional "60" "--max-prompt-audio-seconds" args
            |> fun value -> Math.Max(0.1, Double.Parse(value, CultureInfo.InvariantCulture))
        let maxTurnAudioSeconds =
            optional "60" "--max-turn-audio-seconds" args
            |> fun value -> Math.Max(0.1, Double.Parse(value, CultureInfo.InvariantCulture))
        let cudaGpuMemLimitMb =
            let defaultValue =
                if executionProvider.Equals("cuda", StringComparison.OrdinalIgnoreCase) then "15360" else ""
            let value = optional defaultValue "--cuda-gpu-mem-limit-mb" args
            if String.IsNullOrWhiteSpace value then None else Some(int value)

        Directory.CreateDirectory(workDir) |> ignore
        let tuningOptions =
            { MemoryProfile = ortMemoryProfile
              OptimizedModelCacheDir = optimizedModelCacheDir
              OptimizedModelCacheFormat = optimizedModelCacheFormat
              CudaGpuMemLimitMb = cudaGpuMemLimitMb }
        use runner = new ChromaS2sOnnxRunner(modelDir, bundleDir, executionProvider, memoryMode, tuningOptions)
        let processor = ChromaNativeProcessor(modelDir, thinkerActiveFrames)
        let store = S2sSessionStore(workDir)
        let workQueue = StreamingWorkQueue(maxQueueLength)
        let maxPromptAudioSamples = int (Math.Ceiling(maxPromptAudioSeconds * float processor.PromptSampleRate))
        let maxTurnAudioSamples = int (Math.Ceiling(maxTurnAudioSeconds * float processor.ThinkerSampleRate))

        let builder = WebApplication.CreateBuilder(Array.empty<string>)
        builder.WebHost.UseUrls($"http://localhost:{port}") |> ignore
        let app = builder.Build()
        app.UseWebSockets() |> ignore

        app.MapGet("/", RequestDelegate(fun ctx -> task { do! writeText ctx "text/html; charset=utf-8" indexHtml })) |> ignore
        app.MapGet("/assets/southern_belle.mp3", RequestDelegate(fun ctx -> serveAsset "southern_belle.mp3" "audio/mpeg" ctx)) |> ignore
        app.MapGet("/assets/southern_belle_prompt.txt", RequestDelegate(fun ctx -> serveAsset "southern_belle_prompt.txt" "text/plain; charset=utf-8" ctx)) |> ignore
        app.MapGet(
            "/api/status",
            RequestDelegate(fun ctx ->
                task {
                    let status = runner.Status
                    let payload =
                        {| ready = status.Ready
                           serviceName = "ChromaS2SONNX"
                           mode = "s2s_greedy_streaming"
                           pythonInRequestPath = false
                           pythonBackendAvailable = File.Exists python
                           python = python
                           pythonDevice = pythonDevice
                           modelDir = modelDir
                           bundleDir = bundleDir
                           executionProvider = status.ExecutionProvider
                           memoryMode = runner.MemoryMode
                           ortMemoryProfile = runner.OrtMemoryProfile
                           cudaGpuMemLimitMb = runner.CudaGpuMemLimitMb |> Option.toNullable
                           optimizedModelCacheEnabled = runner.OptimizedModelCacheEnabled
                           optimizedModelCacheDir = runner.OptimizedModelCacheDir
                           optimizedModelCacheFormat = runner.OptimizedModelCacheFormat
                           memory = RuntimeMemory.current()
                           loadedOrtSessions = runner.LoadedSessionNames
                           warmOrtSessions = runner.WarmSessionNames
                           activePagedOrtSessions = runner.ActivePagedSessionNames
                           queueLength = workQueue.QueueLength
                           runningRequestId = nullableString workQueue.RunningId
                           maxQueueLength = workQueue.MaxQueueLength
                           streamDecodeFrames = streamDecodeFrames
                           streamMinFreeVramMb = streamMinFreeVramMb
                           codecStallGuardFrames = codecStallGuardFrames
                           maxPromptAudioSeconds = maxPromptAudioSeconds
                           maxTurnAudioSeconds = maxTurnAudioSeconds
                           globalGpuMemory = RuntimeMemory.tryGlobalGpuMemory() |> Option.toObj
                           peakPrivateGb = runner.PeakPrivateGb
                           peakWorkingSetGb = runner.PeakWorkingSetGb
                           mappedSafetensorShards = runner.MappedShardCount
                           initializerCount = runner.InitializerCount
                           uniqueInitializerSources = runner.UniqueSourceTensorCount
                           uniqueOrtValues = runner.UniqueOrtValueCount
                           sharedPrepackedWeights = runner.SharedPrepackedWeights
                           message = status.Message
                           missingGraphs = status.MissingGraphs
                           availableGraphs = status.AvailableGraphs
                           promptSampleRate = processor.PromptSampleRate
                           thinkerSampleRate = processor.ThinkerSampleRate
                           thinkerFeatureMode = processor.ThinkerFeatureMode
                           thinkerConfiguredActiveFrames = processor.ConfiguredThinkerActiveFrames
                           thinkerTraceFeatureFrames = processor.ThinkerTraceFeatureFrames
                           thinkerTraceSamples = processor.ThinkerTraceSamples |}
                    do! writeJson ctx 200 payload
                })
        ) |> ignore
        app.MapPost("/api/s2s/sessions", RequestDelegate(fun ctx -> createSession processor maxPromptAudioSamples store ctx)) |> ignore
        app.MapGet("/ws/s2s/{id}", RequestDelegate(fun ctx -> handleSocket processor runner modelDir python pythonDevice thinkerActiveFramesArg streamDecodeFrames streamMinFreeVramMb codecStallGuardFrames maxTurnAudioSamples workQueue store ctx)) |> ignore
        app.MapGet("/api/s2s/sessions/{id}/details.json", RequestDelegate(fun ctx -> serveSessionFile store "details.json" "application/json; charset=utf-8" ctx)) |> ignore
        app.MapGet("/api/s2s/sessions/{id}/audio.wav", RequestDelegate(fun ctx -> serveSessionFile store "audio.wav" "audio/wav" ctx)) |> ignore
        app.MapGet("/api/s2s/sessions/{id}/{backend}/details.json", RequestDelegate(fun ctx -> serveBackendSessionFile store "details.json" "application/json; charset=utf-8" ctx)) |> ignore
        app.MapGet("/api/s2s/sessions/{id}/{backend}/audio.wav", RequestDelegate(fun ctx -> serveBackendSessionFile store "audio.wav" "audio/wav" ctx)) |> ignore

        printfn "ChromaS2SONNX service listening on http://localhost:%d" port
        printfn "Python in request path: selectable"
        printfn "Python backend: %s (%s)" python pythonDevice
        printfn "Memory mode: %s" runner.MemoryMode
        printfn "ORT memory profile: %s" runner.OrtMemoryProfile
        match runner.CudaGpuMemLimitMb with
        | Some limit -> printfn "CUDA GPU memory limit: %d MiB" limit
        | None -> ()
        printfn "Stream decode frames: %d" streamDecodeFrames
        printfn "Stream min free VRAM: %d MiB" streamMinFreeVramMb
        printfn "Codec stall guard frames: %d" codecStallGuardFrames
        printfn "Max queue length: %d" workQueue.MaxQueueLength
        printfn "Thinker features: %s" processor.ThinkerFeatureMode
        if runner.OptimizedModelCacheEnabled then
            printfn "Optimized model cache: %s (%s)" runner.OptimizedModelCacheDir runner.OptimizedModelCacheFormat
        printfn "S2S bundle status: %s" runner.Status.Message
        app.Run()
        0

