namespace ChromaOnnx.Service

open System
open System.IO
open System.Net.WebSockets
open System.Text
open System.Text.Json
open System.Text.Json.Nodes
open System.Threading
open System.Threading.Tasks
open Microsoft.AspNetCore.Builder
open Microsoft.AspNetCore.Http
open Microsoft.Extensions.Configuration
open ChromaOnnx.SpeechToSpeech

module S2sWebApp =
    let private jsonOptions =
        JsonSerializerOptions(PropertyNamingPolicy = JsonNamingPolicy.CamelCase, WriteIndented = true)

    let bindOptions (configuration: IConfiguration) =
        let options = S2sRuntimeOptions()
        configuration.GetSection("ChromaOnnx:S2s").Bind(options)
        options

    let bindGemmaOptions (configuration: IConfiguration) =
        let options = GemmaRuntimeOptions()
        configuration.GetSection("ChromaOnnx:Gemma").Bind(options)
        options

    let private indexHtml =
        """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ChromaS2SONNX</title>
  <style>
    :root { color-scheme: light; --ink: #17212b; --muted: #5d6875; --line: #d7dee6; --panel: #f5f7f9; --accent: #0f766e; --danger: #b42318; }
    * { box-sizing: border-box; }
    body { margin: 0; font: 15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--ink); background: #fff; }
    header { min-height: 58px; padding: 14px 22px; border-bottom: 1px solid var(--line); display: flex; align-items: center; justify-content: space-between; gap: 16px; }
    main { min-height: calc(100vh - 58px); display: grid; grid-template-columns: minmax(320px, 430px) 1fr; }
    form { padding: 18px 22px; background: var(--panel); border-right: 1px solid var(--line); display: grid; gap: 13px; align-content: start; }
    label { display: grid; gap: 6px; font-weight: 650; }
    input, textarea, button { font: inherit; }
    textarea, input[type=file], input[type=number], select { width: 100%; border: 1px solid #cbd4dd; border-radius: 6px; padding: 9px 10px; background: white; }
    textarea { min-height: 82px; resize: vertical; }
    button { border: 0; border-radius: 6px; padding: 10px 13px; background: var(--accent); color: white; font-weight: 750; cursor: pointer; }
    button:disabled { opacity: .6; cursor: wait; }
    button.secondary { background: #344054; }
    button.danger { background: var(--danger); }
    .actions { display: grid; grid-template-columns: 1fr auto auto; gap: 10px; }
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
      <label>Mode<select id="conversationMode">
        <option value="s2s" selected>Chroma only</option>
        <option value="agent">Gemma + Chroma</option>
      </select></label>
      <label>System prompt<textarea id="systemPrompt">You are Chroma, an advanced virtual human created by the FlashLabs. You possess the ability to understand auditory inputs and generate both text and speech.</textarea></label>
      <label>Reference text<textarea id="promptText" required>Why don't skeletons fight each other. They don't have the guts.</textarea></label>
      <label>Reference audio<input id="promptPcm" type="file" accept="audio/*,.f32"></label>
      <label>Turn source<select id="inputMode">
        <option value="file" selected>Audio file</option>
        <option value="mic">Microphone</option>
      </select></label>
      <label id="turnFileLabel">Turn audio<input id="turnPcm" type="file" accept="audio/*,.f32"></label>
      <label id="micSecondsLabel" hidden>Mic seconds<input id="micSeconds" type="number" min="1" max="60" value="6"></label>
      <label>Max response seconds<input id="maxResponseSeconds" type="number" min="1" max="72" step="0.5" value="36"></label>
      <div class="actions">
        <button id="sendButton" type="submit">Send</button>
        <button id="newSessionButton" class="secondary" type="button">New</button>
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
    const newSessionButton = document.getElementById('newSessionButton');
    const cancelButton = document.getElementById('cancelButton');
    const message = document.getElementById('message');
    const details = document.getElementById('details');
    const audioResults = document.getElementById('audioResults');
    const queueMetric = document.getElementById('queueMetric');
    const frameMetric = document.getElementById('frameMetric');
    const streamMetric = document.getElementById('streamMetric');
    const inputMode = document.getElementById('inputMode');
    const conversationMode = document.getElementById('conversationMode');
    const turnFileLabel = document.getElementById('turnFileLabel');
    const micSecondsLabel = document.getElementById('micSecondsLabel');
    let audioContext = null;
    let playbackContext = null;
    let playbackCursor = 0;
    let currentSocket = null;
    let currentSession = null;
    let socketReadyResolve = null;
    let socketReadyReject = null;
    let socketReadyPromise = null;
    let activeMicStop = null;
    let pendingChunk = null;
    let streamedSamples = 0;
    const defaultPromptTextUrl = '/assets/southern_belle_prompt.txt';
    const defaultPromptAudioUrl = '/assets/southern_belle.mp3';

    function setBusy(isBusy) {
      button.disabled = isBusy;
      cancelButton.disabled = !isBusy;
      button.textContent = isBusy ? 'Running' : 'Send';
    }

    function cacheBust(url) {
      if (!url) return '';
      return `${url}${url.includes('?') ? '&' : '?'}v=${Date.now()}`;
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
        for (let i = 0; i < frames; i++) mono[i] += data[i] / channels;
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

    async function readAudioBytesAsF32(bytes, fileName, targetRate) {
      if (/\.f32$/i.test(fileName)) return new Uint8Array(bytes);
      audioContext = audioContext || new (window.AudioContext || window.webkitAudioContext)();
      if (audioContext.state === 'suspended') await audioContext.resume();
      const audioBuffer = await audioContext.decodeAudioData(bytes.slice(0));
      const mono = mixToMono(audioBuffer);
      return asByteView(resampleLinear(mono, audioBuffer.sampleRate, targetRate));
    }

    async function readAudioAsF32(file, targetRate) {
      return readAudioBytesAsF32(await file.arrayBuffer(), file.name, targetRate);
    }

    async function readReferenceAudioAsF32(targetRate) {
      const input = document.getElementById('promptPcm');
      if (input.files.length) return readAudioAsF32(input.files[0], targetRate);
      const response = await fetch(defaultPromptAudioUrl);
      if (!response.ok) throw new Error('Default reference audio was not found.');
      return readAudioBytesAsF32(await response.arrayBuffer(), 'southern_belle.mp3', targetRate);
    }

    async function sendBytesInChunks(ws, bytes, chunkBytes = 32768) {
      for (let offset = 0; offset < bytes.length; offset += chunkBytes) {
        if (ws.readyState !== WebSocket.OPEN) throw new Error('WebSocket closed while sending audio.');
        ws.send(bytes.slice(offset, Math.min(offset + chunkBytes, bytes.length)));
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }

    function updateMode() {
      const mic = inputMode.value === 'mic';
      turnFileLabel.hidden = mic;
      micSecondsLabel.hidden = !mic;
    }

    async function streamMicrophone(ws, seconds) {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Microphone capture is not available in this browser.');
      }

      const media = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
        video: false
      });
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

    async function playFloat32Chunk(samples, sampleRate) {
      playbackContext = playbackContext || new (window.AudioContext || window.webkitAudioContext)();
      if (playbackContext.state === 'suspended') await playbackContext.resume();
      const startAt = Math.max(playbackContext.currentTime + 0.03, playbackCursor || 0);
      const buffer = playbackContext.createBuffer(1, samples.length, sampleRate);
      buffer.copyToChannel(samples, 0);
      const source = playbackContext.createBufferSource();
      source.buffer = buffer;
      source.connect(playbackContext.destination);
      source.start(startAt);
      playbackCursor = startAt + buffer.duration;
    }

    function showResult(result) {
      const block = document.createElement('div');
      block.className = 'audioResult';
      const title = document.createElement('strong');
      title.textContent = `Turn ${result.turnIndex || '?'}`;
      const audio = document.createElement('audio');
      audio.controls = true;
      audio.src = cacheBust(result.audioUrl);
      block.append(title, audio);
      audioResults.append(block);
    }

    function updateFromEvent(payload) {
      details.textContent = JSON.stringify(payload, null, 2);
      switch (payload.type) {
        case 'queue.enqueued':
        case 'queue.updated':
          queueMetric.textContent = String(payload.position || payload.queueLength || 0);
          message.textContent = payload.type;
          break;
        case 'queue.started':
        case 'generation.started':
          message.textContent = payload.type;
          break;
        case 'generation.frame':
          frameMetric.textContent = String(payload.frameIndex + 1);
          break;
        case 'audio.deferred':
          message.textContent = 'Audio decode deferred';
          break;
        case 'agent.transcription':
          message.textContent = `Transcript: ${payload.transcript || ''}`;
          break;
        case 'agent.tool_call':
          message.textContent = `Tool: ${payload.name || ''}`;
          break;
        case 'agent.tool_result':
          message.textContent = payload.success ? `Tool result: ${payload.name || ''}` : `Tool failed: ${payload.name || ''}`;
          break;
        case 'agent.final_text':
          message.textContent = payload.text || 'Final text ready';
          break;
        case 'agent.vocalization.started':
          message.textContent = 'Vocalizing with Chroma';
          break;
        case 'agent.done':
          message.textContent = `Agent done: turn ${payload.turnIndex || ''}`;
          setBusy(false);
          break;
        case 'generation.done':
          message.textContent = `Done: turn ${payload.turnIndex || ''}`;
          showResult(payload);
          if (conversationMode.value === 's2s') currentSession = null;
          if (conversationMode.value === 's2s') setBusy(false);
          break;
        case 'error':
          message.textContent = payload.message || 'Error';
          message.classList.add('bad');
          setBusy(false);
          break;
      }
    }

    async function loadStatus() {
      try {
        const response = await fetch('/api/status');
        const status = await response.json();
        const agent = status.agent ? ` | agent ${status.agent.ready ? 'ready' : 'not ready'}` : '';
        runtime.textContent = status.ready ? `${status.executionProvider} ${status.memoryMode} ${status.generationMode}/${status.samplingAlgorithm}${agent}` : `${status.message}${agent}`;
        runtime.className = status.ready ? '' : 'bad';
      } catch (error) {
        runtime.textContent = error.message;
        runtime.className = 'bad';
      }
    }

    function closeConversationSocket(reason = 'new conversation') {
      if (activeMicStop) activeMicStop();
      if (currentSocket && currentSocket.readyState === WebSocket.OPEN) currentSocket.close(1000, reason);
      currentSocket = null;
      socketReadyPromise = null;
      socketReadyResolve = null;
      socketReadyReject = null;
    }

    async function createSession(promptBytes, maxSeconds) {
      const formData = new FormData();
      formData.set('systemPrompt', document.getElementById('systemPrompt').value);
      formData.set('promptText', document.getElementById('promptText').value);
      formData.set('backend', 'fsharp_onnx');
      formData.set('maxNewFrames', String(Math.max(1, Math.ceil(maxSeconds / 0.08))));
      formData.set('promptPcm24k', new Blob([promptBytes], { type: 'application/octet-stream' }), 'prompt.f32');
      const endpoint = conversationMode.value === 'agent' ? '/api/agent/sessions' : '/api/s2s/sessions';
      const sessionResponse = await fetch(endpoint, { method: 'POST', body: formData });
      const session = await sessionResponse.json();
      if (!sessionResponse.ok) throw new Error(session.error || 'Failed to create session.');
      currentSession = session;
      return session;
    }

    async function ensureSocket(session) {
      if (currentSocket && currentSocket.readyState === WebSocket.OPEN && socketReadyPromise) {
        await socketReadyPromise;
        return currentSocket;
      }

      const ws = new WebSocket(`${location.origin.replace('http', 'ws')}${session.websocketUrl}`);
      currentSocket = ws;
      ws.binaryType = 'arraybuffer';
      socketReadyPromise = new Promise((resolve, reject) => {
        socketReadyResolve = resolve;
        socketReadyReject = reject;
      });
      ws.onmessage = async event => {
        if (typeof event.data === 'string') {
          const payload = JSON.parse(event.data);
          if (payload.type === 'session.ready' && socketReadyResolve) {
            socketReadyResolve(ws);
            socketReadyResolve = null;
            socketReadyReject = null;
          }
          if (payload.type === 'audio.chunk') pendingChunk = payload;
          updateFromEvent(payload);
          return;
        }
        if (pendingChunk) {
          const chunk = pendingChunk;
          pendingChunk = null;
          const samples = new Float32Array(event.data);
          streamedSamples += samples.length;
          streamMetric.textContent = `${(streamedSamples / (chunk.sampleRate || 24000)).toFixed(2)} s`;
          await playFloat32Chunk(samples, chunk.sampleRate || 24000);
        }
      };
      ws.onerror = () => {
        message.textContent = 'WebSocket error while generating.';
        message.classList.add('bad');
        setBusy(false);
        if (socketReadyReject) socketReadyReject(new Error('WebSocket error while opening session.'));
      };
      ws.onclose = () => {
        currentSocket = null;
        socketReadyPromise = null;
        if (activeMicStop) activeMicStop();
        if (button.disabled) setBusy(false);
      };
      await socketReadyPromise;
      return ws;
    }

    form.addEventListener('submit', async event => {
      event.preventDefault();
      setBusy(true);
      message.classList.remove('bad');
      message.textContent = 'Preparing audio';
      details.textContent = '{}';
      streamedSamples = 0;
      playbackCursor = 0;

      try {
        const mode = inputMode.value;
        const turnFile = document.getElementById('turnPcm').files[0];
        if (mode === 'file' && !turnFile) throw new Error('Turn audio is required in file mode.');
        const turnBytes = mode === 'file' ? await readAudioAsF32(turnFile, 16000) : null;
        const maxSeconds = Math.max(1, Math.min(72, Number(document.getElementById('maxResponseSeconds').value) || 36));
        const session = currentSession || await createSession(await readReferenceAudioAsF32(24000), maxSeconds);

        message.textContent = mode === 'mic' ? 'Opening microphone' : 'Opening WebSocket';
        const ws = await ensureSocket(session);
        ws.send(JSON.stringify({ type: 'turn.start' }));
        if (mode === 'mic') {
          const seconds = Number(document.getElementById('micSeconds').value) || 6;
          message.textContent = `Recording microphone for ${Math.max(1, seconds)} s`;
          await streamMicrophone(ws, seconds);
        } else {
          await sendBytesInChunks(ws, turnBytes);
          ws.send(JSON.stringify({ type: 'turn.end' }));
        }
      } catch (error) {
        message.textContent = error.message;
        message.classList.add('bad');
        setBusy(false);
      }
    });

    cancelButton.addEventListener('click', () => {
      if (activeMicStop) activeMicStop();
      if (currentSocket && currentSocket.readyState === WebSocket.OPEN) {
        currentSocket.send(JSON.stringify({ type: 'turn.cancel' }));
      }
      setBusy(false);
    });

    newSessionButton.addEventListener('click', () => {
      closeConversationSocket();
      currentSession = null;
      audioResults.innerHTML = '';
      details.textContent = '{}';
      message.classList.remove('bad');
      message.textContent = 'Idle';
    });

    inputMode.addEventListener('change', updateMode);
    conversationMode.addEventListener('change', () => {
      closeConversationSocket('mode changed');
      currentSession = null;
      audioResults.innerHTML = '';
      details.textContent = '{}';
      message.classList.remove('bad');
      message.textContent = 'Idle';
    });
    updateMode();
    loadStatus();
    fetch(defaultPromptTextUrl, { cache: 'no-store' })
      .then(response => response.ok ? response.text() : '')
      .then(text => {
        if (text.trim()) document.getElementById('promptText').value = text.trim();
      })
      .catch(() => {});
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

    let private nullableString (value: string option) : string | null =
        match value with
        | Some text -> text
        | None -> null

    let private optionToObj value : obj | null =
        match value with
        | Some item -> box item
        | None -> null

    let private routeValue (ctx: HttpContext) name =
        match ctx.Request.RouteValues.TryGetValue(name) with
        | true, value when value <> null -> string value
        | _ -> ""

    let private safeId (value: string) =
        not (String.IsNullOrWhiteSpace value)
        && value |> Seq.forall (fun ch -> Char.IsLetterOrDigit(ch) || ch = '_' || ch = '-')

    let private readFormFileBytes (form: IFormCollection) (name: string) =
        match form.Files.GetFile(name) with
        | null -> invalidArg name $"Missing multipart file {name}."
        | file when file.Length = 0L -> invalidArg name $"Multipart file {name} is empty."
        | file ->
            use stream = new MemoryStream()
            file.CopyTo(stream)
            stream.ToArray()

    let private float32PcmFromBytes (bytes: byte array) =
        if bytes.Length >= 12
           && bytes[0] = byte 'R'
           && bytes[1] = byte 'I'
           && bytes[2] = byte 'F'
           && bytes[3] = byte 'F'
           && bytes[8] = byte 'W'
           && bytes[9] = byte 'A'
           && bytes[10] = byte 'V'
           && bytes[11] = byte 'E' then
            invalidArg "bytes" "Expected raw little-endian Float32 PCM, but received a WAV container."
        ChromaOnnx.AudioChunk.float32FromLittleEndianBytes bytes

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

    let private sendJson (socket: WebSocket) (payload: 'T) =
        task {
            let bytes = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(payload, jsonOptions))
            do! socket.SendAsync(ArraySegment<byte>(bytes), WebSocketMessageType.Text, true, CancellationToken.None)
        }

    let private sendBinary (socket: WebSocket) (payload: byte array) =
        task {
            do! socket.SendAsync(ArraySegment<byte>(payload), WebSocketMessageType.Binary, true, CancellationToken.None)
        }

    let private statusPayload (runtime: IS2sRuntime) =
        let status = runtime.Status()
        {| ready = status.Ready
           serviceName = status.ServiceName
           mode = status.Mode
           pythonInRequestPath = status.PythonInRequestPath
           modelDir = status.ModelDir
           bundleDir = status.BundleDir
           executionProvider = status.ExecutionProvider
           memoryMode = status.MemoryMode
           ortMemoryProfile = status.OrtMemoryProfile
           cudaGpuMemLimitMb = status.CudaGpuMemLimitMb
           optimizedModelCacheEnabled = status.OptimizedModelCacheEnabled
           optimizedModelCacheDir = status.OptimizedModelCacheDir
           optimizedModelCacheFormat = status.OptimizedModelCacheFormat
           memory = status.Memory
           loadedOrtSessions = status.LoadedOrtSessions
           warmOrtSessions = status.WarmOrtSessions
           activePagedOrtSessions = status.ActivePagedOrtSessions
           queueLength = status.QueueLength
           runningRequestId = nullableString status.RunningRequestId
           maxQueueLength = status.MaxQueueLength
           streamDecodeFrames = status.StreamDecodeFrames
           streamMinFreeVramMb = status.StreamMinFreeVramMb
           codecStallGuardFrames = status.CodecStallGuardFrames
           generationMode = status.GenerationMode
           samplingAlgorithm = status.SamplingAlgorithm
           samplingTemperature = status.SamplingTemperature
           samplingTopP = status.SamplingTopP
           samplingTopK = status.SamplingTopK
           maxNewFrames = status.MaxNewFrames
           maxPromptAudioSeconds = status.MaxPromptAudioSeconds
           maxTurnAudioSeconds = status.MaxTurnAudioSeconds
           globalGpuMemory = optionToObj status.GlobalGpuMemory
           peakPrivateGb = status.PeakPrivateGb
           peakWorkingSetGb = status.PeakWorkingSetGb
           mappedSafetensorShards = status.MappedSafetensorShards
           initializerCount = status.InitializerCount
           uniqueInitializerSources = status.UniqueInitializerSources
           uniqueOrtValues = status.UniqueOrtValues
           sharedPrepackedWeights = status.SharedPrepackedWeights
           message = status.Message
           missingGraphs = status.MissingGraphs
           availableGraphs = status.AvailableGraphs
           promptSampleRate = status.PromptSampleRate
           thinkerSampleRate = status.ThinkerSampleRate
           bundleGraphMode = status.BundleGraphMode
           bundleThinkerFeatureMode = status.BundleThinkerFeatureMode
           thinkerMaxAudioItems = status.ThinkerMaxAudioItems
           thinkerFeatureMode = status.ThinkerFeatureMode
           thinkerConfiguredActiveFrames = status.ThinkerConfiguredActiveFrames
           thinkerTraceFeatureFrames = status.ThinkerTraceFeatureFrames
           thinkerTraceSamples = status.ThinkerTraceSamples |}

    let private agentStatusPayload (agent: IAgentRuntime) =
        let status = agent.Status()
        {| ready = status.Ready
           serviceName = status.ServiceName
           mode = status.Mode
           modelDir = status.ModelDir
           variant = status.Variant
           executionProvider = status.ExecutionProvider
           maxAudioSeconds = status.MaxAudioSeconds
           asrMaxNewTokens = status.AsrMaxNewTokens
           reasoningMaxNewTokens = status.ReasoningMaxNewTokens
           toolMaxRounds = status.ToolMaxRounds
           maxHistoryTurns = status.MaxHistoryTurns
           chromaReady = status.ChromaReady
           gemma =
            {| ready = status.Gemma.Ready
               modelDir = status.Gemma.ModelDir
               variant = status.Gemma.Variant
               executionProvider = status.Gemma.ExecutionProvider
               missingFiles = status.Gemma.MissingFiles
               loadedSessions = status.Gemma.LoadedSessions
               message = status.Gemma.Message |}
           message = status.Message |}

    let private combinedStatusPayload (runtime: IS2sRuntime) (agent: IAgentRuntime) =
        let node =
            match JsonSerializer.SerializeToNode(statusPayload runtime, jsonOptions) with
            | :? JsonObject as value -> value
            | null -> JsonObject()
            | value ->
                let fallback = JsonObject()
                fallback["s2s"] <- value
                fallback
        node["agent"] <- JsonSerializer.SerializeToNode(agentStatusPayload agent, jsonOptions)
        node :> JsonNode

    let private createSession (runtime: IS2sRuntime) (ctx: HttpContext) =
        task {
            if not ctx.Request.HasFormContentType then
                do! writeJson ctx 400 (error 400 "Expected multipart/form-data.")
            else
                try
                    let! form = ctx.Request.ReadFormAsync()
                    let promptText = form["promptText"].ToString()
                    let systemPrompt = form["systemPrompt"].ToString()
                    let backend = form["backend"].ToString()
                    let maxNewFrames =
                        match Int32.TryParse(form["maxNewFrames"].ToString()) with
                        | true, value -> max 1 value
                        | false, _ -> 450
                    let promptAudioBytes = readFormFileBytes form "promptPcm24k"
                    let maxPromptBytes = runtime.MaxPromptAudioSamples * sizeof<float32>
                    if promptAudioBytes.Length > maxPromptBytes then
                        do!
                            writeJson
                                ctx
                                413
                                (error
                                    413
                                    $"promptPcm24k is too large. The configured maximum is {runtime.MaxPromptAudioSamples} Float32 samples.")
                    else
                        let promptAudio = float32PcmFromBytes promptAudioBytes
                        let session =
                            runtime.CreateSession
                                { PromptText = promptText
                                  SystemPrompt = systemPrompt
                                  Backend = backend
                                  PromptAudio24k = promptAudio
                                  MaxNewFrames = maxNewFrames }
                        do! writeJson ctx 200 session
                with
                | :? ArgumentException as ex -> do! writeJson ctx 400 (error 400 ex.Message)
                | ex -> do! writeJson ctx 500 (error 500 ex.Message)
        }

    let private createAgentSession (runtime: IS2sRuntime) (agent: IAgentRuntime) (ctx: HttpContext) =
        task {
            if not ctx.Request.HasFormContentType then
                do! writeJson ctx 400 (error 400 "Expected multipart/form-data.")
            else
                try
                    let! form = ctx.Request.ReadFormAsync()
                    let promptText = form["promptText"].ToString()
                    let systemPrompt = form["systemPrompt"].ToString()
                    let maxNewFrames =
                        match Int32.TryParse(form["maxNewFrames"].ToString()) with
                        | true, value -> max 1 value
                        | false, _ -> 450
                    let promptAudioBytes = readFormFileBytes form "promptPcm24k"
                    let maxPromptBytes = runtime.MaxPromptAudioSamples * sizeof<float32>
                    if promptAudioBytes.Length > maxPromptBytes then
                        do!
                            writeJson
                                ctx
                                413
                                (error
                                    413
                                    $"promptPcm24k is too large. The configured maximum is {runtime.MaxPromptAudioSamples} Float32 samples.")
                    else
                        let promptAudio = float32PcmFromBytes promptAudioBytes
                        let session =
                            agent.CreateSession
                                { PromptText = promptText
                                  SystemPrompt = systemPrompt
                                  PromptAudio24k = promptAudio
                                  MaxNewFrames = maxNewFrames }
                        do! writeJson ctx 200 session
                with
                | :? ArgumentException as ex -> do! writeJson ctx 400 (error 400 ex.Message)
                | ex -> do! writeJson ctx 500 (error 500 ex.Message)
        }

    let private handleSocket (runtime: IS2sRuntime) (ctx: HttpContext) =
        task {
            let sessionId = routeValue ctx "id"
            if not ctx.WebSockets.IsWebSocketRequest then
                do! writeJson ctx 400 (error 400 "Expected WebSocket request.")
            elif not (safeId sessionId) then
                do! writeJson ctx 400 (error 400 "Invalid session id.")
            else
                match runtime.TryGetSession sessionId with
                | None -> do! writeJson ctx 404 (error 404 "S2S session was not found.")
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
                    let sendBinaryLocked payload =
                        task {
                            do! sendLock.WaitAsync()
                            try
                                if socket.State = WebSocketState.Open then
                                    do! sendBinary socket payload
                            finally
                                sendLock.Release() |> ignore
                        }
                    let trySendJsonLocked payload =
                        task {
                            try do! sendJsonLocked payload with _ -> ()
                        }
                    let runtimeStatus = runtime.Status()
                    do!
                        sendJsonLocked
                            {| ``type`` = "session.ready"
                               id = session.Id
                               maxNewFrames = session.MaxNewFrames
                               runtimeMaxNewFrames = runtimeStatus.MaxNewFrames
                               streamDecodeFrames = runtimeStatus.StreamDecodeFrames
                               streamMinFreeVramMb = runtimeStatus.StreamMinFreeVramMb
                               codecStallGuardFrames = runtimeStatus.CodecStallGuardFrames
                               generationMode = runtimeStatus.GenerationMode
                               samplingAlgorithm = runtimeStatus.SamplingAlgorithm
                               samplingTemperature = runtimeStatus.SamplingTemperature
                               samplingTopP = runtimeStatus.SamplingTopP
                               samplingTopK = runtimeStatus.SamplingTopK
                               bundleGraphMode = runtimeStatus.BundleGraphMode
                               bundleThinkerFeatureMode = runtimeStatus.BundleThinkerFeatureMode
                               thinkerMaxAudioItems = runtimeStatus.ThinkerMaxAudioItems
                               cudaGpuMemLimitMb = runtimeStatus.CudaGpuMemLimitMb
                               queueLength = runtimeStatus.QueueLength
                               maxQueueLength = runtimeStatus.MaxQueueLength |}

                    use turnAudio = new MemoryStream()
                    let mutable running = true
                    while running && socket.State = WebSocketState.Open do
                        let! messageType, payload = receiveMessage socket
                        match messageType with
                        | WebSocketMessageType.Close ->
                            socketCancellation.Cancel()
                            running <- false
                        | WebSocketMessageType.Binary ->
                            let maxTurnBytes = int64 runtime.MaxTurnAudioSamples * int64 sizeof<float32>
                            if turnAudio.Length + int64 payload.Length > maxTurnBytes then
                                socketCancellation.Cancel()
                                running <- false
                                do!
                                    sendJsonLocked
                                        {| ``type`` = "error"
                                           message = $"User turn audio is too large. The configured maximum is {runtime.MaxTurnAudioSamples} Float32 samples." |}
                            else
                                turnAudio.Write(payload, 0, payload.Length)
                                do! sendJsonLocked {| ``type`` = "turn.chunk"; bytes = payload.Length; totalBytes = turnAudio.Length |}
                        | WebSocketMessageType.Text ->
                            try
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
                                    socketCancellation.Cancel()
                                    running <- false
                                    do! trySendJsonLocked {| ``type`` = "generation.canceled"; id = session.Id |}
                                | "turn.end" ->
                                    let turnAudioBytes = turnAudio.ToArray()
                                    if turnAudioBytes.Length = 0 then
                                        do! sendJsonLocked {| ``type`` = "error"; message = "User turn audio is required before turn.end." |}
                                    else
                                        let userAudio = float32PcmFromBytes turnAudioBytes
                                        let emit event =
                                            task {
                                                match event with
                                                | QueueEnqueued(requestId, snapshot) ->
                                                    do!
                                                        sendJsonLocked
                                                            {| ``type`` = "queue.enqueued"
                                                               id = session.Id
                                                               requestId = requestId
                                                               position = snapshot.Position
                                                               queueLength = snapshot.QueueLength
                                                               runningId = nullableString snapshot.RunningId
                                                               maxQueueLength = runtime.MaxQueueLength |}
                                                | QueueUpdated snapshot ->
                                                    do!
                                                        sendJsonLocked
                                                            {| ``type`` = "queue.updated"
                                                               id = session.Id
                                                               requestId = snapshot.Id
                                                               position = snapshot.Position
                                                               queueLength = snapshot.QueueLength
                                                               runningId = nullableString snapshot.RunningId
                                                               isRunning = snapshot.IsRunning |}
                                                | QueueStarted(requestId, queueLength) ->
                                                    do!
                                                        sendJsonLocked
                                                            {| ``type`` = "queue.started"
                                                               id = session.Id
                                                               requestId = requestId
                                                               queueLength = queueLength |}
                                                | GenerationStarted(requestId, maxNewFrames, streamDecodeFrames, streamMinFreeVramMb, codecStallGuardFrames, context) ->
                                                    do!
                                                        sendJsonLocked
                                                            {| ``type`` = "generation.started"
                                                               id = session.Id
                                                               requestId = requestId
                                                               turnIndex = context.TurnIndex
                                                               backend = session.Backend
                                                               backends = [| "fsharp_onnx" |]
                                                               maxNewFrames = maxNewFrames
                                                               streamDecodeFrames = streamDecodeFrames
                                                               streamMinFreeVramMb = streamMinFreeVramMb
                                                               codecStallGuardFrames = codecStallGuardFrames |}
                                                | GenerationFrame(requestId, frame) ->
                                                    do!
                                                        sendJsonLocked
                                                            {| ``type`` = "generation.frame"
                                                               id = session.Id
                                                               requestId = requestId
                                                               frameIndex = frame.FrameIndex
                                                               stepKind = frame.StepKind
                                                               isEos = frame.IsEos
                                                               codes = frame.Codes |}
                                                | AudioChunk(requestId, chunk) ->
                                                    let bytes = ChromaOnnx.AudioChunk.float32ToLittleEndianBytes chunk.Samples
                                                    do!
                                                        sendJsonLocked
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
                                                    do! sendBinaryLocked bytes
                                                | AudioDeferred(requestId, deferred) ->
                                                    do!
                                                        sendJsonLocked
                                                            {| ``type`` = "audio.deferred"
                                                               id = session.Id
                                                               requestId = requestId
                                                               frameCount = deferred.FrameCount
                                                               freeVramMb = deferred.FreeVramMb
                                                               usedVramMb = deferred.UsedVramMb
                                                               totalVramMb = deferred.TotalVramMb
                                                               minFreeVramMb = deferred.MinFreeVramMb
                                                               message = deferred.Message |}
                                                | GenerationDone result ->
                                                    do!
                                                        sendJsonLocked
                                                            {| ``type`` = "generation.done"
                                                               id = result.Id
                                                               requestId = result.RequestId
                                                               turnIndex = result.TurnIndex
                                                               backend = result.Backend
                                                               audioUrl = result.AudioUrl
                                                               detailsUrl = result.DetailsUrl
                                                               results = result.Results |> Array.map (fun item -> item.Details) |}
                                                | GenerationCanceled(id, requestId) ->
                                                    do!
                                                        sendJsonLocked
                                                            {| ``type`` = "generation.canceled"
                                                               id = id
                                                               requestId = nullableString requestId |}
                                            }
                                            :> Task
                                        try
                                            let! _ =
                                                runtime.RunTurnAsync(
                                                    { SessionId = session.Id
                                                      UserAudio16k = userAudio
                                                      RequestId = None },
                                                    emit,
                                                    socketCancellation.Token
                                            )
                                            turnAudio.SetLength(0L)
                                            running <- false
                                        with
                                        | :? OperationCanceledException ->
                                            running <- false
                                            do! trySendJsonLocked {| ``type`` = "generation.canceled"; id = session.Id |}
                                        | ex ->
                                            running <- false
                                            do! trySendJsonLocked {| ``type`` = "error"; message = ex.Message; bundle = statusPayload runtime |}
                                | _ ->
                                    do! sendJsonLocked {| ``type`` = "error"; message = $"Unknown WebSocket event '{eventType}'." |}
                            with ex ->
                                do! sendJsonLocked {| ``type`` = "error"; message = ex.Message; bundle = statusPayload runtime |}
                        | _ ->
                            do! sendJsonLocked {| ``type`` = "error"; message = $"Unsupported WebSocket message type {messageType}." |}

                    if socket.State = WebSocketState.Open || socket.State = WebSocketState.CloseReceived then
                        do! socket.CloseAsync(WebSocketCloseStatus.NormalClosure, "done", CancellationToken.None)
        }

    let private handleAgentSocket (runtime: IS2sRuntime) (agent: IAgentRuntime) (ctx: HttpContext) =
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
                    let sendBinaryLocked payload =
                        task {
                            do! sendLock.WaitAsync()
                            try
                                if socket.State = WebSocketState.Open then
                                    do! sendBinary socket payload
                            finally
                                sendLock.Release() |> ignore
                        }
                    let trySendJsonLocked payload =
                        task {
                            try do! sendJsonLocked payload with _ -> ()
                        }

                    let agentStatus = agent.Status()
                    do!
                        sendJsonLocked
                            {| ``type`` = "session.ready"
                               id = session.Id
                               mode = "agent"
                               maxNewFrames = session.MaxNewFrames
                               maxTurnAudioSamples = agent.MaxTurnAudioSamples
                               gemmaReady = agentStatus.Gemma.Ready
                               gemmaMessage = agentStatus.Gemma.Message
                               toolMaxRounds = agentStatus.ToolMaxRounds
                               maxHistoryTurns = agentStatus.MaxHistoryTurns |}

                    use turnAudio = new MemoryStream()
                    let mutable running = true
                    while running && socket.State = WebSocketState.Open do
                        let! messageType, payload = receiveMessage socket
                        match messageType with
                        | WebSocketMessageType.Close ->
                            socketCancellation.Cancel()
                            running <- false
                        | WebSocketMessageType.Binary ->
                            let maxTurnBytes = int64 agent.MaxTurnAudioSamples * int64 sizeof<float32>
                            if turnAudio.Length + int64 payload.Length > maxTurnBytes then
                                socketCancellation.Cancel()
                                running <- false
                                do!
                                    sendJsonLocked
                                        {| ``type`` = "error"
                                           message = $"User turn audio is too large. The configured maximum is {agent.MaxTurnAudioSamples} Float32 samples." |}
                            else
                                turnAudio.Write(payload, 0, payload.Length)
                                do! sendJsonLocked {| ``type`` = "turn.chunk"; bytes = payload.Length; totalBytes = turnAudio.Length |}
                        | WebSocketMessageType.Text ->
                            try
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
                                    socketCancellation.Cancel()
                                    running <- false
                                    do! trySendJsonLocked {| ``type`` = "generation.canceled"; id = session.Id |}
                                | "turn.end" ->
                                    let turnAudioBytes = turnAudio.ToArray()
                                    if turnAudioBytes.Length = 0 then
                                        do! sendJsonLocked {| ``type`` = "error"; message = "User turn audio is required before turn.end." |}
                                    else
                                        let userAudio = float32PcmFromBytes turnAudioBytes
                                        let emit event =
                                            task {
                                                match event with
                                                | AgentTranscription(id, requestId, turnIndex, transcript) ->
                                                    do!
                                                        sendJsonLocked
                                                            {| ``type`` = "agent.transcription"
                                                               id = id
                                                               requestId = requestId
                                                               turnIndex = turnIndex
                                                               transcript = transcript |}
                                                | AgentToolCall(id, requestId, turnIndex, call) ->
                                                    do!
                                                        sendJsonLocked
                                                            {| ``type`` = "agent.tool_call"
                                                               id = id
                                                               requestId = requestId
                                                               turnIndex = turnIndex
                                                               round = call.Round
                                                               name = call.Name
                                                               arguments = call.Arguments
                                                               rawText = call.RawText |}
                                                | AgentToolResult(id, requestId, turnIndex, result) ->
                                                    do!
                                                        sendJsonLocked
                                                            {| ``type`` = "agent.tool_result"
                                                               id = id
                                                               requestId = requestId
                                                               turnIndex = turnIndex
                                                               round = result.Round
                                                               name = result.Name
                                                               success = result.Success
                                                               result = result.Result
                                                               error = nullableString result.Error |}
                                                | AgentFinalText(id, requestId, turnIndex, finalText) ->
                                                    do!
                                                        sendJsonLocked
                                                            {| ``type`` = "agent.final_text"
                                                               id = id
                                                               requestId = requestId
                                                               turnIndex = turnIndex
                                                               text = finalText |}
                                                | AgentVocalizationStarted(id, requestId, turnIndex, chromaSessionId) ->
                                                    do!
                                                        sendJsonLocked
                                                            {| ``type`` = "agent.vocalization.started"
                                                               id = id
                                                               requestId = requestId
                                                               turnIndex = turnIndex
                                                               chromaSessionId = chromaSessionId |}
                                                | AgentChromaEvent chromaEvent ->
                                                    match chromaEvent with
                                                    | QueueEnqueued(requestId, snapshot) ->
                                                        do!
                                                            sendJsonLocked
                                                                {| ``type`` = "queue.enqueued"
                                                                   id = session.Id
                                                                   requestId = requestId
                                                                   position = snapshot.Position
                                                                   queueLength = snapshot.QueueLength
                                                                   runningId = nullableString snapshot.RunningId
                                                                   maxQueueLength = runtime.MaxQueueLength |}
                                                    | QueueUpdated snapshot ->
                                                        do!
                                                            sendJsonLocked
                                                                {| ``type`` = "queue.updated"
                                                                   id = session.Id
                                                                   requestId = snapshot.Id
                                                                   position = snapshot.Position
                                                                   queueLength = snapshot.QueueLength
                                                                   runningId = nullableString snapshot.RunningId
                                                                   isRunning = snapshot.IsRunning |}
                                                    | QueueStarted(requestId, queueLength) ->
                                                        do!
                                                            sendJsonLocked
                                                                {| ``type`` = "queue.started"
                                                                   id = session.Id
                                                                   requestId = requestId
                                                                   queueLength = queueLength |}
                                                    | GenerationStarted(requestId, maxNewFrames, streamDecodeFrames, streamMinFreeVramMb, codecStallGuardFrames, context) ->
                                                        do!
                                                            sendJsonLocked
                                                                {| ``type`` = "generation.started"
                                                                   id = session.Id
                                                                   requestId = requestId
                                                                   turnIndex = context.TurnIndex
                                                                   backend = "fsharp_onnx"
                                                                   backends = [| "fsharp_onnx" |]
                                                                   maxNewFrames = maxNewFrames
                                                                   streamDecodeFrames = streamDecodeFrames
                                                                   streamMinFreeVramMb = streamMinFreeVramMb
                                                                   codecStallGuardFrames = codecStallGuardFrames |}
                                                    | GenerationFrame(requestId, frame) ->
                                                        do!
                                                            sendJsonLocked
                                                                {| ``type`` = "generation.frame"
                                                                   id = session.Id
                                                                   requestId = requestId
                                                                   frameIndex = frame.FrameIndex
                                                                   stepKind = frame.StepKind
                                                                   isEos = frame.IsEos
                                                                   codes = frame.Codes |}
                                                    | AudioChunk(requestId, chunk) ->
                                                        let bytes = ChromaOnnx.AudioChunk.float32ToLittleEndianBytes chunk.Samples
                                                        do!
                                                            sendJsonLocked
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
                                                        do! sendBinaryLocked bytes
                                                    | AudioDeferred(requestId, deferred) ->
                                                        do!
                                                            sendJsonLocked
                                                                {| ``type`` = "audio.deferred"
                                                                   id = session.Id
                                                                   requestId = requestId
                                                                   frameCount = deferred.FrameCount
                                                                   freeVramMb = deferred.FreeVramMb
                                                                   usedVramMb = deferred.UsedVramMb
                                                                   totalVramMb = deferred.TotalVramMb
                                                                   minFreeVramMb = deferred.MinFreeVramMb
                                                                   message = deferred.Message |}
                                                    | GenerationDone result ->
                                                        do!
                                                            sendJsonLocked
                                                                {| ``type`` = "generation.done"
                                                                   id = session.Id
                                                                   requestId = result.RequestId
                                                                   turnIndex = result.TurnIndex
                                                                   backend = result.Backend
                                                                   audioUrl = result.AudioUrl
                                                                   detailsUrl = result.DetailsUrl
                                                                   results = result.Results |> Array.map (fun item -> item.Details) |}
                                                    | GenerationCanceled(id, requestId) ->
                                                        do!
                                                            sendJsonLocked
                                                                {| ``type`` = "generation.canceled"
                                                                   id = id
                                                                   requestId = nullableString requestId |}
                                                | AgentDone result ->
                                                    do!
                                                        sendJsonLocked
                                                            {| ``type`` = "agent.done"
                                                               id = result.Id
                                                               requestId = result.RequestId
                                                               turnIndex = result.TurnIndex
                                                               transcript = result.Transcript
                                                               finalText = result.FinalText
                                                               chromaSessionId = nullableString result.ChromaSessionId
                                                               chromaTurnIndex = optionToObj result.ChromaTurnIndex
                                                               audioUrl = nullableString result.AudioUrl
                                                               detailsUrl = result.DetailsUrl
                                                               toolCalls = result.ToolCalls
                                                               toolResults = result.ToolResults |}
                                                | AgentCanceled(id, requestId) ->
                                                    do!
                                                        sendJsonLocked
                                                            {| ``type`` = "generation.canceled"
                                                               id = id
                                                               requestId = nullableString requestId |}
                                            }
                                            :> Task
                                        try
                                            let! _ =
                                                agent.RunTurnAsync(
                                                    { SessionId = session.Id
                                                      UserAudio16k = userAudio
                                                      RequestId = None },
                                                    emit,
                                                    socketCancellation.Token
                                                )
                                            turnAudio.SetLength(0L)
                                        with
                                        | :? OperationCanceledException ->
                                            running <- false
                                            do! trySendJsonLocked {| ``type`` = "generation.canceled"; id = session.Id |}
                                        | ex ->
                                            running <- false
                                            do! trySendJsonLocked {| ``type`` = "error"; message = ex.Message; agent = agentStatusPayload agent |}
                                | _ ->
                                    do! sendJsonLocked {| ``type`` = "error"; message = $"Unknown WebSocket event '{eventType}'." |}
                            with ex ->
                                do! sendJsonLocked {| ``type`` = "error"; message = ex.Message; agent = agentStatusPayload agent |}
                        | _ ->
                            do! sendJsonLocked {| ``type`` = "error"; message = $"Unsupported WebSocket message type {messageType}." |}

                    if socket.State = WebSocketState.Open || socket.State = WebSocketState.CloseReceived then
                        do! socket.CloseAsync(WebSocketCloseStatus.NormalClosure, "done", CancellationToken.None)
        }

    let private serveArtifact (runtime: IS2sRuntime) backend fileName (ctx: HttpContext) =
        task {
            let sessionId = routeValue ctx "id"
            if not (safeId sessionId) then
                do! writeJson ctx 400 (error 400 "Invalid session id.")
            else
                match runtime.TryGetArtifact(sessionId, backend, fileName) with
                | None -> do! writeJson ctx 404 (error 404 $"Session artifact {fileName} was not found.")
                | Some artifact ->
                    ctx.Response.ContentType <- artifact.ContentType
                    do! ctx.Response.SendFileAsync(artifact.Path)
        }

    let private serveTurnArtifact (runtime: IS2sRuntime) backend fileName (ctx: HttpContext) =
        task {
            let sessionId = routeValue ctx "id"
            let turnIndexText = routeValue ctx "turnIndex"
            match Int32.TryParse(turnIndexText) with
            | false, _ -> do! writeJson ctx 400 (error 400 "Invalid turn index.")
            | true, turnIndex ->
                if not (safeId sessionId) || turnIndex < 1 then
                    do! writeJson ctx 400 (error 400 "Invalid session id or turn index.")
                else
                    match runtime.TryGetTurnArtifact(sessionId, turnIndex, backend, fileName) with
                    | None -> do! writeJson ctx 404 (error 404 $"Session turn artifact {fileName} was not found.")
                    | Some artifact ->
                        ctx.Response.ContentType <- artifact.ContentType
                        do! ctx.Response.SendFileAsync(artifact.Path)
        }

    let private serveAgentTurnArtifact (agent: IAgentRuntime) fileName (ctx: HttpContext) =
        task {
            let sessionId = routeValue ctx "id"
            let turnIndexText = routeValue ctx "turnIndex"
            match Int32.TryParse(turnIndexText) with
            | false, _ -> do! writeJson ctx 400 (error 400 "Invalid turn index.")
            | true, turnIndex ->
                if not (safeId sessionId) || turnIndex < 1 then
                    do! writeJson ctx 400 (error 400 "Invalid session id or turn index.")
                else
                    match agent.TryGetTurnArtifact(sessionId, turnIndex, fileName) with
                    | None -> do! writeJson ctx 404 (error 404 $"Agent turn artifact {fileName} was not found.")
                    | Some artifact ->
                        ctx.Response.ContentType <- artifact.ContentType
                        do! ctx.Response.SendFileAsync(artifact.Path)
        }

    let private mapCore (app: WebApplication) (runtime: IS2sRuntime) (agent: IAgentRuntime option) =
        app.UseWebSockets() |> ignore
        app.MapGet("/", RequestDelegate(fun ctx -> task { do! writeText ctx "text/html; charset=utf-8" indexHtml })) |> ignore
        app.MapGet("/assets/southern_belle.mp3", RequestDelegate(fun ctx -> serveAsset "southern_belle.mp3" "audio/mpeg" ctx)) |> ignore
        app.MapGet("/assets/southern_belle_prompt.txt", RequestDelegate(fun ctx -> serveAsset "southern_belle_prompt.txt" "text/plain; charset=utf-8" ctx)) |> ignore
        app.MapGet("/healthz", RequestDelegate(fun ctx -> writeJson ctx 200 {| ok = true |})) |> ignore
        app.MapGet(
            "/api/status",
            RequestDelegate(fun ctx ->
                task {
                    match agent with
                    | Some agentRuntime ->
                        do! writeJson ctx 200 (combinedStatusPayload runtime agentRuntime)
                    | None -> do! writeJson ctx 200 (statusPayload runtime)
                })
        )
        |> ignore
        app.MapPost("/api/s2s/sessions", RequestDelegate(fun ctx -> createSession runtime ctx)) |> ignore
        app.MapGet("/ws/s2s/{id}", RequestDelegate(fun ctx -> handleSocket runtime ctx)) |> ignore
        match agent with
        | Some agentRuntime ->
            app.MapPost("/api/agent/sessions", RequestDelegate(fun ctx -> createAgentSession runtime agentRuntime ctx)) |> ignore
            app.MapGet("/ws/agent/{id}", RequestDelegate(fun ctx -> handleAgentSocket runtime agentRuntime ctx)) |> ignore
            app.MapGet("/api/agent/sessions/{id}/turns/{turnIndex}/details.json", RequestDelegate(fun ctx -> serveAgentTurnArtifact agentRuntime "details.json" ctx)) |> ignore
        | None -> ()
        app.MapGet("/api/s2s/sessions/{id}/turns/{turnIndex}/details.json", RequestDelegate(fun ctx -> serveTurnArtifact runtime None "details.json" ctx)) |> ignore
        app.MapGet("/api/s2s/sessions/{id}/turns/{turnIndex}/{backend}/details.json", RequestDelegate(fun ctx -> serveTurnArtifact runtime (Some(routeValue ctx "backend")) "details.json" ctx)) |> ignore
        app.MapGet("/api/s2s/sessions/{id}/turns/{turnIndex}/{backend}/audio.wav", RequestDelegate(fun ctx -> serveTurnArtifact runtime (Some(routeValue ctx "backend")) "audio.wav" ctx)) |> ignore
        app.MapGet("/api/s2s/sessions/{id}/details.json", RequestDelegate(fun ctx -> serveArtifact runtime None "details.json" ctx)) |> ignore
        app.MapGet("/api/s2s/sessions/{id}/audio.wav", RequestDelegate(fun ctx -> serveArtifact runtime None "audio.wav" ctx)) |> ignore
        app.MapGet("/api/s2s/sessions/{id}/{backend}/details.json", RequestDelegate(fun ctx -> serveArtifact runtime (Some(routeValue ctx "backend")) "details.json" ctx)) |> ignore
        app.MapGet("/api/s2s/sessions/{id}/{backend}/audio.wav", RequestDelegate(fun ctx -> serveArtifact runtime (Some(routeValue ctx "backend")) "audio.wav" ctx)) |> ignore
        app

    let map (app: WebApplication) (runtime: IS2sRuntime) =
        mapCore app runtime None

    let mapWithAgent (app: WebApplication) (runtime: IS2sRuntime) (agent: IAgentRuntime) =
        mapCore app runtime (Some agent)
