namespace ChromaOnnx

open System
open System.Collections.Concurrent
open System.Globalization
open System.IO
open System.Net.WebSockets
open System.Text
open System.Text.Json
open System.Threading
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
  <title>Chroma S2S ONNX</title>
  <style>
    body { margin: 0; font: 15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; color: #18212a; }
    header { padding: 16px 22px; border-bottom: 1px solid #d7dde3; display: flex; justify-content: space-between; gap: 16px; }
    main { display: grid; grid-template-columns: minmax(320px, 430px) 1fr; min-height: calc(100vh - 58px); }
    form { padding: 18px 22px; background: #f6f8fa; border-right: 1px solid #d7dde3; display: grid; gap: 14px; align-content: start; }
    label { display: grid; gap: 6px; font-weight: 650; }
    input, textarea, button { font: inherit; }
    textarea, input[type=file], input[type=number], select { border: 1px solid #cfd6de; border-radius: 6px; padding: 9px 10px; background: white; }
    textarea { min-height: 100px; resize: vertical; }
    button { border: 0; border-radius: 6px; padding: 10px 13px; background: #0f766e; color: white; font-weight: 750; cursor: pointer; }
    button:disabled { opacity: .6; cursor: wait; }
    section { padding: 18px 24px; display: grid; gap: 14px; align-content: start; }
    .status { border: 1px solid #d7dde3; border-radius: 8px; padding: 12px; color: #5e6a75; }
    .bad { color: #b42318; }
    pre { margin: 0; max-height: 54vh; overflow: auto; background: #101923; color: #e6edf3; border-radius: 6px; padding: 12px; font-size: 12px; }
    audio { width: min(720px, 100%); }
    .audioResults { display: grid; gap: 12px; }
    .audioResult { display: grid; gap: 6px; }
    .audioResult strong { font-size: 13px; color: #52606d; }
    @media (max-width: 780px) { main { grid-template-columns: 1fr; } form { border-right: 0; border-bottom: 1px solid #d7dde3; } }
  </style>
</head>
<body>
  <header>
    <strong>Chroma S2S ONNX</strong>
    <span id="runtime">Checking runtime...</span>
  </header>
  <main>
    <form id="sessionForm">
      <label>System prompt<textarea id="systemPrompt">You are a helpful assistant.</textarea></label>
      <label>Voice prompt text<textarea id="promptText" required>War and bloodshed throughout the world.</textarea></label>
      <label>Voice prompt audio, 24 kHz target<input id="promptPcm" type="file" accept="audio/*,.f32" required></label>
      <label>User turn audio, 16 kHz target<input id="turnPcm" type="file" accept="audio/*,.f32" required></label>
      <label>Backend<select id="backend">
        <option value="fsharp_onnx" selected>F#/ONNX</option>
        <option value="python">Python Chroma</option>
        <option value="both">Both</option>
      </select></label>
      <label>Max frames<input id="maxNewFrames" type="number" min="1" max="100" value="25"></label>
      <button id="sendButton" type="submit">Create Session And Send Turn</button>
      <div class="status">F#/ONNX is the default path. Python Chroma is available only when selected and uses the same configured thinker window for comparison.</div>
    </form>
    <section>
      <div id="message" class="status">Idle</div>
      <div id="audioResults" class="audioResults"></div>
      <audio id="audio" controls hidden></audio>
      <pre id="details">{}</pre>
    </section>
  </main>
  <script>
    const runtime = document.getElementById('runtime');
    const form = document.getElementById('sessionForm');
    const button = document.getElementById('sendButton');
    const message = document.getElementById('message');
    const details = document.getElementById('details');
    const audio = document.getElementById('audio');
    const audioResults = document.getElementById('audioResults');

    let audioContext = null;
    let currentSocket = null;

    function setBusy(isBusy) {
      button.disabled = isBusy;
      button.textContent = isBusy ? 'Generating...' : 'Create Session And Send Turn';
      button.setAttribute('aria-busy', isBusy ? 'true' : 'false');
    }

    function cacheBust(url) {
      if (!url) return '';
      return `${url}${url.includes('?') ? '&' : '?'}v=${Date.now()}`;
    }

    function renderBackendResults(results) {
      audio.hidden = true;
      audio.removeAttribute('src');
      audioResults.innerHTML = '';
      for (const result of results) {
        const block = document.createElement('div');
        block.className = 'audioResult';

        const label = document.createElement('strong');
        label.textContent = result.label || result.backend || 'Result';

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

        block.append(label, player, links);
        audioResults.append(block);
      }
    }

    async function readFileBytes(input) {
      if (!input.files.length) throw new Error(`${input.id} is required`);
      return new Uint8Array(await input.files[0].arrayBuffer());
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

    async function readAudioAsF32Bytes(input, targetRate) {
      if (!input.files.length) throw new Error(`${input.id} is required`);
      const file = input.files[0];
      if (/\.f32$/i.test(file.name)) return readFileBytes(input);

      audioContext = audioContext || new (window.AudioContext || window.webkitAudioContext)();
      const fileBytes = await file.arrayBuffer();
      const audioBuffer = await audioContext.decodeAudioData(fileBytes.slice(0));
      const mono = mixToMono(audioBuffer);
      const resampled = resampleLinear(mono, audioBuffer.sampleRate, targetRate);
      return asByteView(resampled);
    }

    async function refreshStatus() {
      const response = await fetch('/api/status');
      const payload = await response.json();
      runtime.textContent = payload.ready ? `Ready: ${payload.executionProvider}` : 'Not ready';
      runtime.className = payload.ready ? '' : 'bad';
      details.textContent = JSON.stringify(payload, null, 2);
    }

    form.addEventListener('submit', async event => {
      event.preventDefault();
      if (currentSocket && currentSocket.readyState === WebSocket.OPEN) currentSocket.close(1000, 'new request');
      setBusy(true);
      audio.hidden = true;
      audio.removeAttribute('src');
      audioResults.innerHTML = '';
      message.textContent = 'Creating session...';
      message.className = 'status';
      let finished = false;
      const finish = () => {
        if (!finished) {
          finished = true;
          setBusy(false);
        }
      };
      try {
        const promptPcm = await readAudioAsF32Bytes(document.getElementById('promptPcm'), 24000);
        const turnPcm = await readAudioAsF32Bytes(document.getElementById('turnPcm'), 16000);
        const formData = new FormData();
        formData.set('promptText', document.getElementById('promptText').value);
        formData.set('systemPrompt', document.getElementById('systemPrompt').value);
        formData.set('backend', document.getElementById('backend').value);
        formData.set('maxNewFrames', document.getElementById('maxNewFrames').value);
        formData.set('promptPcm24k', new Blob([promptPcm], { type: 'application/octet-stream' }), 'prompt.f32');
        const sessionResponse = await fetch('/api/s2s/sessions', { method: 'POST', body: formData });
        const session = await sessionResponse.json();
        details.textContent = JSON.stringify(session, null, 2);
        if (!sessionResponse.ok) throw new Error(session.error || sessionResponse.statusText);

        message.textContent = 'Sending turn...';
        const ws = new WebSocket(`${location.origin.replace('http', 'ws')}/ws/s2s/${session.id}`);
        currentSocket = ws;
        ws.binaryType = 'arraybuffer';
        ws.onmessage = event => {
          const payload = JSON.parse(event.data);
          details.textContent = JSON.stringify(payload, null, 2);
          if (payload.type === 'generation.done' && Array.isArray(payload.results)) {
            renderBackendResults(payload.results);
          } else if (payload.type === 'generation.done' && payload.audioUrl) {
            audio.src = cacheBust(payload.audioUrl);
            audio.hidden = false;
          }
          if (payload.type === 'error') {
            message.textContent = payload.message;
            message.className = 'status bad';
            finish();
            if (ws.readyState === WebSocket.OPEN) ws.close(1000, 'error handled');
          } else {
            message.textContent = payload.type;
            message.className = 'status';
            if (payload.type === 'generation.done') {
              finish();
              if (ws.readyState === WebSocket.OPEN) ws.close(1000, 'done');
            }
          }
        };
        ws.onopen = () => {
          ws.send(JSON.stringify({ type: 'turn.start' }));
          ws.send(turnPcm);
          ws.send(JSON.stringify({ type: 'turn.end' }));
        };
        ws.onerror = () => {
          message.textContent = 'WebSocket error while generating.';
          message.className = 'status bad';
          finish();
        };
        ws.onclose = () => {
          if (currentSocket === ws) currentSocket = null;
          finish();
        };
      } catch (error) {
        message.textContent = error.message;
        message.className = 'status bad';
        finish();
      }
    });

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

    let private sendJson (socket: WebSocket) (payload: 'T) =
        task {
            let bytes = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(payload, jsonOptions))
            do! socket.SendAsync(ArraySegment<byte>(bytes), WebSocketMessageType.Text, true, CancellationToken.None)
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

    let private createSession (processor: ChromaNativeProcessor) (store: S2sSessionStore) (ctx: HttpContext) =
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
                        | true, value -> max 1 (min 100 value)
                        | false, _ -> 25

                    if String.IsNullOrWhiteSpace(promptText) then
                        do! writeJson ctx 400 (error 400 "promptText is required.")
                    else
                        let promptAudioBytes = readFormFileBytes form "promptPcm24k"
                        let promptAudio = processor.ReadFloat32PcmFromBytes promptAudioBytes
                        let session = store.Create(promptText, systemPrompt, backend, promptAudio, maxNewFrames)
                        File.WriteAllBytes(Path.Combine(session.WorkDir, "prompt_audio_24k.f32"), promptAudioBytes)
                        let payload =
                            {| id = session.Id
                               mode = "s2s_greedy"
                               backend = session.Backend
                               promptText = session.PromptText
                               systemPrompt = session.SystemPrompt
                               maxNewFrames = session.MaxNewFrames
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
        (session: S2sSession)
        (userAudio: float32 array)
        (prepared: NativeS2sPrepared)
        =
        let backend = "fsharp_onnx"
        let backendDir = Path.Combine(session.WorkDir, backend)
        Directory.CreateDirectory(backendDir) |> ignore
        let memoryBefore = RuntimeMemory.current()
        let result = runner.Generate(prepared, session.MaxNewFrames)
        let memoryAfter = RuntimeMemory.current()
        let codesPath = Path.Combine(backendDir, "audio_codes.i64")
        let rawAudioPath = Path.Combine(backendDir, "audio_values.f32")
        let wavPath = Path.Combine(backendDir, "audio.wav")
        TensorIO.writeInt64s codesPath result.AudioCodes
        TensorIO.writeSingles rawAudioPath result.AudioValues
        let wavStats = Wave.writeMono16 wavPath 24000 result.AudioValues
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
               stopReason = result.StopReason
               stepKinds = result.StepKinds
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
        (generationLock: SemaphoreSlim)
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
                    do! sendJson socket {| ``type`` = "session.ready"; id = session.Id; maxNewFrames = session.MaxNewFrames |}
                    use turnAudio = new MemoryStream()
                    let mutable running = true
                    while running && socket.State = WebSocketState.Open do
                        let! messageType, payload = receiveMessage socket
                        match messageType with
                        | WebSocketMessageType.Close ->
                            running <- false
                        | WebSocketMessageType.Binary ->
                            turnAudio.Write(payload, 0, payload.Length)
                            do! sendJson socket {| ``type`` = "turn.chunk"; bytes = payload.Length; totalBytes = turnAudio.Length |}
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
                                do! sendJson socket {| ``type`` = "turn.accepted"; id = session.Id |}
                            | "turn.end" ->
                                try
                                    let turnAudioBytes = turnAudio.ToArray()
                                    let userAudio = processor.ReadFloat32PcmFromBytes(turnAudioBytes)
                                    let requestedBackends =
                                        match session.Backend with
                                        | "both" -> [| "fsharp_onnx"; "python" |]
                                        | backend -> [| backend |]
                                    File.WriteAllBytes(Path.Combine(session.WorkDir, "user_audio_16k.f32"), turnAudioBytes)
                                    do! sendJson socket {| ``type`` = "generation.started"; id = session.Id; backend = session.Backend; backends = requestedBackends; maxNewFrames = session.MaxNewFrames |}
                                    do! generationLock.WaitAsync()
                                    let results = ResizeArray<JsonElement>()
                                    try
                                        if requestedBackends |> Array.contains "fsharp_onnx" then
                                            use prepared = processor.Prepare(session.PromptText, session.SystemPrompt, session.PromptAudio24k, userAudio)
                                            File.WriteAllText(Path.Combine(session.WorkDir, "conversation.txt"), prepared.ConversationText)
                                            results.Add(runFsharpBackend processor runner session userAudio prepared)

                                        if requestedBackends |> Array.contains "python" then
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
                                    finally
                                        generationLock.Release() |> ignore

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
                                           mode = "s2s_greedy_compare"
                                           backend = session.Backend
                                           maxNewFrames = session.MaxNewFrames
                                           pythonInRequestPath = requestedBackends |> Array.contains "python"
                                           results = results.ToArray() |}
                                    let detailsJson = JsonSerializer.Serialize(details, jsonOptions)
                                    File.WriteAllText(Path.Combine(session.WorkDir, "details.json"), detailsJson)
                                    do!
                                        sendJson
                                            socket
                                            {| ``type`` = "generation.done"
                                               id = session.Id
                                               backend = session.Backend
                                               audioUrl = firstAudioUrl
                                               detailsUrl = firstDetailsUrl
                                               results = results.ToArray() |}
                                with ex ->
                                    do! sendJson socket {| ``type`` = "error"; message = ex.Message; bundle = runner.Status |}
                            | _ ->
                                do! sendJson socket {| ``type`` = "error"; message = $"Unknown WebSocket event '{eventType}'." |}
                        | _ ->
                            do! sendJson socket {| ``type`` = "error"; message = $"Unsupported WebSocket message type {messageType}." |}

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

        Directory.CreateDirectory(workDir) |> ignore
        let tuningOptions =
            { MemoryProfile = ortMemoryProfile
              OptimizedModelCacheDir = optimizedModelCacheDir
              OptimizedModelCacheFormat = optimizedModelCacheFormat }
        use runner = new ChromaS2sOnnxRunner(modelDir, bundleDir, executionProvider, memoryMode, tuningOptions)
        let processor = ChromaNativeProcessor(modelDir, thinkerActiveFrames)
        let store = S2sSessionStore(workDir)
        use generationLock = new SemaphoreSlim(1, 1)

        let builder = WebApplication.CreateBuilder(Array.empty<string>)
        builder.WebHost.UseUrls($"http://localhost:{port}") |> ignore
        let app = builder.Build()
        app.UseWebSockets() |> ignore

        app.MapGet("/", RequestDelegate(fun ctx -> task { do! writeText ctx "text/html; charset=utf-8" indexHtml })) |> ignore
        app.MapGet(
            "/api/status",
            RequestDelegate(fun ctx ->
                task {
                    let status = runner.Status
                    let payload =
                        {| ready = status.Ready
                           mode = "s2s_greedy"
                           pythonInRequestPath = false
                           pythonBackendAvailable = File.Exists python
                           python = python
                           pythonDevice = pythonDevice
                           modelDir = modelDir
                           bundleDir = bundleDir
                           executionProvider = status.ExecutionProvider
                           memoryMode = runner.MemoryMode
                           ortMemoryProfile = runner.OrtMemoryProfile
                           optimizedModelCacheEnabled = runner.OptimizedModelCacheEnabled
                           optimizedModelCacheDir = runner.OptimizedModelCacheDir
                           optimizedModelCacheFormat = runner.OptimizedModelCacheFormat
                           memory = RuntimeMemory.current()
                           loadedOrtSessions = runner.LoadedSessionNames
                           warmOrtSessions = runner.WarmSessionNames
                           activePagedOrtSessions = runner.ActivePagedSessionNames
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
        app.MapPost("/api/s2s/sessions", RequestDelegate(fun ctx -> createSession processor store ctx)) |> ignore
        app.MapGet("/ws/s2s/{id}", RequestDelegate(fun ctx -> handleSocket processor runner modelDir python pythonDevice thinkerActiveFramesArg generationLock store ctx)) |> ignore
        app.MapGet("/api/s2s/sessions/{id}/details.json", RequestDelegate(fun ctx -> serveSessionFile store "details.json" "application/json; charset=utf-8" ctx)) |> ignore
        app.MapGet("/api/s2s/sessions/{id}/audio.wav", RequestDelegate(fun ctx -> serveSessionFile store "audio.wav" "audio/wav" ctx)) |> ignore
        app.MapGet("/api/s2s/sessions/{id}/{backend}/details.json", RequestDelegate(fun ctx -> serveBackendSessionFile store "details.json" "application/json; charset=utf-8" ctx)) |> ignore
        app.MapGet("/api/s2s/sessions/{id}/{backend}/audio.wav", RequestDelegate(fun ctx -> serveBackendSessionFile store "audio.wav" "audio/wav" ctx)) |> ignore

        printfn "Chroma S2S ONNX service listening on http://localhost:%d" port
        printfn "Python in request path: selectable"
        printfn "Python backend: %s (%s)" python pythonDevice
        printfn "Memory mode: %s" runner.MemoryMode
        printfn "ORT memory profile: %s" runner.OrtMemoryProfile
        printfn "Thinker features: %s" processor.ThinkerFeatureMode
        if runner.OptimizedModelCacheEnabled then
            printfn "Optimized model cache: %s (%s)" runner.OptimizedModelCacheDir runner.OptimizedModelCacheFormat
        printfn "S2S bundle status: %s" runner.Status.Message
        app.Run()
        0

