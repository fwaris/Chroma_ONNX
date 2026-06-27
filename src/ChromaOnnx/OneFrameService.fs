namespace ChromaOnnx

open System
open System.Collections.Generic
open System.Diagnostics
open System.Globalization
open System.IO
open System.Linq
open System.Text.Json
open System.Threading
open Microsoft.AspNetCore.Builder
open Microsoft.AspNetCore.Hosting
open Microsoft.AspNetCore.Http
open Microsoft.Extensions.Hosting
open Microsoft.ML.OnnxRuntime.Tensors

module Serve =
    let private jsonOptions =
        JsonSerializerOptions(PropertyNamingPolicy = JsonNamingPolicy.CamelCase, WriteIndented = true)

    let private indexHtml =
        """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Chroma ONNX Lab</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #1d252c;
      --muted: #66727e;
      --line: #d7dde3;
      --panel: #f6f8fa;
      --accent: #0f766e;
      --accent-strong: #0b5f59;
      --danger: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font: 15px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #ffffff;
    }
    header {
      border-bottom: 1px solid var(--line);
      padding: 18px 24px 14px;
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 16px;
    }
    h1 { margin: 0; font-size: 20px; font-weight: 650; letter-spacing: 0; }
    main {
      display: grid;
      grid-template-columns: minmax(320px, 420px) minmax(0, 1fr);
      min-height: calc(100vh - 64px);
    }
    form {
      border-right: 1px solid var(--line);
      padding: 20px 24px;
      display: grid;
      align-content: start;
      gap: 16px;
      background: var(--panel);
    }
    label { display: grid; gap: 7px; font-weight: 600; }
    textarea, input[type="file"] {
      width: 100%;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 6px;
      padding: 10px 11px;
      font: inherit;
    }
    textarea { min-height: 130px; resize: vertical; }
    .row { display: flex; align-items: center; gap: 10px; }
    .row label { display: flex; align-items: center; gap: 8px; font-weight: 500; }
    button {
      border: 0;
      border-radius: 6px;
      padding: 11px 14px;
      background: var(--accent);
      color: white;
      font-weight: 700;
      cursor: pointer;
    }
    button:disabled { opacity: .65; cursor: wait; }
    section { padding: 22px 28px; display: grid; align-content: start; gap: 18px; }
    .status, .result {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #fff;
    }
    .status { color: var(--muted); }
    .pending { color: #8a5a00; }
    .error { color: var(--danger); white-space: pre-wrap; }
    audio { width: min(720px, 100%); }
    pre {
      margin: 0;
      overflow: auto;
      max-height: 42vh;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #0f1720;
      color: #e6edf3;
      padding: 12px;
      font-size: 12px;
    }
    a { color: var(--accent-strong); font-weight: 600; }
    @media (max-width: 780px) {
      main { grid-template-columns: 1fr; }
      form { border-right: 0; border-bottom: 1px solid var(--line); }
    }
  </style>
</head>
<body>
  <header>
    <h1>Chroma ONNX Voice Lab</h1>
    <div id="status" class="pending">Checking runtime...</div>
  </header>
  <main>
    <form id="generateForm">
      <label>
        Prompt text
        <textarea id="promptText" name="promptText" required>War and bloodshed throughout the world.</textarea>
      </label>
      <label>
        Reference audio
        <input id="promptAudio" name="promptAudio" type="file" accept="audio/*" required>
      </label>
      <div class="row">
        <label><input id="compare" name="compare" type="checkbox"> Compare with Python reference</label>
      </div>
      <button id="generateButton" type="submit">Generate</button>
      <div id="message" class="status">V1 serves the validated one-frame F#/ONNX path. Full generate parity is pending.</div>
    </form>
    <section>
      <div class="result">
        <h2>Response</h2>
        <audio id="audio" controls hidden></audio>
        <p id="previewNote" class="status" hidden></p>
        <p><a id="download" hidden>Download WAV</a></p>
      </div>
      <pre id="details">{}</pre>
    </section>
  </main>
  <script>
    const statusEl = document.getElementById('status');
    const form = document.getElementById('generateForm');
    const button = document.getElementById('generateButton');
    const message = document.getElementById('message');
    const audio = document.getElementById('audio');
    const previewNote = document.getElementById('previewNote');
    const download = document.getElementById('download');
    const details = document.getElementById('details');

    async function refreshStatus() {
      const response = await fetch('/api/status');
      const payload = await response.json();
      statusEl.textContent = payload.ready ? `Ready: ${payload.mode}` : 'Not ready';
      details.textContent = JSON.stringify(payload, null, 2);
    }

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      button.disabled = true;
      message.textContent = 'Generating...';
      message.className = 'status pending';
      audio.hidden = true;
      previewNote.hidden = true;
      download.hidden = true;

      try {
        const formData = new FormData(form);
        formData.set('compare', document.getElementById('compare').checked ? 'true' : 'false');
        const response = await fetch('/api/generate', { method: 'POST', body: formData });
        const payload = await response.json();
        details.textContent = JSON.stringify(payload, null, 2);
        if (!response.ok) throw new Error(payload.error || response.statusText);
        audio.src = payload.audioUrl;
        audio.hidden = false;
        if (payload.preview) {
          const duration = Number(payload.preview.durationSeconds || 0).toFixed(3);
          const gain = Number(payload.preview.wavPreviewGain || 1).toFixed(1);
          previewNote.textContent = `One-frame preview: ${duration}s. Browser WAV gain: ${gain}x. Raw .f32 is unchanged.`;
          previewNote.hidden = false;
        }
        download.href = payload.audioUrl;
        download.download = `${payload.id}.wav`;
        download.hidden = false;
        message.textContent = 'Done';
        message.className = 'status';
      } catch (error) {
        message.textContent = error.message;
        message.className = 'status error';
      } finally {
        button.disabled = false;
      }
    });

    refreshStatus().catch(error => {
      statusEl.textContent = error.message;
      statusEl.className = 'error';
    });
  </script>
</body>
</html>
"""

    let private resolveMaybePath (path: string) =
        if path.Contains(string Path.DirectorySeparatorChar)
           || path.Contains(string Path.AltDirectorySeparatorChar)
           || path.StartsWith(".", StringComparison.Ordinal) then
            Path.GetFullPath(path)
        else
            path

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

    let private getStringProperty (name: string) (root: JsonElement) =
        match root.GetProperty(name).GetString() with
        | null -> raise (InvalidDataException($"Helper JSON missing string property {name}."))
        | value -> value

    let private getIntProperty (name: string) (root: JsonElement) =
        root.GetProperty(name).GetInt32()

    let private newRunId () =
        let timestamp = DateTimeOffset.UtcNow.ToString("yyyyMMddHHmmssfff", CultureInfo.InvariantCulture)
        let suffix = Guid.NewGuid().ToString("N").Substring(0, 8)
        $"{timestamp}_{suffix}"

    let private safeExtension (fileName: string) =
        let extension = Path.GetExtension(fileName)
        if String.IsNullOrWhiteSpace(extension) then ".wav" else string extension

    let private routeRunId (ctx: HttpContext) =
        match ctx.Request.RouteValues.TryGetValue("id") with
        | true, value when value <> null -> string value
        | _ -> ""

    let private isSafeRunId (runId: string) =
        runId.Length > 0
        && runId |> Seq.forall (fun ch -> Char.IsLetterOrDigit(ch) || ch = '_' || ch = '-')

    let private serveRunFile (workDir: string) (fileName: string) (contentType: string) (ctx: HttpContext) =
        task {
            let runId = routeRunId ctx
            if not (isSafeRunId runId) then
                do! writeJson ctx 400 (error 400 "Invalid run id.")
            else
                let root = Path.GetFullPath(workDir)
                let path = Path.GetFullPath(Path.Combine(root, runId, fileName))
                if not (path.StartsWith(root, StringComparison.OrdinalIgnoreCase)) then
                    do! writeJson ctx 400 (error 400 "Invalid run path.")
                elif not (File.Exists path) then
                    do! writeJson ctx 404 (error 404 "Run artifact was not found.")
                else
                    ctx.Response.ContentType <- contentType
                    let! bytes = File.ReadAllBytesAsync(path)
                    do! ctx.Response.Body.WriteAsync(bytes)
        }

    let private generate
        (repoRoot: string)
        (modelDir: string)
        (workDir: string)
        (python: string)
        (helperScript: string)
        (runner: ChromaSharedOnnxRunner)
        (generationLock: SemaphoreSlim)
        (ctx: HttpContext)
        =
        task {
            if not ctx.Request.HasFormContentType then
                do! writeJson ctx 400 (error 400 "Expected multipart/form-data.")
            else
                let! form = ctx.Request.ReadFormAsync()
                let promptText = form["promptText"].ToString()
                let compare = String.Equals(form["compare"].ToString(), "true", StringComparison.OrdinalIgnoreCase)
                let promptAudio =
                    match form.Files.GetFile("promptAudio") with
                    | null -> None
                    | file when file.Length = 0L -> None
                    | file -> Some file

                if String.IsNullOrWhiteSpace(promptText) then
                    do! writeJson ctx 400 (error 400 "promptText is required.")
                elif promptAudio.IsNone then
                    do! writeJson ctx 400 (error 400 "promptAudio is required.")
                else
                    let promptAudio = promptAudio.Value
                    let runId = newRunId ()
                    let runDir = Path.Combine(workDir, runId)
                    Directory.CreateDirectory(runDir) |> ignore

                    let uploadPath = Path.Combine(runDir, "prompt_audio" + safeExtension promptAudio.FileName)
                    use uploadStream = File.Create(uploadPath)
                    do! promptAudio.CopyToAsync(uploadStream)
                    uploadStream.Close()

                    let timings = Dictionary<string, int64>(StringComparer.Ordinal)
                    let warnings = ResizeArray<string>()
                    let stopwatch = Stopwatch.StartNew()

                    let! prepareResult =
                        ProcessRunner.run
                            python
                            [ helperScript
                              "prepare"
                              "--model-dir"; modelDir
                              "--prompt-text"; promptText
                              "--prompt-audio"; uploadPath
                              "--output-dir"; runDir ]
                            repoRoot

                    timings["prepare"] <- stopwatch.ElapsedMilliseconds
                    File.WriteAllText(Path.Combine(runDir, "prepare.stdout.json"), prepareResult.Stdout)
                    File.WriteAllText(Path.Combine(runDir, "prepare.stderr.txt"), prepareResult.Stderr)

                    if prepareResult.ExitCode <> 0 then
                        do! writeJson ctx 500 (error 500 $"Python preprocessing failed: {prepareResult.Stderr}")
                    else
                        use preparedDoc = JsonDocument.Parse(prepareResult.Stdout)
                        let prepared = preparedDoc.RootElement
                        let batch = getIntProperty "batch" prepared
                        let textSeq = getIntProperty "text_seq" prepared
                        let audioSamples = getIntProperty "audio_samples" prepared
                        let sampleRate = getIntProperty "sample_rate" prepared

                        let inputIdsPath = getStringProperty "input_ids" prepared
                        let attentionMaskPath = getStringProperty "attention_mask" prepared
                        let inputValuesPath = getStringProperty "input_values" prepared
                        let inputValuesCutoffsPath = getStringProperty "input_values_cutoffs" prepared

                        let inputIds =
                            TensorIO.readInt64s inputIdsPath (batch * textSeq)
                            |> fun values -> TensorIO.denseInt64 values [| batch; textSeq |]

                        let attentionMask =
                            TensorIO.readInt64s attentionMaskPath (batch * textSeq)
                            |> fun values -> TensorIO.denseInt64 values [| batch; textSeq |]

                        let inputValues =
                            TensorIO.readSingles inputValuesPath (batch * audioSamples)
                            |> fun values -> TensorIO.denseFloat values [| batch; 1; audioSamples |]

                        let inputValuesCutoffs =
                            TensorIO.readInt64s inputValuesCutoffsPath batch
                            |> fun values -> TensorIO.denseInt64 values [| batch |]

                        stopwatch.Restart()
                        do! generationLock.WaitAsync()
                        let mutable released = false

                        try
                            let prefill = runner.RunSystemPrefill(inputIds, attentionMask, inputValues, inputValuesCutoffs)
                            let frameCodes = runner.GreedyAudioFrame(prefill, runner.AudioNumCodebooks)
                            let frameCodeValues = Enumerable.ToArray(frameCodes)
                            let codecCodes = DenseTensor<int64>(frameCodeValues, [| batch; runner.AudioNumCodebooks; 1 |])
                            let audio = runner.RunCodecDecode(codecCodes)
                            timings["onnx"] <- stopwatch.ElapsedMilliseconds

                            generationLock.Release() |> ignore
                            released <- true

                            let codesPath = Path.Combine(runDir, "audio_codes.i64")
                            let audioRawPath = Path.Combine(runDir, "audio_values.f32")
                            let audioWavPath = Path.Combine(runDir, "audio.wav")
                            TensorIO.writeInt64s codesPath codecCodes
                            TensorIO.writeSingles audioRawPath audio
                            let previewStats = Wave.writeMono16 audioWavPath sampleRate audio
                            if previewStats.DurationSeconds < 0.25 then
                                warnings.Add(
                                    $"V1 generated one codec frame ({previewStats.DurationSeconds:F3}s). This is expected to sound like a very short preview, not a full spoken response."
                                )

                            if previewStats.WavPreviewGain <> 1.0 then
                                warnings.Add(
                                    $"Browser WAV preview was amplified {previewStats.WavPreviewGain:F1}x. Raw audio_values.f32 is unchanged."
                                )

                            let mutable comparison =
                                use nullDoc = JsonDocument.Parse("null")
                                nullDoc.RootElement.Clone()

                            if compare then
                                stopwatch.Restart()
                                let! compareResult =
                                    ProcessRunner.run
                                        python
                                        [ helperScript
                                          "compare"
                                          "--model-dir"; modelDir
                                          "--prepared-dir"; runDir
                                          "--onnx-codes"; codesPath
                                          "--onnx-audio"; audioRawPath
                                          "--output-dir"; runDir ]
                                        repoRoot

                                timings["compare"] <- stopwatch.ElapsedMilliseconds
                                File.WriteAllText(Path.Combine(runDir, "compare.stdout.json"), compareResult.Stdout)
                                File.WriteAllText(Path.Combine(runDir, "compare.stderr.txt"), compareResult.Stderr)

                                if compareResult.ExitCode = 0 then
                                    use compareDoc = JsonDocument.Parse(compareResult.Stdout)
                                    comparison <- compareDoc.RootElement.Clone()
                                else
                                    warnings.Add($"Python comparison failed: {compareResult.Stderr}")

                            let details =
                                {| id = runId
                                   mode = "one_frame"
                                   runtime = "fsharp_onnx_shared"
                                   fullGenerateParity = "pending"
                                   createdUtc = DateTimeOffset.UtcNow.ToString("O", CultureInfo.InvariantCulture)
                                   promptText = promptText
                                   promptAudioFile = Path.GetFileName(uploadPath)
                                   sampleRate = sampleRate
                                   audioUrl = $"/api/runs/{runId}/audio.wav"
                                   detailsUrl = $"/api/runs/{runId}/details.json"
                                   rawCodesPath = codesPath
                                   rawAudioPath = audioRawPath
                                   preview = previewStats
                                   codes = frameCodeValues
                                   shapes =
                                    {| logits = prefill.Logits.Dimensions.ToArray()
                                       hiddenStates = prefill.HiddenStates.Dimensions.ToArray()
                                       audioCodes = codecCodes.Dimensions.ToArray()
                                       audioValues = audio.Dimensions.ToArray() |}
                                   timingsMs = timings
                                   comparison = comparison
                                   warnings = warnings.ToArray() |}

                            let detailsJson = JsonSerializer.Serialize(details, jsonOptions)
                            File.WriteAllText(Path.Combine(runDir, "details.json"), detailsJson)
                            do! writeJson ctx 200 details
                        with ex ->
                            if not released then
                                generationLock.Release() |> ignore

                            do! writeJson ctx 500 (error 500 ex.Message)
        }

    let run args (tryFind: string -> string list -> string option) (required: string -> string list -> string) (optional: string -> string -> string list -> string) =
        let modelDir = required "--model-dir" args |> Path.GetFullPath
        let bundleDir = required "--bundle-dir" args |> Path.GetFullPath
        let workDir = optional "served_runs" "--work-dir" args |> Path.GetFullPath
        let port = optional "5055" "--port" args |> int
        let python = optional ".venv\\Scripts\\python.exe" "--python" args |> resolveMaybePath
        let helperScript = optional "scripts\\chroma_serve_helper.py" "--helper-script" args |> Path.GetFullPath
        let repoRoot = Directory.GetCurrentDirectory()

        Directory.CreateDirectory(workDir) |> ignore

        use runner = new ChromaSharedOnnxRunner(modelDir, bundleDir)
        use generationLock = new SemaphoreSlim(1, 1)

        let builder = WebApplication.CreateBuilder(Array.empty<string>)
        builder.WebHost.UseUrls($"http://localhost:{port}") |> ignore
        let app = builder.Build()

        app.MapGet(
            "/",
            RequestDelegate(fun ctx ->
                task {
                    do! writeText ctx "text/html; charset=utf-8" indexHtml
                })
        )
        |> ignore

        app.MapGet(
            "/api/status",
            RequestDelegate(fun ctx ->
                task {
                    let payload =
                        {| ready = true
                           mode = "one_frame"
                           runtime = "fsharp_onnx_shared"
                           fullGenerateParity = "pending"
                           modelDir = modelDir
                           bundleDir = bundleDir
                           mappedSafetensorsShards = runner.MappedShardCount
                           safetensorBackedInitializers = runner.InitializerCount
                           uniqueSourceTensors = runner.UniqueSourceTensorCount
                           audioNumCodebooks = runner.AudioNumCodebooks |}

                    do! writeJson ctx 200 payload
                })
        )
        |> ignore

        app.MapPost(
            "/api/generate",
            RequestDelegate(fun ctx ->
                generate repoRoot modelDir workDir python helperScript runner generationLock ctx)
        )
        |> ignore

        app.MapGet(
            "/api/runs/{id}/audio.wav",
            RequestDelegate(fun ctx ->
                serveRunFile workDir "audio.wav" "audio/wav" ctx)
        )
        |> ignore

        app.MapGet(
            "/api/runs/{id}/details.json",
            RequestDelegate(fun ctx ->
                serveRunFile workDir "details.json" "application/json; charset=utf-8" ctx)
        )
        |> ignore

        printfn "Chroma ONNX lab listening on http://localhost:%d" port
        printfn "Mode: one_frame (full generate parity pending)"
        app.Run()
        0

