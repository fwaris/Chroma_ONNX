# FlashLabs Chroma to ONNX plus F#

This workspace converts `FlashLabs/Chroma-4B` to ONNX and runs it from F# with ONNX Runtime. The current primary path is a Python-free F#/ONNX speech-to-speech service using a single merged, safetensors-backed S2S ONNX bundle.

## License

This repository's own source code is licensed under the MIT License. See `LICENSE`.

Chroma model files, weights, remote model code, tokenizer/processor files, and generated artifacts derived from `FlashLabs/Chroma-4B` are not relicensed here. See `THIRD_PARTY_NOTICES.md` and the upstream Chroma terms before redistributing any model or generated ONNX artifacts.

## Current Status

- The default tested S2S bundle is `onnx/chroma-s2s-full-v2`.
- The S2S runtime loads one merged weights-free ONNX graph: `chroma_s2s_merged.weights_free.onnx`.
- Model weights are loaded from the original Hugging Face safetensors through shared ORT initializers.
- F#/ONNX owns preprocessing, cacheful generation, codec decode, WAV output, service state, and persistence.
- Python is not in the F#/ONNX request path. It is available only as an optional UI/backend comparison path and offline parity oracle.
- Deterministic greedy S2S parity has been validated on the sample fixture:
  - 32 generated frames exact against Python Chroma.
  - Code IDs exact.
  - Decoded audio allclose at `1e-5`, with max absolute diff around `4.2e-7`.
- CUDA quality-safe mode disables TF32 in ORT and Python comparison tooling so Mimi/RVQ code selection stays deterministic.

Memory note: `resident-merged` keeps the single merged ORT session loaded for lower request latency, but it has a high private memory footprint. `python-footprint` pages sessions for lower idle memory and slower first request latency.

## Performance Comparison

> The F#-ONNX version is roughly 6X faster than Python-PyTorch on the GTX 3080 16GB and consumes much less OS memory. Why? [See this analysis](/performance_analysis/chroma_onnx_runtime_performance_analysis.md)

The latest warm-generation benchmark compares persistent Python Chroma CUDA against persistent F#/ONNX CUDA on the same fixture. Model load, ONNX session creation, safetensor mapping, processor setup, and warmup requests are excluded. The benchmark is intended to represent a server process that is already loaded and receiving requests one after another.

Benchmark fixture:

- Prompt text: `War and bloodshed throughout the world.`
- Voice prompt: `served_runs/compare_inputs/reference_audio_24k.f32`
- User turn: `served_runs/compare_inputs/make_taco_16k.f32`
- Greedy deterministic generation: `do_sample=False`, `use_cache=True`, `output_audio=True`
- F#/ONNX mode: `resident-merged`, `quality-safe`, `thinker-active-frames 0`
- Bundle/cache: `onnx/chroma-s2s-full-v2` with `ort-cache-ort-local-external`
- Quality mode disables TF32 in both Python comparison tooling and ORT to preserve deterministic RVQ/code selection.

Test machine/environment:

- OS: Windows x64
- GPU: NVIDIA GeForce RTX 3080 Laptop GPU, 16 GiB
- NVIDIA driver: `610.62`
- NVIDIA CUDA UMD: `13.3`
- CUDA Toolkit on PATH: `13.3`
- cuDNN on PATH: `C:\Program Files\NVIDIA\CUDNN\v9.23\bin\13.3\x64`
- .NET SDK: `10.0.301`
- F# ONNX Runtime package: `Microsoft.ML.OnnxRuntime.Gpu 1.27.0`
- Python: `3.11.15`
- PyTorch: `2.7.1+cu128`
- Python CUDA runtime reported by PyTorch: `12.8`
- Python cuDNN reported by PyTorch: `90701`
- Transformers: `5.0.0rc0`
- Python onnxruntime: `1.27.0`

Warm generation time:

| Requested frames | Python Chroma CUDA mean | F#/ONNX CUDA mean | F#/ONNX split | Speedup |
| ---: | ---: | ---: | --- | ---: |
| 8 | `21.97 s` | `2.97 s` | prefill `0.40 s`, generate `2.54 s`, decode `0.03 s` | `~7.4x` |
| 32 | `65.12 s` | `11.18 s` | prefill `0.39 s`, generate `10.76 s`, decode `0.03 s` | `~5.8x` |

Recent F#/ONNX allocation and active-frame updates improved the warm CUDA path versus the previous recorded F#/ONNX benchmark:

| Requested frames | Previous F#/ONNX mean | Latest F#/ONNX mean | Change |
| ---: | ---: | ---: | ---: |
| 8 | `3.63 s` | `2.97 s` | `-18.2%` |
| 32 | `13.65 s` | `11.18 s` | `-18.1%` |

Output parity for the measured artifacts:

| Requested frames | Code IDs | Decoded audio |
| ---: | --- | --- |
| 8 | exact match | `allclose` at `1e-4`, max abs diff `4.77e-7` |
| 32 | exact match | `allclose` at `1e-4`, max abs diff `4.84e-7` |

Host RAM observations from the same warm benchmark:

| Requested frames | Backend | Working set after benchmark | Private bytes after benchmark |
| ---: | --- | ---: | ---: |
| 8 | Python Chroma CUDA | `26.88 GiB` | `45.76 GiB` |
| 8 | F#/ONNX CUDA | `2.15 GiB` | `17.93 GiB` |
| 32 | Python Chroma CUDA | `26.88 GiB` | `45.77 GiB` |
| 32 | F#/ONNX CUDA | `2.11 GiB` | `17.95 GiB` |

On this machine, persistent Python Chroma used substantially more host RAM than the F#/ONNX resident merged path. Python also reported about `22.53 GiB` of PyTorch CUDA reserved memory after the benchmark. The F#/ONNX benchmark JSON did not capture per-process GPU memory because `nvidia-smi --query-compute-apps` did not expose the process values in this shell, so GPU memory should be checked from Task Manager or `nvidia-smi` during a live run when comparing VRAM/shared GPU pressure.

Reproduce the F#/ONNX benchmark:

```powershell
dotnet run --project src\ChromaOnnx -- s2s-benchmark `
  --model-dir models/chroma-4b `
  --bundle-dir onnx/chroma-s2s-full-v2 `
  --prompt-text "War and bloodshed throughout the world." `
  --prompt-audio-f32 served_runs/compare_inputs/reference_audio_24k.f32 `
  --user-audio-f32 served_runs/compare_inputs/make_taco_16k.f32 `
  --frames 32 `
  --warmup 1 `
  --iterations 3 `
  --output-dir served_runs/bench_generation_fsharp_onnx_cuda_32f `
  --execution-provider cuda `
  --memory-mode resident-merged `
  --ort-memory-profile quality-safe `
  --thinker-active-frames 0 `
  --optimized-model-cache-dir onnx/chroma-s2s-full-v2/ort-cache-ort-local-external `
  --optimized-model-cache-format onnx
```

Reproduce the Python benchmark:

```powershell
.venv\Scripts\python.exe scripts\chroma_s2s_benchmark.py `
  --model-dir models/chroma-4b `
  --prompt-text "War and bloodshed throughout the world." `
  --prompt-audio-f32 served_runs/compare_inputs/reference_audio_24k.f32 `
  --user-audio-f32 served_runs/compare_inputs/make_taco_16k.f32 `
  --output-dir served_runs/bench_generation_python_cuda_32f `
  --max-new-frames 32 `
  --warmup 1 `
  --iterations 3 `
  --device cuda `
  --thinker-active-frames 0
```

## Layout

- `scripts/export_chroma_onnx.py` exports component, shared e2e, and shared S2S bundles.
- `scripts/chroma_export/model_structure.py` contains the Python graph wrappers and flat cache I/O structure.
- `scripts/chroma_export/safetensor_weights.py` maps ONNX initializers back to Hugging Face safetensors.
- `scripts/chroma_s2s_backend.py` runs the Python Chroma comparison backend.
- `scripts/chroma_s2s_step_debug.py` dumps Python oracle tensors and compares them with F#/ONNX traces.
- `scripts/rebuild_chroma_local_external_cache.py` rebuilds the ORT 1.27-friendly local-external ONNX cache.
- `src/ChromaOnnx/ModelStructure.fs` contains shared F# contracts and graph result shapes.
- `src/ChromaOnnx/SharedWeights.fs` memory maps safetensors and registers ORT initializers.
- `src/ChromaOnnx/ChromaNativeProcessor.fs` implements tokenizer/chat-template/audio preprocessing.
- `src/ChromaOnnx/S2sOnnxRunner.fs` implements cacheful S2S ONNX inference.
- `src/ChromaOnnx/S2sService.fs` implements the local S2S browser lab and API.
- `src/ChromaOnnx/Cli.fs` contains CLI command implementations.
- `src/ChromaOnnx/Program.fs` is now only the entry point and command dispatch.

## Requirements

Main requirements for the current F#/ONNX CUDA path:

- Windows x64.
- .NET 10 SDK.
- Python 3.11 for export and comparison tooling.
- `Microsoft.ML.OnnxRuntime.Gpu` 1.27.0, already referenced by `src/ChromaOnnx/ChromaOnnx.fsproj`.
- NVIDIA driver with CUDA 13.x user-mode support. The validated setup reported `CUDA UMD Version: 13.3` in `nvidia-smi`.
- CUDA Toolkit 13.3 on `PATH`.
- cuDNN 9 for CUDA 13.3 on `PATH`; validated path:
  `C:\Program Files\NVIDIA\CUDNN\v9.23\bin\13.3\x64`.
- The Hugging Face Chroma model files under `models/chroma-4b`, including the original `.safetensors` shards.

If VS Code or the shell was opened before installing CUDA/cuDNN, prepend the paths for the current terminal:

```powershell
$env:PATH = "C:\Program Files\NVIDIA\CUDNN\v9.23\bin\13.3\x64;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;$env:PATH"
where.exe cudnn64_9.dll
where.exe cublasLt64_13.dll
where.exe cudart64_13.dll
```

Python Chroma comparison uses PyTorch CUDA. Python `onnxruntime-gpu` is not required for the F#/ONNX service path.

## Setup

Use Python 3.11 for export/comparison tooling. The repo also builds the F# runner with .NET 10.

PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-convert.txt
dotnet build src\ChromaOnnx
```

If you need to download the gated Hugging Face model, accept the model terms and provide a token:

```powershell
$env:HF_TOKEN = "hf_..."
.venv\Scripts\python.exe scripts\download_chroma.py --local-dir models/chroma-4b --min-free-gib 80
```

For metadata/code only, for example when weights were downloaded manually:

```powershell
.venv\Scripts\python.exe scripts\download_chroma.py --local-dir models/chroma-4b --skip-weights --min-free-gib 2
```

## Getting Started From Scratch

This repository does not need to include the Chroma weight shards. The large `.safetensors` files should be downloaded from the upstream Hugging Face Chroma model repo after the user accepts that model's terms. The local repo then exports a weights-free ONNX graph and uses the original safetensors through shared ORT initializers.

The quickest working setup on a new Windows machine is:

1. Install system prerequisites:
   - Windows x64.
   - NVIDIA driver with CUDA support. The validated machine used driver `610.62`.
   - CUDA Toolkit `13.3`.
   - cuDNN 9 for CUDA `13.3`.
   - .NET SDK `10`.
   - Python `3.11`.
   - Git.

2. Clone this repo:

```powershell
git clone <this-repo-url>
cd py_to_onnx
```

3. Create the Python environment and build the F# project:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements-convert.txt
dotnet build src\ChromaOnnx
```

4. Make CUDA/cuDNN visible to the current shell:

```powershell
$env:PATH = "C:\Program Files\NVIDIA\CUDNN\v9.23\bin\13.3\x64;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;$env:PATH"
where.exe cudnn64_9.dll
where.exe cublasLt64_13.dll
where.exe cudart64_13.dll
```

5. Accept the upstream Chroma model terms on Hugging Face, then download the model into `models/chroma-4b`:

```powershell
$env:HF_TOKEN = "hf_..."
.venv\Scripts\python.exe scripts\download_chroma.py `
  --local-dir models/chroma-4b `
  --min-free-gib 80
```

Expected important files under `models/chroma-4b`:

```text
config.json
generation_config.json
preprocessor_config.json
tokenizer.json
tokenizer_config.json
chat_template.jinja
model-00001-of-00003.safetensors
model-00002-of-00003.safetensors
model-00003-of-00003.safetensors
model.safetensors.index.json
```

If the safetensors were downloaded manually, place the full Hugging Face model snapshot in `models/chroma-4b`. You can then run the metadata-only downloader to fill in any missing non-weight files:

```powershell
.venv\Scripts\python.exe scripts\download_chroma.py `
  --local-dir models/chroma-4b `
  --skip-weights `
  --min-free-gib 2
```

6. Export the merged F#/ONNX S2S bundle:

```powershell
.venv\Scripts\python.exe scripts\export_chroma_onnx.py `
  --model-dir models/chroma-4b `
  --output-dir onnx/chroma-s2s-full-v2 `
  --bundle safetensor-shared-s2s `
  --device cuda `
  --dtype float32 `
  --thinker-active-frames 0 `
  --single-onnx-s2s `
  --validate
```

7. Build the local-external optimized ONNX cache used by ORT 1.27:

```powershell
.venv\Scripts\python.exe scripts\rebuild_chroma_local_external_cache.py `
  --model-dir models/chroma-4b `
  --bundle-dir onnx/chroma-s2s-full-v2 `
  --provider cuda `
  --memory-profile quality-safe
```

The cache directory contains hardlinks to the upstream safetensors when `models` and `onnx` are on the same NTFS volume. They may appear as large files in Explorer, but they should not consume another full copy of disk space.

8. Run a one-frame smoke test:

```powershell
dotnet run --project src\ChromaOnnx -- s2s-offline `
  --model-dir models/chroma-4b `
  --bundle-dir onnx/chroma-s2s-full-v2 `
  --prompt-text "War" `
  --prompt-audio-f32 served_runs/compare_inputs/reference_audio_24k.f32 `
  --user-audio-f32 served_runs/compare_inputs/make_taco_16k.f32 `
  --frames 1 `
  --output-dir served_runs/smoke/fsharp `
  --execution-provider cuda `
  --memory-mode resident-merged `
  --ort-memory-profile quality-safe `
  --thinker-active-frames 0 `
  --optimized-model-cache-dir onnx/chroma-s2s-full-v2/ort-cache-ort-local-external `
  --optimized-model-cache-format onnx
```

9. Start the local ChromaS2SONNX browser service:

```powershell
dotnet run --project src\ChromaOnnx -- s2s-serve `
  --model-dir models/chroma-4b `
  --bundle-dir onnx/chroma-s2s-full-v2 `
  --work-dir served_runs `
  --port 5055 `
  --execution-provider cuda `
  --memory-mode resident-merged `
  --ort-memory-profile quality-safe `
  --thinker-active-frames 0 `
  --stream-decode-frames 8 `
  --stream-min-free-vram-mb 1024 `
  --cuda-gpu-mem-limit-mb 15360 `
  --max-queue-length 32 `
  --max-prompt-audio-seconds 60 `
  --max-turn-audio-seconds 60 `
  --optimized-model-cache-dir onnx/chroma-s2s-full-v2/ort-cache-ort-local-external `
  --optimized-model-cache-format onnx `
  --python .venv/Scripts/python.exe `
  --python-device cuda
```

Open `http://localhost:5055`.

Notes for publishing this repo:

- Do not commit `models/chroma-4b/*.safetensors`.
- Do not commit generated `onnx/chroma-s2s-full-v2` artifacts unless you intentionally want to publish the ONNX metadata/cache.
- If publishing generated ONNX/model artifacts, include the applicable upstream Chroma notices and comply with the upstream model access terms.
- The essential reproducible source is the code, scripts, project files, and this README.
- New users should obtain the gated Chroma weights directly from the upstream Hugging Face repo using their own accepted license/token.

## Fresh Rebuild After Downloading Chroma

After downloading or refreshing the Hugging Face `FlashLabs/Chroma-4B` repo under `models/chroma-4b`, rebuild the F#/ONNX bundle and local cache in this order.

1. Export the merged safetensor-backed S2S ONNX bundle:

```powershell
.venv\Scripts\python.exe scripts\export_chroma_onnx.py `
  --model-dir models/chroma-4b `
  --output-dir onnx/chroma-s2s-full-v2 `
  --bundle safetensor-shared-s2s `
  --device cuda `
  --dtype float32 `
  --thinker-active-frames 0 `
  --single-onnx-s2s `
  --validate
```

2. Rebuild the ORT 1.27-friendly local-external optimized ONNX cache:

```powershell
.venv\Scripts\python.exe scripts\rebuild_chroma_local_external_cache.py `
  --model-dir models/chroma-4b `
  --bundle-dir onnx/chroma-s2s-full-v2 `
  --provider cuda `
  --memory-profile quality-safe
```

Expected cache output:

```text
onnx/chroma-s2s-full-v2/ort-cache-ort-local-external/
  chroma_s2s_merged.local_external.onnx
  s2s_merged.cuda.quality-safe.optimized.onnx
  local_external_cache_report.json
  model-00001-of-00003.safetensors
  model-00002-of-00003.safetensors
  model-00003-of-00003.safetensors
```

The `model-*.safetensors` files in the cache directory should be hardlinks to `models/chroma-4b`, not separate weight copies, when both directories are on the same NTFS volume. They show their full size in directory listings but do not consume another full copy of disk space.

Important: this project currently uses an optimized/local-external ONNX cache, not a serialized `.ort` model. ORT 1.27 `.ort` serialization was not reloadable for the merged external-weight Chroma graph, while the local-external ONNX cache is validated.

3. Run a CUDA smoke test:

```powershell
dotnet run --project src\ChromaOnnx -- s2s-offline `
  --model-dir models/chroma-4b `
  --bundle-dir onnx/chroma-s2s-full-v2 `
  --prompt-text "War" `
  --prompt-audio-f32 served_runs/compare_inputs/reference_audio_24k.f32 `
  --user-audio-f32 served_runs/compare_inputs/make_taco_16k.f32 `
  --frames 1 `
  --output-dir served_runs/smoke/fsharp `
  --execution-provider cuda `
  --memory-mode resident-merged `
  --ort-memory-profile quality-safe `
  --thinker-active-frames 0 `
  --optimized-model-cache-dir onnx/chroma-s2s-full-v2/ort-cache-ort-local-external `
  --optimized-model-cache-format onnx
```

## Export Bundles

### Single Merged S2S Bundle

This is the current recommended bundle for the F#/ONNX S2S service:

```powershell
.venv\Scripts\python.exe scripts\export_chroma_onnx.py `
  --model-dir models/chroma-4b `
  --output-dir onnx/chroma-s2s-full-v2 `
  --bundle safetensor-shared-s2s `
  --device cuda `
  --dtype float32 `
  --thinker-active-frames 0 `
  --single-onnx-s2s `
  --validate
```

Expected output:

```text
onnx/chroma-s2s-full-v2/
  chroma_s2s_merged.weights_free.onnx
  shared_weights_manifest.json
```

With ORT 1.27, also build the local-external optimized ONNX cache shown in the previous section. It avoids ORT external-data path validation failures by placing hardlinks to the safetensor shards beside the cached ONNX metadata file.

For a simplified visual overview of the merged dispatcher, split logical graphs, and original-vs-optimized ONNX shape, see [diagrams/onnx_s2s_simplified.md](diagrams/onnx_s2s_simplified.md).

The manifest still describes the logical graph phases:

- `generate_prefill`
- `backbone_frame_step`
- `backbone_thinker_step`
- `decoder_prefill`
- `decoder_step`
- `codec_decode`
- `s2s_merged`

The F# runner uses the merged session in `resident-merged` mode and routes each logical phase through named inputs/outputs inside that single ONNX model.

### Shared Low-Level E2E Bundle

The older one-frame e2e lab still works with the three-graph shared bundle:

```powershell
.venv\Scripts\python.exe scripts\export_chroma_onnx.py `
  --model-dir models/chroma-4b `
  --output-dir onnx/chroma-shared `
  --bundle safetensor-shared-e2e `
  --device cuda `
  --dtype float32 `
  --validate
```

This emits:

```text
chroma_system_prefill.weights_free.onnx
chroma_decoder.weights_free.onnx
chroma_codec_decode.weights_free.onnx
shared_weights_manifest.json
```

### Component Bundle

For development/debugging, you can still export separate component graphs:

```powershell
.venv\Scripts\python.exe scripts\export_chroma_onnx.py `
  --model-dir models/chroma-4b `
  --output-dir onnx/chroma-components `
  --bundle components `
  --device cuda `
  --dtype float32 `
  --sequence-length 8
```

## Run ChromaS2SONNX

Start the local ChromaS2SONNX browser lab with the merged bundle:

```powershell
dotnet run --project src\ChromaOnnx -- s2s-serve `
  --model-dir models/chroma-4b `
  --bundle-dir onnx/chroma-s2s-full-v2 `
  --work-dir served_runs `
  --port 5055 `
  --execution-provider cuda `
  --memory-mode resident-merged `
  --ort-memory-profile quality-safe `
  --thinker-active-frames 0 `
  --stream-decode-frames 4 `
  --max-queue-length 32 `
  --max-prompt-audio-seconds 60 `
  --max-turn-audio-seconds 60 `
  --optimized-model-cache-dir onnx/chroma-s2s-full-v2/ort-cache-ort-local-external `
  --optimized-model-cache-format onnx `
  --python .venv/Scripts/python.exe `
  --python-device cuda
```

Open:

```text
http://localhost:5055
```

The UI supports three backend choices:

- `F#/ONNX`: Python-free request path.
- `Python Chroma`: comparison backend only.
- `Both`: runs both and shows separate audio/details.

API shape:

- `GET /` serves the S2S browser lab.
- `GET /api/status` returns bundle readiness, memory mode, queue state, loaded ORT sessions, initializer counts, and thinker feature mode.
- `POST /api/s2s/sessions` creates a session.
- `GET /ws/s2s/{sessionId}` accepts 16 kHz Float32 PCM turn chunks and returns queue, frame, and streaming audio events.
- `GET /api/s2s/sessions/{id}/{backend}/audio.wav` returns generated WAV.
- `GET /api/s2s/sessions/{id}/{backend}/details.json` returns run metadata.

ChromaS2SONNX queues generation FIFO and runs one F#/ONNX audio-processing job at a time. Response audio streaming is available for `F#/ONNX`; the Python comparison backend remains final-result only. On CUDA, the service defaults leave VRAM headroom by capping the ORT CUDA arena and deferring partial audio decode when free VRAM is below the configured threshold.

The browser lab asks for max response seconds and converts that to Chroma audio frames at roughly 12.5 frames per second. If a run reports `stopReason: "max_frames"` or `truncatedByMaxFrames: true`, increase the response seconds; EOS means the model completed naturally.

Important WebSocket events:

- Client sends `turn.start`, binary Float32LE PCM chunks, `turn.end`, and optionally `turn.cancel`.
- Server sends `queue.enqueued`, `queue.updated`, `queue.started`, `generation.started`, `generation.frame`, `audio.chunk`, `audio.deferred`, `generation.done`, `generation.canceled`, and `error`.
- Each `audio.chunk` JSON event is followed by one binary Float32LE 24 kHz payload.

The service path expects canonical audio:

- Voice prompt audio: mono Float32 PCM at 24 kHz.
- User turn audio: mono Float32 PCM at 16 kHz.

The browser lab can upload normal browser-decodable audio and convert it client-side before sending PCM to the service.

## Offline S2S Commands

Run F#/ONNX offline:

```powershell
dotnet run --project src\ChromaOnnx -- s2s-offline `
  --model-dir models/chroma-4b `
  --bundle-dir onnx/chroma-s2s-full-v2 `
  --prompt-text "War" `
  --prompt-audio-f32 served_runs/compare_inputs/reference_audio_24k.f32 `
  --user-audio-f32 served_runs/compare_inputs/make_taco_16k.f32 `
  --frames 32 `
  --output-dir served_runs/offline/fsharp `
  --execution-provider cuda `
  --memory-mode resident-merged `
  --ort-memory-profile quality-safe `
  --thinker-active-frames 0 `
  --optimized-model-cache-dir onnx/chroma-s2s-full-v2/ort-cache-ort-local-external `
  --optimized-model-cache-format onnx
```

Run step-by-step Python/F#/ONNX parity:

```powershell
dotnet run --project src\ChromaOnnx -- s2s-compare `
  --model-dir models/chroma-4b `
  --bundle-dir onnx/chroma-s2s-full-v2 `
  --prompt-text "War" `
  --prompt-audio served_runs/compare_inputs/reference_audio_24k.wav `
  --user-audio served_runs/compare_inputs/make_taco_16k.wav `
  --output-dir served_runs/compare `
  --python .venv/Scripts/python.exe `
  --execution-provider cuda `
  --memory-mode resident-merged `
  --ort-memory-profile quality-safe `
  --frames 32 `
  --thinker-active-frames 0 `
  --optimized-model-cache-dir onnx/chroma-s2s-full-v2/ort-cache-ort-local-external `
  --optimized-model-cache-format onnx
```

Run a memory report:

```powershell
dotnet run --project src\ChromaOnnx -- s2s-memory-report `
  --model-dir models/chroma-4b `
  --bundle-dir onnx/chroma-s2s-full-v2 `
  --prompt-text "War" `
  --prompt-audio-f32 served_runs/compare_inputs/reference_audio_24k.f32 `
  --user-audio-f32 served_runs/compare_inputs/make_taco_16k.f32 `
  --frames 8 `
  --output-dir served_runs/memory `
  --python .venv/Scripts/python.exe `
  --execution-provider cuda `
  --memory-mode resident-merged `
  --ort-memory-profile quality-safe `
  --thinker-active-frames 0 `
  --optimized-model-cache-dir onnx/chroma-s2s-full-v2/ort-cache-ort-local-external `
  --optimized-model-cache-format onnx
```

## Memory Modes

- `resident-merged`: one merged ORT session, best request latency, highest steady memory. `loadedOrtSessions` should report `["s2s_merged"]`.
- `python-footprint`: lower idle memory by creating heavyweight sessions just in time, slower first request.
- `balanced`: keeps smaller sessions warm and pages heavier phases.
- `warm`: keeps separate logical graph sessions warm.

Use `resident-merged` when testing parity and interactive latency. Use `python-footprint` when investigating steady-state memory behavior.

## Legacy Low-Level Commands

Build:

```powershell
dotnet build src\ChromaOnnx
```

Inspect a component ONNX directory:

```powershell
dotnet run --project src\ChromaOnnx -- inspect --onnx-dir onnx/chroma-components
```

Run the old shared one-frame e2e bundle:

```powershell
dotnet run --project src\ChromaOnnx -- shared-e2e `
  --model-dir models/chroma-4b `
  --bundle-dir onnx/chroma-shared `
  --input-ids sample_inputs/input_ids.i64 `
  --attention-mask sample_inputs/text_attention_mask.i64 `
  --input-values sample_inputs/input_values.f32 `
  --input-values-cutoffs sample_inputs/input_values_cutoffs.i64 `
  --batch 1 `
  --text-seq 8 `
  --audio-samples 24000 `
  --output-codes sample_outputs/shared_audio_codes.i64 `
  --output-audio sample_outputs/shared_audio_values.f32
```

Serve the old one-frame web lab:

```powershell
dotnet run --project src\ChromaOnnx -- serve `
  --model-dir models/chroma-4b `
  --bundle-dir onnx/chroma-shared `
  --work-dir served_runs `
  --port 5055 `
  --python .venv/Scripts/python.exe
```

The one-frame lab is kept for low-level regression and debugging. The S2S service is the current primary path.

## Verification

Recommended checks after code changes:

```powershell
dotnet build src\ChromaOnnx
.venv\Scripts\python.exe -m compileall -q scripts
```

For runtime validation, run either:

- `s2s-offline --frames 1` for a quick merged-session smoke test.
- `s2s-compare --frames 8` or `--frames 32` for Python Chroma parity.

The latest checked parity target is deterministic greedy `model.generate(..., do_sample=False, use_cache=True, output_audio=True)` behavior for text/audio Chroma conversations.
