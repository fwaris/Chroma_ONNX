# Runtime Layout

Use two roots when iterating on code:

```text
Chroma_ONNX_assets/        # stable, large, rarely recopied
Chroma_ONNX_A100_runtime/  # small, replaceable publish output
```

`Chroma_ONNX_assets` contains:

- `models/chroma-4b`
- `onnx/chroma-s2s-full-v2/ort-cache-ort-local-external`
- `served_runs/compare_inputs`

`Chroma_ONNX_A100_runtime` contains:

- `app/service`
- `app/cli`
- `onnx_deploy/chroma-s2s-full-v2`
- launch scripts

Prepare the shared assets once:

```powershell
.\scripts\prepare_large_assets.ps1
```

Publish or refresh the small runtime after code changes:

```powershell
.\scripts\publish_a100_runtime.ps1
```

Repair the shared cache links from the model directory:

```powershell
.\scripts\repair_cache_links.ps1 -ModelDir E:\s\temp\Chroma_ONNX_assets\models\chroma-4b
```

The repair script treats `models/chroma-4b` as the source of truth, infers the assets root,
recreates the cache safetensor hardlinks, and validates that the optimized CUDA ONNX cache
file is present.

Run the exported runtime:

```powershell
cd E:\s\temp\Chroma_ONNX_A100_runtime
Set-ExecutionPolicy -Scope Process Bypass
.\smoke-test-a100.ps1
.\run-service-a100.ps1
```

For local repo development, keep using the existing relative defaults, or point the service/CLI at the shared assets with:

```powershell
.\scripts\smoke_local_runtime.ps1
.\scripts\run_local_service.ps1
```

Both scripts keep `onnx_deploy/chroma-s2s-full-v2` in the repo, but read `models/chroma-4b`,
the optimized cache, and smoke-test fixture audio from `Chroma_ONNX_assets`.
