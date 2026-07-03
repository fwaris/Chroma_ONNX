"""Dependency loading and model construction for Chroma export."""

from __future__ import annotations

from pathlib import Path


def require_python_deps():
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except Exception as exc:  # pragma: no cover - only used before dependency install
        raise SystemExit(
            "Missing conversion dependencies. Install with:\n"
            "  pip install -r requirements.txt\n"
            f"Original import error: {exc}"
        ) from exc
    return torch, AutoModelForCausalLM


def require_safetensors():
    try:
        from safetensors import safe_open
    except Exception as exc:  # pragma: no cover - only used before dependency install
        raise SystemExit(
            "Missing dependency 'safetensors'. Install with:\n"
            "  pip install -r requirements.txt\n"
            f"Original import error: {exc}"
        ) from exc
    return safe_open


def torch_dtype(torch, name: str):
    if name == "auto":
        return "auto"
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(name)


def load_model(model_dir: Path, device: str, dtype_name: str):
    torch, AutoModelForCausalLM = require_python_deps()
    from chroma_export.onnx_export import configure_quality_safe_float32

    configure_quality_safe_float32(torch)
    dtype = torch_dtype(torch, dtype_name)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)
    return torch, model
