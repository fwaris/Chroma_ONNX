from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from chroma_export.onnx_export import (
    configure_quality_safe_float32,
    export_with_torch,
    fix_onnxruntime_compat,
    install_onnx_safe_torch_ops,
)


class CodecEncodeDebugWrapper(torch.nn.Module):
    def __init__(self, codec_model):
        super().__init__()
        self.codec_model = codec_model

    def forward(self, input_values):
        model = self.codec_model
        num_quantizers = model.config.num_quantizers
        padding_mask = torch.ones_like(input_values).bool()
        encoder_embeddings = model.encoder(input_values)
        encoder_outputs = model.encoder_transformer(encoder_embeddings.transpose(1, 2), return_dict=False)
        transformer_embeddings = encoder_outputs[0].transpose(1, 2)
        downsample_embeddings = model.downsample(transformer_embeddings)
        codes = model.quantizer.encode(downsample_embeddings, num_quantizers).transpose(0, 1)
        return encoder_embeddings, transformer_embeddings, downsample_embeddings, codes


def write_array(path: Path, value: torch.Tensor | np.ndarray) -> dict:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous().numpy()
    else:
        array = np.ascontiguousarray(value)
    dtype = "i64" if np.issubdtype(array.dtype, np.integer) else "f32"
    suffix = "i64" if dtype == "i64" else "f32"
    array = array.astype(np.int64 if dtype == "i64" else np.float32, copy=False)
    file_name = f"{path.name}.{suffix}"
    array.tofile(path.with_name(file_name))
    return {"file": file_name, "dtype": dtype, "shape": list(array.shape)}


def float_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {
        "shape": list(a.shape),
        "allclose_1e_4": bool(np.allclose(a, b, atol=1e-4, rtol=1e-4)),
        "maxAbsDiff": float(diff.max()) if diff.size else 0.0,
        "meanAbsDiff": float(diff.mean()) if diff.size else 0.0,
        "rmsDiff": float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0,
    }


def int_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    same = a == b
    mismatch = np.argwhere(~same)
    return {
        "shape": list(a.shape),
        "equal": bool(same.all()),
        "matchCount": int(same.sum()),
        "total": int(same.size),
        "firstDiffIndex": None if mismatch.size == 0 else [int(value) for value in mismatch[0]],
        "pythonValue": None if mismatch.size == 0 else int(a[tuple(mismatch[0])]),
        "onnxValue": None if mismatch.size == 0 else int(b[tuple(mismatch[0])]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export and compare Chroma Mimi codec encode debug tensors.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input-f32", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--onnx-safe-torch-ops", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))

    configure_quality_safe_float32(torch)
    if args.onnx_safe_torch_ops:
        install_onnx_safe_torch_ops(torch)

    audio = np.fromfile(Path(args.input_f32).resolve(), dtype=np.float32)
    input_values = torch.from_numpy(audio.reshape(1, 1, -1)).to(args.device)

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).eval()
    model.to(args.device)
    wrapper = CodecEncodeDebugWrapper(model.codec_model).eval()

    names = ["encoder_embeddings", "transformer_embeddings", "downsample_embeddings", "audio_codes"]
    manifest = {"runtime": "python_torch", "tensors": {}}
    with torch.inference_mode():
        torch_outputs = wrapper(input_values)
        for name, value in zip(names, torch_outputs):
            manifest["tensors"][name] = write_array(output_dir / f"python_{name}", value)
    (output_dir / "python_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    onnx_path = output_dir / "chroma_codec_encode_debug.onnx"
    if not args.skip_export:
        export_with_torch(
            torch,
            wrapper,
            (input_values,),
            onnx_path,
            ["input_values"],
            names,
            {
                "input_values": {0: "batch", 2: "samples"},
                "encoder_embeddings": {0: "batch", 2: "encoder_steps"},
                "transformer_embeddings": {0: "batch", 2: "encoder_steps"},
                "downsample_embeddings": {0: "batch", 2: "frames"},
                "audio_codes": {0: "batch", 2: "frames"},
            },
            args.opset,
            False,
        )
        fix_onnxruntime_compat(onnx_path)

    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_outputs = session.run(None, {"input_values": input_values.detach().cpu().numpy()})
    ort_manifest = {"runtime": "onnxruntime_cpu", "tensors": {}}
    comparisons = {}
    for name, py_value, ort_value in zip(names, torch_outputs, ort_outputs):
        ort_manifest["tensors"][name] = write_array(output_dir / f"onnx_{name}", ort_value)
        py_array = py_value.detach().cpu().numpy()
        comparisons[name] = int_metrics(py_array, ort_value) if np.issubdtype(py_array.dtype, np.integer) else float_metrics(py_array, ort_value)

    (output_dir / "onnx_manifest.json").write_text(json.dumps(ort_manifest, indent=2), encoding="utf-8")
    (output_dir / "compare.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    print(json.dumps(comparisons, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
