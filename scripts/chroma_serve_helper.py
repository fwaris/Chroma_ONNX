#!/usr/bin/env python3
"""Preprocess and compare requests for the local Chroma F#/ONNX server."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Create raw tensor files from prompt text and audio.")
    prepare.add_argument("--model-dir", required=True)
    prepare.add_argument("--prompt-text", required=True)
    prepare.add_argument("--prompt-audio", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--sample-rate", type=int, default=24000)
    prepare.add_argument("--max-text-tokens", type=int, default=256)

    compare = subparsers.add_parser("compare", help="Run Python Chroma low-level e2e and compare ONNX output files.")
    compare.add_argument("--model-dir", required=True)
    compare.add_argument("--prepared-dir", required=True)
    compare.add_argument("--onnx-codes", required=True)
    compare.add_argument("--onnx-audio", required=True)
    compare.add_argument("--output-dir", required=True)
    compare.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    return parser.parse_args()


def print_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2), flush=True)


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    import torch
    import torchaudio

    audio, original_sample_rate = torchaudio.load(str(path))
    if audio.ndim != 2:
        raise ValueError(f"Expected audio tensor [channels, samples], got shape {tuple(audio.shape)}")
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    audio = audio.squeeze(0)
    if int(original_sample_rate) != sample_rate:
        audio = torchaudio.functional.resample(audio, int(original_sample_rate), sample_rate)
    return audio.detach().cpu().numpy().astype(np.float32)


def prepare(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    prompt_audio = Path(args.prompt_audio).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    encoded = tokenizer(
        args.prompt_text,
        add_special_tokens=False,
        return_tensors="np",
        truncation=True,
        max_length=args.max_text_tokens,
    )
    input_ids = encoded["input_ids"].astype(np.int64)
    attention_mask = encoded["attention_mask"].astype(np.int64)
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"Expected tokenized prompt shape [1, text_seq], got {input_ids.shape}")
    if input_ids.shape[1] < 1:
        raise ValueError("prompt_text must produce at least one token.")

    audio = load_audio(prompt_audio, args.sample_rate)
    input_values = audio.reshape(1, 1, audio.shape[0]).astype(np.float32)
    input_values_cutoffs = np.array([audio.shape[0]], dtype=np.int64)

    input_ids_path = output_dir / "input_ids.i64"
    attention_mask_path = output_dir / "attention_mask.i64"
    input_values_path = output_dir / "input_values.f32"
    input_values_cutoffs_path = output_dir / "input_values_cutoffs.i64"

    input_ids.tofile(input_ids_path)
    attention_mask.tofile(attention_mask_path)
    input_values.tofile(input_values_path)
    input_values_cutoffs.tofile(input_values_cutoffs_path)

    metadata = {
        "batch": int(input_ids.shape[0]),
        "text_seq": int(input_ids.shape[1]),
        "audio_samples": int(input_values.shape[2]),
        "sample_rate": int(args.sample_rate),
        "input_ids": str(input_ids_path),
        "attention_mask": str(attention_mask_path),
        "input_values": str(input_values_path),
        "input_values_cutoffs": str(input_values_cutoffs_path),
        "prompt_audio": str(prompt_audio),
        "prompt_text": args.prompt_text,
    }
    (output_dir / "prepared.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print_json(metadata)
    return 0


def tensor_from_file(path: Path, dtype, shape: tuple[int, ...]):
    return np.fromfile(path, dtype=dtype).reshape(shape)


def compare(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForCausalLM
    from chroma_export.onnx_export import configure_quality_safe_float32

    configure_quality_safe_float32(torch)

    model_dir = Path(args.model_dir).resolve()
    prepared_dir = Path(args.prepared_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared = json.loads((prepared_dir / "prepared.json").read_text(encoding="utf-8"))
    batch = int(prepared["batch"])
    text_seq = int(prepared["text_seq"])
    audio_samples = int(prepared["audio_samples"])

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    input_ids_np = tensor_from_file(Path(prepared["input_ids"]), np.int64, (batch, text_seq))
    attention_mask_np = tensor_from_file(Path(prepared["attention_mask"]), np.int64, (batch, text_seq))
    input_values_np = tensor_from_file(Path(prepared["input_values"]), np.float32, (batch, 1, audio_samples))
    input_values_cutoffs_np = tensor_from_file(Path(prepared["input_values_cutoffs"]), np.int64, (batch,))

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).eval().to(device)

    audio_num_codebooks = int(model.config.audio_num_codebooks)
    input_ids = torch.from_numpy(input_ids_np).to(device=device, dtype=torch.long)
    attention_mask = torch.from_numpy(attention_mask_np).to(device=device, dtype=torch.long)
    input_values = torch.from_numpy(input_values_np).to(device=device, dtype=next(model.codec_model.parameters()).dtype)
    input_values_cutoffs = torch.from_numpy(input_values_cutoffs_np).to(device=device, dtype=torch.long)

    with torch.no_grad():
        input_embeddings, backbone_attention_mask = model._build_prompt_embeds(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_values=input_values,
            input_values_cutoffs=input_values_cutoffs,
        )
        backbone_outputs = model.backbone(
            input_embeddings=input_embeddings,
            attention_mask=backbone_attention_mask,
            use_cache=False,
            output_hidden_states=True,
            output_attentions=False,
        )
        logits = backbone_outputs.logits
        hidden_states = backbone_outputs.hidden_states[-1]
        first_ids = logits[:, -1, :].argmax(dim=-1)
        hidden = hidden_states[:, -1, :]
        ids = first_ids.reshape(batch, 1)

        for _ in range(1, audio_num_codebooks):
            decoder_outputs = model.decoder(
                input_ids=ids,
                backbone_last_hidden_state=hidden,
                use_cache=False,
                output_hidden_states=False,
                output_attentions=False,
            )
            next_ids = decoder_outputs.logits[:, -1, :].argmax(dim=-1)
            ids = torch.cat([ids, next_ids.reshape(batch, 1)], dim=1)

        python_codes = ids.reshape(batch, audio_num_codebooks, 1).detach().cpu().numpy().astype(np.int64)
        audio_values = model.codec_model.decode(torch.from_numpy(python_codes).to(device=device, dtype=torch.long)).audio_values
        python_audio = audio_values.detach().cpu().numpy().astype(np.float32)

    python_codes_path = output_dir / "python_audio_codes.i64"
    python_audio_path = output_dir / "python_audio_values.f32"
    python_codes.tofile(python_codes_path)
    python_audio.tofile(python_audio_path)

    onnx_codes = np.fromfile(args.onnx_codes, dtype=np.int64).reshape(python_codes.shape)
    onnx_audio = np.fromfile(args.onnx_audio, dtype=np.float32).reshape(python_audio.shape)
    audio_diff = np.abs(onnx_audio - python_audio)

    payload = {
        "device": device,
        "python_codes": str(python_codes_path),
        "python_audio": str(python_audio_path),
        "python_shapes": {
            "logits": list(logits.shape),
            "hidden_states": list(hidden_states.shape),
            "audio_codes": list(python_codes.shape),
            "audio_values": list(python_audio.shape),
        },
        "codes_equal": bool(np.array_equal(onnx_codes, python_codes)),
        "python_codes_flat": python_codes.reshape(-1).astype(int).tolist(),
        "onnx_codes_flat": onnx_codes.reshape(-1).astype(int).tolist(),
        "audio_max_abs_diff": float(audio_diff.max()),
        "audio_mean_abs_diff": float(audio_diff.mean()),
        "audio_allclose_1e_4": bool(np.allclose(onnx_audio, python_audio, atol=1e-4, rtol=1e-4)),
    }
    (output_dir / "comparison.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print_json(payload)
    return 0


def main() -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    if args.command == "prepare":
        return prepare(args)
    if args.command == "compare":
        return compare(args)
    raise ValueError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
