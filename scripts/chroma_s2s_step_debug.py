from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.cache_utils import DynamicCache

from chroma_export.onnx_export import configure_quality_safe_float32, install_onnx_safe_torch_ops


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump Chroma S2S Python intermediate tensors and compare to F#/ONNX.")
    parser.add_argument("--model-dir")
    parser.add_argument("--prompt-text")
    parser.add_argument("--system-prompt", default="You are Chroma, an advanced virtual human created by the FlashLabs. You possess the ability to understand auditory inputs and generate both text and speech.")
    parser.add_argument("--prompt-audio")
    parser.add_argument("--user-audio")
    parser.add_argument("--output-dir")
    parser.add_argument("--onnx-debug-dir")
    parser.add_argument("--python-debug-dir")
    parser.add_argument("--python-manifest", default="debug_manifest.json")
    parser.add_argument("--onnx-manifest", default="debug_manifest.json")
    parser.add_argument("--compare-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--thinker-active-frames", type=int, help="Override active thinker feature frames in the mask.")
    parser.add_argument("--frames", type=int, default=8, help="Number of greedy S2S frames to include in the step trace.")
    parser.add_argument(
        "--onnx-safe-torch-ops",
        action="store_true",
        help="Use the same PyTorch math shims that the ONNX exporter uses before collecting the trace.",
    )
    return parser.parse_args()


def write_tensor(root: Path, manifest: dict, name: str, tensor: torch.Tensor | np.ndarray) -> None:
    if isinstance(tensor, torch.Tensor):
        array = tensor.detach().cpu().contiguous().numpy()
    else:
        array = np.ascontiguousarray(tensor)

    if array.dtype in (np.int64, np.int32, np.uint64, np.uint32):
        array = array.astype(np.int64, copy=False)
        dtype = "i64"
        suffix = "i64"
    else:
        array = array.astype(np.float32, copy=False)
        dtype = "f32"
        suffix = "f32"

    file_name = f"{name}.{suffix}"
    array.tofile(root / file_name)
    manifest["tensors"][name] = {"file": file_name, "dtype": dtype, "shape": list(array.shape)}


def load_debug_tensor(root: Path, manifest: dict, name: str) -> np.ndarray:
    info = manifest["tensors"][name]
    dtype = np.float32 if info["dtype"] == "f32" else np.int64
    return np.fromfile(root / info["file"], dtype=dtype).reshape(info["shape"])


def finite_float_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {
        "shape": list(a.shape),
        "allclose_1e_4": bool(np.allclose(a, b, atol=1e-4, rtol=1e-4)),
        "maxAbsDiff": float(diff.max()) if diff.size else 0.0,
        "meanAbsDiff": float(diff.mean()) if diff.size else 0.0,
        "rmsDiff": float(math.sqrt(np.mean(diff * diff))) if diff.size else 0.0,
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
        "pythonPreview": a.reshape(-1)[: min(16, a.size)].astype(int).tolist(),
        "onnxPreview": b.reshape(-1)[: min(16, b.size)].astype(int).tolist(),
    }


def compare_dirs(
    python_dir: Path,
    onnx_dir: Path,
    python_manifest_name: str = "debug_manifest.json",
    onnx_manifest_name: str = "debug_manifest.json",
) -> dict:
    python_manifest = json.loads((python_dir / python_manifest_name).read_text(encoding="utf-8"))
    onnx_manifest = json.loads((onnx_dir / onnx_manifest_name).read_text(encoding="utf-8"))
    comparisons = {}
    for name, py_info in python_manifest["tensors"].items():
        if name not in onnx_manifest["tensors"]:
            continue
        onnx_info = onnx_manifest["tensors"][name]
        if py_info["shape"] != onnx_info["shape"] or py_info["dtype"] != onnx_info["dtype"]:
            comparisons[name] = {
                "compatible": False,
                "pythonShape": py_info["shape"],
                "onnxShape": onnx_info["shape"],
                "pythonDtype": py_info["dtype"],
                "onnxDtype": onnx_info["dtype"],
            }
            continue

        py = load_debug_tensor(python_dir, python_manifest, name)
        ox = load_debug_tensor(onnx_dir, onnx_manifest, name)
        comparisons[name] = int_metrics(py, ox) if py_info["dtype"] == "i64" else finite_float_metrics(py, ox)

    summary = {
        "prefillCodebook0": comparisons.get("codebook0_ids"),
        "decoderLoopFrame": comparisons.get("decoder_loop_frame_ids"),
        "decoderCacheFrame": comparisons.get("decoder_cache_frame_ids"),
        "decoderGenerateFrame": comparisons.get("decoder_generate_frame_ids"),
        "backboneFrameStepCodebook0": comparisons.get("backbone_frame_step_codebook0_ids"),
        "decoderGenerateVsPythonLoop": None,
        "comparisons": comparisons,
    }

    if "decoder_generate_frame_ids" in python_manifest["tensors"] and "decoder_loop_frame_ids" in python_manifest["tensors"]:
        generated = load_debug_tensor(python_dir, python_manifest, "decoder_generate_frame_ids")
        loop = load_debug_tensor(python_dir, python_manifest, "decoder_loop_frame_ids")
        summary["decoderGenerateVsPythonLoop"] = int_metrics(generated, loop)

    output_name = "compare_to_onnx_debug.json" if python_manifest_name == "debug_manifest.json" else "compare_to_fsharp_prepared.json"
    (python_dir / output_name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def save_prepared_inputs(output_dir: Path, prepared: dict[str, torch.Tensor]) -> None:
    prepared_dir = output_dir / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"runtime": "python_prepared_inputs", "tensors": {}}
    for name in [
        "input_ids",
        "attention_mask",
        "input_values",
        "input_values_cutoffs",
        "thinker_input_ids",
        "thinker_attention_mask",
        "thinker_input_features",
        "thinker_feature_attention_mask",
    ]:
        write_tensor(prepared_dir, manifest, name, prepared[name])
    (prepared_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.compare_only:
        if not args.python_debug_dir or not args.onnx_debug_dir:
            raise SystemExit("--compare-only requires --python-debug-dir and --onnx-debug-dir.")
        summary = compare_dirs(
            Path(args.python_debug_dir).resolve(),
            Path(args.onnx_debug_dir).resolve(),
            args.python_manifest,
            args.onnx_manifest,
        )
        print(json.dumps(summary, indent=2))
        return 0

    missing = [
        name
        for name, value in [
            ("--model-dir", args.model_dir),
            ("--prompt-text", args.prompt_text),
            ("--prompt-audio", args.prompt_audio),
            ("--user-audio", args.user_audio),
            ("--output-dir", args.output_dir),
        ]
        if not value
    ]
    if missing:
        raise SystemExit(f"Missing required argument(s): {', '.join(missing)}")

    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    user_audio_path = Path(args.user_audio).resolve()

    if args.thinker_active_frames is not None and args.thinker_active_frames > 0:
        import librosa
        from scipy.io import wavfile

        target_samples = max(1, int(args.thinker_active_frames) * 160)
        audio, _ = librosa.load(str(user_audio_path), sr=16000, mono=True)
        fixed = np.zeros(target_samples, dtype=np.float32)
        copy_count = min(target_samples, audio.shape[0])
        if copy_count > 0:
            fixed[:copy_count] = audio[:copy_count].astype(np.float32, copy=False)
        user_audio_path = output_dir / f"user_audio_{args.thinker_active_frames}_feature_frames_16k.wav"
        wavfile.write(user_audio_path, 16000, fixed)

    raw_inputs_dir = output_dir / "raw_inputs"
    raw_inputs_dir.mkdir(parents=True, exist_ok=True)
    import librosa

    prompt_audio_24k, _ = librosa.load(str(Path(args.prompt_audio).resolve()), sr=24000, mono=True)
    user_audio_16k, _ = librosa.load(str(user_audio_path), sr=16000, mono=True)
    prompt_audio_24k.astype(np.float32).tofile(raw_inputs_dir / "prompt_audio_24k.f32")
    user_audio_16k.astype(np.float32).tofile(raw_inputs_dir / "user_audio_16k.f32")

    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))

    from export_chroma_onnx import build_chroma_wrappers

    configure_quality_safe_float32(torch)
    if args.onnx_safe_torch_ops:
        install_onnx_safe_torch_ops(torch)

    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).eval()
    model.to(args.device)

    conversation = [
        [
            {"role": "system", "content": [{"type": "text", "text": args.system_prompt}]},
            {"role": "user", "content": [{"type": "audio", "audio": str(user_audio_path)}]},
        ]
    ]
    prepared_cpu = processor(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
        prompt_audio=[str(Path(args.prompt_audio).resolve())],
        prompt_text=[args.prompt_text],
    )
    if args.thinker_active_frames is not None and args.thinker_active_frames > 0:
        mask = prepared_cpu["thinker_feature_attention_mask"].clone()
        mask.zero_()
        mask[:, : args.thinker_active_frames] = 1
        prepared_cpu["thinker_feature_attention_mask"] = mask

    save_prepared_inputs(output_dir, prepared_cpu)
    prepared = {key: value.to(args.device) if hasattr(value, "to") else value for key, value in prepared_cpu.items()}

    wrappers = build_chroma_wrappers(torch, DynamicCache)
    prefill_wrapper = wrappers.GeneratePrefillWrapper(model).eval()
    debug_dir = output_dir / "python"
    debug_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"runtime": "python_torch", "tensors": {}}

    with torch.inference_mode():
        prefill = prefill_wrapper(
            prepared["input_ids"],
            prepared["attention_mask"],
            prepared["input_values"],
            prepared["input_values_cutoffs"],
            prepared["thinker_input_ids"],
            prepared["thinker_attention_mask"],
            prepared["thinker_input_features"],
            prepared["thinker_feature_attention_mask"],
        )
        logits = prefill[0]
        hidden_states = prefill[1]
        next_attention_mask = prefill[2]
        next_thinker_input_ids = prefill[3]
        next_thinker_attention_mask = prefill[4]
        next_thinker_cache_position = prefill[5]
        next_thinker_eos = prefill[6]

        write_tensor(debug_dir, manifest, "prefill_logits", logits)
        write_tensor(debug_dir, manifest, "prefill_hidden_states", hidden_states)
        write_tensor(debug_dir, manifest, "prefill_next_attention_mask", next_attention_mask)
        write_tensor(debug_dir, manifest, "prefill_next_thinker_input_ids", next_thinker_input_ids)
        write_tensor(debug_dir, manifest, "prefill_next_thinker_attention_mask", next_thinker_attention_mask)
        write_tensor(debug_dir, manifest, "prefill_next_thinker_cache_position", next_thinker_cache_position)
        write_tensor(debug_dir, manifest, "prefill_next_thinker_eos", next_thinker_eos.long())

        codebook0 = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        decoder_hidden = hidden_states[:, -1, :]
        write_tensor(debug_dir, manifest, "codebook0_ids", codebook0)
        write_tensor(debug_dir, manifest, "decoder_backbone_last_hidden_state", decoder_hidden)

        ids = codebook0
        for step_index in range(1, int(model.config.audio_num_codebooks)):
            write_tensor(debug_dir, manifest, f"decoder_loop_step_{step_index}_input_ids", ids)
            decoder_outputs = model.decoder(
                input_ids=ids,
                backbone_last_hidden_state=decoder_hidden.clone(),
                use_cache=False,
                output_hidden_states=False,
                output_attentions=False,
            )
            decoder_logits = decoder_outputs.logits
            next_ids = torch.argmax(decoder_logits[:, -1, :], dim=-1, keepdim=True)
            write_tensor(debug_dir, manifest, f"decoder_loop_step_{step_index}_logits", decoder_logits)
            write_tensor(debug_dir, manifest, f"decoder_loop_step_{step_index}_next_ids", next_ids)
            ids = torch.cat([ids, next_ids], dim=1)

        write_tensor(debug_dir, manifest, "decoder_loop_frame_ids", ids)
        cache_ids = codebook0
        write_tensor(debug_dir, manifest, "decoder_cache_step_1_input_ids", cache_ids)
        decoder_outputs = model.decoder(
            input_ids=cache_ids,
            backbone_last_hidden_state=decoder_hidden.clone(),
            attention_mask=torch.ones(
                (cache_ids.shape[0], cache_ids.shape[1] + 1),
                device=cache_ids.device,
                dtype=torch.long,
            ),
            cache_position=torch.arange(0, cache_ids.shape[1], device=cache_ids.device),
            use_cache=True,
            output_hidden_states=False,
            output_attentions=False,
        )
        decoder_logits = decoder_outputs.logits
        decoder_cache = decoder_outputs.past_key_values
        next_ids = torch.argmax(decoder_logits[:, -1, :], dim=-1, keepdim=True)
        write_tensor(debug_dir, manifest, "decoder_cache_step_1_logits", decoder_logits)
        write_tensor(debug_dir, manifest, "decoder_cache_step_1_next_ids", next_ids)
        cache_ids = torch.cat([cache_ids, next_ids], dim=1)

        for step_index in range(2, int(model.config.audio_num_codebooks)):
            step_input = next_ids
            write_tensor(debug_dir, manifest, f"decoder_cache_step_{step_index}_input_ids", step_input)
            past_seen_tokens = decoder_cache.get_seq_length()
            position_start = past_seen_tokens - step_input.shape[1]
            cache_position = torch.arange(
                position_start,
                position_start + step_input.shape[1],
                device=step_input.device,
            )
            decoder_outputs = model.decoder(
                input_ids=step_input,
                past_key_values=decoder_cache,
                attention_mask=torch.ones(
                    (step_input.shape[0], past_seen_tokens + step_input.shape[1]),
                    device=step_input.device,
                    dtype=torch.long,
                ),
                cache_position=cache_position,
                use_cache=True,
                output_hidden_states=False,
                output_attentions=False,
            )
            decoder_logits = decoder_outputs.logits
            decoder_cache = decoder_outputs.past_key_values
            next_ids = torch.argmax(decoder_logits[:, -1, :], dim=-1, keepdim=True)
            write_tensor(debug_dir, manifest, f"decoder_cache_step_{step_index}_logits", decoder_logits)
            write_tensor(debug_dir, manifest, f"decoder_cache_step_{step_index}_next_ids", next_ids)
            cache_ids = torch.cat([cache_ids, next_ids], dim=1)

        write_tensor(debug_dir, manifest, "decoder_cache_frame_ids", cache_ids)
        generated = model.decoder.generate(
            input_ids=codebook0,
            backbone_last_hidden_state=decoder_hidden.clone(),
            max_new_tokens=int(model.config.audio_num_codebooks) - 1,
            min_new_tokens=int(model.config.audio_num_codebooks) - 1,
            do_sample=False,
            use_cache=True,
        )
        write_tensor(debug_dir, manifest, "decoder_generate_frame_ids", generated)

        backbone_layer_count = int(model.config.backbone_config.num_hidden_layers)
        backbone_cache_flat = prefill[7 : 7 + backbone_layer_count * 2]
        frame_step_wrapper = wrappers.BackboneFrameStepWrapper(model, backbone_layer_count).eval()
        frame_input_ids = generated.unsqueeze(1)
        frame_step = frame_step_wrapper(frame_input_ids, next_attention_mask, *backbone_cache_flat)
        frame_logits = frame_step[0]
        frame_hidden_states = frame_step[1]
        frame_codebook0 = torch.argmax(frame_logits[:, -1, :], dim=-1, keepdim=True)
        write_tensor(debug_dir, manifest, "backbone_frame_step_input_ids", frame_input_ids)
        write_tensor(debug_dir, manifest, "backbone_frame_step_logits", frame_logits)
        write_tensor(debug_dir, manifest, "backbone_frame_step_hidden_states", frame_hidden_states)
        write_tensor(debug_dir, manifest, "backbone_frame_step_codebook0_ids", frame_codebook0)

        thinker_layer_count = int(model.config.thinker_config.text_config.num_hidden_layers)
        thinker_cache_flat = prefill[
            7 + backbone_layer_count * 2 : 7 + backbone_layer_count * 2 + thinker_layer_count * 2
        ]
        thinker_step_wrapper = wrappers.BackboneThinkerStepWrapper(
            model,
            backbone_layer_count,
            thinker_layer_count,
        ).eval()

        def decode_frame(step_logits, step_hidden_states):
            codebook0_ids = torch.argmax(step_logits[:, -1, :], dim=-1, keepdim=True)
            return model.decoder.generate(
                input_ids=codebook0_ids,
                backbone_last_hidden_state=step_hidden_states[:, -1, :].clone(),
                max_new_tokens=int(model.config.audio_num_codebooks) - 1,
                min_new_tokens=int(model.config.audio_num_codebooks) - 1,
                do_sample=False,
                use_cache=True,
            )

        trace_current = generated
        trace_input = trace_current.unsqueeze(1)
        trace_attention_mask = next_attention_mask
        trace_thinker_input_ids = next_thinker_input_ids
        trace_thinker_attention_mask = next_thinker_attention_mask
        trace_thinker_cache_position = next_thinker_cache_position
        trace_thinker_eos = next_thinker_eos.long()
        trace_backbone_cache = tuple(backbone_cache_flat)
        trace_thinker_cache = tuple(thinker_cache_flat)
        trace_use_thinker = False

        write_tensor(debug_dir, manifest, "trace_frame_0_ids", trace_current)
        write_tensor(debug_dir, manifest, "trace_frame_0_logits", logits)
        write_tensor(debug_dir, manifest, "trace_frame_0_hidden_states", hidden_states)
        write_tensor(debug_dir, manifest, "trace_state_0_attention_mask", trace_attention_mask)
        write_tensor(debug_dir, manifest, "trace_state_0_thinker_input_ids", trace_thinker_input_ids)
        write_tensor(debug_dir, manifest, "trace_state_0_thinker_attention_mask", trace_thinker_attention_mask)
        write_tensor(debug_dir, manifest, "trace_state_0_thinker_cache_position", trace_thinker_cache_position)
        write_tensor(debug_dir, manifest, "trace_state_0_thinker_eos", trace_thinker_eos)

        for frame_index in range(1, max(1, int(args.frames))):
            thinker_active = bool((trace_thinker_eos == 0).any().item())
            if trace_use_thinker and thinker_active:
                trace_use_thinker = False
                step_kind = 1
                step = thinker_step_wrapper(
                    trace_input,
                    trace_attention_mask,
                    trace_thinker_input_ids,
                    trace_thinker_attention_mask,
                    trace_thinker_cache_position,
                    trace_thinker_eos,
                    *trace_backbone_cache,
                    *trace_thinker_cache,
                )
                step_logits = step[0]
                step_hidden_states = step[1]
                trace_attention_mask = step[2]
                trace_thinker_input_ids = step[3]
                trace_thinker_attention_mask = step[4]
                trace_thinker_cache_position = step[5]
                trace_thinker_eos = step[6].long()
                trace_backbone_cache = tuple(step[7 : 7 + backbone_layer_count * 2])
                trace_thinker_cache = tuple(step[7 + backbone_layer_count * 2 :])
            else:
                trace_use_thinker = thinker_active
                step_kind = 0
                step = frame_step_wrapper(trace_input, trace_attention_mask, *trace_backbone_cache)
                step_logits = step[0]
                step_hidden_states = step[1]
                trace_attention_mask = step[2]
                trace_backbone_cache = tuple(step[3:])

            trace_current = decode_frame(step_logits, step_hidden_states)
            trace_input = trace_current.unsqueeze(1)
            write_tensor(debug_dir, manifest, f"trace_step_{frame_index}_kind", np.array([step_kind], dtype=np.int64))
            write_tensor(debug_dir, manifest, f"trace_step_{frame_index}_logits", step_logits)
            write_tensor(debug_dir, manifest, f"trace_step_{frame_index}_hidden_states", step_hidden_states)
            write_tensor(debug_dir, manifest, f"trace_frame_{frame_index}_ids", trace_current)
            write_tensor(debug_dir, manifest, f"trace_state_{frame_index}_attention_mask", trace_attention_mask)
            write_tensor(debug_dir, manifest, f"trace_state_{frame_index}_thinker_input_ids", trace_thinker_input_ids)
            write_tensor(debug_dir, manifest, f"trace_state_{frame_index}_thinker_attention_mask", trace_thinker_attention_mask)
            write_tensor(debug_dir, manifest, f"trace_state_{frame_index}_thinker_cache_position", trace_thinker_cache_position)
            write_tensor(debug_dir, manifest, f"trace_state_{frame_index}_thinker_eos", trace_thinker_eos)

    (debug_dir / "debug_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"preparedDir": str(output_dir / "prepared"), "pythonDebugDir": str(debug_dir)}, indent=2))

    if args.onnx_debug_dir:
        summary = compare_dirs(debug_dir, Path(args.onnx_debug_dir).resolve())
        print(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
