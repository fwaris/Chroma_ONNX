from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm persistent Python Chroma S2S generation benchmark.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--prompt-audio-f32", required=True)
    parser.add_argument("--user-audio-f32", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-frames", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--thinker-active-frames", type=int, default=0, help="0 keeps Python Chroma's full active audio length; positive values clamp/pad to that many Whisper frames.")
    return parser.parse_args()


def bytes_to_gb(value: int | None) -> float | None:
    if value is None:
        return None
    return round(float(value) / 1024.0 / 1024.0 / 1024.0, 3)


def windows_memory_info() -> tuple[int | None, int | None]:
    if os.name != "nt":
        return None, None

    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(counters)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    if not ok:
        return None, None
    return int(counters.WorkingSetSize), int(counters.PrivateUsage)


def process_memory_bytes() -> tuple[int | None, int | None]:
    try:
        import psutil

        info = psutil.Process(os.getpid()).memory_info()
        private = getattr(info, "private", None)
        if private is None:
            private = getattr(info, "vms", None)
        return int(info.rss), int(private) if private is not None else None
    except Exception:
        return windows_memory_info()


def memory_snapshot(torch_module=None) -> dict:
    working, private = process_memory_bytes()
    snapshot = {
        "processId": os.getpid(),
        "workingSetGb": bytes_to_gb(working),
        "privateGb": bytes_to_gb(private),
    }

    if torch_module is not None and torch_module.cuda.is_available():
        try:
            snapshot["gpuAllocatedGb"] = bytes_to_gb(int(torch_module.cuda.memory_allocated()))
            snapshot["gpuReservedGb"] = bytes_to_gb(int(torch_module.cuda.memory_reserved()))
            snapshot["gpuMaxAllocatedGb"] = bytes_to_gb(int(torch_module.cuda.max_memory_allocated()))
            snapshot["gpuMaxReservedGb"] = bytes_to_gb(int(torch_module.cuda.max_memory_reserved()))
        except Exception:
            snapshot["gpuAllocatedGb"] = None
            snapshot["gpuReservedGb"] = None
            snapshot["gpuMaxAllocatedGb"] = None
            snapshot["gpuMaxReservedGb"] = None

    return snapshot


def load_f32(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.float32)
    if data.size == 0:
        raise ValueError(f"{path} contained no Float32 PCM samples.")
    return np.ascontiguousarray(data)


def write_wav(path: Path, sample_rate: int, samples: np.ndarray) -> None:
    from scipy.io import wavfile

    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sample_rate, np.ascontiguousarray(samples.astype(np.float32, copy=False)))


def write_pcm16_wav(path: Path, sample_rate: int, samples: np.ndarray) -> None:
    from scipy.io import wavfile

    flat = samples.reshape(-1)
    peak = float(np.max(np.abs(flat))) if flat.size else 0.0
    gain = min(80.0, 0.85 / peak) if 1e-8 < peak < 0.5 else 1.0
    pcm = np.clip(flat.astype(np.float64) * gain, -1.0, 1.0)
    pcm = np.round(pcm * 32767.0).astype(np.int16)
    wavfile.write(path, sample_rate, pcm)


def audio_stats(sample_rate: int, samples: np.ndarray) -> dict:
    flat = samples.reshape(-1).astype(np.float64)
    if flat.size == 0:
        return {
            "samples": 0,
            "durationSeconds": 0.0,
            "peakAbs": 0.0,
            "rms": 0.0,
            "meanAbs": 0.0,
            "wavPreviewGain": 1.0,
            "wavPeakAbs": 0.0,
        }

    peak = float(np.max(np.abs(flat)))
    gain = min(80.0, 0.85 / peak) if 1e-8 < peak < 0.5 else 1.0
    return {
        "samples": int(flat.size),
        "durationSeconds": float(flat.size / sample_rate),
        "peakAbs": peak,
        "rms": float(math.sqrt(np.mean(flat * flat))),
        "meanAbs": float(np.mean(np.abs(flat))),
        "wavPreviewGain": float(gain),
        "wavPeakAbs": float(min(1.0, peak * gain)),
    }


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "minMs": None, "maxMs": None, "meanMs": None, "medianMs": None}
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[mid]
    else:
        median = (ordered[mid - 1] + ordered[mid]) / 2.0
    return {
        "count": len(values),
        "minMs": ordered[0],
        "maxMs": ordered[-1],
        "meanMs": float(sum(values) / len(values)),
        "medianMs": float(median),
    }


def to_audio_array(audio_tensor) -> np.ndarray:
    audio = audio_tensor.detach().cpu().contiguous().numpy().astype(np.float32)
    if audio.ndim == 1:
        return audio.reshape(1, 1, audio.shape[0])
    if audio.ndim == 2:
        return audio.reshape(1, audio.shape[0], audio.shape[1])
    if audio.ndim == 3:
        return audio
    raise ValueError(f"Expected audio tensor with 1-3 dimensions, got {audio.shape}.")


def main() -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()

    if args.max_new_frames < 1:
        raise ValueError("--max-new-frames must be positive.")
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative.")
    if args.iterations < 1:
        raise ValueError("--iterations must be positive.")

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    from chroma_export.onnx_export import configure_quality_safe_float32

    configure_quality_safe_float32(torch)

    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    prompt_audio = load_f32(Path(args.prompt_audio_f32).resolve())
    user_audio = load_f32(Path(args.user_audio_f32).resolve())
    if args.thinker_active_frames > 0:
        target_samples = args.thinker_active_frames * 160
        fixed_user_audio = np.zeros(target_samples, dtype=np.float32)
        copy_count = min(target_samples, user_audio.size)
        fixed_user_audio[:copy_count] = user_audio[:copy_count]
        user_audio_for_processor = fixed_user_audio
    else:
        user_audio_for_processor = user_audio

    prompt_wav = output_dir / "prompt_audio_24k.wav"
    user_wav = output_dir / "user_audio_16k.wav"
    write_wav(prompt_wav, 24000, prompt_audio)
    write_wav(user_wav, 16000, user_audio_for_processor)

    snapshots: dict[str, dict] = {"start": memory_snapshot(torch)}
    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).eval()
    model.to(device)
    if device == "cuda":
        torch.cuda.synchronize()
    load_ms = (time.perf_counter() - load_started) * 1000.0
    snapshots["afterLoad"] = memory_snapshot(torch)

    conversation = [
        [
            {"role": "system", "content": [{"type": "text", "text": args.system_prompt}]},
            {"role": "user", "content": [{"type": "audio", "audio": str(user_wav)}]},
        ]
    ]

    prepared = processor(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
        prompt_audio=[str(prompt_wav)],
        prompt_text=[args.prompt_text],
    )
    prepared = {key: value.to(device) if hasattr(value, "to") else value for key, value in prepared.items()}
    if device == "cuda":
        torch.cuda.synchronize()
    snapshots["afterPrepare"] = memory_snapshot(torch)

    def run_generate():
        if device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **prepared,
                do_sample=False,
                use_cache=True,
                max_new_tokens=max(1, int(args.max_new_frames)),
                output_audio=True,
                return_dict_in_generate=True,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return output, elapsed_ms

    for warmup_index in range(args.warmup):
        warmup_output, warmup_ms = run_generate()
        print(f"Warmup {warmup_index + 1}/{args.warmup}: {warmup_ms:.3f} ms", flush=True)
        del warmup_output

    snapshots["afterWarmup"] = memory_snapshot(torch)

    iterations = []
    last_output = None
    for index in range(args.iterations):
        if last_output is not None:
            del last_output
        output, elapsed_ms = run_generate()
        last_output = output
        codes_shape = list(output.sequences.shape)
        audio_shape = list(output.audio[0].shape) if getattr(output, "audio", None) is not None else []
        iterations.append(
            {
                "iteration": index + 1,
                "generateMs": elapsed_ms,
                "frameCount": int(output.sequences.shape[1]) if len(codes_shape) == 3 else None,
                "audioCodesShapePython": codes_shape,
                "audioValuesShapePython": audio_shape,
            }
        )
        print(f"Iteration {index + 1}/{args.iterations}: {elapsed_ms:.3f} ms", flush=True)

    snapshots["afterBenchmark"] = memory_snapshot(torch)

    if last_output is None:
        raise RuntimeError("No measured output was produced.")

    codes_b_t_c = last_output.sequences.detach().cpu().contiguous().numpy().astype(np.int64)
    if codes_b_t_c.ndim != 3:
        raise ValueError(f"Expected generated code shape [B,T,C], got {codes_b_t_c.shape}.")
    codes_b_c_t = np.ascontiguousarray(np.transpose(codes_b_t_c, (0, 2, 1)))
    audio_b_c_t = np.ascontiguousarray(to_audio_array(last_output.audio[0]).astype(np.float32, copy=False))

    codes_path = output_dir / "last_audio_codes.i64"
    raw_audio_path = output_dir / "last_audio_values.f32"
    wav_path = output_dir / "last_audio.wav"
    codes_b_c_t.tofile(codes_path)
    audio_b_c_t.tofile(raw_audio_path)
    wav = audio_stats(24000, audio_b_c_t)
    write_pcm16_wav(wav_path, 24000, audio_b_c_t)

    generate_values = [float(item["generateMs"]) for item in iterations]
    report = {
        "backend": "python_chroma",
        "mode": "s2s_benchmark",
        "generationOnly": True,
        "loadTimeExcluded": True,
        "warmupExcluded": True,
        "device": device,
        "requestedFrames": int(args.max_new_frames),
        "warmupIterations": int(args.warmup),
        "measuredIterations": int(args.iterations),
        "promptAudioSamples": int(prompt_audio.size),
        "userAudioSamples": int(user_audio.size),
        "effectiveUserAudioSamples": int(user_audio_for_processor.size),
        "thinkerActiveFrames": int(args.thinker_active_frames),
        "audioCodesShape": list(codes_b_c_t.shape),
        "audioValuesShape": list(audio_b_c_t.shape),
        "summary": {
            "generateMs": summarize(generate_values),
        },
        "iterations": iterations,
        "timingsMs": {
            "loadMsExcluded": load_ms,
        },
        "memory": snapshots,
        "wav": wav,
        "artifacts": {
            "lastAudioCodes": str(codes_path),
            "lastAudioValues": str(raw_audio_path),
            "lastAudioWav": str(wav_path),
        },
    }
    (output_dir / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
