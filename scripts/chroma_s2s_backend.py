from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Python Chroma S2S generation for the local F#/ONNX comparison service.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument("--system-prompt", default="You are Chroma, an advanced virtual human created by the FlashLabs. You possess the ability to understand auditory inputs and generate both text and speech.")
    parser.add_argument("--prompt-audio-f32", required=True)
    parser.add_argument("--user-audio-f32", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-frames", type=int, default=25)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--thinker-active-frames", type=int, default=0, help="0 keeps Python Chroma's full active audio length; positive values clamp/pad to that many Whisper frames.")
    return parser.parse_args()


def load_f32(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.float32)
    if data.size == 0:
        raise ValueError(f"{path} contained no Float32 PCM samples.")
    return np.ascontiguousarray(data)


def write_wav(path: Path, sample_rate: int, samples: np.ndarray) -> None:
    from scipy.io import wavfile

    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sample_rate, np.ascontiguousarray(samples.astype(np.float32, copy=False)))


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


def write_pcm16_wav(path: Path, sample_rate: int, samples: np.ndarray) -> None:
    from scipy.io import wavfile

    flat = samples.reshape(-1)
    peak = float(np.max(np.abs(flat))) if flat.size else 0.0
    gain = min(80.0, 0.85 / peak) if 1e-8 < peak < 0.5 else 1.0
    pcm = np.clip(flat.astype(np.float64) * gain, -1.0, 1.0)
    pcm = np.round(pcm * 32767.0).astype(np.int16)
    wavfile.write(path, sample_rate, pcm)


def main() -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()

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

    conversation = [
        [
            {"role": "system", "content": [{"type": "text", "text": args.system_prompt}]},
            {"role": "user", "content": [{"type": "audio", "audio": str(user_wav)}]},
        ]
    ]

    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).eval()
    model.to(device)
    load_ms = (time.perf_counter() - started) * 1000.0

    prepared = processor(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
        prompt_audio=[str(prompt_wav)],
        prompt_text=[args.prompt_text],
    )
    prepared = {key: value.to(device) if hasattr(value, "to") else value for key, value in prepared.items()}

    generate_started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **prepared,
            do_sample=False,
            use_cache=True,
            max_new_tokens=max(1, int(args.max_new_frames)),
            output_audio=True,
            return_dict_in_generate=True,
        )
    generate_ms = (time.perf_counter() - generate_started) * 1000.0

    codes_b_t_c = output.sequences.detach().cpu().contiguous().numpy().astype(np.int64)
    if codes_b_t_c.ndim != 3:
        raise ValueError(f"Expected generated code shape [B,T,C], got {codes_b_t_c.shape}.")
    codes_b_c_t = np.ascontiguousarray(np.transpose(codes_b_t_c, (0, 2, 1)))

    audio_tensor = output.audio[0].detach().cpu().contiguous().numpy().astype(np.float32)
    if audio_tensor.ndim == 1:
        audio_b_c_t = audio_tensor.reshape(1, 1, audio_tensor.shape[0])
    elif audio_tensor.ndim == 2:
        audio_b_c_t = audio_tensor.reshape(1, audio_tensor.shape[0], audio_tensor.shape[1])
    elif audio_tensor.ndim == 3:
        audio_b_c_t = audio_tensor
    else:
        raise ValueError(f"Expected audio tensor with 1-3 dimensions, got {audio_tensor.shape}.")
    audio_b_c_t = np.ascontiguousarray(audio_b_c_t.astype(np.float32, copy=False))

    codes_path = output_dir / "audio_codes.i64"
    raw_audio_path = output_dir / "audio_values.f32"
    wav_path = output_dir / "audio.wav"
    codes_b_c_t.tofile(codes_path)
    audio_b_c_t.tofile(raw_audio_path)
    wav = audio_stats(24000, audio_b_c_t)
    write_pcm16_wav(wav_path, 24000, audio_b_c_t)

    frame_count = int(codes_b_c_t.shape[2])
    stop_reason = "eos" if frame_count < int(args.max_new_frames) else "max_frames"
    total_ms = (time.perf_counter() - started) * 1000.0
    details = {
        "backend": "python",
        "mode": "python_chroma_generate",
        "device": device,
        "pythonInRequestPath": True,
        "promptAudioSamples": int(prompt_audio.size),
        "userAudioSamples": int(user_audio.size),
        "effectiveUserAudioSamples": int(user_audio_for_processor.size),
        "thinkerActiveFrames": int(args.thinker_active_frames),
        "frameCount": frame_count,
        "stopReason": stop_reason,
        "audioCodesShape": list(codes_b_c_t.shape),
        "audioValuesShape": list(audio_b_c_t.shape),
        "timingsMs": {
            "loadMs": load_ms,
            "generateMs": generate_ms,
            "totalMs": total_ms,
        },
        "wav": wav,
        "artifacts": {
            "audioCodes": str(codes_path),
            "audioValues": str(raw_audio_path),
            "audioWav": str(wav_path),
        },
    }
    (output_dir / "details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(json.dumps(details, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
