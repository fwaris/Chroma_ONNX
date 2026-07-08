param(
    [string]$ModelDir = "",
    [string]$OutputDir = "",
    [string]$ReferenceWav = "",
    [string]$ReferenceName = "reference",
    [ValidateSet("q4f16", "q4", "fp16", "fp32")]
    [string]$LanguageModelVariant = "q4f16",
    [ValidateSet("cuda", "cpu")]
    [string]$ExecutionProvider = "cuda",
    [int]$MaxNewTokens = 128,
    [double]$Exaggeration = 0.5,
    [string]$AsrPython = "",
    [switch]$SkipAsr,
    [switch]$IncludeSouthernBelle
)

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptRoot

function Resolve-FirstExisting([string[]]$Candidates, [string]$Label) {
    foreach ($candidate in $Candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {
            return [System.IO.Path]::GetFullPath($candidate)
        }
    }
    throw "$Label was not found. Tried: $($Candidates -join ', ')"
}

if (-not $ModelDir) {
    $ModelDir = Resolve-FirstExisting @(
        "E:\s\temp\VoiceAgent_assets\models\chatterbox-onnx",
        (Join-Path $RepoRoot "VoiceAgent_assets\models\chatterbox-onnx"),
        (Join-Path $RepoRoot "models\chatterbox-onnx")
    ) "Chatterbox ONNX model directory"
} else {
    $ModelDir = [System.IO.Path]::GetFullPath($ModelDir)
}

if (-not $OutputDir) {
    $baseTemp = if (Test-Path -LiteralPath "E:\s\temp") { "E:\s\temp" } else { [System.IO.Path]::GetTempPath() }
    $OutputDir = Join-Path $baseTemp ("chatterbox_onnx_parity_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
}
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

function Assert-PathExists([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Label was not found: $Path"
    }
}

Assert-PathExists $ModelDir "Chatterbox ONNX model directory"
Assert-PathExists (Join-Path $ModelDir "tokenizer.json") "Chatterbox tokenizer"
Assert-PathExists (Join-Path $ModelDir "default_voice.wav") "Chatterbox default voice"
Assert-PathExists (Join-Path $ModelDir "onnx\speech_encoder.onnx") "Chatterbox speech encoder"
Assert-PathExists (Join-Path $ModelDir "onnx\speech_encoder.onnx_data") "Chatterbox speech encoder external data"
Assert-PathExists (Join-Path $ModelDir "onnx\embed_tokens.onnx") "Chatterbox embed tokens"
Assert-PathExists (Join-Path $ModelDir "onnx\embed_tokens.onnx_data") "Chatterbox embed tokens external data"
Assert-PathExists (Join-Path $ModelDir "onnx\conditional_decoder.onnx") "Chatterbox conditional decoder"
Assert-PathExists (Join-Path $ModelDir "onnx\conditional_decoder.onnx_data") "Chatterbox conditional decoder external data"

$lmFile = switch ($LanguageModelVariant) {
    "q4f16" { "language_model_q4f16.onnx" }
    "q4" { "language_model_q4.onnx" }
    "fp16" { "language_model_fp16.onnx" }
    "fp32" { "language_model.onnx" }
}
$lmDataFile = "$lmFile`_data"
Assert-PathExists (Join-Path $ModelDir "onnx\$lmFile") "Chatterbox language model"
Assert-PathExists (Join-Path $ModelDir "onnx\$lmDataFile") "Chatterbox language model external data"

if (-not $AsrPython -and -not $SkipAsr) {
    $candidate = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $candidate) {
        $AsrPython = [System.IO.Path]::GetFullPath($candidate)
    }
}

$cases = @(
    [pscustomobject]@{
        name = "default_hello"
        reference = Join-Path $ModelDir "default_voice.wav"
        text = "Hello, this is a quick test."
    },
    [pscustomobject]@{
        name = "default_taco"
        reference = Join-Path $ModelDir "default_voice.wav"
        text = "Mixing seasoned meat with toppings in a warm tortilla is the basic method."
    },
    [pscustomobject]@{
        name = "default_agent_answer"
        reference = Join-Path $ModelDir "default_voice.wav"
        text = "Making tacos is simple. First, prepare your filling. Warm the tortillas, add the seasoned filling, and finish with your favorite toppings."
    }
)

if ($ReferenceWav) {
    $ReferenceWav = [System.IO.Path]::GetFullPath($ReferenceWav)
    Assert-PathExists $ReferenceWav "Reference WAV"
    $cases += @(
        [pscustomobject]@{
            name = "$ReferenceName`_hello"
            reference = $ReferenceWav
            text = "Hello, this is a quick test."
        },
        [pscustomobject]@{
            name = "$ReferenceName`_taco"
            reference = $ReferenceWav
            text = "Mixing seasoned meat with toppings in a warm tortilla is the basic method."
        }
    )
}

if ($IncludeSouthernBelle) {
    $southernBelleMp3 = Join-Path $RepoRoot "assets\southern_belle.mp3"
    if (Test-Path -LiteralPath $southernBelleMp3) {
        $southernBelleWav = Join-Path $OutputDir "southern_belle.wav"
        $ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
        if (-not $ffmpeg) {
            Write-Warning "ffmpeg was not found; skipping southern_belle reference."
        } else {
            & $ffmpeg.Source -y -hide_banner -loglevel error -i $southernBelleMp3 -ac 1 -ar 24000 $southernBelleWav
            if ($LASTEXITCODE -ne 0) {
                throw "ffmpeg failed while converting southern_belle.mp3"
            }
            $cases += @(
                [pscustomobject]@{
                    name = "southern_belle_hello"
                    reference = $southernBelleWav
                    text = "Hello, this is a quick test."
                },
                [pscustomobject]@{
                    name = "southern_belle_taco"
                    reference = $southernBelleWav
                    text = "Mixing seasoned meat with toppings in a warm tortilla is the basic method."
                }
            )
        }
    } else {
        Write-Warning "southern_belle.mp3 was not found at $southernBelleMp3"
    }
}

$casesPath = Join-Path $OutputDir "cases.json"
$cases | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $casesPath -Encoding UTF8

$pythonPath = if ($AsrPython -and (Test-Path -LiteralPath $AsrPython)) {
    $AsrPython
} else {
    Resolve-FirstExisting @(
        (Join-Path $RepoRoot ".venv\Scripts\python.exe"),
        "python"
    ) "Python"
}

$runnerPath = Join-Path $OutputDir "_run_chatterbox_onnx.py"
@'
import argparse
import json
import os
import re
import time
import warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
warnings.filterwarnings("ignore")

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
from transformers import AutoTokenizer
from transformers.utils import logging as transformers_logging

transformers_logging.set_verbosity_error()

S3GEN_SR = 24000
START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562


class RepetitionPenaltyLogitsProcessor:
    def __init__(self, penalty: float):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` must be a strictly positive float, but is {penalty}")
        self.penalty = penalty

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        score = np.take_along_axis(scores, input_ids, axis=1)
        score = np.where(score < 0, score * self.penalty, score / self.penalty)
        scores_processed = scores.copy()
        np.put_along_axis(scores_processed, input_ids, score, axis=1)
        return scores_processed


def dtype_from_ort(type_name: str):
    if type_name == "tensor(float16)":
        return np.float16
    if type_name == "tensor(double)":
        return np.float64
    return np.float32


def shape_dim(shape, index, fallback):
    if index >= len(shape):
        return fallback
    value = shape[index]
    return value if isinstance(value, int) and value > 0 else fallback


def wav_stats(path: Path):
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.size == 0:
        peak = rms = mean_abs = 0.0
    else:
        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
        mean_abs = float(np.mean(np.abs(audio)))
    return {
        "sampleRate": int(sr),
        "samples": int(audio.shape[0]),
        "durationSeconds": round(float(audio.shape[0]) / float(sr), 3) if sr else 0.0,
        "peakAbs": round(peak, 6),
        "rms": round(rms, 6),
        "meanAbs": round(mean_abs, 6),
        "bytes": path.stat().st_size,
    }


def make_session(path: Path, providers):
    return ort.InferenceSession(str(path), providers=providers)


def generate_case(case, sessions, tokenizer, args):
    speech_encoder_session, embed_tokens_session, lm_session, decoder_session = sessions
    output_path = Path(args.output_dir) / f"{case['name']}.wav"
    text = case["text"].strip()
    ref_path = case["reference"]
    audio_values, _ = librosa.load(ref_path, sr=S3GEN_SR)
    audio_values = audio_values[np.newaxis, :].astype(np.float32)

    input_ids = tokenizer(text, return_tensors="np")["input_ids"].astype(np.int64)
    position_ids = np.where(
        input_ids >= START_SPEECH_TOKEN,
        0,
        np.arange(input_ids.shape[1], dtype=np.int64)[np.newaxis, :] - 1,
    ).astype(np.int64)
    embed_inputs = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "exaggeration": np.array([args.exaggeration], dtype=np.float32),
    }

    past_inputs = [i for i in lm_session.get_inputs() if i.name.startswith("past_key_values.")]
    if not past_inputs:
        raise RuntimeError("Language model has no past_key_values inputs; this smoke expects the exported cache graph.")
    first_past = past_inputs[0]
    cache_dtype = dtype_from_ort(first_past.type)
    batch_size = 1
    num_key_value_heads = shape_dim(first_past.shape, 1, 16)
    head_dim = shape_dim(first_past.shape, 3, 64)
    past_key_names = [i.name for i in past_inputs]

    repetition_penalty = RepetitionPenaltyLogitsProcessor(penalty=1.2)
    generated = np.array([[START_SPEECH_TOKEN]], dtype=np.int64)

    start = time.perf_counter()
    stopped_on_stop_token = False
    prompt_token = None
    ref_x_vector = None
    prompt_feat = None
    attention_mask = None
    past_key_values = None
    tokens_generated = 0

    for i in range(args.max_new_tokens):
        inputs_embeds = embed_tokens_session.run(None, embed_inputs)[0]
        if i == 0:
            cond_emb, prompt_token, ref_x_vector, prompt_feat = speech_encoder_session.run(
                None, {"audio_values": audio_values}
            )
            inputs_embeds = np.concatenate((cond_emb, inputs_embeds), axis=1)
            batch_size, seq_len, _ = inputs_embeds.shape
            past_key_values = {
                name: np.zeros([batch_size, num_key_value_heads, 0, head_dim], dtype=cache_dtype)
                for name in past_key_names
            }
            attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)

        lm_inputs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            **past_key_values,
        }
        outputs = lm_session.run(None, lm_inputs)
        logits, present_key_values = outputs[0], outputs[1:]
        logits = logits[:, -1, :]
        next_logits = repetition_penalty(generated, logits)
        next_token = np.argmax(next_logits, axis=-1, keepdims=True).astype(np.int64)
        generated = np.concatenate((generated, next_token), axis=-1)
        tokens_generated += 1

        if (next_token.flatten() == STOP_SPEECH_TOKEN).all():
            stopped_on_stop_token = True
            break

        embed_inputs["input_ids"] = next_token
        embed_inputs["position_ids"] = np.full((input_ids.shape[0], 1), i + 1, dtype=np.int64)
        attention_mask = np.concatenate([attention_mask, np.ones((batch_size, 1), dtype=np.int64)], axis=1)
        for j, key in enumerate(past_key_names):
            past_key_values[key] = present_key_values[j]

    speech_tokens = generated[:, 1:]
    if speech_tokens.shape[1] > 0 and int(speech_tokens[0, -1]) == STOP_SPEECH_TOKEN:
        speech_tokens = speech_tokens[:, :-1]
    if prompt_token is not None:
        speech_tokens = np.concatenate([prompt_token, speech_tokens], axis=1)

    wav = decoder_session.run(
        None,
        {
            "speech_tokens": speech_tokens,
            "speaker_embeddings": ref_x_vector,
            "speaker_features": prompt_feat,
        },
    )[0]
    wav = np.squeeze(wav)
    sf.write(str(output_path), wav, S3GEN_SR)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    stats = wav_stats(output_path)
    rtf = (elapsed_ms / 1000.0) / max(0.001, stats["durationSeconds"])
    return {
        "name": case["name"],
        "text": text,
        "reference": ref_path,
        "outputWav": str(output_path),
        "tokensGenerated": tokens_generated,
        "stoppedOnStopToken": stopped_on_stop_token,
        "inferenceMs": round(elapsed_ms, 3),
        "rtf": round(rtf, 3),
        "wav": stats,
    }


def run_asr(results, python_asr_enabled):
    if not python_asr_enabled:
        return
    from transformers import pipeline

    pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")
    for result in results:
        try:
            result["asrText"] = pipe(result["outputWav"]).get("text", "").strip()
        except Exception as exc:
            result["asrText"] = f"ASR_ERROR: {exc}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--variant", default="q4f16")
    parser.add_argument("--provider", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--exaggeration", type=float, default=0.5)
    parser.add_argument("--asr", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    onnx_dir = model_dir / "onnx"
    lm_file = {
        "q4f16": "language_model_q4f16.onnx",
        "q4": "language_model_q4.onnx",
        "fp16": "language_model_fp16.onnx",
        "fp32": "language_model.onnx",
    }[args.variant]
    providers = ["CPUExecutionProvider"]
    if args.provider == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    sessions = (
        make_session(onnx_dir / "speech_encoder.onnx", providers),
        make_session(onnx_dir / "embed_tokens.onnx", providers),
        make_session(onnx_dir / lm_file, providers),
        make_session(onnx_dir / "conditional_decoder.onnx", providers),
    )
    provider_report = {
        "speechEncoder": sessions[0].get_providers(),
        "embedTokens": sessions[1].get_providers(),
        "languageModel": sessions[2].get_providers(),
        "conditionalDecoder": sessions[3].get_providers(),
        "availableProviders": ort.get_available_providers(),
    }
    with open(args.cases, "r", encoding="utf-8-sig") as f:
        cases = json.load(f)

    results = []
    for case in cases:
        print(f"Running {case['name']}...", flush=True)
        results.append(generate_case(case, sessions, tokenizer, args))
    run_asr(results, args.asr)

    report = {
        "modelDir": str(model_dir),
        "variant": args.variant,
        "provider": args.provider,
        "providers": provider_report,
        "maxNewTokens": args.max_new_tokens,
        "exaggeration": args.exaggeration,
        "results": results,
    }
    report_path = Path(args.output_dir) / "chatterbox_onnx_parity_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {report_path}")
    for result in results:
        asr = result.get("asrText", "(ASR skipped)")
        print(
            f"{result['name']}: duration={result['wav']['durationSeconds']}s "
            f"rms={result['wav']['rms']} rtf={result['rtf']} "
            f"tokens={result['tokensGenerated']} stop={result['stoppedOnStopToken']} "
            f"asr='{asr}'"
        )


if __name__ == "__main__":
    main()
'@ | Set-Content -LiteralPath $runnerPath -Encoding UTF8

$arguments = @(
    $runnerPath,
    "--model-dir", $ModelDir,
    "--output-dir", $OutputDir,
    "--cases", $casesPath,
    "--variant", $LanguageModelVariant,
    "--provider", $ExecutionProvider,
    "--max-new-tokens", [string]$MaxNewTokens,
    "--exaggeration", ([string]::Format([System.Globalization.CultureInfo]::InvariantCulture, "{0}", $Exaggeration))
)
if (-not $SkipAsr -and $AsrPython) {
    $arguments += "--asr"
}

Write-Host "Running Chatterbox ONNX parity smoke..."
Write-Host "ModelDir: $ModelDir"
Write-Host "OutputDir: $OutputDir"
Write-Host "Variant: $LanguageModelVariant"
Write-Host "ExecutionProvider: $ExecutionProvider"
& $pythonPath @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Chatterbox ONNX parity smoke failed with exit code $LASTEXITCODE"
}
