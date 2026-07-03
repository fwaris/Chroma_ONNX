#!/usr/bin/env python3
"""Export FlashLabs Chroma ONNX component graphs.

The full Chroma generate loop is Python control flow across a thinker model,
the audio-codebook backbone, the depth decoder, and the Mimi codec. This script
exports the tensor graphs that can be coordinated from another runtime.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

from chroma_export.dependencies import load_model
from chroma_export.model_structure import (
    BACKBONE_CACHE_PREFIX,
    DECODER_CACHE_PREFIX,
    THINKER_CACHE_PREFIX,
    build_chroma_wrappers,
    cache_dynamic_axes,
    cache_io_names,
)
from chroma_export.onnx_export import (
    check_onnx,
    export_with_torch,
    fix_onnxruntime_compat,
    install_onnx_safe_torch_ops,
    module_dtype,
    tensor_to_numpy,
    validate_onnxruntime,
    write_manifest,
)
from chroma_export.safetensor_weights import (
    S2S_REQUIRED_GRAPH_FILES,
    create_shared_bundle,
    validate_shared_onnxruntime,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/chroma-4b")
    parser.add_argument("--output-dir", default="onnx/chroma")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--dtype", default="float32", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--audio-samples", type=int, default=24000)
    parser.add_argument("--bundle", default="components", choices=["components", "safetensor-shared-e2e", "safetensor-shared-s2s"])
    parser.add_argument(
        "--thinker-active-frames",
        type=int,
        default=100,
        help="Active Whisper/thinker feature frames used for S2S export tracing; the exported batch-1 graph keeps the runtime feature mask dynamic.",
    )
    parser.add_argument(
        "--single-onnx-s2s",
        action="store_true",
        help="For safetensor-shared-s2s, keep only chroma_s2s_merged.weights_free.onnx plus the manifest.",
    )
    parser.add_argument("--skip-thinker", action="store_true")
    parser.add_argument("--skip-system-prefill", action="store_true")
    parser.add_argument("--skip-codec", action="store_true")
    parser.add_argument("--no-external-data", action="store_true", help="Disable ONNX external data files. Not recommended for Chroma-sized graphs.")
    parser.add_argument("--validate", action="store_true", help="Run onnx.checker and one ONNX Runtime inference per exported graph.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not model_dir.exists():
        raise SystemExit(f"Model directory does not exist: {model_dir}")

    torch, model = load_model(model_dir, args.device, args.dtype)
    from transformers.cache_utils import DynamicCache

    install_onnx_safe_torch_ops(torch)
    config = model.config
    hidden_size = int(config.backbone_config.hidden_size)
    audio_num_codebooks = int(config.audio_num_codebooks)
    vocab_size = int(config.backbone_config.vocab_size)
    backbone_layer_count = int(config.backbone_config.num_hidden_layers)
    backbone_kv_heads = int(config.backbone_config.num_key_value_heads)
    backbone_head_dim = int(config.backbone_config.head_dim)
    decoder_layer_count = int(config.decoder_config.num_hidden_layers)
    decoder_kv_heads = int(config.decoder_config.num_key_value_heads)
    decoder_head_dim = int(config.decoder_config.head_dim)
    thinker_text_config = config.thinker_config.text_config
    thinker_layer_count = int(thinker_text_config.num_hidden_layers)
    thinker_kv_heads = int(thinker_text_config.num_key_value_heads)
    thinker_head_dim = int(thinker_text_config.hidden_size // thinker_text_config.num_attention_heads)
    thinker_feature_size = int(config.thinker_config.audio_config.num_mel_bins)
    thinker_max_frames = int(json.loads((model_dir / "preprocessor_config.json").read_text(encoding="utf-8")).get("nb_max_frames", 30000))
    codec_dtype = module_dtype(model.codec_model, torch.float32)
    external_data = not args.no_external_data
    e2e_shared_bundle = args.bundle == "safetensor-shared-e2e"
    s2s_shared_bundle = args.bundle == "safetensor-shared-s2s"
    shared_bundle = e2e_shared_bundle or s2s_shared_bundle
    if args.thinker_active_frames < 1 or args.thinker_active_frames > thinker_max_frames:
        raise SystemExit(f"--thinker-active-frames must be between 1 and {thinker_max_frames}.")
    if args.single_onnx_s2s and not s2s_shared_bundle:
        raise SystemExit("--single-onnx-s2s is only valid with --bundle safetensor-shared-s2s.")

    if shared_bundle and (args.skip_system_prefill or args.skip_codec):
        raise SystemExit(f"--bundle {args.bundle} requires system_prefill, decoder, and codec_decode exports.")

    wrappers = build_chroma_wrappers(torch, DynamicCache)
    ThinkerTextWrapper = wrappers.ThinkerTextWrapper
    TextEmbeddingWrapper = wrappers.TextEmbeddingWrapper
    SystemPrefillWrapper = wrappers.SystemPrefillWrapper
    BackboneWrapper = wrappers.BackboneWrapper
    DecoderWrapper = wrappers.DecoderWrapper
    DecoderPrefillWrapper = wrappers.DecoderPrefillWrapper
    DecoderStepWrapper = wrappers.DecoderStepWrapper
    CodecDecodeWrapper = wrappers.CodecDecodeWrapper
    CodecEncodeWrapper = wrappers.CodecEncodeWrapper
    GeneratePrefillWrapper = wrappers.GeneratePrefillWrapper
    BackboneFrameStepWrapper = wrappers.BackboneFrameStepWrapper
    BackboneThinkerStepWrapper = wrappers.BackboneThinkerStepWrapper

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "model_dir": str(model_dir),
        "opset": args.opset,
        "dtype": args.dtype,
        "external_data": external_data,
        "hidden_size": hidden_size,
        "audio_num_codebooks": audio_num_codebooks,
        "vocab_size": vocab_size,
        "graphs": {},
    }
    validation_inputs: dict[str, dict[str, object]] = {}

    dummy_text_ids = torch.zeros(args.batch, args.sequence_length, dtype=torch.long, device=args.device)
    dummy_text_attention = torch.ones(args.batch, args.sequence_length, dtype=torch.long, device=args.device)

    if not args.skip_thinker and not shared_bundle:
        thinker_text_path = output_dir / "chroma_thinker_text.onnx"
        export_with_torch(
            torch,
            ThinkerTextWrapper(model.thinker),
            (dummy_text_ids, dummy_text_attention),
            thinker_text_path,
            ["input_ids", "attention_mask"],
            ["logits", "hidden_states"],
            {
                "input_ids": {0: "batch", 1: "text_sequence"},
                "attention_mask": {0: "batch", 1: "text_sequence"},
                "logits": {0: "batch", 1: "text_sequence"},
                "hidden_states": {0: "batch", 1: "text_sequence"},
            },
            args.opset,
            external_data,
        )
        manifest["graphs"]["thinker_text"] = {
            "path": str(thinker_text_path),
            "inputs": ["input_ids", "attention_mask"],
            "outputs": ["logits", "hidden_states"],
            "description": "Qwen2.5-Omni thinker text-only graph.",
        }
        validation_inputs["thinker_text"] = {
            "input_ids": tensor_to_numpy(dummy_text_ids),
            "attention_mask": tensor_to_numpy(dummy_text_attention),
        }
        print(f"Exported {thinker_text_path}")

        text_embeddings_path = output_dir / "chroma_text_embeddings.onnx"
        export_with_torch(
            torch,
            TextEmbeddingWrapper(model),
            (dummy_text_ids,),
            text_embeddings_path,
            ["input_ids"],
            ["embeddings"],
            {
                "input_ids": {0: "batch", 1: "text_sequence"},
                "embeddings": {0: "batch", 1: "text_sequence"},
            },
            args.opset,
            external_data,
        )
        manifest["graphs"]["text_embeddings"] = {
            "path": str(text_embeddings_path),
            "inputs": ["input_ids"],
            "outputs": ["embeddings"],
            "description": "Chroma thinker token embedding table as an ONNX graph.",
        }
        validation_inputs["text_embeddings"] = {
            "input_ids": tensor_to_numpy(dummy_text_ids),
        }
        print(f"Exported {text_embeddings_path}")

    if not args.skip_system_prefill:
        dummy_audio = torch.zeros(args.batch, 1, args.audio_samples, dtype=codec_dtype, device=args.device)
        dummy_audio_cutoffs = torch.full((args.batch,), args.audio_samples, dtype=torch.long, device=args.device)
        system_prefill_path = output_dir / "chroma_system_prefill.onnx"
        export_with_torch(
            torch,
            SystemPrefillWrapper(model),
            (dummy_text_ids, dummy_text_attention, dummy_audio, dummy_audio_cutoffs),
            system_prefill_path,
            ["input_ids", "attention_mask", "input_values", "input_values_cutoffs"],
            ["logits", "hidden_states", "backbone_attention_mask"],
            {
                "input_ids": {0: "batch", 1: "text_sequence"},
                "attention_mask": {0: "batch", 1: "text_sequence"},
                "input_values": {0: "batch", 2: "audio_samples"},
                "input_values_cutoffs": {0: "batch"},
                "logits": {0: "batch", 1: "backbone_sequence"},
                "hidden_states": {0: "batch", 1: "backbone_sequence"},
                "backbone_attention_mask": {0: "batch", 1: "backbone_sequence"},
            },
            args.opset,
            external_data,
        )
        manifest["graphs"]["system_prefill"] = {
            "path": str(system_prefill_path),
            "inputs": ["input_ids", "attention_mask", "input_values", "input_values_cutoffs"],
            "outputs": ["logits", "hidden_states", "backbone_attention_mask"],
            "description": "Prompt audio/text construction plus Chroma backbone prefill.",
        }
        validation_inputs["system_prefill"] = {
            "input_ids": tensor_to_numpy(dummy_text_ids),
            "attention_mask": tensor_to_numpy(dummy_text_attention),
            "input_values": tensor_to_numpy(dummy_audio),
            "input_values_cutoffs": tensor_to_numpy(dummy_audio_cutoffs),
        }
        print(f"Exported {system_prefill_path}")

    if s2s_shared_bundle:
        audio_placeholder_count = 25
        dummy_thinker_audio_count = 5
        dummy_thinker_sequence = max(args.sequence_length, dummy_thinker_audio_count * audio_placeholder_count + 4)
        dummy_thinker_ids = torch.zeros(args.batch, dummy_thinker_sequence, dtype=torch.long, device=args.device)
        for audio_index in range(dummy_thinker_audio_count):
            start = 1 + audio_index * audio_placeholder_count
            dummy_thinker_ids[:, start : start + audio_placeholder_count] = int(config.thinker_config.audio_token_index)
        dummy_thinker_attention = torch.ones(args.batch, dummy_thinker_sequence, dtype=torch.long, device=args.device)
        dummy_thinker_features = torch.zeros(
            dummy_thinker_audio_count,
            thinker_feature_size,
            thinker_max_frames,
            dtype=module_dtype(model.thinker, torch.float32),
            device=args.device,
        )
        dummy_thinker_feature_mask = torch.zeros(dummy_thinker_audio_count, thinker_max_frames, dtype=torch.long, device=args.device)
        dummy_thinker_feature_mask[:, : args.thinker_active_frames] = 1

        prefill_backbone_outputs = cache_io_names(BACKBONE_CACHE_PREFIX, backbone_layer_count, "present")
        prefill_thinker_outputs = cache_io_names(THINKER_CACHE_PREFIX, thinker_layer_count, "present")
        generate_prefill_outputs = [
            "logits",
            "hidden_states",
            "next_attention_mask",
            "next_thinker_input_ids",
            "next_thinker_attention_mask",
            "next_thinker_cache_position",
            "next_thinker_eos",
            *prefill_backbone_outputs,
            *prefill_thinker_outputs,
        ]
        generate_prefill_path = output_dir / "chroma_generate_prefill.onnx"
        generate_prefill_dynamic_axes = {
            "input_ids": {0: "batch", 1: "text_sequence"},
            "attention_mask": {0: "batch", 1: "text_sequence"},
            "input_values": {0: "batch", 2: "audio_samples"},
            "input_values_cutoffs": {0: "batch"},
            "thinker_input_ids": {0: "batch", 1: "thinker_prompt_sequence"},
            "thinker_attention_mask": {0: "batch", 1: "thinker_prompt_sequence"},
            "thinker_input_features": {0: "thinker_audio_items", 2: "thinker_feature_frames"},
            "thinker_feature_attention_mask": {0: "thinker_audio_items", 1: "thinker_feature_frames"},
            "logits": {0: "batch", 1: "backbone_sequence"},
            "hidden_states": {0: "batch", 1: "backbone_sequence"},
            "next_attention_mask": {0: "batch", 1: "next_backbone_sequence"},
            "next_thinker_input_ids": {0: "batch", 1: "thinker_next_sequence"},
            "next_thinker_attention_mask": {0: "batch", 1: "thinker_sequence"},
            "next_thinker_cache_position": {0: "thinker_cache_positions"},
            "next_thinker_eos": {0: "batch"},
        }
        generate_prefill_dynamic_axes.update(
            cache_dynamic_axes(BACKBONE_CACHE_PREFIX, backbone_layer_count, "present", "backbone_present_sequence")
        )
        generate_prefill_dynamic_axes.update(
            cache_dynamic_axes(THINKER_CACHE_PREFIX, thinker_layer_count, "present", "thinker_present_sequence")
        )
        export_with_torch(
            torch,
            GeneratePrefillWrapper(model),
            (
                dummy_text_ids,
                dummy_text_attention,
                dummy_audio,
                dummy_audio_cutoffs,
                dummy_thinker_ids,
                dummy_thinker_attention,
                dummy_thinker_features,
                dummy_thinker_feature_mask,
            ),
            generate_prefill_path,
            [
                "input_ids",
                "attention_mask",
                "input_values",
                "input_values_cutoffs",
                "thinker_input_ids",
                "thinker_attention_mask",
                "thinker_input_features",
                "thinker_feature_attention_mask",
            ],
            generate_prefill_outputs,
            generate_prefill_dynamic_axes,
            args.opset,
            external_data,
        )
        manifest["graphs"]["generate_prefill"] = {
            "path": str(generate_prefill_path),
            "inputs": [
                "input_ids",
                "attention_mask",
                "input_values",
                "input_values_cutoffs",
                "thinker_input_ids",
                "thinker_attention_mask",
                "thinker_input_features",
                "thinker_feature_attention_mask",
            ],
            "outputs": generate_prefill_outputs,
            "description": "Chroma generate prefill: prompt construction, first thinker injection, backbone logits, and flat caches.",
        }
        validation_inputs["generate_prefill"] = {
            "input_ids": tensor_to_numpy(dummy_text_ids),
            "attention_mask": tensor_to_numpy(dummy_text_attention),
            "input_values": tensor_to_numpy(dummy_audio),
            "input_values_cutoffs": tensor_to_numpy(dummy_audio_cutoffs),
            "thinker_input_ids": tensor_to_numpy(dummy_thinker_ids),
            "thinker_attention_mask": tensor_to_numpy(dummy_thinker_attention),
            "thinker_input_features": tensor_to_numpy(dummy_thinker_features),
            "thinker_feature_attention_mask": tensor_to_numpy(dummy_thinker_feature_mask),
        }
        print(f"Exported {generate_prefill_path}")

        backbone_cache_dtype = module_dtype(model.backbone, torch.float32)
        thinker_cache_dtype = module_dtype(model.thinker, torch.float32)
        dummy_backbone_past_sequence = 4
        dummy_thinker_past_sequence = 4
        dummy_frame_codes = torch.zeros(args.batch, 1, audio_num_codebooks, dtype=torch.long, device=args.device)
        dummy_step_attention = torch.ones(
            args.batch,
            dummy_backbone_past_sequence + 1,
            dtype=torch.float32,
            device=args.device,
        )
        dummy_backbone_cache = tuple(
            torch.zeros(
                args.batch,
                backbone_kv_heads,
                dummy_backbone_past_sequence,
                backbone_head_dim,
                dtype=backbone_cache_dtype,
                device=args.device,
            )
            for _ in range(backbone_layer_count * 2)
        )
        backbone_past_inputs = cache_io_names(BACKBONE_CACHE_PREFIX, backbone_layer_count, "past")
        backbone_present_outputs = cache_io_names(BACKBONE_CACHE_PREFIX, backbone_layer_count, "present")
        backbone_frame_step_outputs = [
            "logits",
            "hidden_states",
            "next_attention_mask",
            *backbone_present_outputs,
        ]
        backbone_frame_step_path = output_dir / "chroma_backbone_frame_step.onnx"
        backbone_frame_step_dynamic_axes = {
            "frame_codes": {0: "batch", 1: "frame_sequence"},
            "attention_mask": {0: "batch", 1: "backbone_input_sequence"},
            "logits": {0: "batch", 1: "frame_sequence"},
            "hidden_states": {0: "batch", 1: "frame_sequence"},
            "next_attention_mask": {0: "batch", 1: "backbone_next_sequence"},
        }
        backbone_frame_step_dynamic_axes.update(
            cache_dynamic_axes(BACKBONE_CACHE_PREFIX, backbone_layer_count, "past", "backbone_past_sequence")
        )
        backbone_frame_step_dynamic_axes.update(
            cache_dynamic_axes(BACKBONE_CACHE_PREFIX, backbone_layer_count, "present", "backbone_present_sequence")
        )
        export_with_torch(
            torch,
            BackboneFrameStepWrapper(model, backbone_layer_count),
            (dummy_frame_codes, dummy_step_attention, *dummy_backbone_cache),
            backbone_frame_step_path,
            ["frame_codes", "attention_mask", *backbone_past_inputs],
            backbone_frame_step_outputs,
            backbone_frame_step_dynamic_axes,
            args.opset,
            external_data,
        )
        manifest["graphs"]["backbone_frame_step"] = {
            "path": str(backbone_frame_step_path),
            "inputs": ["frame_codes", "attention_mask", *backbone_past_inputs],
            "outputs": backbone_frame_step_outputs,
            "description": "One Chroma audio-frame-only backbone generation step with flat K/V cache inputs and outputs.",
        }
        validation_inputs["backbone_frame_step"] = {
            "frame_codes": tensor_to_numpy(dummy_frame_codes),
            "attention_mask": tensor_to_numpy(dummy_step_attention),
            **{name: tensor_to_numpy(value) for name, value in zip(backbone_past_inputs, dummy_backbone_cache)},
        }
        print(f"Exported {backbone_frame_step_path}")

        dummy_thinker_step_ids = torch.zeros(args.batch, 1, dtype=torch.long, device=args.device)
        dummy_thinker_step_attention = torch.ones(args.batch, dummy_thinker_past_sequence, dtype=torch.long, device=args.device)
        dummy_thinker_cache_position = torch.arange(dummy_thinker_past_sequence, dtype=torch.long, device=args.device)
        dummy_thinker_eos = torch.zeros(args.batch, dtype=torch.long, device=args.device)
        dummy_thinker_cache = tuple(
            torch.zeros(
                args.batch,
                thinker_kv_heads,
                dummy_thinker_past_sequence,
                thinker_head_dim,
                dtype=thinker_cache_dtype,
                device=args.device,
            )
            for _ in range(thinker_layer_count * 2)
        )
        thinker_past_inputs = cache_io_names(THINKER_CACHE_PREFIX, thinker_layer_count, "past")
        thinker_present_outputs = cache_io_names(THINKER_CACHE_PREFIX, thinker_layer_count, "present")
        backbone_thinker_step_outputs = [
            "logits",
            "hidden_states",
            "next_attention_mask",
            "next_thinker_input_ids",
            "next_thinker_attention_mask",
            "next_thinker_cache_position",
            "next_thinker_eos",
            *backbone_present_outputs,
            *thinker_present_outputs,
        ]
        backbone_thinker_step_path = output_dir / "chroma_backbone_thinker_step.onnx"
        backbone_thinker_step_dynamic_axes = {
            "frame_codes": {0: "batch", 1: "frame_sequence"},
            "attention_mask": {0: "batch", 1: "backbone_input_sequence"},
            "thinker_input_ids": {0: "batch", 1: "thinker_step_sequence"},
            "thinker_attention_mask": {0: "batch", 1: "thinker_input_sequence"},
            "thinker_cache_position": {0: "thinker_input_cache_positions"},
            "thinker_eos": {0: "batch"},
            "logits": {0: "batch", 1: "backbone_step_sequence"},
            "hidden_states": {0: "batch", 1: "backbone_step_sequence"},
            "next_attention_mask": {0: "batch", 1: "backbone_next_sequence"},
            "next_thinker_input_ids": {0: "batch", 1: "thinker_next_sequence"},
            "next_thinker_attention_mask": {0: "batch", 1: "thinker_next_sequence"},
            "next_thinker_cache_position": {0: "thinker_output_cache_positions"},
            "next_thinker_eos": {0: "batch"},
        }
        backbone_thinker_step_dynamic_axes.update(
            cache_dynamic_axes(BACKBONE_CACHE_PREFIX, backbone_layer_count, "past", "backbone_past_sequence")
        )
        backbone_thinker_step_dynamic_axes.update(
            cache_dynamic_axes(BACKBONE_CACHE_PREFIX, backbone_layer_count, "present", "backbone_present_sequence")
        )
        backbone_thinker_step_dynamic_axes.update(
            cache_dynamic_axes(THINKER_CACHE_PREFIX, thinker_layer_count, "past", "thinker_past_sequence")
        )
        backbone_thinker_step_dynamic_axes.update(
            cache_dynamic_axes(THINKER_CACHE_PREFIX, thinker_layer_count, "present", "thinker_present_sequence")
        )
        backbone_thinker_inputs = [
            "frame_codes",
            "attention_mask",
            "thinker_input_ids",
            "thinker_attention_mask",
            "thinker_cache_position",
            "thinker_eos",
            *backbone_past_inputs,
            *thinker_past_inputs,
        ]
        export_with_torch(
            torch,
            BackboneThinkerStepWrapper(model, backbone_layer_count, thinker_layer_count),
            (
                dummy_frame_codes,
                dummy_step_attention,
                dummy_thinker_step_ids,
                dummy_thinker_step_attention,
                dummy_thinker_cache_position,
                dummy_thinker_eos,
                *dummy_backbone_cache,
                *dummy_thinker_cache,
            ),
            backbone_thinker_step_path,
            backbone_thinker_inputs,
            backbone_thinker_step_outputs,
            backbone_thinker_step_dynamic_axes,
            args.opset,
            external_data,
        )
        manifest["graphs"]["backbone_thinker_step"] = {
            "path": str(backbone_thinker_step_path),
            "inputs": backbone_thinker_inputs,
            "outputs": backbone_thinker_step_outputs,
            "description": "One Chroma backbone step that appends an audio frame and injects the next thinker token pair.",
        }
        validation_inputs["backbone_thinker_step"] = {
            "frame_codes": tensor_to_numpy(dummy_frame_codes),
            "attention_mask": tensor_to_numpy(dummy_step_attention),
            "thinker_input_ids": tensor_to_numpy(dummy_thinker_step_ids),
            "thinker_attention_mask": tensor_to_numpy(dummy_thinker_step_attention),
            "thinker_cache_position": tensor_to_numpy(dummy_thinker_cache_position),
            "thinker_eos": tensor_to_numpy(dummy_thinker_eos),
            **{name: tensor_to_numpy(value) for name, value in zip(backbone_past_inputs, dummy_backbone_cache)},
            **{name: tensor_to_numpy(value) for name, value in zip(thinker_past_inputs, dummy_thinker_cache)},
        }
        print(f"Exported {backbone_thinker_step_path}")

    if not shared_bundle:
        dummy_embeddings = torch.zeros(
            args.batch,
            args.sequence_length,
            hidden_size,
            dtype=next(model.backbone.parameters()).dtype,
            device=args.device,
        )
        dummy_attention = torch.ones(args.batch, args.sequence_length, dtype=torch.long, device=args.device)
        backbone_path = output_dir / "chroma_backbone.onnx"
        export_with_torch(
            torch,
            BackboneWrapper(model.backbone),
            (dummy_embeddings, dummy_attention),
            backbone_path,
            ["input_embeddings", "attention_mask"],
            ["logits", "hidden_states"],
            {
                "input_embeddings": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "sequence"},
                "hidden_states": {0: "batch", 1: "sequence"},
            },
            args.opset,
            external_data,
        )
        manifest["graphs"]["backbone"] = {
            "path": str(backbone_path),
            "inputs": ["input_embeddings", "attention_mask"],
            "outputs": ["logits", "hidden_states"],
        }
        validation_inputs["backbone"] = {
            "input_embeddings": tensor_to_numpy(dummy_embeddings),
            "attention_mask": tensor_to_numpy(dummy_attention),
        }
        print(f"Exported {backbone_path}")

    dummy_input_ids = torch.zeros(args.batch, 1, dtype=torch.long, device=args.device)
    dummy_hidden = torch.zeros(
        args.batch,
        hidden_size,
        dtype=next(model.decoder.parameters()).dtype,
        device=args.device,
    )
    dummy_decoder_prefill_attention_mask = torch.ones(args.batch, 2, dtype=torch.long, device=args.device)
    dummy_decoder_prefill_cache_position = torch.arange(0, 1, dtype=torch.long, device=args.device)
    decoder_path = output_dir / "chroma_decoder.onnx"
    export_with_torch(
        torch,
        DecoderWrapper(model.decoder),
        (dummy_input_ids, dummy_hidden),
        decoder_path,
        ["input_ids", "backbone_last_hidden_state"],
        ["logits"],
        {
            "input_ids": {0: "batch", 1: "decoder_sequence"},
            "backbone_last_hidden_state": {0: "batch"},
            "logits": {0: "batch", 1: "decoder_sequence_plus_prompt"},
        },
        args.opset,
        external_data,
    )
    manifest["graphs"]["decoder"] = {
        "path": str(decoder_path),
        "inputs": ["input_ids", "backbone_last_hidden_state"],
        "outputs": ["logits"],
    }
    validation_inputs["decoder"] = {
        "input_ids": tensor_to_numpy(dummy_input_ids),
        "backbone_last_hidden_state": tensor_to_numpy(dummy_hidden),
    }
    print(f"Exported {decoder_path}")

    decoder_cache_dtype = module_dtype(model.decoder, torch.float32)
    decoder_present_outputs = cache_io_names(DECODER_CACHE_PREFIX, decoder_layer_count, "present")
    decoder_prefill_outputs = ["logits", *decoder_present_outputs]
    decoder_prefill_path = output_dir / "chroma_decoder_prefill.onnx"
    decoder_prefill_dynamic_axes = {
        "input_ids": {0: "batch", 1: "decoder_sequence"},
        "backbone_last_hidden_state": {0: "batch"},
        "attention_mask": {0: "batch", 1: "decoder_attention_sequence"},
        "cache_position": {0: "decoder_sequence"},
        "logits": {0: "batch", 1: "decoder_prefill_sequence"},
    }
    decoder_prefill_dynamic_axes.update(
        cache_dynamic_axes(DECODER_CACHE_PREFIX, decoder_layer_count, "present", "decoder_present_sequence")
    )
    export_with_torch(
        torch,
        DecoderPrefillWrapper(model.decoder),
        (dummy_input_ids, dummy_hidden, dummy_decoder_prefill_attention_mask, dummy_decoder_prefill_cache_position),
        decoder_prefill_path,
        ["input_ids", "backbone_last_hidden_state", "attention_mask", "cache_position"],
        decoder_prefill_outputs,
        decoder_prefill_dynamic_axes,
        args.opset,
        external_data,
    )
    manifest["graphs"]["decoder_prefill"] = {
        "path": str(decoder_prefill_path),
        "inputs": ["input_ids", "backbone_last_hidden_state", "attention_mask", "cache_position"],
        "outputs": decoder_prefill_outputs,
        "description": "Cacheful Chroma depth decoder prefill for codebook-0 plus backbone hidden state.",
    }
    validation_inputs["decoder_prefill"] = {
        "input_ids": tensor_to_numpy(dummy_input_ids),
        "backbone_last_hidden_state": tensor_to_numpy(dummy_hidden),
        "attention_mask": tensor_to_numpy(dummy_decoder_prefill_attention_mask),
        "cache_position": tensor_to_numpy(dummy_decoder_prefill_cache_position),
    }
    print(f"Exported {decoder_prefill_path}")

    dummy_decoder_past_sequence = 2
    dummy_decoder_step_attention_mask = torch.ones(args.batch, dummy_decoder_past_sequence + 1, dtype=torch.long, device=args.device)
    dummy_decoder_step_cache_position = torch.arange(
        dummy_decoder_past_sequence - 1,
        dummy_decoder_past_sequence,
        dtype=torch.long,
        device=args.device,
    )
    dummy_decoder_cache = tuple(
        torch.zeros(
            args.batch,
            decoder_kv_heads,
            dummy_decoder_past_sequence,
            decoder_head_dim,
            dtype=decoder_cache_dtype,
            device=args.device,
        )
        for _ in range(decoder_layer_count * 2)
    )
    decoder_past_inputs = cache_io_names(DECODER_CACHE_PREFIX, decoder_layer_count, "past")
    decoder_step_outputs = ["logits", *decoder_present_outputs]
    decoder_step_path = output_dir / "chroma_decoder_step.onnx"
    decoder_step_dynamic_axes = {
        "input_ids": {0: "batch", 1: "decoder_step_sequence"},
        "attention_mask": {0: "batch", 1: "decoder_step_attention_sequence"},
        "cache_position": {0: "decoder_step_sequence"},
        "logits": {0: "batch", 1: "decoder_step_sequence"},
    }
    decoder_step_dynamic_axes.update(
        cache_dynamic_axes(DECODER_CACHE_PREFIX, decoder_layer_count, "past", "decoder_past_sequence")
    )
    decoder_step_dynamic_axes.update(
        cache_dynamic_axes(DECODER_CACHE_PREFIX, decoder_layer_count, "present", "decoder_present_sequence")
    )
    export_with_torch(
        torch,
        DecoderStepWrapper(model.decoder, decoder_layer_count),
        (dummy_input_ids, dummy_decoder_step_attention_mask, dummy_decoder_step_cache_position, *dummy_decoder_cache),
        decoder_step_path,
        ["input_ids", "attention_mask", "cache_position", *decoder_past_inputs],
        decoder_step_outputs,
        decoder_step_dynamic_axes,
        args.opset,
        external_data,
    )
    manifest["graphs"]["decoder_step"] = {
        "path": str(decoder_step_path),
        "inputs": ["input_ids", "attention_mask", "cache_position", *decoder_past_inputs],
        "outputs": decoder_step_outputs,
        "description": "One cacheful Chroma depth decoder step for subsequent audio codebooks.",
    }
    validation_inputs["decoder_step"] = {
        "input_ids": tensor_to_numpy(dummy_input_ids),
        "attention_mask": tensor_to_numpy(dummy_decoder_step_attention_mask),
        "cache_position": tensor_to_numpy(dummy_decoder_step_cache_position),
        **{name: tensor_to_numpy(value) for name, value in zip(decoder_past_inputs, dummy_decoder_cache)},
    }
    print(f"Exported {decoder_step_path}")

    if not args.skip_codec:
        dummy_codes = torch.zeros(
            args.batch,
            audio_num_codebooks,
            max(1, args.sequence_length),
            dtype=torch.long,
            device=args.device,
        )
        codec_decode_path = output_dir / "chroma_codec_decode.onnx"
        export_with_torch(
            torch,
            CodecDecodeWrapper(model.codec_model),
            (dummy_codes,),
            codec_decode_path,
            ["audio_codes"],
            ["audio_values"],
            {
                "audio_codes": {0: "batch", 2: "frames"},
                "audio_values": {0: "batch", 2: "samples"},
            },
            args.opset,
            external_data,
        )
        manifest["graphs"]["codec_decode"] = {
            "path": str(codec_decode_path),
            "inputs": ["audio_codes"],
            "outputs": ["audio_values"],
        }
        validation_inputs["codec_decode"] = {
            "audio_codes": tensor_to_numpy(dummy_codes),
        }
        print(f"Exported {codec_decode_path}")

        if not shared_bundle:
            dummy_audio = torch.zeros(args.batch, 1, args.audio_samples, dtype=codec_dtype, device=args.device)
            codec_encode_path = output_dir / "chroma_codec_encode.onnx"
            export_with_torch(
                torch,
                CodecEncodeWrapper(model.codec_model),
                (dummy_audio,),
                codec_encode_path,
                ["input_values"],
                ["audio_codes"],
                {
                    "input_values": {0: "batch", 2: "samples"},
                    "audio_codes": {0: "batch", 2: "frames"},
                },
                args.opset,
                external_data,
            )
            manifest["graphs"]["codec_encode"] = {
                "path": str(codec_encode_path),
                "inputs": ["input_values"],
                "outputs": ["audio_codes"],
            }
            validation_inputs["codec_encode"] = {
                "input_values": tensor_to_numpy(dummy_audio),
            }
            print(f"Exported {codec_encode_path}")

    if shared_bundle:
        write_manifest(output_dir, manifest)
        del model
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

        if s2s_shared_bundle:
            capabilities = {
                "mode": "s2s_greedy",
                "python_request_path": False,
                "ready": True,
                "required_graphs": sorted(S2S_REQUIRED_GRAPH_FILES),
                "implemented_graphs": sorted(S2S_REQUIRED_GRAPH_FILES),
                "single_onnx_model": bool(args.single_onnx_s2s),
                "thinker_feature_mode": "dynamic_batch1_multi_audio_full_length",
                "thinker_max_audio_items": int(dummy_thinker_audio_count),
                "thinker_trace_active_frames": int(args.thinker_active_frames),
                "thinker_max_frames": int(thinker_max_frames),
                "note": (
                    "Safetensor-backed S2S graph bundle for greedy Chroma generation. "
                    "Python is intended only for offline comparison."
                ),
            }
            shared_manifest = create_shared_bundle(
                output_dir,
                model_dir,
                manifest,
                bundle_name="safetensor-shared-s2s",
                graph_files=S2S_REQUIRED_GRAPH_FILES,
                capabilities=capabilities,
                single_model_only=args.single_onnx_s2s,
            )
        else:
            shared_manifest = create_shared_bundle(output_dir, model_dir, manifest)
        if args.validate:
            validate_shared_onnxruntime(shared_manifest, model_dir, validation_inputs)
        print(f"Wrote {output_dir / 'shared_weights_manifest.json'}")
        return 0

    if args.validate:
        for graph_name, graph in manifest["graphs"].items():
            path = Path(graph["path"])
            fix_onnxruntime_compat(path)
            check_onnx(path)
            print(f"Checked {path.name}")
            shapes = validate_onnxruntime(path, validation_inputs[graph_name])
            print(f"ONNX Runtime {graph_name} outputs: {shapes}")

    write_manifest(output_dir, manifest)
    print(f"Wrote {output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
