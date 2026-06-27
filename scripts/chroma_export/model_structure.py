"""Chroma graph structure wrappers used by the ONNX exporter."""

from __future__ import annotations

from types import SimpleNamespace


BACKBONE_CACHE_PREFIX = "backbone"
THINKER_CACHE_PREFIX = "thinker"
DECODER_CACHE_PREFIX = "decoder"


def cache_io_names(prefix: str, layer_count: int, state: str) -> list[str]:
    return [
        f"{prefix}_{state}_{layer_idx}_{kind}"
        for layer_idx in range(layer_count)
        for kind in ("key", "value")
    ]


def cache_dynamic_axes(prefix: str, layer_count: int, state: str, sequence_axis_name: str) -> dict[str, dict[int, str]]:
    return {
        name: {0: "batch", 2: sequence_axis_name}
        for name in cache_io_names(prefix, layer_count, state)
    }


def build_chroma_wrappers(torch, DynamicCache):
    """Build torch.nn.Module wrappers without importing torch at module import time."""

    def install_batch1_dynamic_audio_tower(chroma):
        """Patch Qwen2.5-Omni thinker audio export to avoid trace-time Split sizes.

        The upstream batch/general path packs active audio features, converts
        tensor lengths to Python lists, then splits into tower windows. Legacy
        torch.onnx.export bakes those list lengths into Split attributes. Chroma
        S2S V1 is batch=1, so we can preserve the same windowed audio-tower math
        with a static max feature tensor [1, 128, 30000] and a dynamic mask that
        selects the real active length.
        """

        thinker = chroma.thinker
        audio_tower = thinker.audio_tower
        if getattr(audio_tower, "_chroma_batch1_dynamic_onnx_audio", False):
            return
        audio_tower.config._attn_implementation = "eager"

        def dynamic_audio_tower_forward(input_features, feature_lens=None, aftercnn_lens=None, **kwargs):
            if input_features.dim() == 3:
                features = input_features[0]
            else:
                features = input_features

            total_frames = features.shape[-1]
            window = audio_tower.n_window * 2
            if total_frames % window != 0:
                raise RuntimeError(
                    f"Dynamic Chroma audio export expects feature frames divisible by {window}, got {total_frames}."
                )

            max_chunks = total_frames // window
            feature_len = feature_lens[0].to(torch.long)
            frame_positions = torch.arange(total_frames, device=features.device, dtype=torch.long)
            valid_frames = frame_positions < feature_len
            valid_chunks = valid_frames.reshape(max_chunks, window)

            padded_feature = features.reshape(audio_tower.num_mel_bins, max_chunks, window).permute(1, 0, 2)
            active_chunk_count = torch.clamp((feature_len + window - 1) // window, min=1, max=max_chunks)
            chunk_positions = torch.arange(max_chunks, device=features.device, dtype=torch.long)
            active_chunk_indices = torch.nonzero(chunk_positions < active_chunk_count, as_tuple=False).squeeze(1)
            padded_feature = torch.index_select(padded_feature, 0, active_chunk_indices)
            valid_chunks = torch.index_select(valid_chunks, 0, active_chunk_indices)
            padded_feature = padded_feature * valid_chunks.unsqueeze(1).to(padded_feature.dtype)
            padded_mask = valid_chunks.unsqueeze(1).to(padded_feature.dtype)

            padded_embed = torch.nn.functional.gelu(audio_tower.conv1(padded_feature)) * padded_mask
            padded_embed = torch.nn.functional.gelu(audio_tower.conv2(padded_embed)).transpose(1, 2)
            padded_embed = padded_embed + audio_tower.positional_embedding.positional_embedding[
                : padded_embed.shape[1], :
            ].unsqueeze(0).to(padded_embed.dtype)

            chunk_lengths = valid_chunks.long().sum(dim=1)
            zero_lengths = torch.zeros_like(chunk_lengths)
            feature_lens_after_cnn = torch.where(chunk_lengths > 0, (chunk_lengths - 1) // 2 + 1, zero_lengths)
            cnn_positions = torch.arange(padded_embed.shape[1], device=features.device, dtype=torch.long)
            padded_mask_after_cnn = cnn_positions.unsqueeze(0) < feature_lens_after_cnn.unsqueeze(1)

            hidden_states = padded_embed
            key_attention_mask = torch.where(
                padded_mask_after_cnn[:, None, None, :],
                torch.zeros((), dtype=hidden_states.dtype, device=hidden_states.device),
                torch.full((), torch.finfo(hidden_states.dtype).min, dtype=hidden_states.dtype, device=hidden_states.device),
            )
            for encoder_layer in audio_tower.layers:
                residual = hidden_states
                layer_states = encoder_layer.self_attn_layer_norm(hidden_states)
                seq_length = layer_states.shape[1]
                attention = encoder_layer.self_attn
                query_states = attention.q_proj(layer_states).reshape(
                    layer_states.shape[0], seq_length, attention.num_heads, attention.head_dim
                )
                key_states = attention.k_proj(layer_states).reshape(
                    layer_states.shape[0], seq_length, attention.num_heads, attention.head_dim
                )
                value_states = attention.v_proj(layer_states).reshape(
                    layer_states.shape[0], seq_length, attention.num_heads, attention.head_dim
                )
                query_states = query_states.transpose(1, 2)
                key_states = key_states.transpose(1, 2)
                value_states = value_states.transpose(1, 2)
                attention_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * attention.scaling
                attention_weights = attention_weights + key_attention_mask[:, :, :, : key_states.shape[-2]]
                attention_weights = torch.nn.functional.softmax(attention_weights, dim=-1, dtype=torch.float32).to(
                    query_states.dtype
                )
                attention_output = torch.matmul(attention_weights, value_states)
                attention_output = attention_output.transpose(1, 2).contiguous().reshape(
                    layer_states.shape[0], seq_length, -1
                )
                attention_output = attention.out_proj(attention_output)
                hidden_states = residual + attention_output
                residual = hidden_states
                hidden_states = encoder_layer.final_layer_norm(hidden_states)
                hidden_states = encoder_layer.fc1(hidden_states)
                hidden_states = encoder_layer.activation_fn(hidden_states)
                hidden_states = encoder_layer.fc2(hidden_states)
                hidden_states = residual + hidden_states
                hidden_states = hidden_states * padded_mask_after_cnn.unsqueeze(-1).to(hidden_states.dtype)

            audio_state_len = aftercnn_lens[0].to(torch.long)
            flat_hidden_states = hidden_states.reshape(hidden_states.shape[0] * hidden_states.shape[1], hidden_states.shape[2])
            flat_valid = padded_mask_after_cnn.reshape(hidden_states.shape[0] * hidden_states.shape[1])
            valid_indices = torch.nonzero(flat_valid, as_tuple=False).squeeze(1)
            hidden_states = torch.index_select(flat_hidden_states, 0, valid_indices)
            audio_positions = torch.arange(hidden_states.shape[0], device=hidden_states.device, dtype=torch.long)
            hidden_states = hidden_states[audio_positions < audio_state_len]
            token_audio = audio_tower.avg_pooler(hidden_states.transpose(0, 1)).transpose(0, 1)
            token_audio = audio_tower.ln_post(token_audio)
            token_audio = audio_tower.proj(token_audio)
            return SimpleNamespace(last_hidden_state=token_audio)

        def dynamic_get_audio_features(input_features, feature_attention_mask=None, audio_feature_lengths=None):
            if feature_attention_mask is not None:
                audio_feature_lengths = torch.sum(feature_attention_mask, dim=1).to(torch.long)
            elif audio_feature_lengths is None:
                audio_feature_lengths = torch.full(
                    (input_features.shape[0],),
                    input_features.shape[-1],
                    dtype=torch.long,
                    device=input_features.device,
                )
            else:
                audio_feature_lengths = audio_feature_lengths.to(torch.long)

            audio_feat_lengths, _audio_output_lengths = thinker.audio_tower._get_feat_extract_output_lengths(
                audio_feature_lengths
            )
            audio_outputs = thinker.audio_tower(
                input_features,
                feature_lens=audio_feature_lengths,
                aftercnn_lens=audio_feat_lengths,
            )
            return audio_outputs.last_hidden_state

        audio_tower.forward = dynamic_audio_tower_forward
        thinker.get_audio_features = dynamic_get_audio_features
        audio_tower._chroma_batch1_dynamic_onnx_audio = True

    def cache_from_flat(flat_cache, cache_config):
        pairs = []
        for index in range(0, len(flat_cache), 2):
            pairs.append((flat_cache[index], flat_cache[index + 1]))
        return DynamicCache(ddp_cache_data=pairs, config=cache_config)

    def flatten_cache(cache):
        flat = []
        for layer in cache.layers:
            flat.append(layer.keys)
            flat.append(layer.values)
        return tuple(flat)

    class ThinkerTextWrapper(torch.nn.Module):
        def __init__(self, thinker):
            super().__init__()
            self.thinker = thinker

        def forward(self, input_ids, attention_mask):
            outputs = self.thinker(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                output_attentions=False,
                return_dict=True,
                use_audio_in_video=False,
            )
            return outputs.logits, outputs.hidden_states[-1]

    class TextEmbeddingWrapper(torch.nn.Module):
        def __init__(self, chroma):
            super().__init__()
            self.chroma = chroma
            install_batch1_dynamic_audio_tower(chroma)

        def forward(self, input_ids):
            return self.chroma._embed_text_tokens(input_ids)

    class SystemPrefillWrapper(torch.nn.Module):
        def __init__(self, chroma):
            super().__init__()
            self.chroma = chroma

        def forward(self, input_ids, attention_mask, input_values, input_values_cutoffs):
            input_embeddings, backbone_attention_mask = self.chroma._build_prompt_embeds(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_values=input_values,
                input_values_cutoffs=input_values_cutoffs,
            )
            outputs = self.chroma.backbone(
                input_embeddings=input_embeddings,
                attention_mask=backbone_attention_mask,
                use_cache=False,
                output_hidden_states=True,
                output_attentions=False,
            )
            return outputs.logits, outputs.hidden_states[-1], backbone_attention_mask

    class BackboneWrapper(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, input_embeddings, attention_mask):
            outputs = self.backbone(
                input_embeddings=input_embeddings,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                output_attentions=False,
            )
            return outputs.logits, outputs.hidden_states[-1]

    class DecoderWrapper(torch.nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder

        def forward(self, input_ids, backbone_last_hidden_state):
            outputs = self.decoder(
                input_ids=input_ids,
                backbone_last_hidden_state=backbone_last_hidden_state,
                use_cache=False,
                output_hidden_states=False,
                output_attentions=False,
            )
            return outputs.logits

    class DecoderPrefillWrapper(torch.nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder
            self.decoder.model.config._attn_implementation = "eager"

        def _attention_mask_4d(self, input_ids, attention_mask, dtype):
            batch = input_ids.shape[0]
            query_length = input_ids.shape[1] + 1
            query_positions = torch.arange(query_length, device=input_ids.device).view(1, 1, query_length, 1)
            key_positions = torch.arange(query_length, device=input_ids.device).view(1, 1, 1, query_length)
            causal = key_positions <= query_positions
            valid = attention_mask[:, None, None, :query_length].to(torch.bool)
            allowed = causal & valid
            zeros = torch.zeros((batch, 1, query_length, query_length), device=input_ids.device, dtype=dtype)
            blocked = torch.full_like(zeros, torch.finfo(dtype).min)
            return torch.where(allowed.expand(batch, 1, query_length, query_length), zeros, blocked)

        def forward(self, input_ids, backbone_last_hidden_state, attention_mask, cache_position):
            attention_mask_4d = self._attention_mask_4d(
                input_ids,
                attention_mask,
                backbone_last_hidden_state.dtype,
            )
            outputs = self.decoder(
                input_ids=input_ids,
                backbone_last_hidden_state=backbone_last_hidden_state,
                attention_mask=attention_mask_4d,
                cache_position=cache_position,
                use_cache=True,
                output_hidden_states=False,
                output_attentions=False,
            )
            return (outputs.logits, *flatten_cache(outputs.past_key_values))

    class DecoderStepWrapper(torch.nn.Module):
        def __init__(self, decoder, decoder_cache_layer_count):
            super().__init__()
            self.decoder = decoder
            self.decoder_cache_layer_count = decoder_cache_layer_count
            self.decoder.model.config._attn_implementation = "eager"

        def _attention_mask_4d(self, input_ids, attention_mask, dtype):
            batch = input_ids.shape[0]
            query_length = input_ids.shape[1]
            key_length = attention_mask.shape[1]
            valid = attention_mask[:, None, None, :key_length].to(torch.bool)
            zeros = torch.zeros((batch, 1, query_length, key_length), device=input_ids.device, dtype=dtype)
            blocked = torch.full_like(zeros, torch.finfo(dtype).min)
            return torch.where(valid.expand(batch, 1, query_length, key_length), zeros, blocked)

        def forward(self, input_ids, attention_mask, cache_position, *decoder_flat_cache):
            decoder_cache = cache_from_flat(decoder_flat_cache, self.decoder.model.config)
            attention_mask_4d = self._attention_mask_4d(
                input_ids,
                attention_mask,
                decoder_flat_cache[0].dtype,
            )
            outputs = self.decoder(
                input_ids=input_ids,
                past_key_values=decoder_cache,
                attention_mask=attention_mask_4d,
                cache_position=cache_position,
                use_cache=True,
                output_hidden_states=False,
                output_attentions=False,
            )
            return (outputs.logits, *flatten_cache(outputs.past_key_values))

    class CodecDecodeWrapper(torch.nn.Module):
        def __init__(self, codec_model):
            super().__init__()
            self.codec_model = codec_model

        def forward(self, audio_codes):
            return self.codec_model.decode(audio_codes).audio_values

    class CodecEncodeWrapper(torch.nn.Module):
        def __init__(self, codec_model):
            super().__init__()
            self.codec_model = codec_model

        def forward(self, input_values):
            return self.codec_model.encode(input_values).audio_codes

    class GeneratePrefillWrapper(torch.nn.Module):
        def __init__(self, chroma):
            super().__init__()
            self.chroma = chroma
            install_batch1_dynamic_audio_tower(chroma)

        def forward(
            self,
            input_ids,
            attention_mask,
            input_values,
            input_values_cutoffs,
            thinker_input_ids,
            thinker_attention_mask,
            thinker_input_features,
            thinker_feature_attention_mask,
        ):
            input_embeddings, backbone_attention_mask = self.chroma._build_prompt_embeds(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_values=input_values,
                input_values_cutoffs=input_values_cutoffs,
            )
            thinker_cache = DynamicCache(config=self.chroma.thinker.model.config)
            thinker_cache_position = torch.arange(
                0,
                thinker_input_ids.shape[1],
                device=thinker_input_ids.device,
            )
            thinker_outputs = self.chroma.thinker(
                input_ids=thinker_input_ids,
                input_features=thinker_input_features,
                attention_mask=thinker_attention_mask,
                feature_attention_mask=thinker_feature_attention_mask,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=False,
                return_dict=True,
                past_key_values=thinker_cache,
                cache_position=thinker_cache_position,
                use_audio_in_video=False,
            )
            thinker_hidden_states = thinker_outputs.hidden_states[-1]
            thinker_next_ids = thinker_outputs.logits[:, -1:, :].argmax(dim=-1)
            next_token_emb = self.chroma._embed_text_tokens(thinker_next_ids)
            thinker_eos = thinker_next_ids.squeeze(-1) == self.chroma.config.im_end_token_id
            thinker_input_embeddings = torch.cat([thinker_hidden_states[:, -1:, :], next_token_emb], dim=1)
            input_embeddings = torch.cat([input_embeddings, thinker_input_embeddings], dim=1)
            thinker_attention_values = backbone_attention_mask.new_ones((backbone_attention_mask.shape[0], 1))
            backbone_attention_mask = torch.cat(
                [backbone_attention_mask, thinker_attention_values, thinker_attention_values],
                dim=1,
            )
            backbone_cache_position = torch.arange(
                0,
                input_embeddings.shape[1],
                device=input_embeddings.device,
            )
            outputs = self.chroma.backbone(
                input_embeddings=input_embeddings,
                attention_mask=backbone_attention_mask,
                cache_position=backbone_cache_position,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=False,
            )
            next_attention_mask = torch.cat(
                [
                    backbone_attention_mask,
                    backbone_attention_mask.new_ones((backbone_attention_mask.shape[0], 1)),
                ],
                dim=1,
            )
            return (
                outputs.logits,
                outputs.hidden_states[-1],
                next_attention_mask,
                thinker_next_ids,
                thinker_attention_mask,
                thinker_cache_position,
                thinker_eos.long(),
                *flatten_cache(outputs.past_key_values),
                *flatten_cache(thinker_outputs.past_key_values),
            )

    class BackboneFrameStepWrapper(torch.nn.Module):
        def __init__(self, chroma, backbone_cache_layer_count):
            super().__init__()
            self.chroma = chroma
            self.backbone_cache_layer_count = backbone_cache_layer_count

        def forward(self, frame_codes, attention_mask, *backbone_flat_cache):
            backbone_cache = cache_from_flat(backbone_flat_cache, self.chroma.backbone.model.config)
            input_embeddings = self.chroma.backbone.emb_audio_frames(frame_codes.to(self.chroma.device))
            past_seen_tokens = backbone_cache.get_seq_length()
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + input_embeddings.shape[1],
                device=input_embeddings.device,
            )
            outputs = self.chroma.backbone(
                input_embeddings=input_embeddings,
                attention_mask=attention_mask,
                past_key_values=backbone_cache,
                cache_position=cache_position,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=False,
            )
            next_attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                dim=1,
            )
            return (
                outputs.logits,
                outputs.hidden_states[-1],
                next_attention_mask,
                *flatten_cache(outputs.past_key_values),
            )

    class BackboneThinkerStepWrapper(torch.nn.Module):
        def __init__(self, chroma, backbone_cache_layer_count, thinker_cache_layer_count):
            super().__init__()
            self.chroma = chroma
            self.backbone_cache_layer_count = backbone_cache_layer_count
            self.thinker_cache_layer_count = thinker_cache_layer_count

        def forward(
            self,
            frame_codes,
            attention_mask,
            thinker_input_ids,
            thinker_attention_mask,
            thinker_cache_position,
            thinker_eos,
            *flat_cache,
        ):
            backbone_count = self.backbone_cache_layer_count * 2
            backbone_flat_cache = flat_cache[:backbone_count]
            thinker_flat_cache = flat_cache[backbone_count:]
            backbone_cache = cache_from_flat(backbone_flat_cache, self.chroma.backbone.model.config)
            thinker_cache = cache_from_flat(thinker_flat_cache, self.chroma.thinker.model.config)
            model_inputs = self.chroma.prepare_inputs_for_generation(
                input_ids=frame_codes,
                attention_mask=attention_mask,
                past_key_values=backbone_cache,
                thinker_input_ids=thinker_input_ids,
                thinker_attention_mask=thinker_attention_mask,
                thinker_cache_position=thinker_cache_position,
                thinker_past_key_values=thinker_cache,
                thinker_flag=True,
                thinker_eos=thinker_eos.bool(),
            )
            outputs = self.chroma(**model_inputs, return_dict=True)
            next_attention_mask = torch.cat(
                [
                    outputs.attention_mask,
                    outputs.attention_mask.new_ones((outputs.attention_mask.shape[0], 1)),
                ],
                dim=1,
            )
            thinker_input_ids_out = outputs.thinker_input_ids
            if thinker_input_ids_out is None:
                thinker_input_ids_out = torch.zeros_like(thinker_input_ids)
            return (
                outputs.logits,
                outputs.hidden_states[-1],
                next_attention_mask,
                thinker_input_ids_out,
                outputs.thinker_attention_mask,
                outputs.thinker_cache_position,
                outputs.thinker_eos.long(),
                *flatten_cache(outputs.past_key_values),
                *flatten_cache(outputs.thinker_past_key_values),
            )

    return SimpleNamespace(
        ThinkerTextWrapper=ThinkerTextWrapper,
        TextEmbeddingWrapper=TextEmbeddingWrapper,
        SystemPrefillWrapper=SystemPrefillWrapper,
        BackboneWrapper=BackboneWrapper,
        DecoderWrapper=DecoderWrapper,
        DecoderPrefillWrapper=DecoderPrefillWrapper,
        DecoderStepWrapper=DecoderStepWrapper,
        CodecDecodeWrapper=CodecDecodeWrapper,
        CodecEncodeWrapper=CodecEncodeWrapper,
        GeneratePrefillWrapper=GeneratePrefillWrapper,
        BackboneFrameStepWrapper=BackboneFrameStepWrapper,
        BackboneThinkerStepWrapper=BackboneThinkerStepWrapper,
    )
