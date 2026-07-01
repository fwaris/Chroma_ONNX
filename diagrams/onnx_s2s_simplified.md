# Chroma S2S ONNX Simplified Diagrams

These diagrams collapse the large ONNX graphs into the subsystem blocks that matter when reading the runtime shape. They describe the local generated artifacts in this workspace, especially `onnx/chroma-s2s-full-v2` and `onnx/chroma-s2s-split-trt`.

## S2S Runtime Overview

```mermaid
flowchart LR
    Text[Prompt text] --> Prep[F# native preprocessing]
    PromptAudio[Prompt audio] --> Prep
    UserAudio[User turn audio] --> Prep

    Prep --> Inputs[Prepared tensors]
    Inputs --> Prefill[generate_prefill]
    Prefill --> State[Backbone and thinker KV cache]
    Prefill --> Loop[Frame generation loop]

    State --> Loop
    Loop -->|frame code input| FrameStep[backbone_frame_step]
    Loop -->|thinker active frame| ThinkerStep[backbone_thinker_step]
    FrameStep --> State
    ThinkerStep --> State
    FrameStep --> Decoder[decoder_prefill / decoder_step]
    ThinkerStep --> Decoder
    Decoder --> Codes[Audio codebooks]
    Codes --> Codec[codec_decode]
    Codec --> Audio[Decoded audio values]
    Audio --> Wav[WAV / service response]

    Weights[(Shared safetensor weights)] -. registered initializers .-> Prefill
    Weights -. registered initializers .-> FrameStep
    Weights -. registered initializers .-> ThinkerStep
    Weights -. registered initializers .-> Decoder
    Weights -. registered initializers .-> Codec
```

## Merged S2S Dispatcher

The `chroma_s2s_merged.weights_free.onnx` file is small at the top level because it is a dispatcher. The real work is inside nested `If` branch graphs selected by the scalar `s2s_mode` input.

```mermaid
flowchart TD
    Mode[s2s_mode int64 scalar] --> Dispatch{Nested ONNX If dispatcher}

    Dispatch -->|0| GP[generate_prefill]
    Dispatch -->|1| BFS[backbone_frame_step]
    Dispatch -->|2| BTS[backbone_thinker_step]
    Dispatch -->|3| DEC[decoder]
    Dispatch -->|4| DP[decoder_prefill]
    Dispatch -->|5| DS[decoder_step]
    Dispatch -->|6| CD[codec_decode]

    GP --> Out[Shared merged output list]
    BFS --> Out
    BTS --> Out
    DEC --> Out
    DP --> Out
    DS --> Out
    CD --> Out

    Out --> Named[F# runner keeps the named outputs for the selected logical graph]
```

Each branch returns the same merged output list shape. Outputs that do not belong to the selected logical graph are filled with dummy branch outputs, so the top-level ONNX signature stays stable.

## Split Logical Graphs

The split S2S bundle exposes the same logical phases as separate ONNX files. The diagrams below intentionally hide repeated transformer layers, attention-mask plumbing, rotary math, reshapes, and cache tensor fan-out.

```mermaid
flowchart LR
    subgraph generate_prefill
        GPIn[Text, audio features, thinker prompt] --> Thinker[Thinker stack]
        GPIn --> AudioTower[Audio feature path]
        Thinker --> Backbone[Backbone stack]
        AudioTower --> Backbone
        Backbone --> GPLogits[Backbone logits and hidden states]
        Backbone --> GPCache[Initial backbone KV cache]
        Thinker --> ThinkerCache[Initial thinker KV cache]
    end

    subgraph generation_steps
        FrameCodes[Prior frame codes] --> FrameBackbone[Backbone frame step]
        FrameBackbone --> FrameLogits[Next frame logits]
        FrameBackbone --> NextBackboneCache[Updated backbone KV cache]

        FrameCodes --> ThinkerBackbone[Backbone plus thinker step]
        ThinkerBackbone --> ThinkerNext[Next thinker token state]
        ThinkerBackbone --> NextThinkerCache[Updated thinker KV cache]
        ThinkerBackbone --> ThinkerBackboneCache[Updated backbone KV cache]
    end

    subgraph decoder_and_codec
        Hidden[Backbone hidden state] --> DecPrefill[decoder_prefill]
        DecPrefill --> DecCache[Decoder KV cache]
        DecCache --> DecStep[decoder_step]
        DecStep --> CodeLogits[Audio code logits]
        CodeLogits --> AudioCodes[8 codebooks per frame]
        AudioCodes --> Codec[codec_decode]
        Codec --> AudioValues[Audio waveform values]
    end

    SharedWeights[(Shared safetensor weights)] -.-> Thinker
    SharedWeights -.-> Backbone
    SharedWeights -.-> FrameBackbone
    SharedWeights -.-> ThinkerBackbone
    SharedWeights -.-> DecPrefill
    SharedWeights -.-> DecStep
    SharedWeights -.-> Codec
```

## Original vs CUDA Optimized Graph Shape

The optimized CUDA cache can look much smaller and more provider-specific than the original split graph, while preserving the same public callable contract.

```mermaid
flowchart LR
    Inputs[Same graph inputs] --> Original[Original weights-free ONNX]
    Inputs --> Optimized[CUDA optimized ONNX cache]

    Original --> OrigConstants[Many Constant nodes in graph]
    Original --> OrigOps[Exported ONNX compute nodes]
    Original --> OrigOutputs[Same graph outputs]

    Optimized --> OptInitializers[Folded constants as initializers]
    Optimized --> OptOps[Remaining compute nodes]
    Optimized --> Memcpy[CUDA provider memory-copy nodes]
    Optimized --> OptOutputs[Same graph outputs]

    OrigOutputs -. same names and shapes .-> OptOutputs
```

Observed local split-graph examples:

| Logical graph | Original nodes | CUDA optimized nodes | Public I/O |
| --- | ---: | ---: | --- |
| `generate_prefill` | 20515 | 14084 | same |
| `codec_decode` | 2010 | 1299 | same |
| `decoder_prefill` | 1189 | 765 | same |
| `decoder_step` | 1170 | 761 | same |

The main visual difference is that ONNX Runtime folded many `Constant` nodes into initializers and added CUDA provider memory-transfer nodes such as `MemcpyFromHost` and `MemcpyToHost`. In this workspace, the TensorRT optimized `.onnx` files often match the local-external original structure more closely than the CUDA optimized files.

For `onnx/chroma-s2s-full-v2`, the merged original and the merged optimized cache both have the same tiny top-level dispatcher shape: `Constant`, `Equal`, and `If`. Their large graphs live inside nested branch graphs, so the top-level model view can be misleadingly simple.
