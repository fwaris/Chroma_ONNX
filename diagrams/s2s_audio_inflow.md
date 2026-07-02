# Chroma S2S Audio Inflow

This diagram focuses on how audio enters the standalone F#/ONNX speech-to-speech service and becomes model-ready tensors, then streamed response audio. It covers both the reference voice prompt audio and the user turn audio because they enter the model through different paths.

## Browser To Runtime

```mermaid
flowchart TD
    subgraph Browser["Browser test page"]
        DefaultVoice["Default reference audio\nassets/southern_belle.mp3"]
        UploadedVoice["Optional uploaded reference audio\nMP3/WAV/browser-decodable audio or .f32"]
        ReferenceText["Reference text\nassets/southern_belle_prompt.txt or edited text"]
        TurnFile["User turn audio file\nMP3/WAV/browser-decodable audio or .f32"]
        TurnMic["Microphone capture\ngetUserMedia"]

        DefaultVoice --> RefDecode["AudioContext decode\nmix to mono\nresample to 24 kHz"]
        UploadedVoice --> RefDecode
        RefDecode --> PromptPcm["promptPcm24k\nFloat32LE mono PCM"]

        TurnFile --> TurnDecode["AudioContext decode\nmix to mono\nresample to 16 kHz"]
        TurnMic --> MicFrames["ScriptProcessor audio frames"]
        MicFrames --> TurnDecode
        TurnDecode --> TurnPcm["turn audio\nFloat32LE mono PCM"]
    end

    subgraph HttpApi["ASP.NET service API"]
        CreateSession["POST /api/s2s/sessions\nmultipart/form-data"]
        WebSocketTurn["GET /ws/s2s/{sessionId}\nturn.start + binary chunks + turn.end"]
    end

    PromptPcm --> CreateSession
    ReferenceText --> CreateSession
    TurnPcm --> WebSocketTurn

    subgraph AppLayer["ChromaOnnx.SpeechToSpeech"]
        RuntimeCreate["ChromaS2sRuntime.CreateSession"]
        SessionStore["Session store\nprompt text, system prompt,\nreference audio, max frames"]
        PromptArtifact["served_runs/.../prompt_audio_24k.f32"]
        RuntimeTurn["ChromaS2sRuntime.RunTurnAsync"]
        TurnArtifact["served_runs/.../user_audio_16k.f32"]
        WorkQueue["StreamingWorkQueue\nsingle active generation job"]
    end

    CreateSession --> RuntimeCreate
    RuntimeCreate --> SessionStore
    RuntimeCreate --> PromptArtifact
    WebSocketTurn --> RuntimeTurn
    RuntimeTurn --> TurnArtifact
    RuntimeTurn --> WorkQueue
```

## Native Preprocessing And ONNX Generation

```mermaid
flowchart TD
    subgraph Inputs["Runtime inputs"]
        PromptText["Reference text"]
        SystemPrompt["System prompt"]
        PromptAudio["Reference voice audio\n24 kHz Float32 PCM"]
        UserAudio["User turn audio\n16 kHz Float32 PCM"]
    end

    subgraph NativeProcessor["ChromaNativeProcessor.Prepare"]
        PromptTokenizer["Tokenize reference text"]
        PromptAudioTensor["Wrap prompt audio as input_values\nshape: [1, 1, samples]"]
        PromptCutoff["input_values_cutoffs\nprompt sample count"]
        ThinkerFeatures["Whisper-style log-mel extraction\nfrom 16 kHz user turn audio"]
        AudioTokenCount["Derive audio token count\nfrom active thinker frames"]
        Conversation["Render system/user conversation\nwith <|AUDIO|> placeholders"]
        ThinkerTokenizer["Tokenize rendered conversation"]
        Prepared["NativeS2sPrepared\nmodel input tensors"]
    end

    PromptText --> PromptTokenizer --> Prepared
    PromptAudio --> PromptAudioTensor --> Prepared
    PromptAudio --> PromptCutoff --> Prepared
    UserAudio --> ThinkerFeatures --> Prepared
    ThinkerFeatures --> AudioTokenCount --> Conversation
    SystemPrompt --> Conversation
    Conversation --> ThinkerTokenizer --> Prepared

    subgraph OnnxRuntime["ChromaS2sOnnxRunner.GenerateStreaming"]
        Prefill["generate_prefill\ns2s_mode = 0"]
        State["Backbone + thinker KV state"]
        Loop["Frame generation loop"]
        FrameStep["backbone_frame_step\nor backbone_thinker_step"]
        Decoder["decoder_prefill / decoder_step"]
        Codes["Generated audio codebook frames"]
        PartialDecode["Optional partial codec_decode\nfor streaming chunks"]
        FinalDecode["Final codec_decode"]
    end

    Prepared --> Prefill
    Prefill --> State
    Prefill --> Loop
    State --> Loop
    Loop --> FrameStep --> State
    FrameStep --> Decoder --> Codes
    Codes --> PartialDecode
    Codes --> FinalDecode

    subgraph Outputs["Service outputs"]
        JsonEvents["WebSocket JSON events\ngeneration.frame, audio.chunk, generation.done"]
        BinaryChunks["Binary Float32LE 24 kHz audio chunks"]
        Artifacts["served_runs artifacts\naudio.wav, audio_values.f32,\naudio_codes.i64, details.json"]
    end

    PartialDecode --> JsonEvents
    PartialDecode --> BinaryChunks
    FinalDecode --> Artifacts
    FinalDecode --> JsonEvents
```

## Important Rates And Boundaries

| Boundary | Format | Owner |
| --- | --- | --- |
| Reference voice prompt into session creation | Mono Float32 PCM at 24 kHz | Browser page sends `promptPcm24k`; runtime stores it on the session |
| User turn into WebSocket | Mono Float32 PCM at 16 kHz | Browser sends binary chunks after `turn.start` |
| Thinker audio features | Log-mel tensor plus attention mask | `ChromaNativeProcessor` |
| Response stream | Mono Float32 PCM at 24 kHz | `ChromaS2sOnnxRunner.GenerateStreaming` emits `audio.chunk` metadata followed by binary audio |
| Persisted preview | WAV plus raw `.f32` and codes | `ChromaS2sRuntime` under `served_runs` |

The key split is that reference audio conditions the voice through `input_values`, while user turn audio is transformed into thinker features that drive the assistant response generation.
