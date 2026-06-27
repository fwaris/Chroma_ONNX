namespace ChromaOnnx

open System
open System.IO
open System.Text.Json
open Microsoft.ML.OnnxRuntime.Tensors
open Tokenizers.HuggingFace.Tokenizer
open TorchSharp

type ChromaNativeProcessor(modelDir: string, ?thinkerActiveFrames: int) =
    let tokenizerPath = Path.Combine(modelDir, "tokenizer.json")
    let configPath = Path.Combine(modelDir, "config.json")
    let preprocessorConfigPath = Path.Combine(modelDir, "preprocessor_config.json")
    let tokenizer = Tokenizer.FromFile(tokenizerPath)
    let promptSampleRate = 24000
    let thinkerSampleRate = 16000

    let readIntPropertyFrom (path: string) (name: string) (fallback: int) =
        if not (File.Exists path) then
            fallback
        else
            use doc = JsonDocument.Parse(File.ReadAllText(path))
            match doc.RootElement.TryGetProperty(name) with
            | true, value when value.ValueKind = JsonValueKind.Number -> value.GetInt32()
            | _ -> fallback

    let readConfigInt = readIntPropertyFrom configPath
    let readPreprocessorInt = readIntPropertyFrom preprocessorConfigPath

    let audioFeatureSize = readPreprocessorInt "feature_size" 128
    let thinkerMaxFrames = readPreprocessorInt "nb_max_frames" 30000
    let thinkerHopLength = readPreprocessorInt "hop_length" 160
    let thinkerNfft = readPreprocessorInt "n_fft" 400
    let thinkerMaxSamples = readPreprocessorInt "n_samples" (thinkerSampleRate * 300)
    let configuredThinkerActiveFrames =
        let value = defaultArg thinkerActiveFrames 0
        if value < 0 || value > thinkerMaxFrames then
            invalidArg (nameof thinkerActiveFrames) $"Thinker active frames must be 0 for full length or between 1 and {thinkerMaxFrames}, got {value}."
        value
    let thinkerActiveFrameLimit =
        if configuredThinkerActiveFrames = 0 then thinkerMaxFrames else configuredThinkerActiveFrames
    let audioFrameFreq = readConfigInt "audio_frame_freq" 1920
    let melFrequencyBins = thinkerNfft / 2 + 1

    let hertzToSlaneyMel (frequency: float) =
        let minLogHertz = 1000.0
        let minLogMel = 15.0
        let logStep = 27.0 / Math.Log(6.4)
        let linearMel = 3.0 * frequency / 200.0
        if frequency >= minLogHertz then
            minLogMel + Math.Log(frequency / minLogHertz) * logStep
        else
            linearMel

    let slaneyMelToHertz (mel: float) =
        let minLogMel = 15.0
        let logStep = Math.Log(6.4) / 27.0
        if mel >= minLogMel then
            1000.0 * Math.Exp(logStep * (mel - minLogMel))
        else
            200.0 * mel / 3.0

    let linspace start stop count =
        if count <= 1 then
            [| start |]
        else
            let step = (stop - start) / float (count - 1)
            Array.init count (fun index -> start + step * float index)

    let melFilterBank =
        lazy
            let minMel = hertzToSlaneyMel 0.0
            let maxMel = hertzToSlaneyMel (float (thinkerSampleRate / 2))
            let melFrequencies =
                linspace minMel maxMel (audioFeatureSize + 2)
                |> Array.map slaneyMelToHertz
            let fftFrequencies = linspace 0.0 (float (thinkerSampleRate / 2)) melFrequencyBins
            let filterValues = Array.zeroCreate<float32> (audioFeatureSize * melFrequencyBins)

            for filterIndex in 0 .. audioFeatureSize - 1 do
                let left = melFrequencies[filterIndex]
                let center = melFrequencies[filterIndex + 1]
                let right = melFrequencies[filterIndex + 2]
                let downDenominator = center - left
                let upDenominator = right - center
                let normalization = 2.0 / (right - left)

                for frequencyIndex in 0 .. melFrequencyBins - 1 do
                    let frequency = fftFrequencies[frequencyIndex]
                    let downSlope = (frequency - left) / downDenominator
                    let upSlope = (right - frequency) / upDenominator
                    let weight = max 0.0 (min downSlope upSlope)
                    filterValues[filterIndex * melFrequencyBins + frequencyIndex] <- float32 (weight * normalization)

            filterValues

    let encode (text: string) =
        let encoding =
            tokenizer.Encode(
                text,
                false,
                null,
                false,
                false,
                false,
                false,
                false,
                true,
                false
            )

        let ids =
            encoding
            |> Seq.head
            |> fun value -> value.Ids
            |> Seq.map int64
            |> Seq.toArray

        if ids.Length = 0 then
            invalidArg (nameof text) "Text produced no tokenizer ids."

        ids

    let monoFloat32FromLittleEndian (bytes: byte array) =
        if bytes.Length >= 12
           && bytes[0] = byte 'R'
           && bytes[1] = byte 'I'
           && bytes[2] = byte 'F'
           && bytes[3] = byte 'F'
           && bytes[8] = byte 'W'
           && bytes[9] = byte 'A'
           && bytes[10] = byte 'V'
           && bytes[11] = byte 'E' then
            invalidArg "bytes" "Expected raw little-endian Float32 PCM, but received a WAV container. Decode the audio to .f32 PCM before sending it."

        if bytes.Length % sizeof<float32> <> 0 then
            invalidArg "bytes" $"PCM Float32 payload length must be divisible by 4, got {bytes.Length} bytes."

        Array.init (bytes.Length / sizeof<float32>) (fun index -> BitConverter.ToSingle(bytes, index * sizeof<float32>))

    let denseInt64Row (values: int64 array) =
        DenseTensor<int64>(values, [| 1; values.Length |])

    let denseOnes length =
        DenseTensor<int64>(Array.create length 1L, [| 1; length |])

    let denseFloatAudio (values: float32 array) =
        DenseTensor<float32>(values, [| 1; 1; values.Length |])

    let audioTokenOutputLength activeFrames =
        if activeFrames <= 0 then
            0
        else
            let afterFirstConv = ((activeFrames - 1) / 2) + 1
            max 0 (((afterFirstConv - 2) / 2) + 1)

    let renderConversation (systemPrompt: string) audioTokenCount =
        let normalizedSystemPrompt =
            if String.IsNullOrWhiteSpace(systemPrompt) then
                "You are a helpful assistant."
            else
                systemPrompt.Trim()

        let audioPlaceholders = String.replicate (max 1 audioTokenCount) "<|AUDIO|>"

        String.concat
            ""
            [ "<|im_start|>system\n"
              normalizedSystemPrompt
              "<|im_end|>\n"
              "<|im_start|>user\n"
              "<|audio_bos|>"
              audioPlaceholders
              "<|audio_eos|><|im_end|>\n"
              "<|im_start|>assistant\n" ]

    let extractThinkerFeatures (pcm16k: float32 array) =
        let sampleCount = min pcm16k.Length thinkerMaxSamples
        let padded = Array.zeroCreate<float32> thinkerMaxSamples
        if sampleCount > 0 then
            Array.Copy(pcm16k, padded, sampleCount)

        let activeFrames =
            if sampleCount <= 0 then
                0
            else
                min thinkerMaxFrames (((sampleCount - 1) / thinkerHopLength) + 1)

        let activeFramesForExport = min activeFrames thinkerActiveFrameLimit

        let mask =
            Array.init thinkerMaxFrames (fun frameIndex -> if frameIndex < activeFramesForExport then 1L else 0L)

        use waveform = torch.tensor(padded, dtype = Nullable(torch.float32))
        use window = torch.hann_window(int64 thinkerNfft, dtype = Nullable(torch.float32))
        use stft =
            torch.stft(
                waveform,
                int64 thinkerNfft,
                hop_length = int64 thinkerHopLength,
                win_length = int64 thinkerNfft,
                window = window,
                center = true,
                normalized = false,
                onesided = true,
                return_complex = true
            )
        use magnitude = stft.abs()
        use powerSpectrum = magnitude * magnitude
        use melFilters =
            torch
                .tensor(melFilterBank.Value, dtype = Nullable(torch.float32))
                .reshape([| int64 audioFeatureSize; int64 melFrequencyBins |])
        use melSpectrum = torch.matmul(melFilters, powerSpectrum).clamp_min(1e-10).log10()
        use trimmed = melSpectrum.narrow(1L, 0L, int64 thinkerMaxFrames)
        use floor = torch.maximum(trimmed, trimmed.max() - 8.0)
        use normalized = (floor + 4.0) / 4.0
        let featureValues = normalized.contiguous().data<float32>().ToArray()

        DenseTensor<float32>(featureValues, [| 1; audioFeatureSize; thinkerMaxFrames |]),
        DenseTensor<int64>(mask, [| 1; thinkerMaxFrames |]),
        audioTokenOutputLength activeFramesForExport

    member _.PromptSampleRate = promptSampleRate
    member _.ThinkerSampleRate = thinkerSampleRate
    member _.ConfiguredThinkerActiveFrames = configuredThinkerActiveFrames
    member _.ThinkerTraceFeatureFrames = thinkerActiveFrameLimit
    member _.ThinkerTraceSamples = thinkerActiveFrameLimit * thinkerHopLength
    member _.ThinkerFeatureMode =
        if configuredThinkerActiveFrames = 0 then
            $"full-length up to {thinkerActiveFrameLimit} feature frames"
        else
            $"{thinkerActiveFrameLimit} clamped feature frames"
    member _.AudioFrameFreq = audioFrameFreq

    member _.ReadFloat32Pcm(path: string) =
        File.ReadAllBytes(path) |> monoFloat32FromLittleEndian

    member _.ReadFloat32PcmFromBytes(bytes: byte array) =
        monoFloat32FromLittleEndian bytes

    member _.Prepare(promptText: string, systemPrompt: string, promptAudio24k: float32 array, userAudio16k: float32 array) =
        if promptAudio24k.Length = 0 then
            invalidArg (nameof promptAudio24k) "Voice prompt PCM must contain at least one sample."
        if userAudio16k.Length = 0 then
            invalidArg (nameof userAudio16k) "User turn PCM must contain at least one sample."

        let promptIds = encode promptText
        let thinkerFeatures, thinkerFeatureMask, audioTokenCount = extractThinkerFeatures userAudio16k
        let conversationText = renderConversation systemPrompt audioTokenCount
        let thinkerIds = encode conversationText

        { InputIds = denseInt64Row promptIds
          AttentionMask = denseOnes promptIds.Length
          InputValues = denseFloatAudio promptAudio24k
          InputValuesCutoffs = DenseTensor<int64>([| int64 promptAudio24k.Length |], [| 1 |])
          ThinkerInputIds = denseInt64Row thinkerIds
          ThinkerAttentionMask = denseOnes thinkerIds.Length
          ThinkerInputFeatures = thinkerFeatures
          ThinkerFeatureAttentionMask = thinkerFeatureMask
          PromptAudioSamples = promptAudio24k.Length
          UserAudioSamples = userAudio16k.Length
          ConversationText = conversationText }

