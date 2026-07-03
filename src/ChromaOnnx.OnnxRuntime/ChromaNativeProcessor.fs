namespace ChromaOnnx

open System
open System.IO
open System.Text
open System.Text.Json
open Microsoft.ML.OnnxRuntime.Tensors
open Tokenizers.HuggingFace.Tokenizer
open TorchSharp

type NativeS2sConversationAudio =
    { Role: string
      Audio16k: float32 array }

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

    let denseFloatAudio (values: float32 array) =
        DenseTensor<float32>(values, [| 1; 1; values.Length |])

    let rentedInt64TensorFromValues (values: int64 array) (dimensions: int array) =
        let buffer = RentedTensorBuffer.rent<int64> values.Length false
        try
            values.AsSpan().CopyTo(buffer.Memory.Span)
            DenseTensor<int64>(buffer.Memory, ReadOnlySpan<int>(dimensions), false), buffer :> IDisposable
        with
        | _ ->
            (buffer :> IDisposable).Dispose()
            reraise()

    let rentedInt64TensorFilled length (value: int64) (dimensions: int array) =
        let buffer = RentedTensorBuffer.rent<int64> length false
        try
            buffer.Memory.Span.Fill(value)
            DenseTensor<int64>(buffer.Memory, ReadOnlySpan<int>(dimensions), false), buffer :> IDisposable
        with
        | _ ->
            (buffer :> IDisposable).Dispose()
            reraise()

    let rentedInt64TensorSingleton value =
        let buffer = RentedTensorBuffer.rent<int64> 1 false
        try
            buffer.Memory.Span[0] <- value
            DenseTensor<int64>(buffer.Memory, ReadOnlySpan<int>([| 1 |]), false), buffer :> IDisposable
        with
        | _ ->
            (buffer :> IDisposable).Dispose()
            reraise()

    let audioTokenOutputLength activeFrames =
        if activeFrames <= 0 then
            0
        else
            let afterFirstConv = ((activeFrames - 1) / 2) + 1
            max 0 (((afterFirstConv - 2) / 2) + 1)

    let normalizedRole role =
        match (if String.IsNullOrWhiteSpace role then "" else role.Trim().ToLowerInvariant()) with
        | "assistant" -> "assistant"
        | "user" | "" -> "user"
        | value -> invalidArg (nameof role) $"Unsupported S2S conversation role '{value}'. Use user or assistant."

    let renderAudioContent audioTokenCount =
        String.concat
            ""
            [ "<|audio_bos|>"
              String.replicate (max 1 audioTokenCount) "<|AUDIO|>"
              "<|audio_eos|>" ]

    let renderConversation (systemPrompt: string) (items: (string * int) array) =
        let normalizedSystemPrompt =
            if String.IsNullOrWhiteSpace(systemPrompt) then
                "You are a helpful assistant."
            else
                systemPrompt.Trim()

        let builder = StringBuilder()
        builder.Append("<|im_start|>system\n").Append(normalizedSystemPrompt).Append("<|im_end|>\n") |> ignore
        for role, audioTokenCount in items do
            builder
                .Append("<|im_start|>")
                .Append(normalizedRole role)
                .Append("\n")
                .Append(renderAudioContent audioTokenCount)
                .Append("<|im_end|>\n")
            |> ignore
        builder.Append("<|im_start|>assistant\n") |> ignore
        builder.ToString()

    let fillThinkerFeatures
        audioIndex
        (pcm16k: float32 array)
        (featureBuffer: RentedTensorBuffer<float32>)
        (maskBuffer: RentedTensorBuffer<int64>)
        =
        if pcm16k.Length = 0 then
            invalidArg (nameof pcm16k) "Conversation audio PCM must contain at least one sample."

        let sampleCount = min pcm16k.Length thinkerMaxSamples
        let activeFrames =
            if sampleCount <= 0 then
                0
            else
                min thinkerMaxFrames (((sampleCount - 1) / thinkerHopLength) + 1)

        let activeFramesForExport = min activeFrames thinkerActiveFrameLimit
        let maskOffset = audioIndex * thinkerMaxFrames
        let mask = maskBuffer.Memory.Span.Slice(maskOffset, thinkerMaxFrames)
        mask.Clear()
        for frameIndex in 0 .. activeFramesForExport - 1 do
            mask[frameIndex] <- 1L

        if activeFramesForExport > 0 then
            let framesToCompute = activeFrames
            let samplesNeededForComputedFrames = min sampleCount (framesToCompute * thinkerHopLength)
            let workingSampleCount = max 1 (min thinkerMaxSamples (samplesNeededForComputedFrames + thinkerNfft))
            use workingBuffer = RentedTensorBuffer.rent<float32> workingSampleCount true
            let workingPcm = workingBuffer.Memory.Span
            workingPcm.Clear()
            pcm16k
                .AsSpan(0, min sampleCount workingSampleCount)
                .CopyTo(workingPcm)

            use waveform =
                torch.tensor(
                    workingBuffer.Memory,
                    ReadOnlySpan<int64>([| int64 workingSampleCount |]),
                    dtype = Nullable(torch.float32)
                )
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
            use trimmed = melSpectrum.narrow(1L, 0L, int64 framesToCompute)
            use floor = torch.maximum(trimmed, trimmed.max() - 8.0)
            use normalized = (floor + 4.0) / 4.0
            use contiguous = normalized.contiguous()
            let activeFeatureValues = contiguous.data<float32>()
            let featureValues = featureBuffer.Memory.Span
            let audioOffset = audioIndex * audioFeatureSize * thinkerMaxFrames

            for featureIndex in 0 .. audioFeatureSize - 1 do
                activeFeatureValues.CopyTo(
                    featureValues.Slice(audioOffset + featureIndex * thinkerMaxFrames, activeFramesForExport),
                    0,
                    int64 (featureIndex * framesToCompute)
                )

        audioTokenOutputLength activeFramesForExport, sampleCount

    let extractThinkerFeatures (audios16k: float32 array array) =
        if audios16k.Length = 0 then
            invalidArg (nameof audios16k) "At least one conversation audio segment is required."

        let featureBuffer = RentedTensorBuffer.rent<float32> (audios16k.Length * audioFeatureSize * thinkerMaxFrames) true
        let maskBuffer = RentedTensorBuffer.rent<int64> (audios16k.Length * thinkerMaxFrames) true

        try
            featureBuffer.Memory.Span.Clear()
            maskBuffer.Memory.Span.Clear()
            let tokenCounts = Array.zeroCreate<int> audios16k.Length
            let sampleCounts = Array.zeroCreate<int> audios16k.Length
            for audioIndex in 0 .. audios16k.Length - 1 do
                let tokenCount, sampleCount =
                    fillThinkerFeatures audioIndex audios16k[audioIndex] featureBuffer maskBuffer
                tokenCounts[audioIndex] <- tokenCount
                sampleCounts[audioIndex] <- sampleCount

            DenseTensor<float32>(featureBuffer.Memory, ReadOnlySpan<int>([| audios16k.Length; audioFeatureSize; thinkerMaxFrames |]), false),
            DenseTensor<int64>(maskBuffer.Memory, ReadOnlySpan<int>([| audios16k.Length; thinkerMaxFrames |]), false),
            tokenCounts,
            sampleCounts,
            [| featureBuffer :> IDisposable; maskBuffer :> IDisposable |]
        with
        | _ ->
            (featureBuffer :> IDisposable).Dispose()
            (maskBuffer :> IDisposable).Dispose()
            reraise()

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

    member _.PrepareConversation
        (
            promptText: string,
            systemPrompt: string,
            promptAudio24k: float32 array,
            conversationAudios: NativeS2sConversationAudio array,
            currentUserAudioSamples: int
        ) =
        if promptAudio24k.Length = 0 then
            invalidArg (nameof promptAudio24k) "Voice prompt PCM must contain at least one sample."
        if conversationAudios.Length = 0 then
            invalidArg (nameof conversationAudios) "At least one conversation audio segment is required."
        if currentUserAudioSamples <= 0 then
            invalidArg (nameof currentUserAudioSamples) "Current user turn PCM must contain at least one sample."

        let promptIds = encode promptText
        let thinkerFeatures, thinkerFeatureMask, audioTokenCounts, _, ownedBuffers =
            conversationAudios
            |> Array.map _.Audio16k
            |> extractThinkerFeatures
        let owned = ResizeArray<IDisposable>(ownedBuffers)

        try
            let conversationText =
                conversationAudios
                |> Array.mapi (fun index item -> normalizedRole item.Role, audioTokenCounts[index])
                |> renderConversation systemPrompt
            let thinkerIds = encode conversationText

            let inputIds, inputIdsBuffer = rentedInt64TensorFromValues promptIds [| 1; promptIds.Length |]
            owned.Add(inputIdsBuffer)
            let attentionMask, attentionMaskBuffer = rentedInt64TensorFilled promptIds.Length 1L [| 1; promptIds.Length |]
            owned.Add(attentionMaskBuffer)
            let inputValuesCutoffs, inputValuesCutoffsBuffer = rentedInt64TensorSingleton (int64 promptAudio24k.Length)
            owned.Add(inputValuesCutoffsBuffer)
            let thinkerInputIds, thinkerInputIdsBuffer = rentedInt64TensorFromValues thinkerIds [| 1; thinkerIds.Length |]
            owned.Add(thinkerInputIdsBuffer)
            let thinkerAttentionMask, thinkerAttentionMaskBuffer = rentedInt64TensorFilled thinkerIds.Length 1L [| 1; thinkerIds.Length |]
            owned.Add(thinkerAttentionMaskBuffer)

            new NativeS2sPrepared(
                inputIds,
                attentionMask,
                denseFloatAudio promptAudio24k,
                inputValuesCutoffs,
                thinkerInputIds,
                thinkerAttentionMask,
                thinkerFeatures,
                thinkerFeatureMask,
                promptAudio24k.Length,
                currentUserAudioSamples,
                conversationText,
                owned.ToArray()
            )
        with
        | _ ->
            for buffer in owned do
                buffer.Dispose()
            reraise()

    member this.Prepare(promptText: string, systemPrompt: string, promptAudio24k: float32 array, userAudio16k: float32 array) =
        if userAudio16k.Length = 0 then
            invalidArg (nameof userAudio16k) "User turn PCM must contain at least one sample."

        this.PrepareConversation(
            promptText,
            systemPrompt,
            promptAudio24k,
            [| { Role = "user"; Audio16k = userAudio16k } |],
            userAudio16k.Length
        )

