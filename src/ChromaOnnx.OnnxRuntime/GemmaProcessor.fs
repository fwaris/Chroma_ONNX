namespace ChromaOnnx

open System
open System.Collections.Generic
open System.Globalization
open System.IO
open System.Text
open System.Text.Json
open System.Text.RegularExpressions
open Microsoft.ML.OnnxRuntime.Tensors
open Tokenizers.HuggingFace.Tokenizer
open TorchSharp

type GemmaProcessor(modelDir: string, ?maxAudioSeconds: float) =
    let tokenizerPath = Path.Combine(modelDir, "tokenizer.json")
    let processorConfigPath = Path.Combine(modelDir, "processor_config.json")
    let configPath = Path.Combine(modelDir, "config.json")
    let tokenizer =
        lazy
            if not (File.Exists tokenizerPath) then
                invalidOp $"Gemma tokenizer was not found: {tokenizerPath}"
            Tokenizer.FromFile(tokenizerPath)

    let maxAudioSeconds = defaultArg maxAudioSeconds 30.0 |> max 0.1

    let readJson path =
        if File.Exists path then
            Some(JsonDocument.Parse(File.ReadAllText(path)))
        else
            None

    let tryProperty (path: string) (pathParts: string array) =
        readJson path
        |> Option.bind (fun doc ->
            let mutable current = doc.RootElement
            let mutable found = true
            for part in pathParts do
                if found then
                    match current.TryGetProperty(part) with
                    | true, value -> current <- value
                    | false, _ -> found <- false
            if found then Some(current.Clone()) else None)

    let intFrom path parts fallback =
        match tryProperty path parts with
        | Some value when value.ValueKind = JsonValueKind.Number ->
            match value.TryGetInt32() with
            | true, number -> number
            | _ -> fallback
        | _ -> fallback

    let floatFrom path parts fallback =
        match tryProperty path parts with
        | Some value when value.ValueKind = JsonValueKind.Number ->
            match value.TryGetDouble() with
            | true, number -> number
            | _ -> fallback
        | _ -> fallback

    let stringFrom path parts fallback =
        match tryProperty path parts with
        | Some value when value.ValueKind = JsonValueKind.String ->
            value.GetString() |> Option.ofObj |> Option.defaultValue fallback
        | _ -> fallback

    let arrayIntFrom path parts fallback =
        match tryProperty path parts with
        | Some value when value.ValueKind = JsonValueKind.Array ->
            value.EnumerateArray()
            |> Seq.choose (fun item ->
                match item.ValueKind with
                | JsonValueKind.Number ->
                    match item.TryGetInt64() with
                    | true, number -> Some number
                    | _ -> None
                | _ -> None)
            |> Seq.toArray
        | Some value when value.ValueKind = JsonValueKind.Number ->
            match value.TryGetInt64() with
            | true, number -> [| number |]
            | _ -> fallback
        | _ -> fallback

    let samplingRate = intFrom processorConfigPath [| "feature_extractor"; "sampling_rate" |] 16000
    let featureSize = intFrom processorConfigPath [| "feature_extractor"; "feature_size" |] 128
    let fftLength = intFrom processorConfigPath [| "feature_extractor"; "fft_length" |] 512
    let frameLength = intFrom processorConfigPath [| "feature_extractor"; "frame_length" |] 320
    let hopLength = intFrom processorConfigPath [| "feature_extractor"; "hop_length" |] 160
    let minFrequency = floatFrom processorConfigPath [| "feature_extractor"; "min_frequency" |] 0.0
    let maxFrequency = floatFrom processorConfigPath [| "feature_extractor"; "max_frequency" |] 8000.0
    let melFloor = floatFrom processorConfigPath [| "feature_extractor"; "mel_floor" |] 0.001
    let paddingValue = floatFrom processorConfigPath [| "feature_extractor"; "padding_value" |] 0.0
    let preemphasis = floatFrom processorConfigPath [| "feature_extractor"; "preemphasis" |] 0.0
    let audioSeqLength = intFrom processorConfigPath [| "audio_seq_length" |] 750
    let audioMsPerToken = intFrom processorConfigPath [| "audio_ms_per_token" |] 40
    let audioTokenId = intFrom configPath [| "audio_token_id" |] 258881
    let bosTokenId = intFrom configPath [| "text_config"; "bos_token_id" |] 2
    let eosTokenIds =
        let generationConfigPath = Path.Combine(modelDir, "generation_config.json")
        let configured = arrayIntFrom generationConfigPath [| "eos_token_id" |] Array.empty
        if configured.Length > 0 then configured else arrayIntFrom configPath [| "eos_token_id" |] [| 1L; 106L; 50L |]

    let tokenStringFromTokenizerConfig name fallback =
        match tryProperty (Path.Combine(modelDir, "tokenizer_config.json")) [| name |] with
        | Some value when value.ValueKind = JsonValueKind.String -> value.GetString() |> Option.ofObj |> Option.defaultValue fallback
        | _ -> fallback

    let audioToken = tokenStringFromTokenizerConfig "audio_token" "<|audio|>"
    let boaToken = tokenStringFromTokenizerConfig "boa_token" "<|audio>"
    let eoaToken = tokenStringFromTokenizerConfig "eoa_token" "<audio|>"
    let bosToken = tokenStringFromTokenizerConfig "bos_token" "<bos>"

    let hertzToHtkMel (frequency: float) =
        2595.0 * Math.Log10(1.0 + frequency / 700.0)

    let htkMelToHertz (mel: float) =
        700.0 * (Math.Pow(10.0, mel / 2595.0) - 1.0)

    let linspace start stop count =
        if count <= 1 then
            [| start |]
        else
            let step = (stop - start) / float (count - 1)
            Array.init count (fun index -> start + step * float index)

    let melFilterBank =
        lazy
            let numFrequencyBins = fftLength / 2 + 1
            let melMin = hertzToHtkMel minFrequency
            let melMax = hertzToHtkMel maxFrequency
            let filterFreqs =
                linspace melMin melMax (featureSize + 2)
                |> Array.map htkMelToHertz
            let fftFreqs = linspace 0.0 (Math.Floor(float samplingRate / 2.0)) numFrequencyBins
            let values = Array.zeroCreate<float32> (featureSize * numFrequencyBins)

            for melIndex in 0 .. featureSize - 1 do
                let left = filterFreqs[melIndex]
                let center = filterFreqs[melIndex + 1]
                let right = filterFreqs[melIndex + 2]
                for freqIndex in 0 .. numFrequencyBins - 1 do
                    let freq = fftFreqs[freqIndex]
                    let down =
                        if center = left then 0.0 else (freq - left) / (center - left)
                    let up =
                        if right = center then 0.0 else (right - freq) / (right - center)
                    values[melIndex * numFrequencyBins + freqIndex] <- float32 (max 0.0 (min down up))

            values

    let encode (text: string) =
        let encoding =
            tokenizer.Value.Encode(
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

        encoding
        |> Seq.head
        |> fun value -> value.Ids
        |> Seq.map int64
        |> Seq.toArray

    let decodeByReflection (ids: int64 array) =
        let tok = tokenizer.Value
        let methods =
            tok.GetType().GetMethods()
            |> Array.filter (fun method -> method.Name = "Decode")
            |> Array.sortBy (fun method -> method.GetParameters().Length)

        let idsForType (targetType: Type) =
            if targetType = typeof<int64 array> || targetType.IsAssignableFrom(typeof<int64 array>) then
                box ids
            elif targetType = typeof<int array> || targetType.IsAssignableFrom(typeof<int array>) then
                ids |> Array.map int |> box
            elif targetType = typeof<uint32 array> || targetType.IsAssignableFrom(typeof<uint32 array>) then
                ids |> Array.map uint32 |> box
            elif targetType = typeof<uint64 array> || targetType.IsAssignableFrom(typeof<uint64 array>) then
                ids |> Array.map uint64 |> box
            elif targetType.IsAssignableFrom(typeof<IEnumerable<uint32>>) then
                ids |> Array.map uint32 |> box
            else
                box ids

        let tryInvoke (method: Reflection.MethodInfo) =
            let parameters = method.GetParameters()
            try
                let args =
                    match parameters.Length with
                    | 1 ->
                        [| idsForType parameters[0].ParameterType |]
                    | 2 ->
                        let first = parameters[0].ParameterType
                        [| idsForType first; box true |]
                    | _ -> [||]
                if args.Length = parameters.Length then
                    match method.Invoke(tok, args) with
                    | :? string as text -> Some text
                    | null -> None
                    | value -> value.ToString() |> Option.ofObj
                else
                    None
            with _ ->
                None

        methods
        |> Array.tryPick tryInvoke
        |> Option.defaultWith (fun () -> String.Join(" ", ids))

    let escapeText (text: string) =
        if Object.ReferenceEquals(text, null) then "" else text

    let quoteValue (text: string) =
        let value = if Object.ReferenceEquals(text, null) then "" else text
        "<|\"|>" + value.Replace("<|\"|>", "\"") + "<|\"|>"

    let renderToolDeclaration (tool: GemmaToolDeclaration) =
        let properties =
            tool.Parameters
            |> Array.map (fun parameter ->
                let description =
                    if String.IsNullOrWhiteSpace parameter.Description then
                        ""
                    else
                        ",description:" + quoteValue parameter.Description
                $"{parameter.Name}:{{type:{quoteValue (parameter.Type.ToUpperInvariant())}{description}}}")
            |> String.concat ","
        let required =
            tool.Parameters
            |> Array.filter _.Required
            |> Array.map (fun parameter -> quoteValue parameter.Name)
            |> String.concat ","
        let objectType = quoteValue "OBJECT"
        $"<|tool>declaration:{tool.Name}{{description:{quoteValue tool.Description},parameters:{{properties:{{{properties}}},required:[{required}],type:{objectType}}}}}<tool|>"

    let normalizedRole role =
        match role with
        | GemmaChatRole.System -> "system"
        | GemmaChatRole.User -> "user"
        | GemmaChatRole.Model -> "model"
        | GemmaChatRole.Tool -> "tool"

    let renderMessage (message: GemmaChatMessage) =
        match message.Role with
        | GemmaChatRole.Tool ->
            let name = message.ToolName |> Option.defaultValue "tool"
            $"<|turn>tool\n<|tool_response>response:{name}{{value:{quoteValue message.Content}}}<tool_response|><turn|>\n"
        | _ ->
            $"<|turn>{normalizedRole message.Role}\n{escapeText message.Content}<turn|>\n"

    let computeAudioTokenCount sampleCount =
        let padLeft = frameLength / 2
        let mutable frames = int (Math.Floor((float (sampleCount + padLeft - frameLength - 1)) / float hopLength)) + 1
        if frames <= 0 then
            0
        else
            for _ in 0 .. 1 do
                frames <- ((frames - 1) / 2) + 1
            min frames audioSeqLength

    let expandAudioToken (prompt: string) (audio: float32 array option) =
        match audio with
        | None -> prompt
        | Some samples ->
            let tokens = max 1 (min audioSeqLength (computeAudioTokenCount samples.Length))
            prompt.Replace(audioToken, "\n\n" + boaToken + String.replicate tokens audioToken + eoaToken + "\n\n")

    let parseToolArguments (body: string) =
        let arguments = Dictionary<string, string>(StringComparer.Ordinal)
        let pattern = @"(?<name>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*(?:<\|""\|>(?<quoted>.*?)<\|""\|>|(?<bare>[^,}]+))"
        for item in Regex.Matches(body, pattern, RegexOptions.Singleline) do
            let name = item.Groups["name"].Value
            let value =
                if item.Groups["quoted"].Success then
                    item.Groups["quoted"].Value
                else
                    item.Groups["bare"].Value.Trim()
            arguments[name] <- value
        arguments |> Seq.map (fun item -> item.Key, item.Value) |> Map.ofSeq

    member _.ModelDir = modelDir
    member _.TokenizerPath = tokenizerPath
    member _.SamplingRate = samplingRate
    member _.FeatureSize = featureSize
    member _.FrameLength = frameLength
    member _.HopLength = hopLength
    member _.AudioSeqLength = audioSeqLength
    member _.AudioMsPerToken = audioMsPerToken
    member _.AudioTokenId = int64 audioTokenId
    member _.BosTokenId = int64 bosTokenId
    member _.EosTokenIds = eosTokenIds
    member _.AudioToken = audioToken

    member _.Encode(text: string) = encode text

    member _.Decode(ids: int64 array) = decodeByReflection ids

    member _.ComputeAudioTokenCount(sampleCount: int) = computeAudioTokenCount sampleCount

    member _.RenderChat(messages: GemmaChatMessage array, tools: GemmaToolDeclaration array, addGenerationPrompt: bool, ?audio: float32 array) =
        let builder = StringBuilder()
        builder.Append(bosToken) |> ignore
        let systemMessages = messages |> Array.filter (fun message -> message.Role = GemmaChatRole.System)
        let nonSystemMessages = messages |> Array.filter (fun message -> message.Role <> GemmaChatRole.System)
        if systemMessages.Length > 0 || tools.Length > 0 then
            builder.Append("<|turn>system\n") |> ignore
            if systemMessages.Length > 0 then
                systemMessages
                |> Array.map _.Content
                |> String.concat "\n\n"
                |> escapeText
                |> builder.Append
                |> ignore
            if tools.Length > 0 then
                if systemMessages.Length > 0 then builder.Append("\n\n") |> ignore
                tools |> Array.iter (fun tool -> builder.Append(renderToolDeclaration tool).Append("\n") |> ignore)
            builder.Append("<turn|>\n") |> ignore
        for message in nonSystemMessages do
            builder.Append(renderMessage message) |> ignore
        if addGenerationPrompt then
            builder.Append("<|turn>model\n") |> ignore

        expandAudioToken (builder.ToString()) audio

    member this.Prepare(request: GemmaGenerationRequest) =
        let prompt = this.RenderChat(request.Messages, request.Tools, request.AddGenerationPrompt, ?audio = request.Audio16k)
        let ids = encode prompt
        if ids.Length = 0 then invalidArg (nameof request) "Gemma prompt produced no tokenizer ids."
        let inputIds = DenseTensor<int64>(ids, [| 1; ids.Length |])
        let mask = DenseTensor<int64>(Array.create ids.Length 1L, [| 1; ids.Length |])
        let audioFeatures = request.Audio16k |> Option.map this.ExtractAudioFeatures
        { Prompt = prompt
          InputIds = inputIds
          AttentionMask = mask
          AudioFeatures = audioFeatures }

    member _.ExtractAudioFeatures(audio16k: float32 array) =
        if samplingRate <> 16000 then
            invalidOp $"Gemma audio feature extractor expects a 16 kHz model config, found {samplingRate} Hz."
        if audio16k.Length = 0 then
            invalidArg (nameof audio16k) "Gemma audio input must contain at least one sample."
        let maxSamples = int (Math.Ceiling(maxAudioSeconds * float samplingRate))
        let original =
            if audio16k.Length > maxSamples then
                audio16k[0 .. maxSamples - 1]
            else
                audio16k
        let padToMultiple = 128
        let paddedLength =
            let remainder = original.Length % padToMultiple
            if remainder = 0 then original.Length else original.Length + (padToMultiple - remainder)
        let paddedAudio = Array.zeroCreate<float32> paddedLength
        Array.Copy(original, paddedAudio, original.Length)
        if paddingValue <> 0.0 then
            for index in original.Length .. paddedLength - 1 do
                paddedAudio[index] <- float32 paddingValue

        let padLeft = frameLength / 2
        let semicausal = Array.zeroCreate<float32> (paddedAudio.Length + padLeft)
        Array.Copy(paddedAudio, 0, semicausal, padLeft, paddedAudio.Length)
        let numFrames = max 0 (int (Math.Floor(float (semicausal.Length - frameLength) / float hopLength)) + 1)
        let frameCountForMask =
            max 0 (int (Math.Floor(float (paddedAudio.Length + padLeft - (frameLength + 1)) / float hopLength)) + 1)
        let featureValues = Array.zeroCreate<float32> (max 0 numFrames * featureSize)
        let maskValues = Array.zeroCreate<bool> (max 0 numFrames)

        if numFrames > 0 then
            let numFrequencyBins = fftLength / 2 + 1
            use waveform =
                torch.tensor(
                    semicausal,
                    ReadOnlySpan<int64>([| int64 semicausal.Length |]),
                    dtype = Nullable(torch.float32)
                )
            use window = torch.hann_window(int64 frameLength, dtype = Nullable(torch.float32))
            use stft =
                torch.stft(
                    waveform,
                    int64 fftLength,
                    hop_length = int64 hopLength,
                    win_length = int64 frameLength,
                    window = window,
                    center = false,
                    normalized = false,
                    onesided = true,
                    return_complex = true
                )
            use magnitude = stft.abs()
            use melFilters =
                torch
                    .tensor(melFilterBank.Value, dtype = Nullable(torch.float32))
                    .reshape([| int64 featureSize; int64 numFrequencyBins |])
            use melSpectrum = torch.matmul(melFilters, magnitude)
            use logMel = (melSpectrum + float32 melFloor).log()
            use transposed = logMel.transpose(0L, 1L).contiguous()
            let active = transposed.data<float32>()
            active.CopyTo(featureValues.AsSpan(0, featureValues.Length), 0, 0L)

            let sampleMask = Array.zeroCreate<byte> (original.Length + padLeft + padToMultiple)
            for index in padLeft .. padLeft + original.Length - 1 do
                sampleMask[index] <- 1uy
            let frameSizeForUnfold = frameLength + 1
            let validFrameCount = min numFrames frameCountForMask
            for frameIndex in 0 .. validFrameCount - 1 do
                let sampleIndex = frameIndex * hopLength + frameSizeForUnfold - 1
                let valid = sampleIndex >= 0 && sampleIndex < sampleMask.Length && sampleMask[sampleIndex] <> 0uy
                maskValues[frameIndex] <- valid
                if not valid then
                    Array.Clear(featureValues, frameIndex * featureSize, featureSize)
            for frameIndex in validFrameCount .. numFrames - 1 do
                Array.Clear(featureValues, frameIndex * featureSize, featureSize)

        { InputFeatures = DenseTensor<float32>(featureValues, [| 1; numFrames; featureSize |])
          InputFeaturesMask = DenseTensor<bool>(maskValues, [| 1; numFrames |])
          AudioTokenCount = computeAudioTokenCount original.Length
          FrameCount = numFrames
          ValidFrameCount = maskValues |> Array.filter id |> Array.length
          SampleCount = original.Length }

    member _.TryParseToolCall(text: string) =
        let source = if Object.ReferenceEquals(text, null) then "" else text
        let patterns =
            [| @"<\|tool_call\>\s*call:(?<name>[A-Za-z_][A-Za-z0-9_]*)\s*\{(?<args>.*?)\}\s*<tool_call\|>"
               @"(?:^|\s)call:(?<name>[A-Za-z_][A-Za-z0-9_]*)\s*\{(?<args>.*?)\}(?:\s|$)" |]
        let matchValue =
            patterns
            |> Array.map (fun pattern -> Regex.Match(source, pattern, RegexOptions.Singleline))
            |> Array.tryFind _.Success
            |> Option.defaultValue Match.Empty
        if matchValue.Success then
            Some
                { Name = matchValue.Groups["name"].Value
                  Arguments = parseToolArguments matchValue.Groups["args"].Value
                  RawText = matchValue.Value }
        else
            None
