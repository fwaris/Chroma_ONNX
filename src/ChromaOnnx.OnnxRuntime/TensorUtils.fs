namespace ChromaOnnx

open System
open System.IO
open System.Linq
open System.Runtime.InteropServices
open System.Text.Json
open Microsoft.ML.OnnxRuntime.Tensors

module TensorIO =
    let private expectedBytes<'T> count =
        count * System.Runtime.InteropServices.Marshal.SizeOf<'T>()

    let private ensureParentDirectory (path: string) =
        match Path.GetDirectoryName(Path.GetFullPath(path)) with
        | null | "" -> ()
        | directory -> Directory.CreateDirectory(directory) |> ignore

    let private elementCount (dimensions: ReadOnlySpan<int>) =
        let mutable count = 1
        for dimension in dimensions do
            count <- count * dimension
        count

    let private isContiguous (tensor: DenseTensor<'T>) =
        let dimensions = tensor.Dimensions
        let strides = tensor.Strides
        let mutable expectedStride = 1
        let mutable index = dimensions.Length - 1
        let mutable contiguous = true

        while contiguous && index >= 0 do
            if strides[index] <> expectedStride then
                contiguous <- false
            expectedStride <- expectedStride * dimensions[index]
            index <- index - 1

        contiguous

    let readSingles (path: string) (count: int) =
        let bytes = File.ReadAllBytes path
        let expected = expectedBytes<float32> count
        if bytes.Length <> expected then
            invalidArg path $"Expected {expected} bytes for {count} float32 values, found {bytes.Length}."

        let values = Array.zeroCreate<float32> count
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length)
        values

    let readInt64s (path: string) (count: int) =
        let bytes = File.ReadAllBytes path
        let expected = expectedBytes<int64> count
        if bytes.Length <> expected then
            invalidArg path $"Expected {expected} bytes for {count} int64 values, found {bytes.Length}."

        let values = Array.zeroCreate<int64> count
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length)
        values

    let denseFloat (data: float32 array) (dims: int array) =
        DenseTensor<float32>(data, dims)

    let denseInt64 (data: int64 array) (dims: int array) =
        DenseTensor<int64>(data, dims)

    let writeSingles (path: string) (tensor: DenseTensor<float32>) =
        ensureParentDirectory path
        if isContiguous tensor then
            let data = tensor.Buffer.Span.Slice(0, elementCount tensor.Dimensions)
            File.WriteAllBytes(path, MemoryMarshal.AsBytes(data))
        else
            let data = Enumerable.ToArray(tensor)
            let bytes = Array.zeroCreate<byte> (data.Length * sizeof<float32>)
            Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length)
            File.WriteAllBytes(path, bytes)

    let writeInt64s (path: string) (tensor: DenseTensor<int64>) =
        ensureParentDirectory path
        if isContiguous tensor then
            let data = tensor.Buffer.Span.Slice(0, elementCount tensor.Dimensions)
            File.WriteAllBytes(path, MemoryMarshal.AsBytes(data))
        else
            let data = Enumerable.ToArray(tensor)
            let bytes = Array.zeroCreate<byte> (data.Length * sizeof<int64>)
            Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length)
            File.WriteAllBytes(path, bytes)

    let cloneFloatTensor (tensor: Microsoft.ML.OnnxRuntime.Tensors.Tensor<float32>) =
        DenseTensor<float32>(Enumerable.ToArray(tensor), tensor.Dimensions.ToArray())

    let cloneInt64Tensor (tensor: Microsoft.ML.OnnxRuntime.Tensors.Tensor<int64>) =
        DenseTensor<int64>(Enumerable.ToArray(tensor), tensor.Dimensions.ToArray())

    let cloneFloatTensorOwned (tensor: Microsoft.ML.OnnxRuntime.Tensors.Tensor<float32>) =
        let dimensions = tensor.Dimensions.ToArray()
        let count = elementCount tensor.Dimensions
        let buffer = RentedTensorBuffer.rent<float32> count false
        try
            let destination = buffer.Memory.Span
            match tensor with
            | :? DenseTensor<float32> as dense when isContiguous dense ->
                dense.Buffer.Span.Slice(0, count).CopyTo(destination)
            | _ ->
                let mutable index = 0
                for value in tensor do
                    destination[index] <- value
                    index <- index + 1
            DenseTensor<float32>(buffer.Memory, ReadOnlySpan<int>(dimensions), false), buffer :> IDisposable
        with
        | _ ->
            (buffer :> IDisposable).Dispose()
            reraise()

    let cloneInt64TensorOwned (tensor: Microsoft.ML.OnnxRuntime.Tensors.Tensor<int64>) =
        let dimensions = tensor.Dimensions.ToArray()
        let count = elementCount tensor.Dimensions
        let buffer = RentedTensorBuffer.rent<int64> count false
        try
            let destination = buffer.Memory.Span
            match tensor with
            | :? DenseTensor<int64> as dense when isContiguous dense ->
                dense.Buffer.Span.Slice(0, count).CopyTo(destination)
            | _ ->
                let mutable index = 0
                for value in tensor do
                    destination[index] <- value
                    index <- index + 1
            DenseTensor<int64>(buffer.Memory, ReadOnlySpan<int>(dimensions), false), buffer :> IDisposable
        with
        | _ ->
            (buffer :> IDisposable).Dispose()
            reraise()

module Manifest =
    let private tryGetInt (name: string) (root: JsonElement) =
        match root.TryGetProperty(name) with
        | true, value when value.ValueKind = JsonValueKind.Number ->
            match value.TryGetInt32() with
            | true, number -> Some number
            | _ -> None
        | _ -> None

    let load (onnxDir: string) =
        let path = Path.Combine(onnxDir, "manifest.json")
        if not (File.Exists path) then
            { HiddenSize = None; AudioNumCodebooks = None }
        else
            use doc = JsonDocument.Parse(File.ReadAllText(path))
            { HiddenSize = tryGetInt "hidden_size" doc.RootElement
              AudioNumCodebooks = tryGetInt "audio_num_codebooks" doc.RootElement }

module TensorMath =
    let argmaxLast (logits: DenseTensor<float32>) =
        let dims = logits.Dimensions
        if dims.Length <> 3 then
            invalidArg (nameof logits) "Expected logits shape [batch, sequence, vocab]."

        let batch = dims[0]
        let sequence = dims[1]
        let vocab = dims[2]
        if batch < 1 || sequence < 1 || vocab < 1 then
            invalidArg (nameof logits) "Expected logits shape [batch, sequence, vocab] with positive dimensions."

        let values = logits.Buffer.Span
        let strides = logits.Strides
        let ids = Array.zeroCreate<int64> batch

        for batchIndex in 0 .. batch - 1 do
            let rowStart = (batchIndex * strides[0]) + ((sequence - 1) * strides[1])
            let mutable bestIndex = 0
            let mutable bestValue = values[rowStart]

            for vocabIndex in 1 .. vocab - 1 do
                let value = values[rowStart + (vocabIndex * strides[2])]
                if value > bestValue then
                    bestValue <- value
                    bestIndex <- vocabIndex

            ids[batchIndex] <- int64 bestIndex

        ids

    let lastHidden (hiddenStates: DenseTensor<float32>) =
        let dims = hiddenStates.Dimensions
        if dims.Length <> 3 then
            invalidArg (nameof hiddenStates) "Expected hidden states shape [batch, sequence, hidden]."

        let batch = dims[0]
        let sequence = dims[1]
        let hidden = dims[2]
        if batch < 1 || sequence < 1 || hidden < 1 then
            invalidArg (nameof hiddenStates) "Expected hidden states shape [batch, sequence, hidden] with positive dimensions."

        let source = hiddenStates.Buffer.Span
        let strides = hiddenStates.Strides
        let values = Array.zeroCreate<float32> (batch * hidden)

        for batchIndex in 0 .. batch - 1 do
            let sourceStart = (batchIndex * strides[0]) + ((sequence - 1) * strides[1])
            let destination = values.AsSpan(batchIndex * hidden, hidden)
            if strides[2] = 1 then
                source.Slice(sourceStart, hidden).CopyTo(destination)
            else
                for hiddenIndex in 0 .. hidden - 1 do
                    destination[hiddenIndex] <- source[sourceStart + (hiddenIndex * strides[2])]

        DenseTensor<float32>(values, [| batch; hidden |])

    let appendColumn (ids: DenseTensor<int64>) (nextIds: int64 array) =
        let dims = ids.Dimensions
        if dims.Length <> 2 then
            invalidArg (nameof ids) "Expected ids shape [batch, sequence]."

        let batch = dims[0]
        let sequence = dims[1]
        if batch < 1 || sequence < 0 then
            invalidArg (nameof ids) "Expected ids shape [batch, sequence] with non-empty batch and non-negative sequence."

        if nextIds.Length <> batch then
            invalidArg (nameof nextIds) "Expected one next id per batch row."

        let source = ids.Buffer.Span
        let strides = ids.Strides
        let nextSequence = sequence + 1
        let values = Array.zeroCreate<int64> (batch * nextSequence)

        for batchIndex in 0 .. batch - 1 do
            let sourceStart = batchIndex * strides[0]
            let destinationStart = batchIndex * nextSequence
            if strides[1] = 1 then
                source.Slice(sourceStart, sequence).CopyTo(values.AsSpan(destinationStart, sequence))
            else
                for sequenceIndex in 0 .. sequence - 1 do
                    values[destinationStart + sequenceIndex] <- source[sourceStart + (sequenceIndex * strides[1])]
            values[destinationStart + sequence] <- nextIds[batchIndex]

        DenseTensor<int64>(values, [| batch; nextSequence |])

