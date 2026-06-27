namespace ChromaOnnx

open System
open System.IO
open System.Linq
open System.Text.Json
open Microsoft.ML.OnnxRuntime.Tensors
open TorchSharp

module TensorIO =
    let private expectedBytes<'T> count =
        count * System.Runtime.InteropServices.Marshal.SizeOf<'T>()

    let private ensureParentDirectory (path: string) =
        match Path.GetDirectoryName(Path.GetFullPath(path)) with
        | null | "" -> ()
        | directory -> Directory.CreateDirectory(directory) |> ignore

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
        let data = Enumerable.ToArray(tensor)
        let bytes = Array.zeroCreate<byte> (data.Length * sizeof<float32>)
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length)
        ensureParentDirectory path
        File.WriteAllBytes(path, bytes)

    let writeInt64s (path: string) (tensor: DenseTensor<int64>) =
        let data = Enumerable.ToArray(tensor)
        let bytes = Array.zeroCreate<byte> (data.Length * sizeof<int64>)
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length)
        ensureParentDirectory path
        File.WriteAllBytes(path, bytes)

    let cloneFloatTensor (tensor: Microsoft.ML.OnnxRuntime.Tensors.Tensor<float32>) =
        DenseTensor<float32>(Enumerable.ToArray(tensor), tensor.Dimensions.ToArray())

    let cloneInt64Tensor (tensor: Microsoft.ML.OnnxRuntime.Tensors.Tensor<int64>) =
        DenseTensor<int64>(Enumerable.ToArray(tensor), tensor.Dimensions.ToArray())

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

module TorchTensor =
    let private shape (dims: int array) =
        dims |> Array.map int64

    let fromDenseFloat (tensor: DenseTensor<float32>) =
        let data = Enumerable.ToArray(tensor)
        torch.tensor(data, dtype = Nullable(torch.float32)).reshape(shape (tensor.Dimensions.ToArray()))

    let fromDenseInt64 (tensor: DenseTensor<int64>) =
        let data = Enumerable.ToArray(tensor)
        torch.tensor(data, dtype = Nullable(torch.int64)).reshape(shape (tensor.Dimensions.ToArray()))

    let toDenseFloat (tensor: torch.Tensor) (dims: int array) =
        use contiguous = tensor.contiguous()
        DenseTensor<float32>(contiguous.data<float32>().ToArray(), dims)

    let toDenseInt64 (tensor: torch.Tensor) (dims: int array) =
        use contiguous = tensor.contiguous()
        DenseTensor<int64>(contiguous.data<int64>().ToArray(), dims)

module TensorMath =
    let argmaxLast (logits: DenseTensor<float32>) =
        let dims = logits.Dimensions.ToArray()
        if dims.Length <> 3 then
            invalidArg (nameof logits) "Expected logits shape [batch, sequence, vocab]."

        let batch = dims[0]
        let sequence = dims[1]
        use tensor = TorchTensor.fromDenseFloat logits
        use last = tensor.select(1L, int64 (sequence - 1))
        use ids = last.argmax(1L)
        ids.data<int64>().ToArray()

    let lastHidden (hiddenStates: DenseTensor<float32>) =
        let dims = hiddenStates.Dimensions.ToArray()
        if dims.Length <> 3 then
            invalidArg (nameof hiddenStates) "Expected hidden states shape [batch, sequence, hidden]."

        let batch = dims[0]
        let sequence = dims[1]
        let hidden = dims[2]
        use tensor = TorchTensor.fromDenseFloat hiddenStates
        use last = tensor.select(1L, int64 (sequence - 1))
        TorchTensor.toDenseFloat last [| batch; hidden |]

    let appendColumn (ids: DenseTensor<int64>) (nextIds: int64 array) =
        let dims = ids.Dimensions.ToArray()
        if dims.Length <> 2 then
            invalidArg (nameof ids) "Expected ids shape [batch, sequence]."

        let batch = dims[0]
        let sequence = dims[1]
        if nextIds.Length <> batch then
            invalidArg (nameof nextIds) "Expected one next id per batch row."

        use current = TorchTensor.fromDenseInt64 ids
        use next = torch.tensor(nextIds, dtype = Nullable(torch.int64)).reshape([| int64 batch; 1L |])
        use appended = torch.cat([| current; next |], 1L)
        TorchTensor.toDenseInt64 appended [| batch; sequence + 1 |]

