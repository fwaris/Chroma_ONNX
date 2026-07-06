namespace ChromaOnnx

open System
open System.ComponentModel
open System.Collections.Generic
open System.IO
open System.IO.MemoryMappedFiles
open System.Linq
open System.Runtime.InteropServices
open System.Text.Json
open Microsoft.ML.OnnxRuntime
open Microsoft.ML.OnnxRuntime.Tensors

type SafetensorTensorMetadata =
    { Name: string
      Dtype: string
      Shape: int64 array
      DataOffset: int64
      ByteLength: int64 }

type MappedSafetensorTensor =
    { SourceShard: string
      SourceTensor: string
      Dtype: string
      Shape: int64 array
      Pointer: IntPtr
      ByteLength: int64 }

type SharedInitializerEntry =
    { Graph: string
      OnnxInitializer: string
      SourceShard: string
      SourceTensor: string
      Dtype: string
      Shape: int64 array
      ByteLength: int64
      Transform: string option }

type SharedGraphInfo =
    { Name: string
      Path: string
      Inputs: string array
      Outputs: string array }

type SharedBundleCapabilities =
    { GraphMode: string option
      ThinkerFeatureMode: string option
      ThinkerMaxAudioItems: int option }

type SharedBundleManifest =
    { BundleDir: string
      HiddenSize: int option
      AudioNumCodebooks: int option
      Capabilities: SharedBundleCapabilities
      Graphs: Dictionary<string, SharedGraphInfo>
      Initializers: SharedInitializerEntry array }

type SafetensorShard(path: string) =
    let fileInfo = FileInfo(path)

    let readHeader () =
        if not fileInfo.Exists then
            invalidArg (nameof path) $"Safetensors shard does not exist: {path}"

        use stream = File.OpenRead(path)
        use reader = new BinaryReader(stream)
        if stream.Length < 8L then
            invalidArg (nameof path) $"Safetensors shard is too small: {path}"

        let headerLength = int64 (reader.ReadUInt64())
        if headerLength < 0L || headerLength > int64 Int32.MaxValue || 8L + headerLength > stream.Length then
            invalidArg (nameof path) $"Invalid safetensors header length in {path}: {headerLength}"

        let headerBytes = reader.ReadBytes(int headerLength)
        let dataStart = 8L + headerLength
        use doc = JsonDocument.Parse(headerBytes)
        let tensors = Dictionary<string, SafetensorTensorMetadata>(StringComparer.Ordinal)

        for property in doc.RootElement.EnumerateObject() do
            if property.Name <> "__metadata__" then
                let element = property.Value

                let dtype =
                    match element.GetProperty("dtype").GetString() with
                    | null -> invalidArg property.Name "Safetensors tensor is missing dtype."
                    | value -> value

                let shape =
                    element.GetProperty("shape").EnumerateArray()
                    |> Seq.map (fun item -> item.GetInt64())
                    |> Seq.toArray

                let offsets =
                    element.GetProperty("data_offsets").EnumerateArray()
                    |> Seq.map (fun item -> item.GetInt64())
                    |> Seq.toArray

                if offsets.Length <> 2 || offsets[1] < offsets[0] then
                    invalidArg property.Name $"Invalid safetensors offsets for tensor {property.Name} in {path}."

                let absoluteOffset = dataStart + offsets[0]
                let byteLength = offsets[1] - offsets[0]
                if absoluteOffset < dataStart || absoluteOffset + byteLength > stream.Length then
                    invalidArg property.Name $"Safetensors tensor {property.Name} points outside {path}."

                tensors.Add(
                    property.Name,
                    { Name = property.Name
                      Dtype = dtype
                      Shape = shape
                      DataOffset = absoluteOffset
                      ByteLength = byteLength }
                )

        tensors

    let tensors = readHeader ()
    let map = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, fileInfo.Length, MemoryMappedFileAccess.Read)
    let accessor = map.CreateViewAccessor(0L, fileInfo.Length, MemoryMappedFileAccess.Read)
    let baseAddress = accessor.SafeMemoryMappedViewHandle.DangerousGetHandle()

    member _.Path = path
    member _.TensorNames = tensors.Keys :> seq<string>

    member _.GetTensor(sourceTensor: string) =
        match tensors.TryGetValue(sourceTensor) with
        | true, metadata ->
            let shardName =
                match Path.GetFileName(path) with
                | null | "" -> path
                | value -> value

            { SourceShard = shardName
              SourceTensor = sourceTensor
              Dtype = metadata.Dtype
              Shape = metadata.Shape
              Pointer = IntPtr(baseAddress.ToInt64() + metadata.DataOffset)
              ByteLength = metadata.ByteLength }
        | false, _ ->
            invalidArg (nameof sourceTensor) $"Safetensors tensor {sourceTensor} was not found in {path}."

    interface IDisposable with
        member _.Dispose() =
            accessor.Dispose()
            map.Dispose()

type SafetensorWeightStore(modelDir: string, entries: SharedInitializerEntry array) =
    let shardNames =
        entries
        |> Seq.map (fun entry -> entry.SourceShard)
        |> Seq.distinct
        |> Seq.sort
        |> Seq.toArray

    let shards =
        shardNames
        |> Array.map (fun shardName ->
            let shardPath = Path.Combine(modelDir, shardName)
            shardName, new SafetensorShard(shardPath))
        |> dict

    member _.MappedShardCount = shardNames.Length

    member _.GetTensor(sourceShard: string, sourceTensor: string) =
        match shards.TryGetValue(sourceShard) with
        | true, shard -> shard.GetTensor(sourceTensor)
        | false, _ -> invalidArg (nameof sourceShard) $"Safetensors shard {sourceShard} is not loaded from {modelDir}."

    interface IDisposable with
        member _.Dispose() =
            for shard in shards.Values do
                (shard :> IDisposable).Dispose()

type SharedInitializerRegistry(store: SafetensorWeightStore, entries: SharedInitializerEntry array) =
    let normalizeDtype (dtype: string) =
        dtype.ToUpperInvariant()

    let elementTypeFor (dtype: string) =
        match normalizeDtype dtype with
        | "F32" -> TensorElementType.Float
        | "BF16" -> TensorElementType.BFloat16
        | value -> invalidOp $"Unsupported shared initializer dtype {value}. Chroma shared V1 supports F32 and BF16."

    let validateShape (entry: SharedInitializerEntry) (tensor: MappedSafetensorTensor) =
        if not (tensor.Shape.SequenceEqual(entry.Shape)) then
            let manifestShape = String.Join(", ", entry.Shape)
            let safetensorShape = String.Join(", ", tensor.Shape)
            invalidOp (
                $"Shape mismatch for {entry.OnnxInitializer}: manifest [{manifestShape}], "
                + $"safetensors [{safetensorShape}]."
            )

    let initializerKey (entry: SharedInitializerEntry) =
        let shape = String.Join("x", entry.Shape)
        $"{entry.SourceShard}::{entry.SourceTensor}::{normalizeDtype entry.Dtype}::{shape}::{entry.ByteLength}"

    let sharedValues = Dictionary<string, OrtValue>(StringComparer.Ordinal)

    let createSharedValue (entry: SharedInitializerEntry) =
        let key = initializerKey entry
        match sharedValues.TryGetValue(key) with
        | true, value -> value
        | false, _ ->
            let elementType = elementTypeFor entry.Dtype
            let tensor = store.GetTensor(entry.SourceShard, entry.SourceTensor)
            let sourceElementType = elementTypeFor tensor.Dtype
            if sourceElementType <> elementType then
                invalidOp $"Dtype mismatch for {entry.OnnxInitializer}: manifest {entry.Dtype}, safetensors {tensor.Dtype}."

            validateShape entry tensor
            if tensor.ByteLength <> entry.ByteLength then
                invalidOp $"Byte length mismatch for {entry.OnnxInitializer}: manifest {entry.ByteLength}, safetensors {tensor.ByteLength}."

            let value =
                OrtValue.CreateTensorValueWithData(
                    OrtMemoryInfo.DefaultInstance,
                    elementType,
                    entry.Shape,
                    tensor.Pointer,
                    tensor.ByteLength
                )

            sharedValues[key] <- value
            value

    let initializers =
        entries
        |> Array.map (fun entry ->
            let value = createSharedValue entry
            entry, value)

    member _.InitializerCount = initializers.Length

    member _.UniqueSourceTensorCount =
        initializers
        |> Seq.map (fun (entry, _) -> $"{entry.SourceShard}::{entry.SourceTensor}")
        |> Seq.distinct
        |> Seq.length

    member _.UniqueOrtValueCount = sharedValues.Count

    member _.AddInitializers(options: SessionOptions, graphName: string) =
        for entry, value in initializers do
            if entry.Graph = graphName then
                options.AddInitializer(entry.OnnxInitializer, value)

    member _.AddAllInitializers(options: SessionOptions) =
        let added = HashSet<string>(StringComparer.Ordinal)
        for entry, value in initializers do
            if added.Add(entry.OnnxInitializer) then
                options.AddInitializer(entry.OnnxInitializer, value)

    interface IDisposable with
        member _.Dispose() =
            for value in sharedValues.Values do
                value.Dispose()

module SharedWeights =
    [<DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)>]
    extern bool private CreateHardLinkW(string lpFileName, string lpExistingFileName, nativeint lpSecurityAttributes)

    [<DllImport("libc", SetLastError = true, EntryPoint = "link")>]
    extern int private LinkUnix(string oldPath, string newPath)

    let private pathComparison =
        if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
            StringComparison.OrdinalIgnoreCase
        else
            StringComparison.Ordinal

    let private pathEquals left right =
        String.Equals(Path.GetFullPath(left), Path.GetFullPath(right), pathComparison)

    let private hardLink sourcePath targetPath =
        if RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then
            if not (CreateHardLinkW(targetPath, sourcePath, 0n)) then
                let error = Marshal.GetLastWin32Error()
                raise (IOException($"Could not create hardlink {targetPath} -> {sourcePath}.", Win32Exception(error)))
        else
            let result = LinkUnix(sourcePath, targetPath)
            if result <> 0 then
                let error = Marshal.GetLastWin32Error()
                raise (IOException($"Could not create hardlink {targetPath} -> {sourcePath}.", Win32Exception(error)))

    let private linkOrSymlink sourcePath targetPath =
        try
            hardLink sourcePath targetPath
        with hardlinkEx ->
            try
                File.CreateSymbolicLink(targetPath, sourcePath) |> ignore
            with symlinkEx ->
                raise (
                    IOException(
                        $"Could not create local external-data link {targetPath} for safetensor shard {sourcePath}. "
                        + "Place the model and ONNX bundle on a volume that supports hardlinks, enable symlink creation, "
                        + "or rebuild the local-external cache with --copy-if-hardlink-fails.",
                        AggregateException(hardlinkEx, symlinkEx)
                    )
                )

    let ensureLocalExternalDataLinks (modelDir: string) (graphPath: string) (entries: SharedInitializerEntry seq) =
        let graphDir =
            match Path.GetDirectoryName(graphPath) with
            | null | "" -> Directory.GetCurrentDirectory()
            | value -> value

        Directory.CreateDirectory(graphDir) |> ignore

        entries
        |> Seq.map (fun entry -> entry.SourceShard)
        |> Seq.distinct
        |> Seq.iter (fun sourceShard ->
            let sourcePath = Path.GetFullPath(Path.Combine(modelDir, sourceShard))
            if not (File.Exists sourcePath) then
                invalidArg (nameof modelDir) $"Safetensors shard was not found for external-data link: {sourcePath}"

            let targetPath = Path.GetFullPath(Path.Combine(graphDir, sourceShard))
            if not (pathEquals sourcePath targetPath) then
                if File.Exists targetPath then
                    let sourceInfo = FileInfo(sourcePath)
                    let targetInfo = FileInfo(targetPath)
                    if sourceInfo.Length <> targetInfo.Length then
                        invalidOp (
                            $"External-data link target already exists with a different size: {targetPath}. "
                            + $"Expected {sourceInfo.Length} bytes from {sourcePath}, found {targetInfo.Length} bytes."
                        )
                else
                    linkOrSymlink sourcePath targetPath)

    let private tryGetInt (name: string) (root: JsonElement) =
        match root.TryGetProperty(name) with
        | true, value when value.ValueKind = JsonValueKind.Number ->
            match value.TryGetInt32() with
            | true, number -> Some number
            | _ -> None
        | _ -> None

    let private requiredString (name: string) (element: JsonElement) =
        match element.GetProperty(name).GetString() with
        | null -> invalidArg name $"Missing string property {name}."
        | value -> value

    let private optionalString (name: string) (element: JsonElement) =
        match element.TryGetProperty(name) with
        | true, value when value.ValueKind = JsonValueKind.String -> value.GetString() |> Option.ofObj
        | _ -> None

    let private requiredInt64 (name: string) (element: JsonElement) =
        element.GetProperty(name).GetInt64()

    let private int64Array (name: string) (element: JsonElement) =
        element.GetProperty(name).EnumerateArray()
        |> Seq.map (fun item -> item.GetInt64())
        |> Seq.toArray

    let private optionalStringArray (name: string) (element: JsonElement) =
        match element.TryGetProperty(name) with
        | true, value when value.ValueKind = JsonValueKind.Array ->
            value.EnumerateArray()
            |> Seq.choose (fun item ->
                if item.ValueKind = JsonValueKind.String then
                    item.GetString() |> Option.ofObj
                else
                    None)
            |> Seq.toArray
        | _ -> Array.empty

    let private resolvePath (bundleDir: string) (path: string) =
        if Path.IsPathRooted(path) then
            if File.Exists path then
                path
            else
                match Path.GetFileName path with
                | null | "" -> path
                | fileName ->
                    let localPath = Path.GetFullPath(Path.Combine(bundleDir, fileName))
                    if File.Exists localPath then localPath else path
        else
            Path.GetFullPath(Path.Combine(bundleDir, path))

    let loadManifest (bundleDir: string) =
        let manifestPath = Path.Combine(bundleDir, "shared_weights_manifest.json")
        if not (File.Exists manifestPath) then
            invalidArg (nameof bundleDir) $"Shared weights manifest was not found: {manifestPath}"

        use doc = JsonDocument.Parse(File.ReadAllText(manifestPath))
        let root = doc.RootElement
        let graphs = Dictionary<string, SharedGraphInfo>(StringComparer.Ordinal)
        let capabilitiesElement =
            match root.TryGetProperty("capabilities") with
            | true, value when value.ValueKind = JsonValueKind.Object -> Some value
            | _ -> None
        let thinkerMaxAudioItems =
            capabilitiesElement |> Option.bind (tryGetInt "thinker_max_audio_items")
        let graphMode =
            capabilitiesElement |> Option.bind (optionalString "s2s_graph_mode")
        let capabilities =
            { GraphMode = graphMode
              ThinkerFeatureMode = capabilitiesElement |> Option.bind (optionalString "thinker_feature_mode")
              ThinkerMaxAudioItems = thinkerMaxAudioItems }

        for property in root.GetProperty("graphs").EnumerateObject() do
            let path = requiredString "path" property.Value |> resolvePath bundleDir
            graphs.Add(
                property.Name,
                { Name = property.Name
                  Path = path
                  Inputs = optionalStringArray "inputs" property.Value
                  Outputs = optionalStringArray "outputs" property.Value }
            )

        let initializers =
            root.GetProperty("initializers").EnumerateArray()
            |> Seq.map (fun item ->
                { Graph = requiredString "graph" item
                  OnnxInitializer = requiredString "onnx_initializer" item
                  SourceShard = requiredString "source_shard" item
                  SourceTensor = requiredString "source_tensor" item
                  Dtype = requiredString "dtype" item
                  Shape = int64Array "shape" item
                  ByteLength = requiredInt64 "byte_length" item
                  Transform = optionalString "transform" item })
            |> Seq.toArray

        { BundleDir = bundleDir
          HiddenSize = tryGetInt "hidden_size" root
          AudioNumCodebooks = tryGetInt "audio_num_codebooks" root
          Capabilities = capabilities
          Graphs = graphs
          Initializers = initializers }
