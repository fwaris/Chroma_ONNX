namespace ChromaOnnx

open System
open System.IO
open System.Linq
open System.Text
open Microsoft.ML.OnnxRuntime.Tensors

module Wave =
    type PreviewStats =
        { Samples: int
          DurationSeconds: float
          PeakAbs: float
          Rms: float
          MeanAbs: float
          WavPreviewGain: float
          WavPeakAbs: float }

    let private writeAscii (writer: BinaryWriter) (text: string) =
        writer.Write(Encoding.ASCII.GetBytes(text))

    let private clamp16 (sample: float) =
        let clamped =
            if sample > 1.0 then 1.0
            elif sample < -1.0 then -1.0
            else sample

        int16 (Math.Round(clamped * 32767.0))

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

    let private stats (sampleRate: int) (data: Span<float32>) =
        if data.Length = 0 then
            {| PeakAbs = 0.0; Rms = 0.0; MeanAbs = 0.0; DurationSeconds = 0.0 |}
        else
            let mutable peakAbs = 0.0
            let mutable sumSquares = 0.0
            let mutable sumAbs = 0.0

            for index in 0 .. data.Length - 1 do
                let value = float data[index]
                let absValue = Math.Abs(value)
                peakAbs <- max peakAbs absValue
                sumSquares <- sumSquares + value * value
                sumAbs <- sumAbs + absValue

            {| PeakAbs = peakAbs
               Rms = Math.Sqrt(sumSquares / float data.Length)
               MeanAbs = sumAbs / float data.Length
               DurationSeconds = float data.Length / float sampleRate |}

    let private writeMono16Span (path: string) (sampleRate: int) (data: Span<float32>) =
        let dataBytes = data.Length * sizeof<int16>
        let audioStats = stats sampleRate data
        let wavPreviewGain =
            if audioStats.PeakAbs > 1e-8 && audioStats.PeakAbs < 0.5 then
                min 80.0 (0.85 / audioStats.PeakAbs)
            else
                1.0

        match Path.GetDirectoryName(Path.GetFullPath(path)) with
        | null | "" -> ()
        | directory -> Directory.CreateDirectory(directory) |> ignore

        use stream = File.Create(path)
        use writer = new BinaryWriter(stream, Encoding.ASCII, leaveOpen = false)
        writeAscii writer "RIFF"
        writer.Write(36 + dataBytes)
        writeAscii writer "WAVE"
        writeAscii writer "fmt "
        writer.Write(16)
        writer.Write(int16 1)
        writer.Write(int16 1)
        writer.Write(sampleRate)
        writer.Write(sampleRate * sizeof<int16>)
        writer.Write(int16 sizeof<int16>)
        writer.Write(int16 16)
        writeAscii writer "data"
        writer.Write(dataBytes)

        for index in 0 .. data.Length - 1 do
            writer.Write(clamp16 (float data[index] * wavPreviewGain))

        { Samples = data.Length
          DurationSeconds = audioStats.DurationSeconds
          PeakAbs = audioStats.PeakAbs
          Rms = audioStats.Rms
          MeanAbs = audioStats.MeanAbs
          WavPreviewGain = wavPreviewGain
          WavPeakAbs = min 1.0 (audioStats.PeakAbs * wavPreviewGain) }

    let writeMono16 (path: string) (sampleRate: int) (tensor: DenseTensor<float32>) =
        if isContiguous tensor then
            writeMono16Span path sampleRate (tensor.Buffer.Span.Slice(0, elementCount tensor.Dimensions))
        else
            let data = Enumerable.ToArray(tensor)
            writeMono16Span path sampleRate (data.AsSpan())

