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

    let private stats (sampleRate: int) (data: float32 array) =
        if data.Length = 0 then
            {| PeakAbs = 0.0; Rms = 0.0; MeanAbs = 0.0; DurationSeconds = 0.0 |}
        else
            let mutable peakAbs = 0.0
            let mutable sumSquares = 0.0
            let mutable sumAbs = 0.0

            for sample in data do
                let value = float sample
                let absValue = Math.Abs(value)
                peakAbs <- max peakAbs absValue
                sumSquares <- sumSquares + value * value
                sumAbs <- sumAbs + absValue

            {| PeakAbs = peakAbs
               Rms = Math.Sqrt(sumSquares / float data.Length)
               MeanAbs = sumAbs / float data.Length
               DurationSeconds = float data.Length / float sampleRate |}

    let writeMono16 (path: string) (sampleRate: int) (tensor: DenseTensor<float32>) =
        let data = Enumerable.ToArray(tensor)
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

        for sample in data do
            writer.Write(clamp16 (float sample * wavPreviewGain))

        { Samples = data.Length
          DurationSeconds = audioStats.DurationSeconds
          PeakAbs = audioStats.PeakAbs
          Rms = audioStats.Rms
          MeanAbs = audioStats.MeanAbs
          WavPreviewGain = wavPreviewGain
          WavPeakAbs = min 1.0 (audioStats.PeakAbs * wavPreviewGain) }

