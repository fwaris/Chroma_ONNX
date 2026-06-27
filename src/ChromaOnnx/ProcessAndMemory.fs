namespace ChromaOnnx

open System
open System.Diagnostics

type ProcessResult =
    { ExitCode: int
      Stdout: string
      Stderr: string }

module ProcessRunner =
    let run (fileName: string) (arguments: string seq) (workingDirectory: string) =
        task {
            let startInfo = ProcessStartInfo()
            startInfo.FileName <- fileName
            startInfo.WorkingDirectory <- workingDirectory
            startInfo.UseShellExecute <- false
            startInfo.RedirectStandardOutput <- true
            startInfo.RedirectStandardError <- true
            startInfo.CreateNoWindow <- true
            startInfo.Environment["TOKENIZERS_PARALLELISM"] <- "false"

            for argument in arguments do
                startInfo.ArgumentList.Add(argument)

            use proc = new Process()
            proc.StartInfo <- startInfo

            if not (proc.Start()) then
                invalidOp $"Failed to start process: {fileName}"

            let stdoutTask = proc.StandardOutput.ReadToEndAsync()
            let stderrTask = proc.StandardError.ReadToEndAsync()
            do! proc.WaitForExitAsync()
            let! stdout = stdoutTask
            let! stderr = stderrTask
            return { ExitCode = proc.ExitCode; Stdout = stdout; Stderr = stderr }
        }

module RuntimeMemory =
    type Snapshot =
        { ProcessId: int
          WorkingSetGb: float
          PrivateGb: float
          GcHeapGb: float
          GpuMb: Nullable<int>
          GpuGb: Nullable<float> }

    let private gb bytes =
        Math.Round(float bytes / 1024.0 / 1024.0 / 1024.0, 3)

    let private tryCurrentGpuMemoryMb () =
        try
            let processId = Process.GetCurrentProcess().Id
            let startInfo = ProcessStartInfo()
            startInfo.FileName <- "nvidia-smi"
            startInfo.UseShellExecute <- false
            startInfo.RedirectStandardOutput <- true
            startInfo.RedirectStandardError <- true
            startInfo.CreateNoWindow <- true
            startInfo.ArgumentList.Add("--query-compute-apps=pid,used_memory")
            startInfo.ArgumentList.Add("--format=csv,noheader,nounits")

            use proc = new Process()
            proc.StartInfo <- startInfo
            if not (proc.Start()) then
                None
            elif proc.WaitForExit(1000) then
                let output = proc.StandardOutput.ReadToEnd()
                output.Split([| '\r'; '\n' |], StringSplitOptions.RemoveEmptyEntries)
                |> Array.choose (fun line ->
                    let parts = line.Split(',', StringSplitOptions.TrimEntries ||| StringSplitOptions.RemoveEmptyEntries)
                    if parts.Length >= 2 then
                        match Int32.TryParse(parts[0]), Int32.TryParse(parts[1]) with
                        | (true, pid), (true, usedMb) when pid = processId -> Some usedMb
                        | _ -> None
                    else
                        None)
                |> Array.tryHead
            else
                try proc.Kill(true) with _ -> ()
                None
        with _ ->
            None
    let current () =
        let proc = Process.GetCurrentProcess()
        proc.Refresh()
        let gpuMb = tryCurrentGpuMemoryMb ()
        { ProcessId = proc.Id
          WorkingSetGb = gb proc.WorkingSet64
          PrivateGb = gb proc.PrivateMemorySize64
          GcHeapGb = gb (GC.GetTotalMemory(false))
          GpuMb = gpuMb |> Option.toNullable
          GpuGb = gpuMb |> Option.map (fun value -> Math.Round(float value / 1024.0, 3)) |> Option.toNullable }

