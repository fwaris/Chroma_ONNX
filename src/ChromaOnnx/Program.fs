namespace ChromaOnnx

open System

module Program =
    [<EntryPoint>]
    let main argv =
        try
            match argv |> Array.toList with
            | "inspect" :: args -> Cli.inspect args
            | "backbone" :: args -> Cli.backbone args
            | "e2e" :: args -> Cli.e2e args
            | "shared-e2e" :: args -> Cli.sharedE2e args
            | "serve" :: args -> Cli.serve args
            | "s2s-serve" :: args -> Cli.s2sServe args
            | "s2s-offline" :: args -> Cli.s2sOffline args
            | "s2s-debug-onnx" :: args -> Cli.s2sDebugOnnx args
            | "codec-encode-onnx-debug" :: args -> Cli.codecEncodeOnnxDebug args
            | "s2s-benchmark" :: args -> Cli.s2sBenchmark args
            | "s2s-memory-report" :: args -> Cli.s2sMemoryReport args
            | "s2s-compare" :: args -> Cli.s2sCompare args
            | _ ->
                Cli.printUsage()
                2
        with ex ->
            eprintfn "%O" ex
            1
