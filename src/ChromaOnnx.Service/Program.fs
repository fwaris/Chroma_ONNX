namespace ChromaOnnx.Service

open System
open Microsoft.AspNetCore.Builder
open Microsoft.Extensions.DependencyInjection
open ChromaOnnx.SpeechToSpeech

module Program =
    [<EntryPoint>]
    let main argv =
        try
            let builder = WebApplication.CreateBuilder(argv)
            let options = S2sWebApp.bindOptions builder.Configuration
            let gemmaOptions = S2sWebApp.bindGemmaOptions builder.Configuration
            printfn "ChromaS2SONNX initializing runtime."
            printfn "Configured model: %s" options.ModelDir
            printfn "Configured bundle: %s" options.BundleDir
            printfn "Configured memory mode: %s" options.MemoryMode
            printfn "Configured generation mode: %s" options.GenerationMode
            printfn "Configured sampling algorithm: %s" options.SamplingAlgorithm
            if options.MemoryMode.Trim().Equals("resident-merged", StringComparison.OrdinalIgnoreCase) then
                printfn "Resident merged mode loads the merged ONNX session before the service starts listening; first startup can take several minutes and use substantial host RAM."
            Console.Out.Flush()

            let runtime = new ChromaS2sRuntime(options)
            builder.Services.AddSingleton<IS2sRuntime>(runtime) |> ignore
            let agentRuntime = new GemmaChromaAgentRuntime(gemmaOptions, runtime)
            builder.Services.AddSingleton<IAgentRuntime>(agentRuntime) |> ignore

            let app = builder.Build()
            let runtime = app.Services.GetRequiredService<IS2sRuntime>()
            let agentRuntime = app.Services.GetRequiredService<IAgentRuntime>()
            app.Lifetime.ApplicationStopped.Register(fun () ->
                match agentRuntime with
                | :? IDisposable as disposable -> disposable.Dispose()
                | _ -> ()
                (runtime :?> IDisposable).Dispose()) |> ignore
            S2sWebApp.mapWithAgent app runtime agentRuntime |> ignore

            let status = runtime.Status()
            let agentStatus = agentRuntime.Status()
            printfn "ChromaS2SONNX standalone service listening."
            printfn "Model: %s" status.ModelDir
            printfn "Bundle: %s" status.BundleDir
            printfn "Execution provider: %s" status.ExecutionProvider
            printfn "Memory mode: %s" status.MemoryMode
            printfn "Generation mode: %s" status.GenerationMode
            printfn "Sampling algorithm: %s" status.SamplingAlgorithm
            printfn "Bundle graph mode: %s" status.BundleGraphMode
            printfn "Bundle thinker feature mode: %s, max audio items: %d" status.BundleThinkerFeatureMode status.ThinkerMaxAudioItems
            printfn "S2S bundle status: %s" status.Message
            printfn "Gemma agent model: %s (%s)" agentStatus.ModelDir agentStatus.Variant
            printfn "Gemma agent status: %s" agentStatus.Message
            app.Run()
            0
        with ex ->
            eprintfn "%O" ex
            1
