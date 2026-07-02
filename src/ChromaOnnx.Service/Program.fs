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
            let runtime = new ChromaS2sRuntime(options)
            builder.Services.AddSingleton<IS2sRuntime>(runtime) |> ignore

            let app = builder.Build()
            let runtime = app.Services.GetRequiredService<IS2sRuntime>()
            app.Lifetime.ApplicationStopped.Register(fun () -> (runtime :?> IDisposable).Dispose()) |> ignore
            S2sWebApp.map app runtime |> ignore

            let status = runtime.Status()
            printfn "ChromaS2SONNX standalone service listening."
            printfn "Model: %s" status.ModelDir
            printfn "Bundle: %s" status.BundleDir
            printfn "Execution provider: %s" status.ExecutionProvider
            printfn "Memory mode: %s" status.MemoryMode
            printfn "S2S bundle status: %s" status.Message
            app.Run()
            0
        with ex ->
            eprintfn "%O" ex
            1
