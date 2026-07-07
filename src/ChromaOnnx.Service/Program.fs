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
            let options = VoiceAgentWebApp.bindOptions builder.Configuration
            printfn "GemmaVoiceAgent initializing runtime."
            printfn "Configured Gemma model: %s" options.Gemma.ModelDir
            printfn "Configured Gemma runtime: %s" options.Gemma.Runtime
            printfn "Configured TTS model: %s" options.Tts.ModelDir
            printfn "Configured TTS runtime: %s" options.Tts.Runtime
            printfn "Configured TTS provider: %s" options.Tts.ExecutionProvider
            Console.Out.Flush()

            let agentRuntime = new GemmaVoiceAgentRuntime(options)
            builder.Services.AddSingleton<IVoiceAgentRuntime>(agentRuntime) |> ignore

            let app = builder.Build()
            let agentRuntime = app.Services.GetRequiredService<IVoiceAgentRuntime>()
            app.Lifetime.ApplicationStopped.Register(fun () ->
                match agentRuntime with
                | :? IDisposable as disposable -> disposable.Dispose()
                | _ -> ()) |> ignore
            VoiceAgentWebApp.map app agentRuntime |> ignore

            let agentStatus = agentRuntime.Status()
            printfn "GemmaVoiceAgent service listening."
            printfn "Gemma status: %s" agentStatus.Gemma.Message
            printfn "STT status: %s" agentStatus.Stt.Message
            printfn "TTS status: %s" agentStatus.Tts.Message
            printfn "Agent status: %s" agentStatus.Message
            app.Run()
            0
        with ex ->
            eprintfn "%O" ex
            1
