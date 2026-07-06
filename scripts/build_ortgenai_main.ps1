param(
    [string]$OrtGenAiRepo = "E:\s\repos\onnxruntime-genai",
    [string]$Python = "",
    [string]$CudaHome = $env:CUDA_PATH,
    [string]$Configuration = "Release",
    [string]$BuildName = "WindowsNinjaCudaNoPython5"
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Python)) {
    $repoVenvPython = Join-Path (Split-Path -Parent $PSScriptRoot) ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $repoVenvPython) {
        $Python = $repoVenvPython
    } else {
        $Python = "python"
    }
}

if (-not (Test-Path -LiteralPath $OrtGenAiRepo)) {
    git clone --depth 1 https://github.com/microsoft/onnxruntime-genai.git $OrtGenAiRepo
} else {
    git -C $OrtGenAiRepo fetch --depth 1 origin main
    git -C $OrtGenAiRepo checkout main
    git -C $OrtGenAiRepo pull --ff-only
}

$externalDeps = Join-Path $OrtGenAiRepo "cmake\external\onnxruntime_external_deps.cmake"
$depsText = Get-Content -LiteralPath $externalDeps -Raw
if ($depsText -notmatch "GIT_SUBMODULES\s+`"`"") {
    $depsText = $depsText.Replace(
        "  GIT_TAG `${DEP_SHA1_onnxruntime_extensions}`r`n)",
        "  GIT_TAG `${DEP_SHA1_onnxruntime_extensions}`r`n  GIT_SUBMODULES `"`"`r`n)"
    )
    Set-Content -LiteralPath $externalDeps -Value $depsText -Encoding UTF8
}

if ([string]::IsNullOrWhiteSpace($CudaHome) -or -not (Test-Path -LiteralPath (Join-Path $CudaHome "bin\nvcc.exe"))) {
    throw "CUDA_HOME/CUDA_PATH must point to a CUDA toolkit containing bin\nvcc.exe."
}

$vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path -LiteralPath $vswhere)) {
    throw "vswhere.exe was not found. Install Visual Studio Build Tools with the C++ workload."
}

$vsRoot = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if ([string]::IsNullOrWhiteSpace($vsRoot)) {
    throw "No Visual Studio C++ toolchain was found."
}

$vcvars = Join-Path $vsRoot "VC\Auxiliary\Build\vcvars64.bat"
$cmake = Join-Path $vsRoot "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$ctest = Join-Path $vsRoot "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe"
$ninja = Join-Path $vsRoot "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
$nvcc = Join-Path $CudaHome "bin\nvcc.exe"

foreach ($path in @($vcvars, $cmake, $ctest, $ninja, $nvcc)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required build tool was not found: $path"
    }
}

cmd.exe /s /c "call `"$vcvars`" && set" | ForEach-Object {
    if ($_ -match "^(.*?)=(.*)$") {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
    }
}

$env:CUDACXX = $nvcc
$buildDir = Join-Path $OrtGenAiRepo "build\$BuildName"
$msvcFlags = "/DWIN32 /D_WINDOWS /EHsc /wd4875 /D_SILENCE_EXPERIMENTAL_COROUTINE_DEPRECATION_WARNINGS"

& $Python (Join-Path $OrtGenAiRepo "build.py") `
    --build_dir $buildDir `
    --config $Configuration `
    --update --build `
    --skip_tests --skip_wheel --skip_examples `
    --use_cuda --cuda_home $CudaHome `
    --cmake_generator Ninja `
    --cmake_path $cmake `
    --ctest_path $ctest `
    --parallel `
    --cmake_extra_defines `
    "ENABLE_PYTHON=OFF" `
    "ENABLE_TESTS=OFF" `
    "CMAKE_MAKE_PROGRAM=$ninja" `
    "CMAKE_CUDA_COMPILER=$nvcc" `
    "CMAKE_CXX_FLAGS=$msvcFlags"

$nativeDir = Join-Path $buildDir $Configuration
dotnet build (Join-Path $OrtGenAiRepo "src\csharp\Microsoft.ML.OnnxRuntimeGenAI.csproj") `
    -c $Configuration `
    /p:Platform=AnyCPU `
    /p:NativeBuildOutputDir=$nativeDir

Write-Host "ORT GenAI managed artifacts: $(Join-Path $OrtGenAiRepo "src\csharp\bin\$Configuration\net8.0")"
Write-Host "ORT GenAI native artifacts:  $nativeDir"
Write-Host "ORT native artifacts:        $(Join-Path $nativeDir $Configuration)"
