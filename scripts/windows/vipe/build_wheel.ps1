<#
.SYNOPSIS
  Build the Windows ViPE wheel that install.bat -Vipe installs.

.DESCRIPTION
  ViPE (https://github.com/nv-tlabs/vipe) is published as source only, and pip
  compiles its CUDA extensions during installation. This script does that once,
  for GEM-X Studio's environment (Python 3.12, PyTorch 2.10, CUDA 13.0):

    1. checks out ViPE at the pinned commit and applies windows.patch
       (MSVC compiler flags, torch/types.h in CUDA sources, int64_t indices:
       Windows' long is 32-bit);
    2. assembles a private CUDA 13.0 toolkit from NVIDIA's pip packages;
    3. compiles with Visual Studio's C++ tools against .venv's PyTorch.

  Needs Visual Studio 2022 (or Build Tools) with "Desktop development with C++",
  git, uv and an installed .venv (install.bat). Rebuild after changing the
  PyTorch version. Nothing is installed system-wide; the work folder can be
  deleted afterwards.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\windows\vipe\build_wheel.ps1
#>
[CmdletBinding()]
param(
    # GPU architectures: CUDA 13 supports Turing (7.5) and newer; +PTX for future GPUs.
    [string]$Arch = '7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX',
    [string]$Work = (Join-Path $env:TEMP 'gem-vipe-build'),
    # Default: scripts\windows\vipe\dist ($PSScriptRoot is empty in Windows PowerShell 5.1 defaults).
    [string]$Out = '',
    # Parallel compile jobs; CUDA compilation is CPU and memory heavy.
    [int]$Jobs = 6
)

$ErrorActionPreference = 'Continue'
if (-not $Out) { $Out = Join-Path $PSScriptRoot 'dist' }
$ProgressPreference = 'SilentlyContinue'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$Commit = '8c9f36144e08d8f8c8cf60d20ad70c100d9cff07'

function Fail([string]$Text) { Write-Host "BUILD FAILED: $Text" -ForegroundColor Red; exit 1 }

if (-not (Test-Path $Python)) { Fail 'Run install.bat first: the wheel is built against .venv''s PyTorch.' }
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = if (Test-Path $vswhere) { & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath }
if (-not $vs) { Fail 'Visual Studio C++ build tools not found (install "Desktop development with C++").' }
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) { $uv = Join-Path $Root '.tools\uv\uv.exe' }
if (-not (Test-Path $uv)) { Fail 'uv not found.' }

New-Item -ItemType Directory -Force $Work, $Out | Out-Null
$env:UV_CACHE_DIR = Join-Path $Work 'uv-cache'

# 1. ViPE source at the pinned commit, patched for Windows.
$source = Join-Path $Work 'vipe'
if (-not (Test-Path (Join-Path $source '.git'))) {
    git clone --quiet https://github.com/nv-tlabs/vipe.git $source
    if ($LASTEXITCODE -ne 0) { Fail 'git clone failed' }
}
git -C $source checkout --quiet --force $Commit
git -C $source clean --quiet -fdx -e csrc/include
git -C $source apply --whitespace=nowarn (Join-Path $PSScriptRoot 'windows.patch')
if ($LASTEXITCODE -ne 0) { Fail 'windows.patch did not apply' }

# 2. CUDA 13.0 compiler, headers and libraries from pip (laid out as nvidia\cu13).
$cudaPip = Join-Path $Work 'cuda-pip'
& $uv pip install --quiet --target $cudaPip 'nvidia-cuda-nvcc==13.0.*' 'nvidia-cuda-runtime==13.0.*' `
    'nvidia-cuda-cccl==13.0.*' 'nvidia-cuda-crt==13.0.*' 'nvidia-nvvm==13.0.*' 'nvidia-cublas==13.*' `
    'nvidia-cusparse==12.*' 'nvidia-cusolver==12.*'
if ($LASTEXITCODE -ne 0) { Fail 'Could not fetch the CUDA toolkit packages' }
$tools = Join-Path $Work 'tools'
& $uv pip install --quiet --target $tools wheel ninja
if ($LASTEXITCODE -ne 0) { Fail 'Could not fetch wheel/ninja' }
$cuda = Join-Path $cudaPip 'nvidia\cu13'

# 3. Compile inside the Visual Studio x64 environment.
$script = Join-Path $Work 'build.cmd'
@"
@echo off
call "$vs\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set CUDA_HOME=$cuda
set CUDA_PATH=$cuda
set PATH=$cuda\bin;$tools\bin;%PATH%
set PYTHONPATH=$tools
set TORCH_CUDA_ARCH_LIST=$Arch
set MAX_JOBS=$Jobs
set DISTUTILS_USE_SDK=1
set PIP_CACHE_DIR=$Work\pip-cache
cd /d "$source" || exit /b 1
"$Python" -m pip wheel --no-deps --no-build-isolation -w "$Out" .
"@ | Set-Content -Path $script -Encoding ascii
cmd /c "`"$script`""
if ($LASTEXITCODE -ne 0) { Fail 'Compilation failed; see the output above' }
Get-ChildItem $Out -Filter 'nvidia_vipe-*.whl' | ForEach-Object { Write-Host "Built $($_.FullName)" -ForegroundColor Green }
