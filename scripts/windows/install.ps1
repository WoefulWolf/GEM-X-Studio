<#
.SYNOPSIS
  One-shot installer for GEM-X and GEM-X Studio on Windows.

.DESCRIPTION
  Creates a self-contained environment inside the repository:
    .tools\uv         uv package manager (downloaded if not already on PATH)
    .tools\node       portable Node.js, only if Node is missing (to build the UI)
    .tools\python     Python 3.12 interpreter (managed by uv)
    .venv             virtual environment with PyTorch, GEM-X and Studio
  It picks the right PyTorch build for your NVIDIA driver, builds the web UI,
  optionally downloads models, and adds Desktop / Start Menu shortcuts.
  Nothing is installed system-wide. Re-running it is safe and fast.

.EXAMPLE
  .\install.bat
  .\install.bat -Models body_pose,body_motion -Yes
  .\install.bat -Torch cpu -NoShortcut
#>
[CmdletBinding()]
param(
    [ValidateSet('auto', 'cpu', 'cu126', 'cu128', 'cu130')]
    [string]$Torch = 'auto',
    # body_pose, hand_pose, body_motion, all, none. Empty = ask.
    [string[]]$Models = @(),
    [switch]$NoShortcut,
    [switch]$Yes,
    [switch]$Launch
)

# 'Continue': Windows PowerShell 5.1 turns any native stderr output into a terminating
# error under 'Stop'. Native calls are checked through $LASTEXITCODE instead.
$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'   # Invoke-WebRequest is much faster without it
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Tools = Join-Path $Root '.tools'
$Venv = Join-Path $Root '.venv'
$Python = Join-Path $Venv 'Scripts\python.exe'
$PythonW = Join-Path $Venv 'Scripts\pythonw.exe'
$Web = Join-Path $Root 'studio\web'
$Static = Join-Path $Root 'studio\gem_studio\static'
$PythonVersion = '3.12'
$script:Step = 0
$StepCount = 7

function Write-Banner {
    Write-Host ''
    Write-Host '  ============================================' -ForegroundColor DarkGreen
    Write-Host '     GEM-X Studio installer for Windows' -ForegroundColor Green
    Write-Host '     Body Pose  |  Hand Pose  |  Body Motion' -ForegroundColor Gray
    Write-Host '  ============================================' -ForegroundColor DarkGreen
    Write-Host "  Install folder: $Root" -ForegroundColor DarkGray
    Write-Host ''
}

function Write-Step([string]$Text) {
    $script:Step++
    Write-Host ''
    Write-Host ("[{0}/{1}] {2}" -f $script:Step, $StepCount, $Text) -ForegroundColor Cyan
}

function Write-Ok([string]$Text) { Write-Host "      OK  $Text" -ForegroundColor Green }
function Write-Info([string]$Text) { Write-Host "      $Text" -ForegroundColor Gray }
function Write-Warn([string]$Text) { Write-Host "      !   $Text" -ForegroundColor Yellow }

function Fail([string]$Text) {
    Write-Host ''
    Write-Host "  INSTALL FAILED: $Text" -ForegroundColor Red
    Write-Host '  Fix the problem above and run install.bat again; finished steps are skipped.' -ForegroundColor Red
    exit 1
}

function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments, [string]$What)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { Fail "$What (exit code $LASTEXITCODE)" }
}

function Get-Download([string]$Url, [string]$Target) {
    Write-Info "Downloading $Url"
    $partial = "$Target.part"
    try {
        Invoke-WebRequest -Uri $Url -OutFile $partial -UseBasicParsing -ErrorAction Stop
    } catch {
        Fail "Download failed: $Url ($($_.Exception.Message))"
    }
    Move-Item -Force $partial $Target -ErrorAction Stop
}

function Ask([string]$Question, [string]$Default = 'Y') {
    if ($Yes) { return $Default }
    $answer = Read-Host "      $Question"
    if ([string]::IsNullOrWhiteSpace($answer)) { return $Default }
    return $answer.Trim()
}

# ---------------------------------------------------------------------------
Write-Banner

if ([Environment]::OSVersion.Platform -ne 'Win32NT' -or -not [Environment]::Is64BitOperatingSystem) {
    Fail 'This installer requires 64-bit Windows 10 or 11.'
}

# 1. uv ----------------------------------------------------------------------
Write-Step 'Package manager (uv)'
$uv = $null
$localUv = Join-Path $Tools 'uv\uv.exe'
if (Test-Path $localUv) {
    $uv = $localUv
} else {
    $found = Get-Command uv -ErrorAction SilentlyContinue
    if ($found) { $uv = $found.Source }
}
if (-not $uv) {
    New-Item -ItemType Directory -Force (Join-Path $Tools 'uv') | Out-Null
    $zip = Join-Path $Tools 'uv.zip'
    Get-Download 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' $zip
    Expand-Archive -ErrorAction Stop -Force $zip (Join-Path $Tools 'uv')
    Remove-Item $zip
    $uv = $localUv
}
$env:UV_LINK_MODE = 'copy'
# Keep the managed Python interpreter inside the repository as well.
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Tools 'python'
Write-Ok ("uv " + ((& $uv --version) -replace '^uv ', ''))

# 2. Python environment ------------------------------------------------------
Write-Step "Python $PythonVersion environment (.venv)"
if (Test-Path $Python) {
    $existing = (& $Python -c "import sys; print('%d.%d' % sys.version_info[:2])") 2>$null
    if ($existing -ne $PythonVersion) {
        Write-Warn ".venv uses Python $existing; recreating it with $PythonVersion"
        Remove-Item -Recurse -Force $Venv
    }
}
if (-not (Test-Path $Python)) {
    Invoke-Checked $uv @('venv', $Venv, '--python', $PythonVersion, '--seed') 'Creating the virtual environment'
}
Write-Ok (& $Python --version)

# 3. PyTorch -----------------------------------------------------------------
Write-Step 'PyTorch'
$backend = $Torch
$gpuName = $null
if ($backend -eq 'auto') {
    $backend = 'cpu'
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($smi) {
        $smiText = (& $smi.Source) -join "`n"
        $gpuName = ((& $smi.Source --query-gpu=name --format=csv,noheader) | Select-Object -First 1)
        $driver = ((& $smi.Source --query-gpu=driver_version --format=csv,noheader) | Select-Object -First 1)
        $cuda = 0
        # Older drivers print "CUDA Version: 12.8"; newer ones "CUDA UMD Version: 13.5".
        if ($smiText -match 'CUDA(?: UMD)? Version:\s*(\d+)\.(\d+)') {
            $cuda = [int]$Matches[1] * 100 + [int]$Matches[2]
        } elseif ($driver -match '^(\d+)') {
            $major = [int]$Matches[1]
            if ($major -ge 580) { $cuda = 1300 } elseif ($major -ge 570) { $cuda = 1208 } elseif ($major -ge 560) { $cuda = 1206 }
        }
        if ($gpuName) {
            Write-Info ("NVIDIA GPU: {0}, driver {1} (CUDA {2}.{3})" -f $gpuName, $driver, [math]::Floor($cuda / 100), ($cuda % 100))
            # torch 2.10 (the version GEM is tested with) ships for cu130/cu128/cu126.
            if ($cuda -ge 1300) { $backend = 'cu130' }
            elseif ($cuda -ge 1208) { $backend = 'cu128' }
            elseif ($cuda -ge 1206) { $backend = 'cu126' }
            else { Write-Warn 'Your NVIDIA driver is too old for CUDA 12.6+. Update it for GPU speed; using CPU for now.' }
        }
    } else {
        Write-Warn 'No NVIDIA GPU detected - installing the CPU build (inference will be slow).'
    }
}
# Pinned to the 2.10 series GEM is validated with; newer builds have caused CUDA
# faults (illegal memory access, driver resets) on some GPUs.
$TorchSpec = 'torch>=2.10,<2.11'
$haveTorch = (& $Python -c "import torch; v = torch.__version__; print(((v.split('+') + ['cpu'])[1]) if v.startswith('2.10.') else 'other')" 2>$null)
if ($haveTorch -eq $backend) {
    Write-Ok "PyTorch 2.10 already installed ($backend)"
} else {
    Write-Info "Installing PyTorch 2.10 ($backend) - this is a large download, please wait..."
    Invoke-Checked $uv @('pip', 'install', '--python', $Python, $TorchSpec,
        '--index-url', "https://download.pytorch.org/whl/$backend", '--reinstall-package', 'torch') 'Installing PyTorch'
    Write-Ok "PyTorch installed ($backend)"
}

# 4. GEM-X + Studio ----------------------------------------------------------
Write-Step 'GEM-X and Studio packages'
Push-Location $Root
try {
    Invoke-Checked $uv @('pip', 'install', '--python', $Python, '-e', '.[video,hand]', '-e', './studio[native]') 'Installing GEM-X'
} finally { Pop-Location }
# GEM's person/hand detectors run on ONNX Runtime, which is CPU-only by default and
# would max out the processor. onnxruntime-gpu targets CUDA 13, matching the cu13x
# PyTorch builds; both packages own the same "onnxruntime" folder, so the CPU one
# is removed and the GPU one reinstalled to restore any files that removal deleted.
$torchTag = (& $Python -c "import torch; print((torch.__version__.split('+') + ['cpu'])[1])" 2>$null)
if ($torchTag -match '^cu13') {
    Write-Info 'Enabling GPU detectors (onnxruntime-gpu)...'
    & $uv pip uninstall --python $Python onnxruntime 2>$null | Out-Null
    Invoke-Checked $uv @('pip', 'install', '--python', $Python, 'onnxruntime-gpu>=1.23,<2', '--reinstall-package', 'onnxruntime-gpu') 'Installing onnxruntime-gpu'
}
$check = & $Python -c "import torch, gem, gem_studio, webview; print(torch.__version__, torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { Fail 'Python packages did not import correctly.' }
$check = ($check | Select-Object -Last 1)
Write-Ok "Packages ready (torch $($check.Split(' ')[0]), CUDA available: $($check.Split(' ')[1]))"
# uv's venv launchers are console programs, even pythonw.exe, so the app window would
# come with an empty console window. Put the real windowless interpreter (and the DLLs
# it loads from its own folder) into the venv; pyvenv.cfg next to it keeps the venv active.
$venvHome = ((Get-Content (Join-Path $Venv 'pyvenv.cfg')) | Where-Object { $_ -match '^home\s*=' }) -replace '^home\s*=\s*', ''
$copied = 0
foreach ($file in @('pythonw.exe', 'python3.dll', ('python' + ($PythonVersion -replace '\.', '') + '.dll'), 'vcruntime140.dll', 'vcruntime140_1.dll')) {
    $source = Join-Path $venvHome $file
    if (Test-Path $source) {
        try {
            Copy-Item $source (Join-Path $Venv 'Scripts') -Force -ErrorAction Stop
            $copied++
        } catch {
            Write-Warn "Could not update $file (is GEM-X Studio running? close it and re-run install.bat)"
        }
    }
}
# pythonw is a GUI program: PowerShell's & would not wait for it, so use Start-Process.
$probe = Start-Process -FilePath $PythonW -ArgumentList '-c', '"import sys, gem_studio; sys.exit(0 if sys.prefix.lower().endswith(''.venv'') else 1)"' -Wait -PassThru -NoNewWindow
if ($probe.ExitCode -eq 0) { Write-Ok 'Windowless launcher ready' } else { Write-Warn 'Windowless launcher check failed; studio.bat --browser still works' }

# 5. Web UI ------------------------------------------------------------------
Write-Step 'Studio user interface'
$needsBuild = -not (Test-Path (Join-Path $Static 'index.html'))
if (-not $needsBuild) {
    $built = (Get-Item (Join-Path $Static 'index.html')).LastWriteTimeUtc
    $newest = Get-ChildItem -Recurse -File (Join-Path $Web 'src'), (Join-Path $Web 'package.json'), (Join-Path $Web 'index.html') |
        Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
    $needsBuild = $newest.LastWriteTimeUtc -gt $built
}
if ($needsBuild) {
    $npm = $null
    $localNode = Join-Path $Tools 'node'
    if (Test-Path (Join-Path $localNode 'npm.cmd')) {
        $npm = Join-Path $localNode 'npm.cmd'
    } else {
        $found = Get-Command npm.cmd -ErrorAction SilentlyContinue
        $nodeOk = $false
        if ($found) {
            $major = [int](((& node --version) -replace '^v', '').Split('.')[0])
            $nodeOk = $major -ge 20
        }
        if ($nodeOk) { $npm = $found.Source }
    }
    if (-not $npm) {
        Write-Info 'Node.js 20+ not found; fetching a portable copy (used only to build the UI).'
        $index = Invoke-RestMethod 'https://nodejs.org/dist/index.json' -ErrorAction Stop
        $lts = ($index | Where-Object { $_.lts } | Select-Object -First 1).version
        $name = "node-$lts-win-x64"
        $zip = Join-Path $Tools "$name.zip"
        New-Item -ItemType Directory -Force $Tools | Out-Null
        Get-Download "https://nodejs.org/dist/$lts/$name.zip" $zip
        Expand-Archive -ErrorAction Stop -Force $zip $Tools
        Remove-Item $zip
        if (Test-Path $localNode) { Remove-Item -Recurse -Force $localNode }
        Rename-Item (Join-Path $Tools $name) 'node'
        $npm = Join-Path $localNode 'npm.cmd'
    }
    $env:PATH = (Split-Path $npm) + ';' + $env:PATH
    Push-Location $Web
    try {
        Write-Info 'Installing UI dependencies...'
        Invoke-Checked $npm @('ci', '--no-audit', '--no-fund', '--loglevel=error') 'npm ci'
        Write-Info 'Building UI...'
        Invoke-Checked $npm @('run', 'build', '--silent') 'npm run build'
    } finally { Pop-Location }
    Write-Ok 'UI built'
} else {
    Write-Ok 'UI is up to date'
}

# 6. Models ------------------------------------------------------------------
Write-Step 'Models'
& $Python -m gem_studio.models status
if ($Models.Count -eq 0) {
    Write-Host ''
    Write-Info 'Download models now? You can also do this later from the Models page in Studio.'
    Write-Info '  1) Body Pose    (~4 GB)'
    Write-Info '  2) Hand Pose    (~11.6 GB)'
    Write-Info '  3) Body Motion  (~7.5 GB, includes Body Pose)'
    Write-Info '  A) Everything   (~19 GB)'
    Write-Info '  S) Skip'
    $choice = Ask 'Choose one or more, e.g. "1 3" [S]:' 'S'
    $map = @{ '1' = 'body_pose'; '2' = 'hand_pose'; '3' = 'body_motion'; 'A' = 'all' }
    foreach ($token in ($choice.ToUpper() -split '[\s,]+')) {
        if ($map.ContainsKey($token)) { $Models += $map[$token] }
    }
}
$Models = @($Models | Where-Object { $_ -and $_ -ne 'none' })
if ($Models.Count -gt 0) {
    & $Python -m gem_studio.models download @Models
    if ($LASTEXITCODE -ne 0) { Write-Warn 'Some downloads failed; retry them from the Models page in Studio.' }
} else {
    Write-Info 'Skipped. Open the Models page in Studio to download with one click.'
}

# 7. Shortcuts ---------------------------------------------------------------
Write-Step 'Shortcuts'
if ($NoShortcut) {
    Write-Info 'Skipped (-NoShortcut).'
} else {
    $shell = New-Object -ComObject WScript.Shell
    $icon = Join-Path $Root 'studio\gem_studio\icon.ico'
    $places = @(
        [Environment]::GetFolderPath('Desktop'),
        (Join-Path ([Environment]::GetFolderPath('Programs')) '')
    )
    foreach ($place in $places) {
        $link = $shell.CreateShortcut((Join-Path $place 'GEM-X Studio.lnk'))
        $link.TargetPath = $PythonW
        $link.Arguments = '-m gem_studio'
        $link.WorkingDirectory = $Root
        $link.Description = 'GEM-X Studio - body, hand and motion estimation'
        if (Test-Path $icon) { $link.IconLocation = "$icon,0" }
        $link.Save()
    }
    Write-Ok 'Desktop and Start Menu shortcuts created'
}

# WebView2 is built into Windows 11 and current Windows 10; Studio falls back to the browser otherwise.
$webview2 = @(
    'HKLM:\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}',
    'HKCU:\SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}'
) | Where-Object { Test-Path $_ }
if (-not $webview2) {
    Write-Warn 'Microsoft Edge WebView2 Runtime not found; Studio will open in your browser instead.'
    Write-Warn 'Install it from https://developer.microsoft.com/microsoft-edge/webview2/ for the app window.'
}

Write-Host ''
Write-Host '  Installation complete!' -ForegroundColor Green
Write-Host '  Start GEM-X Studio from the Desktop shortcut, or run studio.bat' -ForegroundColor Gray
Write-Host ''
if ($Launch -or (-not $Yes -and ((Ask 'Launch GEM-X Studio now? [Y/n]' 'Y') -match '^[Yy]'))) {
    Start-Process -FilePath $PythonW -ArgumentList '-m', 'gem_studio' -WorkingDirectory $Root
}
