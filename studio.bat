@echo off
rem Start GEM-X Studio. Pass --browser to use your web browser instead of the app window.
setlocal
set ROOT=%~dp0
if not exist "%ROOT%.venv\Scripts\pythonw.exe" (
    echo GEM-X Studio is not installed yet. Running the installer first...
    call "%ROOT%install.bat"
    exit /b %ERRORLEVEL%
)
cd /d "%ROOT%"
if /I "%~1"=="--browser" (
    "%ROOT%.venv\Scripts\python.exe" -m gem_studio %*
) else (
    start "" "%ROOT%.venv\Scripts\pythonw.exe" -m gem_studio %*
)
