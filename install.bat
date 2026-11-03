@echo off
rem GEM-X Studio installer: double-click, or run "install.bat -Help" for options.
setlocal
if /I "%~1"=="-Help" (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Help '%~dp0scripts\windows\install.ps1' -Detailed"
    exit /b 0
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\install.ps1" %*
set CODE=%ERRORLEVEL%
if not "%CODE%"=="0" (
    echo.
    echo Installation did not finish. Scroll up for the error.
)
rem Keep the window open when launched by double-click.
echo %CMDCMDLINE% | find /I "%~0" >nul && pause
exit /b %CODE%
