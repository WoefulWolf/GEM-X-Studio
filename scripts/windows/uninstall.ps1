<#
.SYNOPSIS
  Remove the GEM-X Studio environment and shortcuts. Models, assets and your
  Studio runs are kept unless -All is given.
#>
[CmdletBinding()]
param([switch]$All)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path

foreach ($place in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) {
    $link = Join-Path $place 'GEM-X Studio.lnk'
    if (Test-Path $link) { Remove-Item $link; Write-Host "Removed $link" }
}
foreach ($name in @('.venv', '.venv-vipe', '.tools', 'studio\web\node_modules', 'studio\gem_studio\static')) {
    $path = Join-Path $Root $name
    if (Test-Path $path) { Remove-Item -Recurse -Force $path; Write-Host "Removed $path" }
}
if ($All) {
    foreach ($name in @('weights', 'assets\soma', 'studio_data')) {
        $path = Join-Path $Root $name
        if (Test-Path $path) { Remove-Item -Recurse -Force $path; Write-Host "Removed $path" }
    }
} else {
    Write-Host 'Kept weights\, assets\soma\ and studio_data\ (use -All to remove them too).'
}
$webview = Join-Path $env:LOCALAPPDATA 'GEM-X Studio'
if (Test-Path $webview) { Remove-Item -Recurse -Force $webview }
Write-Host 'GEM-X Studio uninstalled.'
