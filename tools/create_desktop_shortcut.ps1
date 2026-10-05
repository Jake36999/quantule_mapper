param(
    [int]$Port = 8000
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$deployScript = Join-Path $repoRoot 'tools\deploy_webapp.ps1'
$desktop = [Environment]::GetFolderPath('Desktop')
$shortcutPath = Join-Path $desktop 'Quantule Mapper Webapp.lnk'
$powershellExe = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'

if (-not (Test-Path $deployScript)) {
    throw "Missing deploy script: $deployScript"
}

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $powershellExe
$shortcut.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$deployScript`" -Port $Port"
$shortcut.WorkingDirectory = $repoRoot
$shortcut.Description = 'Build, deploy, launch, and open the Quantule Mapper webapp.'
$shortcut.IconLocation = "$env:SystemRoot\System32\SHELL32.dll,220"
$shortcut.Save()

Write-Host "[shortcut] Created $shortcutPath"
