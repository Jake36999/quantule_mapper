param(
    [string]$Config = "burn_in_config.json",
    [switch]$RunPreflight,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$pythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Error "Missing virtual environment Python at $pythonExe"
}

$fileEncoding = if ($PSVersionTable.PSVersion.Major -ge 6) { 'utf8NoBOM' } else { 'utf8' }

if ($RunPreflight) {
    Write-Host "[INFO] Running preflight reset..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force simulation_data, provenance_reports, input_configs, runtime_logs, pareto_snapshots, archive_runs | Out-Null
    '[]' | Out-File -Encoding $fileEncoding backlog_queue.json
    '{}' | Out-File -Encoding $fileEncoding backlog_queue.json.claims.json
    '{}' | Out-File -Encoding $fileEncoding backlog_queue.json.workers.json
    '[]' | Out-File -Encoding $fileEncoding result_queue.json
    if (Test-Path stop_after_gen.txt) { Remove-Item stop_after_gen.txt -Force }
    Write-Host "[INFO] Preflight complete." -ForegroundColor Green
}

$logDir = Join-Path $PSScriptRoot "runtime_logs"
New-Item -ItemType Directory -Force $logDir | Out-Null

function Start-AsteWindow {
    param(
        [Parameter(Mandatory = $true)][string]$Title,
        [Parameter(Mandatory = $true)][string]$Command
    )
    Start-Process -FilePath "powershell" -WorkingDirectory $PSScriptRoot -ArgumentList @(
        "-NoExit",
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-Command", $Command
    ) -WindowStyle Normal | Out-Null
    Write-Host "[STARTED] $Title" -ForegroundColor Green
}

Write-Host "[INFO] Starting API on 127.0.0.1:8000..." -ForegroundColor Cyan
$apiCommand = "& '$pythonExe' -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload"
Start-AsteWindow -Title "ASTE API" -Command $apiCommand
Start-Sleep -Seconds 2

# Health check before continuing
$healthy = $false
for ($i = 0; $i -lt 10; $i++) {
    try {
        $resp = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/debug/terminals" -UseBasicParsing -TimeoutSec 2
        if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 300) { $healthy = $true; break }
    } catch {
        Start-Sleep -Milliseconds 750
    }
}
if (-not $healthy) {
    Write-Warning "API health check did not pass within timeout. Continuing anyway."
}

Write-Host "[INFO] Starting orchestrator with $Config..." -ForegroundColor Cyan
$orchestratorCommand = "& '$pythonExe' -m orchestrator.orchestrator_service --config '$Config'"
Start-AsteWindow -Title "ASTE Orchestrator" -Command $orchestratorCommand
Start-Sleep -Seconds 1

if (-not $NoBrowser) {
    Start-Sleep -Seconds 1
    Start-Process "http://127.0.0.1:8000/" | Out-Null
}

Write-Host "[INFO] Launch complete." -ForegroundColor Green
Write-Host "[INFO] Workers are intentionally not auto-started by launch_all.ps1." -ForegroundColor Yellow
Write-Host "[INFO] Start manually when ready:" -ForegroundColor Yellow
Write-Host "[INFO]   `$env:CUDA_VISIBLE_DEVICES='0'; & '$pythonExe' worker_daemon.py" -ForegroundColor Yellow
Write-Host "[INFO] Use .\stop_all.bat to stop all ASTE processes." -ForegroundColor Green
