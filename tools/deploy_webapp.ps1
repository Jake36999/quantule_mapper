param(
    [int]$Port = 8000,
    [switch]$NoBrowser,
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$uiRoot = Join-Path $repoRoot 'UI\components'
$pythonExe = Join-Path $repoRoot '.venv\Scripts\python.exe'
$logDir = Join-Path $repoRoot 'runtime_logs'
$outLog = Join-Path $logDir 'webapp_stdout.log'
$errLog = Join-Path $logDir 'webapp_stderr.log'
$url = "http://127.0.0.1:$Port/"

function Stop-WebappProcess {
    param([int]$TargetPort)

    $processes = Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'pythonw.exe'" |
        Where-Object {
            $cmd = [string]$_.CommandLine
            $cmd -match 'uvicorn\s+app:app' -and $cmd -match "--port\s+$TargetPort(\s|$)"
        }

    foreach ($proc in $processes) {
        Write-Host "[deploy] Stopping existing webapp PID=$($proc.ProcessId)"
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
    }

    $listener = Get-NetTCPConnection -LocalPort $TargetPort -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($listener) {
        $owner = Get-CimInstance Win32_Process -Filter "ProcessId = $($listener.OwningProcess)" -ErrorAction SilentlyContinue
        $cmd = [string]$owner.CommandLine
        if ($cmd -match 'uvicorn\s+app:app') {
            Write-Host "[deploy] Stopping webapp port owner PID=$($listener.OwningProcess)"
            Stop-Process -Id $listener.OwningProcess -Force -ErrorAction SilentlyContinue
        } else {
            throw "Port $TargetPort is already in use by PID=$($listener.OwningProcess). Command: $cmd"
        }
    }
}

function Wait-WebappReady {
    param(
        [string]$StatusUrl,
        [int]$Attempts = 30
    )

    for ($i = 0; $i -lt $Attempts; $i++) {
        try {
            $response = Invoke-WebRequest -Uri $StatusUrl -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300) {
                return $true
            }
        } catch {
            Start-Sleep -Milliseconds 750
        }
    }
    return $false
}

Push-Location $repoRoot
try {
    if (-not (Test-Path $pythonExe)) {
        throw "Missing virtual environment Python at $pythonExe"
    }

    New-Item -ItemType Directory -Force $logDir | Out-Null

    if (-not $SkipBuild) {
        Write-Host "[deploy] Building React dashboard..."
        Push-Location $uiRoot
        try {
            if (-not (Test-Path 'node_modules')) {
                Write-Host "[deploy] Installing frontend dependencies..."
                npm install --legacy-peer-deps
            }
            npm run build
        } finally {
            Pop-Location
        }

        Write-Host "[deploy] Publishing frontend build..."
        New-Item -ItemType Directory -Force (Join-Path $repoRoot 'build') | Out-Null
        Copy-Item -Recurse -Force (Join-Path $uiRoot 'build\*') (Join-Path $repoRoot 'build')
    }

    Stop-WebappProcess -TargetPort $Port

    Write-Host "[deploy] Starting webapp at $url"
    $process = Start-Process -FilePath $pythonExe `
        -ArgumentList @('-m', 'uvicorn', 'app:app', '--host', '127.0.0.1', '--port', [string]$Port) `
        -WorkingDirectory $repoRoot `
        -RedirectStandardOutput $outLog `
        -RedirectStandardError $errLog `
        -WindowStyle Hidden `
        -PassThru

    if (-not (Wait-WebappReady -StatusUrl "$url`api/run/status")) {
        throw "Webapp did not become healthy. Check $outLog and $errLog."
    }

    Write-Host "[deploy] Webapp healthy. PID=$($process.Id)"
    if (-not $NoBrowser) {
        Start-Process $url | Out-Null
    }
    Write-Host "[deploy] Complete: $url"
} finally {
    Pop-Location
}
