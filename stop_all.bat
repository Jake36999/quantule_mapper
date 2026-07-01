@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo [INFO] Requesting graceful orchestrator stop...
type nul > stop_after_gen.txt

set "WAIT_SECONDS=%~1"
if "%WAIT_SECONDS%"=="" set "WAIT_SECONDS=20"
echo [INFO] Waiting %WAIT_SECONDS%s before force-stop fallback...
timeout /t %WAIT_SECONDS% /nobreak >nul

echo [INFO] Stopping ASTE Python processes (api/orchestrator/worker)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$targets = @('uvicorn app:app','adaptive_hunt_orchestrator.py','worker_daemon.py'); $procs = Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\"; foreach($p in $procs){ $cmd = [string]$p.CommandLine; if($null -ne $cmd){ foreach($t in $targets){ if($cmd -like ('*' + $t + '*')){ try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop; Write-Host ('[KILLED] PID=' + $p.ProcessId + ' CMD=' + $cmd) } catch { Write-Host ('[SKIP] PID=' + $p.ProcessId + ' (' + $_.Exception.Message + ')') }; break } } } }"

if exist stop_after_gen.txt del /q stop_after_gen.txt
echo [INFO] Stop sequence complete.
exit /b 0
