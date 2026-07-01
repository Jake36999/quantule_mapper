@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [ERROR] Missing virtual environment Python at %PY%
  echo [HINT] Create it with: python -m venv .venv
  pause
  exit /b 1
)

set "CONFIG=%~1"
if "%CONFIG%"=="" set "CONFIG=burn_in_config.json"
set "OPEN_BROWSER=%~2"
if "%OPEN_BROWSER%"=="" set "OPEN_BROWSER=1"

echo [INFO] Launching ASTE stack from %CD%
echo [INFO] API URL: http://127.0.0.1:8000/
echo [INFO] Config: %CONFIG%
echo [INFO] Worker launch: manual only (not started by this script)

if not exist "runtime_logs" mkdir "runtime_logs"

start "ASTE API" cmd /k "\"%PY%\" -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload"
timeout /t 2 /nobreak >nul

start "ASTE Orchestrator" cmd /k "\"%PY%\" orchestrator/orchestrator_service.py --config \"%CONFIG%\""
timeout /t 1 /nobreak >nul

if "%OPEN_BROWSER%"=="1" (
  timeout /t 2 /nobreak >nul
  start "" "http://127.0.0.1:8000/"
)

echo [INFO] Workers are intentionally not auto-started.
echo [INFO] Start a worker manually when ready:
echo [INFO]   set CUDA_VISIBLE_DEVICES=0 ^&^& "%PY%" worker_daemon.py

echo [INFO] Launch commands sent.
echo [INFO] To stop, run stop_all.bat
exit /b 0
