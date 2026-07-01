# ASTE V11.0 Launch And Use Guide

This guide is an operator-focused manual for launching ASTE locally, running hunts, monitoring progress, and scaling worker capacity across additional VMs.

## What Runs In ASTE

- `app.py`: FastAPI backend, WebSocket telemetry, control APIs, and static UI serving.
- `adaptive_hunt_orchestrator.py`: Orchestrator loop that dispatches/coordinates hunt work.
- `worker_daemon.py`: Worker process that claims jobs and executes simulation/evaluation.
- `burn_in_config.json`: Baseline runtime settings (batching, queue behavior, lease/heartbeat settings).
- `runtime_logs/`: Rolling logs for API, orchestrator, workers, and preflight tasks.

## Quick Launch (Windows PowerShell)

Run these in order from repo root:

```powershell
# 1) Activate environment
& .\.venv\Scripts\Activate.ps1

# 2) Preflight reset (queues + required dirs)
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { .\.venv\Scripts\Activate.ps1; $ErrorActionPreference='Stop'; New-Item -ItemType Directory -Force simulation_data, provenance_reports, input_configs, runtime_logs, pareto_snapshots, archive_runs | Out-Null; '[]' | Out-File -Encoding utf8NoBOM backlog_queue.json; '{}' | Out-File -Encoding utf8NoBOM backlog_queue.json.claims.json; '{}' | Out-File -Encoding utf8NoBOM backlog_queue.json.workers.json; '[]' | Out-File -Encoding utf8NoBOM result_queue.json; Write-Host 'Preflight complete: queues reset and dirs ensured.' }"

# 3) Start API (terminal A)
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload

# 4) Start orchestrator (terminal B)
python adaptive_hunt_orchestrator.py --config burn_in_config.json

# 5) Start worker (terminal C)
$env:CUDA_VISIBLE_DEVICES='0'; python worker_daemon.py
```

Open the UI at http://127.0.0.1:8000/.

## One-Click Launchers (Windows)

Three helper scripts are available at repo root:

- `launch_all.bat`: quick one-click launcher from CMD (API/UI + orchestrator only).
- `stop_all.bat`: graceful-stop signal plus force-stop fallback for ASTE processes.
- `launch_all.ps1`: safer PowerShell launcher with optional preflight and health check (API/UI + orchestrator only).

Batch usage:

```bat
launch_all.bat [config_path] [open_browser]

:: examples
launch_all.bat
launch_all.bat burn_in_config.json 1
stop_all.bat
```

PowerShell usage:

```powershell
.\launch_all.ps1
.\launch_all.ps1 -RunPreflight -Config burn_in_config.json
.\launch_all.ps1 -NoBrowser
```

All launchers target `http://127.0.0.1:8000/`.
Workers are intentionally started manually.

Manual worker example:

```powershell
$env:CUDA_VISIBLE_DEVICES='0'; python worker_daemon.py
```

Important: start workers after you stage/initiate a hunt. If no active run exists yet, workers exit with an active run pointer error.

## Unix/Linux Notes

- Activate env: `source .venv/bin/activate`
- Set GPU variable (example): `CUDA_VISIBLE_DEVICES=0 python worker_daemon.py`
- Other commands are the same script/module names.

## Prerequisites

- Python 3.10+
- Optional GPU acceleration: NVIDIA GPU + working CUDA stack
- Python dependencies installed from `requirements.txt`
- Node/npm only required when rebuilding frontend assets

## Environment Setup

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you need a frontend rebuild:

```powershell
Set-Location UI/components
npm install
npm run build
Set-Location ../..
Copy-Item -Path UI/components/build/* -Destination build -Recurse -Force
```

## Full Operator Flow

### 1. Preflight Initialization

Purpose:
Initialize required directories and reset queue files before launching runtime processes.

Expected result:
`simulation_data/`, `runtime_logs/`, `input_configs/`, `archive_runs/`, and queue JSON files exist.

### 2. Launch API

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

The API handles:

- Mission control staging/start endpoints
- Telemetry streaming
- Static UI serving from `UI/components/build` (preferred) or fallback `build`

### 3. Launch Orchestrator

```powershell
python adaptive_hunt_orchestrator.py --config burn_in_config.json
```

The orchestrator reads queued work and emits dispatch activity into runtime logs.

### 4. Launch Worker(s)

Single worker on GPU 0:

```powershell
$env:CUDA_VISIBLE_DEVICES='0'; python worker_daemon.py
```

Multiple local workers example:

```powershell
# terminal C
$env:CUDA_VISIBLE_DEVICES='0'; python worker_daemon.py

# terminal D
$env:CUDA_VISIBLE_DEVICES='1'; python worker_daemon.py
```

### 5. Stage And Initiate Hunt In UI

1. Open Mission Control in UI.
2. Either provide an existing config path or fill generation/population/simulation fields.
3. Click `GENERATE/STAGE INITIATING CONFIG`.
4. Confirm staged path/hash appears.
5. Click `INITIATE HUNT`.

### 6. Monitor Runtime

- UI status panel and telemetry feed
- `runtime_logs/api_preview.log`
- `runtime_logs/orchestrator.log`
- `runtime_logs/worker_gpu0.log` (and additional worker logs if configured)
- `simulation_data/` and `provenance_reports/` outputs

### 7. Stop/Abort

- Use `ABORT` in Mission Control to issue stop signal.
- Stop terminal processes when done.

## Scaling Workers To Additional VMs

Use this pattern when scaling beyond one machine:

1. Keep one orchestrator owner for a given queue set (do not run two orchestrators on the same queue set unless intentionally sharded).
2. Ensure each worker VM has the same code version and Python dependencies.
3. Ensure workers can reach the same shared queue/session state used by orchestrator (file paths, mounts, or your selected distributed strategy).
4. Launch worker per available GPU on each VM.
5. Set `CUDA_VISIBLE_DEVICES` per process to avoid GPU contention.
6. Verify heartbeats/leases are healthy (check worker heartbeat TTL and job lease timeout values in runtime config).

Worker bring-up checklist per VM:

1. Activate environment.
2. Confirm queue/session path visibility.
3. Start worker process.
4. Confirm worker heartbeat updates in queue worker metadata.
5. Confirm jobs are claimed and completed.

## Backlog Queue Schema

`backlog_queue.json` must be a JSON array. Each item must be a config-like object that can be staged as a run seed.

Minimal example:

```json
[
  {
    "hunt_name": "seed_001",
    "mode": "full",
    "generations": 20,
    "batch_size": 8,
    "population_size": 256,
    "seeds_per_candidate": 4,
    "n_grid": 128,
    "t_steps": 600,
    "dt": 0.01
  }
]
```

Path safety rules for Mission Control `backlog_source`:

1. Relative paths are resolved from repository root.
2. Absolute paths are accepted only when they resolve inside repository root.
3. Paths that escape the repo are rejected with a stage-time validation error.

## Troubleshooting

### UI does not show latest changes

1. Rebuild UI in `UI/components` with `npm run build`.
2. Copy build artifacts to root `build` directory.
3. Reload UI at `http://127.0.0.1:8000/`.

### Orchestrator runs but nothing is processed

1. Confirm preflight was run and queue files exist.
2. Confirm at least one worker process is running.
3. Check `runtime_logs/orchestrator.log` and worker log for lease/claim issues.

### Stage/start fails with backlog source error

1. Use a path under repository root (example: `input_configs/backlog_queue.json`).
2. Ensure the file exists and is valid JSON array (`[]` when empty).
3. Avoid external absolute paths; they are intentionally blocked by path hardening.

### Debug console shows "Awaiting ... creation"

1. This is a one-time system warning that a configured log source file does not exist yet.
2. Start the corresponding process (worker/orchestrator/API) to create the log file.
3. If process is running, check path and file permissions in `runtime_logs/`.

### Fleet worker stale warning appears

1. `CRITICAL` stale alerts are emitted when worker heartbeat age exceeds 60 seconds.
2. Verify worker process health and queue visibility.
3. Restart workers that no longer update heartbeat metadata.

### Worker starts but never claims jobs

1. Confirm worker sees the same queue/session files as orchestrator.
2. Confirm heartbeat TTL and lease timeout are not too aggressive for your hardware.
3. Confirm no stale claims are blocking queue progress.

### API is up but UI page is blank or missing

1. Confirm build output contains `index.html` and static assets.
2. Confirm API process has access to `UI/components/build` or root `build`.
3. Inspect browser console/network for static 404s.

## Validation And Smoke Checks

- Run deploy + smoke:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "& .\tools\deploy_local.ps1 -RunSmoke"
```

- Run selected tests:

```powershell
python -m pytest tests/test_e2e_integration.py tests/test_phase_d_static_compile.py tests/test_no_subprocess_bypass.py tests/test_schema_concurrent.py tests/test_seed_determinism.py tests/test_oom_fallback.py -q
```

## Useful Commands

```powershell
python adaptive_hunt_orchestrator.py --help
python worker_daemon.py --help
python -m uvicorn --help
```

## Project References

- Notebook walkthrough: `notebooks/ASTE_Quickstart.ipynb`
- WebSocket/API notes: `WEBSOCKET_API.md`
