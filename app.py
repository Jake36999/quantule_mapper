import os
import asyncio
import logging
import json
import hashlib
import re
import sqlite3
import subprocess
import sys
import time
import traceback
from typing import Any, Dict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

# --- WINDOWS ASYNCIO SUBPROCESS FIX ---
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
# --------------------------------------
import psutil  # For the ghost worker purge
from fastapi import FastAPI, Response, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from orchestrator.schema_utils import initialize_ledger_schema
from config_utils import (
    ActiveRunPointerError,
    create_session_workspace,
    write_active_run_pointer_atomic,
    resolve_active_session_paths,
    clear_active_run_pointer,
    read_active_run_pointer,
    resolve_generation_dir,
    init_run_manifest,
    ensure_generation_dir,
    update_run_manifest_generation,
    run_manifest_path,
    normalize_run_mode,
    resolve_backlog_source_path,
    RUNS_ROOT_DIR,
    SESSION_DB_BASENAME,
)

try:
    from orchestrator.job_manifest import JobManifest
    from orchestrator.scheduling.queue_manager import QueueManager
    IPC_AVAILABLE = True
except ImportError:
    IPC_AVAILABLE = False

# ---- Process Management State ----
state = {
    "worker_processes": [],
    "orchestrator_process": None,
    "stream_tasks": [],
}

# ---- Configuration Constants ----
DB_FILE = os.getenv("ASTE_DB_FILE", "simulation_ledger.db")
BACKLOG_QUEUE_FILE = "backlog_queue.json"
BACKLOG_RESULT_FILE = "result_queue.json"
DATA_DIR = os.getenv("ASTE_DATA_DIR", "simulation_data")
SIM_DATA_DIR = Path(DATA_DIR)
ALLOWED_POINTCLOUD_DATASETS = frozenset({
    "psi_final",
    "rho_final",
    "A_final",
    "A_dot_final",
    "omega_sq_final",
    "N_a_stage",
    "N_b_stage",
    "N_c_stage",
})
ACTIVE_RUN_POINTER_ERROR = "No active hunt session. Start a hunt to initialize session workspace."
ROOT_BUILD_DIR = Path(__file__).parent / "build"
UI_COMPONENTS_BUILD_DIR = Path(__file__).parent / "UI" / "components" / "build"
RUNTIME_LOGS_DIR = Path(__file__).parent / "runtime_logs"
DEBUG_TEST_TIMEOUT_SECONDS = int(os.getenv("ASTE_DEBUG_TEST_TIMEOUT_SECONDS", "900"))
SYSTEM_TELEMETRY_TTL_SECONDS = float(os.getenv("ASTE_WORKER_HEARTBEAT_TTL_SECONDS", "90.0"))
PLUGIN_EXECUTION_TIMEOUT_SECONDS = int(os.getenv("ASTE_PLUGIN_EXECUTION_TIMEOUT_SECONDS", "300"))
PLUGIN_EXECUTION_MAX_ARGS = int(os.getenv("ASTE_PLUGIN_EXECUTION_MAX_ARGS", "8"))
PLUGIN_EXECUTION_MAX_ARG_LENGTH = int(os.getenv("ASTE_PLUGIN_EXECUTION_MAX_ARG_LENGTH", "128"))
PLUGIN_EXECUTION_MAX_PATH_LENGTH = int(os.getenv("ASTE_PLUGIN_EXECUTION_MAX_PATH_LENGTH", "4096"))
PLUGIN_EXECUTION_ALLOWED_EXTENSIONS = {
    ".h5",
    ".hdf5",
    ".csv",
    ".json",
    ".npy",
    ".npz",
}
PLUGIN_TIER_VALUES = {"GOLDEN", "SILVER", "BRONZE"}
PLUGIN_CONTRACTS = {
    "gif_pipeline_manager": "gif_pipeline",
    "fss_scaling_analyzer": "fss_scaling",
}

RUN_DIR = Path(__file__).parent / "run"
RUN_DIR.mkdir(exist_ok=True)
WORKERS_PID_FILE = RUN_DIR / "workers.pid"
ORCHESTRATOR_PID_FILE = RUN_DIR / "orchestrator.pid"

def _resolve_frontend_build_dir() -> Path:
    """Prefer the live UI/components build; fallback to legacy root build."""
    if (UI_COMPONENTS_BUILD_DIR / "index.html").exists() and (UI_COMPONENTS_BUILD_DIR / "static").exists():
        return UI_COMPONENTS_BUILD_DIR
    return ROOT_BUILD_DIR

ACTIVE_BUILD_DIR = _resolve_frontend_build_dir()

DEBUG_LOG_SOURCES = [
    {"id": "api", "name": "API Preview", "path": RUNTIME_LOGS_DIR / "api_preview.log"},
    {"id": "orchestrator", "name": "Orchestrator", "path": RUNTIME_LOGS_DIR / "orchestrator.log"},
    {"id": "worker", "name": "Worker GPU0", "path": RUNTIME_LOGS_DIR / "worker_gpu0.log"},
    {"id": "preflight", "name": "Preflight", "path": RUNTIME_LOGS_DIR / "preflight.log"},
]

DEBUG_TEST_SUITES = {
    "e2e": ["python", "-m", "pytest", "tests/test_e2e_integration.py", "-q"],
    "regression": [
        "python", "-m", "pytest",
        "tests/test_e2e_integration.py",
        "tests/test_phase_d_static_compile.py",
        "tests/test_no_subprocess_bypass.py",
        "tests/test_schema_concurrent.py",
        "tests/test_seed_determinism.py",
        "tests/test_oom_fallback.py",
        "-q",
    ],
}
TESTS_DIR = Path(__file__).parent / "tests"

# ==============================================================================
# APP INITIALIZATION
# ==============================================================================
app = FastAPI()

# ---- CORS Middleware: Allow localhost dev UI ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:3000",
        "http://localhost:8000",
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_connections: set[WebSocket] = set()
observer_instance: object | None = None
telemetry_queue: asyncio.Queue = asyncio.Queue()
telemetry_ticker_task: asyncio.Task | None = None
log_tailer_task_instance: asyncio.Task | None = None
missing_log_warned_sources: set[str] = set()
stale_worker_alert_cache: dict[str, float] = {}


# ==============================================================================
# GHOST PURGE, PID TRACKING & TERMINAL PIPE HELPERS
# ==============================================================================
def _save_pids():
    """Write active subprocess PIDs to disk for zombie recovery."""
    try:
        worker_pids = [str(p.pid) for p in state["worker_processes"] if p.returncode is None]
        WORKERS_PID_FILE.write_text("\n".join(worker_pids))
        
        if state["orchestrator_process"] and state["orchestrator_process"].returncode is None:
            ORCHESTRATOR_PID_FILE.write_text(str(state["orchestrator_process"].pid))
        else:
            if ORCHESTRATOR_PID_FILE.exists():
                ORCHESTRATOR_PID_FILE.unlink()
    except Exception as exc:
        logging.warning(f"Failed to save PID files: {exc}")

def _kill_saved_pids():
    """Kill any PIDs leftover from previous runs to prevent resource leakage."""
    for pid_file in [WORKERS_PID_FILE, ORCHESTRATOR_PID_FILE]:
        if pid_file.exists():
            try:
                pids = pid_file.read_text().splitlines()
                for pid_str in pids:
                    if pid_str.strip():
                        try:
                            proc = psutil.Process(int(pid_str.strip()))
                            proc.terminate()
                            proc.wait(timeout=3)
                        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                            try:
                                proc.kill()
                            except Exception:
                                pass
            except Exception:
                pass
            try:
                pid_file.unlink(missing_ok=True)
            except Exception:
                pass

async def _stream_subprocess_output(stream, feed_id: str, level: str = "INFO"):
    """Asynchronously pipe subprocess stdout/stderr directly to the UI Terminal Matrix."""
    if not stream:
        return
    while True:
        line = await stream.readline()
        if not line:
            break
        line_str = line.decode('utf-8', errors='replace').rstrip()
        if line_str:
            _emit_terminal_debug(feed_id, line_str, level)

def _purge_ghost_workers():
    """Finds and violently kills any lingering Python processes running worker scripts."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline') or []
            if any('worker_daemon.py' in cmd for cmd in cmdline) or any('worker_cupy.py' in cmd for cmd in cmdline):
                proc.terminate()
                proc.wait(timeout=3)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            try:
                proc.kill()
            except Exception:
                pass

# ==============================================================================
# CLUSTER KILL ENDPOINTS
# ==============================================================================
@app.post("/api/control/kill_cluster")
async def kill_cluster():
    """Violently terminate all active workers and the orchestrator, and clean the DB."""
    try:
        # 1. Kill OS Processes
        _purge_ghost_workers()
        _kill_saved_pids()
        
        # Cancel dangling stream tasks
        for task in state.get("stream_tasks", []):
            task.cancel()
        state["stream_tasks"] = []

        for proc in state["worker_processes"]:
            try:
                proc.kill()
            except Exception:
                pass
        state["worker_processes"] = []
        
        if state["orchestrator_process"] is not None:
            try:
                state["orchestrator_process"].kill()
            except Exception:
                pass
            state["orchestrator_process"] = None
            
        _save_pids()
            
        # 2. Clean the Queue Database
        qm = QueueManager(queue_file=BACKLOG_QUEUE_FILE, result_file=BACKLOG_RESULT_FILE)
        qm.clear_all_workers()
        _emit_terminal_debug("backend", "Cluster terminated. All workers killed and DB cleared.", "WARN")
        return {"status": "success", "message": "Cluster completely terminated."}
    except Exception as exc:
        logging.error(f"Failed to kill cluster:\n{traceback.format_exc()}")
        return {"status": "error", "message": "Failed to kill cluster. See server logs for details."}

@app.post("/api/control/kill_worker/{worker_id}")
async def kill_worker(worker_id: str):
    """Terminate a specific worker and re-queue its jobs."""
    try:
        qm = QueueManager(queue_file=BACKLOG_QUEUE_FILE, result_file=BACKLOG_RESULT_FILE)
        qm.clear_worker(worker_id)
        _emit_terminal_debug("backend", f"Targeted kill issued for {worker_id}. Jobs re-queued.", "WARN")
        return {"status": "success", "message": f"Worker {worker_id} terminated."}
    except Exception as exc:
        logging.error(f"Failed to kill worker:\n{traceback.format_exc()}")
        return {"status": "error", "message": "Failed to kill worker. See server logs for details."}


# ---- Pydantic Models ----
class ControlStartRequest(BaseModel):
    staged_path: str
    hunt_name: str | None = None
    origin: str | None = None
    staged_at: str | None = None
    staged_config_hash: str | None = None
    mode: str | None = None
    backlog_source: str | None = None
    worker_count: int | None = None
    workerCount: int | None = None
    param_D: float = 1.0
    param_eta: float = 0.1
    param_a: float = 0.0
    param_rho_vac: float = 0.0


class ControlStageRequest(BaseModel):
    config_path: str | None = None
    hunt_name: str | None = None
    generations: int | None = None
    batch_size: int | None = None
    origin: str | None = None
    population_size: int | None = None
    seeds_per_candidate: int | None = None
    n_grid: int | None = None
    t_steps: int | None = None
    dt: float | None = None
    l_domain: float | None = None
    collapse_threshold: float | None = None
    mutation_rate: float | None = None
    crossover_rate: float | None = None
    survival_fraction: float | None = None
    predator_sweep_frequency: int | None = None
    min_queue_depth: int | None = None
    poll_interval: float | None = None
    bleed_workers: int | None = None
    artifact_gc_min_age_seconds: int | None = None
    job_lease_timeout_seconds: int | None = None
    result_lease_timeout_seconds: int | None = None
    worker_heartbeat_ttl_seconds: int | None = None
    max_queue_depth: int | None = None
    mode: str | None = None          # "evolution" (default) | "backlog"
    backlog_source: str | None = None  # path to backlog queue file; defaults to backlog_queue.json
    param_D: float = 1.0
    param_eta: float = 0.1
    param_a: float = 0.0
    param_rho_vac: float = 0.0


class DebugTestRunRequest(BaseModel):
    suite: str = "e2e"


class PluginExecuteRequest(BaseModel):
    plugin_id: str
    artifact_path: str | None = None
    sse: float | None = None
    tier: str | None = None
    args: list[str] = Field(default_factory=list)


class LedgerRunsResponse(BaseModel):
    status: str
    runs: list[dict[str, Any]]


class LedgerRowsResponse(BaseModel):
    status: str
    run_id: str | None = None
    session_dir: str | None = None
    db_file: str | None = None
    rows: list[dict[str, Any]] = []
    pagination: dict[str, int] | None = None
    message: str | None = None


class FleetWorkerStatus(BaseModel):
    worker_id: str
    last_heartbeat_epoch: float
    age_seconds: float
    state: str
    in_flight_claims: int


class FleetTelemetryResponse(BaseModel):
    status: str
    timestamp: str
    queue_depth: int
    dlq_count: int
    active_workers: list[str]
    stale_workers: list[str]
    total_claims_processed: int
    worker_heartbeat_ttl_seconds: float
    workers: list[FleetWorkerStatus]
    message: str | None = None


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

async def _broadcast_payload(payload: dict) -> None:
    message = json.dumps(payload)
    
    async def send_msg(connection: WebSocket):
        try:
            await asyncio.wait_for(connection.send_text(message), timeout=2.0)
        except Exception:
            active_connections.discard(connection)
            
    if active_connections:
        await asyncio.gather(*(send_msg(conn) for conn in list(active_connections)), return_exceptions=True)


def _enqueue_telemetry_event(payload: dict) -> None:
    try:
        telemetry_queue.put_nowait(payload)
    except Exception:
        return


def _emit_terminal_debug(feed_id: str, line: str, level: str = "INFO") -> None:
    clean_feed = str(feed_id or "backend").strip() or "backend"
    clean_level = str(level or "INFO").strip().upper() or "INFO"
    clean_line = str(line or "").strip()
    if not clean_line:
        return
    # FAILSAFE: Force the log to the physical PowerShell console
    print(f"[{clean_feed.upper()}] {clean_level}: {clean_line}")
    _enqueue_telemetry_event(
        {
            "type": "terminal_log",
            "feed_id": clean_feed,
            "level": clean_level,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "line": f"[{clean_feed.upper()}] {clean_level}: {clean_line}",
        }
    )


async def telemetry_ticker() -> None:
    """Drain telemetry queue every 0.5s and emit one JSON array batch."""
    while True:
        await asyncio.sleep(0.5)
        batch: list[dict] = []
        while not telemetry_queue.empty():
            try:
                item = telemetry_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if isinstance(item, dict):
                batch.append(item)

        if not batch or not active_connections:
            continue

        message = json.dumps(batch)
        for connection in list(active_connections):
            try:
                await asyncio.wait_for(connection.send_text(message), timeout=2.0)
            except asyncio.TimeoutError:
                active_connections.discard(connection)
            except Exception:
                active_connections.discard(connection)


def _initialize_log_tail_positions() -> dict[str, int]:
    file_positions: dict[str, int] = {}
    for source in DEBUG_LOG_SOURCES:
        source_id = str(source.get("id", "")).strip()
        if not source_id:
            continue
        path = Path(str(source.get("path", "")))
        if not path.exists():
            _emit_terminal_debug(
                "backend",
                f"Debug log source missing at startup: {source_id} -> {path}",
                "WARN",
            )
        try:
            file_positions[source_id] = int(path.stat().st_size) if path.exists() else 0
        except Exception:
            file_positions[source_id] = 0
    return file_positions


async def _tail_logs_once(file_positions: dict[str, int]) -> None:
    global missing_log_warned_sources
    for source in DEBUG_LOG_SOURCES:
        source_id = str(source.get("id", "")).strip()
        source_name = str(source.get("name", source_id)).strip() or source_id
        if not source_id:
            continue

        path = Path(str(source.get("path", "")))
        if not path.exists():
            if source_id not in missing_log_warned_sources:
                _emit_terminal_debug(
                    "system",
                    f"Awaiting {source_name} creation...",
                    "WARN",
                )
                missing_log_warned_sources.add(source_id)
            continue
        missing_log_warned_sources.discard(source_id)

        try:
            current_size = int(path.stat().st_size)
            pos = int(file_positions.get(source_id, 0))
            if current_size < pos:
                pos = 0

            if current_size <= pos:
                file_positions[source_id] = pos
                continue

            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                handle.seek(pos)
                new_data = handle.read()
                file_positions[source_id] = int(handle.tell())

            if not new_data:
                continue

            for line in new_data.splitlines():
                if not line.strip():
                    continue
                _enqueue_telemetry_event(
                    {
                        "type": "terminal_log",
                        "feed_id": source_id,
                        "line": f"[{source_name}] {line}",
                    }
                )
        except Exception as exc:
            logging.warning(f"Error tailing {source_name}: {exc}")
            _emit_terminal_debug("backend", f"Error tailing {source_name}: {exc}", "WARN")


async def log_tailer_task() -> None:
    """Background task to watch DEBUG_LOG_SOURCES and stream new lines to the UI."""
    file_positions = _initialize_log_tail_positions()
    while True:
        await asyncio.sleep(1.0)
        await _tail_logs_once(file_positions)


def _tail_file_lines(path: Path, max_lines: int) -> list[str]:
    if not path.exists() or not path.is_file():
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            content = handle.read().splitlines()
        return content if max_lines <= 0 else content[-max_lines:]
    except Exception:
        return []


def _load_pde_history_from_h5(artifact_path: Path) -> list[dict]:
    """Load PDE telemetry history points from HDF5 artifact for websocket broadcast."""
    if not artifact_path.exists() or not artifact_path.is_file():
        return []
    try:
        import h5py  # type: ignore
    except Exception:
        return []

    try:
        with h5py.File(artifact_path, "r") as handle:
            if "telemetry" not in handle:
                return []
            telemetry = handle["telemetry"]
            required = ("step", "energy", "C_invariant", "max_amplitude")
            if any(key not in telemetry for key in required):
                return []

            steps = list(telemetry["step"][:])
            energy = list(telemetry["energy"][:])
            c_invariant = list(telemetry["C_invariant"][:])
            max_amplitude = list(telemetry["max_amplitude"][:])

        point_count = min(len(steps), len(energy), len(c_invariant), len(max_amplitude))
        history: list[dict] = []
        for idx in range(point_count):
            history.append(
                {
                    "time": float(steps[idx]),
                    "energy": float(energy[idx]),
                    "c_invariant": float(c_invariant[idx]),
                    "max_amplitude": float(max_amplitude[idx]),
                }
            )
        return history
    except Exception:
        return []

def _plugin_allowed_roots() -> list[Path]:
    repo_root = Path(__file__).parent.resolve()
    roots = [repo_root]

    data_dir = Path(DATA_DIR).resolve()
    if data_dir.exists() and data_dir.is_dir():
        roots.append(data_dir)

    try:
        active_paths = resolve_active_session_paths(require_exists=True)
        session_dir = Path(str(active_paths["session_dir"])).resolve()
        roots.append(session_dir)
    except (ActiveRunPointerError, ValueError, KeyError):
        pass

    archive_dir = (repo_root / "archive_runs").resolve()
    if archive_dir.exists() and archive_dir.is_dir():
        roots.append(archive_dir)

    unique_roots: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        unique_roots.append(root)
    return unique_roots


def _resolve_safe_artifact_path(raw_path: str | None) -> Path | None:
    if raw_path is None:
        return None
    candidate_raw = str(raw_path).strip()
    if not candidate_raw:
        return None
    if len(candidate_raw) > PLUGIN_EXECUTION_MAX_PATH_LENGTH:
        raise ValueError("artifact_path exceeds maximum length")

    candidate = Path(candidate_raw).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if not candidate.exists() or not candidate.is_file():
        raise ValueError("artifact_path does not exist")

    if candidate.suffix.lower() not in PLUGIN_EXECUTION_ALLOWED_EXTENSIONS:
        raise ValueError("artifact_path extension is not allowed")

    for root in _plugin_allowed_roots():
        try:
            candidate.relative_to(root)
            return candidate
        except ValueError:
            continue

    raise ValueError("artifact_path must be under an approved workspace root")

CONTROL_REQUIRED_FIELDS = (
    "generations",
    "batch_size",
    "population_size",
    "seeds_per_candidate",
    "n_grid",
    "t_steps",
    "dt",
)


def _canonicalize_control_payload(payload: dict) -> dict:
    """Normalize control payload keys so generated artifacts are consistent."""
    normalized = dict(payload)
    if "N_grid" in normalized and "n_grid" not in normalized:
        normalized["n_grid"] = normalized.pop("N_grid")
    if "T_steps" in normalized and "t_steps" not in normalized:
        normalized["t_steps"] = normalized.pop("T_steps")
    return normalized


def _validate_control_payload(payload: dict) -> tuple[bool, str | None]:
    max_backlog_bytes = 50 * 1024 * 1024

    for key in CONTROL_REQUIRED_FIELDS:
        value = payload.get(key)
        if value is None:
            return False, f"missing required config field '{key}'"
        if key == "dt":
            try:
                if float(value) <= 0:
                    return False, "dt must be > 0"
            except Exception:
                return False, "dt must be numeric"
        else:
            try:
                if int(value) <= 0:
                    return False, f"{key} must be > 0"
            except Exception:
                return False, f"{key} must be an integer"
    try:
        mode = normalize_run_mode(payload.get("mode"))
    except ValueError as exc:
        return False, str(exc)

    if mode == "backlog":
        # In backlog mode: population_size == chunk size, generations == purge cadence.
        try:
            if int(payload.get("population_size", 0)) <= 0:
                return False, "population_size (chunk size) must be > 0 in backlog mode"
            if int(payload.get("generations", 0)) <= 0:
                return False, "generations (purge cadence) must be > 0 in backlog mode"
        except Exception:
            return False, "population_size and generations must be positive integers in backlog mode"
        try:
            backlog_source_path = resolve_backlog_source_path(
                payload.get("backlog_source"),
                BACKLOG_QUEUE_FILE,
                require_exists=True,
            )
        except ValueError as exc:
            return False, str(exc)

        if backlog_source_path.is_file():
            backlog_size = backlog_source_path.stat().st_size
            if backlog_size > max_backlog_bytes:
                return (
                    False,
                    f"backlog_source file exceeds 50MB: {backlog_source_path} "
                    f"({backlog_size} bytes)",
                )

        if backlog_source_path.is_dir():
            for candidate in backlog_source_path.glob("*.json"):
                if not candidate.is_file():
                    continue
                backlog_size = candidate.stat().st_size
                if backlog_size > max_backlog_bytes:
                    return (
                        False,
                        f"backlog entry exceeds 50MB: {candidate} ({backlog_size} bytes)",
                    )

    return True, None


def _build_control_payload_from_request(req: ControlStageRequest) -> dict:
    # Dump all fields directly from the request
    payload = req.dict(exclude_none=True)
    # Inject the required orchestrator system path
    payload["system"] = {"archive_dir": str(RUNS_ROOT_DIR)}
    return payload


def _resolve_control_payload(req: ControlStageRequest) -> tuple[dict | None, str | None]:
    if req.config_path:
        # Use safe resolver to prevent path traversal
        try:
            requested = _resolve_safe_artifact_path(req.config_path)
        except ValueError as exc:
            return None, f"Invalid config_path: {exc}"
            
        if not requested or not requested.exists() or not requested.is_file():
            return None, f"config_path not found or disallowed: {req.config_path}"
            
        try:
            with open(requested, "r", encoding="utf-8") as handle:
                file_payload = json.load(handle)
        except Exception as exc:
            return None, f"failed to read config_path '{req.config_path}': {exc}"

        canonical = _canonicalize_control_payload(file_payload)
        if "origin" not in canonical or not canonical.get("origin"):
            canonical["origin"] = req.origin or "UI_CONTROL"
        return canonical, None

    payload = _build_control_payload_from_request(req)
    payload = _canonicalize_control_payload(payload)
    payload["origin"] = payload.get("origin") or "UI_CONTROL"
    return payload, None


def _active_paths_or_error() -> tuple[dict | None, JSONResponse | None]:
    try:
        return resolve_active_session_paths(require_exists=True), None
    except ActiveRunPointerError as exc:
        return None, JSONResponse(status_code=409, content={"status": "error", "message": str(exc) or ACTIVE_RUN_POINTER_ERROR})
    except ValueError as exc:
        return None, JSONResponse(status_code=400, content={"status": "error", "message": str(exc)})


def _load_plugin_manifest(manifest_path: Path) -> dict:
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    plugins = manifest.get("plugins", [])
    if isinstance(plugins, list):
        for plugin in plugins:
            if not isinstance(plugin, dict):
                continue
            script = plugin.get("script")
            if not script:
                continue
            script_path = (manifest_path.parent / str(script)).resolve()
            plugin["script_exists"] = script_path.exists()
            plugin["resolved_path"] = str(script_path).replace("\\", "/")
    return manifest


def _collect_plugin_registry() -> dict[str, dict[str, Any]]:
    repo_root = Path(__file__).parent.resolve()
    plugin_root = (repo_root / "plugins").resolve()
    manifests = (
        ("tools", plugin_root / "tools" / "manifest.json"),
        ("visualizers", plugin_root / "visualizers" / "manifest.json"),
    )
    registry: dict[str, dict[str, Any]] = {}

    for kind, manifest_path in manifests:
        if not manifest_path.exists() or not manifest_path.is_file():
            continue
        manifest = _load_plugin_manifest(manifest_path)
        for plugin in manifest.get("plugins", []):
            if not isinstance(plugin, dict):
                continue
            plugin_id = str(plugin.get("id") or "").strip()
            script_rel = str(plugin.get("script") or "").strip()
            if not plugin_id or not script_rel:
                continue

            script_path = (manifest_path.parent / script_rel).resolve()
            if not script_path.exists() or not script_path.is_file():
                continue
            try:
                script_path.relative_to(plugin_root)
            except ValueError:
                continue

            registry[plugin_id] = {
                "id": plugin_id,
                "kind": kind,
                "script_path": script_path,
                "script": script_rel,
            }

    return registry


def _validate_plugin_args(raw_args: list[str]) -> list[str]:
    if len(raw_args) > PLUGIN_EXECUTION_MAX_ARGS:
        raise ValueError(f"args length exceeds maximum of {PLUGIN_EXECUTION_MAX_ARGS}")

    validated: list[str] = []
    for value in raw_args:
        arg = str(value)
        if len(arg) > PLUGIN_EXECUTION_MAX_ARG_LENGTH:
            raise ValueError("plugin arg exceeds maximum length")
        if "\x00" in arg:
            raise ValueError("plugin args cannot contain null bytes")
        validated.append(arg)
    return validated


def _build_plugin_command(plugin_id: str, script_path: Path, req: PluginExecuteRequest) -> list[str]:
    contract = PLUGIN_CONTRACTS.get(plugin_id)
    if not contract:
        raise ValueError(f"plugin execution contract is not defined for '{plugin_id}'")

    if contract == "gif_pipeline":
        artifact = _resolve_safe_artifact_path(req.artifact_path)
        if artifact is None:
            raise ValueError("artifact_path is required for gif_pipeline_manager")

        sse_value = float(req.sse) if req.sse is not None else 999.0
        tier_value = str(req.tier or "SILVER").upper()
        if tier_value not in PLUGIN_TIER_VALUES:
            raise ValueError(f"tier must be one of {sorted(PLUGIN_TIER_VALUES)}")

        return [
            sys.executable,
            str(script_path),
            "--input",
            str(artifact),
            "--sse",
            str(sse_value),
            "--tier",
            tier_value,
        ]

    if contract == "fss_scaling":
        if req.artifact_path:
            _resolve_safe_artifact_path(req.artifact_path)
        return [sys.executable, str(script_path)]

    raise ValueError(f"unsupported plugin execution contract: {contract}")


def _record_plugin_execution_event(event: dict[str, Any]) -> None:
    RUNTIME_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    event_path = RUNTIME_LOGS_DIR / "plugin_executions.jsonl"
    with open(event_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=True) + "\n")


def _artifact_hit(base: Path, root_name: str, path: Path) -> dict:
    stat = path.stat()
    return {
        "name": path.name,
        "relative_path": str(path.relative_to(base)).replace("\\", "/"),
        "source": root_name,
        "size": stat.st_size,
        "modified": int(stat.st_mtime),
        "type": path.suffix,
    }


def _is_valid_run_id(run_id: str) -> bool:
    return bool(re.fullmatch(r"[A-Fa-f0-9]{8}", run_id.strip()))


def _is_valid_config_hash(config_hash: str) -> bool:
    return bool(re.fullmatch(r"[A-Fa-f0-9]{64}", config_hash.strip()))


def _coerce_pagination(limit: int, offset: int, *, max_limit: int = 500) -> tuple[int, int]:
    safe_limit = max(1, min(int(limit), max_limit))
    safe_offset = max(0, int(offset))
    return safe_limit, safe_offset


def _session_metadata_from_dir(session_dir: Path) -> dict[str, Any] | None:
    if not session_dir.exists() or not session_dir.is_dir():
        return None

    session_name = session_dir.name
    run_id = ""
    hunt_name = session_name
    if "_" in session_name:
        hunt_name, run_id = session_name.rsplit("_", 1)

    db_file = session_dir / SESSION_DB_BASENAME
    provenance_dir = session_dir / "provenance_reports"
    created_at = datetime.fromtimestamp(session_dir.stat().st_ctime, tz=timezone.utc).isoformat()

    return {
        "hunt_name": hunt_name,
        "run_id": run_id,
        "session_name": session_name,
        "session_dir": str(session_dir.resolve()),
        "db_file": str(db_file.resolve()),
        "provenance_dir": str(provenance_dir.resolve()),
        "created_at": created_at,
        "is_active": False,
    }


def _list_known_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()

    try:
        active_payload = read_active_run_pointer()
    except ActiveRunPointerError:
        active_payload = None

    if active_payload:
        active_entry: dict[str, Any] = dict(active_payload)
        active_entry["session_name"] = (
            active_payload.get("session_name")
            or f"{active_payload.get('hunt_name', '')}_{active_payload.get('run_id', '')}"
        )
        active_entry["is_active"] = True
        runs.append(active_entry)
        seen_run_ids.add(str(active_entry.get("run_id") or ""))

    runs_root = Path(__file__).parent / RUNS_ROOT_DIR
    if runs_root.exists() and runs_root.is_dir():
        for candidate in sorted(runs_root.iterdir(), key=lambda path: path.name.lower(), reverse=True):
            entry = _session_metadata_from_dir(candidate)
            if not entry:
                continue
            run_id = str(entry.get("run_id") or "")
            if not _is_valid_run_id(run_id) or run_id in seen_run_ids:
                continue
            runs.append(entry)
            seen_run_ids.add(run_id)

    runs.sort(
        key=lambda entry: (
            0 if entry.get("is_active") else 1,
            str(entry.get("created_at") or ""),
        ),
        reverse=False,
    )
    return runs


def _resolve_ledger_run_or_error(run_id: str) -> tuple[dict[str, Any] | None, JSONResponse | None]:
    normalized_run_id = str(run_id or "").strip()
    if not _is_valid_run_id(normalized_run_id):
        return None, JSONResponse(
            status_code=400,
            content={"status": "error", "message": "run_id must be an 8-character hex identifier"},
        )

    try:
        active_payload = read_active_run_pointer()
    except ActiveRunPointerError:
        active_payload = None

    if active_payload and str(active_payload.get("run_id")) == normalized_run_id:
        try:
            active_paths = resolve_active_session_paths(require_exists=True)
        except ActiveRunPointerError as exc:
            return None, JSONResponse(status_code=409, content={"status": "error", "message": str(exc)})
        except ValueError as exc:
            return None, JSONResponse(status_code=400, content={"status": "error", "message": str(exc)})

        return {
            **active_paths,
            "session_name": f"{active_paths['hunt_name']}_{active_paths['run_id']}",
            "is_active": True,
        }, None

    runs_root = (Path(__file__).parent / RUNS_ROOT_DIR).resolve()
    if not runs_root.exists() or not runs_root.is_dir():
        return None, JSONResponse(status_code=404, content={"status": "error", "message": f"run_id not found: {normalized_run_id}"})

    for candidate in runs_root.iterdir():
        entry = _session_metadata_from_dir(candidate)
        if not entry:
            continue
        if str(entry.get("run_id")) != normalized_run_id:
            continue
        db_file = Path(str(entry["db_file"]))
        if not db_file.exists():
            return None, JSONResponse(status_code=404, content={"status": "error", "message": f"ledger database missing for run_id: {normalized_run_id}"})
        return entry, None

    return None, JSONResponse(status_code=404, content={"status": "error", "message": f"run_id not found: {normalized_run_id}"})


def _fetch_ledger_rows(db_file: Path, limit: int, offset: int) -> tuple[list[dict[str, Any]], int]:
    """Synchronous SQLite execution logic. Isolated to run in a threadpool."""
    conn = sqlite3.connect(str(db_file), timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=30000;")

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) AS total FROM runs")
        total_rows = int(cursor.fetchone()[0])

        cursor.execute(
            """
            SELECT
                runs.config_hash,
                runs.generation,
                runs.status,
                runs.fitness,
                runs.origin,
                runs.timestamp,
                metrics.log_prime_sse,
                metrics.bragg_peaks_detected,
                metrics.pcs,
                metrics.collapse_event_count,
                parameters.param_D,
                parameters.param_eta,
                parameters.param_rho_vac
            FROM runs
            LEFT JOIN metrics ON metrics.config_hash = runs.config_hash
            LEFT JOIN parameters ON parameters.config_hash = runs.config_hash
            ORDER BY runs.timestamp DESC, runs.generation DESC, runs.config_hash DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )
        rows = [dict(row) for row in cursor.fetchall()]
        return rows, total_rows
    finally:
        conn.close()


def _search_artifact_root(base: Path, query: str, remaining: int, source: str) -> list[dict]:
    if remaining <= 0 or not base.exists() or not base.is_dir():
        return []

    lowered = query.strip().lower()
    allowed = {".h5", ".csv", ".npz", ".json"}
    hits: list[dict] = []
    for candidate in base.rglob("*"):
        if not candidate.is_file() or candidate.suffix.lower() not in allowed:
            continue
        if lowered and lowered not in candidate.name.lower():
            continue
        hits.append(_artifact_hit(base, source, candidate))
        if len(hits) >= remaining:
            break
    return hits

def _extract_tensor_binary_sync(safe_path: Path) -> bytes:
    """Synchronous HDF5 read to run in a threadpool."""
    with h5py.File(safe_path, "r") as f:
        if "rho_final" in f:
            tensor_data = f["rho_final"][::4, ::4, ::4]
        elif "rho_history" in f:
            rho_history = f["rho_history"]
            tensor_data = rho_history[-1, ::4, ::4, ::4] if rho_history.ndim == 4 else rho_history[::4, ::4, ::4]
        elif "psi_final" in f:
            psi_stride = f["psi_final"][::4, ::4, ::4]
            tensor_data = np.abs(psi_stride).astype(np.float32) ** 2
        else:
            raise ValueError("No supported tensor dataset found in artifact")

    return np.asarray(tensor_data, dtype=np.float32).flatten().tobytes()

def _extract_pointcloud_sync(safe_path: Path, dataset_name: str, safe_threshold: float) -> bytes:
    """Synchronous HDF5 read and filtering to run in a threadpool."""
    with h5py.File(safe_path, 'r') as f:
        if dataset_name not in f:
            raise KeyError(f"Dataset {dataset_name} not found.")

        data = f[dataset_name][::2, ::2, ::2]

        if np.iscomplexobj(data):
            data = np.abs(data) ** 2

        max_val = float(np.max(data)) if data.size else 0.0
        if max_val > 0.0:
            data = data / max_val

        z, y, x = np.where(data > safe_threshold)
        intensities = data[z, y, x]

        center = data.shape[0] / 2.0
        point_cloud = np.column_stack((x - center, y - center, z - center, intensities)).astype(np.float32)
        return point_cloud.ravel().tobytes()

# ---- Control Endpoints ----
@app.post("/api/control/stage")
async def control_stage(req: ControlStageRequest):
    """
    Persist mission-control inputs to a unique staged JSON before launch.
    This gives the frontend a traceable config path for reproducibility.
    """
    try:
        staged_payload, resolve_error = _resolve_control_payload(req)
        if resolve_error:
            return {"status": "error", "message": resolve_error}
        if staged_payload is None:
            return {"status": "error", "message": "failed to resolve control payload"}

        source_payload = dict(staged_payload)

        is_valid, validation_error = _validate_control_payload(staged_payload)
        if not is_valid:
            return {"status": "error", "message": validation_error}

        staged_at = datetime.now(timezone.utc).isoformat()
        try:
            _raw_mode = normalize_run_mode(staged_payload.get("mode"))
            _backlog_path = resolve_backlog_source_path(
                staged_payload.get("backlog_source"),
                BACKLOG_QUEUE_FILE,
                require_exists=_raw_mode == "backlog",
            )
            _backlog_source = str(_backlog_path).replace("\\", "/")
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}

        staged_payload = {
            "hunt_name": str(source_payload.get("hunt_name") or "").strip(),
            "generations": int(source_payload["generations"]),
            "batch_size": int(source_payload["batch_size"]),
            "origin": str(source_payload.get("origin") or "UI_CONTROL"),
            "population_size": int(source_payload["population_size"]),
            "seeds_per_candidate": int(source_payload["seeds_per_candidate"]),
            "n_grid": int(source_payload["n_grid"]),
            "t_steps": int(source_payload["t_steps"]),
            "dt": float(source_payload["dt"]),
            "mode": _raw_mode,
            "backlog_source": _backlog_source,
            "staged_at": staged_at,
            "initiating_config": True,
        }
        optional_numeric_keys = (
            "l_domain",
            "collapse_threshold",
            "mutation_rate",
            "crossover_rate",
            "survival_fraction",
            "predator_sweep_frequency",
            "min_queue_depth",
            "poll_interval",
            "bleed_workers",
            "artifact_gc_min_age_seconds",
            "job_lease_timeout_seconds",
            "result_lease_timeout_seconds",
            "worker_heartbeat_ttl_seconds",
            "max_queue_depth",
        )
        for key in optional_numeric_keys:
            value = source_payload.get(key)
            if value is None:
                continue
            if key in {
                "l_domain",
                "collapse_threshold",
                "mutation_rate",
                "crossover_rate",
                "survival_fraction",
                "poll_interval",
            }:
                staged_payload[key] = float(value)
            else:
                staged_payload[key] = int(value)
        # --- [ALETHEIA V4.4] Propagate Physics & System Dials ---
        for key in ["param_D", "param_eta", "param_a", "param_rho_vac", "system"]:
            if key in source_payload:
                staged_payload[key] = source_payload[key]
        # --------------------------------------------------------
        if not staged_payload["hunt_name"]:
            return {"status": "error", "message": "hunt_name is required for session workspace naming"}

        canonical = json.dumps(staged_payload, sort_keys=True)
        config_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        config_dir = Path("input_configs")
        config_dir.mkdir(parents=True, exist_ok=True)
        staged_name = f"initiating_config_{config_hash[:12]}.json"
        staged_path = config_dir / staged_name

        with open(staged_path, "w", encoding="utf-8") as handle:
            json.dump(staged_payload, handle, indent=2)

        await _broadcast_payload(
            {
                "type": "log",
                "level": "INFO",
                "message": f"Staged initiating config: {staged_name}",
            }
        )
        _emit_terminal_debug(
            "backend",
            f"Staged initiating config {staged_name} mode={_raw_mode} backlog_source={_backlog_source}",
            "INFO",
        )
        _emit_terminal_debug(
            "orchestrator",
            f"Mode shifted to {_raw_mode.upper()}. Target: staged config {staged_name}.",
            "INFO",
        )

        return {
            "status": "success",
            "config_hash": config_hash,
            "staged_path": str(staged_path).replace("\\", "/"),
            "staged_at": staged_at,
            "origin": staged_payload["origin"],
            "hunt_name": staged_payload["hunt_name"],
            "initiating_config": True,
        }
    except Exception as exc:
        logging.error(f"Stage request failed:\n{traceback.format_exc()}")
        _emit_terminal_debug("backend", "Stage request failed.", "ERROR")
        return {"status": "error", "message": "An internal server error occurred while staging."}


@app.post("/api/control/start")
async def control_start(req: ControlStartRequest):
    """
    Queue a new ASTE generation run via IPC.
    DO NOT spawn orchestrator directly — push manifest to cluster queue.
    """
    if not IPC_AVAILABLE:
        return {
            "status": "error",
            "message": "orchestrator package not available; cluster IPC disabled"
        }

    import sys
    try:
        qm = QueueManager(queue_file=BACKLOG_QUEUE_FILE, result_file=BACKLOG_RESULT_FILE)
        staged_path = (req.staged_path or "").strip() or None
        staged_at = (req.staged_at or "").strip() or None
        staged_config_hash = (req.staged_config_hash or "").strip() or None
        origin = (req.origin or "UI_CONTROL").strip() or "UI_CONTROL"

        if not staged_path:
            return {
                "status": "error",
                "message": "staged_path is required. Stage an initiating config before start.",
            }

        staged_file = Path(staged_path)
        if not staged_file.exists():
            return {
                "status": "error",
                "message": f"staged_path not found: {staged_path}",
            }

        with open(staged_file, "r", encoding="utf-8") as handle:
            staged_payload = _canonicalize_control_payload(json.load(handle))

        is_valid, validation_error = _validate_control_payload(staged_payload)
        if not is_valid:
            return {
                "status": "error",
                "message": f"invalid staged config: {validation_error}",
            }

        if not staged_config_hash:
            canonical = json.dumps(staged_payload, sort_keys=True)
            staged_config_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        if not staged_at:
            staged_at = str(staged_payload.get("staged_at") or datetime.now(timezone.utc).isoformat())

        hunt_name = (req.hunt_name or staged_payload.get("hunt_name") or "").strip()
        if not hunt_name:
            return {
                "status": "error",
                "message": "hunt_name is required. Stage/start must provide explicit hunt_name.",
            }

        session_info = create_session_workspace(hunt_name)
        initialize_ledger_schema(session_info["db_file"])
        # Phase 0 Hardening: create the Smart Sim Manifest immediately at run start.
        try:
            init_run_manifest(session_info)
        except RuntimeError as _mf_exc:
            return {"status": "error", "message": f"Failed to initialise run manifest: {_mf_exc}"}
        pointer_payload = {
            "hunt_name": session_info["hunt_name"],
            "run_id": session_info["run_id"],
            "session_dir": session_info["session_dir"],
            "db_file": session_info["db_file"],
            "provenance_dir": session_info["provenance_dir"],
            "created_at": session_info["created_at"],
        }
        pointer_payload = write_active_run_pointer_atomic(pointer_payload)

        simulation_payload: Dict[str, Any] = {
            "N_grid": int(staged_payload["n_grid"]),
            "T_steps": int(staged_payload["t_steps"]),
            "dt": float(staged_payload["dt"]),
        }
        params: Dict[str, Any] = {
            "generations": int(staged_payload["generations"]),
            "batch_size": int(staged_payload["batch_size"]),
            "origin": origin,
            "population_size": int(staged_payload["population_size"]),
            "seeds_per_candidate": int(staged_payload["seeds_per_candidate"]),
            "n_grid": int(staged_payload["n_grid"]),
            "t_steps": int(staged_payload["t_steps"]),
            "dt": float(staged_payload["dt"]),
            "simulation": simulation_payload,
            "mode": normalize_run_mode(staged_payload.get("mode") or req.mode),
            "backlog_source": "",
        }
        backlog_path = resolve_backlog_source_path(
            staged_payload.get("backlog_source") or req.backlog_source,
            BACKLOG_QUEUE_FILE,
            require_exists=params["mode"] == "backlog",
        )
        params["backlog_source"] = str(backlog_path).replace("\\", "/")

        advanced_passthrough_keys = (
            "min_queue_depth",
            "poll_interval",
            "bleed_workers",
            "artifact_gc_min_age_seconds",
            "job_lease_timeout_seconds",
            "result_lease_timeout_seconds",
            "worker_heartbeat_ttl_seconds",
            "max_queue_depth",
        )
        for key in advanced_passthrough_keys:
            value = staged_payload.get(key)
            if value is not None:
                params[key] = value

        if staged_payload.get("l_domain") is not None:
            simulation_payload["L_domain"] = float(staged_payload["l_domain"])
        if staged_payload.get("collapse_threshold") is not None:
            simulation_payload["collapse_threshold"] = float(staged_payload["collapse_threshold"])

        evolution_payload: Dict[str, Any] = {}
        if staged_payload.get("mutation_rate") is not None:
            evolution_payload["mutation_rate"] = float(staged_payload["mutation_rate"])
        if staged_payload.get("crossover_rate") is not None:
            evolution_payload["crossover_rate"] = float(staged_payload["crossover_rate"])
        if staged_payload.get("survival_fraction") is not None:
            evolution_payload["survival_fraction"] = float(staged_payload["survival_fraction"])
        if staged_payload.get("predator_sweep_frequency") is not None:
            evolution_payload["predator_sweep_frequency"] = int(staged_payload["predator_sweep_frequency"])

        if evolution_payload:
            params["evolution"] = evolution_payload
        # --- [ALETHEIA V4.4] Propagate Physics & System Dials to Engine ---
        for key in ["param_D", "param_eta", "param_a", "param_rho_vac", "system"]:
            if key in staged_payload:
                params[key] = staged_payload[key]
        # ------------------------------------------------------------------

        manifest = None
        queued_count = 0
        if params["mode"] == "backlog":
            jobs_to_push = []
            backlog_source_path = Path(params["backlog_source"])

            if backlog_source_path.is_dir():
                for f in sorted(backlog_source_path.glob("*.json")):
                    job_params = json.loads(f.read_text(encoding="utf-8"))
                    if not isinstance(job_params, dict):
                        return {
                            "status": "error",
                            "message": f"Invalid backlog entry in {f}: expected JSON object",
                        }
                    merged_params = {**params, **job_params}
                    job_manifest = JobManifest.from_params(
                        params=merged_params, generation=0, mode="backlog", origin=origin,
                        hunt_name=pointer_payload["hunt_name"], session_dir=pointer_payload["session_dir"],
                        session_db_path=pointer_payload["db_file"],
                        generation_dir=str(resolve_generation_dir(pointer_payload["session_dir"], 0))
                    )
                    jobs_to_push.append(job_manifest.to_json())
                    manifest = job_manifest
            elif backlog_source_path.is_file():
                data = json.loads(backlog_source_path.read_text(encoding="utf-8"))
                if not isinstance(data, list):
                    return {
                        "status": "error",
                        "message": f"Invalid backlog file {backlog_source_path}: expected JSON list",
                    }
                for job_params in data:
                    if not isinstance(job_params, dict):
                        return {
                            "status": "error",
                            "message": f"Invalid backlog entry in {backlog_source_path}: expected JSON object items",
                        }
                    merged_params = {**params, **job_params}
                    job_manifest = JobManifest.from_params(
                        params=merged_params, generation=0, mode="backlog", origin=origin,
                        hunt_name=pointer_payload["hunt_name"], session_dir=pointer_payload["session_dir"],
                        session_db_path=pointer_payload["db_file"],
                        generation_dir=str(resolve_generation_dir(pointer_payload["session_dir"], 0))
                    )
                    jobs_to_push.append(job_manifest.to_json())
                    manifest = job_manifest

            queued_count = len(jobs_to_push)
            if queued_count == 0:
                return {
                    "status": "error",
                    "message": f"No backlog jobs discovered at {params['backlog_source']}",
                }

            qm.push_jobs_batch(jobs_to_push)
            _emit_terminal_debug("backend", f"Queued {queued_count} backlog jobs from {params['backlog_source']}", "INFO")
        else:
            manifest = JobManifest.from_params(
                params=params, generation=0, seed=0, origin=origin,
                staged_path=staged_path, staged_at=staged_at, staged_config_hash=staged_config_hash,
                hunt_name=pointer_payload["hunt_name"], session_dir=pointer_payload["session_dir"],
                session_db_path=pointer_payload["db_file"],
                generation_dir=str(resolve_generation_dir(pointer_payload["session_dir"], 0)),
                mode=params["mode"], backlog_source=params["backlog_source"]
            )
            qm.push_job(manifest.to_json())
            _emit_terminal_debug("backend", f"Queued evolution control job_id={manifest.job_id}", "INFO")
        target_runs = int(params.get("generations", 0)) * int(params.get("batch_size", 0))
        _emit_terminal_debug(
            "orchestrator",
            f"Mode shifted to {str(params['mode']).upper()}. Target: {target_runs} runs.",
            "INFO",
        )
        # --- Auto-Booter: Launch orchestrator and workers ---
        # 1. Purge any lingering ghosts before starting fresh
        try:
            _purge_ghost_workers()
            _kill_saved_pids()
            _emit_terminal_debug("backend", "Phantom workers purged successfully.", "INFO")
        except Exception as e:
            logging.warning(f"Ghost purge failed: {e}")

        RUNTIME_LOGS_DIR.mkdir(parents=True, exist_ok=True)

        # 2. Launch orchestrator process
        if state["orchestrator_process"] is None or getattr(state["orchestrator_process"], "returncode", None) is not None:
            orchestrator_env = os.environ.copy()
            orchestrator_env["PYTHONPATH"] = str(Path(__file__).parent.resolve())
            
            orchestrator_cmd = [
                sys.executable, 
                "-u", 
                "-m", "orchestrator.orchestrator_service", 
                "--config", str(staged_path)
            ]
            orch_log = open(RUNTIME_LOGS_DIR / "orchestrator.log", "a", encoding="utf-8")
            state["orchestrator_process"] = subprocess.Popen(
                orchestrator_cmd,
                env=orchestrator_env,
                stdout=orch_log,
                stderr=subprocess.STDOUT,
                cwd=str(Path(__file__).parent.resolve()),
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )

        # 3. Launch worker daemons
        worker_count = req.worker_count or req.workerCount or 1

        for proc in state["worker_processes"]:
            try:
                proc.terminate()
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        state["worker_processes"] = []

        for i in range(worker_count):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(i)
            worker_id = f"worker_{i}"
            env["ASTE_WORKER_ID"] = worker_id
            env["PYTHONPATH"] = str(Path(__file__).parent.resolve())
            
            worker_cmd = [sys.executable, "-u", "worker_daemon.py"]
            w_log = open(RUNTIME_LOGS_DIR / f"worker_gpu{i}.log", "a", encoding="utf-8")
            proc = subprocess.Popen(
                worker_cmd,
                env=env,
                stdout=w_log,
                stderr=subprocess.STDOUT,
                cwd=str(Path(__file__).parent.resolve()),
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )
            state["worker_processes"].append(proc)

        # Track PIDs locally to manage state crashes
        _save_pids()

        _emit_terminal_debug("backend", f"Cluster Online: {worker_count} GPU Workers Booted via Popen.", "INFO")
        # --- End Auto-Booter ---
        if params["mode"] == "backlog":
            return {
                "status": "success",
                "message": f"Successfully queued {queued_count} backlog jobs",
                "mode": params["mode"],
                "queued_count": queued_count,
                "job_id": "backlog_batch",
                "config_hash": "backlog_batch",
                "staged_path": staged_path,
                "staged_at": staged_at,
                "staged_config_hash": staged_config_hash,
                "hunt_name": pointer_payload["hunt_name"],
                "session_dir": pointer_payload["session_dir"],
                "session_db_path": pointer_payload["db_file"],
                "initiating_config": True,
                "workers_launched": worker_count,
            }
        return {
            "status": "success",
            "message": f"Queued {params['generations']} generations (batch_size={params['batch_size']})",
            "job_id": manifest.job_id if manifest else None,
            "config_hash": manifest.config_hash if manifest else None,
            "staged_path": manifest.staged_path if manifest else None,
            "staged_at": manifest.staged_at if manifest else None,
            "staged_config_hash": manifest.staged_config_hash if manifest else None,
            "hunt_name": pointer_payload["hunt_name"],
            "session_dir": pointer_payload["session_dir"],
            "session_db_path": pointer_payload["db_file"],
            "initiating_config": True,
            "workers_launched": worker_count,
        }
    except Exception as exc:
        logging.error(f"Start request failed:\n{traceback.format_exc()}")
        _emit_terminal_debug("backend", "Start request failed. Check logs.", "ERROR")
        return {"status": "error", "message": "Failed to start the sequence. Check logs for details."}

@app.post("/api/control/stop")
async def control_stop():
    """
    Request graceful orchestrator stop and clear active pointer.
    """
    from orchestrator.storage.artifact_gc import purge_old_artifacts
    
    # Cancel dangling stream tasks
    for task in state.get("stream_tasks", []):
        task.cancel()
    state["stream_tasks"] = []

    # --- Kill Switch: Terminate orchestrator and workers (asyncio) ---
    for proc in state["worker_processes"]:
        try:
            proc.kill()
        except Exception:
            pass
    state["worker_processes"] = []
    if state["orchestrator_process"] is not None:
        try:
            state["orchestrator_process"].kill()
        except Exception:
            pass
        state["orchestrator_process"] = None
        
    _save_pids()

    # --- [ALETHEIA V4.4] GC Sweep for Aborted Backlog ---
    try:
        db_file, session_dir, prov_dir = resolve_active_session_paths()
        if session_dir and session_dir.exists():
            purged = purge_old_artifacts(session_dir, min_age_seconds=0)
            _emit_terminal_debug("backend", f"Aborted: Purged {purged} orphaned artifacts.", "INFO")
    except Exception:
        pass
        
    stop_file = Path("stop_after_gen.txt")
    try:
        stop_file.write_text(
            f"stop_requested_at={datetime.now(timezone.utc).isoformat()}\n",
            encoding="utf-8",
        )
    except Exception as exc:
        logging.error(f"Failed to write stop signal file:\n{traceback.format_exc()}")
        _emit_terminal_debug("backend", f"Failed to write stop signal file", "ERROR")
        return {"status": "error", "message": f"failed to request stop."}

    clear_active_run_pointer()
    _emit_terminal_debug(
        "backend",
        "Stop requested; orchestrator and workers terminated; active pointer cleared.",
        "WARN",
    )
    return {
        "status": "success",
        "message": "Stop requested; orchestrator and workers terminated; active session pointer cleared",
        "stop_signal_file": str(stop_file.resolve()).replace("\\", "/"),
    }
    

@app.get("/api/debug/terminals")
async def debug_terminals(lines: int = 150):
    max_lines = max(20, min(lines, 500))
    feeds = []
    combined: list[str] = []

    for source in DEBUG_LOG_SOURCES:
        source_path = Path(str(source["path"]))
        tail = _tail_file_lines(source_path, max_lines)
        feeds.append(
            {
                "id": source["id"],
                "name": source["name"],
                "path": str(source_path).replace("\\", "/"),
                "lines": tail,
            }
        )
        combined.extend([f"[{source['name']}] {line}" for line in tail])

    return {
        "status": "success",
        "feeds": feeds,
        "combined": combined[-max_lines:],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/debug/tests/suites")
async def debug_test_suites():
    file_suites = []
    if TESTS_DIR.exists() and TESTS_DIR.is_dir():
        for test_file in sorted(TESTS_DIR.glob("test_*.py")):
            rel_path = str(test_file.relative_to(Path(__file__).parent)).replace("\\", "/")
            file_suites.append(
                {
                    "id": f"file:{rel_path}",
                    "label": test_file.stem,
                    "command": f"python -m pytest {rel_path} -q",
                }
            )

    return {
        "status": "success",
        "suites": [
            {"id": suite_id, "label": suite_id.upper(), "command": " ".join(cmd)}
            for suite_id, cmd in DEBUG_TEST_SUITES.items()
        ] + file_suites,
    }


@app.post("/api/debug/tests/run")
async def debug_run_tests(req: DebugTestRunRequest):
    suite = (req.suite or "").strip().lower()
    cmd: list[str]
    if suite.startswith("file:"):
        raw_rel = suite[5:]
        candidate = (Path(__file__).parent / raw_rel).resolve()
        tests_root = TESTS_DIR.resolve()
        if not str(candidate).startswith(str(tests_root)) or not candidate.exists() or candidate.suffix != ".py":
            return {
                "status": "error",
                "message": f"Unknown test file suite '{suite}'.",
            }
        cmd = ["python", "-m", "pytest", str(candidate.relative_to(Path(__file__).parent)).replace("\\", "/"), "-q"]
    elif suite in DEBUG_TEST_SUITES:
        cmd = DEBUG_TEST_SUITES[suite]
    else:
        return {
            "status": "error",
            "message": f"Unknown suite '{suite}'.",
            "available_suites": sorted(DEBUG_TEST_SUITES.keys()),
        }

    started_at = datetime.now(timezone.utc)

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(Path(__file__).parent),
            text=True,
            capture_output=True,
            timeout=DEBUG_TEST_TIMEOUT_SECONDS,
            check=False,
        )
        output = (completed.stdout or "")
        if completed.stderr:
            output += ("\n" if output else "") + completed.stderr
        output_lines = output.splitlines()
        duration = (datetime.now(timezone.utc) - started_at).total_seconds()

        RUNTIME_LOGS_DIR.mkdir(parents=True, exist_ok=True)
        test_log = RUNTIME_LOGS_DIR / "test_runner.log"
        with open(test_log, "a", encoding="utf-8") as handle:
            handle.write(
                f"\n=== [{started_at.isoformat()}] suite={suite} exit={completed.returncode} duration={duration:.2f}s ===\n"
            )
            handle.write(output + "\n")

        return {
            "status": "success" if completed.returncode == 0 else "fail",
            "suite": suite,
            "exit_code": completed.returncode,
            "duration_seconds": duration,
            "output": "\n".join(output_lines[-500:]),
            "started_at": started_at.isoformat(),
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "suite": suite,
            "message": f"Test run timed out after {DEBUG_TEST_TIMEOUT_SECONDS}s.",
        }


@app.get("/api/system/telemetry")
async def get_system_telemetry(ttl_seconds: float | None = None):
    global stale_worker_alert_cache
    try:
        effective_ttl = float(ttl_seconds) if ttl_seconds is not None else SYSTEM_TELEMETRY_TTL_SECONDS
        if effective_ttl <= 0:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "ttl_seconds must be a positive number"},
            )
    except (TypeError, ValueError):
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "ttl_seconds must be numeric"},
        )

    try:
        queue_manager = QueueManager(queue_file=BACKLOG_QUEUE_FILE, result_file=BACKLOG_RESULT_FILE)
        queue_depth = queue_manager.size()
        heartbeats = queue_manager.get_worker_heartbeats()
        claims_by_worker = queue_manager.get_claim_counts_by_worker()
        active_workers = queue_manager.list_active_workers(effective_ttl)
        stale_workers = queue_manager.list_stale_workers(effective_ttl)
        counters = queue_manager.get_telemetry_counters()
        dlq_count = 0
        try:
            active_paths = resolve_active_session_paths(require_exists=True)
            db_path = Path(str(active_paths["session_dir"])) / SESSION_DB_BASENAME
            if db_path.exists():
                # Utilizing run_in_threadpool slightly manually for a rapid query here:
                def _get_dlq_count():
                    conn = sqlite3.connect(str(db_path), timeout=5.0)
                    try:
                        cur = conn.cursor()
                        try:
                            cur.execute("SELECT COUNT(*) FROM dead_letter_queue")
                            row = cur.fetchone()
                            return int(row[0]) if row and row[0] is not None else 0
                        except sqlite3.OperationalError:
                            cur.execute("SELECT COUNT(*) FROM runs WHERE status LIKE '%fail%' OR status LIKE '%error%'")
                            row = cur.fetchone()
                            return int(row[0]) if row and row[0] is not None else 0
                    finally:
                        conn.close()
                dlq_count = await run_in_threadpool(_get_dlq_count)
        except Exception:
            dlq_count = 0
            
        now = time.time()
        workers: list[FleetWorkerStatus] = []
        for worker_id in sorted(heartbeats.keys()):
            heartbeat_ts = float(heartbeats[worker_id])
            age_seconds = max(0.0, now - heartbeat_ts)
            if age_seconds > 60.0:
                last_alert = float(stale_worker_alert_cache.get(worker_id, 0.0))
                if now - last_alert >= 60.0:
                    _emit_terminal_debug(
                        "orchestrator",
                        f"Worker {worker_id} heartbeat stale ({age_seconds:.1f}s). Possible CUDA OOM or hang.",
                        "CRITICAL",
                    )
                    stale_worker_alert_cache[worker_id] = now
            else:
                stale_worker_alert_cache.pop(worker_id, None)
            workers.append(
                FleetWorkerStatus(
                    worker_id=worker_id,
                    last_heartbeat_epoch=heartbeat_ts,
                    age_seconds=age_seconds,
                    state="stale" if worker_id in stale_workers else "active",
                    in_flight_claims=int(claims_by_worker.get(worker_id, 0)),
                )
            )

        return FleetTelemetryResponse(
            status="success",
            timestamp=datetime.now(timezone.utc).isoformat(),
            queue_depth=queue_depth,
            dlq_count=dlq_count,
            active_workers=active_workers,
            stale_workers=stale_workers,
            total_claims_processed=int(counters.get("total_claims_processed", 0)),
            worker_heartbeat_ttl_seconds=effective_ttl,
            workers=workers,
        )
    except Exception as exc:
        logging.error(f"Failed to gather telemetry:\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Failed to gather system telemetry due to internal error."},
        )

# ---- Data Endpoints ----
@app.get("/api/data/files")
async def list_data_files():
    """List all .h5 and .csv files in active session workspace."""
    try:
        active_paths, error_response = _active_paths_or_error()
        if error_response is not None:
            return error_response

        data_path = Path(str(active_paths["session_dir"]))
        if not data_path.exists() or not data_path.is_dir():
            return {"files": []}

        files = sorted([
            {
                "name": str(f.relative_to(data_path)).replace("\\", "/"),
                "size": f.stat().st_size,
                "modified": int(f.stat().st_mtime),
                "type": f.suffix
            }
            for f in data_path.rglob("*")
            if f.is_file() and f.suffix in {".h5", ".csv"}
        ])
        return {"files": files}
    except Exception as exc:
        logging.error(f"Error listing data files:\n{traceback.format_exc()}")
        return {"error": "Failed to retrieve data files", "files": []}


@app.get("/api/data/ledger/runs")
async def list_ledger_runs():
    try:
        return LedgerRunsResponse(status="success", runs=_list_known_runs())
    except Exception as exc:
        logging.error(f"Error listing ledger runs:\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"status": "error", "message": "Internal error retrieving runs", "runs": []})


@app.get("/api/data/ledger/{run_id}")
async def get_ledger_rows(run_id: str, limit: int = 50, offset: int = 0):
    ledger_run, error_response = _resolve_ledger_run_or_error(run_id)
    if error_response is not None:
        return error_response
    if ledger_run is None:
        return JSONResponse(status_code=404, content={"status": "error", "message": f"run_id not found: {run_id}"})

    try:
        safe_limit, safe_offset = _coerce_pagination(limit, offset)
    except (TypeError, ValueError):
        return JSONResponse(status_code=400, content={"status": "error", "message": "limit and offset must be integers"})

    db_file = Path(str(ledger_run["db_file"]))
    if not db_file.exists() or not db_file.is_file():
        return JSONResponse(status_code=404, content={"status": "error", "message": f"ledger database missing for run_id: {run_id}"})

    try:
        # PUSHED TO THREADPOOL to avoid blocking the event loop on large SQLite read
        rows, total_rows = await run_in_threadpool(_fetch_ledger_rows, db_file, safe_limit, safe_offset)
    except sqlite3.Error as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": f"failed to query ledger: {exc}"})
    except Exception as exc:
        logging.error(f"Internal error fetching ledger rows:\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"status": "error", "message": "Internal error fetching ledger rows"})

    return LedgerRowsResponse(
        status="success",
        run_id=str(ledger_run.get("run_id") or run_id),
        session_dir=str(ledger_run.get("session_dir") or ""),
        db_file=str(db_file),
        rows=rows,
        pagination={
            "limit": safe_limit,
            "offset": safe_offset,
            "returned": len(rows),
            "total": total_rows,
        },
    )

@app.get("/api/data/download/{file_path:path}")
async def download_data_file(file_path: str):
    """Serve a .h5 or .csv file from active session workspace."""
    active_paths, error_response = _active_paths_or_error()
    if error_response is not None:
        return error_response
    if active_paths is None:
        return JSONResponse(status_code=409, content={"status": "error", "message": ACTIVE_RUN_POINTER_ERROR})

    session_dir = Path(str(active_paths["session_dir"]))
    candidate = (session_dir / file_path).resolve()
    if not str(candidate).startswith(str(session_dir.resolve())):
        return JSONResponse(status_code=400, content={"status": "error", "message": "invalid file path"})

    # Safety: prevent path traversal and invalid file types
    if not candidate.exists() or candidate.suffix not in {".h5", ".csv"}:
        return {"error": "file not found or invalid type"}
    return FileResponse(candidate, filename=candidate.name)


@app.get("/api/plugins/visualizers")
async def list_visualizer_plugins():
    manifest_path = Path(__file__).parent / "plugins" / "visualizers" / "manifest.json"
    if not manifest_path.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": "visualizer manifest not found"})

    try:
        manifest = _load_plugin_manifest(manifest_path)
        return {
            "status": "success",
            "manifest_path": str(manifest_path).replace("\\", "/"),
            "manifest": manifest,
            "plugins": manifest.get("plugins", []),
        }
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@app.get("/api/plugins/tools")
async def list_tool_plugins():
    manifest_path = Path(__file__).parent / "plugins" / "tools" / "manifest.json"
    if not manifest_path.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": "tools manifest not found"})

    try:
        manifest = _load_plugin_manifest(manifest_path)
        return {
            "status": "success",
            "manifest_path": str(manifest_path).replace("\\", "/"),
            "manifest": manifest,
        }
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@app.post("/api/plugins/execute")
async def execute_plugin(req: PluginExecuteRequest):
    plugin_id = str(req.plugin_id or "").strip()
    if not plugin_id:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "plugin_id is required"},
        )

    try:
        plugin_args = _validate_plugin_args(req.args or [])
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"status": "error", "message": str(exc)})
    if plugin_args:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "custom args are not supported for this endpoint"},
        )

    registry = _collect_plugin_registry()
    plugin_entry = registry.get(plugin_id)
    if not plugin_entry:
        allowed_ids = sorted(registry.keys())
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": f"plugin_id '{plugin_id}' is not allowlisted",
                "allowed_plugin_ids": allowed_ids,
            },
        )

    script_path = Path(str(plugin_entry["script_path"]))
    if not script_path.exists() or not script_path.is_file():
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": f"plugin script missing: {script_path}"},
        )

    try:
        cmd = _build_plugin_command(plugin_id, script_path, req)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"status": "error", "message": str(exc)})

    artifact_path = str(req.artifact_path or "").strip() or None
    execution_id = hashlib.sha256(
        f"{plugin_id}:{artifact_path or ''}:{datetime.now(timezone.utc).isoformat()}".encode("utf-8")
    ).hexdigest()[:12]
    submitted_at = datetime.now(timezone.utc).isoformat()
    log_path = RUNTIME_LOGS_DIR / f"plugin_execute_{execution_id}.log"
    log_handle = None

    popen_kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "close_fds": True,
        "cwd": str(Path(__file__).parent.resolve()),
        "env": {
            **os.environ,
            "ASTE_PLUGIN_EXECUTION_ID": execution_id,
            "ASTE_PLUGIN_ID": plugin_id,
            "ASTE_PLUGIN_TIMEOUT_SECONDS": str(PLUGIN_EXECUTION_TIMEOUT_SECONDS),
        },
    }

    try:
        RUNTIME_LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_path, "a", encoding="utf-8")
        popen_kwargs["stdout"] = log_handle
        popen_kwargs["stderr"] = log_handle

        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        else:
            popen_kwargs["start_new_session"] = True

        process = subprocess.Popen(cmd, **popen_kwargs)
        _record_plugin_execution_event(
            {
                "event": "spawned",
                "execution_id": execution_id,
                "plugin_id": plugin_id,
                "pid": process.pid,
                "submitted_at": submitted_at,
                "artifact_path": artifact_path,
                "script_path": str(script_path).replace("\\", "/"),
            }
        )
    except Exception as exc:
        _record_plugin_execution_event(
            {
                "event": "failed_to_spawn",
                "execution_id": execution_id,
                "plugin_id": plugin_id,
                "submitted_at": submitted_at,
                "artifact_path": artifact_path,
                "error": str(exc),
            }
        )
        return JSONResponse(status_code=500, content={"status": "error", "message": f"failed to launch plugin: {exc}"})
    finally:
        if log_handle:
            log_handle.close()

    return {
        "status": "success",
        "mocked": False,
        "accepted": True,
        "plugin_id": plugin_id,
        "artifact_path": artifact_path,
        "execution_id": execution_id,
        "submitted_at": submitted_at,
        "pid": process.pid,
        "log_path": str(log_path).replace("\\", "/"),
        "command": cmd,
        "message": "Plugin execution accepted and launched asynchronously.",
    }


@app.get("/api/data/artifacts/search")
async def search_artifacts(q: str = "", limit: int = 100):
    capped_limit = max(1, min(int(limit), 1000))
    results: list[dict] = []

    session_dir: Path | None = None
    try:
        active_paths = resolve_active_session_paths(require_exists=True)
        session_dir = Path(str(active_paths["session_dir"]))
    except (ActiveRunPointerError, ValueError):
        session_dir = None

    if session_dir is not None:
        session_hits = _search_artifact_root(session_dir, q, capped_limit - len(results), "active_session")
        results.extend(session_hits)

    archive_dir = Path(__file__).parent / "archive_runs"
    archive_hits = _search_artifact_root(archive_dir, q, capped_limit - len(results), "archive_runs")
    results.extend(archive_hits)

    return {
        "status": "success",
        "query": q,
        "limit": capped_limit,
        "searched": {
            "active_session": str(session_dir).replace("\\", "/") if session_dir else None,
            "archive_runs": str(archive_dir).replace("\\", "/"),
        },
        "results": results,
        "artifacts": results,
    }


def _find_artifact_for_config_hash(config_hash: str) -> Path | None:
    target_name = f"rho_history_{config_hash}.h5"

    try:
        active_paths = resolve_active_session_paths(require_exists=True)
        session_dir = Path(str(active_paths["session_dir"]))
        for hit in session_dir.rglob(target_name):
            if hit.is_file():
                return hit.resolve()
    except Exception:
        pass

    data_dir = Path(DATA_DIR)
    direct_path = (data_dir / target_name).resolve()
    if direct_path.exists() and direct_path.is_file():
        return direct_path

    archive_dir = Path(__file__).parent / "archive_runs"
    for hit in archive_dir.rglob(target_name):
        if hit.is_file():
            return hit.resolve()

    return None


@app.get("/api/data/tensor/{config_hash}")
async def get_tensor_binary(config_hash: str):
    artifact_path = _find_artifact_for_config_hash(config_hash)
    if artifact_path is None:
        return JSONResponse(status_code=404, content={"status": "error", "message": "Artifact not found"})

    try:
        safe_path = _resolve_safe_artifact_path(str(artifact_path))
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"status": "error", "message": str(exc)})

    try:
        # Prevent event loop blocking using run_in_threadpool wrapper
        tensor_bytes = await run_in_threadpool(_extract_tensor_binary_sync, safe_path)
        return Response(content=tensor_bytes, media_type="application/octet-stream")
    except ValueError as exc:
        return JSONResponse(status_code=422, content={"status": "error", "message": str(exc)})
    except Exception as exc:
        logging.error(f"Tensor extraction failed:\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"status": "error", "message": "Internal error extracting tensor data"})


@app.get("/api/data/pointcloud/hash/{config_hash}/{dataset_name}")
async def get_pointcloud_by_hash(config_hash: str, dataset_name: str, threshold: float = 0.05):
    """
    Streams selected HDF5 tensor data as an [N x 4] Float32 point cloud buffer.
    """
    if not _is_valid_config_hash(config_hash):
        raise HTTPException(status_code=400, detail="Invalid config_hash format.")
    if dataset_name not in ALLOWED_POINTCLOUD_DATASETS:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found.")

    safe_threshold = float(np.clip(threshold, 0.0, 1.0))
    artifact_path = _find_artifact_for_config_hash(config_hash)
    if artifact_path is None:
        raise HTTPException(status_code=404, detail="Artifact not found.")

    try:
        safe_path = _resolve_safe_artifact_path(str(artifact_path))
        if safe_path is None:
            raise HTTPException(status_code=404, detail="Artifact not found.")

        # Prevent event loop blocking using run_in_threadpool wrapper
        binary_buffer = await run_in_threadpool(_extract_pointcloud_sync, safe_path, dataset_name, safe_threshold)

        return Response(
            content=binary_buffer,
            media_type="application/octet-stream",
            headers={"Content-Length": str(len(binary_buffer))},
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        logging.error(f"PointCloud hash extraction error:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal processing error")


# ==========================================
# WEBGL POINT CLOUD STREAMER
# ==========================================
@app.get("/api/data/pointcloud/{filename}/{dataset_name}")
async def get_pointcloud(filename: str, dataset_name: str, threshold: float = 0.05):
    """
    Streams HDF5 tensor data as a raw Float32 binary buffer.
    Performs server-side 'vacuum filtering' to save massive network bandwidth.
    """
    if dataset_name not in ALLOWED_POINTCLOUD_DATASETS:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found.")

    file_path = (Path(DATA_DIR) / filename).resolve()
    if not file_path.exists() or file_path.suffix.lower() not in {'.h5', '.hdf5'}:
        raise HTTPException(status_code=404, detail="Artifact not found.")

    try:
        safe_path = _resolve_safe_artifact_path(str(file_path))
        if safe_path is None:
            raise HTTPException(status_code=404, detail="Artifact not found.")

        safe_threshold = float(np.clip(threshold, 0.0, 1.0))
        
        # Prevent event loop blocking using run_in_threadpool wrapper
        binary_buffer = await run_in_threadpool(_extract_pointcloud_sync, safe_path, dataset_name, safe_threshold)

        return Response(
            content=binary_buffer,
            media_type="application/octet-stream",
            headers={"Content-Length": str(len(binary_buffer))}
        )

    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"PointCloud extraction error:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal processing error")


@app.get("/api/run/active")
async def get_active_run():
    try:
        payload = read_active_run_pointer()
        # Phase 3: augment with backlog observability fields
        session_dir = Path(payload.get("session_dir", ""))
        session_name = (
            payload.get("session_name")
            or f"{payload.get('hunt_name', '')}_{payload.get('run_id', '')}"
        )
        manifest_p = run_manifest_path(session_dir, session_name) if session_dir.as_posix() != "." else None

        mode = "evolution"
        queue_remaining: int | None = None
        chunks_completed = 0
        purge_cycles_completed = 0
        backlog_source: str | None = None
        manifest_data: dict | None = None

        if manifest_p and manifest_p.exists():
            try:
                with open(manifest_p, "r", encoding="utf-8") as _mf:
                    manifest_data = json.load(_mf)
            except Exception:
                pass

        # Peek the active backlog source for queue depth
        _backlog_path = Path(BACKLOG_QUEUE_FILE)
        if _backlog_path.exists():
            try:
                with open(_backlog_path, "r", encoding="utf-8") as _bf:
                    _bl_data = json.load(_bf)
                queue_remaining = len(_bl_data) if isinstance(_bl_data, list) else None
            except Exception:
                pass

        if manifest_data:
            gens = manifest_data.get("generations", [])
            # If any generation entry carries mode=backlog the run is a backlog run
            if any(g.get("mode") == "backlog" for g in gens):
                mode = "backlog"
                backlog_entries = [g for g in gens if g.get("mode") == "backlog"]
                chunks_completed = int(max((g.get("chunks_completed") or g.get("chunk") or 0) for g in backlog_entries) if backlog_entries else 0)
                purge_cycles_completed = int(max((g.get("purge_cycles_completed") or 0) for g in backlog_entries) if backlog_entries else 0)
                for entry in reversed(backlog_entries):
                    if entry.get("backlog_source"):
                        backlog_source = str(entry.get("backlog_source"))
                        break

        observability: dict = {
            "mode": mode,
            "queue_remaining": queue_remaining,
            "chunks_completed": chunks_completed,
            "purge_cycles_completed": purge_cycles_completed,
            "backlog_source": backlog_source,
            "manifest_path": str(manifest_p).replace("\\", "/") if manifest_p else None,
        }

        return {"status": "success", "active_run": payload, "observability": observability}
    except ActiveRunPointerError as exc:
        return JSONResponse(status_code=409, content={"status": "error", "message": str(exc)})


@app.get("/api/stream/status")
async def stream_status_compat():
    """
    Backward-compatible SSE endpoint for older frontend builds.
    Emits lightweight status heartbeat while websocket migration settles.
    """

    async def event_gen():
        while True:
            payload = {
                "type": "status",
                "state": "running" if active_connections else "idle",
                "details": "SSE compatibility stream (prefer /ws/telemetry)",
            }
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    try:
        while True:
            msg = await websocket.receive_text()
            try:
                data = json.loads(msg)
            except Exception:
                continue
            if data.get("event") == "START_HUNT":
                await websocket.send_text(json.dumps({
                    "type": "log",
                    "message": "Hunt sequence initialized by Sovereign Auditor."
                }))
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "state": "running",
                    "details": "Hunt sequence initialized"
                }))
                # Optionally trigger hunt task here (no nested app or imports)
    except WebSocketDisconnect:
        logging.info("WebSocket disconnected.")
    finally:
        active_connections.discard(websocket)

# --- Ensure GIFS directory exists ---
GIFS_DIR = Path(__file__).parent / "GIFS"
GIFS_DIR.mkdir(exist_ok=True)

# Mount /static/gifs to GIFS folder
app.mount("/static/gifs", StaticFiles(directory=GIFS_DIR), name="gifs")

# Mount React build static assets
if (ACTIVE_BUILD_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=ACTIVE_BUILD_DIR / "static"), name="static")
else:
    # Fallback: mount project root for legacy assets
    PROJECT_ROOT = Path(__file__).parent
    app.mount("/static", StaticFiles(directory=PROJECT_ROOT), name="static")

# Serve React app (catch-all after API/WS routes)
@app.get("/", response_class=HTMLResponse)
async def serve_react_root():
    """Serve build/index.html for React SPA."""
    index_path = ACTIVE_BUILD_DIR / "index.html"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(
                content=f.read(),
                headers={
                    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                    "Pragma": "no-cache",
                },
            )
    # Fallback to legacy UI if build not present
    legacy_path = Path(__file__).parent / "UI" / "index.html"
    if legacy_path.exists():
        with open(legacy_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>React build not found</h1>"

@app.get("/{full_path:path}", response_class=HTMLResponse)
async def serve_react_catch_all(full_path: str):
    """
    Catch-all for React Router: serve build/index.html unless the path
    is an API or WebSocket route (those are matched by their specific handlers above).
    """
    # Prevent shadowing API/WS routes and static files
    if full_path.startswith(("api", "ws", "static")):
        return JSONResponse(status_code=404, content={"error": "not found"})

    index_path = ACTIVE_BUILD_DIR / "index.html"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(
                content=f.read(),
                headers={
                    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                    "Pragma": "no-cache",
                },
            )
    return "<h1>React build not found</h1>"

# --- Directory Monitoring for GIF Updates ---
class AsyncDebouncer:
    def __init__(self, loop: asyncio.AbstractEventLoop, wait_seconds: float, callback):
        self.loop = loop
        self.wait_seconds = wait_seconds
        self.callback = callback
        self._timer: asyncio.TimerHandle | None = None

    def call(self, *args, **kwargs) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = self.loop.call_later(
            self.wait_seconds,
            lambda: asyncio.create_task(self.callback(*args, **kwargs)),
        )


class GifWatcherHandler(FileSystemEventHandler):
    def __init__(self, loop):
        self.loop = loop
        self._debouncer = AsyncDebouncer(loop, 1.0, self.broadcast_gif_update)

    def on_modified(self, event):
        if event.src_path.endswith(".gif"):
            self._debouncer.call()

    async def broadcast_gif_update(self):
        payload = {
            "type": "gif_update",
            "control_path": "/static/gifs/control_run.gif" if os.path.exists("GIFS/control_run.gif") else None,
            "prev_path": "/static/gifs/previous_best.gif" if os.path.exists("GIFS/previous_best.gif") else None,
            "new_path": "/static/gifs/new_best.gif" if os.path.exists("GIFS/new_best.gif") else None,
        }
        _enqueue_telemetry_event(payload)

        render_meta_path = Path(__file__).parent / "GIFS" / "render_meta.json"
        if render_meta_path.exists():
            try:
                with open(render_meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                metrics_payload = {
                    "type": "metrics",
                    "sse": float(meta.get("new_sse", 999.0)),
                    "pcs": float(meta.get("pcs", 0.0)),
                    "ic": float(meta.get("ic", 0.0)),
                    "timestamp": str(meta.get("updated_at", "")),
                }
                _enqueue_telemetry_event(metrics_payload)

                status_payload = {
                    "type": "status",
                    "state": "running",
                    "details": f"Tier={str(meta.get('tier', 'SILVER')).upper()} SSE={float(meta.get('new_sse', 999.0)):.6f}",
                }
                _enqueue_telemetry_event(status_payload)

                artifact_path_raw = str(meta.get("artifact_path") or "").strip()
                if artifact_path_raw:
                    artifact_path = Path(artifact_path_raw).expanduser()
                    if not artifact_path.is_absolute():
                        artifact_path = (Path(__file__).parent / artifact_path).resolve()
                    history = _load_pde_history_from_h5(artifact_path)
                    if history:
                        _enqueue_telemetry_event(
                            {
                                "type": "pde_history",
                                "payload": history,
                            }
                        )
            except Exception as exc:
                logging.warning(f"WebSocket metrics/status broadcast failed: {exc}")

@app.on_event("startup")
async def startup_event():
    global observer_instance, telemetry_ticker_task, log_tailer_task_instance
    
    # Run our zombie killer!
    _kill_saved_pids()
    
    loop = asyncio.get_running_loop()
    event_handler = GifWatcherHandler(loop)
    observer_instance = Observer()
    gifs_dir = Path(__file__).parent / "GIFS"
    observer_instance.schedule(event_handler, str(gifs_dir), recursive=False)
    observer_instance.start()
    telemetry_ticker_task = asyncio.create_task(telemetry_ticker())
    log_tailer_task_instance = asyncio.create_task(log_tailer_task())
    _emit_terminal_debug("backend", "Startup complete: telemetry ticker and log tailer running.", "INFO")


@app.on_event("shutdown")
async def shutdown_event():
    global observer_instance, telemetry_ticker_task, log_tailer_task_instance
    from orchestrator.storage.artifact_gc import purge_old_artifacts

    # Cancel dangling stream tasks
    for task in state.get("stream_tasks", []):
        task.cancel()
    state["stream_tasks"] = []

    # 1. Kill OS Processes
    for proc in state["worker_processes"]:
        try:
            proc.kill()
        except Exception:
            pass
    state["worker_processes"] = []
    if state["orchestrator_process"] is not None:
        try:
            state["orchestrator_process"].kill()
        except Exception:
            pass
        state["orchestrator_process"] = None

    # --- [ALETHEIA V4.4] GC Sweep for Shutdown ---
    try:
        db_file, session_dir, prov_dir = resolve_active_session_paths()
        if session_dir and session_dir.exists():
            purged = purge_old_artifacts(session_dir, min_age_seconds=0)
            _emit_terminal_debug("backend", f"Shutdown: Purged {purged} orphaned artifacts.", "INFO")
    except Exception:
        pass

    # 2. Stop Background Tasks
    if observer_instance is not None:
        observer_instance.stop()
        observer_instance.join(timeout=5)
        observer_instance = None

    for task in [telemetry_ticker_task, log_tailer_task_instance]:
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    # 3. Write PIDs to disk to prevent zombies on next boot
    try:
        _save_pids()
    except NameError:
        pass # Fallback if _save_pids isn't defined

    _emit_terminal_debug("backend", "Shutdown complete: Processes and tasks terminated.", "WARN")