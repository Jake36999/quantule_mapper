import asyncio
import json
import os
import sqlite3
import subprocess as std_subprocess
import sys
import time
import types
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import app as app_module
import worker_daemon
from config_utils import clear_active_run_pointer, read_active_run_pointer, resolve_generation_dir
from orchestrator.job_manifest import JobManifest
from orchestrator.result_processor import ResultProcessor
from orchestrator.scheduling.job_dispatcher import JobDispatcher
from orchestrator.scheduling.queue_manager import QueueManager


REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"
MOCK_WORKER = FIXTURE_DIR / "mock_worker_cupy.py"
MOCK_VALIDATOR = FIXTURE_DIR / "mock_validation_pipeline.py"


@pytest.fixture
def app_client(monkeypatch: pytest.MonkeyPatch, sandbox: Path):
    monkeypatch.setattr(app_module, "BACKLOG_QUEUE_FILE", str(sandbox / "backlog_queue.json"))
    monkeypatch.setattr(app_module, "BACKLOG_RESULT_FILE", str(sandbox / "result_queue.json"))
    monkeypatch.setattr(app_module, "DB_FILE", str(sandbox / "simulation_ledger.db"))
    monkeypatch.setattr(app_module, "DATA_DIR", str(sandbox / "simulation_data"))
    with TestClient(app_module.app) as client:
        yield client


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _session_paths() -> tuple[Path, Path]:
    pointer = read_active_run_pointer()
    session_dir = Path(pointer["session_dir"])
    gen0_dir = resolve_generation_dir(session_dir, 0)
    return session_dir, gen0_dir


def _seed_ledger_rows(db_path: Path, count: int = 3) -> None:
    conn = sqlite3.connect(str(db_path), timeout=5.0, check_same_thread=False)
    cursor = conn.cursor()
    for index in range(count):
      config_hash = f"cfg{index:04d}"
      cursor.execute(
          "INSERT OR REPLACE INTO runs (config_hash, generation, status, fitness, origin, timestamp) VALUES (?, ?, ?, ?, ?, datetime('now', ?))",
          (config_hash, index, "SUCCESS", 0.1 + index, "TEST", f"-{index} minutes"),
      )
      cursor.execute(
          "INSERT OR REPLACE INTO metrics (config_hash, log_prime_sse, bragg_peaks_detected, pcs, collapse_event_count) VALUES (?, ?, ?, ?, ?)",
          (config_hash, 0.5 + index, 2 + index, 0.9 - (index * 0.1), index),
      )
      cursor.execute(
          "INSERT OR REPLACE INTO parameters (config_hash, param_D, param_eta, param_rho_vac) VALUES (?, ?, ?, ?)",
          (config_hash, 1.0 + index, 2.0 + index, 3.0 + index),
      )
    conn.commit()
    conn.close()


def _patch_worker_daemon_execution_contract(
    monkeypatch: pytest.MonkeyPatch,
    sandbox: Path,
    worker_sleep: float = 0.0,
    validator_sleep: float = 0.0,
):
    trace_file = sandbox / "trace_events.jsonl"

    monkeypatch.setattr(worker_daemon, "QUEUE_FILE", str(sandbox / "backlog_queue.json"))
    monkeypatch.setattr(worker_daemon, "RESULT_FILE", str(sandbox / "result_queue.json"))
    monkeypatch.setattr(worker_daemon, "DB_FILE", str(sandbox / "simulation_ledger.db"))

    real_run = std_subprocess.run

    def _proxy_run(cmd, *args, **kwargs):
        cmd_list = list(cmd)
        if len(cmd_list) >= 2 and cmd_list[1] == "worker_cupy.py":
            cmd_list[1] = str(MOCK_WORKER)
        elif len(cmd_list) >= 2 and cmd_list[1] == "validation_pipeline.py":
            cmd_list[1] = str(MOCK_VALIDATOR)

        env = dict(os.environ)
        env.update(kwargs.pop("env", {}))
        env["ASTE_TRACE_FILE"] = str(trace_file)
        env["ASTE_MOCK_SLEEP_SECONDS"] = str(worker_sleep)
        env["ASTE_MOCK_VALIDATION_SLEEP_SECONDS"] = str(validator_sleep)

        kwargs["env"] = env
        kwargs["cwd"] = str(sandbox)
        return real_run(cmd_list, *args, **kwargs)

    monkeypatch.setattr(worker_daemon.subprocess, "run", _proxy_run)
    return trace_file


def test_control_start_creates_manifest_in_queue(app_client: TestClient, sandbox: Path):
    stage = app_client.post(
        "/api/control/stage",
        json={
            "hunt_name": "E2E_SESSION",
            "generations": 5,
            "batch_size": 2,
            "population_size": 32,
            "seeds_per_candidate": 2,
            "n_grid": 16,
            "t_steps": 100,
            "dt": 0.01,
        },
    )
    assert stage.status_code == 200
    assert stage.json()["status"] == "success"

    response = app_client.post(
        "/api/control/start",
        json={"staged_path": stage.json()["staged_path"], "hunt_name": "E2E_SESSION", "origin": "UI_CONTROL"},
    )
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "success"
    assert payload.get("job_id")

    queue = _read_json(sandbox / "backlog_queue.json")
    assert len(queue) == 1
    job = queue[0]
    assert job["job_id"] == payload["job_id"]
    assert job["origin"] == "UI_CONTROL"
    assert job["params"]["generations"] == 5
    assert job["params"]["batch_size"] == 2


def test_list_data_files_discovery(app_client: TestClient, sandbox: Path):
    session_dir, gen0_dir = _session_paths()
    gen0_dir.mkdir(parents=True, exist_ok=True)
    (gen0_dir / "a.h5").write_bytes(b"h5")
    (gen0_dir / "b.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    (session_dir / "ignore.txt").write_text("no", encoding="utf-8")

    response = app_client.get("/api/data/files")
    assert response.status_code == 200
    payload = response.json()
    assert "files" in payload
    files = payload["files"]
    assert isinstance(files, list)
    for item in files:
        assert item["type"] in {".h5", ".csv"}


def test_download_data_file_safety(app_client: TestClient, sandbox: Path):
    _session_dir, gen0_dir = _session_paths()
    gen0_dir.mkdir(parents=True, exist_ok=True)
    target = gen0_dir / "safe.h5"
    target.write_bytes(b"safe")

    ok = app_client.get("/api/data/download/gen_0/safe.h5")
    assert ok.status_code == 200
    assert ok.content == b"safe"

    bad = app_client.get("/api/data/download/not_allowed.txt")
    assert bad.status_code == 200
    assert bad.json().get("error")


def test_plugin_execute_launches_allowlisted_visualizer(app_client: TestClient, sandbox: Path):
    _session_dir, gen0_dir = _session_paths()
    artifact = gen0_dir / "rho_history_test.h5"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"h5")

    proc = types.SimpleNamespace(pid=4242)
    with patch("app.subprocess.Popen", return_value=proc) as popen_mock:
        response = app_client.post(
            "/api/plugins/execute",
            json={
                "plugin_id": "gif_pipeline_manager",
                "artifact_path": str(artifact),
                "sse": 0.22,
                "tier": "golden",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["mocked"] is False
    assert payload["accepted"] is True
    assert payload["plugin_id"] == "gif_pipeline_manager"
    assert payload["pid"] == 4242
    assert popen_mock.called


def test_plugin_execute_rejects_unknown_plugin_id(app_client: TestClient, sandbox: Path):
    _session_dir, gen0_dir = _session_paths()
    artifact = gen0_dir / "rho_history_test.h5"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"h5")

    response = app_client.post(
        "/api/plugins/execute",
        json={
            "plugin_id": "totally_unknown_plugin",
            "artifact_path": str(artifact),
        },
    )
    assert response.status_code == 400
    payload = response.json()
    assert payload["status"] == "error"
    assert "allowlisted" in payload["message"]


def test_list_ledger_runs_includes_active_run(app_client: TestClient):
    pointer = read_active_run_pointer()

    response = app_client.get("/api/data/ledger/runs")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "success"
    assert isinstance(payload["runs"], list)
    assert any(run["run_id"] == pointer["run_id"] and run.get("is_active") for run in payload["runs"])


def test_get_ledger_rows_paginates_active_run(app_client: TestClient):
    pointer = read_active_run_pointer()
    _seed_ledger_rows(Path(pointer["db_file"]), count=4)

    response = app_client.get(f"/api/data/ledger/{pointer['run_id']}?limit=2&offset=1")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "success"
    assert payload["run_id"] == pointer["run_id"]
    assert payload["pagination"]["limit"] == 2
    assert payload["pagination"]["offset"] == 1
    assert payload["pagination"]["returned"] == 2
    assert payload["pagination"]["total"] == 4
    assert len(payload["rows"]) == 2


def test_get_ledger_rows_clamps_pagination(app_client: TestClient):
    pointer = read_active_run_pointer()
    _seed_ledger_rows(Path(pointer["db_file"]), count=2)

    response = app_client.get(f"/api/data/ledger/{pointer['run_id']}?limit=5000&offset=-10")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "success"
    assert payload["pagination"]["limit"] == 500
    assert payload["pagination"]["offset"] == 0
    assert payload["pagination"]["returned"] == 2


def test_get_ledger_rows_unknown_run_id_returns_404(app_client: TestClient):
    response = app_client.get("/api/data/ledger/deadbeef?limit=50&offset=0")
    assert response.status_code == 404
    payload = response.json()
    assert payload["status"] == "error"
    assert "run_id not found" in payload["message"]


def test_system_telemetry_snapshot_includes_worker_health_and_claims(app_client: TestClient, sandbox: Path):
    queue_path = sandbox / "backlog_queue.json"
    result_path = sandbox / "result_queue.json"
    queue_manager = QueueManager(str(queue_path), str(result_path))

    queue_manager.push_job(
        JobManifest.from_params(
            params={"param_D": 0.15, "generation": 0, "origin": "TEST"},
            generation=0,
            seed=0,
            origin="TEST",
        ).to_json()
    )
    queue_manager.push_job(
        JobManifest.from_params(
            params={"param_D": 0.25, "generation": 0, "origin": "TEST"},
            generation=0,
            seed=1,
            origin="TEST",
        ).to_json()
    )

    claim = queue_manager.claim_job("worker_active")
    assert claim is not None
    queue_manager.complete_job(claim["token"])

    queue_manager.set_worker_heartbeat("worker_active", time.time())
    queue_manager.set_worker_heartbeat("worker_stale", time.time() - 300.0)

    response = app_client.get("/api/system/telemetry?ttl_seconds=90")
    assert response.status_code == 200
    payload = response.json()

    assert payload["status"] == "success"
    assert payload["queue_depth"] == 1
    assert payload["total_claims_processed"] >= 1
    assert "worker_active" in payload["active_workers"]
    assert "worker_stale" in payload["stale_workers"]
    assert any(worker["state"] == "active" for worker in payload["workers"])
    assert any(worker["state"] == "stale" for worker in payload["workers"])


def test_system_telemetry_rejects_invalid_ttl(app_client: TestClient):
    response = app_client.get("/api/system/telemetry?ttl_seconds=-5")
    assert response.status_code == 400
    payload = response.json()
    assert payload["status"] == "error"


def test_data_endpoints_fail_fast_without_active_pointer(app_client: TestClient):
    clear_active_run_pointer()

    files_resp = app_client.get("/api/data/files")
    assert files_resp.status_code == 409
    assert files_resp.json().get("status") == "error"

    download_resp = app_client.get("/api/data/download/gen_0/missing.h5")
    assert download_resp.status_code == 409
    assert download_resp.json().get("status") == "error"


def test_worker_daemon_claims_and_executes_job(monkeypatch: pytest.MonkeyPatch, sandbox: Path):
    trace_file = _patch_worker_daemon_execution_contract(monkeypatch, sandbox)
    monkeypatch.setattr(worker_daemon, "SIM_TIMEOUT", 5.0)
    monkeypatch.setattr(worker_daemon, "VAL_TIMEOUT", 5.0)

    manifest = JobManifest.from_params(
        params={"param_D": 0.1, "generation": 0, "origin": "TEST", "mock_sse": 0.42},
        generation=0,
        seed=0,
        origin="TEST",
    )
    QueueManager(str(sandbox / "backlog_queue.json"), str(sandbox / "result_queue.json")).push_job(
        manifest.to_json()
    )

    worker_daemon.main()

    queue = _read_json(sandbox / "backlog_queue.json")
    claims = _read_json(sandbox / "backlog_queue.json.claims.json")
    results = _read_json(sandbox / "result_queue.json")

    assert queue == []
    assert claims == {}
    assert len(results) == 1
    assert results[0]["status"] == "SUCCESS"
    assert results[0]["config_hash"] == manifest.config_hash

    temp_manifest_files = list(sandbox.glob("temp_manifest_*.json"))
    temp_param_files = list(sandbox.glob("temp_params_*.json"))
    assert temp_manifest_files == []
    assert temp_param_files == []

    session_dir, gen0_dir = _session_paths()
    output_h5 = gen0_dir / f"rho_history_{manifest.config_hash}.h5"
    provenance = session_dir / "provenance_reports" / f"provenance_{manifest.config_hash}.json"
    assert output_h5.exists()
    assert provenance.exists()

    trace_lines = trace_file.read_text(encoding="utf-8").strip().splitlines()
    stages = {json.loads(line)["stage"] for line in trace_lines if line.strip()}
    assert "mock_worker" in stages
    assert "mock_validation" in stages


def test_worker_daemon_timeout_handling(monkeypatch: pytest.MonkeyPatch, sandbox: Path):
    _patch_worker_daemon_execution_contract(monkeypatch, sandbox, worker_sleep=0.2)
    monkeypatch.setattr(worker_daemon, "SIM_TIMEOUT", 0.01)
    monkeypatch.setattr(worker_daemon, "VAL_TIMEOUT", 2.0)

    manifest = JobManifest.from_params(
        params={"param_D": 0.2, "generation": 0, "origin": "TEST_TIMEOUT"},
        generation=0,
        seed=0,
        origin="TEST_TIMEOUT",
    )
    QueueManager(str(sandbox / "backlog_queue.json"), str(sandbox / "result_queue.json")).push_job(
        manifest.to_json()
    )

    worker_daemon.main()
    results = _read_json(sandbox / "result_queue.json")
    assert len(results) == 1
    assert results[0]["status"] == "FAIL"
    assert results[0]["reason"] == "timeout"


def test_queue_corruption_recovery(sandbox: Path):
    backlog_file = sandbox / "backlog_queue.json"
    backlog_file.write_text("{broken", encoding="utf-8")

    dispatcher = JobDispatcher(
        {
            "seeds_per_candidate": 1,
            "backlog_seed_json_decode_retries": 1,
            "backlog_seed_json_decode_backoff_seconds": 0.001,
            "origin": "TEST",
        }
    )
    seeds = dispatcher._pop_backlog_seed_configs(1)
    assert seeds == []


def test_backlog_path_resolution(app_client: TestClient, sandbox: Path):
    outside_queue = (sandbox.parent / "outside_backlog_queue.json").resolve()
    outside_queue.write_text("[]", encoding="utf-8")

    response = app_client.post(
        "/api/control/stage",
        json={
            "hunt_name": "PATH_GUARD_TEST",
            "generations": 1,
            "batch_size": 1,
            "population_size": 4,
            "seeds_per_candidate": 1,
            "n_grid": 32,
            "t_steps": 10,
            "dt": 0.01,
            "mode": "backlog",
            "backlog_source": str(outside_queue),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "error"
    assert (
        "backlog_source must resolve inside repository root" in payload["message"]
        or "backlog_source not found under repo roots" in payload["message"]
    )


def test_dropdown_error_propagation(
    app_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    pointer = read_active_run_pointer()

    def _boom(*_args, **_kwargs):
        raise RuntimeError("Backend unreachable [503]")

    monkeypatch.setattr(app_module, "_fetch_ledger_rows", _boom)

    response = app_client.get(f"/api/data/ledger/{pointer['run_id']}?limit=50&offset=0")
    assert response.status_code == 500
    payload = response.json()
    assert payload["status"] == "error"
    assert "Backend unreachable [503]" in payload["message"]


def test_ws_missing_log_broadcast(
    monkeypatch: pytest.MonkeyPatch,
    sandbox: Path,
):
    missing_path = sandbox / "runtime_logs" / "worker_gpu0.log"
    monkeypatch.setattr(
        app_module,
        "DEBUG_LOG_SOURCES",
        [{"id": "worker", "name": "Worker GPU0", "path": missing_path}],
    )
    app_module.missing_log_warned_sources.clear()

    while not app_module.telemetry_queue.empty():
        app_module.telemetry_queue.get_nowait()

    asyncio.run(app_module._tail_logs_once({}))

    events = []
    while not app_module.telemetry_queue.empty():
        events.append(app_module.telemetry_queue.get_nowait())

    assert any(
        evt.get("type") == "terminal_log"
        and "Awaiting Worker GPU0 creation" in str(evt.get("line", ""))
        for evt in events
    )


def test_websocket_connects_and_receives_broadcasts(app_client: TestClient):
    with app_client.websocket_connect("/ws/telemetry") as ws:
        ws.send_text(json.dumps({"event": "START_HUNT"}))
        first = ws.receive_json()
        second = ws.receive_json()

        assert first["type"] == "log"
        assert second["type"] == "status"

        asyncio.run(app_module._broadcast_payload({"type": "metrics", "sse": 1.23, "pcs": 0.6, "ic": 0.4}))
        third = ws.receive_json()
        assert third["type"] == "metrics"
        assert float(third["sse"]) == 1.23


def test_websocket_metrics_history_slice_contract():
    history = []
    for idx in range(150):
        history = (history + [{"sse": idx}])[-100:]
    assert len(history) == 100
    assert history[0]["sse"] == 50
    assert history[-1]["sse"] == 149


def test_orchestrator_processes_result_queue(sandbox: Path):
    if "psutil" not in sys.modules:
        sys.modules["psutil"] = types.SimpleNamespace(
            disk_usage=lambda *_: types.SimpleNamespace(used=0)
        )
    from orchestrator.orchestrator_engine import OrchestratorEngine

    session_dir, gen0_dir = _session_paths()
    config_hash = "abc123" * 10 + "abcd"
    artifact = gen0_dir / f"rho_history_{config_hash}.h5"
    gen0_dir.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"artifact")

    provenance_path = session_dir / "provenance_reports" / f"provenance_{config_hash}.json"
    _write_json(
        provenance_path,
        {
            "spectral_fidelity": {"log_prime_sse": 0.8},
            "aletheia_metrics": {"pcs": 0.4, "ic": 0.2},
        },
    )

    _write_json(
        sandbox / "result_queue.json",
        [
            {
                "job_id": "job1",
                "generation": 0,
                "config_hash": config_hash,
                "artifact_url": str(artifact),
                "status": "SUCCESS",
                "provenance_path": str(provenance_path),
                "config": {"origin": "TEST", "generation": 0},
            }
        ],
    )

    engine = OrchestratorEngine(
        {
            "generations": 1,
            "population_size": 1,
            "seeds_per_candidate": 1,
            "db_path": str(session_dir / "simulation_ledger.db"),
            "ledger_db_path": str(session_dir / "simulation_ledger.db"),
            "data_dir": str(session_dir),
            "provenance_dir": str(session_dir / "provenance_reports"),
            "archive_dir": str(sandbox / "archive"),
            "audit_log": str(sandbox / "runtime_logs" / "run_lifecycle_audit.jsonl"),
            "poll_interval": 0.01,
        }
    )

    def _fake_process_result(result_data):
        result_data["_ingested_log_prime_sse"] = 0.8
        return True

    engine.result_processor.process_result = _fake_process_result

    engine._process_pending_results()

    assert _read_json(sandbox / "result_queue.json") == []
    assert any(token.startswith("gen_0_") for token in engine.completed_jobs)
    assert engine.global_best_hash == config_hash
    assert pytest.approx(engine.global_best_sse, rel=0, abs=1e-9) == 0.8


def test_result_validation_contract(sandbox: Path):
    session_dir, gen0_dir = _session_paths()
    config_hash = "def456" * 10 + "def4"
    gen0_dir.mkdir(parents=True, exist_ok=True)
    artifact = gen0_dir / f"rho_history_{config_hash}.h5"
    artifact.write_bytes(b"artifact")

    provenance_path = session_dir / "provenance_reports" / f"provenance_{config_hash}.json"
    _write_json(
        provenance_path,
        {
            "spectral_fidelity": {"log_prime_sse": 0.314159},
            "aletheia_metrics": {"pcs": 0.5, "ic": 0.25},
        },
    )

    rp = ResultProcessor(
        {
            "db_path": str(session_dir / "simulation_ledger.db"),
            "data_dir": str(session_dir),
            "provenance_dir": str(session_dir / "provenance_reports"),
        }
    )
    validated = rp._validate_result(
        {
            "job_id": "job2",
            "generation": 0,
            "config_hash": config_hash,
            "artifact_url": str(artifact),
            "provenance_path": str(provenance_path),
        }
    )

    assert validated is not None
    assert pytest.approx(float(validated["log_prime_sse"]), rel=0, abs=1e-9) == 0.314159


def test_e2e_ui_to_worker_to_dashboard(
    monkeypatch: pytest.MonkeyPatch,
    app_client: TestClient,
    sandbox: Path,
):
    trace_file = _patch_worker_daemon_execution_contract(monkeypatch, sandbox)
    monkeypatch.setattr(worker_daemon, "SIM_TIMEOUT", 5.0)
    monkeypatch.setattr(worker_daemon, "VAL_TIMEOUT", 5.0)

    stage = app_client.post(
        "/api/control/stage",
        json={
            "hunt_name": "UI_E2E_SESSION",
            "generations": 1,
            "batch_size": 1,
            "population_size": 8,
            "seeds_per_candidate": 1,
            "n_grid": 8,
            "t_steps": 10,
            "dt": 0.01,
        },
    )
    assert stage.status_code == 200
    assert stage.json()["status"] == "success"

    start = app_client.post(
        "/api/control/start",
        json={"staged_path": stage.json()["staged_path"], "hunt_name": "UI_E2E_SESSION", "origin": "UI_CONTROL"},
    )
    assert start.status_code == 200
    assert start.json()["status"] == "success"

    queued = _read_json(sandbox / "backlog_queue.json")
    assert len(queued) == 1

    worker_daemon.main()

    results = _read_json(sandbox / "result_queue.json")
    assert len(results) == 1
    assert results[0]["status"] == "SUCCESS"

    files_resp = app_client.get("/api/data/files")
    assert files_resp.status_code == 200
    files = files_resp.json()["files"]
    assert files

    h5_name = files[0]["name"]
    download = app_client.get(f"/api/data/download/{h5_name}")
    assert download.status_code == 200
    assert download.content

    with app_client.websocket_connect("/ws/telemetry") as ws:
        ws.send_text(json.dumps({"event": "START_HUNT"}))
        assert ws.receive_json()["type"] == "log"

    trace_lines = trace_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(trace_lines) >= 2
