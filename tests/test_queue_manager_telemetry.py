import json
import time
from pathlib import Path

from orchestrator.job_manifest import JobManifest
from orchestrator.scheduling.queue_manager import QueueManager


def test_complete_job_increments_total_claims_processed(tmp_path: Path):
    queue_path = tmp_path / "queue.json"
    result_path = tmp_path / "result.json"
    qm = QueueManager(queue_file=str(queue_path), result_file=str(result_path))

    manifest = JobManifest.from_params(params={"param_D": 0.1, "generation": 0, "origin": "TEST"}, generation=0, seed=0)
    qm.push_job(manifest.to_json())
    claimed = qm.claim_job("worker-1")
    assert claimed is not None

    before = qm.get_telemetry_counters()["total_claims_processed"]
    assert qm.complete_job(claimed["token"]) is True
    after = qm.get_telemetry_counters()["total_claims_processed"]
    assert after == before + 1

    # Unknown claim token should not mutate counters.
    assert qm.complete_job("missing-token") is False
    assert qm.get_telemetry_counters()["total_claims_processed"] == after


def test_non_mutating_worker_lists_do_not_requeue_or_clear(tmp_path: Path):
    queue_path = tmp_path / "queue.json"
    result_path = tmp_path / "result.json"
    qm = QueueManager(queue_file=str(queue_path), result_file=str(result_path))

    manifest_a = JobManifest.from_params(params={"param_D": 0.2, "generation": 0, "origin": "TEST"}, generation=0, seed=0)
    manifest_b = JobManifest.from_params(params={"param_D": 0.3, "generation": 0, "origin": "TEST"}, generation=0, seed=1)
    qm.push_job(manifest_a.to_json())
    qm.push_job(manifest_b.to_json())
    claimed = qm.claim_job("worker-stale")
    assert claimed is not None

    qm.set_worker_heartbeat("worker-active", time.time())
    qm.set_worker_heartbeat("worker-stale", time.time() - 240.0)

    stale_workers = qm.list_stale_workers(90.0)
    active_workers = qm.list_active_workers(90.0)
    claims_before = qm._read_claims()
    queue_before = qm.peek_all()

    assert "worker-stale" in stale_workers
    assert "worker-active" in active_workers
    assert claimed["token"] in claims_before
    assert len(queue_before) == 1

    # Confirm non-mutating semantics: stale discovery does not clear worker or claims.
    claims_after = qm._read_claims()
    workers_after = qm.get_worker_heartbeats()
    queue_after = qm.peek_all()

    assert claims_after == claims_before
    assert workers_after.get("worker-stale") is not None
    assert queue_after == queue_before
