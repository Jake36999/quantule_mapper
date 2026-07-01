#!/usr/bin/env python3
"""Autonomous Backlog Worker Daemon - Bypasses broken Orchestrator/Redis queues"""

import json
import logging
import os
import subprocess
import sys
import uuid
from pathlib import Path

from orchestrator.diagnostics.runtime_audit import log_lifecycle_event
from orchestrator.contracts import parse_job_manifest_payload, parse_job_result_payload
from orchestrator.scheduling.queue_manager import QueueManager
from config_utils import (
    ActiveRunPointerError,
    resolve_active_session_paths,
    resolve_generation_dir,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

QUEUE_FILE = "backlog_queue.json"
RESULT_FILE = "result_queue.json"
# Compatibility constant retained for legacy tests/injection hooks.
DB_FILE = "simulation_ledger.db"
SIM_TIMEOUT = float(os.environ.get("ASTE_SIM_TIMEOUT", "7200"))
VAL_TIMEOUT = float(os.environ.get("ASTE_VAL_TIMEOUT", "1800"))


def _build_energy_sparkline(output_h5_path: str) -> list[float] | None:
    """Extract precomputed worker energy sparkline from HDF5 telemetry."""
    try:
        import h5py  # type: ignore

        with h5py.File(output_h5_path, "r") as f:
            if "telemetry" not in f or "energy_sparkline" not in f["telemetry"]:
                return None
            sparkline = f["telemetry"]["energy_sparkline"][:]

        return [float(v) for v in sparkline.tolist()]
    except Exception:
        return None


def record_fail_result(queue_manager: QueueManager, job_id, job_params, config_hash, output_h5_path, reason=None):
    """Enforce fail accounting sink: validated result queue payload."""
    payload = {
        "job_id": job_id,
        "generation": int(job_params.get("generation", -1)),
        "config_hash": config_hash,
        "artifact_url": output_h5_path,
        "status": "FAIL",
        "origin": job_params.get("origin"),
        "staged_path": job_params.get("staged_path"),
        "staged_at": job_params.get("staged_at"),
        "staged_config_hash": job_params.get("staged_config_hash"),
    }
    if reason:
        payload["error"] = reason
        payload["reason"] = reason
    payload["config"] = job_params
    model = parse_job_result_payload(payload)
    queue_manager.push_result(model.model_dump_json())


def main():
    logging.info("[WorkerDaemon] Autonomous Backlog Engine Started.")

    repo_root = os.path.dirname(os.path.abspath(__file__))
    worker_id = os.environ.get("ASTE_WORKER_ID", f"worker_{os.getpid()}")
    try:
        bootstrap_paths = resolve_active_session_paths(require_exists=True)
    except ActiveRunPointerError as exc:
        logging.error(f"[FATAL] Active run pointer unavailable: {exc}")
        sys.exit(1)

    # ASTE Priority-1: Enforce orchestrator-first schema ownership
    from orchestrator.schema_utils import ensure_ledger_ready
    if not ensure_ledger_ready(bootstrap_paths["db_file"], raise_on_fail=False):
        logging.error(
            "[FATAL] Ledger schema not initialized. "
            "Orchestrator must initialize schema before worker spawn. "
            f"DB path: {bootstrap_paths['db_file']}"
        )
        sys.exit(1)

    queue_manager = QueueManager(
        queue_file=QUEUE_FILE,
        result_file=RESULT_FILE,
        ledger_db_path=str(bootstrap_paths["db_file"]),
    )

    jobs_completed = 0

    while True:
        queue_manager.set_worker_heartbeat(worker_id)
        claimed_job = queue_manager.claim_job(worker_id)
        if not claimed_job:
            logging.info("[WorkerDaemon] Queue empty. All backlog jobs complete!")
            break

        claim_token = claimed_job["token"]
        claim_released = False
        job_params = {}
        job_id = str(uuid.uuid4())[:16]
        generation = -1
        config_hash = "UNKNOWN"
        temp_manifest_path = ""
        temp_params_path = ""
        output_h5_path = ""
        provenance_path = ""
        staged_path = None
        staged_at = None
        staged_config_hash = None
        job_origin = "NATURAL"

        try:
            manifest_model = parse_job_manifest_payload(claimed_job["payload"])
            job_params = dict(manifest_model.params)
            job_id = str(manifest_model.job_id or str(uuid.uuid4())[:16])
            generation = int(manifest_model.generation)
            config_hash = str(manifest_model.config_hash)
            job_origin = str(manifest_model.origin)
            staged_path = manifest_model.staged_path
            staged_at = manifest_model.staged_at
            staged_config_hash = manifest_model.staged_config_hash

            validation_params = dict(job_params)
            validation_params.setdefault("config_hash", config_hash)
            if staged_path:
                validation_params.setdefault("staged_path", staged_path)
            if staged_at:
                validation_params.setdefault("staged_at", staged_at)
            if staged_config_hash:
                validation_params.setdefault("staged_config_hash", staged_config_hash)

            temp_manifest_path = f"temp_manifest_{config_hash}_{job_id}.json"
            temp_params_path = f"temp_params_{config_hash}_{job_id}.json"

            active_paths = resolve_active_session_paths(require_exists=True)
            generation_value = int(job_params.get("generation", generation if generation >= 0 else 0))
            generation_dir = resolve_generation_dir(active_paths["session_dir"], generation_value)
            generation_dir.mkdir(parents=True, exist_ok=True)
            provenance_dir = Path(active_paths["provenance_dir"]).resolve()
            provenance_dir.mkdir(parents=True, exist_ok=True)

            output_h5_path = str((generation_dir / f"rho_history_{config_hash}.h5").resolve())
            provenance_path = str((provenance_dir / f"provenance_{config_hash}.json").resolve())

            with open(temp_manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest_model.model_dump(), handle, indent=2)
            with open(temp_params_path, "w", encoding="utf-8") as handle:
                json.dump(validation_params, handle, indent=2)

            logging.info(f"--- Starting Job {jobs_completed+1} | Hash: {config_hash[:12]} ---")

            log_lifecycle_event(
                stage="worker_start",
                config_hash=config_hash,
                generation=-1,
                job_id=job_id,
                details={"worker_id": "autonomous_daemon"}
            )
        except Exception as exc:
            logging.exception(f"❌ Pre-execution setup failed for claim {claim_token}: {exc}")
            record_fail_result(
                queue_manager,
                job_id,
                job_params,
                config_hash,
                output_h5_path,
                reason="pre_execution_setup_failed",
            )
            queue_manager.complete_job(claim_token)
            claim_released = True
            continue

        try:

            # 1. Run ETDRK4 Integration
            subprocess.run(
                [sys.executable, "-u", "worker_cupy.py", "--manifest", temp_manifest_path, "--output", output_h5_path],
                cwd=repo_root,
                check=True,
                timeout=SIM_TIMEOUT,
            )
            queue_manager.set_worker_heartbeat(worker_id)
            queue_manager.set_worker_heartbeat(worker_id)

            quantule_csv_path = Path(provenance_path).parent / f"{config_hash}_quantule_events.csv"
            if quantule_csv_path.exists():
                generation_csv_path = generation_dir / quantule_csv_path.name
                try:
                    quantule_csv_path.replace(generation_csv_path)
                except OSError:
                    pass

            # 2. Emit typed result payload for Orchestrator Batch Validation
            result_model = parse_job_result_payload(
                {
                    "job_id": job_id,
                    "generation": int(job_params.get("generation", -1)),
                    "config_hash": config_hash,
                    "artifact_url": output_h5_path,
                    "status": "PENDING_VALIDATION",
                    "provenance_path": "",
                    "origin": job_origin,
                    "staged_path": staged_path,
                    "staged_at": staged_at,
                    "staged_config_hash": staged_config_hash,
                    "config": job_params,
                }
            )
            queue_manager.push_result(result_model.model_dump_json())
            logging.info(f"✅ Handoff complete: Queued {config_hash[:12]} as PENDING_VALIDATION.")
            jobs_completed += 1

        except subprocess.CalledProcessError:
            logging.error(f"❌ Job {config_hash[:12]} failed during execution. Recording FAIL to DB.")
            record_fail_result(
                queue_manager,
                job_id,
                job_params,
                config_hash,
                output_h5_path,
                reason="subprocess_failed",
            )
        except subprocess.TimeoutExpired as exc:
            timed_out_stage = "validation" if "validation_pipeline.py" in str(exc.cmd) else "simulation"
            logging.error(
                f"⏱️ Job {config_hash[:12]} timed out during {timed_out_stage}. "
                f"Timeout={exc.timeout}s. Recording FAIL to DB."
            )

            timeout_model = parse_job_result_payload(
                {
                    "job_id": job_id,
                    "generation": int(job_params.get("generation", -1)),
                    "config_hash": config_hash,
                    "artifact_url": output_h5_path,
                    "status": "FAIL",
                    "reason": "timeout",
                    "error": "timeout",
                    "origin": job_origin,
                    "staged_path": staged_path,
                    "staged_at": staged_at,
                    "staged_config_hash": staged_config_hash,
                    "config": job_params,
                }
            )
            queue_manager.push_result(timeout_model.model_dump_json())
            queue_manager.complete_job(claim_token)
            claim_released = True
        except Exception as exc:
            logging.exception(f"❌ Job {config_hash[:12]} experienced an unhandled error: {exc}")
            record_fail_result(
                queue_manager,
                job_id,
                job_params,
                config_hash,
                output_h5_path,
                reason="unhandled_exception",
            )
        finally:
            if not claim_released:
                queue_manager.complete_job(claim_token)
            if temp_manifest_path and os.path.exists(temp_manifest_path):
                os.remove(temp_manifest_path)
            if temp_params_path and os.path.exists(temp_params_path):
                os.remove(temp_params_path)

if __name__ == "__main__":
    main()