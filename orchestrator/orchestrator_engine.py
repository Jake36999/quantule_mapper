#!/usr/bin/env python3

"""
orchestrator_engine.py
Core orchestration engine with generation accounting, completion gating,
disk-pressure backpressure, and GC integration.
"""

import os
import json
import logging
import time
import traceback
import sys
import subprocess
import types
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
from config_utils import ActiveRunPointerError, resolve_active_session_paths
from aste_hunter import Hunter

try:
    import psutil  # type: ignore
except Exception:
    psutil = types.SimpleNamespace(disk_usage=lambda *_args, **_kwargs: types.SimpleNamespace(used=0))

from .scheduling.job_dispatcher import JobDispatcher
from .result_processor import ResultProcessor
from .storage.artifact_bleed_manager import ArtifactBleedManager
from .scheduling.queue_manager import QueueManager
from .storage.artifact_gc import purge_old_artifacts
from .diagnostics.runtime_audit import log_lifecycle_event
from .schema_utils import initialize_ledger_schema, ensure_ledger_ready
from orchestrator.contracts import SOLVER_CONTRACT_VERSION


class OrchestratorEngine:
    """Main orchestration engine coordinating the simulation lifecycle."""

    global_best_hash: Optional[str] = None
    global_best_sse: float = 999.0

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize orchestration runtime state.

        Ownership Contract (Phase B Freeze):
        - Startup checkpoint restoration ownership lives here via `_load_checkpoint()`.
        - `current_generation`, `global_best_hash`, and `global_best_sse` loaded here
          are the authoritative in-memory state for loop control and GC protection.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.ledger_db_path = str(config.get('ledger_db_path', 'simulation_ledger.db'))
        self.state_file = config.get('orchestrator_state_file', 'orchestrator_state.json')
        self.stop_signal_file = str(config.get('stop_signal_file', 'stop_after_gen.txt'))
        self._refresh_active_paths(raise_on_missing=False)

        # Initialize components
        self.job_dispatcher = JobDispatcher(config)
        self.result_processor = ResultProcessor(config)
        self.bleed_manager = ArtifactBleedManager(config)
        self.queue_manager = QueueManager(
            queue_file="backlog_queue.json",
            result_file="result_queue.json",
            ledger_db_path=self.ledger_db_path,
        )
        self.hunter = Hunter(db_file=self.ledger_db_path)

        # Generation state
        self.current_generation = 0
        self.generations_total = config.get('generations', 100)
        self.jobs_per_generation = config.get('population_size', 10) * config.get('seeds_per_candidate', 1)
        self.completed_jobs: set[str] = set()
        self._result_status_by_key: Dict[str, str] = {}
        self.global_best_hash: Optional[str] = None
        self.global_best_sse: float = 999.0

        # Disk pressure settings
        self.max_disk_gb = config.get('max_disk_gb', 100)
        self.trigger_bleed_threshold_gb = config.get('trigger_bleed_threshold_gb', 80)

        # Polling settings
        self.poll_interval = config.get('poll_interval', 5)
        self.worker_heartbeat_ttl_seconds = float(config.get('worker_heartbeat_ttl_seconds', 90.0))
        self.iteration_error_backoff_max_seconds = float(config.get('iteration_error_backoff_max_seconds', 60.0))
        self.artifact_read_timeout_seconds = float(config.get('artifact_read_timeout_seconds', 120.0))

        # Threading
        self.running = False
        self.engine_thread: Optional[threading.Thread] = None

        self._initialize_ledger_schema()
        self._load_checkpoint()

    def start(self):
        """Start the orchestration engine."""
        if self.running:
            self.logger.warning("Engine already running")
            return

        if self.stop_signal_file and os.path.exists(self.stop_signal_file):
            try:
                os.remove(self.stop_signal_file)
                self.logger.info(f"Cleared stale stop signal file: {self.stop_signal_file}")
            except OSError as exc:
                self.logger.warning(f"Could not clear stale stop signal file: {exc}")

        self.running = True
        self.engine_thread = threading.Thread(target=self._orchestration_loop, daemon=True)
        self.engine_thread.start()
        self.logger.info("Orchestrator engine started")

    def stop(self):
        """Stop the orchestration engine."""
        self.running = False
        if self.engine_thread:
            self.engine_thread.join(timeout=10)
        self.logger.info("Orchestrator engine stopped")

    def _orchestration_loop(self):
        """Main orchestration loop."""
        consecutive_failures = 0
        while self.running and self.current_generation < self.generations_total:
            try:
                if self.stop_signal_file and os.path.exists(self.stop_signal_file):
                    self.logger.warning(
                        f"Stop signal file detected at {self.stop_signal_file}; stopping orchestrator loop"
                    )
                    self.running = False
                    break

                self._refresh_active_paths(raise_on_missing=True)

                # Check disk pressure
                if self._check_disk_pressure():
                    self.logger.info("Disk pressure detected, pausing orchestration")
                    time.sleep(60)
                    continue

                # Process any completed results
                self._process_pending_results()

                # Recover stale workers and requeue lost claims
                self._recover_stale_workers()

                # Check if current generation is complete
                if self._is_generation_complete():
                    self._trigger_batch_validation()
                    self._advance_generation()
                    continue

                # Dispatch jobs if queue is low
                queue_depth = self.job_dispatcher.get_queue_depth()
                min_queue_depth = self.config.get('min_queue_depth', 5)

                if queue_depth < min_queue_depth:
                    # Generate candidate configs (dispatcher can still seed from backlog when empty)
                    candidate_configs = self._generate_candidate_configs()
                    jobs_dispatched = self.job_dispatcher.dispatch_generation(
                        self.current_generation, candidate_configs
                    )
                    if jobs_dispatched == 0:
                        self.logger.warning(
                            f"No jobs dispatched for generation {self.current_generation}; "
                            "awaiting backlog/candidate availability"
                        )
                    else:
                        self.logger.info(
                            f"Dispatched {jobs_dispatched} jobs for generation {self.current_generation}"
                        )

                # Run GC if needed
                self._run_garbage_collection()

                # Sleep before next iteration
                time.sleep(self.poll_interval)
                consecutive_failures = 0

            except Exception as e:
                consecutive_failures += 1
                backoff_seconds = min(
                    self.iteration_error_backoff_max_seconds,
                    float(2 ** min(consecutive_failures, 5))
                )
                self.logger.error(
                    "Transient orchestration iteration error. "
                    f"failure_count={consecutive_failures}, backoff={backoff_seconds:.1f}s, error={e}"
                )
                self.logger.error(traceback.format_exc())
                time.sleep(backoff_seconds)
                continue

        self.logger.info("Orchestration loop completed")

    def _trigger_batch_validation(self):
        """Trigger batch validation for all pending artifacts using thread pool."""
        import sqlite3
        import concurrent.futures
        
        executor = None
        try:
            conn = sqlite3.connect(self.ledger_db_path, timeout=30.0, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute("SELECT artifact_url, config_hash FROM runs WHERE status = 'PENDING_VALIDATION'")
            pending = cursor.fetchall()
            conn.close()
            
            if not pending:
                self.logger.info("No pending validation artifacts found.")
                return
                
            self.logger.info(f"Triggering batch validation for {len(pending)} artifacts.")
            from notebooks.validation_pipeline import run as validation_run  # Adjust import as needed
            
            max_workers = max(1, os.cpu_count() - 1)
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
            
            futures = [executor.submit(validation_run, artifact_url, config_hash) for artifact_url, config_hash in pending]
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    self.logger.info(f"Validation completed: {result}")
                except Exception as exc:
                    self.logger.error(f"Validation error: {exc}")
        except Exception as exc:
            self.logger.error(f"Batch validation trigger failed: {exc}")
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
                self.logger.info("Validation thread pool cleanly shut down and locks released.")

    def _check_disk_pressure(self) -> bool:
        """Check if disk usage exceeds threshold."""
        try:
            data_dir = str(self.config.get('data_dir') or '.')
            disk_usage = psutil.disk_usage(str(Path(data_dir).parent))
            used_gb = disk_usage.used / (1024**3)

            if used_gb > self.trigger_bleed_threshold_gb:
                self.logger.warning(f"Disk usage {used_gb:.1f}GB exceeds threshold {self.trigger_bleed_threshold_gb}GB")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking disk pressure: {e}")
            return False

    def _process_pending_results(self):
        """
        Process any pending results from workers.

        Ownership Contract (Phase B Freeze):
        - Completion accounting ownership is centralized here (`self.completed_jobs`).
        - Champion promotion ownership is centralized here (`global_best_*` updates).
        - Champion checkpoint writes are triggered here via `_save_state_atomic()`.
        """
        import os
        import time
        try:
            results = self.queue_manager.get_results()
            for result_data in results:
                generation = result_data.get('generation', self.current_generation)
                try:
                    generation_int = int(generation)
                except (TypeError, ValueError):
                    generation_int = self.current_generation
                config_hash = result_data.get('config_hash')
                completion_id = str(result_data.get('job_id') or config_hash or "unknown")
                status = str(result_data.get('status', '')).upper()
                dedupe_key = f"gen_{generation_int}_{str(config_hash or completion_id)}"

                prior_status = self._result_status_by_key.get(dedupe_key)
                if prior_status == 'SUCCESS' or (prior_status == 'FAIL' and status == 'FAIL'):
                    self.logger.info(
                        f"Skipping duplicate result replay for key={dedupe_key} status={status or 'UNKNOWN'}"
                    )
                    continue

                # ASTE D.3 Fix: Atomic HDF5 Release Wait with network filesystem guard
                artifact_path = result_data.get('artifact_url')
                if artifact_path and os.path.exists(artifact_path):
                    started = time.time()
                    while (time.time() - started) < self.artifact_read_timeout_seconds:
                        try:
                            if os.path.getsize(artifact_path) > 0:
                                with open(artifact_path, 'rb+'):
                                    break
                        except IOError:
                            pass
                        time.sleep(0.5)

                success = self.result_processor.process_result(result_data)
                self._result_status_by_key[dedupe_key] = 'SUCCESS' if success else (status or 'UNKNOWN')

                if success and result_data.get('artifact_url'):
                    self.completed_jobs.add(f"gen_{generation_int}_{completion_id}")

                    ingested_sse = float(result_data.get('_ingested_log_prime_sse', 999.0))
                    refinement_status = result_data.get('_refinement_status')
                    contract_version = result_data.get('_solver_contract_version', '')

                    champion_eligible = bool(result_data.get('_champion_eligible', True))
                    if not champion_eligible:
                        # Compatibility gate (result_processor): incompatible solver
                        # contract / variant / affect topology / grid.  Strictly
                        # excluded from comparing against or unseating the champion.
                        self.logger.warning(
                            f"[Gate] {str(config_hash)[:12]} not champion-eligible "
                            f"({result_data.get('_incompatible_reason', 'incompatible variant')}); "
                            "recorded but excluded from champion comparison."
                        )
                    elif contract_version != SOLVER_CONTRACT_VERSION:
                        self.logger.warning(
                            f"[Gate] Physics uncertified for {str(config_hash)[:12]}: "
                            f"contract={contract_version!r} (expected {SOLVER_CONTRACT_VERSION!r}). "
                            "Champion promotion blocked."
                        )
                    elif refinement_status == 'VALIDATED_PROVISIONAL':
                        self._dispatch_refinement_variants(result_data, config_hash, ingested_sse)
                    elif refinement_status == 'REFINEMENT_STABLE':
                        if config_hash and ingested_sse < self.global_best_sse:
                            self.global_best_sse = ingested_sse
                            self.global_best_hash = str(config_hash)
                            self.result_processor.trigger_champion_visual_async(result_data, ingested_sse)
                            self.result_processor._trigger_predator_sweep_async(result_data)
                            self.logger.info(
                                f"New champion accepted (REFINEMENT_STABLE): "
                                f"{self.global_best_hash[:12]} (SSE {self.global_best_sse:.6f})"
                            )
                            self._save_state_atomic()
                    elif refinement_status == 'REFINEMENT_REJECTED':
                        self.logger.info(
                            f"[Gate] Candidate {str(config_hash)[:12]} REFINEMENT_REJECTED — not promoted."
                        )
                    elif ingested_sse < self.global_best_sse and refinement_status is None:
                        # Legacy artifact (no contract field): allow champion update but log clearly.
                        self.global_best_sse = ingested_sse
                        self.global_best_hash = str(config_hash)
                        self.result_processor.trigger_champion_visual_async(result_data, ingested_sse)
                        self.logger.info(
                            f"[Legacy] Champion updated without certification: "
                            f"{self.global_best_hash[:12]} (SSE {self.global_best_sse:.6f}). "
                            "TRIAGE_ONLY / NOT_SCIENTIFIC_EVIDENCE"
                        )
                        self._save_state_atomic()

                    # Queue artifact for bleeding if successful
                    self.bleed_manager.queue_for_bleed(
                        result_data['artifact_url'],
                        result_data.get('generation', 0),
                        result_data.get('config_hash', 'unknown')
                    )
                elif status == 'FAIL':
                    self.completed_jobs.add(f"gen_{generation_int}_{completion_id}")
                if not success:
                    self.logger.warning(f"Failed to process result for job {result_data.get('job_id')}")
        except Exception as e:
            self.logger.error(f"Error processing pending results: {e}")

    def _dispatch_refinement_variants(
        self,
        result_data: Dict[str, Any],
        config_hash: str,
        ingested_sse: float,
    ) -> None:
        """Dispatch falsifiability-ladder variants for a VALIDATED_PROVISIONAL candidate."""
        import sqlite3 as _sqlite3
        try:
            from validation_pipeline import RefinementManifestGenerator
        except ImportError:
            self.logger.error("RefinementManifestGenerator import failed; refinement skipped.")
            return

        base_params = result_data.get('config', {})
        if not base_params:
            base_params = result_data.get('params', {})
        if not base_params:
            self.logger.warning(f"[Refinement] No params for {str(config_hash)[:12]}; cannot dispatch variants.")
            return

        variants = RefinementManifestGenerator.generate(base_params)

        conn = _sqlite3.connect(self.ledger_db_path, timeout=30.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        try:
            for v in variants:
                if not v.get('feasible', True):
                    conn.execute(
                        "INSERT OR IGNORE INTO refinement_variants "
                        "(baseline_config_hash, variant_label, status) VALUES (?, ?, 'SKIPPED')",
                        (str(config_hash), v['label']),
                    )
                    continue
                conn.execute(
                    "INSERT OR IGNORE INTO refinement_variants "
                    "(baseline_config_hash, variant_label, status) VALUES (?, ?, 'PENDING')",
                    (str(config_hash), v['label']),
                )
                try:
                    from orchestrator.contracts import JobManifest
                    manifest = JobManifest.from_params(
                        v['params'],
                        generation=result_data.get('generation', 0),
                        origin='REFINEMENT',
                    )
                    self.queue_manager.push_job(manifest.model_dump_json())
                except Exception as enq_err:
                    self.logger.warning(f"[Refinement] Enqueue failed for {v['label']}: {enq_err}")
            conn.commit()
        finally:
            conn.close()

        self.logger.info(
            f"[Refinement] Dispatched {len(variants)} variants for {str(config_hash)[:12]} "
            f"(SSE {ingested_sse:.4f}). Marked REFINEMENT_PENDING."
        )
        log_lifecycle_event(
            stage='refinement_dispatch',
            generation=result_data.get('generation', 0),
            config_hash=config_hash,
            details={"n_variants": len(variants), "baseline_sse": ingested_sse},
        )

    def _is_generation_complete(self) -> bool:
        """Check if all jobs for current generation are complete."""
        expected_jobs = self.jobs_per_generation
        completed_count = len([
            job_key
            for job_key in self.completed_jobs
            if job_key.startswith(f"gen_{self.current_generation}_")
        ])

        return completed_count >= expected_jobs

    def _advance_generation(self):
        """Advance to next generation."""
        self.logger.info(f"Generation {self.current_generation} complete, advancing to {self.current_generation + 1}")
        self.current_generation += 1
        self.completed_jobs.clear()
        self._result_status_by_key.clear()
        self._save_state_atomic()
        self._trigger_fss_predictor_async()

        # Emit generation advance event
        log_lifecycle_event(
            stage='generation_advance',
            generation=self.current_generation,
            config_hash=self.config.get('config_hash', 'unknown')
        )

    def _trigger_fss_predictor_async(self) -> None:
        """Launch FSS scaling analyzer as a detached process for next-generation prediction."""
        repo_root = Path(__file__).resolve().parent.parent
        script_path = repo_root / "plugins" / "tools" / "fss_scaling_analyzer.py"
        if not script_path.exists() or not script_path.is_file():
            self.logger.warning(f"FSS analyzer script missing, skipping async launch: {script_path}")
            return

        cmd = [
            sys.executable,
            str(script_path),
            "--generation",
            str(self.current_generation + 1),
            "--db",
            str(self.ledger_db_path),
        ]

        popen_kwargs: Dict[str, Any] = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "stdin": subprocess.DEVNULL,
            "close_fds": True,
            "cwd": str(repo_root),
        }

        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        else:
            popen_kwargs["start_new_session"] = True

        try:
            subprocess.Popen(cmd, **popen_kwargs)
            self.logger.info("Launched async FSS predictor for next generation")
        except Exception as exc:
            self.logger.warning(f"Failed to launch async FSS predictor: {exc}")

    def _generate_candidate_configs(self) -> List[Dict[str, Any]]:
        """Generate candidate configurations using the NSGA-II Hunter."""
        return self.hunter.generate_next_generation(
            population_size=self.config.get('population_size', 10),
            bounds=self.config.get('bounds')
        )

    def _run_garbage_collection(self):
        """Run garbage collection on old artifacts."""
        try:
            self._refresh_active_paths(raise_on_missing=True)

            # Safely fetch known archived files to prevent deleting un-backed-up data
            archived = set()
            if hasattr(self.bleed_manager, 'get_archived_files'):
                archived = self.bleed_manager.get_archived_files()
            elif hasattr(self.bleed_manager, '_archived_files'):
                archived = self.bleed_manager._archived_files

            protected_files = set()
            data_dir = str(self.config.get('data_dir') or '.')
            if self.global_best_hash:
                protected_files.add(
                    os.path.abspath(
                        os.path.join(
                            data_dir,
                            f"rho_history_{self.global_best_hash}.h5"
                        )
                    )
                )

            purged_count = purge_old_artifacts(
                data_dir=data_dir,
                protected_files=protected_files,
                current_generation=self.current_generation,
                require_archived=self.config.get('gc_require_archived', True),
                min_age_seconds=self.config.get('artifact_gc_min_age_seconds', 300),
                archived_files=archived,
                bleed_locked_files=getattr(self.bleed_manager, 'locked_for_bleed', set()),
            )

            if purged_count > 0:
                self.logger.info(f"Garbage collected {purged_count} artifacts")

        except Exception as e:
            self.logger.error(f"Error during garbage collection: {e}")

    def _recover_stale_workers(self) -> None:
        """Revoke stale worker claims and requeue jobs if heartbeat TTL is exceeded."""
        if self.worker_heartbeat_ttl_seconds <= 0:
            return
        try:
            stale_workers = self.queue_manager.recover_stale_workers(self.worker_heartbeat_ttl_seconds)
            for worker_id in stale_workers:
                self.logger.warning(f"Recovered stale worker claim set: {worker_id}")
                log_lifecycle_event(
                    stage='worker_stale_recovered',
                    generation=self.current_generation,
                    config_hash=self.config.get('config_hash', 'unknown'),
                    details={"worker_id": worker_id, "ttl_seconds": self.worker_heartbeat_ttl_seconds}
                )
        except Exception as exc:
            self.logger.error(f"Error during stale worker recovery: {exc}")

    def get_status(self) -> Dict[str, Any]:
        """Get current engine status."""
        return {
            'running': self.running,
            'current_generation': self.current_generation,
            'total_generations': self.generations_total,
            'queue_depth': self.job_dispatcher.get_queue_depth(),
            'completed_jobs': len(self.completed_jobs),
            'global_best_hash': self.global_best_hash,
            'global_best_sse': self.global_best_sse,
            'disk_pressure': self._check_disk_pressure()
        }

    def _load_checkpoint(self) -> None:
        """Restore generation + champion state from orchestrator_state.json if present."""
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file, 'r', encoding='utf-8') as handle:
                state = json.load(handle)
            self.current_generation = int(state.get('current_generation', self.current_generation))
            self.global_best_hash = state.get('global_best_hash')
            self.global_best_sse = float(state.get('global_best_sse', self.global_best_sse))
            self.logger.info(
                "Restored orchestrator state: "
                f"gen={self.current_generation}, champion={self.global_best_hash}, sse={self.global_best_sse}"
            )
        except Exception as exc:
            self.logger.warning(f"Could not restore orchestrator state: {exc}")

    def _load_state(self) -> None:
        """Backward-compatible alias for checkpoint restore."""
        self._load_checkpoint()

    def _save_state_atomic(self) -> None:
        """Persist generation + champion state atomically via temp-write + replace."""
        try:
            payload = {
                'current_generation': self.current_generation,
                'global_best_hash': self.global_best_hash,
                'global_best_sse': self.global_best_sse,
            }
            tmp_path = f"{self.state_file}.tmp"
            with open(tmp_path, 'w', encoding='utf-8') as handle:
                json.dump(payload, handle, indent=2)
            os.replace(tmp_path, self.state_file)
        except Exception as exc:
            self.logger.warning(f"Could not persist orchestrator state: {exc}")

    def _initialize_ledger_schema(self) -> None:
        """Initialize simulation ledger schema once at orchestrator startup."""
        try:
            initialize_ledger_schema(self.ledger_db_path)
            ensure_ledger_ready(self.ledger_db_path, raise_on_fail=True)
            self.logger.info(f"Ledger schema initialized at {self.ledger_db_path}")
        except Exception as exc:
            self.logger.error(f"Failed to initialize ledger schema: {exc}")
            raise

    def _refresh_active_paths(self, raise_on_missing: bool = True) -> bool:
        """Refresh runtime paths from .active_run_pointer.json across long-lived processes."""
        try:
            active = resolve_active_session_paths(require_exists=True)
            self.ledger_db_path = str(active['db_file'])
            self.config['ledger_db_path'] = self.ledger_db_path
            self.config['db_path'] = self.ledger_db_path
            self.config['data_dir'] = str(active['session_dir'])
            self.config['provenance_dir'] = str(active['provenance_dir'])
            self.state_file = str(Path(active['session_dir']) / 'orchestrator_state.json')

            if hasattr(self, 'result_processor') and self.result_processor is not None:
                self.result_processor.db_path = self.ledger_db_path
                self.result_processor.data_dir = self.config['data_dir']
                self.result_processor.provenance_dir = self.config['provenance_dir']

            if hasattr(self, 'bleed_manager') and self.bleed_manager is not None:
                self.bleed_manager.data_dir = Path(self.config['data_dir'])

            return True
        except ActiveRunPointerError as exc:
            if raise_on_missing:
                raise RuntimeError(f"Active session pointer unavailable: {exc}") from exc
            self.logger.warning(f"Active session pointer unavailable: {exc}")
            return False