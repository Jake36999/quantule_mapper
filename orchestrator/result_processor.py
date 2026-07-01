#!/usr/bin/env python3

"""
result_processor.py
Processes simulation results with validation and lifecycle auditing.
"""

import os
import requests
import json
import logging
import sqlite3
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

from orchestrator.contracts import (
    JobResult,
    parse_job_result_payload,
    SOLVER_CONTRACT_VERSION,
    PROVISIONAL_SSE_THRESHOLD,
)
from orchestrator import run_identity as ri
from .diagnostics.runtime_audit import log_lifecycle_event


class ResultProcessor:
    """Processes and validates simulation results."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = config['db_path']
        self.data_dir = config['data_dir']
        self.provenance_dir = config['provenance_dir']
        self.logger = logging.getLogger(__name__)

    def process_result(self, result_data: Dict[str, Any]) -> bool:
        """
        Process a single result from a worker.
        Returns True if processing successful.
        """
        try:
            result_model = parse_job_result_payload(result_data)
            canonical_result = result_model.model_dump()
            result_data.clear()
            result_data.update(canonical_result)

            job_id = canonical_result.get('job_id')
            generation = canonical_result.get('generation')
            config_hash = canonical_result.get('config_hash')

            if str(canonical_result.get("status", "")).upper() == "FAIL":
                self._write_worker_result_to_ledger(result_model)
                log_lifecycle_event(
                    stage='result_fail_ingest',
                    generation=generation,
                    config_hash=config_hash,
                    job_id=job_id,
                    details={"reason": canonical_result.get("reason") or canonical_result.get("error") or "worker_reported_fail"}
                )
                return False

            # Validate result using validation pipeline
            validation_result = self._validate_result(canonical_result)
            if not validation_result:
                self.logger.warning(f"Validation failed for job {job_id}")
                return False

            # Compatibility gate: a run may only compete for champion against runs
            # of the same solver contract / variant / affect topology / grid.  We
            # compute eligibility BEFORE the ledger write so it is persisted, and
            # incompatible runs are recorded but excluded from champion ranking.
            identity_fields = self._run_identity_fields(result_model)
            gate = self._evaluate_compatibility_gate(result_model, identity_fields)
            result_data["_champion_eligible"] = gate["champion_eligible"]
            result_data["_compatibility_key"] = gate["incoming_key"]
            if not gate["champion_eligible"]:
                result_data["_incompatible_reason"] = gate["reason"]
                self.logger.warning(
                    f"[Gate] {str(config_hash)[:12]} not champion-eligible: {gate['reason']}. "
                    "Recorded in ledger but excluded from champion comparison."
                )

            self._write_worker_result_to_ledger(
                result_model, validation_result,
                identity_fields=identity_fields,
                champion_eligible=gate["champion_eligible"],
            )

            # Store in database
            self._store_result(canonical_result, validation_result)

            log_prime_sse = float(validation_result.get("log_prime_sse", 999.0))
            result_data["_ingested_log_prime_sse"] = log_prime_sse

            # Extract solver contract from provenance and set certification status.
            provenance_payload = validation_result.get("provenance", {})
            contract = provenance_payload.get("solver_contract") if isinstance(provenance_payload, dict) else None
            contract_version = (contract or {}).get("solver_contract_version", "") if isinstance(contract, dict) else ""
            result_data["_solver_contract_version"] = contract_version

            if not gate["champion_eligible"]:
                # Incompatible variant: record, but never rank/unseat the champion.
                result_data["_refinement_status"] = "PHYSICS_UNCERTIFIED"
            elif contract_version != SOLVER_CONTRACT_VERSION:
                result_data["_refinement_status"] = "PHYSICS_UNCERTIFIED"
            elif log_prime_sse < PROVISIONAL_SSE_THRESHOLD:
                result_data["_refinement_status"] = "VALIDATED_PROVISIONAL"

            triage_tier: Optional[str] = None
            if log_prime_sse < 1.0:
                triage_tier = "GOLDEN"
            elif log_prime_sse < 3.0:
                triage_tier = "SILVER"
            if triage_tier:
                result_data["_triage_tier"] = triage_tier
                self._trigger_visual_observer_async(result_data, log_prime_sse, triage_tier)
                # Predator sweep is intentionally NOT triggered here.
                # It is gated on REFINEMENT_STABLE in orchestrator_engine.

            # Emit audit event
            log_lifecycle_event(
                stage='result_ingest',
                generation=generation,
                config_hash=config_hash,
                job_id=job_id,
                details={
                    "validation_score": validation_result.get('composite_fitness', 0.0),
                    "log_prime_sse": log_prime_sse,
                    "triage_tier": triage_tier,
                    "golden_threshold": 1.0,
                    "silver_threshold": 3.0,
                }
            )

            # Handle hunter persistence if needed
            if canonical_result.get('persist_hunter', False):
                self._persist_hunter_state(canonical_result)
                log_lifecycle_event(
                    stage='hunter_persist',
                    generation=generation,
                    config_hash=config_hash,
                    job_id=job_id
                )

            self.logger.info(f"Successfully processed result for job {job_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error processing result: {e}")
            return False

    def trigger_champion_visual_async(self, result_data: Dict[str, Any], ingested_sse: float) -> None:
        """Launch champion visual observer pipeline for newly accepted global best results."""
        try:
            payload = {
                "plugin_id": "gif_pipeline_manager",
                "artifact_path": result_data.get('artifact_url'),
                "sse": ingested_sse
            }
            # The backend execution endpoint
            requests.post("http://localhost:8000/api/plugins/execute", json=payload, timeout=3.0)
        except Exception as e:
            self.logger.warning(f"Failed to trigger GIF pipeline: {e}")

    def _run_identity_fields(self, result_model: JobResult) -> Dict[str, Any]:
        """
        Resolve the discriminator identity for a run.  Prefers the artifact's
        /identity group (stamped by the solver), falling back to deriving the
        variant fields from params.  Never raises.
        """
        params = dict(result_model.config or {})
        identity: Dict[str, Any] = {}
        artifact = getattr(result_model, "artifact_url", None)
        try:
            import h5py
            if artifact and os.path.exists(artifact):
                with h5py.File(artifact, "r") as f:
                    identity = ri.read_identity_group(f)
        except Exception:
            identity = {}

        seed = identity.get("seed")
        if seed is None:
            try:
                seed = int(params.get("global_seed", 0) or 0)
            except (TypeError, ValueError):
                seed = 0

        artifact_hash = None
        try:
            if artifact and os.path.exists(artifact):
                artifact_hash = ri.sha256_file(artifact)
        except Exception:
            artifact_hash = None

        return {
            "seed": int(seed),
            "run_id": identity.get("run_id") or getattr(result_model, "job_id", None),
            "hunt_name": identity.get("hunt_name"),
            "utc_start": identity.get("utc_start"),
            "solver_contract_version": identity.get("solver_contract_version")
            or ri.solver_contract_version_for_params(params),
            "variant_label": identity.get("variant_label") or ri.variant_label_for_params(params),
            "affect_topology": identity.get("affect_topology") or ri.affect_topology_for_params(params),
            "affect_strength": identity.get("affect_strength", ri.affect_strength_for_params(params)),
            "git_commit": identity.get("git_commit"),
            "n_grid": identity.get("N_grid"),
            "dt": identity.get("dt"),
            "t_steps": identity.get("T_steps"),
            "gpu_backend": identity.get("gpu_backend"),
            "artifact_hash": artifact_hash,
        }

    def _reference_compatibility_key(
        self,
        conn: sqlite3.Connection,
        exclude_config_hash: Optional[str] = None,
        exclude_seed: Optional[int] = None,
    ):
        """
        Resolve the compatibility key of the current champion = the best-fitness,
        champion-eligible SUCCESS run already in the ledger (excluding the run
        being processed).  Returns None when no prior eligible run exists, in
        which case the incoming run establishes the comparison class.
        """
        try:
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(runs)")
            cols = {row[1] for row in cur.fetchall()}
            needed = {"solver_contract_version", "variant_label", "affect_topology", "n_grid", "fitness", "status"}
            if not needed.issubset(cols):
                return None
            where = ["status = 'SUCCESS'", "fitness IS NOT NULL"]
            params: list = []
            if "champion_eligible" in cols:
                where.append("COALESCE(champion_eligible, 1) = 1")
            if exclude_config_hash is not None:
                where.append("NOT (config_hash = ? AND seed = ?)")
                params.extend([exclude_config_hash, int(exclude_seed or 0)])
            cur.execute(
                "SELECT solver_contract_version, variant_label, affect_topology, n_grid "
                f"FROM runs WHERE {' AND '.join(where)} ORDER BY fitness ASC LIMIT 1",
                params,
            )
            row = cur.fetchone()
            if not row:
                return None
            return ri.compatibility_key({
                "solver_contract_version": row[0],
                "variant_label": row[1],
                "affect_topology": row[2],
                "N_grid": row[3],
            })
        except Exception:
            return None

    def _evaluate_compatibility_gate(
        self,
        result_model: JobResult,
        identity_fields: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Decide whether this run may compete for champion.  Eligible iff its
        compatibility key (solver contract, variant, affect topology, grid)
        matches the current champion's.  With no champion yet, the run is
        eligible and establishes the comparison class.  Incompatible runs are
        still recorded in the ledger; they are only excluded from ranking.
        """
        incoming = ri.compatibility_key({
            "solver_contract_version": identity_fields.get("solver_contract_version"),
            "variant_label": identity_fields.get("variant_label"),
            "affect_topology": identity_fields.get("affect_topology"),
            "N_grid": identity_fields.get("n_grid"),
        })
        reference = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
            try:
                reference = self._reference_compatibility_key(
                    conn,
                    exclude_config_hash=str(result_model.config_hash),
                    exclude_seed=identity_fields.get("seed"),
                )
            finally:
                conn.close()
        except Exception:
            reference = None
        eligible = reference is None or incoming == reference
        reason = "" if eligible else f"incompatible variant {incoming} != champion {reference}"
        return {
            "champion_eligible": eligible,
            "incoming_key": list(incoming),
            "reference_key": list(reference) if reference is not None else None,
            "reason": reason,
        }

    @staticmethod
    def _upsert_run_row(
        cursor: sqlite3.Cursor,
        run_columns: set,
        config_hash: str,
        generation: int,
        status: str,
        fitness: float,
        origin: str,
        staged_path: Optional[str],
        staged_at: Optional[str],
        identity_fields: Dict[str, Any],
    ) -> None:
        """Build an INSERT OR REPLACE that only references columns that exist,
        so it works on both DC-v1.0 (composite PK) and legacy schemas."""
        row: Dict[str, Any] = {
            "config_hash": config_hash,
            "generation": generation,
            "status": status,
            "fitness": fitness,
            "origin": origin,
            "staged_path": staged_path,
            "staged_at": staged_at,
        }
        row.update(identity_fields)
        cols = [c for c in row if c in run_columns]
        # config_hash is always required.
        if "config_hash" not in cols:
            cols.insert(0, "config_hash")
        placeholders = ", ".join("?" for _ in cols)
        cursor.execute(
            f"INSERT OR REPLACE INTO runs ({', '.join(cols)}) VALUES ({placeholders})",
            [row.get(c) for c in cols],
        )

    def _write_worker_result_to_ledger(
        self,
        result_model: JobResult,
        validation_result: Optional[Dict[str, Any]] = None,
        identity_fields: Optional[Dict[str, Any]] = None,
        champion_eligible: Optional[bool] = None,
    ) -> None:
        """Authoritative ledger persistence for worker outputs (orchestrator-owned writes)."""
        from orchestrator.schema_utils import initialize_ledger_schema

        initialize_ledger_schema(self.db_path)
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=30000;")
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(runs)")
            run_columns = {row[1] for row in cursor.fetchall()}

            params = dict(result_model.config or {})
            generation = int(result_model.generation)
            config_hash = str(result_model.config_hash)
            origin = str(result_model.origin or params.get("origin") or "NATURAL")

            staged_path = result_model.staged_path
            staged_at = result_model.staged_at

            status = str(result_model.status).upper()
            if identity_fields is None:
                identity_fields = self._run_identity_fields(result_model)
            if champion_eligible is not None and "champion_eligible" in run_columns:
                identity_fields = {**identity_fields, "champion_eligible": 1 if champion_eligible else 0}
            if status == "SUCCESS" and validation_result is not None:
                log_prime_sse = float(validation_result.get("log_prime_sse", 999.0))
                self._upsert_run_row(
                    cursor, run_columns, config_hash, generation, "SUCCESS",
                    log_prime_sse, origin, staged_path, staged_at, identity_fields,
                )

                param_columns = {row[1] for row in cursor.execute("PRAGMA table_info(parameters)").fetchall()}
                if "param_affect_coupling" in param_columns:
                    # param_omega0 defaults to param_rho_vac (back-compat) when unset.
                    _omega0 = params.get("param_omega0", params.get("param_rho_vac"))
                    cursor.execute(
                        "INSERT OR REPLACE INTO parameters (config_hash, param_D, param_eta, param_rho_vac, param_omega0, param_a_coupling, param_splash_coupling, param_splash_fraction, param_affect_coupling, param_affect_topology) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            config_hash,
                            params.get("param_D"),
                            params.get("param_eta"),
                            params.get("param_rho_vac"),
                            _omega0,
                            params.get("param_a_coupling"),
                            params.get("param_splash_coupling"),
                            params.get("param_splash_fraction"),
                            identity_fields.get("affect_strength"),
                            identity_fields.get("affect_topology"),
                        ),
                    )
                else:
                    cursor.execute(
                        "INSERT OR REPLACE INTO parameters (config_hash, param_D, param_eta, param_rho_vac, param_a_coupling, param_splash_coupling, param_splash_fraction) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            config_hash,
                            params.get("param_D"),
                            params.get("param_eta"),
                            params.get("param_rho_vac"),
                            params.get("param_a_coupling"),
                            params.get("param_splash_coupling"),
                            params.get("param_splash_fraction"),
                        ),
                    )

                provenance_payload = validation_result.get("provenance", {})
                spectral = provenance_payload.get("spectral_fidelity", {}) if isinstance(provenance_payload, dict) else {}
                aletheia = provenance_payload.get("aletheia_metrics", {}) if isinstance(provenance_payload, dict) else {}
                contract = provenance_payload.get("solver_contract") if isinstance(provenance_payload, dict) else {}
                contract_version = (contract or {}).get("solver_contract_version", "") if isinstance(contract, dict) else ""
                refinement_status = None
                if contract_version != SOLVER_CONTRACT_VERSION:
                    refinement_status = "PHYSICS_UNCERTIFIED"
                elif log_prime_sse < PROVISIONAL_SSE_THRESHOLD:
                    refinement_status = "VALIDATED_PROVISIONAL"
                peak_count = spectral.get("bragg_peaks_detected", spectral.get("n_bragg_peaks", 0))
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO metrics
                    (config_hash, log_prime_sse, bragg_peaks_detected, n_bragg_peaks, bragg_prime_sse, collapse_event_count, pcs, refinement_status, solver_contract_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        config_hash,
                        spectral.get("log_prime_sse"),
                        peak_count,
                        peak_count,
                        spectral.get("bragg_prime_sse", spectral.get("bragg_lattice_sse")),
                        spectral.get("collapse_event_count", 0),
                        aletheia.get("pcs"),
                        refinement_status,
                        contract_version,
                    ),
                )
            else:
                self._upsert_run_row(
                    cursor, run_columns, config_hash, generation, "FAIL",
                    999.0, origin, staged_path, staged_at, identity_fields,
                )

            conn.commit()
        finally:
            conn.close()

    def _validate_result(self, result_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Validate result using validation pipeline.

        Ownership Contract (Phase B Freeze):
        - Canonical ingest SSE (`log_prime_sse`) is sourced here from provenance.
        - Downstream orchestrator accounting/champion logic must consume this value
          through `process_result` output payload fields, not recompute SSE elsewhere.
        """
        try:
            provenance_path = result_data.get("provenance_path")
            if provenance_path and os.path.exists(provenance_path):
                with open(provenance_path, "r", encoding="utf-8") as handle:
                    provenance_payload = json.load(handle)

                spectral = provenance_payload.get("spectral_fidelity", {})
                log_prime_sse = float(spectral.get("log_prime_sse", 999.0))
                composite_fitness = 0.0 if log_prime_sse >= 999.0 else (1.0 / (log_prime_sse + 1e-9))

                return {
                    "composite_fitness": composite_fitness,
                    "log_prime_sse": log_prime_sse,
                    "provenance": provenance_payload,
                }

            sys.path.insert(0, str(Path(__file__).parent.parent))
            from validation_pipeline import ValidationPipeline

            artifact_path = result_data.get("artifact_url")
            params_path = result_data.get("params_path")
            if not artifact_path or not params_path:
                self.logger.error("Validation fallback requires artifact_url and params_path")
                return None

            pipeline = ValidationPipeline(
                input_path=artifact_path,
                params_path=params_path,
                output_dir=self.provenance_dir,
            )
            if not pipeline.run():
                return None

            config_hash = str(result_data.get("config_hash", ""))
            # Collision-free provenance name, consistent with the writer.  Prefer
            # the path the pipeline actually wrote; fall back to the artifact-derived
            # name (which folds in seed/run_id via the /identity group).
            generated_provenance = getattr(pipeline, "provenance_path", None) or ri.provenance_path_for_artifact(
                self.provenance_dir, artifact_path, config_hash
            )
            if not generated_provenance or not os.path.exists(generated_provenance):
                return None

            with open(generated_provenance, "r", encoding="utf-8") as handle:
                provenance_payload = json.load(handle)
            spectral = provenance_payload.get("spectral_fidelity", {})
            log_prime_sse = float(spectral.get("log_prime_sse", 999.0))
            composite_fitness = 0.0 if log_prime_sse >= 999.0 else (1.0 / (log_prime_sse + 1e-9))
            return {
                "composite_fitness": composite_fitness,
                "log_prime_sse": log_prime_sse,
                "provenance": provenance_payload,
            }
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return None

    def _trigger_visual_observer_async(self, result_data: Dict[str, Any], log_prime_sse: float, triage_tier: str) -> None:
        """Launch visual observer asynchronously with no blocking wait/check."""
        artifact_path = result_data.get("artifact_url")
        if not artifact_path or not os.path.exists(artifact_path):
            return

        repo_root = Path(__file__).resolve().parent.parent
        script_path = repo_root / "plugins" / "visualizers" / "gif_pipeline_manager.py"
        if not script_path.exists():
            self.logger.warning(f"Visual observer script missing: {script_path}")
            return

        try:
            sse_value = float(log_prime_sse)
        except (TypeError, ValueError):
            sse_value = 999.0

        tier_value = str(triage_tier or "SILVER").upper()
        cmd = [
            sys.executable,
            str(script_path),
            "--input",
            str(artifact_path),
            "--sse",
            str(sse_value),
            "--tier",
            tier_value,
        ]

        popen_kwargs: Dict[str, Any] = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "stdin": subprocess.DEVNULL,
            "close_fds": True,
        }

        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        else:
            popen_kwargs["start_new_session"] = True

        try:
            subprocess.Popen(cmd, cwd=str(repo_root), **popen_kwargs)
            self.logger.info(
                f"Launched async visual observer for {tier_value} result "
                f"(SSE={sse_value:.6f}, hash={result_data.get('config_hash')})"
            )
        except Exception as exc:
            self.logger.warning(f"Failed to launch async visual observer: {exc}")

    def _trigger_predator_sweep_async(self, result_data: Dict[str, Any]) -> None:
        """Launch predator sweep asynchronously for GOLDEN results."""
        config_hash = result_data.get("config_hash")
        if not config_hash:
            return

        repo_root = Path(__file__).resolve().parent.parent
        script_path = repo_root / "predator_sweep.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--target_hash",
            str(config_hash),
            "--db",
            str(repo_root / "simulation_ledger.db"),
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
            self.logger.info(f"Launched async Predator sweep for GOLDEN hash {str(config_hash)[:12]}")
        except Exception as exc:
            self.logger.warning(f"Failed to launch Predator sweep: {exc}")

    def _store_result(self, result_data: Dict[str, Any], validation_result: Dict[str, Any]):
        """Store result in simulation ledger database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS simulation_results (
                    job_id TEXT PRIMARY KEY,
                    generation INTEGER,
                    config_hash TEXT,
                    config_json TEXT,
                    result_json TEXT,
                    validation_json TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # Insert or update result record
            cursor.execute('''
                INSERT OR REPLACE INTO simulation_results
                (job_id, generation, config_hash, config_json, result_json, validation_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
            ''', (
                result_data['job_id'],
                result_data['generation'],
                result_data['config_hash'],
                json.dumps(result_data.get('config', {})),
                json.dumps(result_data),
                json.dumps(validation_result)
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"Database error: {e}")
            raise

    def _persist_hunter_state(self, result_data: Dict[str, Any]):
        """Persist hunter state if required."""
        # Implementation depends on hunter specifics
        # For now, just log
        self.logger.info(f"Persisting hunter state for job {result_data['job_id']}")