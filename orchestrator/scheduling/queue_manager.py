"""Job queue management (SQLite WAL-backed with legacy JSON bootstrap migration)."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional, Sequence

from orchestrator.contracts import parse_job_manifest_payload, parse_job_result_payload
from orchestrator.schema_utils import (
    initialize_ledger_schema,
    initialize_queue_schema,
    migrate_json_queue_to_sqlite,
)



class QueueManager:
    """SQLite-backed job queue with worker management and durable telemetry."""

    def _initialize_schema(self, conn):
        """Ensure the 'simulations' table exists with the required columns."""
        conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS simulations (
                config_hash TEXT PRIMARY KEY,
                status TEXT,
                log_prime_sse REAL,
                param_D REAL,
                param_eta REAL,
                param_rho_vac REAL,
                param_a_coupling REAL,
                param_splash_coupling REAL,
                param_splash_fraction REAL,
                artifact_url TEXT
            )
            '''
        )

    def clear_all_workers(self) -> None:
        """Drop all active claims back to the backlog and clear worker registry."""
        conn = self._connect_queue()
        try:
            cur = conn.cursor()
            cur.execute("BEGIN EXCLUSIVE")
            cur.execute("SELECT claim_token, payload, claimed_at, retry_count FROM queue_active")
            rows = cur.fetchall()
            for claim_token, payload_str, claimed_at, retry_count in rows:
                payload_dict = self._row_payload_to_dict(str(payload_str))
                cur.execute(
                    "INSERT INTO queue_backlog (job_id, config_hash, payload, created_at, retry_count) VALUES (?, ?, ?, ?, ?)",
                    (
                        str(payload_dict.get("job_id") or ""),
                        str(payload_dict.get("config_hash") or ""),
                        str(payload_str),
                        float(claimed_at) if claimed_at is not None else float(time.time()),
                        int(retry_count),
                    ),
                )
            cur.execute("DELETE FROM queue_active")
            cur.execute("DELETE FROM queue_workers")
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _coerce_manifest_dict(payload: Any) -> Dict[str, Any]:
        model = parse_job_manifest_payload(payload)
        return model.model_dump()

    @staticmethod
    def _coerce_result_dict(payload: Any) -> Dict[str, Any]:
        model = parse_job_result_payload(payload)
        return model.model_dump()

    def __init__(
        self,
        queue_file: str = "job_queue.json",
        result_file: str = "result_queue.json",
        queue_db_path: str | None = None,
        ledger_db_path: str | None = None,
        max_claim_retries: int = 3,
    ):
        self.queue_file = queue_file
        self.result_file = result_file
        self.workers_file = f"{self.queue_file}.workers.json"
        self.claims_file = f"{self.queue_file}.claims.json"
        self.telemetry_file = f"{self.queue_file}.telemetry.json"

        queue_dir = os.path.dirname(os.path.abspath(self.queue_file)) or os.getcwd()
        self.queue_db_path = queue_db_path or os.path.join(queue_dir, "queue_runtime.db")
        self.ledger_db_path = ledger_db_path or self._resolve_ledger_db_path(queue_dir)
        self.max_claim_retries = int(max_claim_retries)

        initialize_queue_schema(self.queue_db_path)
        migrate_json_queue_to_sqlite(
            queue_db_path=self.queue_db_path,
            queue_file=self.queue_file,
            claims_file=self.claims_file,
            result_file=self.result_file,
            workers_file=self.workers_file,
            telemetry_file=self.telemetry_file,
        )
        # Drop malformed legacy rows that are not valid JobManifest payloads.
        self._sanitize_backlog_payloads()
        # Legacy JSON mirror is optional and disabled by default on Windows due to file lock contention.
        if os.environ.get("ASTE_ENABLE_LEGACY_JSON_MIRROR_BOOTSTRAP", "0") == "1":
            self._sync_legacy_json_from_db()

    def _sanitize_backlog_payloads(self) -> None:
        """Remove queue_backlog rows that cannot be parsed as JobManifest payloads."""
        conn = self._connect_queue()
        try:
            cur = conn.cursor()
            cur.execute("BEGIN IMMEDIATE")
            cur.execute("SELECT id, payload FROM queue_backlog ORDER BY created_at ASC, id ASC")
            rows = cur.fetchall()
            for row_id, payload_str in rows:
                try:
                    payload_obj = json.loads(str(payload_str))
                    manifest_dict = self._coerce_manifest_dict(payload_obj)
                    normalized_payload = json.dumps(manifest_dict, sort_keys=True)
                    if normalized_payload != str(payload_str):
                        cur.execute(
                            "UPDATE queue_backlog SET job_id = ?, config_hash = ?, payload = ? WHERE id = ?",
                            (
                                str(manifest_dict.get("job_id") or ""),
                                str(manifest_dict.get("config_hash") or ""),
                                normalized_payload,
                                int(row_id),
                            ),
                        )
                except Exception:
                    cur.execute("DELETE FROM queue_backlog WHERE id = ?", (int(row_id),))
            conn.commit()
        finally:
            conn.close()

    def _atomic_json_write(self, path: str, payload: Any) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        last_error: Exception | None = None
        for attempt in range(6):
            tmp = f"{path}.tmp.{os.getpid()}.{attempt}"
            try:
                with open(tmp, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2)
                os.replace(tmp, path)
                return
            except PermissionError as exc:
                last_error = exc
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except Exception:
                    pass
                time.sleep(0.02 * (attempt + 1))
            except Exception as exc:
                last_error = exc
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except Exception:
                    pass
                break

        # Compatibility mirror is best-effort; queue truth remains in SQLite.
        if last_error is not None:
            return

    def _sync_legacy_json_from_db(self) -> None:
        """Compatibility mirror for legacy JSON queue observers and existing tests."""
        conn = self._connect_queue()
        try:
            cur = conn.cursor()
            cur.execute("SELECT payload FROM queue_backlog ORDER BY created_at ASC, id ASC")
            backlog_rows = cur.fetchall()
            backlog_payload = [self._row_payload_to_dict(str(row[0])) for row in backlog_rows]

            cur.execute(
                "SELECT claim_token, worker_id, claimed_at, retry_count, payload FROM queue_active ORDER BY claimed_at ASC, id ASC"
            )
            claim_rows = cur.fetchall()
            claims_payload: Dict[str, Dict[str, Any]] = {}
            for claim_token, worker_id, claimed_at, retry_count, payload in claim_rows:
                claims_payload[str(claim_token)] = {
                    "worker_id": str(worker_id),
                    "claimed_at": float(claimed_at),
                    "retry_count": int(retry_count),
                    "job": self._row_payload_to_dict(str(payload)),
                }

            cur.execute("SELECT payload FROM queue_results ORDER BY created_at ASC, id ASC")
            result_rows = cur.fetchall()
            result_payload = [self._row_payload_to_dict(str(row[0])) for row in result_rows]

            cur.execute("SELECT worker_id, heartbeat_epoch FROM queue_workers")
            worker_rows = cur.fetchall()
            workers_payload = {str(worker_id): float(ts) for worker_id, ts in worker_rows}

            cur.execute("SELECT key, value FROM queue_telemetry")
            telemetry_rows = cur.fetchall()
            telemetry_payload = {str(key): int(value) for key, value in telemetry_rows}
        finally:
            conn.close()

        self._atomic_json_write(self.queue_file, backlog_payload)
        self._atomic_json_write(self.claims_file, claims_payload)
        self._atomic_json_write(self.result_file, result_payload)
        self._atomic_json_write(self.workers_file, workers_payload)
        self._atomic_json_write(self.telemetry_file, telemetry_payload)

    def _resolve_ledger_db_path(self, fallback_dir: str) -> str:
        try:
            from config_utils import resolve_active_session_paths

            active = resolve_active_session_paths(require_exists=True)
            return str(active["db_file"])
        except Exception:
            return os.path.join(fallback_dir, "simulation_ledger.db")

    def _connect_queue(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.queue_db_path, timeout=30.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        return conn

    def _connect_ledger(self) -> sqlite3.Connection:
        initialize_ledger_schema(self.ledger_db_path)
        conn = sqlite3.connect(self.ledger_db_path, timeout=30.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        self._initialize_schema(conn)
        return conn

    def _row_payload_to_dict(self, payload_raw: str) -> Dict[str, Any]:
        payload = json.loads(payload_raw)
        if isinstance(payload, dict):
            return payload
        return {"raw_payload": payload}

    def _route_claim_to_dead_letter(
        self,
        claim_token: str,
        worker_id: str,
        retry_count: int,
        payload_dict: Dict[str, Any],
        reason: str,
    ) -> None:
        conn = self._connect_ledger()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO dead_letter_queue
                (claim_token, worker_id, retry_count, payload, failed_at, reason, job_id, config_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(claim_token),
                    str(worker_id),
                    int(retry_count),
                    json.dumps(payload_dict, sort_keys=True),
                    float(time.time()),
                    str(reason),
                    str(payload_dict.get("job_id") or ""),
                    str(payload_dict.get("config_hash") or ""),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def push_job(self, manifest_payload: Any) -> None:
        self.push_jobs_batch([manifest_payload])

    def push_jobs_batch(self, manifest_payloads: Sequence[Any]) -> int:
        if not manifest_payloads:
            return 0

        parsed_jobs: List[Dict[str, Any]] = [self._coerce_manifest_dict(p) for p in manifest_payloads]

        conn = self._connect_queue()
        try:
            cur = conn.cursor()
            now = time.time()
            cur.execute("BEGIN IMMEDIATE")
            for idx, payload in enumerate(parsed_jobs):
                payload_str = json.dumps(payload, sort_keys=True)
                cur.execute(
                    "INSERT INTO queue_backlog (job_id, config_hash, payload, created_at, retry_count) VALUES (?, ?, ?, ?, 0)",
                    (
                        str(payload.get("job_id") or ""),
                        str(payload.get("config_hash") or ""),
                        payload_str,
                        now + (idx * 1e-6),
                    ),
                )
            conn.commit()
            return len(parsed_jobs)
        finally:
            conn.close()

    def pop_job(self) -> Optional[Dict[str, Any]]:
        popped = self.pop_backlog_jobs(1)
        return popped[0] if popped else None

    def pop_backlog_jobs(self, count: int) -> List[Dict[str, Any]]:
        if count <= 0:
            return []

        conn = self._connect_queue()
        try:
            cur = conn.cursor()
            cur.execute("BEGIN EXCLUSIVE")
            cur.execute(
                "SELECT id, payload FROM queue_backlog ORDER BY created_at ASC, id ASC LIMIT ?",
                (int(count),),
            )
            rows = cur.fetchall()
            if not rows:
                conn.commit()
                return []

            ids = [int(row[0]) for row in rows]
            payloads = [self._row_payload_to_dict(str(row[1])) for row in rows]
            cur.executemany("DELETE FROM queue_backlog WHERE id = ?", [(row_id,) for row_id in ids])
            conn.commit()
            return payloads
        finally:
            conn.close()

    def claim_job(self, worker_id: str) -> Optional[Dict[str, Any]]:
        conn = self._connect_queue()
        try:
            cur = conn.cursor()
            cur.execute("BEGIN EXCLUSIVE")
            cur.execute(
                "SELECT id, payload, retry_count FROM queue_backlog ORDER BY created_at ASC, id ASC LIMIT 1"
            )
            row = cur.fetchone()
            if not row:
                conn.commit()
                return None

            backlog_id = int(row[0])
            payload_str = str(row[1])
            retry_count = int(row[2] or 0)
            payload_dict = self._row_payload_to_dict(payload_str)
            now_ts = time.time()
            payload_hash = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()[:8]
            claim_token = f"{worker_id}_{int(now_ts)}_{payload_hash}"

            cur.execute(
                """
                INSERT INTO queue_active
                (claim_token, job_id, config_hash, payload, worker_id, claimed_at, retry_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    claim_token,
                    str(payload_dict.get("job_id") or ""),
                    str(payload_dict.get("config_hash") or ""),
                    payload_str,
                    str(worker_id),
                    float(now_ts),
                    int(retry_count),
                ),
            )
            cur.execute("DELETE FROM queue_backlog WHERE id = ?", (backlog_id,))
            conn.commit()
            return {"token": claim_token, "payload": payload_str}
        finally:
            conn.close()

    def complete_job(self, claim_token: str, result_payload: Any | None = None) -> bool:
        conn = self._connect_queue()
        try:
            cur = conn.cursor()
            cur.execute("BEGIN IMMEDIATE")
            cur.execute("DELETE FROM queue_active WHERE claim_token = ?", (str(claim_token),))
            removed = cur.rowcount > 0
            if removed:
                cur.execute(
                    "UPDATE queue_telemetry SET value = value + 1 WHERE key = 'total_claims_processed'"
                )
            if removed and result_payload is not None:
                result_dict = self._coerce_result_dict(result_payload)
                cur.execute(
                    "INSERT INTO queue_results (payload, created_at) VALUES (?, ?)",
                    (json.dumps(result_dict, sort_keys=True), float(time.time())),
                )
            conn.commit()
            return removed
        finally:
            conn.close()

    def push_result(self, result_payload: Any) -> None:
        result_dict = self._coerce_result_dict(result_payload)
        conn = self._connect_queue()
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO queue_results (payload, created_at) VALUES (?, ?)",
                (json.dumps(result_dict, sort_keys=True), float(time.time())),
            )
            conn.commit()
        finally:
            conn.close()

    def get_results(self) -> List[Dict[str, Any]]:
        conn = self._connect_queue()
        try:
            cur = conn.cursor()
            cur.execute("BEGIN EXCLUSIVE")
            cur.execute("SELECT id, payload FROM queue_results ORDER BY created_at ASC, id ASC")
            rows = cur.fetchall()
            if not rows:
                conn.commit()
                return []

            ids = [int(row[0]) for row in rows]
            results = [self._row_payload_to_dict(str(row[1])) for row in rows]
            cur.executemany("DELETE FROM queue_results WHERE id = ?", [(row_id,) for row_id in ids])
            conn.commit()
            return results
        finally:
            conn.close()

    def set_worker_heartbeat(self, worker_id: str, timestamp: Optional[float] = None) -> None:
        conn = self._connect_queue()
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT OR REPLACE INTO queue_workers (worker_id, heartbeat_epoch) VALUES (?, ?)",
                (str(worker_id), float(timestamp if timestamp is not None else time.time())),
            )
            conn.commit()
        finally:
            conn.close()

    def clear_worker(self, worker_id: str) -> None:
        conn = self._connect_queue()
        try:
            cur = conn.cursor()
            cur.execute("BEGIN EXCLUSIVE")
            cur.execute(
                "SELECT claim_token, payload, claimed_at, retry_count FROM queue_active WHERE worker_id = ?",
                (str(worker_id),),
            )
            rows = cur.fetchall()
            for claim_token, payload_str, claimed_at, retry_count in rows:
                payload_dict = self._row_payload_to_dict(str(payload_str))
                cur.execute(
                    "INSERT INTO queue_backlog (job_id, config_hash, payload, created_at, retry_count) VALUES (?, ?, ?, ?, ?)",
                    (
                        str(payload_dict.get("job_id") or ""),
                        str(payload_dict.get("config_hash") or ""),
                        str(payload_str),
                        float(claimed_at) if claimed_at is not None else float(time.time()),
                        int(retry_count),
                    ),
                )
                cur.execute("DELETE FROM queue_active WHERE claim_token = ?", (str(claim_token),))

            cur.execute("DELETE FROM queue_workers WHERE worker_id = ?", (str(worker_id),))
            conn.commit()
        finally:
            conn.close()

    def peek_all(self) -> List[Dict[str, Any]]:
        conn = self._connect_queue()
        try:
            cur = conn.cursor()
            cur.execute("SELECT payload FROM queue_backlog ORDER BY created_at ASC, id ASC")
            rows = cur.fetchall()
            return [self._row_payload_to_dict(str(row[0])) for row in rows]
        finally:
            conn.close()

    def size(self) -> int:
        conn = self._connect_queue()
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM queue_backlog")
            row = cur.fetchone()
            return int(row[0]) if row and row[0] is not None else 0
        finally:
            conn.close()

    def get_worker_heartbeats(self) -> Dict[str, float]:
        conn = self._connect_queue()
        try:
            cur = conn.cursor()
            cur.execute("SELECT worker_id, heartbeat_epoch FROM queue_workers")
            return {str(worker_id): float(ts) for worker_id, ts in cur.fetchall()}
        finally:
            conn.close()

    def list_stale_workers(self, stale_after_seconds: float) -> List[str]:
        now = time.time()
        cutoff = now - float(stale_after_seconds)
        conn = self._connect_queue()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT worker_id FROM queue_workers WHERE heartbeat_epoch < ? ORDER BY worker_id ASC",
                (float(cutoff),),
            )
            return [str(row[0]) for row in cur.fetchall()]
        finally:
            conn.close()

    def list_active_workers(self, stale_after_seconds: float) -> List[str]:
        stale = set(self.list_stale_workers(stale_after_seconds))
        all_workers = self.get_worker_heartbeats().keys()
        return sorted([worker_id for worker_id in all_workers if worker_id not in stale])

    def _read_claims(self) -> Dict[str, Dict[str, Any]]:
        """Compatibility helper retained for existing tests."""
        conn = self._connect_queue()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT claim_token, worker_id, claimed_at, retry_count, payload FROM queue_active"
            )
            claims: Dict[str, Dict[str, Any]] = {}
            for claim_token, worker_id, claimed_at, retry_count, payload in cur.fetchall():
                claims[str(claim_token)] = {
                    "worker_id": str(worker_id),
                    "claimed_at": float(claimed_at),
                    "retry_count": int(retry_count),
                    "job": self._row_payload_to_dict(str(payload)),
                }
            return claims
        finally:
            conn.close()

    def get_claim_counts_by_worker(self) -> Dict[str, int]:
        conn = self._connect_queue()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT worker_id, COUNT(*) FROM queue_active GROUP BY worker_id"
            )
            return {str(worker_id): int(count) for worker_id, count in cur.fetchall()}
        finally:
            conn.close()

    def get_telemetry_counters(self) -> Dict[str, int]:
        conn = self._connect_queue()
        try:
            cur = conn.cursor()
            cur.execute("SELECT value FROM queue_telemetry WHERE key='total_claims_processed'")
            row = cur.fetchone()
            return {"total_claims_processed": int(row[0]) if row and row[0] is not None else 0}
        finally:
            conn.close()

    def recover_stale_workers(self, stale_after_seconds: float) -> List[str]:
        stale_workers = self.list_stale_workers(stale_after_seconds)
        if not stale_workers:
            return []

        conn = self._connect_queue()
        try:
            cur = conn.cursor()
            cur.execute("BEGIN EXCLUSIVE")
            now = time.time()

            for worker_id in stale_workers:
                cur.execute(
                    "SELECT claim_token, payload, claimed_at, retry_count FROM queue_active WHERE worker_id = ?",
                    (str(worker_id),),
                )
                rows = cur.fetchall()

                for claim_token, payload_str, claimed_at, retry_count in rows:
                    payload_dict = self._row_payload_to_dict(str(payload_str))
                    next_retry = int(retry_count) + 1

                    if next_retry > self.max_claim_retries:
                        self._route_claim_to_dead_letter(
                            claim_token=str(claim_token),
                            worker_id=str(worker_id),
                            retry_count=next_retry,
                            payload_dict=payload_dict,
                            reason="stale_worker_retry_exhausted",
                        )
                    else:
                        cur.execute(
                            "INSERT INTO queue_backlog (job_id, config_hash, payload, created_at, retry_count) VALUES (?, ?, ?, ?, ?)",
                            (
                                str(payload_dict.get("job_id") or ""),
                                str(payload_dict.get("config_hash") or ""),
                                str(payload_str),
                                float(now),
                                int(next_retry),
                            ),
                        )

                    cur.execute("DELETE FROM queue_active WHERE claim_token = ?", (str(claim_token),))

                cur.execute("DELETE FROM queue_workers WHERE worker_id = ?", (str(worker_id),))

            conn.commit()
            return sorted(stale_workers)
        finally:
            conn.close()
