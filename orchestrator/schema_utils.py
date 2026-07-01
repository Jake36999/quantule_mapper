"""Schema health check utilities for orchestrator and worker coordination."""
import json
import os
import sqlite3
import time
import random
from typing import Dict, Set, Optional, Iterable, Tuple


# DC-v1.0: the `runs` table is keyed on the composite (config_hash, seed) so that
# multi-seed runs of the same parameter set are distinct ledger rows instead of
# collapsing under INSERT OR REPLACE.  Discriminator columns carry the full run
# identity (see orchestrator/run_identity.py and docs/DATA_CONTRACT.md).  This
# DDL is the single source of truth, reused by initialize_ledger_schema() and
# migrate_runs_to_composite_pk().
_RUNS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    config_hash TEXT NOT NULL,
    seed INTEGER NOT NULL DEFAULT 0,
    generation INTEGER,
    status TEXT,
    fitness REAL,
    origin TEXT DEFAULT 'NATURAL',
    parent_1 TEXT,
    parent_2 TEXT,
    staged_path TEXT,
    staged_at TEXT,
    staged_config_hash TEXT,
    run_id TEXT,
    hunt_name TEXT,
    utc_start TEXT,
    solver_contract_version TEXT,
    variant_label TEXT,
    affect_topology TEXT,
    affect_strength REAL,
    git_commit TEXT,
    n_grid INTEGER,
    dt REAL,
    t_steps INTEGER,
    gpu_backend TEXT,
    artifact_hash TEXT,
    provenance_hash TEXT,
    champion_eligible INTEGER DEFAULT 1,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (config_hash, seed)
)
"""


def _safe_add_columns(cursor: sqlite3.Cursor, table_name: str, columns: Iterable[Tuple[str, str]]) -> None:
    cursor.execute(f"PRAGMA table_info({table_name})")
    existing = {row[1] for row in cursor.fetchall()}
    for column_name, column_spec in columns:
        if column_name not in existing:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_spec}")


def get_column_set(db_path: str, table_name: str) -> Set[str]:
    """Get set of column names in a table; return empty set if table doesn't exist."""
    try:
        conn = sqlite3.connect(db_path, timeout=5.0, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()
        return columns
    except sqlite3.OperationalError:
        return set()


def ensure_ledger_ready(db_path: str, raise_on_fail: bool = False) -> bool:
    """
    Verify ledger has baseline schema (runs, parameters, metrics tables).
    Returns True if schema is ready, False otherwise.
    
    If raise_on_fail is True, raises RuntimeError instead of returning False.
    
    NOTE: This only checks for schema existence. Use initialize_ledger_schema()
    to create schema if missing.
    """
    try:
        conn = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        required = {"runs", "parameters", "metrics", "results", "pareto_archive", "spectral_basins"}
        
        if not required.issubset(tables):
            missing = required - tables
            msg = f"Ledger schema incomplete. Missing tables: {missing}"
            conn.close()
            if raise_on_fail:
                raise RuntimeError(msg)
            return False
        
        conn.close()
        return True
    except Exception as e:
        if raise_on_fail:
            raise RuntimeError(f"Ledger health check failed: {e}")
        return False


def initialize_ledger_schema(db_path: str, _retries: int = 3) -> bool:
    """
    Initialize simulation ledger schema if it doesn't exist.
    Creates and migrates all ASTE tables used by orchestrator and hunter modules.
    Returns True if successful, raises RuntimeError on failure.
    Retries up to 3 times on 'database is locked' (safe under concurrent init).
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        cursor = conn.cursor()
        
        # Create runs table (composite PK (config_hash, seed); see _RUNS_TABLE_DDL)
        cursor.execute(_RUNS_TABLE_DDL)
        
        # Create parameters table
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS parameters (
                config_hash TEXT PRIMARY KEY,
                param_D REAL,
                param_eta REAL,
                param_rho_vac REAL,
                param_a_coupling REAL,
                param_splash_coupling REAL,
                param_splash_fraction REAL
            )
            '''
        )
        
        # Create metrics table (superset schema)
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS metrics (
                config_hash TEXT PRIMARY KEY,
                log_prime_sse REAL,
                bragg_peaks_detected INTEGER,
                n_bragg_peaks INTEGER,
                bragg_prime_sse REAL,
                collapse_event_count INTEGER,
                pcs REAL,
                primary_harmonic_error REAL,
                missing_peak_penalty REAL,
                noise_penalty REAL,
                sse_null_phase_scramble REAL,
                sse_null_target_shuffle REAL,
                pli REAL,
                ic REAL,
                c4_contrast REAL,
                ablated_c4_contrast REAL,
                j_info_mean REAL,
                grad_phase_var REAL,
                max_amp_peak REAL,
                clamp_fraction_mean REAL,
                omega_sat_mean REAL,
                dominant_peak_k REAL,
                secondary_peak_k REAL,
                stage4_early_reject INTEGER DEFAULT 0,
                basin_id INTEGER DEFAULT -1,
                origin TEXT DEFAULT 'NATURAL'
            )
            '''
        )

        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS results (
                config_hash TEXT PRIMARY KEY,
                generation INTEGER,
                param_D REAL,
                param_eta REAL,
                param_rho_vac REAL,
                param_a_coupling REAL,
                param_splash_coupling REAL,
                param_splash_fraction REAL,
                log_prime_sse REAL,
                fitness REAL,
                parent_1 TEXT,
                parent_2 TEXT
            )
            '''
        )

        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS pareto_archive (
                config_hash TEXT PRIMARY KEY,
                generation INTEGER,
                log_prime_sse REAL,
                fitness REAL
            )
            '''
        )

        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS spectral_basins (
                basin_id INTEGER PRIMARY KEY AUTOINCREMENT,
                k1_mean REAL,
                k2_mean REAL,
                pcs_mean REAL,
                best_error REAL,
                population_size INTEGER,
                generation_first_seen INTEGER,
                generation_last_seen INTEGER,
                stagnant_generations INTEGER DEFAULT 0
            )
            '''
        )

        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS dead_letter_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_token TEXT,
                worker_id TEXT,
                retry_count INTEGER DEFAULT 0,
                payload TEXT NOT NULL,
                failed_at REAL,
                reason TEXT,
                job_id TEXT,
                config_hash TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            '''
        )

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_generation ON runs(generation)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_sse ON metrics(log_prime_sse ASC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_generation ON runs(generation DESC)")
        
        _safe_add_columns(
            cursor,
            "runs",
            (
                ("origin", "TEXT DEFAULT 'NATURAL'"),
                ("timestamp", "DATETIME"),
                ("parent_1", "TEXT"),
                ("parent_2", "TEXT"),
                ("staged_path", "TEXT"),
                ("staged_at", "TEXT"),
                ("staged_config_hash", "TEXT"),
                ("status", "TEXT"), # <-- ADDED BY PATCH
                ("fitness", "REAL"), # <-- ADDED BY PATCH
                # DC-v1.0 discriminator columns (idempotent for pre-existing DBs).
                # NOTE: ALTER TABLE cannot change the PRIMARY KEY; existing DBs keep
                # a config_hash-only PK until migrate_runs_to_composite_pk() is run.
                ("seed", "INTEGER DEFAULT 0"),
                ("run_id", "TEXT"),
                ("hunt_name", "TEXT"),
                ("utc_start", "TEXT"),
                ("solver_contract_version", "TEXT"),
                ("variant_label", "TEXT"),
                ("affect_topology", "TEXT"),
                ("affect_strength", "REAL"),
                ("git_commit", "TEXT"),
                ("n_grid", "INTEGER"),
                ("dt", "REAL"),
                ("t_steps", "INTEGER"),
                ("gpu_backend", "TEXT"),
                ("artifact_hash", "TEXT"),
                ("provenance_hash", "TEXT"),
                ("champion_eligible", "INTEGER DEFAULT 1"),
            ),
        )

        _safe_add_columns(
            cursor,
            "parameters",
            (
                ("param_D", "REAL"),
                ("param_eta", "REAL"),
                ("param_a", "REAL"),
                ("param_rho_vac", "REAL"),
                ("param_omega0", "REAL"),
                ("param_affect_coupling", "REAL DEFAULT 0.0"),
                ("param_affect_topology", "TEXT DEFAULT 'none'"),
            ),
        )

        _safe_add_columns(
            cursor,
            "metrics",
            (
                ("bragg_peaks_detected", "INTEGER"),
                ("n_bragg_peaks", "INTEGER"),
                ("bragg_prime_sse", "REAL"),
                ("collapse_event_count", "INTEGER DEFAULT 0"),
                ("pcs", "REAL"),
                ("primary_harmonic_error", "REAL DEFAULT 999.0"),
                ("missing_peak_penalty", "REAL DEFAULT 0.0"),
                ("noise_penalty", "REAL DEFAULT 0.0"),
                ("sse_null_phase_scramble", "REAL"),
                ("sse_null_target_shuffle", "REAL"),
                ("pli", "REAL"),
                ("ic", "REAL"),
                ("c4_contrast", "REAL"),
                ("ablated_c4_contrast", "REAL"),
                ("j_info_mean", "REAL"),
                ("grad_phase_var", "REAL"),
                ("max_amp_peak", "REAL"),
                ("clamp_fraction_mean", "REAL"),
                ("omega_sat_mean", "REAL"),
                ("dominant_peak_k", "REAL DEFAULT 0.0"),
                ("secondary_peak_k", "REAL DEFAULT 0.0"),
                ("stage4_early_reject", "INTEGER DEFAULT 0"),
                ("basin_id", "INTEGER DEFAULT -1"),
                ("origin", "TEXT DEFAULT 'NATURAL'"),
                ("fitness", "REAL DEFAULT 0.0"),
                ("quantum_falsifiability", "REAL"),
                ("refinement_status", "TEXT DEFAULT NULL"),
                ("solver_contract_version", "TEXT DEFAULT NULL"),
            ),
        )

        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS refinement_variants (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                baseline_config_hash TEXT NOT NULL,
                variant_label TEXT NOT NULL,
                variant_config_hash TEXT,
                status TEXT DEFAULT 'PENDING',
                log_prime_sse REAL,
                dominant_peak_k REAL,
                sse_directional_min REAL,
                audit_verdict TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                completed_at DATETIME DEFAULT NULL
            )
            '''
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_refinement_baseline ON refinement_variants(baseline_config_hash)"
        )

        conn.commit()
        conn.close()
        return True
    except sqlite3.OperationalError as e:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        if "database is locked" in str(e).lower() and _retries > 0:
            sleep_s = 0.2 * (2 ** (3 - _retries)) + random.uniform(0.0, 0.1)
            time.sleep(sleep_s)
            return initialize_ledger_schema(db_path, _retries=_retries - 1)
        raise RuntimeError(f"Failed to initialize ledger schema: {e}")
    except Exception as e:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        raise RuntimeError(f"Failed to initialize ledger schema: {e}")


def runs_primary_key(db_path: str) -> Set[str]:
    """Return the set of columns forming the `runs` PRIMARY KEY (empty if no table)."""
    try:
        conn = sqlite3.connect(db_path, timeout=5.0, check_same_thread=False)
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(runs)")
        pk = {row[1] for row in cur.fetchall() if row[5] > 0}
        conn.close()
        return pk
    except sqlite3.OperationalError:
        return set()


def migrate_runs_to_composite_pk(db_path: str) -> bool:
    """
    Opt-in migration: rebuild an existing `runs` table so its PRIMARY KEY is
    (config_hash, seed) instead of config_hash alone, preserving all rows.

    Pre-DC-v1.0 ledgers keyed `runs` on config_hash only, which collapses
    multi-seed runs (same params, different seed) into a single row under
    INSERT OR REPLACE.  This migration is deliberately **not** automatic: run it
    once against a live ledger before launching multi-seed hunts.  It is a no-op
    when the composite PK is already present.

    Returns True if the table now has the composite PK, False if there is no
    `runs` table to migrate.
    """
    if not os.path.exists(db_path):
        return False
    # Ensure all discriminator columns exist first (also creates runs if absent).
    initialize_ledger_schema(db_path)
    if runs_primary_key(db_path) == {"config_hash", "seed"}:
        return True

    conn = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)
    try:
        conn.execute("PRAGMA busy_timeout=30000;")
        conn.execute("PRAGMA foreign_keys=OFF;")
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(runs)")
        old_cols = [row[1] for row in cur.fetchall()]
        if not old_cols:
            return False

        cur.execute("BEGIN IMMEDIATE")
        cur.execute("DROP TABLE IF EXISTS runs_premigration")
        cur.execute("ALTER TABLE runs RENAME TO runs_premigration")
        cur.execute(_RUNS_TABLE_DDL)  # fresh table with composite PK
        cur.execute("PRAGMA table_info(runs)")
        new_cols = [row[1] for row in cur.fetchall()]
        shared = [c for c in new_cols if c in old_cols]
        shared_list = ", ".join(shared)
        # COALESCE seed to 0 so the NOT NULL composite PK is satisfied for
        # legacy rows that predate the seed column.
        select_list = ", ".join(
            ("COALESCE(seed, 0)" if c == "seed" else c) for c in shared
        )
        cur.execute(
            f"INSERT OR IGNORE INTO runs ({shared_list}) "
            f"SELECT {select_list} FROM runs_premigration"
        )
        cur.execute("DROP TABLE runs_premigration")
        conn.commit()
        cursor_pk = runs_primary_key(db_path)
        return cursor_pk == {"config_hash", "seed"}
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        conn.close()


def initialize_queue_schema(queue_db_path: str, _retries: int = 3) -> bool:
    """
    Initialize queue runtime schema in a dedicated SQLite database.
    Returns True if successful, raises RuntimeError on failure.
    """
    conn = None
    try:
        os.makedirs(os.path.dirname(os.path.abspath(queue_db_path)) or ".", exist_ok=True)
        conn = sqlite3.connect(queue_db_path, timeout=30.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        cursor = conn.cursor()

        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS queue_backlog (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT,
                config_hash TEXT,
                payload TEXT NOT NULL,
                created_at REAL NOT NULL,
                retry_count INTEGER NOT NULL DEFAULT 0
            )
            '''
        )
        _safe_add_columns(
            cursor,
            "queue_backlog",
            (("retry_count", "INTEGER NOT NULL DEFAULT 0"),),
        )

        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS queue_active (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_token TEXT UNIQUE NOT NULL,
                job_id TEXT,
                config_hash TEXT,
                payload TEXT NOT NULL,
                worker_id TEXT NOT NULL,
                claimed_at REAL NOT NULL,
                retry_count INTEGER NOT NULL DEFAULT 0
            )
            '''
        )

        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS queue_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                payload TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            '''
        )

        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS queue_workers (
                worker_id TEXT PRIMARY KEY,
                heartbeat_epoch REAL NOT NULL
            )
            '''
        )

        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS queue_telemetry (
                key TEXT PRIMARY KEY,
                value INTEGER NOT NULL
            )
            '''
        )
        cursor.execute(
            "INSERT OR IGNORE INTO queue_telemetry (key, value) VALUES ('total_claims_processed', 0)"
        )
        cursor.execute(
            "INSERT OR IGNORE INTO queue_telemetry (key, value) VALUES ('json_import_done', 0)"
        )

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_queue_backlog_created_at ON queue_backlog(created_at, id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_queue_active_worker ON queue_active(worker_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_queue_active_claimed_at ON queue_active(claimed_at)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_queue_results_created_at ON queue_results(created_at, id)"
        )
        

        conn.commit()
        conn.close()
        return True
    except sqlite3.OperationalError as e:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        if "database is locked" in str(e).lower() and _retries > 0:
            sleep_s = 0.2 * (2 ** (3 - _retries)) + random.uniform(0.0, 0.1)
            time.sleep(sleep_s)
            return initialize_queue_schema(queue_db_path, _retries=_retries - 1)
        raise RuntimeError(f"Failed to initialize queue schema: {e}")
    except Exception as e:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        raise RuntimeError(f"Failed to initialize queue schema: {e}")


def migrate_json_queue_to_sqlite(
    queue_db_path: str,
    queue_file: str,
    claims_file: str,
    result_file: str,
    workers_file: str,
    telemetry_file: str,
) -> Dict[str, int]:
    """
    One-way, idempotent bootstrap migration from legacy JSON queue files to SQLite queue tables.
    Returns a summary of imported row counts.
    """
    initialize_queue_schema(queue_db_path)

    summary = {
        "backlog_imported": 0,
        "active_imported": 0,
        "results_imported": 0,
        "workers_imported": 0,
        "telemetry_imported": 0,
    }

    conn = sqlite3.connect(queue_db_path, timeout=30.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=30000;")

    try:
        cur = conn.cursor()
        cur.execute("SELECT value FROM queue_telemetry WHERE key='json_import_done'")
        row = cur.fetchone()
        if row and int(row[0]) == 1:
            return summary

        now = time.time()

        def _read_json(path: str, default):
            if not os.path.exists(path):
                return default
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    return json.load(handle)
            except Exception:
                return default

        backlog = _read_json(queue_file, [])
        if isinstance(backlog, list):
            for payload in backlog:
                try:
                    payload_str = json.dumps(payload, sort_keys=True)
                    job_id = str(payload.get("job_id") or "") if isinstance(payload, dict) else ""
                    config_hash = str(payload.get("config_hash") or "") if isinstance(payload, dict) else ""
                    cur.execute(
                        "INSERT INTO queue_backlog (job_id, config_hash, payload, created_at, retry_count) VALUES (?, ?, ?, ?, 0)",
                        (job_id, config_hash, payload_str, now),
                    )
                    summary["backlog_imported"] += 1
                except Exception:
                    continue

        claims = _read_json(claims_file, {})
        if isinstance(claims, dict):
            for claim_token, claim_data in claims.items():
                if not isinstance(claim_data, dict):
                    continue
                payload = claim_data.get("job", {})
                try:
                    payload_str = json.dumps(payload, sort_keys=True)
                    cur.execute(
                        "INSERT OR IGNORE INTO queue_active (claim_token, job_id, config_hash, payload, worker_id, claimed_at, retry_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            str(claim_token),
                            str((payload or {}).get("job_id") or ""),
                            str((payload or {}).get("config_hash") or ""),
                            payload_str,
                            str(claim_data.get("worker_id") or "unknown"),
                            float(claim_data.get("claimed_at") or now),
                            int(claim_data.get("retry_count") or 0),
                        ),
                    )
                    summary["active_imported"] += 1
                except Exception:
                    continue

        results = _read_json(result_file, [])
        if isinstance(results, list):
            for payload in results:
                try:
                    payload_str = json.dumps(payload, sort_keys=True)
                    cur.execute(
                        "INSERT INTO queue_results (payload, created_at) VALUES (?, ?)",
                        (payload_str, now),
                    )
                    summary["results_imported"] += 1
                except Exception:
                    continue

        workers = _read_json(workers_file, {})
        if isinstance(workers, dict):
            for worker_id, heartbeat in workers.items():
                try:
                    cur.execute(
                        "INSERT OR REPLACE INTO queue_workers (worker_id, heartbeat_epoch) VALUES (?, ?)",
                        (str(worker_id), float(heartbeat)),
                    )
                    summary["workers_imported"] += 1
                except Exception:
                    continue

        telemetry = _read_json(telemetry_file, {})
        if isinstance(telemetry, dict):
            try:
                total_claims = int(telemetry.get("total_claims_processed", 0))
                cur.execute(
                    "INSERT OR REPLACE INTO queue_telemetry (key, value) VALUES ('total_claims_processed', ?)",
                    (total_claims,),
                )
                summary["telemetry_imported"] = 1
            except Exception:
                pass

        cur.execute("INSERT OR REPLACE INTO queue_telemetry (key, value) VALUES ('json_import_done', 1)")
        conn.commit()
        return summary
    finally:
        conn.close()
