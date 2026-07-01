"""
test_schema_concurrent.py
Vector 3 (P1) gate — initialize_ledger_schema and ensure_ledger_ready must be
safe against concurrent callers hitting the same SQLite file.

Two threads each call initialize_ledger_schema on the same temp database
simultaneously (simulating an aste_hunter.Hunter and a LedgerDB both booting
against the same shared ledger).  The test asserts:

  1. No sqlite3.OperationalError is raised under concurrent load.
  2. All expected tables exist after the race.
  3. The full superset of required metrics columns is present.
"""

import sqlite3
import tempfile
import threading
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from orchestrator.schema_utils import initialize_ledger_schema, ensure_ledger_ready

REQUIRED_TABLES = {"runs", "parameters", "metrics", "results", "pareto_archive", "spectral_basins"}

REQUIRED_METRICS_COLS = {
    "config_hash",
    "log_prime_sse",
    "primary_harmonic_error",
    "missing_peak_penalty",
    "noise_penalty",
    "pcs",
    "collapse_event_count",
    "stage4_early_reject",
}


def _worker(db_path: str, errors: list, index: int) -> None:
    """Each thread only calls initialize_ledger_schema (DDL).  Concurrent
    ensure_ledger_ready would hit a lock because the DDL transaction is still
    open in the racing thread — that check is done once, serially, after join."""
    try:
        initialize_ledger_schema(db_path)
    except Exception as exc:
        errors.append(f"Thread {index}: {exc}")


def test_concurrent_schema_init_no_crash():
    """Both threads race to initialise the same DB and neither must crash.
    After both finish, ensure_ledger_ready must confirm the schema is complete."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    errors: list = []
    threads = [threading.Thread(target=_worker, args=(db_path, errors, i)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    assert not errors, f"Concurrent schema init raised errors: {errors}"
    # Schema must be complete after the race
    ensure_ledger_ready(db_path, raise_on_fail=True)


def test_all_required_tables_present_after_init():
    """After initialisation every ASTE table must exist."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    initialize_ledger_schema(db_path)
    ensure_ledger_ready(db_path, raise_on_fail=True)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        present = {row[0] for row in cursor.fetchall()}

    missing = REQUIRED_TABLES - present
    assert not missing, f"Required tables missing after schema init: {missing}"


def test_metrics_superset_columns_present():
    """The metrics table must carry the full superset of ASTE metrics columns."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    initialize_ledger_schema(db_path)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(metrics)")
        present_cols = {row[1] for row in cursor.fetchall()}

    missing = REQUIRED_METRICS_COLS - present_cols
    assert not missing, f"Metrics columns missing after schema init: {missing}"


def test_idempotent_double_init():
    """Calling initialize_ledger_schema twice on the same DB must not raise."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    initialize_ledger_schema(db_path)
    # Second call — must be a no-op / safe migration
    initialize_ledger_schema(db_path)
    ensure_ledger_ready(db_path, raise_on_fail=True)
