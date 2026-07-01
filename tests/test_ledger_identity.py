"""
tests/test_ledger_identity.py

Ledger schema hardening (DC-v1.0): composite (config_hash, seed) primary key,
discriminator columns, and the opt-in legacy migration.  Pure sqlite3 — runs
anywhere.
"""
import os
import sqlite3
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from orchestrator.schema_utils import (  # noqa: E402
    initialize_ledger_schema,
    migrate_runs_to_composite_pk,
    runs_primary_key,
)

DISCRIMINATORS = [
    "seed", "run_id", "hunt_name", "utc_start", "solver_contract_version",
    "variant_label", "affect_topology", "affect_strength", "git_commit",
    "n_grid", "dt", "t_steps", "gpu_backend", "artifact_hash", "provenance_hash",
]


def _cols(conn, table):
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cur.fetchall()]


class TestFreshSchema:
    def test_runs_composite_pk(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "ledger.db")
            initialize_ledger_schema(db)
            assert runs_primary_key(db) == {"config_hash", "seed"}

    def test_runs_has_all_discriminators(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "ledger.db")
            initialize_ledger_schema(db)
            conn = sqlite3.connect(db)
            cols = _cols(conn, "runs")
            conn.close()
            for c in DISCRIMINATORS:
                assert c in cols, f"runs missing discriminator column {c}"

    def test_parameters_has_affect_coupling(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "ledger.db")
            initialize_ledger_schema(db)
            conn = sqlite3.connect(db)
            cols = _cols(conn, "parameters")
            conn.close()
            assert "param_affect_coupling" in cols
            assert "param_affect_topology" in cols

    def test_multi_seed_rows_are_distinct(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "ledger.db")
            initialize_ledger_schema(db)
            conn = sqlite3.connect(db)
            cur = conn.cursor()
            for seed in (1, 2, 3):
                cur.execute(
                    "INSERT OR REPLACE INTO runs (config_hash, seed, generation, status, fitness) VALUES (?, ?, ?, ?, ?)",
                    ("HASH", seed, 0, "SUCCESS", 0.5),
                )
            conn.commit()
            cur.execute("SELECT COUNT(*) FROM runs WHERE config_hash='HASH'")
            n = cur.fetchone()[0]
            conn.close()
            assert n == 3, "multi-seed runs of the same config must be 3 distinct rows"

    def test_same_config_same_seed_replaces(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "ledger.db")
            initialize_ledger_schema(db)
            conn = sqlite3.connect(db)
            cur = conn.cursor()
            cur.execute("INSERT OR REPLACE INTO runs (config_hash, seed, fitness) VALUES ('H', 1, 9.0)")
            cur.execute("INSERT OR REPLACE INTO runs (config_hash, seed, fitness) VALUES ('H', 1, 0.1)")
            conn.commit()
            cur.execute("SELECT COUNT(*), MIN(fitness) FROM runs WHERE config_hash='H'")
            count, fit = cur.fetchone()
            conn.close()
            assert count == 1 and abs(fit - 0.1) < 1e-9, "re-run with same seed must replace in place"


class TestLegacyMigration:
    def _make_legacy(self, db):
        """Create a pre-DC-v1.0 runs table keyed on config_hash only, with rows."""
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE runs (
                config_hash TEXT PRIMARY KEY,
                generation INTEGER,
                status TEXT,
                fitness REAL,
                origin TEXT DEFAULT 'NATURAL'
            )
            """
        )
        cur.executemany(
            "INSERT INTO runs (config_hash, generation, status, fitness) VALUES (?, ?, ?, ?)",
            [("A", 0, "SUCCESS", 0.4), ("B", 1, "FAIL", 999.0)],
        )
        conn.commit()
        conn.close()

    def test_legacy_pk_detected(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "ledger.db")
            self._make_legacy(db)
            assert runs_primary_key(db) == {"config_hash"}

    def test_migration_changes_pk_and_preserves_rows(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "ledger.db")
            self._make_legacy(db)
            ok = migrate_runs_to_composite_pk(db)
            assert ok is True
            assert runs_primary_key(db) == {"config_hash", "seed"}
            conn = sqlite3.connect(db)
            cur = conn.cursor()
            cur.execute("SELECT config_hash, seed, fitness FROM runs ORDER BY config_hash")
            rows = cur.fetchall()
            conn.close()
            assert rows == [("A", 0, 0.4), ("B", 0, 999.0)], rows
            # legacy temp table must be cleaned up
            conn = sqlite3.connect(db)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {r[0] for r in cur.fetchall()}
            conn.close()
            assert "runs_premigration" not in tables

    def test_migration_is_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "ledger.db")
            self._make_legacy(db)
            assert migrate_runs_to_composite_pk(db) is True
            # running again is a no-op
            assert migrate_runs_to_composite_pk(db) is True
            assert runs_primary_key(db) == {"config_hash", "seed"}

    def test_post_migration_multi_seed_works(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "ledger.db")
            self._make_legacy(db)
            migrate_runs_to_composite_pk(db)
            conn = sqlite3.connect(db)
            cur = conn.cursor()
            cur.execute("INSERT OR REPLACE INTO runs (config_hash, seed, fitness) VALUES ('A', 5, 0.2)")
            conn.commit()
            cur.execute("SELECT COUNT(*) FROM runs WHERE config_hash='A'")
            n = cur.fetchone()[0]
            conn.close()
            assert n == 2, "after migration A should have seed 0 and seed 5 rows"
