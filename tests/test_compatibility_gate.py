"""
tests/test_compatibility_gate.py

The champion compatibility gate (result_processor): incompatible solver
contract / variant / affect topology / grid runs are recorded in the ledger but
excluded from champion comparison.  Pure sqlite3 + run_identity — runs anywhere.
"""
import os
import sqlite3
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# result_processor imports `requests` at module top; skip cleanly if unavailable.
ResultProcessor = pytest.importorskip("orchestrator.result_processor").ResultProcessor
from orchestrator.contracts import JobResult  # noqa: E402
from orchestrator.schema_utils import initialize_ledger_schema  # noqa: E402


LOCAL_RHO = "IRER-SNCGL-LOCAL-RHO-ETDRK4-v1"
CAUSAL = "IRER-SNCGL-CAUSAL-AFFECT-ETDRK4-v1"


def _processor(db):
    return ResultProcessor({
        "db_path": db,
        "data_dir": os.path.dirname(db),
        "provenance_dir": os.path.dirname(db),
    })


def _insert_champion(db, *, contract=LOCAL_RHO, variant="LOCAL-RHO",
                     topology="none", n_grid=64, fitness=0.5, eligible=1):
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT OR REPLACE INTO runs "
        "(config_hash, seed, status, fitness, solver_contract_version, variant_label, affect_topology, n_grid, champion_eligible) "
        "VALUES (?, ?, 'SUCCESS', ?, ?, ?, ?, ?, ?)",
        ("champ", 0, fitness, contract, variant, topology, n_grid, eligible),
    )
    conn.commit()
    conn.close()


def _identity(*, contract=LOCAL_RHO, variant="LOCAL-RHO", topology="none", n_grid=64, seed=1):
    return {
        "seed": seed,
        "solver_contract_version": contract,
        "variant_label": variant,
        "affect_topology": topology,
        "n_grid": n_grid,
    }


def _result(config_hash="incoming"):
    return JobResult(
        job_id="job123", generation=1, config_hash=config_hash,
        artifact_url="/nonexistent.h5", status="SUCCESS", params_path="x.json",
    )


class TestGateLogic:
    def test_first_run_is_eligible_no_champion(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "ledger.db")
            initialize_ledger_schema(db)
            gate = _processor(db)._evaluate_compatibility_gate(_result(), _identity())
            assert gate["champion_eligible"] is True
            assert gate["reference_key"] is None

    def test_matching_variant_is_eligible(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "ledger.db")
            initialize_ledger_schema(db)
            _insert_champion(db)
            gate = _processor(db)._evaluate_compatibility_gate(_result(), _identity())
            assert gate["champion_eligible"] is True

    def test_different_contract_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "ledger.db")
            initialize_ledger_schema(db)
            _insert_champion(db)  # LOCAL-RHO champion
            gate = _processor(db)._evaluate_compatibility_gate(
                _result(), _identity(contract=CAUSAL, variant="CAUSAL-AFFECT", topology="vacuum_ref")
            )
            assert gate["champion_eligible"] is False
            assert "incompatible" in gate["reason"]

    def test_different_grid_blocked_same_contract(self):
        # The subtle case are_rankable closes: same LOCAL-RHO contract, different N_grid.
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "ledger.db")
            initialize_ledger_schema(db)
            _insert_champion(db, n_grid=64)
            gate = _processor(db)._evaluate_compatibility_gate(_result(), _identity(n_grid=128))
            assert gate["champion_eligible"] is False

    def test_run_does_not_gate_against_itself(self):
        # A run already in the ledger must not be its own champion reference.
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "ledger.db")
            initialize_ledger_schema(db)
            _insert_champion(db, n_grid=128)  # only row is config_hash='champ'
            # incoming IS 'champ' at a different grid; excluding itself => no reference => eligible
            gate = _processor(db)._evaluate_compatibility_gate(
                _result(config_hash="champ"), _identity(n_grid=64, seed=0)
            )
            assert gate["champion_eligible"] is True


class TestGatePersistence:
    def test_incompatible_run_recorded_as_ineligible(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "ledger.db")
            initialize_ledger_schema(db)
            proc = _processor(db)
            ident = _identity(contract=CAUSAL, variant="CAUSAL-AFFECT", topology="vacuum_ref")
            proc._write_worker_result_to_ledger(
                _result(), validation_result={"log_prime_sse": 0.3, "provenance": {}},
                identity_fields=ident, champion_eligible=False,
            )
            conn = sqlite3.connect(db)
            row = conn.execute(
                "SELECT champion_eligible, variant_label, n_grid FROM runs WHERE config_hash='incoming'"
            ).fetchone()
            conn.close()
            assert row is not None
            assert row[0] == 0, "incompatible run must be recorded with champion_eligible=0"
            assert row[1] == "CAUSAL-AFFECT"

    def test_ineligible_champion_not_used_as_reference(self):
        # An incompatible run already in the ledger (eligible=0) must not become the reference.
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "ledger.db")
            initialize_ledger_schema(db)
            _insert_champion(db, contract=LOCAL_RHO, variant="LOCAL-RHO", n_grid=64, fitness=0.5, eligible=1)
            # Insert a *better* SSE run of a foreign variant but marked ineligible.
            conn = sqlite3.connect(db)
            conn.execute(
                "INSERT OR REPLACE INTO runs (config_hash, seed, status, fitness, solver_contract_version, variant_label, affect_topology, n_grid, champion_eligible) "
                "VALUES ('foreign', 0, 'SUCCESS', 0.01, ?, 'CAUSAL-AFFECT', 'vacuum_ref', 64, 0)",
                (CAUSAL,),
            )
            conn.commit()
            conn.close()
            # A LOCAL-RHO/64 incoming should match the eligible champion, not the better foreign run.
            gate = _processor(db)._evaluate_compatibility_gate(_result(), _identity())
            assert gate["champion_eligible"] is True
            assert gate["reference_key"][0] == LOCAL_RHO
