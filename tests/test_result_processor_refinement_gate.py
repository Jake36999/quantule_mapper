"""
tests/test_result_processor_refinement_gate.py

Tests that result_processor correctly stamps _refinement_status and
_solver_contract_version based on solver_contract presence in provenance.
"""
import json
import os
import sqlite3
import tempfile
import pytest
from unittest.mock import MagicMock, patch

from orchestrator.contracts import SOLVER_CONTRACT_VERSION, PROVISIONAL_SSE_THRESHOLD
from orchestrator.result_processor import ResultProcessor


_GOOD_CONTRACT = {
    "solver_contract_version": SOLVER_CONTRACT_VERSION,
    "geometry_source": "local_stage_rho",
    "auxiliary_geometry": False,
    "topology_cap_in_simulation": False,
    "linear_operator": "-D*k^2 - eta + i*rho_vac",
}


def _make_processor():
    cfg = {
        "db_path": ":memory:",
        "data_dir": "/tmp",
        "provenance_dir": "/tmp",
    }
    return ResultProcessor(cfg)


def _make_provenance(sse: float, contract=None) -> str:
    payload = {
        "spectral_fidelity": {"log_prime_sse": sse},
        "solver_contract": contract,
    }
    return json.dumps(payload)


class TestRefinementStatusStamping:
    def _run_process_result(self, provenance: dict, db_path: str):
        processor = ResultProcessor({
            "db_path": db_path,
            "data_dir": "/tmp",
            "provenance_dir": "/tmp",
        })
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as pf:
            json.dump(provenance, pf)
            prov_path = pf.name

        try:
            result_data = {
                "job_id": "test-job-001",
                "generation": 1,
                "config_hash": "abc123",
                "artifact_url": "/nonexistent/path.h5",
                "status": "SUCCESS",
                "provenance_path": prov_path,
                "config": {},
            }
            with patch.object(processor, "_store_result"), patch.object(processor, "_trigger_visual_observer_async"):
                assert processor.process_result(result_data)
            return db_path
        finally:
            os.unlink(prov_path)

    def _read_metrics(self, db_path: str):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT refinement_status, solver_contract_version FROM metrics WHERE config_hash = ?",
            ("abc123",),
        )
        row = cursor.fetchone()
        conn.close()
        return row

    def test_uncertified_when_contract_absent_persists_metrics(self, tmp_path):
        db_path = str(tmp_path / "ledger.db")
        provenance = {"spectral_fidelity": {"log_prime_sse": 0.5}, "solver_contract": None}
        self._run_process_result(provenance, db_path)
        row = self._read_metrics(db_path)

        assert row == ("PHYSICS_UNCERTIFIED", "")

    def test_uncertified_when_contract_wrong_version_persists_metrics(self, tmp_path):
        db_path = str(tmp_path / "ledger.db")
        bad_contract = dict(_GOOD_CONTRACT)
        bad_contract["solver_contract_version"] = "OLD-VERSION-v0"
        provenance = {"spectral_fidelity": {"log_prime_sse": 0.5}, "solver_contract": bad_contract}
        self._run_process_result(provenance, db_path)
        row = self._read_metrics(db_path)

        assert row == ("PHYSICS_UNCERTIFIED", "OLD-VERSION-v0")

    def test_validated_provisional_persists_metrics(self, tmp_path):
        db_path = str(tmp_path / "ledger.db")
        provenance = {"spectral_fidelity": {"log_prime_sse": 0.5}, "solver_contract": _GOOD_CONTRACT}
        self._run_process_result(provenance, db_path)
        row = self._read_metrics(db_path)

        assert row == ("VALIDATED_PROVISIONAL", SOLVER_CONTRACT_VERSION)

    def test_certified_non_provisional_persists_contract_only(self, tmp_path):
        db_path = str(tmp_path / "ledger.db")
        provenance = {"spectral_fidelity": {"log_prime_sse": 2.5}, "solver_contract": _GOOD_CONTRACT}
        self._run_process_result(provenance, db_path)
        row = self._read_metrics(db_path)

        assert row == (None, SOLVER_CONTRACT_VERSION)

    def test_predator_not_triggered_from_result_processor(self):
        """_trigger_predator_sweep_async must not be called by process_result."""
        processor = _make_processor()

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as pf:
            json.dump(
                {"spectral_fidelity": {"log_prime_sse": 0.4}, "solver_contract": _GOOD_CONTRACT},
                pf,
            )
            prov_path = pf.name

        try:
            result_data = {
                "job_id": "test-job-001",
                "generation": 1,
                "config_hash": "abc123",
                "artifact_url": "/nonexistent/path.h5",
                "status": "SUCCESS",
                "provenance_path": prov_path,
                "config": {},
            }

            with patch.object(processor, "_trigger_predator_sweep_async") as mock_pred, \
                 patch.object(processor, "_write_worker_result_to_ledger"), \
                 patch.object(processor, "_store_result"), \
                 patch.object(processor, "_trigger_visual_observer_async"):
                processor.process_result(result_data)
                mock_pred.assert_not_called()
        finally:
            os.unlink(prov_path)
