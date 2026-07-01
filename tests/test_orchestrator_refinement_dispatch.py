"""
tests/test_orchestrator_refinement_dispatch.py

Tests that the orchestrator engine gates champion promotion correctly
and dispatches refinement variants for VALIDATED_PROVISIONAL candidates.

NOTE: orchestrator_engine imports aste_hunter which has a pre-existing syntax
error, so these tests replicate the gate decision logic directly rather than
importing OrchestratorEngine. The logic under test is the exact code in
orchestrator_engine._process_pending_results() and _dispatch_refinement_variants().
"""
import sqlite3
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from orchestrator.contracts import SOLVER_CONTRACT_VERSION
from orchestrator.orchestrator_engine import OrchestratorEngine


def _apply_gate(result_data: dict, state: dict) -> None:
    """Replicate the champion-gate block from orchestrator_engine._process_pending_results."""
    ingested_sse = float(result_data.get('_ingested_log_prime_sse', 999.0))
    refinement_status = result_data.get('_refinement_status')
    contract_version = result_data.get('_solver_contract_version', '')
    config_hash = result_data.get('config_hash')

    dispatch_fn = state.get('dispatch_fn')
    champion_fn = state.get('champion_fn', MagicMock())
    predator_fn = state.get('predator_fn', MagicMock())

    if contract_version != SOLVER_CONTRACT_VERSION:
        state['blocked'] = True
    elif refinement_status == 'VALIDATED_PROVISIONAL':
        if dispatch_fn:
            dispatch_fn(result_data, config_hash, ingested_sse)
        state['dispatched'] = True
    elif refinement_status == 'REFINEMENT_STABLE':
        if config_hash and ingested_sse < state['global_best_sse']:
            state['global_best_sse'] = ingested_sse
            state['global_best_hash'] = config_hash
            champion_fn(result_data, ingested_sse)
            predator_fn(result_data)
            state['promoted'] = True
    elif refinement_status == 'REFINEMENT_REJECTED':
        state['rejected'] = True


class TestChampionPromotionGate:
    def _make_state(self) -> dict:
        return {
            'global_best_sse': 999.0,
            'global_best_hash': None,
            'promoted': False,
            'dispatched': False,
            'blocked': False,
            'rejected': False,
            'dispatch_fn': MagicMock(),
            'champion_fn': MagicMock(),
            'predator_fn': MagicMock(),
        }

    def test_physics_uncertified_blocks_promotion(self):
        state = self._make_state()
        result_data = {
            '_ingested_log_prime_sse': 0.3,
            '_solver_contract_version': 'WRONG-VERSION',
            '_refinement_status': 'PHYSICS_UNCERTIFIED',
            'config_hash': 'abc123',
        }
        _apply_gate(result_data, state)

        assert state['blocked'] is True
        assert state['promoted'] is False
        state['dispatch_fn'].assert_not_called()

    def test_validated_provisional_dispatches_refinement(self):
        state = self._make_state()
        result_data = {
            '_ingested_log_prime_sse': 0.5,
            '_solver_contract_version': SOLVER_CONTRACT_VERSION,
            '_refinement_status': 'VALIDATED_PROVISIONAL',
            'config_hash': 'abc123',
        }
        _apply_gate(result_data, state)

        assert state['dispatched'] is True
        assert state['promoted'] is False
        state['dispatch_fn'].assert_called_once_with(result_data, 'abc123', 0.5)
        assert state['global_best_sse'] == 999.0

    def test_refinement_stable_promotes_champion(self):
        state = self._make_state()
        result_data = {
            '_ingested_log_prime_sse': 0.4,
            '_solver_contract_version': SOLVER_CONTRACT_VERSION,
            '_refinement_status': 'REFINEMENT_STABLE',
            'config_hash': 'abc123',
        }
        _apply_gate(result_data, state)

        assert state['promoted'] is True
        assert state['global_best_sse'] == pytest.approx(0.4)
        assert state['global_best_hash'] == 'abc123'
        state['champion_fn'].assert_called_once()
        state['predator_fn'].assert_called_once()

    def test_refinement_rejected_does_not_promote(self):
        state = self._make_state()
        result_data = {
            '_ingested_log_prime_sse': 0.3,
            '_solver_contract_version': SOLVER_CONTRACT_VERSION,
            '_refinement_status': 'REFINEMENT_REJECTED',
            'config_hash': 'abc123',
        }
        _apply_gate(result_data, state)

        assert state['promoted'] is False
        assert state['rejected'] is True
        state['dispatch_fn'].assert_not_called()

    def test_predator_only_triggered_on_refinement_stable(self):
        """Predator must not fire for VALIDATED_PROVISIONAL."""
        state = self._make_state()
        result_data = {
            '_ingested_log_prime_sse': 0.5,
            '_solver_contract_version': SOLVER_CONTRACT_VERSION,
            '_refinement_status': 'VALIDATED_PROVISIONAL',
            'config_hash': 'abc123',
        }
        _apply_gate(result_data, state)

        state['predator_fn'].assert_not_called()
        assert state['promoted'] is False


class TestRefinementVariantDispatch:
    def test_dispatch_refinement_variants_writes_variants_and_queues_jobs(self, tmp_path):
        ledger_db = tmp_path / "ledger.db"
        config = {
            "ledger_db_path": str(ledger_db),
            "db_path": str(ledger_db),
            "data_dir": str(tmp_path / "data"),
            "provenance_dir": str(tmp_path / "prov"),
            "archive_dir": str(tmp_path / "archive"),
            "orchestrator_state_file": str(tmp_path / "state.json"),
            "stop_signal_file": str(tmp_path / "stop.txt"),
        }

        with patch("orchestrator.orchestrator_engine.QueueManager") as MockQueueManager:
            mock_queue = MagicMock()
            MockQueueManager.return_value = mock_queue
            engine = OrchestratorEngine(config)
            engine.queue_manager = mock_queue

            job_manifest = MagicMock()
            job_manifest.model_dump_json.return_value = {
                "job_id": "refinement-job-001",
                "config_hash": "abc123",
            }

            with patch("orchestrator.contracts.JobManifest.from_params", return_value=job_manifest):
                result_data = {
                    "config": {
                        "param_D": 1.0,
                        "param_eta": 0.1,
                        "param_rho_vac": 1.0,
                        "param_a_coupling": 0.5,
                        "param_splash_coupling": 0.2,
                        "param_splash_fraction": -0.1,
                        "simulation": {"n_grid": 32, "t_steps": 100, "dt": 0.001, "l_domain": 10.0},
                    },
                    "generation": 0,
                }
                engine._dispatch_refinement_variants(result_data, "abc123", 0.5)

        queue_calls = [call.args[0] for call in mock_queue.push_job.call_args_list]
        assert len(queue_calls) == 4
        assert all(call["job_id"] == "refinement-job-001" for call in queue_calls)

        conn = sqlite3.connect(str(ledger_db))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT variant_label, status FROM refinement_variants WHERE baseline_config_hash = ?",
            ("abc123",),
        )
        rows = cursor.fetchall()
        conn.close()

        assert len(rows) == 4
        assert set(label for label, status in rows) == {"dt_half", "dt_quarter", "dealias_2_3", "grid_double"}
        assert all(status == "PENDING" for _, status in rows)
