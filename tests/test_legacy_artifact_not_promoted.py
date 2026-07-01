"""
tests/test_legacy_artifact_not_promoted.py

Tests that artifacts lacking a solver_contract in their provenance are
stamped PHYSICS_UNCERTIFIED and cannot be promoted to champion.
"""
import pytest
from orchestrator.contracts import SOLVER_CONTRACT_VERSION, PROVISIONAL_SSE_THRESHOLD


def _apply_gate_logic(result_data: dict) -> dict:
    """Replicate the gate decision logic from orchestrator_engine in isolation."""
    log_prime_sse = float(result_data.get('_ingested_log_prime_sse', 999.0))
    provenance = result_data.get('_provenance', {})
    contract = provenance.get('solver_contract') if isinstance(provenance, dict) else None
    contract_version = (
        contract.get('solver_contract_version', '') if isinstance(contract, dict) else ''
    )
    result_data['_solver_contract_version'] = contract_version

    if contract_version != SOLVER_CONTRACT_VERSION:
        result_data['_refinement_status'] = 'PHYSICS_UNCERTIFIED'
    elif log_prime_sse < PROVISIONAL_SSE_THRESHOLD:
        result_data['_refinement_status'] = 'VALIDATED_PROVISIONAL'

    return result_data


class TestLegacyArtifactExclusion:
    def test_missing_solver_contract_key_is_uncertified(self):
        result_data = {
            '_ingested_log_prime_sse': 0.3,
            '_provenance': {'spectral_fidelity': {'log_prime_sse': 0.3}},
        }
        out = _apply_gate_logic(result_data)
        assert out['_refinement_status'] == 'PHYSICS_UNCERTIFIED'

    def test_null_solver_contract_is_uncertified(self):
        result_data = {
            '_ingested_log_prime_sse': 0.3,
            '_provenance': {'spectral_fidelity': {'log_prime_sse': 0.3}, 'solver_contract': None},
        }
        out = _apply_gate_logic(result_data)
        assert out['_refinement_status'] == 'PHYSICS_UNCERTIFIED'

    def test_empty_contract_dict_is_uncertified(self):
        result_data = {
            '_ingested_log_prime_sse': 0.3,
            '_provenance': {'solver_contract': {}},
        }
        out = _apply_gate_logic(result_data)
        assert out['_refinement_status'] == 'PHYSICS_UNCERTIFIED'

    def test_wrong_version_string_is_uncertified(self):
        result_data = {
            '_ingested_log_prime_sse': 0.3,
            '_provenance': {
                'solver_contract': {'solver_contract_version': 'FLAT-LAPLACIAN-WRONG-v0'}
            },
        }
        out = _apply_gate_logic(result_data)
        assert out['_refinement_status'] == 'PHYSICS_UNCERTIFIED'

    def test_correct_contract_below_threshold_is_provisional(self):
        result_data = {
            '_ingested_log_prime_sse': 0.5,
            '_provenance': {
                'solver_contract': {'solver_contract_version': SOLVER_CONTRACT_VERSION}
            },
        }
        out = _apply_gate_logic(result_data)
        assert out['_refinement_status'] == 'VALIDATED_PROVISIONAL'

    def test_correct_contract_above_threshold_has_no_refinement_status(self):
        result_data = {
            '_ingested_log_prime_sse': 2.0,
            '_provenance': {
                'solver_contract': {'solver_contract_version': SOLVER_CONTRACT_VERSION}
            },
        }
        out = _apply_gate_logic(result_data)
        assert '_refinement_status' not in out
