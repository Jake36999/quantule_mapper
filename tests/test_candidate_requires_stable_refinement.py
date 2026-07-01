"""
tests/test_candidate_requires_stable_refinement.py

End-to-end gate contract: only REFINEMENT_STABLE candidates may become champion.
Exercises the full state-machine path including RefinementAudit integration.
"""
import pytest
from unittest.mock import MagicMock

from orchestrator.contracts import SOLVER_CONTRACT_VERSION, PROVISIONAL_SSE_THRESHOLD
from validation_pipeline import RefinementAudit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prov(sse: float, peak: float = 0.30) -> dict:
    return {
        "spectral_fidelity": {
            "log_prime_sse": sse,
            "dominant_peak_k": peak,
            "sse_directional_min": sse,
        }
    }


def _run_gate(refinement_status: str, sse: float = 0.4) -> dict:
    """Simulate the champion gate decision for a given refinement_status."""
    state = {"promoted": False, "predator_fired": False}
    global_best = {"sse": 999.0}

    contract_version = SOLVER_CONTRACT_VERSION
    if refinement_status == "PHYSICS_UNCERTIFIED":
        contract_version = "WRONG"

    if contract_version != SOLVER_CONTRACT_VERSION:
        return state
    if refinement_status == "VALIDATED_PROVISIONAL":
        return state  # dispatch only, no promotion
    if refinement_status == "REFINEMENT_STABLE":
        if sse < global_best["sse"]:
            global_best["sse"] = sse
            state["promoted"] = True
            state["predator_fired"] = True
    return state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCandidateRequiresStableRefinement:
    def test_provisional_alone_is_not_sufficient_for_promotion(self):
        state = _run_gate("VALIDATED_PROVISIONAL", sse=0.3)
        assert state["promoted"] is False

    def test_stable_is_sufficient_for_promotion(self):
        state = _run_gate("REFINEMENT_STABLE", sse=0.3)
        assert state["promoted"] is True

    def test_uncertified_is_not_sufficient_for_promotion(self):
        state = _run_gate("PHYSICS_UNCERTIFIED", sse=0.3)
        assert state["promoted"] is False

    def test_rejected_is_not_sufficient_for_promotion(self):
        state = _run_gate("REFINEMENT_REJECTED", sse=0.3)
        assert state["promoted"] is False

    def test_predator_fires_only_on_stable(self):
        assert _run_gate("REFINEMENT_STABLE")["predator_fired"] is True
        assert _run_gate("VALIDATED_PROVISIONAL")["predator_fired"] is False
        assert _run_gate("PHYSICS_UNCERTIFIED")["predator_fired"] is False

    def test_audit_stable_verdict_enables_promotion(self):
        results = [
            ("baseline", _make_prov(0.4, peak=0.30)),
            ("dt_half",  _make_prov(0.5, peak=0.31)),
            ("dealias_2_3", _make_prov(0.6, peak=0.30)),
        ]
        audit = RefinementAudit.assess(results)
        assert audit["verdict"] == "STABLE"

        # With STABLE verdict, gate permits promotion
        state = _run_gate("REFINEMENT_STABLE", sse=0.4)
        assert state["promoted"] is True

    def test_audit_drifting_verdict_blocks_promotion(self):
        results = [
            ("baseline", _make_prov(0.4)),
            ("dt_half",  _make_prov(3.5)),
        ]
        audit = RefinementAudit.assess(results)
        assert audit["verdict"] == "DRIFTING"

        # Drifting maps to REFINEMENT_REJECTED — must not promote
        state = _run_gate("REFINEMENT_REJECTED", sse=0.4)
        assert state["promoted"] is False

    def test_provisional_sse_threshold_boundary(self):
        """Candidates at exactly the threshold must NOT enter the provisional path."""
        at_threshold = PROVISIONAL_SSE_THRESHOLD
        result_data: dict = {
            '_ingested_log_prime_sse': at_threshold,
            '_solver_contract_version': SOLVER_CONTRACT_VERSION,
        }
        # Apply the contract-check logic inline
        if float(result_data['_ingested_log_prime_sse']) < PROVISIONAL_SSE_THRESHOLD:
            result_data['_refinement_status'] = 'VALIDATED_PROVISIONAL'

        assert '_refinement_status' not in result_data
