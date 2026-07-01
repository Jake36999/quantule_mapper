"""
tests/test_refinement_lifecycle_e2e.py

End-to-end lifecycle test: baseline candidate → VALIDATED_PROVISIONAL →
variants dispatched → mocked variant provenance → RefinementAudit.assess() →
REFINEMENT_STABLE or REFINEMENT_REJECTED → only STABLE can promote.

No real DB, no subprocesses — all variant provenance is constructed in memory.
"""
import pytest
from orchestrator.contracts import SOLVER_CONTRACT_VERSION, PROVISIONAL_SSE_THRESHOLD
from validation_pipeline import RefinementAudit, RefinementManifestGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prov(sse: float, peak: float = 0.30, contract: bool = True) -> dict:
    prov = {
        "spectral_fidelity": {
            "log_prime_sse": sse,
            "dominant_peak_k": peak,
            "sse_directional_min": sse,
        }
    }
    if contract:
        prov["solver_contract"] = {"solver_contract_version": SOLVER_CONTRACT_VERSION}
    return prov


def _stamp_result(result_data: dict, provenance: dict) -> dict:
    """Replicate result_processor contract-stamp logic."""
    contract = provenance.get("solver_contract") if isinstance(provenance, dict) else None
    contract_version = (
        (contract or {}).get("solver_contract_version", "") if isinstance(contract, dict) else ""
    )
    result_data["_solver_contract_version"] = contract_version
    log_prime_sse = float(provenance.get("spectral_fidelity", {}).get("log_prime_sse", 999.0))
    result_data["_ingested_log_prime_sse"] = log_prime_sse

    if contract_version != SOLVER_CONTRACT_VERSION:
        result_data["_refinement_status"] = "PHYSICS_UNCERTIFIED"
    elif log_prime_sse < PROVISIONAL_SSE_THRESHOLD:
        result_data["_refinement_status"] = "VALIDATED_PROVISIONAL"

    return result_data


def _apply_gate(result_data: dict, global_best_sse: float):
    """Replicate orchestrator_engine champion gate logic."""
    state = {"promoted": False, "dispatched": False, "predator_fired": False,
             "global_best_sse": global_best_sse}
    contract_version = result_data.get("_solver_contract_version", "")
    refinement_status = result_data.get("_refinement_status")
    sse = float(result_data.get("_ingested_log_prime_sse", 999.0))

    if contract_version != SOLVER_CONTRACT_VERSION:
        return state
    if refinement_status == "VALIDATED_PROVISIONAL":
        state["dispatched"] = True
    elif refinement_status == "REFINEMENT_STABLE":
        if sse < state["global_best_sse"]:
            state["global_best_sse"] = sse
            state["promoted"] = True
            state["predator_fired"] = True
    return state


# ---------------------------------------------------------------------------
# Lifecycle helpers
# ---------------------------------------------------------------------------

def _run_full_lifecycle(baseline_sse: float, variant_provs: list[tuple[str, dict]]):
    """
    Simulate:
      1. Baseline result stamped by ResultProcessor
      2. Gate check → should dispatch (VALIDATED_PROVISIONAL)
      3. Variant provenances assembled and assessed by RefinementAudit
      4. Final status set based on audit verdict
      5. Gate re-checked with final status
    Returns final state dict.
    """
    baseline_prov = _make_prov(baseline_sse)
    result_data = {"config_hash": "abc123", "generation": 1}
    _stamp_result(result_data, baseline_prov)

    initial_gate = _apply_gate(result_data, global_best_sse=999.0)
    assert initial_gate["dispatched"], "Expected VALIDATED_PROVISIONAL dispatch"

    audit = RefinementAudit.assess(variant_provs)

    if audit["verdict"] == "STABLE":
        result_data["_refinement_status"] = "REFINEMENT_STABLE"
    else:
        result_data["_refinement_status"] = "REFINEMENT_REJECTED"

    final_gate = _apply_gate(result_data, global_best_sse=999.0)
    final_gate["audit"] = audit
    return final_gate


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRefinementLifecycleE2E:
    def test_stable_variants_lead_to_promotion(self):
        """Baseline below threshold + stable variants → REFINEMENT_STABLE → promoted."""
        variant_provs = [
            ("baseline",    _make_prov(0.40, peak=0.30)),
            ("dt_half",     _make_prov(0.45, peak=0.30)),
            ("dealias_2_3", _make_prov(0.50, peak=0.31)),
        ]
        state = _run_full_lifecycle(baseline_sse=0.40, variant_provs=variant_provs)

        assert state["audit"]["verdict"] == "STABLE"
        assert state["promoted"] is True
        assert state["predator_fired"] is True

    def test_drifting_variants_block_promotion(self):
        """Baseline below threshold + drifting variants → REFINEMENT_REJECTED → not promoted."""
        variant_provs = [
            ("baseline", _make_prov(0.40, peak=0.30)),
            ("dt_half",  _make_prov(3.80, peak=0.30)),
        ]
        state = _run_full_lifecycle(baseline_sse=0.40, variant_provs=variant_provs)

        assert state["audit"]["verdict"] == "DRIFTING"
        assert state["promoted"] is False
        assert state["predator_fired"] is False

    def test_baseline_above_threshold_never_dispatches(self):
        """Baseline SSE above PROVISIONAL_SSE_THRESHOLD → no VALIDATED_PROVISIONAL → no dispatch."""
        baseline_prov = _make_prov(PROVISIONAL_SSE_THRESHOLD + 0.1)
        result_data = {"config_hash": "xyz999", "generation": 1}
        _stamp_result(result_data, baseline_prov)

        assert result_data.get("_refinement_status") is None
        gate = _apply_gate(result_data, global_best_sse=999.0)
        assert gate["dispatched"] is False
        assert gate["promoted"] is False

    def test_uncertified_baseline_never_dispatches(self):
        """Baseline with no solver_contract → PHYSICS_UNCERTIFIED → no dispatch, no promotion."""
        prov = _make_prov(0.40, contract=False)
        result_data = {"config_hash": "legacy001", "generation": 1}
        _stamp_result(result_data, prov)

        assert result_data["_refinement_status"] == "PHYSICS_UNCERTIFIED"
        gate = _apply_gate(result_data, global_best_sse=999.0)
        assert gate["dispatched"] is False
        assert gate["promoted"] is False

    def test_predator_not_fired_for_provisional(self):
        """VALIDATED_PROVISIONAL dispatch must never trigger predator."""
        prov = _make_prov(0.40)
        result_data = {"config_hash": "prov001", "generation": 1}
        _stamp_result(result_data, prov)

        gate = _apply_gate(result_data, global_best_sse=999.0)
        assert gate["dispatched"] is True
        assert gate["predator_fired"] is False

    def test_variant_manifest_generates_four_variants(self):
        """RefinementManifestGenerator must emit exactly 4 variant configs."""
        base_params = {
            "config_hash": "abc123",
            "global_seed": 42,
            "param_D": 1.0,
            "param_eta": 0.1,
            "param_rho_vac": 1.0,
            "param_a_coupling": 0.5,
            "param_splash_coupling": 0.2,
            "param_splash_fraction": -0.1,
            "simulation": {"n_grid": 32, "t_steps": 100, "dt": 0.001},
        }
        variants = RefinementManifestGenerator.generate(base_params)
        assert len(variants) == 4
        labels = {v["label"] for v in variants}
        assert labels == {"dt_half", "dt_quarter", "dealias_2_3", "grid_double"}
