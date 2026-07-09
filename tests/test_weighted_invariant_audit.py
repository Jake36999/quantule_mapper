import numpy as np


def test_recommendations_are_restricted():
    from tools.run_weighted_invariant_audit import RECOMMENDATIONS

    assert RECOMMENDATIONS == {
        "ORDINARY_NORM_NOT_CONSERVED_WEIGHTED_CANDIDATE_FOUND",
        "NO_WEIGHTED_INVARIANT_FOUND",
        "CONTRACT_REVIEW_REQUIRED",
    }


def test_weight_specs_include_required_candidates():
    from tools.run_weighted_invariant_audit import build_weight_specs

    labels = [spec["label"] for spec in build_weight_specs()]

    assert "ordinary_norm" in labels
    assert "dx_scaled_norm" in labels
    assert "sqrt_g_weighted_norm" in labels
    assert "inverse_sqrt_g_weighted_norm" in labels
    assert "omega_exact_helper_weight" in labels
    assert "omega_sq_exact_helper_weight" in labels
    assert "omega_power_p-3_exploratory" in labels
    assert "omega_power_p3_exploratory" in labels


def test_weight_array_from_spec_handles_power_and_dx():
    from tools.run_weighted_invariant_audit import compute_weight_from_spec

    omega = np.array([2.0, 4.0])
    omega_sq = omega**2

    assert np.allclose(compute_weight_from_spec({"kind": "ordinary"}, omega, omega_sq), [1.0, 1.0])
    assert np.allclose(compute_weight_from_spec({"kind": "omega_power", "power": -1.0}, omega, omega_sq), [0.5, 0.25])
    assert np.allclose(compute_weight_from_spec({"kind": "omega_sq"}, omega, omega_sq), [4.0, 16.0])
    assert np.allclose(compute_weight_from_spec({"kind": "inverse_omega_sq"}, omega, omega_sq), [0.25, 0.0625])


def test_weighted_invariant_and_flux_use_same_weight():
    from tools.run_weighted_invariant_audit import weighted_fractional_flux, weighted_invariant

    psi = np.array([1.0 + 0.0j, 2.0 + 0.0j])
    rhs = np.array([0.0 + 1.0j, 0.0 + 2.0j])
    weight = np.array([1.0, 3.0])

    assert weighted_invariant(psi, weight, scale=1.0) == 13.0
    assert weighted_fractional_flux(psi, rhs, weight) == 0.0


def test_candidate_classifier_requires_improvement_in_drift_and_adjointness():
    from tools.run_weighted_invariant_audit import classify_recommendation

    found = classify_recommendation(
        ordinary_abs_drift=1e-3,
        best_weight_abs_drift=1e-5,
        ordinary_adjoint_mismatch=1e-3,
        best_weight_adjoint_mismatch=1e-5,
        best_label="sqrt_g_weighted_norm",
    )
    missing = classify_recommendation(
        ordinary_abs_drift=1e-3,
        best_weight_abs_drift=9e-4,
        ordinary_adjoint_mismatch=1e-3,
        best_weight_adjoint_mismatch=9e-4,
        best_label="omega_power_p2_exploratory",
    )

    assert found["recommendation"] == "ORDINARY_NORM_NOT_CONSERVED_WEIGHTED_CANDIDATE_FOUND"
    assert missing["recommendation"] == "NO_WEIGHTED_INVARIANT_FOUND"


def test_manifest_uses_t0_to_t1_rk4_trajectory():
    from tools.run_weighted_invariant_audit import build_manifest

    manifest = build_manifest()

    assert manifest["N"] == 48
    assert manifest["L"] == 10.0
    assert manifest["dt"] == 0.001
    assert manifest["physical_time"] == 1.0
    assert manifest["sample_times"] == [0.0, 0.25, 0.5, 0.75, 1.0]
