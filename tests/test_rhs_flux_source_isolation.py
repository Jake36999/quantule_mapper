import numpy as np


def test_parseval_flux_matches_physical_inner_product_for_numpy_ffts():
    from tools.run_rhs_flux_source_isolation import physical_flux_raw, spectral_flux_raw

    rng = np.random.default_rng(123)
    psi = rng.normal(size=(4, 4, 4)) + 1j * rng.normal(size=(4, 4, 4))
    rhs = rng.normal(size=(4, 4, 4)) + 1j * rng.normal(size=(4, 4, 4))
    psi_k = np.fft.fftn(psi)
    rhs_k = np.fft.fftn(rhs)

    assert np.isclose(physical_flux_raw(psi, rhs), spectral_flux_raw(psi_k, rhs_k))


def test_flux_status_thresholds_match_integrity_batch():
    from tools.run_rhs_flux_source_isolation import flux_status

    assert flux_status(1e-11) == "numerical_zero"
    assert flux_status(2e-8) == "warning"
    assert flux_status(2e-6) == "fail"


def test_source_isolation_manifest_uses_requested_samples():
    from tools.run_rhs_flux_source_isolation import build_manifest

    manifest = build_manifest()

    assert manifest["N"] == 48
    assert manifest["L"] == 10.0
    assert manifest["dt"] == 0.001
    assert manifest["physical_time"] == 1.0
    assert manifest["sample_times"] == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_component_classification_prioritizes_measurement_mismatch():
    from tools.run_rhs_flux_source_isolation import classify_source

    result = classify_source(
        max_physical_spectral_diff=1e-4,
        max_roundtrip_diff=1e-12,
        max_linear_flux=1e-12,
        max_full_flux=1e-4,
        max_nonlinear_flux=1e-4,
    )

    assert result["final_decision"] == "RHS_FLUX_MEASUREMENT_CONVENTION_FAIL"


def test_component_classification_detects_nonlinear_algebraic_flux():
    from tools.run_rhs_flux_source_isolation import classify_source

    result = classify_source(
        max_physical_spectral_diff=1e-12,
        max_roundtrip_diff=1e-12,
        max_linear_flux=1e-12,
        max_full_flux=2e-4,
        max_nonlinear_flux=2e-4,
    )

    assert result["final_decision"] == "RHS_NONLINEAR_ALGEBRAIC_FLUX_CONFIRMED"


def test_final_decision_labels_are_restricted():
    from tools.run_rhs_flux_source_isolation import FINAL_DECISIONS

    assert FINAL_DECISIONS == {
        "RHS_NONLINEAR_ALGEBRAIC_FLUX_CONFIRMED",
        "RHS_FLUX_MEASUREMENT_CONVENTION_FAIL",
        "RHS_FLUX_RECONSTRUCTION_FAIL",
        "RHS_LINEAR_TERM_SOURCE",
        "RHS_FLUX_SOURCE_UNCLEAR",
    }
