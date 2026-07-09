import numpy as np


def test_final_decisions_are_restricted():
    from tools.run_rhs_term_flux_decomposition import FINAL_DECISIONS

    assert FINAL_DECISIONS == {
        "TERM_RECOMBINATION_MISMATCH",
        "RHS_TERM_CONTRACT_REVIEW_REQUIRED",
        "DISCRETE_OPERATOR_ADJOINT_FAILURE",
        "RHS_NONCONSERVATIVE_BY_CONTRACT",
        "CONSERVATIVE_LABEL_AMBIGUOUS",
        "FLUX_SOURCE_UNCLEAR",
    }


def test_manifest_contains_required_samples_and_controls():
    from tools.run_rhs_term_flux_decomposition import build_manifest

    manifest = build_manifest(max_wallclock_minutes=60)

    assert manifest["N"] == 48
    assert manifest["L"] == 10.0
    assert manifest["dt"] == 0.001
    assert manifest["sample_times"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert [case["case_id"] for case in manifest["symmetry_controls"]] == [
        "uniform_constant",
        "single_gaussian",
        "triangle_symmetric",
        "triangle_amp_minus_5pct",
        "triangle_phase_pi8",
        "triangle_position_jitter",
        "ablated_triangle",
        "tetrahedron",
        "triangular_prism",
    ]


def test_term_labels_match_c2_rhs_terms():
    from tools.run_rhs_term_flux_decomposition import TERM_LABELS

    assert TERM_LABELS == [
        "geometry_covariant_correction",
        "cubic_density_a",
        "quintic_density_s",
        "septic_density_f",
    ]


def test_recombination_classifier_stops_on_mismatch():
    from tools.run_rhs_term_flux_decomposition import classify_recombination

    ok = classify_recombination(max_abs_error=1e-11, relative_l2_error=1e-12)
    bad = classify_recombination(max_abs_error=1e-5, relative_l2_error=1e-8)

    assert ok == "pass"
    assert bad == "TERM_RECOMBINATION_MISMATCH"


def test_flux_percentage_uses_total_flux_and_handles_zero():
    from tools.run_rhs_term_flux_decomposition import percentage_contribution

    assert percentage_contribution(-2.0, -4.0) == 50.0
    assert percentage_contribution(1.0, 0.0) is None


def test_adjoint_classifier_thresholds():
    from tools.run_rhs_term_flux_decomposition import classify_adjoint_mismatch

    assert classify_adjoint_mismatch(1e-12) == "ADJOINT_PASS"
    assert classify_adjoint_mismatch(1e-7) == "ADJOINT_WARNING"
    assert classify_adjoint_mismatch(1e-4) == "ADJOINT_FAIL"


def test_contract_decision_prefers_operator_failure():
    from tools.run_rhs_term_flux_decomposition import classify_final_decision

    result = classify_final_decision(
        recombination_status="pass",
        dominant_term="geometry_covariant_correction",
        max_term_flux=3e-4,
        adjoint_status="ADJOINT_FAIL",
        docs_state="ambiguous",
    )

    assert result["final_decision"] == "DISCRETE_OPERATOR_ADJOINT_FAILURE"


def test_contract_decision_uses_ambiguous_label_when_docs_unclear():
    from tools.run_rhs_term_flux_decomposition import classify_final_decision

    result = classify_final_decision(
        recombination_status="pass",
        dominant_term="cubic_density_a",
        max_term_flux=3e-4,
        adjoint_status="ADJOINT_PASS",
        docs_state="ambiguous",
    )

    assert result["final_decision"] == "CONSERVATIVE_LABEL_AMBIGUOUS"


def test_parseval_flux_helper_matches_numpy_ffts():
    from tools.run_rhs_term_flux_decomposition import physical_flux_raw, spectral_flux_raw

    rng = np.random.default_rng(17)
    psi = rng.normal(size=(4, 4, 4)) + 1j * rng.normal(size=(4, 4, 4))
    rhs = rng.normal(size=(4, 4, 4)) + 1j * rng.normal(size=(4, 4, 4))

    assert np.isclose(physical_flux_raw(psi, rhs), spectral_flux_raw(np.fft.fftn(psi), np.fft.fftn(rhs)))


def test_protected_file_list_contains_reference_paths():
    from tools.run_rhs_term_flux_decomposition import PROTECTED_FILES

    assert "solver/core.py" in PROTECTED_FILES
    assert "worker_cupy.py" in PROTECTED_FILES
    assert "jax_scout/physics.py" in PROTECTED_FILES
    assert "jax_scout/phase_d_c2_2_loss_source.py" in PROTECTED_FILES


def test_stringified_none_specs_count_as_absent():
    from tools.run_rhs_term_flux_decomposition import _spec_absent

    assert _spec_absent(None) is True
    assert _spec_absent("None") is True
    assert _spec_absent("ModuleSpec(name='jax')") is False
