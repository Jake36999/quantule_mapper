import math

import numpy as np


def test_stepper_audit_paths_include_required_deliverables(tmp_path):
    from tools.conservative_stepper_contract_audit import stepper_audit_paths

    paths = stepper_audit_paths(tmp_path)

    assert paths["contract_snapshot"].name == "contract_snapshot.json"
    assert paths["rhs_flux_report"].name == "rhs_flux_report.md"
    assert paths["csv"].name == "conservative_stepper_contract_audit_results.csv"
    assert paths["report"].name == "conservative_stepper_contract_audit_report.md"
    assert paths["history"].name == "conservative_stepper_contract_audit_history.json"
    assert paths["norm_plot"].name == "norm_loss_vs_dt.png"
    assert paths["one_step_plot"].name == "one_step_norm_defect_vs_dt.png"
    assert paths["rho_plot"].name == "rho_max_vs_time.png"


def test_rhs_flux_thresholds_have_zero_warning_and_hard_stop_regions():
    from tools.conservative_stepper_contract_audit import classify_rhs_flux

    assert classify_rhs_flux(1e-11) == "numerical_zero"
    assert classify_rhs_flux(-5e-8) == "warning"
    assert classify_rhs_flux(2e-6) == "hard_stop"


def test_norm_conventions_report_diagnostic_and_physical_grid_norms():
    from tools.conservative_stepper_contract_audit import norm_conventions

    psi = np.ones((4, 4, 4), dtype=np.complex128) * (1.0 + 1.0j)
    norms = norm_conventions(psi, L=8.0)

    assert math.isclose(norms["diagnostic_norm"], 128.0)
    assert math.isclose(norms["dx"], 2.0)
    assert math.isclose(norms["physical_grid_norm"], 1024.0)


def test_fractional_rhs_flux_uses_physical_space_inner_product():
    from tools.conservative_stepper_contract_audit import fractional_rhs_flux

    psi = np.ones((4, 4, 4), dtype=np.complex128)
    skew_rhs = 1j * psi
    leaking_rhs = 0.25 * psi

    skew = fractional_rhs_flux(psi, skew_rhs, L=10.0)
    leak = fractional_rhs_flux(psi, leaking_rhs, L=10.0)

    assert abs(skew["fractional_flux_raw"]) <= 1e-12
    assert math.isclose(leak["fractional_flux_raw"], 0.5)
    assert "physical_grid_norm" in leak


def test_classification_requires_tested_native_profile_for_profile_mismatch_primary():
    from tools.conservative_stepper_contract_audit import classify_audit_result

    result = classify_audit_result(
        contract_ok=True,
        rhs_flux_hard_stop=False,
        timestep_error=True,
        material_norm_change=True,
        profile_mismatch_evidence=True,
        native_profile_tested=False,
        total_norm_conserved=False,
        rho_or_profile_changed=True,
        secondary_flags=[],
    )

    assert result["primary_classification"] == "etdrk4_timestep_error"
    assert "profile_mismatch_untested" in result["secondary_flags"]
    assert result["final_decision"] == "ETDRK4_ERROR_CONFIRMED"


def test_true_dispersion_classification_requires_conserved_norm():
    from tools.conservative_stepper_contract_audit import classify_audit_result

    conserved = classify_audit_result(
        contract_ok=True,
        rhs_flux_hard_stop=False,
        timestep_error=False,
        material_norm_change=False,
        profile_mismatch_evidence=False,
        native_profile_tested=False,
        total_norm_conserved=True,
        rho_or_profile_changed=True,
        secondary_flags=[],
    )
    leaking = classify_audit_result(
        contract_ok=True,
        rhs_flux_hard_stop=False,
        timestep_error=False,
        material_norm_change=True,
        profile_mismatch_evidence=False,
        native_profile_tested=False,
        total_norm_conserved=False,
        rho_or_profile_changed=True,
        secondary_flags=[],
    )

    assert conserved["primary_classification"] == "true_conservative_dispersion_without_norm_loss"
    assert leaking["primary_classification"] != "true_conservative_dispersion_without_norm_loss"


def test_contract_mismatch_maps_to_wrapper_parity_review_required():
    from tools.conservative_stepper_contract_audit import classify_audit_result

    result = classify_audit_result(
        contract_ok=False,
        rhs_flux_hard_stop=False,
        timestep_error=False,
        material_norm_change=False,
        profile_mismatch_evidence=False,
        native_profile_tested=False,
        total_norm_conserved=False,
        rho_or_profile_changed=False,
        secondary_flags=["fft_domain_mismatch"],
    )

    assert result["primary_classification"] == "cupy_wrapper_contract_mismatch"
    assert result["final_decision"] == "WRAPPER_PARITY_REVIEW_REQUIRED"


def test_uniform_control_is_first_rhs_flux_probe():
    from tools.conservative_stepper_contract_audit import rhs_flux_probe_specs

    specs = rhs_flux_probe_specs(spacing=0.45)

    assert specs[0]["case_id"] == "uniform_constant_control"
    assert [s["case_id"] for s in specs[:4]] == [
        "uniform_constant_control",
        "single_gaussian_node",
        "ablated_triangle_spacing_0.45",
        "triangle_spacing_0.45",
    ]
