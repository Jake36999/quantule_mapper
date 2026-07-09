import math

import numpy as np


def test_rk4_diagnostic_paths_include_required_outputs(tmp_path):
    from tools.conservative_rk4_stepper_diagnostic import rk4_diagnostic_paths

    paths = rk4_diagnostic_paths(tmp_path)

    assert paths["safety_checkpoint"].name == "safety_checkpoint.md"
    assert paths["protected_diff_before"].name == "protected_diff_before.txt"
    assert paths["one_step_csv"].name == "one_step_rk4_vs_etdrk4.csv"
    assert paths["multistep_csv"].name == "multistep_rk4_vs_etdrk4_results.csv"
    assert paths["geometry_csv"].name == "rk4_geometry_replay_results.csv"
    assert paths["final_report"].name == "rk4_stepper_diagnostic_final_report.md"


def test_parser_exposes_smoke_and_audit_subcommands():
    from tools.conservative_rk4_stepper_diagnostic import build_parser

    parser = build_parser()
    smoke = parser.parse_args(["smoke", "--out", "x"])
    audit = parser.parse_args(["audit", "--out", "x"])

    assert smoke.command == "smoke"
    assert audit.command == "audit"


def test_protected_files_include_reference_and_production_paths():
    from tools.conservative_rk4_stepper_diagnostic import PROTECTED_FILES

    required = {
        "solver/core.py",
        "solver/run.py",
        "worker_cupy.py",
        "jax_scout/physics.py",
        "jax_scout/phase_d_c2_transport.py",
        "jax_scout/phase_d_c2_soliton_scout.py",
        "jax_scout/phase_d_c2_2_loss_source.py",
    }

    assert required.issubset(set(PROTECTED_FILES))


def test_final_decision_labels_are_restricted_to_requested_set():
    from tools.conservative_rk4_stepper_diagnostic import FINAL_DECISIONS

    assert FINAL_DECISIONS == {
        "RK4_STEPPER_REJECTED",
        "RK4_STEPPER_DIAGNOSTIC_PROMISING",
        "RK4_GEOMETRY_REPLAY_PROMISING",
        "GEOMETRY_DEPENDED_ON_LOSSY_ETDRK4",
        "CONSERVATIVE_STEPPER_STILL_BLOCKED",
    }


def test_rk4_stepper_projects_external_psi0_to_complex128_spectral_state():
    import cupy as cp

    from tools.conservative_geometry_campaign import build_geometry_ic, points_for_case
    from tools.conservative_rk4_stepper_diagnostic import ConservativeC2RK4Stepper

    N = 16
    points = points_for_case("triangle", 0.45, {}, seed=123)
    psi0 = build_geometry_ic(N=N, L=10.0, points_box=points, width_box=0.08333333333333333, amplitude=1.0)
    stepper = ConservativeC2RK4Stepper(N=N, L=10.0, dt=1e-5)
    psi_k, psi_phys = stepper.project_psi0(psi0)

    assert psi_k.shape == (N, N, N)
    assert psi_phys.shape == (N, N, N)
    assert psi_k.dtype == cp.complex128
    assert psi_phys.dtype == cp.complex128


def test_one_rk4_step_is_finite_and_does_not_normalize_amplitude():
    import cupy as cp

    from tools.conservative_geometry_campaign import build_geometry_ic, points_for_case
    from tools.conservative_rk4_stepper_diagnostic import ConservativeC2RK4Stepper

    N = 16
    points = points_for_case("triangle", 0.45, {}, seed=123)
    psi0 = build_geometry_ic(N=N, L=10.0, points_box=points, width_box=0.08333333333333333, amplitude=1.0)
    stepper = ConservativeC2RK4Stepper(N=N, L=10.0, dt=1e-5)
    psi_k, _ = stepper.project_psi0(psi0)
    norm_before = float(cp.sum(cp.abs(stepper.to_physical(psi_k)) ** 2, dtype=cp.float64))
    psi_next_k = stepper.step(psi_k)
    psi_next = stepper.to_physical(psi_next_k)
    norm_after = float(cp.sum(cp.abs(psi_next) ** 2, dtype=cp.float64))

    assert bool(cp.isfinite(psi_next).all())
    assert math.isfinite(norm_after)
    assert abs(norm_after - norm_before) > 0.0
    assert abs((norm_after - norm_before) / norm_before) < 1e-6


def test_linear_only_rk4_short_run_preserves_norm_at_tiny_dt():
    import cupy as cp

    from tools.conservative_geometry_campaign import build_geometry_ic, points_for_case
    from tools.conservative_rk4_stepper_diagnostic import ConservativeC2RK4Stepper

    N = 16
    points = points_for_case("triangle", 0.45, {}, seed=123)
    psi0 = build_geometry_ic(N=N, L=10.0, points_box=points, width_box=0.08333333333333333, amplitude=1.0)
    stepper = ConservativeC2RK4Stepper(N=N, L=10.0, dt=1e-6, nonlinear_enabled=False)
    psi_k, _ = stepper.project_psi0(psi0)
    norm_before = float(cp.sum(cp.abs(stepper.to_physical(psi_k)) ** 2, dtype=cp.float64))
    for _ in range(5):
        psi_k = stepper.step(psi_k)
    norm_after = float(cp.sum(cp.abs(stepper.to_physical(psi_k)) ** 2, dtype=cp.float64))

    assert abs((norm_after - norm_before) / norm_before) < 1e-6


def test_rhs_flux_controls_remain_numerical_zero():
    from tools.conservative_rk4_stepper_diagnostic import rhs_flux_controls_are_zero

    result = rhs_flux_controls_are_zero(N=16, L=10.0, dt=0.001, spacing=0.45)

    assert result["uniform_constant_control"] == "numerical_zero"
    assert result["triangle_spacing_0.45"] == "numerical_zero"
