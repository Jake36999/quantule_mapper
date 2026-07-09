import re


def test_manifest_contains_required_integrity_and_optional_n64_cases():
    from tools.run_rk4_integrity_batch import build_manifest

    manifest = build_manifest()
    groups = [case["group"] for case in manifest["cases"]]
    labels = [case.get("label") for case in manifest["cases"]]

    assert groups.count("dt_integrity_fixed_T") == 4
    assert groups.count("dt_integrity_fixed_dt") == 3
    assert groups.count("trajectory_rhs_flux") == 1
    assert groups.count("n64_geometry_replay_optional") == 6
    assert "coarse_dt_sanity" in labels


def test_timestamped_output_path_is_fresh_by_default(tmp_path):
    from tools.run_rk4_integrity_batch import resolve_output_dir

    out = resolve_output_dir(tmp_path / "rk4_integrity_diagnostic", timestamp="20260708_120000", resume=False)

    assert out.name == "rk4_integrity_diagnostic_20260708_120000"
    assert re.match(r"rk4_integrity_diagnostic_\d{8}_\d{6}", out.name)


def test_resume_is_explicit_and_disabled_by_default():
    from tools.run_rk4_integrity_batch import build_parser

    args = build_parser().parse_args(["--out", "x"])

    assert args.resume is False


def test_step_count_derivation_and_final_time_integrity():
    from tools.run_rk4_integrity_batch import expected_steps, validate_time_integrity

    assert expected_steps(1.0, 0.004) == 250
    assert expected_steps(2.0, 0.001) == 2000

    ok = validate_time_integrity({"requested_dt": 0.001, "actual_dt": 0.001, "requested_physical_time": 1.0, "actual_physical_time": 1.0, "expected_steps": 1000, "actual_steps": 1000})
    bad = validate_time_integrity({"requested_dt": 0.001, "actual_dt": 0.001, "requested_physical_time": 1.0, "actual_physical_time": 0.999, "expected_steps": 1000, "actual_steps": 999})

    assert ok == ""
    assert "time/step integrity mismatch" in bad


def test_sample_schedule_always_includes_final_step():
    from tools.run_rk4_integrity_batch import sample_steps_for_times

    steps = sample_steps_for_times(total_steps=1000, dt=0.001, target_times=[0.0, 0.25, 0.5, 0.75, 1.0])

    assert steps[0] == 0
    assert steps[-1] == 1000
    assert 250 in steps


def test_hash_comparison_flags_identical_nonidentical_cases_only():
    from tools.run_rk4_integrity_batch import detect_suspicious_hash_reuse

    rows = [
        {"case_id": "a", "requested_dt": 0.004, "requested_physical_time": 1.0, "final_state_hash": "same", "sampled_history_hash": "same"},
        {"case_id": "b", "requested_dt": 0.001, "requested_physical_time": 1.0, "final_state_hash": "same", "sampled_history_hash": "same"},
        {"case_id": "c", "requested_dt": 0.0005, "requested_physical_time": 1.0, "final_state_hash": "c", "sampled_history_hash": "c"},
    ]

    flags = detect_suspicious_hash_reuse(rows)

    assert flags
    assert "a" in flags[0] and "b" in flags[0]


def test_close_metrics_with_different_hashes_do_not_fail_dt_integrity():
    from tools.run_rk4_integrity_batch import classify_dt_integrity

    rows = [
        {"case_id": "a", "requested_dt": 0.004, "actual_dt": 0.004, "requested_physical_time": 1.0, "actual_physical_time": 1.0, "expected_steps": 250, "actual_steps": 250, "final_state_hash": "a", "sampled_history_hash": "ha"},
        {"case_id": "b", "requested_dt": 0.001, "actual_dt": 0.001, "requested_physical_time": 1.0, "actual_physical_time": 1.0, "expected_steps": 1000, "actual_steps": 1000, "final_state_hash": "b", "sampled_history_hash": "hb"},
    ]

    result = classify_dt_integrity(rows)

    assert result["passed"] is True


def test_rhs_flux_classifier_thresholds():
    from tools.run_rk4_integrity_batch import rhs_flux_status

    assert rhs_flux_status(1e-11) == "numerical_zero"
    assert rhs_flux_status(2e-8) == "warning"
    assert rhs_flux_status(2e-6) == "fail"


def test_n64_t2_requires_twenty_minutes_remaining():
    from tools.run_rk4_integrity_batch import should_run_n64_t2

    assert should_run_n64_t2(20 * 60 + 1) is True
    assert should_run_n64_t2(20 * 60 - 1) is False


def test_final_decisions_are_restricted():
    from tools.run_rk4_integrity_batch import FINAL_DECISIONS

    assert FINAL_DECISIONS == {
        "RK4_DT_INTEGRITY_FAIL",
        "RK4_RHS_FLUX_FAIL",
        "RK4_INTEGRITY_PASS_N64_NOT_RUN",
        "RK4_INTEGRITY_PASS_N64_PROMISING",
        "RK4_INTEGRITY_UNCLEAR",
    }


def test_protected_files_match_required_list():
    from tools.run_rk4_integrity_batch import PROTECTED_FILES

    assert "solver/core.py" in PROTECTED_FILES
    assert "worker_cupy.py" in PROTECTED_FILES
    assert "jax_scout/physics.py" in PROTECTED_FILES
    assert "jax_scout/phase_d_c2_transport.py" in PROTECTED_FILES
