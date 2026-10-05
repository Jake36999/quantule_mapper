from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np


def test_config_defaults_include_expected_metric_paths():
    from tools.external_validation.io_utils import default_config

    cfg = default_config()
    assert "sweep_runs/C29_ROBUST" in cfg["metrics"]["v1"]["paths"]
    assert "sweep_runs/C27_REDERIVE" in cfg["metrics"]["v2"]["paths"]
    assert "sweep_runs/C3_EXACT_VK2" in cfg["metrics"]["v3"]["paths"]


def test_v1_estimates_known_quadratic_acceleration():
    from tools.external_validation.metrics.v1_phase_force_fit import estimate_qddot, expected_sign

    t = np.linspace(0.0, 1.0, 8)
    q = 3.0 - 0.25 * t**2
    sep = 2.0 * q
    assert np.isclose(estimate_qddot(t, sep), -0.5)
    assert expected_sign(0.0) == -1
    assert expected_sign(np.pi) == 1
    assert expected_sign(np.pi / 2) == 0


def test_v2_linear_fit_reports_exact_line():
    from tools.external_validation.metrics.v2_galilean_transport import fit_line

    x = np.array([0.5, 1.0, 1.5])
    y = 2.0 * x
    fit = fit_line(x, y)
    assert np.isclose(fit["slope"], 2.0)
    assert np.isclose(fit["intercept"], 0.0)
    assert np.isclose(fit["r2"], 1.0)


def test_v7_excludes_saturated_regions_for_exponent():
    from tools.external_validation.metrics.v7_omega_exponent_diagnosis import estimate_exponent

    rho = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.1])
    omega = np.array([1.0, 1.25, 1.66, 2.5, 100.0, 100.0])
    fit = estimate_exponent(rho, omega)
    assert fit["fit_available"] is True
    assert fit["cap_fraction"] > 0.0


def test_runner_selects_all_metrics_and_decision_labels_are_restricted():
    from tools.external_validation.io_utils import BEHAVIOURAL_DECISIONS, FINAL_DECISIONS
    from tools.external_validation.run_external_validation import classify_behavioural, classify_suite, select_metrics

    assert select_metrics("all") == ["v1", "v2", "v3", "v5", "v6", "v7"]
    assert classify_suite([{"status": "RAN"}]) == "VALIDATION_HARNESS_READY_WITH_RESULTS"
    assert classify_suite([{"status": "RAN_WEAK"}]) == "VALIDATION_HARNESS_READY_WITH_GAPS"
    assert classify_suite([{"status": "SKIPPED_WITH_REASON"}]) == "VALIDATION_HARNESS_READY_WITH_GAPS"
    assert classify_suite([{"status": "FAILED"}]) == "VALIDATION_HARNESS_BLOCKED"
    assert classify_behavioural([{"status": "RAN", "behavioural_result": "PASS"}]) == "BEHAVIOURAL_VALIDATION_READY_FOR_CLAUDE_REVIEW"
    assert classify_behavioural([{"status": "RAN_WEAK", "behavioural_result": "FAIL"}]) == "BEHAVIOURAL_VALIDATION_READY_WITH_GAPS"
    assert FINAL_DECISIONS == {
        "VALIDATION_HARNESS_READY_WITH_RESULTS",
        "VALIDATION_HARNESS_READY_WITH_GAPS",
        "VALIDATION_HARNESS_BLOCKED",
    }
    assert BEHAVIOURAL_DECISIONS == {
        "BEHAVIOURAL_VALIDATION_READY_FOR_CLAUDE_REVIEW",
        "BEHAVIOURAL_VALIDATION_READY_WITH_GAPS",
        "BEHAVIOURAL_VALIDATION_BLOCKED",
    }


def test_metric_skip_writes_required_schema(tmp_path):
    from tools.external_validation.metrics import v2_galilean_transport

    cfg = {"metrics": {"v2": {"paths": [str(tmp_path / "missing")]}}}
    result = v2_galilean_transport.run(cfg, tmp_path, make_plots=False)
    assert result["status"] == "SKIPPED_WITH_REASON"
    for key in [
        "metric_id",
        "status",
        "data_sources",
        "files_read",
        "formula_target",
        "fit_parameters",
        "fit_quality",
        "caveats",
        "match_level_candidate",
        "validation_depth",
        "behavioural_result",
        "no_new_simulation",
        "provisional_until_claude_review",
    ]:
        assert key in result
    assert (tmp_path / "v2" / "result.json").exists()


def test_guardrail_phrase_detector():
    from tools.external_validation.io_utils import ensure_empty_guardrail_text

    assert ensure_empty_guardrail_text("same-family comparison") == []
    assert ensure_empty_guardrail_text("IRER is validated experimentally") == [
        "IRER is validated experimentally"
    ]


def test_runner_can_emit_suite_manifest_with_missing_inputs(tmp_path, monkeypatch):
    import tools.external_validation.run_external_validation as runner

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "output_root: ignored\nmetrics:\n  v2:\n    paths:\n      - definitely_missing\n",
        encoding="utf-8",
    )
    real_write_text = runner.write_text

    def patched_write_text(path, text):
        if str(path).endswith("EXTERNAL_VALIDATION_AUTOMATION_REPORT.md"):
            return real_write_text(tmp_path / "report.md", text)
        return real_write_text(path, text)

    monkeypatch.setattr(runner, "write_text", patched_write_text)
    monkeypatch.setattr(runner, "write_depth_audit", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(runner, "write_behavioural_report", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner,
        "write_external_data_manifest",
        lambda *_args, **_kwargs: {"status": "SKIPPED_WITH_REASON", "reason": "test", "rows": []},
    )
    args = Namespace(metric="v2", input=None, config=str(cfg), out=str(tmp_path / "out"), no_plots=True)
    results, decision, out_root = runner.run_suite(args)
    assert decision == "VALIDATION_HARNESS_READY_WITH_GAPS"
    assert results[0]["metric_id"] == "v2"
    assert (out_root / "manifest.json").exists()
    payload = json.loads((out_root / "manifest.json").read_text())
    assert payload["provisional_until_claude_review"] is True


def test_external_data_manifest_classifies_without_forcing_comparison(tmp_path):
    from tools.external_validation.external_data import write_external_data_manifest

    result = write_external_data_manifest(tmp_path / "external")
    assert result["status"] == "SKIPPED_WITH_REASON"
    classes = {row["classification"] for row in result["rows"]}
    assert "MACHINE_READABLE_NOW" in classes
    assert "DIGITIZATION_REQUIRED" in classes
    assert (tmp_path / "external" / "external_data_manifest.csv").exists()


def test_depth_audit_classifies_existing_result(tmp_path):
    import json

    from tools.external_validation.depth_audit import audit_generated_metrics

    metric_dir = tmp_path / "v2"
    metric_dir.mkdir()
    (metric_dir / "result.json").write_text(
        json.dumps(
            {
                "metric_id": "v2",
                "status": "RAN",
                "validation_depth": "INTERNAL_DATA_BEHAVIOUR_CHECK",
                "behavioural_result": "PASS",
                "fit_quality": {"slope": 2.0},
            }
        ),
        encoding="utf-8",
    )
    (metric_dir / "read_files.txt").write_text("some/file.json\n", encoding="utf-8")
    rows = audit_generated_metrics(tmp_path)
    assert rows[0]["validation_depth"] == "INTERNAL_DATA_BEHAVIOUR_CHECK"
    assert rows[0]["claude_reviewable"] is True
