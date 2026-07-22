from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.external_validation.io_utils import (  # noqa: E402
    BEHAVIOURAL_DECISIONS,
    FINAL_DECISIONS,
    discover_git_commit,
    ensure_empty_guardrail_text,
    load_config,
    protected_diff,
    repo_path,
    write_json,
    write_text,
)
from tools.external_validation.depth_audit import write_depth_audit  # noqa: E402
from tools.external_validation.external_data import write_external_data_manifest  # noqa: E402
from tools.external_validation.metrics import (  # noqa: E402
    v1_phase_force_fit,
    v2_galilean_transport,
    v3_vk_branch,
    v5_collision_phase_grid,
    v6_old_vs_corrected_c2,
    v7_omega_exponent_diagnosis,
)
from tools.external_validation.report_writer import suite_report  # noqa: E402


METRIC_RUNNERS: dict[str, Callable] = {
    "v1": v1_phase_force_fit.run,
    "v2": v2_galilean_transport.run,
    "v3": v3_vk_branch.run,
    "v5": v5_collision_phase_grid.run,
    "v6": v6_old_vs_corrected_c2.run,
    "v7": v7_omega_exponent_diagnosis.run,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run read-only external-validation metric harnesses.")
    parser.add_argument("--metric", default="all", choices=["all", *METRIC_RUNNERS.keys()])
    parser.add_argument("--input", default=None, help="Optional input path override for a single metric.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("validation_config.yaml")))
    parser.add_argument("--out", default=None, help="Output root; defaults to config output_root.")
    parser.add_argument("--no-plots", action="store_true", help="Skip optional PNG plots.")
    parser.add_argument(
        "--mode",
        default="internal_analytic",
        choices=["internal_analytic", "external_data", "audit_depth"],
        help="internal_analytic compares project data to analytic laws; external_data only uses ready external datasets; audit_depth classifies existing result depth.",
    )
    return parser.parse_args(argv)


def select_metrics(metric: str) -> list[str]:
    return list(METRIC_RUNNERS) if metric == "all" else [metric]


def classify_suite(results: list[dict]) -> str:
    if any(r.get("status") == "FAILED" for r in results):
        return "VALIDATION_HARNESS_BLOCKED"
    if any(r.get("behavioural_result") in {"PARTIAL", "FAIL", "INCONCLUSIVE"} for r in results):
        return "VALIDATION_HARNESS_READY_WITH_GAPS"
    if results and all(r.get("status") == "RAN" for r in results):
        return "VALIDATION_HARNESS_READY_WITH_RESULTS"
    return "VALIDATION_HARNESS_READY_WITH_GAPS"


def classify_behavioural(results: list[dict]) -> str:
    if any(r.get("status") == "FAILED" for r in results):
        return "BEHAVIOURAL_VALIDATION_BLOCKED"
    if any(r.get("status") in {"RAN_WEAK", "SKIPPED_WITH_REASON"} for r in results):
        return "BEHAVIOURAL_VALIDATION_READY_WITH_GAPS"
    if any(r.get("behavioural_result") in {"PARTIAL", "FAIL", "INCONCLUSIVE"} for r in results):
        return "BEHAVIOURAL_VALIDATION_READY_WITH_GAPS"
    return "BEHAVIOURAL_VALIDATION_READY_FOR_CLAUDE_REVIEW"


def write_behavioural_report(results: list[dict], decision: str, external_data_status: dict | None = None) -> None:
    assert decision in BEHAVIOURAL_DECISIONS
    lines = [
        "# External Validation Behavioural Comparison Report",
        "",
        "All results remain `PROVISIONAL_UNTIL_CLAUDE_REVIEW`.",
        "",
        f"Final decision: `{decision}`",
        "",
        "## Depth Audit Result",
        "",
        "The harness distinguishes formalism alignment from measured-data behavioural comparison. IRER formulations remain primary; external models are comparison instruments, not replacements.",
        "",
        "## Numeric Summary",
        "",
        "| Metric | Status | Depth | Behavioural result | Key numeric fields |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        fq = r.get("fit_quality") or {}
        keys = [
            "phase_sign_accuracy",
            "slope_percent_error",
            "r_squared",
            "dQ_domega_fit",
            "anti_phase_transmission_cells",
            "corrected_v_over_2Dk",
            "fitted_effective_exponent",
            "cliff_detected",
        ]
        vals = ", ".join(f"{k}={fq[k]}" for k in keys if k in fq)
        lines.append(
            f"| `{r.get('metric_id')}` | `{r.get('status')}` | `{r.get('validation_depth')}` | "
            f"`{r.get('behavioural_result')}` | {vals or 'see result.json'} |"
        )
    lines += [
        "",
        "## External-Data Readiness",
        "",
    ]
    if external_data_status:
        lines.append(f"- External data mode status: `{external_data_status.get('status')}`")
        lines.append(f"- Reason: {external_data_status.get('reason')}")
    else:
        lines.append("- External machine-readable dynamics correlation was not run in `internal_analytic` mode.")
    lines += [
        "",
        "## Limitations",
        "",
        "- V1 uses measured separation tracks and free analytic-law fits; it does not assert canonical pure-NLS constants.",
        "- V5 uses existing collision labels and telemetry only; it does not reinterpret outcomes.",
        "- V7 is a design-constraint and geometry-law diagnostic, not a gravity result.",
        "- No external dataset is promoted beyond the readiness labels in `DATASET_CANDIDATES.md`.",
        "",
        "## Claude Review Checklist",
        "",
        "- Review whether each behavioural threshold is appropriate.",
        "- Review V1 acceleration-window choice and whether later fits should use more robust trajectory models.",
        "- Review whether any external dataset should be digitised before Level-3 comparison.",
        "",
        "Required statements:",
        "",
        "- No simulations were run.",
        "- No production physics changed.",
        "- No verdicts changed.",
        "- No physical-correspondence claims added.",
        "- IRER formulations were preserved as primary.",
    ]
    write_text("docs/external_validation/EXTERNAL_VALIDATION_BEHAVIOURAL_COMPARISON_REPORT.md", "\n".join(lines) + "\n")


def run_suite(args: argparse.Namespace) -> tuple[list[dict], str, Path]:
    config = load_config(args.config)
    out_root = repo_path(args.out or config.get("output_root", "docs/external_validation/generated_metrics"))
    out_root.mkdir(parents=True, exist_ok=True)
    mode = getattr(args, "mode", "internal_analytic")
    if mode == "external_data":
        external_status = write_external_data_manifest()
        write_behavioural_report([], "BEHAVIOURAL_VALIDATION_READY_WITH_GAPS", external_status)
        return [], "VALIDATION_HARNESS_READY_WITH_GAPS", out_root
    if mode == "audit_depth":
        write_depth_audit(out_root)
        write_behavioural_report([], "BEHAVIOURAL_VALIDATION_READY_WITH_GAPS")
        return [], "VALIDATION_HARNESS_READY_WITH_GAPS", out_root
    metric_ids = select_metrics(args.metric)
    results: list[dict] = []
    for metric_id in metric_ids:
        runner = METRIC_RUNNERS[metric_id]
        override = args.input if args.metric != "all" else None
        try:
            results.append(runner(config, out_root, override, make_plots=not args.no_plots))
        except Exception as exc:
            results.append(
                {
                    "metric_id": metric_id,
                    "status": "FAILED",
                    "data_sources": [],
                    "files_read": [],
                    "formula_target": "",
                    "fit_parameters": {},
                    "fit_quality": {},
                    "caveats": [f"Metric failed: {exc}"],
                    "match_level_candidate": "UNVERIFIED",
                    "validation_depth": "PLACEHOLDER_OR_WEAK",
                    "behavioural_result": "INCONCLUSIVE",
                    "no_new_simulation": True,
                    "provisional_until_claude_review": True,
                }
            )
    decision = classify_suite(results)
    assert decision in FINAL_DECISIONS
    manifest = {
        "git_commit": discover_git_commit(),
        "metrics": results,
        "final_decision": decision,
        "no_new_simulation": True,
        "provisional_until_claude_review": True,
    }
    write_json(out_root / "manifest.json", manifest)
    diff = protected_diff()
    report = suite_report(
        out_root,
        results,
        commands=[" ".join([sys.executable, *sys.argv])],
        protected_diff=diff,
        decision=decision,
    )
    forbidden = ensure_empty_guardrail_text(report)
    if forbidden:
        raise RuntimeError(f"Forbidden guardrail phrases in report: {forbidden}")
    write_text("docs/external_validation/EXTERNAL_VALIDATION_AUTOMATION_REPORT.md", report)
    write_text(out_root / "protected_diff.txt", diff)
    depth_rows = write_depth_audit(out_root)
    external_status = write_external_data_manifest()
    behavioural_decision = classify_behavioural(results)
    write_behavioural_report(results, behavioural_decision, external_status)
    return results, decision, out_root


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results, decision, out_root = run_suite(args)
    print(f"external validation decision: {decision}")
    print(f"output root: {out_root}")
    for r in results:
        print(f"{r.get('metric_id')}: {r.get('status')} ({r.get('match_level_candidate')})")
    return 0 if decision != "VALIDATION_HARNESS_BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
