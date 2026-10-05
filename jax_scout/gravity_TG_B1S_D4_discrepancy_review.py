"""CPU review for the TG-B1S D4 larger-box discrepancy.

This is a post-processing script only. It does not evolve fields and does not
change the frozen state-load model. It collects the completed D4 row artifacts,
Claude's FC-box analytic discriminator, and the D4 gate outputs into one
machine-readable review bundle.

Use when the queue row `CX_TG_B1S_D4_LARGER_BOX_DISCREPANCY_REVIEW` is selected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def f(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return float(value)
    except Exception:
        return default


def row_by_id(rows: list[dict[str, str]], run_id: str) -> dict[str, str]:
    return next((row for row in rows if row.get("run_id") == run_id or row.get("case") == run_id), {})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_hashes(outdir: Path) -> list[dict[str, str]]:
    rows = []
    for path in sorted(outdir.rglob("*")):
        if path.is_file() and path.name != "artifact_hashes.csv":
            rows.append({"path": str(path.relative_to(outdir)).replace("\\", "/"), "sha256": sha256(path)})
    return rows


def git_text(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT)
    except Exception as exc:
        return f"COMMAND_FAILED {args}: {exc}\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "sweep_runs" / f"TG_B1S_D4_DISCREPANCY_REVIEW_{time.strftime('%Y%m%d_%H%M%S')}"))
    parser.add_argument("--d4-run", default=str(ROOT / "sweep_runs" / "TG_B1S_D4_ROWS_GPU_20260715_091558"))
    parser.add_argument("--fc-box-run", default=str(ROOT / "sweep_runs" / "TG_B1S_BOX_DEPENDENCE_20260715_222213"))
    parser.add_argument("--reference-run", default=str(ROOT / "sweep_runs" / "TG_B1S_DRIFT_DECOMPOSITION_GPU_20260714_184736"))
    args = parser.parse_args()

    outdir = Path(args.out)
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    d4_run = Path(args.d4_run)
    fc_run = Path(args.fc_box_run)
    reference_run = Path(args.reference_run)

    (outdir / "git_state_before.txt").write_text(git_text(["git", "status", "--short"]), encoding="utf-8")
    write_json(
        outdir / "review_inputs.json",
        {
            "d4_run": str(d4_run),
            "fc_box_run": str(fc_run),
            "reference_run": str(reference_run),
            "command": " ".join([sys.executable, *sys.argv]),
            "scope": "CPU post-processing only; no field evolution",
        },
    )

    numerical = read_csv(d4_run / "numerical_validation.csv")
    asym = read_csv(d4_run / "asymptotic_frequency.csv")
    structural = read_csv(d4_run / "structural_trends.csv")
    energy = read_csv(d4_run / "energy_ledger.csv")
    boundary = read_csv(d4_run / "boundary_flux.csv")
    field = read_csv(d4_run / "field_boundedness.csv")
    fc_summary = json.loads((fc_run / "summary.json").read_text(encoding="utf-8")) if (fc_run / "summary.json").exists() else {}

    cases = [
        "D4_baseline_50P",
        "D4_output_cadence_50P",
        "D4_dt_half_50P",
        "D4_grid_refined_50P",
        "D4_larger_box_50P",
        "D4_absorber_wider_50P",
    ]
    rows = []
    baseline = row_by_id(numerical, "D4_baseline_50P")
    base_omega = f(baseline, "delta_omega_infty")
    for case in cases:
        nrow = row_by_id(numerical, case)
        arow = row_by_id(asym, case)
        brow = row_by_id(boundary, case)
        erow = row_by_id(energy, case)
        fld = row_by_id(field, case)
        delta = f(nrow, "delta_omega_infty")
        rows.append(
            {
                "case": case,
                "d4_pass": nrow.get("pass", ""),
                "fail_reason": nrow.get("fail_reason", ""),
                "delta_omega_infty": delta,
                "ratio_to_d4_baseline": delta / base_omega if base_omega and base_omega == base_omega else "",
                "relative_frequency_difference": nrow.get("relative_frequency_difference", ""),
                "early_late_relative_difference": nrow.get("early_late_relative_difference", ""),
                "phase_translation_aligned_distance_final": nrow.get("phase_translation_aligned_distance_final", ""),
                "profile_overlap": nrow.get("profile_overlap", ""),
                "modal_leakage_max": nrow.get("modal_leakage_max", ""),
                "ledger_residual_abs": nrow.get("ledger_residual_abs", ""),
                "boundary_flux_proxy_max": nrow.get("boundary_flux_proxy_max", brow.get("boundary_flux_proxy_max", "")),
                "T_class": nrow.get("T_class", fld.get("T_class", "")),
                "G_class": nrow.get("G_class", fld.get("G_class", "")),
                "constant_slope_supported": arow.get("constant_slope_supported", ""),
                "energy_row_present": bool(erow),
            }
        )

    fc_geoms = fc_summary.get("geometries", [])
    fc_rows = []
    for geom in fc_geoms:
        fc_rows.append(
            {
                "name": geom.get("name"),
                "N": geom.get("N"),
                "L": geom.get("L"),
                "dx": geom.get("dx"),
                "analytic_d_omega_fixed_profile": geom.get("d_omega_fp"),
                "analytic_ratio_to_baseline": geom.get("analytic_ratio_to_baseline"),
                "measured_d_omega": geom.get("measured_d_omega"),
                "screening_length_fit": geom.get("screening_length_fit"),
                "qball_residual": geom.get("residual"),
                "int_grad2": geom.get("int_grad2"),
            }
        )

    larger = row_by_id(numerical, "D4_larger_box_50P")
    grid = row_by_id(numerical, "D4_grid_refined_50P")
    absorber = row_by_id(numerical, "D4_absorber_wider_50P")
    larger_failed_only_frequency = (
        str(larger.get("pass")) == "False"
        and larger.get("fail_reason") == "frequency_scale"
        and not larger.get("structural_channels")
    )
    grid_passed = str(grid.get("pass")) == "True"
    absorber_passed = str(absorber.get("pass")) == "True"
    analytic_ratio = float(fc_summary.get("analytic_larger_box_over_baseline", "nan")) if fc_summary else float("nan")
    measured_ratio = float(fc_summary.get("measured", {}).get("measured_larger_box_over_baseline", "nan")) if fc_summary else float("nan")
    analytic_contradicts_measured_drop = bool(
        analytic_ratio == analytic_ratio
        and measured_ratio == measured_ratio
        and analytic_ratio > 1.0
        and measured_ratio < 1.0
    )
    recommendation = {
        "status": "D4_LARGER_BOX_DISCREPANCY_REVIEWED",
        "d5_should_remain_blocked": True,
        "larger_box_failed_only_frequency_scale": larger_failed_only_frequency,
        "grid_refined_passed": grid_passed,
        "absorber_wider_passed": absorber_passed,
        "analytic_larger_box_over_baseline": analytic_ratio,
        "measured_larger_box_over_baseline": measured_ratio,
        "analytic_contradicts_measured_drop": analytic_contradicts_measured_drop,
        "primary_inference": "D4 larger-box failure is not explained by resolution or the local fixed-profile frequency mechanism. It is most consistent with a geometry/branch/transient/boundary-comparability issue requiring a targeted rerun or stricter same-geometry interpretation.",
        "recommended_next": "Queue a targeted same-geometry L=12 baseline/off/full rerun or absorber/relief isolation row before D5. Do not relax the D4 gate post hoc.",
        "bounded_labels_only": [
            "TG_STATE_LOAD_D4_NUMERICAL_VALIDATION_FAILED",
            "TG_STATE_LOAD_LONG_TIME_DRIFT_UNRESOLVED",
        ],
    }

    write_csv(outdir / "d4_case_comparison.csv", rows)
    write_csv(outdir / "fc_box_mechanism_comparison.csv", fc_rows)
    write_json(outdir / "recommendation.json", recommendation)
    write_csv(
        outdir / "falsification_results.csv",
        [
            {"test": "larger_box_frequency_gate", "pass": False, "evidence": larger.get("fail_reason", "")},
            {"test": "resolution_explanation", "pass": False, "evidence": "grid_refined passed and FC-box dx effect was negligible"},
            {"test": "local_static_mechanism_explains_drop", "pass": False, "evidence": "analytic ratio rises while measured ratio drops"},
            {"test": "structural_drift_in_larger_box", "pass": False, "evidence": "larger-box row remained structurally clean"},
        ],
    )
    handoff = [
        "# TG-B1S D4 Larger-Box Discrepancy Review",
        "",
        f"Run directory: `{outdir.as_posix()}`.",
        "",
        "This is a CPU post-processing review. No field evolution was run.",
        "",
        "## Finding",
        "",
        "- `D4_larger_box_50P` remains a real D4 gate failure on frequency scale.",
        "- The row is structurally clean: high overlap, bounded T/G classes, low leakage, and no structural-channel fail.",
        "- The grid-refined and absorber-wider rows pass, so the failure is specific to the larger-box geometry/branch/transient comparison.",
        "- FC-box predicts the local fixed-profile mechanism should increase in the larger box, while the measured D4 shift decreases.",
        "",
        "## Boundary",
        "",
        "This does not falsify the TG-B1S feed-forward chain or robust-but-weak backreaction. It blocks D5-style promotion until a targeted same-geometry or relief/boundary audit is run.",
    ]
    (outdir / "TECHNICAL_HANDOFF.md").write_text("\n".join(handoff) + "\n", encoding="utf-8")
    (outdir / "OPEN_QUESTIONS.md").write_text(
        "# Open Questions\n\n"
        "- Is the larger-box row selecting a different Petviashvili branch/profile despite same nominal Q-ball target?\n"
        "- Does absorber radius alter T/G tail ring-in or boundary energy accounting over 50 periods?\n"
        "- Would an L=12 row compared only to a local L=12 reference clear the frequency-scale gate?\n",
        encoding="utf-8",
    )
    (outdir / "git_state_after.txt").write_text(git_text(["git", "status", "--short"]), encoding="utf-8")
    write_csv(outdir / "artifact_hashes.csv", artifact_hashes(outdir))
    write_json(outdir / "RUN_COMPLETE.json", {"status": "D4_LARGER_BOX_DISCREPANCY_REVIEWED", "outdir": str(outdir)})
    print(json.dumps({"status": "D4_LARGER_BOX_DISCREPANCY_REVIEWED", "outdir": str(outdir)}, indent=2))


if __name__ == "__main__":
    main()
