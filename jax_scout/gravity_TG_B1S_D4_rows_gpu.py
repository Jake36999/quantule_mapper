"""TG-B1S D4 row-wise numerical validation runner.

This is a D4-only recovery/deployment runner for the frozen TG-B1S
STATE_LOAD_FEEDBACK model.  It runs the remaining numerical-validation rows
serially and writes a marker after every row so partial progress can be reviewed
without relying on a single final marker.

It does not run D5 and does not alter the model.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

import jax

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jax_scout import gravity_TG_B1S_D4_D5_closure_gpu as close  # noqa: E402
from jax_scout import gravity_TG_B1S_drift_decomposition_gpu as dec  # noqa: E402
from jax_scout import gravity_TG_B1S_state_load_feedback_gpu as b1s  # noqa: E402


from jax_scout.provenance import write_json  # noqa: E402  (shared: stamps run metadata)

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def selected_cases(args: argparse.Namespace, cfg: dict[str, Any]) -> list[tuple[str, dict[str, Any], float]]:
    specs = {
        "output_cadence": ("D4_output_cadence_50P", {}, args.output_cadence_sample_dt),
        "dt_half": ("D4_dt_half_50P", {"dt": cfg["dt"] / 2.0}, args.sample_dt),
        "grid_refined": ("D4_grid_refined_50P", {"N": 56}, args.sample_dt),
        "larger_box": ("D4_larger_box_50P", {"N": 56, "L": 12.0}, args.sample_dt),
        "absorber_wider": ("D4_absorber_wider_50P", {"absorb_width": cfg["absorb_width"] * 1.25}, args.sample_dt),
    }
    out = []
    for key in [x.strip() for x in args.cases.split(",") if x.strip()]:
        if key not in specs:
            raise ValueError(f"unknown D4 case {key!r}; valid={sorted(specs)}")
        out.append(specs[key])
    return out


def import_prior_baseline(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not path:
        return [], [], [], [], [], []
    validation = []
    asym = []
    structural = []
    field = []
    energy = []
    boundary = []
    for row in read_csv(path / "numerical_validation.csv"):
        if row.get("case") == "D4_baseline_50P":
            row["source_run"] = str(path)
            validation.append(row)
    for row in read_csv(path / "asymptotic_frequency.csv"):
        if row.get("run_id") == "D4_baseline_50P":
            row["source_run"] = str(path)
            asym.append(row)
    for row in read_csv(path / "structural_trends.csv"):
        if row.get("run_id") == "D4_baseline_50P":
            row["source_run"] = str(path)
            structural.append(row)
    for row in read_csv(path / "field_boundedness.csv"):
        if row.get("run_id") == "D4_baseline_50P":
            row["source_run"] = str(path)
            field.append(row)
    for row in read_csv(path / "energy_ledger.csv"):
        if row.get("run_id") == "D4_baseline_50P":
            row["source_run"] = str(path)
            energy.append(row)
    for row in read_csv(path / "boundary_flux.csv"):
        if row.get("run_id") == "D4_baseline_50P":
            row["source_run"] = str(path)
            boundary.append(row)
    return validation, asym, structural, field, energy, boundary


def d4_row(
    case: str,
    overrides: dict[str, Any],
    sample_dt: float,
    cfg: dict[str, Any],
    gates: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    vcfg = dict(cfg)
    vcfg.update(overrides)
    vphi, _vprof, vrefs, _vbase = close.solve_phi_refs(vcfg)
    summary, pair_rows, _full_rows, _off_rows = dec.run_pair_decomposition(case, args.d4_periods, 1.0, vcfg, vphi, vrefs, sample_dt)
    structural_rows = dec.drift_channel_rows(case, pair_rows)
    asym = dec.asymptotic_frequency_row(case, pair_rows)
    field = close.field_boundedness(case, pair_rows)
    boundary = close.boundary_flux(case, pair_rows)
    passed, reason = close.pass_d4_row(case, summary, asym, structural_rows, field, boundary, gates)
    ref = float(gates["reference_delta_omega"])
    rel = abs(float(asym.get("delta_omega_infty", np.nan)) - ref) / max(abs(ref), 1e-30)
    structural_channels = close.structural_channels_for(case, structural_rows, gates)
    validation = {
        "case": case,
        "run_id": case,
        "pass": passed,
        "fail_reason": reason,
        "delta_omega_infty": asym.get("delta_omega_infty"),
        "reference_delta_omega": ref,
        "relative_frequency_difference": rel,
        "early_late_relative_difference": asym.get("early_late_relative_difference"),
        "phase_translation_aligned_distance_final": summary["phase_translation_aligned_distance_final"],
        "phase_translation_aligned_distance_max": summary["phase_translation_aligned_distance_max"],
        "profile_overlap": summary["full_profile_overlap"],
        "modal_leakage_max": summary["full_modal_leakage_max"],
        "ledger_residual_abs": summary["ledger_residual_abs"],
        "boundary_flux_proxy_max": boundary["shell_flux_proxy_max"],
        "T_class": field["T_class"],
        "G_class": field["G_class"],
        "structural_channels": ",".join(structural_channels),
        "numerical_accumulation_suspected": summary["phase_translation_aligned_distance_final"] < 0.5 * 4.9140079664143094e-05,
    }
    energy = {
        "run_id": case,
        "ledger_residual_abs": summary["ledger_residual_abs"],
        "T_peak": summary["T_peak"],
        "G_peak": summary["G_peak"],
    }
    return validation, pair_rows, asym, field, energy, boundary, structural_rows


def write_summary(outdir: Path, validation_rows: list[dict[str, Any]], started: str, status: str) -> None:
    passed = all(str(row.get("pass", "")).lower() == "true" or row.get("pass") is True for row in validation_rows)
    failed = [row.get("case") for row in validation_rows if not (str(row.get("pass", "")).lower() == "true" or row.get("pass") is True)]
    label = "TG_STATE_LOAD_FREQUENCY_SHIFT_NUMERICALLY_CONVERGED_BASIN_OPEN" if passed else "TG_STATE_LOAD_D4_NUMERICAL_VALIDATION_FAILED"
    payload = {
        "status": status,
        "label": label,
        "d4_passed": passed,
        "completed_cases": [row.get("case") for row in validation_rows],
        "failed_cases": failed,
        "run_start_time": started,
        "run_end_time": now_iso(),
        "note": "D4-only row-wise runner; D5 not run.",
    }
    write_json(outdir / "D4_RUN_COMPLETE.json", payload)
    write_json(outdir / "D4_SUMMARY.json", payload)
    (outdir / "TECHNICAL_HANDOFF.md").write_text(
        "\n".join(
            [
                "# TG-B1S D4 Row-wise Technical Handoff",
                "",
                f"Status: `{status}`.",
                f"Label: `{label}`.",
                f"D4 passed: `{passed}`.",
                f"Completed cases: `{', '.join(str(x) for x in payload['completed_cases'])}`.",
                f"Failed cases: `{', '.join(str(x) for x in failed)}`.",
                "",
                "D5 was intentionally not run by this D4-only recovery runner.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference-run", required=True)
    ap.add_argument("--import-baseline-run", default="")
    ap.add_argument("--cases", default="output_cadence,dt_half,grid_refined,larger_box,absorber_wider")
    ap.add_argument("--d4-periods", type=float, default=50.0)
    ap.add_argument("--sample-dt", type=float, default=1.0)
    ap.add_argument("--output-cadence-sample-dt", type=float, default=2.0)
    ap.add_argument("--N", type=int, default=48)
    ap.add_argument("--L", type=float, default=10.0)
    ap.add_argument("--c", type=float, default=0.5477)
    ap.add_argument("--m", type=float, default=1.0)
    ap.add_argument("--a", type=float, default=0.8)
    ap.add_argument("--s", type=float, default=-0.5)
    ap.add_argument("--f", type=float, default=-0.1)
    ap.add_argument("--w", type=float, default=0.964)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--alpha-T", type=float, default=0.35)
    ap.add_argument("--omega-T", type=float, default=1.25)
    ap.add_argument("--omega-G", type=float, default=0.85)
    ap.add_argument("--gamma-T", type=float, default=0.08)
    ap.add_argument("--gamma-G", type=float, default=0.06)
    ap.add_argument("--kappa-TG", type=float, default=0.55)
    ap.add_argument("--epsilon-G", type=float, default=0.06)
    ap.add_argument("--cT", type=float, default=0.7)
    ap.add_argument("--cG", type=float, default=0.55)
    ap.add_argument("--absorb-width", type=float, default=1.6)
    ap.add_argument("--absorb-strength", type=float, default=0.02)
    ap.add_argument("--core-radius", type=float, default=2.0)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    started = now_iso()
    try:
        (outdir / "git_state_before.txt").write_text(b1s.git_state(), encoding="utf-8")
        pf = b1s.preflight(outdir)
        cfg = close.default_config(args)
        ref_delta = close.reference_delta_omega(Path(args.reference_run))
        gates = close.default_gates(ref_delta)
        write_json(outdir / "environment_versions.json", {**pf, "stage": "TG-B1S_D4_ROW_WISE"})
        write_json(outdir / "model_spec.json", {"source": "STATE_LOAD_FEEDBACK", "S0": b1s.S0_AUTHORITATIVE, "disabled_sources": ["R_relax", "L_lock", "P_threshold"], "frozen_model": True})
        write_json(outdir / "preregistered_gates.json", gates)
        write_json(outdir / "preregistered_matrix.json", {"D4_cases": args.cases, "d4_periods": args.d4_periods, "reference_run": args.reference_run, "import_baseline_run": args.import_baseline_run})
        phi, prof, refs, base = close.solve_phi_refs(cfg)
        write_csv(outdir / "baseline_reproduction.csv", [{**base, "reference_delta_omega": ref_delta}])
        validation_rows, asym_rows, structural_rows, field_rows, energy_rows, boundary_rows = import_prior_baseline(Path(args.import_baseline_run)) if args.import_baseline_run else ([], [], [], [], [], [])
        orbital_rows: list[dict[str, Any]] = []
        phase_rows: list[dict[str, Any]] = []
        attractor_rows: list[dict[str, Any]] = []
        write_csv(outdir / "numerical_validation.csv", validation_rows)
        write_csv(outdir / "asymptotic_frequency.csv", asym_rows)
        write_csv(outdir / "structural_trends.csv", structural_rows)
        write_csv(outdir / "field_boundedness.csv", field_rows)
        write_csv(outdir / "energy_ledger.csv", energy_rows)
        write_csv(outdir / "boundary_flux.csv", boundary_rows)
        for case, overrides, sample_dt in selected_cases(args, cfg):
            write_json(outdir / f"ROW_{case}_STARTED.json", {"case": case, "start": now_iso()})
            validation, pairs, asym, field, energy, boundary, structural = d4_row(case, overrides, sample_dt, cfg, gates, args)
            validation_rows.append(validation)
            asym_rows.append(asym)
            structural_rows.extend(structural)
            field_rows.append(field)
            energy_rows.append(energy)
            boundary_rows.append(boundary)
            orbital_rows.extend(pairs)
            phase_rows.extend(
                {
                    "run_id": row["run_id"],
                    "t": row["t"],
                    "period": row["period"],
                    "alpha_star": row["alpha_star"],
                    "alpha_star_translation": row["alpha_star_translation"],
                    "delta_theta": row["delta_theta"],
                    "delta_omega_inst": row["delta_omega_inst"],
                }
                for row in pairs
            )
            attractor_rows.append({"stage": "D4", "class": "ORBITALLY_STABLE_FREQUENCY_SHIFT" if validation["pass"] else "D4_VALIDATION_FAILED", **validation})
            write_csv(outdir / "numerical_validation.csv", validation_rows)
            write_csv(outdir / "asymptotic_frequency.csv", asym_rows)
            write_csv(outdir / "structural_trends.csv", structural_rows)
            write_csv(outdir / "field_boundedness.csv", field_rows)
            write_csv(outdir / "energy_ledger.csv", energy_rows)
            write_csv(outdir / "boundary_flux.csv", boundary_rows)
            write_csv(outdir / "orbital_distance.csv", orbital_rows)
            write_csv(outdir / "phase_alignment.csv", phase_rows)
            write_csv(outdir / "attractor_classification.csv", attractor_rows)
            write_json(outdir / f"ROW_{case}_COMPLETE.json", {"case": case, "complete": now_iso(), "pass": validation["pass"], "fail_reason": validation["fail_reason"]})
        falsification = [{"test": "D4_numerical_validation", "pass": all(str(row.get("pass", "")).lower() == "true" or row.get("pass") is True for row in validation_rows)}]
        write_csv(outdir / "falsification_results.csv", falsification)
        (outdir / "git_state_after.txt").write_text(b1s.git_state(), encoding="utf-8")
        write_csv(outdir / "artifact_hashes.csv", b1s.artifact_hashes(outdir))
        write_summary(outdir, validation_rows, started, "D4_ROW_WISE_COMPLETE")
    except Exception as exc:  # pragma: no cover - overnight failure capture.
        tb = traceback.format_exc()
        write_json(outdir / "D4_RUN_FAILED.json", {"status": "D4_RUN_FAILED", "run_start_time": started, "run_end_time": now_iso(), "error": str(exc), "traceback": tb})
        (outdir / "DISCREPANCY_REPORT.md").write_text(f"# TG-B1S D4 Row-wise Discrepancy Report\n\n```text\n{tb}\n```\n", encoding="utf-8")
        try:
            (outdir / "git_state_after.txt").write_text(b1s.git_state(), encoding="utf-8")
            write_csv(outdir / "artifact_hashes.csv", b1s.artifact_hashes(outdir))
        finally:
            raise


if __name__ == "__main__":
    main()
