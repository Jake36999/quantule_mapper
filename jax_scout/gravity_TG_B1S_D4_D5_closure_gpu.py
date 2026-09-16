"""TG-B1S D4/D5 closure runner.

Self-gating overnight closure for the frozen STATE_LOAD_FEEDBACK model.

This script runs only:
- D4 numerical validation of the orbitally bounded frequency-shift result;
- D5 basin-orbital stability, only if D4 passes.

It intentionally reuses the TG-B1S-D drift-decomposition machinery and does not
change the state-load source, normalization, feedback coefficients, damping,
coupling signs, T/G equations, coefficient maps, Q-ball initialization,
absorber formulation or disabled source families.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jax_scout import gravity_TG_B1S_drift_decomposition_gpu as dec  # noqa: E402
from jax_scout import gravity_TG_B1S_long_time_gpu as lt  # noqa: E402
from jax_scout import gravity_TG_B1S_state_load_feedback_gpu as b1s  # noqa: E402


from jax_scout.provenance import write_json  # noqa: E402  (shared: stamps run metadata)

REFERENCE_DELTA_OMEGA_DEFAULT = -2.150936882840006e-06
ORIGINAL_MAKE_INITIAL_STATE_LT = lt.make_initial_state_lt


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


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def patched_make_initial_state_lt(
    phi: np.ndarray,
    cfg: dict[str, Any],
    kind: str = "base",
    t_seed: float = 0.0,
    g_seed: float = 0.0,
) -> tuple[jnp.ndarray, ...]:
    if kind in ("amplitude_plus", "amplitude_minus", "width_plus", "width_minus"):
        if kind == "amplitude_plus":
            psi = (1.001 * phi).astype(np.complex128)
        elif kind == "amplitude_minus":
            psi = (0.999 * phi).astype(np.complex128)
        else:
            op = b1s.build_kg(cfg["N"], cfg["L"], cfg["c"], cfg["m"], cfg["dt"])
            grid = b1s.make_grid(op)
            rr = np.asarray(grid["R"])
            shape = rr * rr / (np.mean(rr * rr) + 1e-30) - 1.0
            sign = 1.0 if kind == "width_plus" else -1.0
            psi = (phi * (1.0 + sign * 1.0e-3 * shape)).astype(np.complex128)
        pi = (-1j * cfg["w"] * psi).astype(np.complex128)
        real_zero = jnp.zeros_like(jnp.real(jnp.asarray(psi)))
        op = b1s.build_kg(cfg["N"], cfg["L"], cfg["c"], cfg["m"], cfg["dt"])
        grid = b1s.make_grid(op)
        seed_shape = jnp.exp(-0.5 * (grid["R"] / 1.5) ** 2)
        T0 = t_seed * seed_shape if t_seed else real_zero
        G0 = g_seed * seed_shape if g_seed else real_zero
        return (jnp.asarray(psi), jnp.asarray(pi), T0, real_zero, G0, real_zero)
    return ORIGINAL_MAKE_INITIAL_STATE_LT(phi, cfg, kind=kind, t_seed=t_seed, g_seed=g_seed)


def install_basin_initial_state_patch() -> None:
    lt.make_initial_state_lt = patched_make_initial_state_lt


def period_from_cfg(cfg: dict[str, Any]) -> float:
    return 2.0 * math.pi / float(cfg["w"])


def default_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "N": args.N,
        "L": args.L,
        "c": args.c,
        "m": args.m,
        "a": args.a,
        "s": args.s,
        "f": args.f,
        "w": args.w,
        "dt": args.dt,
        "T": args.T,
        "sample_dt": args.sample_dt,
        "alpha_T": args.alpha_T,
        "omega_T": args.omega_T,
        "omega_G": args.omega_G,
        "gamma_T": args.gamma_T,
        "gamma_G": args.gamma_G,
        "kappa_TG": args.kappa_TG,
        "epsilon_G": args.epsilon_G,
        "cT": args.cT,
        "cG": args.cG,
        "absorb_width": args.absorb_width,
        "absorb_strength": args.absorb_strength,
        "core_radius": args.core_radius,
    }


def default_gates(reference_delta_omega: float) -> dict[str, Any]:
    return {
        "reference_delta_omega": reference_delta_omega,
        "frequency_relative_tolerance": 0.60,
        "early_late_relative_difference_max": 0.35,
        "forbidden_structural_classifications": ["LINEAR_SECULAR", "QUADRATIC_OR_ACCELERATING"],
        "phase_translation_aligned_distance_max": 2.5e-4,
        "phase_translation_aligned_distance_final": 2.0e-4,
        "profile_overlap_min": 0.999,
        "modal_leakage_max": 1.0e-3,
        "ledger_residual_abs_max": 1.0e-4,
        "boundary_flux_proxy_max": 5.0e-7,
        "bounded_field_classes": ["APPROACH_FIXED_PROFILE", "REMAIN_BOUNDED_OSCILLATORY", "DECAY_TO_NULL"],
        "basin_allowed_classes": [
            "ORBITALLY_STABLE_FREQUENCY_SHIFT",
            "DAMPED_TO_FREQUENCY_SHIFTED_ORBIT",
            "BOUNDED_OSCILLATORY_ORBIT",
        ],
    }


def reference_delta_omega(reference_run: Path) -> float:
    rows = read_csv_dicts(reference_run / "asymptotic_frequency.csv")
    for row in rows:
        if row.get("run_id") == "D3_100P_lam1" and row.get("delta_omega_infty"):
            return float(row["delta_omega_infty"])
    return REFERENCE_DELTA_OMEGA_DEFAULT


def solve_phi_refs(cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any], dict[str, float], dict[str, Any]]:
    phi, prof = b1s.solve_qball(cfg)
    op = b1s.build_kg(cfg["N"], cfg["L"], cfg["c"], cfg["m"], cfg["dt"])
    refs = b1s.ref_source_norms(phi, cfg, op)
    base = b1s.baseline_contract(cfg, phi, prof)
    return phi, prof, refs, base


def structural_channels_for(run_id: str, drift_rows: list[dict[str, Any]], gates: dict[str, Any]) -> list[str]:
    forbidden = set(gates["forbidden_structural_classifications"])
    structural_names = {"delta_A_core", "delta_width", "delta_E_core", "d_orbital", "modal_leakage"}
    return [
        str(row["channel"])
        for row in drift_rows
        if row.get("run_id") == run_id
        and row.get("channel") in structural_names
        and row.get("classification") in forbidden
    ]


def field_boundedness(run_id: str, pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    T_class, T_slope = lt.classify_field_boundedness(pair_rows, "T_peak")
    G_class, G_slope = lt.classify_field_boundedness(pair_rows, "G_peak")
    return {
        "run_id": run_id,
        "T_class": T_class,
        "T_peak_slope": T_slope,
        "T_peak": max(float(r["T_peak"]) for r in pair_rows),
        "G_class": G_class,
        "G_peak_slope": G_slope,
        "G_peak": max(float(r["G_peak"]) for r in pair_rows),
    }


def boundary_flux(run_id: str, pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "shell_flux_proxy_max": max(float(r["shell_flux_proxy"]) for r in pair_rows),
    }


def pass_d4_row(
    run_id: str,
    summary: dict[str, Any],
    asym: dict[str, Any],
    drift_rows: list[dict[str, Any]],
    field_row: dict[str, Any],
    boundary_row: dict[str, Any],
    gates: dict[str, Any],
) -> tuple[bool, str]:
    ref = float(gates["reference_delta_omega"])
    domega = float(asym.get("delta_omega_infty", float("nan")))
    reasons: list[str] = []
    if not np.isfinite(domega) or np.sign(domega) != np.sign(ref):
        reasons.append("frequency_sign")
    rel = abs(domega - ref) / max(abs(ref), 1e-30)
    if rel > float(gates["frequency_relative_tolerance"]):
        reasons.append("frequency_scale")
    if float(asym.get("early_late_relative_difference", float("inf"))) > float(gates["early_late_relative_difference_max"]):
        reasons.append("early_late_slope")
    structural = structural_channels_for(run_id, drift_rows, gates)
    if structural:
        reasons.append("structural_channels:" + ",".join(structural))
    if float(summary["phase_translation_aligned_distance_max"]) > float(gates["phase_translation_aligned_distance_max"]):
        reasons.append("orbital_distance_max")
    if float(summary["phase_translation_aligned_distance_final"]) > float(gates["phase_translation_aligned_distance_final"]):
        reasons.append("orbital_distance_final")
    if float(summary["full_profile_overlap"]) < float(gates["profile_overlap_min"]):
        reasons.append("profile_overlap")
    if float(summary["full_modal_leakage_max"]) > float(gates["modal_leakage_max"]):
        reasons.append("modal_leakage")
    if float(summary["ledger_residual_abs"]) > float(gates["ledger_residual_abs_max"]):
        reasons.append("ledger")
    if float(boundary_row["shell_flux_proxy_max"]) > float(gates["boundary_flux_proxy_max"]):
        reasons.append("boundary_flux")
    if field_row["T_class"] not in gates["bounded_field_classes"] or field_row["G_class"] not in gates["bounded_field_classes"]:
        reasons.append("T_G_boundedness")
    return not reasons, ";".join(reasons)


def basin_class(
    run_id: str,
    summary: dict[str, Any],
    asym: dict[str, Any],
    drift_rows: list[dict[str, Any]],
    field_row: dict[str, Any],
    gates: dict[str, Any],
) -> str:
    structural = structural_channels_for(run_id, drift_rows, gates)
    if (
        float(summary["full_profile_overlap"]) < float(gates["profile_overlap_min"])
        or float(summary["full_modal_leakage_max"]) > float(gates["modal_leakage_max"])
        or field_row["T_class"] not in gates["bounded_field_classes"]
        or field_row["G_class"] not in gates["bounded_field_classes"]
        or float(summary["ledger_residual_abs"]) > float(gates["ledger_residual_abs_max"])
    ):
        return "DESTABILIZED"
    if structural:
        return "STRUCTURAL_CONTINUOUS_DRIFT"
    channel_rows = [r for r in drift_rows if r.get("run_id") == run_id]
    theta = next((r for r in channel_rows if r.get("channel") == "delta_theta"), {})
    orbital = next((r for r in channel_rows if r.get("channel") == "d_orbital"), {})
    if theta.get("classification") == "LINEAR_SECULAR" and bool(asym.get("constant_slope_supported", False)):
        return "ORBITALLY_STABLE_FREQUENCY_SHIFT"
    if theta.get("classification") in ("SATURATING", "CONSTANT_OFFSET"):
        return "DAMPED_TO_FREQUENCY_SHIFTED_ORBIT"
    if orbital.get("classification") == "BOUNDED_OSCILLATORY":
        return "BOUNDED_OSCILLATORY_ORBIT"
    return "NUMERICALLY_UNRESOLVED"


def write_empty_outputs(outdir: Path) -> None:
    for name in [
        "numerical_validation.csv",
        "asymptotic_frequency.csv",
        "phase_alignment.csv",
        "orbital_distance.csv",
        "structural_trends.csv",
        "basin_orbital_stability.csv",
        "field_boundedness.csv",
        "energy_ledger.csv",
        "boundary_flux.csv",
        "attractor_classification.csv",
        "falsification_results.csv",
    ]:
        if not (outdir / name).exists():
            write_csv(outdir / name, [])


def write_docs(outdir: Path, status: str, labels: list[str], summary: dict[str, Any]) -> None:
    docdir = ROOT / "docs" / "gravity_maturity"
    docdir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# TG-B1S D4/D5 Closure Results",
        "",
        f"Timestamp: {datetime.now().date().isoformat()}.",
        f"Run directory: `{outdir.as_posix()}`.",
        f"Status: `{status}`.",
        "",
        "## Labels",
        "",
        *(f"- `{label}`" for label in labels),
        "",
        "## Summary",
        "",
        f"- D4 passed: `{summary.get('d4_passed')}`.",
        f"- D5 passed: `{summary.get('d5_passed')}`.",
        f"- Final label: `{status}`.",
        "",
        "No gravity, photon, objective-time, geodesic, universal-free-fall, production or IRER-validation claim is made.",
    ]
    (docdir / "TG_B1S_D4_D5_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(docdir / "TG_B1S_D4_D5_SUMMARY.json", {"status": status, "labels": labels, "run_directory": str(outdir), **summary})
    inputs = [
        "# TG-B1S D4/D5 Documentation Inputs",
        "",
        f"- Status: `{status}`.",
        f"- Labels: `{', '.join(labels)}`.",
        f"- Artifacts: `{outdir.as_posix()}`.",
        "- Frozen `STATE_LOAD_FEEDBACK` model only.",
        "- `R_relax`, `L_lock` and `P_threshold` disabled.",
        "- Preserve prior `TG_STATE_LOAD_CONTINUOUS_SLOW_DRIFT` as a historical pre-decomposition label.",
    ]
    (docdir / "TG_B1S_D4_D5_DOCUMENTATION_INPUTS.md").write_text("\n".join(inputs), encoding="utf-8")


def write_handoff(outdir: Path, status: str, labels: list[str], summary: dict[str, Any]) -> None:
    (outdir / "TECHNICAL_HANDOFF.md").write_text(
        "\n".join(
            [
                "# TG-B1S D4/D5 Closure Technical Handoff",
                "",
                f"Status: `{status}`.",
                f"Labels: `{', '.join(labels)}`.",
                f"D4 passed: `{summary.get('d4_passed')}`.",
                f"D5 passed: `{summary.get('d5_passed')}`.",
                "",
                "Frozen state-load model only; no source, coefficient, damping, coupling, absorber, photon, gravity or production claim.",
            ]
        ),
        encoding="utf-8",
    )
    (outdir / "OPEN_QUESTIONS.md").write_text(
        "# TG-B1S D4/D5 Open Questions\n\n- If positive, should the master catalogue record the precise frequency-shift label rather than the broader bounded-feedback label?\n- If D4 or D5 fails, which single numerical or basin branch should be audited next?\n",
        encoding="utf-8",
    )


def marker(
    outdir: Path,
    name: str,
    status: str,
    start_time: str,
    completed_stages: list[str],
    failed_gates: list[str],
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "status": status,
        "completed_stages": completed_stages,
        "failed_gates": failed_gates,
        "run_start_time": start_time,
        "run_end_time": now_iso(),
        "key_artifact_paths": [
            "numerical_validation.csv",
            "basin_orbital_stability.csv",
            "falsification_results.csv",
            "artifact_hashes.csv",
        ],
    }
    if extra:
        payload.update(extra)
    write_json(outdir / name, payload)


def final_label(d4_passed: bool, d5_passed: bool, validation_rows: list[dict[str, Any]], basin_rows: list[dict[str, Any]]) -> tuple[str, list[str]]:
    all_rows = [*validation_rows, *basin_rows]
    if any(str(row.get("numerical_accumulation_suspected", "")).lower() == "true" for row in validation_rows):
        return "TG_STATE_LOAD_DRIFT_NUMERICALLY_INDUCED", ["TG_STATE_LOAD_DRIFT_NUMERICALLY_INDUCED"]
    if any(str(row.get("structural_channels", "")) for row in all_rows):
        return "TG_STATE_LOAD_STRUCTURAL_SLOW_DRIFT_CONFIRMED", ["TG_STATE_LOAD_STRUCTURAL_SLOW_DRIFT_CONFIRMED"]
    if not d4_passed:
        return "TG_STATE_LOAD_D4_NUMERICAL_VALIDATION_FAILED", ["TG_STATE_LOAD_D4_NUMERICAL_VALIDATION_FAILED"]
    if not d5_passed:
        return "TG_STATE_LOAD_D5_BASIN_STABILITY_FAILED", ["TG_STATE_LOAD_D5_BASIN_STABILITY_FAILED"]
    return (
        "TG_STATE_LOAD_ORBITALLY_BOUNDED_FREQUENCY_SHIFT_SUPPORTED",
        [
            "TG_STATE_LOAD_ORBITALLY_BOUNDED_FREQUENCY_SHIFT_SUPPORTED",
            "TG_STATE_LOAD_BOUNDED_FEEDBACK_SUPPORTED",
        ],
    )


def run_campaign(args: argparse.Namespace, outdir: Path) -> None:
    start_time = now_iso()
    install_basin_initial_state_patch()
    reference_run = Path(args.reference_run)
    ref_delta = reference_delta_omega(reference_run)
    gates = default_gates(ref_delta)
    cfg = default_config(args)
    cfg["config_hash"] = b1s.config_hash(cfg)

    (outdir / "git_state_before.txt").write_text(b1s.git_state(), encoding="utf-8")
    preflight = b1s.preflight(outdir)
    write_json(outdir / "environment_versions.json", {**preflight, "stage": "TG-B1S_D4_D5_CLOSURE"})
    write_json(outdir / "model_spec.json", {"source": "STATE_LOAD_FEEDBACK", "S0": b1s.S0_AUTHORITATIVE, "disabled_sources": ["R_relax", "L_lock", "P_threshold"], "frozen_model": True})
    write_json(outdir / "preregistered_gates.json", gates)

    d4_specs = [
        ("D4_baseline_50P", {}, args.sample_dt),
        ("D4_output_cadence_50P", {}, args.output_cadence_sample_dt),
        ("D4_dt_half_50P", {"dt": cfg["dt"] / 2.0}, args.sample_dt),
        ("D4_grid_refined_50P", {"N": 56}, args.sample_dt),
        ("D4_larger_box_50P", {"N": 56, "L": 12.0}, args.sample_dt),
        ("D4_absorber_wider_50P", {"absorb_width": cfg["absorb_width"] * 1.25}, args.sample_dt),
    ]
    d5_specs = [
        ("D5_amp_plus", "amplitude_plus", None, 0.0, 0.0),
        ("D5_amp_minus", "amplitude_minus", None, 0.0, 0.0),
        ("D5_width_plus", "width_plus", None, 0.0, 0.0),
        ("D5_width_minus", "width_minus", None, 0.0, 0.0),
        ("D5_global_phase", "global_phase", None, 0.0, 0.0),
        ("D5_translated", "translated", "translated", 0.0, 0.0),
        ("D5_initial_T_seed", "base", None, 1.0e-5, 0.0),
        ("D5_initial_G_seed", "base", None, 0.0, 1.0e-5),
    ]
    write_json(outdir / "preregistered_matrix.json", {"D4": d4_specs, "D5": d5_specs, "d4_periods": args.d4_periods, "d5_periods": args.d5_periods, "reference_run": str(reference_run), "reference_delta_omega": ref_delta})

    phi, _prof, refs, base = solve_phi_refs(cfg)
    write_csv(outdir / "baseline_reproduction.csv", [{**base, "reference_delta_omega": ref_delta}])
    if base["status"] != "TG_SOURCE_NODE_BASELINE_CLOSED":
        status, labels = "TG_SOURCE_NODE_NOT_STATIONARY", ["TG_SOURCE_NODE_NOT_STATIONARY"]
        write_csv(outdir / "falsification_results.csv", [{"test": "baseline_reproduction", "pass": False, "status": base["status"]}])
        write_docs(outdir, status, labels, {"d4_passed": False, "d5_passed": False})
        write_handoff(outdir, status, labels, {"d4_passed": False, "d5_passed": False})
        write_empty_outputs(outdir)
        write_csv(outdir / "artifact_hashes.csv", b1s.artifact_hashes(outdir))
        (outdir / "git_state_after.txt").write_text(b1s.git_state(), encoding="utf-8")
        marker(outdir, "RUN_FAILED.json", status, start_time, ["preflight"], ["baseline_reproduction"])
        return

    all_orbital: list[dict[str, Any]] = []
    all_phase: list[dict[str, Any]] = []
    all_structural: list[dict[str, Any]] = []
    all_asym: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    energy_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    attractor_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    basin_rows: list[dict[str, Any]] = []
    falsification_rows: list[dict[str, Any]] = []
    completed_stages = ["preflight", "baseline_reproduction"]

    for case, overrides, sample_dt in d4_specs:
        vcfg = dict(cfg)
        vcfg.update(overrides)
        vphi, _vprof, vrefs, _vbase = solve_phi_refs(vcfg)
        summary, pair_rows, _full_rows, _off_rows = dec.run_pair_decomposition(case, args.d4_periods, 1.0, vcfg, vphi, vrefs, sample_dt)
        drift_rows = dec.drift_channel_rows(case, pair_rows)
        asym = dec.asymptotic_frequency_row(case, pair_rows)
        field = field_boundedness(case, pair_rows)
        boundary = boundary_flux(case, pair_rows)
        passed, reason = pass_d4_row(case, summary, asym, drift_rows, field, boundary, gates)
        rel = abs(float(asym.get("delta_omega_infty", np.nan)) - ref_delta) / max(abs(ref_delta), 1e-30)
        structural = structural_channels_for(case, drift_rows, gates)
        row = {
            "case": case,
            "run_id": case,
            "pass": passed,
            "fail_reason": reason,
            "delta_omega_infty": asym.get("delta_omega_infty"),
            "reference_delta_omega": ref_delta,
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
            "structural_channels": ",".join(structural),
            "numerical_accumulation_suspected": summary["phase_translation_aligned_distance_final"] < 0.5 * 4.9140079664143094e-05,
        }
        validation_rows.append(row)
        attractor_rows.append({"run_id": case, "stage": "D4", "class": "ORBITALLY_STABLE_FREQUENCY_SHIFT" if passed else "D4_VALIDATION_FAILED", **row})
        all_orbital.extend(pair_rows)
        all_phase.extend({"run_id": r["run_id"], "t": r["t"], "period": r["period"], "alpha_star": r["alpha_star"], "alpha_star_translation": r["alpha_star_translation"], "delta_theta": r["delta_theta"], "delta_omega_inst": r["delta_omega_inst"]} for r in pair_rows)
        all_structural.extend(drift_rows)
        all_asym.append(asym)
        field_rows.append(field)
        energy_rows.append({"run_id": case, "ledger_residual_abs": summary["ledger_residual_abs"], "T_peak": summary["T_peak"], "G_peak": summary["G_peak"]})
        boundary_rows.append(boundary)
        write_csv(outdir / "numerical_validation.csv", validation_rows)
        write_csv(outdir / "orbital_distance.csv", all_orbital)
        write_csv(outdir / "phase_alignment.csv", all_phase)
        write_csv(outdir / "structural_trends.csv", all_structural)
        write_csv(outdir / "asymptotic_frequency.csv", all_asym)
        write_csv(outdir / "field_boundedness.csv", field_rows)
        write_csv(outdir / "energy_ledger.csv", energy_rows)
        write_csv(outdir / "boundary_flux.csv", boundary_rows)
        write_csv(outdir / "attractor_classification.csv", attractor_rows)

    completed_stages.append("D4")
    d4_passed = all(bool(row["pass"]) for row in validation_rows)
    if not d4_passed and args.stop_on_d4_fail:
        status, labels = final_label(False, False, validation_rows, basin_rows)
        falsification_rows = [{"test": "D4_numerical_validation", "pass": False, "failed_cases": ",".join(row["case"] for row in validation_rows if not row["pass"])}]
        write_csv(outdir / "falsification_results.csv", falsification_rows)
        write_docs(outdir, status, labels, {"d4_passed": False, "d5_passed": False, "failed_d4_cases": [row for row in validation_rows if not row["pass"]]})
        write_handoff(outdir, status, labels, {"d4_passed": False, "d5_passed": False})
        (outdir / "DISCREPANCY_REPORT.md").write_text("# TG-B1S D4/D5 Discrepancy Report\n\nD4 failed; D5 was not launched.\n", encoding="utf-8")
        write_empty_outputs(outdir)
        (outdir / "git_state_after.txt").write_text(b1s.git_state(), encoding="utf-8")
        write_csv(outdir / "artifact_hashes.csv", b1s.artifact_hashes(outdir))
        marker(outdir, "RUN_STOPPED_AT_D4.json", status, start_time, completed_stages, ["D4_numerical_validation"], {"failed_cases": [row["case"] for row in validation_rows if not row["pass"]]})
        return

    for case, kind, mode_kind, t_seed, g_seed in d5_specs:
        summary, pair_rows, _full_rows, _off_rows = dec.run_pair_decomposition(case, args.d5_periods, 1.0, cfg, phi, refs, args.sample_dt, kind=kind, mode_kind=mode_kind, t_seed=t_seed, g_seed=g_seed)
        drift_rows = dec.drift_channel_rows(case, pair_rows)
        asym = dec.asymptotic_frequency_row(case, pair_rows)
        field = field_boundedness(case, pair_rows)
        boundary = boundary_flux(case, pair_rows)
        cls = basin_class(case, summary, asym, drift_rows, field, gates)
        passed = cls in gates["basin_allowed_classes"]
        structural = structural_channels_for(case, drift_rows, gates)
        row = {
            "case": case,
            "run_id": case,
            "class": cls,
            "pass": passed,
            "delta_omega_infty": asym.get("delta_omega_infty"),
            "phase_translation_aligned_distance_final": summary["phase_translation_aligned_distance_final"],
            "phase_translation_aligned_distance_max": summary["phase_translation_aligned_distance_max"],
            "profile_overlap": summary["full_profile_overlap"],
            "modal_leakage_max": summary["full_modal_leakage_max"],
            "ledger_residual_abs": summary["ledger_residual_abs"],
            "boundary_flux_proxy_max": boundary["shell_flux_proxy_max"],
            "T_class": field["T_class"],
            "G_class": field["G_class"],
            "structural_channels": ",".join(structural),
        }
        basin_rows.append(row)
        attractor_rows.append({"stage": "D5", **row})
        all_orbital.extend(pair_rows)
        all_phase.extend({"run_id": r["run_id"], "t": r["t"], "period": r["period"], "alpha_star": r["alpha_star"], "alpha_star_translation": r["alpha_star_translation"], "delta_theta": r["delta_theta"], "delta_omega_inst": r["delta_omega_inst"]} for r in pair_rows)
        all_structural.extend(drift_rows)
        all_asym.append(asym)
        field_rows.append(field)
        energy_rows.append({"run_id": case, "ledger_residual_abs": summary["ledger_residual_abs"], "T_peak": summary["T_peak"], "G_peak": summary["G_peak"]})
        boundary_rows.append(boundary)
        write_csv(outdir / "basin_orbital_stability.csv", basin_rows)
        write_csv(outdir / "orbital_distance.csv", all_orbital)
        write_csv(outdir / "phase_alignment.csv", all_phase)
        write_csv(outdir / "structural_trends.csv", all_structural)
        write_csv(outdir / "asymptotic_frequency.csv", all_asym)
        write_csv(outdir / "field_boundedness.csv", field_rows)
        write_csv(outdir / "energy_ledger.csv", energy_rows)
        write_csv(outdir / "boundary_flux.csv", boundary_rows)
        write_csv(outdir / "attractor_classification.csv", attractor_rows)

    completed_stages.append("D5")
    d5_passed = all(bool(row["pass"]) for row in basin_rows)
    status, labels = final_label(d4_passed, d5_passed, validation_rows, basin_rows)
    falsification_rows = [
        {"test": "D4_numerical_validation", "pass": d4_passed},
        {"test": "D5_basin_orbital_stability", "pass": d5_passed},
        {"test": "bounded_feedback_secondary", "pass": status == "TG_STATE_LOAD_ORBITALLY_BOUNDED_FREQUENCY_SHIFT_SUPPORTED" and d4_passed and d5_passed},
    ]
    write_csv(outdir / "falsification_results.csv", falsification_rows)
    write_docs(outdir, status, labels, {"d4_passed": d4_passed, "d5_passed": d5_passed, "numerical_validation": validation_rows, "basin_orbital_stability": basin_rows})
    write_handoff(outdir, status, labels, {"d4_passed": d4_passed, "d5_passed": d5_passed})
    (outdir / "DISCREPANCY_REPORT.md").write_text("# TG-B1S D4/D5 Discrepancy Report\n\nNo runtime discrepancy was raised. Review failed gates, if any, in `falsification_results.csv`.\n", encoding="utf-8")
    write_empty_outputs(outdir)
    (outdir / "git_state_after.txt").write_text(b1s.git_state(), encoding="utf-8")
    write_csv(outdir / "artifact_hashes.csv", b1s.artifact_hashes(outdir))
    marker(outdir, "RUN_COMPLETE.json", status, start_time, completed_stages, [] if d4_passed and d5_passed else ["D5_basin_orbital_stability"], {"labels": labels})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference-run", required=True)
    ap.add_argument("--matrix", choices=("closure",), default="closure")
    ap.add_argument("--d4-periods", type=float, default=50.0)
    ap.add_argument("--d5-periods", type=float, default=25.0)
    ap.add_argument("--sample-dt", type=float, default=1.0)
    ap.add_argument("--output-cadence-sample-dt", type=float, default=2.0)
    ap.add_argument("--stop-on-d4-fail", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=True)
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
    try:
        run_campaign(args, outdir)
    except Exception as exc:  # pragma: no cover - used for overnight failure capture.
        tb = traceback.format_exc()
        write_json(outdir / "RUN_FAILED.json", {"status": "RUN_FAILED", "run_end_time": now_iso(), "error": str(exc), "traceback": tb})
        (outdir / "DISCREPANCY_REPORT.md").write_text(f"# TG-B1S D4/D5 Discrepancy Report\n\nRunner failed with exception.\n\n```text\n{tb}\n```\n", encoding="utf-8")
        try:
            (outdir / "git_state_after.txt").write_text(b1s.git_state(), encoding="utf-8")
            write_empty_outputs(outdir)
            write_csv(outdir / "artifact_hashes.csv", b1s.artifact_hashes(outdir))
        finally:
            raise


if __name__ == "__main__":
    main()
