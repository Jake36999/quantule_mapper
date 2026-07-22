"""TG-B1S-LT long-time boundedness closure.

This script determines whether the robust-but-weak state-load backreaction from
TG-B1S-R remains a genuinely bounded shifted node over the preregistered 25/50
period horizon.

Frozen scope:
- STATE_LOAD_FEEDBACK only.
- R_relax, L_lock and P_threshold remain disabled.
- Q-ball, S_state, fixed S0 normalization, T/G equations, coupling signs,
  damping, coefficient maps, absorbing boundary and feedback operator are not
  changed.

Diagnostic additions:
- translated modal-reference correction;
- long paired off/full runs at 25 and 50 periods;
- phase-drift fits;
- field boundedness classification;
- basin perturbations;
- representative 25-period numerical validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
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

from jax_scout import gravity_TG_B1S_backreaction_robustness_gpu as br  # noqa: E402
from jax_scout import gravity_TG_B1S_state_load_feedback_gpu as b1s  # noqa: E402


S0_AUTHORITATIVE = b1s.S0_AUTHORITATIVE


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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float), encoding="utf-8")


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


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


def mode_for_kind(kind: str, phi: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    psi, _ = b1s.initial_state(kind, phi, cfg)
    return psi.astype(np.complex128)


def make_initial_state_lt(
    phi: np.ndarray,
    cfg: dict[str, Any],
    kind: str = "base",
    t_seed: float = 0.0,
    g_seed: float = 0.0,
) -> tuple[jnp.ndarray, ...]:
    if kind == "amplitude_perturb":
        psi = (1.001 * phi).astype(np.complex128)
        pi = (-1j * cfg["w"] * psi).astype(np.complex128)
    elif kind == "width_perturb":
        op = b1s.build_kg(cfg["N"], cfg["L"], cfg["c"], cfg["m"], cfg["dt"])
        g = b1s.make_grid(op)
        rr = np.asarray(g["R"])
        shape = rr * rr / (np.mean(rr * rr) + 1e-30) - 1.0
        psi = (phi * (1.0 + 1.0e-3 * shape)).astype(np.complex128)
        pi = (-1j * cfg["w"] * psi).astype(np.complex128)
    else:
        psi, pi = b1s.initial_state(kind, phi, cfg)
    real_zero = jnp.zeros_like(jnp.real(jnp.asarray(psi)))
    op = b1s.build_kg(cfg["N"], cfg["L"], cfg["c"], cfg["m"], cfg["dt"])
    g = b1s.make_grid(op)
    seed_shape = jnp.exp(-0.5 * (g["R"] / 1.5) ** 2)
    T0 = t_seed * seed_shape if t_seed else real_zero
    G0 = g_seed * seed_shape if g_seed else real_zero
    return (
        jnp.asarray(psi),
        jnp.asarray(pi),
        T0,
        real_zero,
        G0,
        real_zero,
    )


def run_case_lt(
    run_id: str,
    lambda_fb: float,
    cfg: dict[str, Any],
    phi: np.ndarray,
    refs: dict[str, float],
    periods: float,
    kind: str = "base",
    mode_kind: str | None = None,
    t_seed: float = 0.0,
    g_seed: float = 0.0,
    write_rows_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    duration = periods * period_from_cfg(cfg)
    op = b1s.build_kg(cfg["N"], cfg["L"], cfg["c"], cfg["m"], cfg["dt"])
    g = b1s.make_grid(op)
    cfgv = b1s.cfg_array(cfg)
    refv = b1s.refs_array(refs)
    mode_source = mode_for_kind(mode_kind or kind, phi, cfg)
    mode = jnp.asarray(mode_source.astype(np.complex128))
    state = make_initial_state_lt(phi, cfg, kind=kind, t_seed=t_seed, g_seed=g_seed)
    initial_phi = np.asarray(state[0])
    flags = jnp.asarray(br.arm_flags("full_loop"), dtype=jnp.float64)
    every = max(1, int(round(cfg["sample_dt"] / cfg["dt"])))
    steps = int(round(duration / cfg["dt"]))
    chunks = max(1, steps // every)
    rows: list[dict[str, float]] = []
    d0 = br.modal_diagnostics(state, cfgv, refv, g, mode)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), d0)
    rows.append({"t": 0.0, **{k: float(np.asarray(v)) for k, v in d0.items()}})
    start = time.time()
    for idx in range(chunks):
        state = br.evolve_n_lambda(state, cfgv, refv, g, flags, jnp.asarray(lambda_fb, dtype=jnp.float64), every)
        d = br.modal_diagnostics(state, cfgv, refv, g, mode)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), d)
        rows.append({"t": (idx + 1) * every * cfg["dt"], **{k: float(np.asarray(v)) for k, v in d.items()}})
    summary = br.summarize_rows(run_id, "full_loop", lambda_fb, cfg, rows, state, initial_phi, duration, periods)
    summary["execution_time_s"] = time.time() - start
    summary["kind"] = kind
    summary["mode_kind"] = mode_kind or kind
    summary["periods"] = periods
    summary["duration"] = duration
    summary["config_hash"] = b1s.config_hash({**cfg, "lambda_fb": lambda_fb, "periods": periods, "kind": kind, "mode_kind": mode_kind or kind})
    if write_rows_path is not None:
        write_csv(write_rows_path, rows)
    return summary, rows


def phase_drift(rows_full: list[dict[str, float]], rows_off: list[dict[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(rows_full), len(rows_off))
    t = np.asarray([r["t"] for r in rows_full[:n]], dtype=float)
    pf = np.unwrap(np.asarray([r["modal_phase"] for r in rows_full[:n]], dtype=float))
    po = np.unwrap(np.asarray([r["modal_phase"] for r in rows_off[:n]], dtype=float))
    return t, (pf - pf[0]) - (po - po[0])


def fit_phase_models(run_id: str, t: np.ndarray, drift: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if len(t) < 5:
        return rows
    x = t - t[0]
    y = drift
    coef = np.polyfit(x, y, 1)
    pred = np.polyval(coef, x)
    rows.append(
        {
            "run_id": run_id,
            "model": "constant_frequency_shift",
            "slope_delta_omega": coef[0],
            "offset": coef[1],
            "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
            "max_abs_residual": float(np.max(np.abs(y - pred))),
        }
    )
    # Saturating model by preregistered tau grid; theta_inf is linear for fixed tau.
    best_sat: dict[str, Any] | None = None
    tau_grid = np.geomspace(max(float(x[-1]) / 100.0, 1e-6), max(float(x[-1]) * 10.0, 1e-5), 48)
    for tau in tau_grid:
        basis = 1.0 - np.exp(-x / tau)
        theta_inf = float(np.dot(basis, y) / (np.dot(basis, basis) + 1e-30))
        pred = theta_inf * basis
        rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
        cand = {
            "run_id": run_id,
            "model": "saturating_phase_shift",
            "theta_inf": theta_inf,
            "tau": float(tau),
            "rmse": rmse,
            "max_abs_residual": float(np.max(np.abs(y - pred))),
        }
        if best_sat is None or rmse < best_sat["rmse"]:
            best_sat = cand
    if best_sat:
        rows.append(best_sat)
    # Oscillatory exchange: linear trend plus one sinusoid. Search frequency grid.
    best_osc: dict[str, Any] | None = None
    if x[-1] > 0:
        freq_grid = np.linspace(1.0 / x[-1], 12.0 / x[-1], 48)
        for freq in freq_grid:
            omega = 2.0 * math.pi * freq
            X = np.column_stack([x, np.sin(omega * x), np.cos(omega * x), np.ones_like(x)])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            pred = X @ beta
            rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
            cand = {
                "run_id": run_id,
                "model": "linear_plus_oscillatory_exchange",
                "slope_delta_omega": float(beta[0]),
                "sin_coeff": float(beta[1]),
                "cos_coeff": float(beta[2]),
                "offset": float(beta[3]),
                "omega_exchange": float(omega),
                "rmse": rmse,
                "max_abs_residual": float(np.max(np.abs(y - pred))),
            }
            if best_osc is None or rmse < best_osc["rmse"]:
                best_osc = cand
    if best_osc:
        rows.append(best_osc)
    lin_rmse = rows[0]["rmse"]
    for row in rows:
        row["rmse_improvement_vs_linear"] = (lin_rmse - row["rmse"]) / max(lin_rmse, 1e-30)
    return rows


def trend_slope(rows: list[dict[str, float]], key: str, frac: float = 0.5) -> float:
    t = np.asarray([r["t"] for r in rows], dtype=float)
    y = np.asarray([r[key] for r in rows], dtype=float)
    if len(t) < 5:
        return float("nan")
    start = int((1.0 - frac) * len(t))
    return float(np.polyfit(t[start:], y[start:], 1)[0])


def classify_field_boundedness(rows: list[dict[str, float]], key: str) -> tuple[str, float]:
    y = np.asarray([r[key] for r in rows], dtype=float)
    slope = trend_slope(rows, key)
    amp = float(np.max(np.abs(y))) + 1e-30
    rel = abs(slope) * (rows[-1]["t"] - rows[0]["t"]) / amp
    if not np.isfinite(slope):
        return "NUMERICALLY_UNRESOLVED", slope
    if amp < 1e-12:
        return "DECAY_TO_NULL", slope
    if rel < 0.02:
        return "APPROACH_FIXED_PROFILE", slope
    if rel < 0.20:
        return "REMAIN_BOUNDED_OSCILLATORY", slope
    if slope > 0:
        return "GROW_UNBOUNDED", slope
    return "DRIFT_MONOTONICALLY", slope


def classify_pair(summary: dict[str, Any], rows_full: list[dict[str, float]], delta: dict[str, float], gates: dict[str, float]) -> str:
    amp_slope = trend_slope(rows_full, "core_amp")
    width_slope = trend_slope(rows_full, "node_width")
    duration = max(rows_full[-1]["t"] - rows_full[0]["t"], 1e-30)
    amp_change = abs(amp_slope) * duration
    width_change = abs(width_slope) * duration
    if summary["profile_overlap"] <= gates["profile_overlap_min"] or summary["modal_leakage_max"] >= gates["modal_leakage_max"]:
        return "DESTABILIZED"
    if amp_change > gates["core_amp_trend_max"] or width_change > gates["width_trend_max"]:
        return "CONTINUOUS_DRIFT"
    if abs(delta.get("delta_modal_phase_slope", 0.0)) > 1e-12:
        return "STABLE_SHIFTED_NODE"
    return "DAMPED_TO_SHIFTED_NODE"


def pair_delta(full: dict[str, Any], off: dict[str, Any], rows_full: list[dict[str, float]], rows_off: list[dict[str, float]]) -> dict[str, float]:
    return br.delta_between(full, off, rows_full, rows_off)


def run_pair(
    label: str,
    periods: float,
    lambda_fb: float,
    cfg: dict[str, Any],
    phi: np.ndarray,
    refs: dict[str, float],
    outdir: Path,
    write_trajectory: bool = False,
    kind: str = "base",
    mode_kind: str | None = None,
    t_seed: float = 0.0,
    g_seed: float = 0.0,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, float]], list[dict[str, float]], dict[str, float]]:
    traj_off = outdir / f"{label}_off_trajectory.csv" if write_trajectory else None
    traj_full = outdir / f"{label}_full_trajectory.csv" if write_trajectory else None
    off, off_rows = run_case_lt(f"{label}_off", 0.0, cfg, phi, refs, periods, kind=kind, mode_kind=mode_kind, t_seed=t_seed, g_seed=g_seed, write_rows_path=traj_off)
    full, full_rows = run_case_lt(f"{label}_full", lambda_fb, cfg, phi, refs, periods, kind=kind, mode_kind=mode_kind, t_seed=t_seed, g_seed=g_seed, write_rows_path=traj_full)
    return off, full, off_rows, full_rows, pair_delta(full, off, full_rows, off_rows)


def write_docs(outdir: Path, status: str, labels: list[str], key: dict[str, Any]) -> None:
    docdir = ROOT / "docs" / "gravity_maturity"
    docdir.mkdir(parents=True, exist_ok=True)
    results = [
        "# TG-B1S-LT Long-Time Boundedness Results",
        "",
        "Timestamp: 2026-07-14.",
        f"Run directory: `{outdir.as_posix()}`.",
        f"Status: `{status}`.",
        "",
        "## Labels",
        "",
        *(f"- `{label}`" for label in labels),
        "",
        "## Key Metrics",
        "",
        f"- Max period horizon completed: `{key.get('max_periods_completed')}`.",
        f"- 50-period attractor class: `{key.get('period50_class')}`.",
        f"- 50-period modal leakage max: `{key.get('period50_modal_leakage_max')}`.",
        f"- 50-period profile overlap: `{key.get('period50_profile_overlap')}`.",
        f"- 50-period modal phase drift: `{key.get('period50_delta_modal_phase_final')}`.",
        "",
        "## Scope",
        "",
        "This is a state-load long-time boundedness closure. It does not enable `R_relax`, `L_lock`, or `P_threshold`, and it does not test photon emission or radiative relaxation.",
        "",
        "No gravity, objective-time, geodesic, universal-free-fall, photon, production or IRER-validation claim is made.",
    ]
    (docdir / "TG_B1S_LT_RESULTS.md").write_text("\n".join(results), encoding="utf-8")
    write_json(docdir / "TG_B1S_LT_SUMMARY.json", {"status": status, "labels": labels, "run_directory": str(outdir), **key})
    inputs = [
        "# TG-B1S-LT Documentation Inputs",
        "",
        f"- Status: `{status}`.",
        f"- Labels: `{', '.join(labels)}`.",
        f"- Artifacts: `{outdir.as_posix()}`.",
        "- Source: `STATE_LOAD_FEEDBACK` only.",
        "- Preserved labels: `TG_STATE_LOAD_FEEDFORWARD_CHAIN_SUPPORTED`, `TG_STATE_LOAD_GEOMETRIC_BACKREACTION_DETECTED`, `TG_STATE_LOAD_BACKREACTION_ROBUST`, `TG_STATE_LOAD_BACKREACTION_ROBUST_BUT_WEAK`.",
        "",
        "## Strongest Tables",
        "",
        "- `translated_modal_reference.csv`",
        "- `long_time_pairs.csv`",
        "- `phase_drift_fits.csv`",
        "- `field_boundedness.csv`",
        "- `basin_tests.csv`",
        "- `numerical_validation.csv`",
        "- `falsification_results.csv`",
        "",
        "## Caveats",
        "",
        "- This branch does not require radiation.",
        "- Any failing numerical-validation row blocks bounded-feedback promotion.",
        "- Do not update the master catalogue until review.",
    ]
    (docdir / "TG_B1S_LT_DOCUMENTATION_INPUTS.md").write_text("\n".join(inputs), encoding="utf-8")


def classify_final(long_rows: list[dict[str, Any]], basin_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]], field_rows: list[dict[str, Any]], gates: dict[str, float]) -> tuple[str, list[str], list[dict[str, Any]]]:
    p50 = next((r for r in long_rows if abs(float(r["periods"]) - 50.0) < 1e-9 and abs(float(r["lambda_fb"]) - 1.0) < 1e-9), None)
    p25 = next((r for r in long_rows if abs(float(r["periods"]) - 25.0) < 1e-9 and abs(float(r["lambda_fb"]) - 1.0) < 1e-9), None)
    node_localized = p50 is not None and float(p50["profile_overlap"]) > gates["profile_overlap_min"] and float(p50["modal_leakage_max"]) < gates["modal_leakage_max"]
    field_bounded = all(r["T_class"] in ("APPROACH_FIXED_PROFILE", "REMAIN_BOUNDED_OSCILLATORY") and r["G_class"] in ("APPROACH_FIXED_PROFILE", "REMAIN_BOUNDED_OSCILLATORY") for r in field_rows if abs(float(r.get("lambda_fb", 1.0)) - 1.0) < 1e-9)
    phase_consistent = p25 is not None and p50 is not None and np.sign(float(p25["delta_modal_phase_slope"])) == np.sign(float(p50["delta_modal_phase_slope"]))
    basin_ok = bool(basin_rows) and all(str(r["pass"]).lower() == "true" and r["class"] in ("STABLE_SHIFTED_NODE", "DAMPED_TO_SHIFTED_NODE", "BOUNDED_PERIODIC_FEEDBACK") for r in basin_rows)
    validation_ok = bool(validation_rows) and all(str(r["pass"]).lower() == "true" for r in validation_rows)
    ledger_ok = p50 is not None and abs(float(p50["ledger_residual_abs"])) < gates["ledger_abs_max"]
    no_drift = p50 is not None and p50["class"] in ("STABLE_SHIFTED_NODE", "DAMPED_TO_SHIFTED_NODE", "BOUNDED_PERIODIC_FEEDBACK")
    tests = [
        {"test": "50_period_node_localized", "pass": node_localized},
        {"test": "T_G_bounded", "pass": field_bounded},
        {"test": "phase_consistent_25_50", "pass": phase_consistent},
        {"test": "basin_qualitative_consistency", "pass": basin_ok},
        {"test": "numerical_validation", "pass": validation_ok},
        {"test": "energy_ledger", "pass": ledger_ok},
        {"test": "no_unresolved_monotonic_drift", "pass": no_drift},
    ]
    if all(t["pass"] for t in tests):
        return "TG_STATE_LOAD_BOUNDED_FEEDBACK_SUPPORTED", ["TG_STATE_LOAD_BOUNDED_FEEDBACK_SUPPORTED"], tests
    if node_localized and phase_consistent and p50 is not None and p50["class"] == "STABLE_SHIFTED_NODE":
        return "TG_STATE_LOAD_LONG_TIME_STABLE_SHIFT_SUPPORTED", ["TG_STATE_LOAD_LONG_TIME_STABLE_SHIFT_SUPPORTED"], tests
    if p50 is not None and p50["class"] == "CONTINUOUS_DRIFT":
        return "TG_STATE_LOAD_CONTINUOUS_SLOW_DRIFT", ["TG_STATE_LOAD_CONTINUOUS_SLOW_DRIFT"], tests
    if p50 is not None and p50["class"] == "DESTABILIZED":
        return "TG_STATE_LOAD_FEEDBACK_LONG_TIME_DESTABILIZATION", ["TG_STATE_LOAD_FEEDBACK_LONG_TIME_DESTABILIZATION"], tests
    return "TG_STATE_LOAD_LONG_TIME_NUMERICALLY_UNRESOLVED", ["TG_STATE_LOAD_LONG_TIME_NUMERICALLY_UNRESOLVED"], tests


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--matrix", choices=("closure", "primary", "quick"), default="closure")
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
    ap.add_argument("--sample-dt", type=float, default=0.5)
    ap.add_argument("--basin-periods", type=float, default=25.0)
    ap.add_argument("--validation-periods", type=float, default=25.0)
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

    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.out) if args.out else ROOT / "sweep_runs" / f"TG_B1S_LONG_TIME_GPU_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "git_state_before.txt").write_text(b1s.git_state(), encoding="utf-8")
    pf = b1s.preflight(outdir)
    cfg = default_config(args)
    cfg["config_hash"] = b1s.config_hash(cfg)
    write_json(outdir / "environment_versions.json", {**pf, "stage": "TG-B1S-LT_LONG_TIME_BOUNDEDNESS"})
    gates = {
        "profile_overlap_min": 0.999,
        "modal_leakage_max": 1e-3,
        "core_amp_trend_max": 2e-4,
        "width_trend_max": 5e-4,
        "ledger_abs_max": 1e-4,
        "phase_slope_relative_tolerance": 0.35,
    }
    write_json(
        outdir / "preregistered_matrix.json",
        {
            "matrix": args.matrix,
            "primary_lambda": 1.0,
            "long_pairs": [
                {"periods": 25.0, "lambda_fb": 1.0},
                {"periods": 50.0, "lambda_fb": 1.0},
                {"periods": 25.0, "lambda_fb": 0.5},
                {"periods": 25.0, "lambda_fb": 2.0},
            ],
            "basin_periods": args.basin_periods,
            "validation_periods": args.validation_periods,
            "gates": gates,
            "state_load_only": True,
            "disabled_sources": ["R_relax", "L_lock", "P_threshold"],
            "S0_authoritative_TG_S": S0_AUTHORITATIVE,
        },
    )
    write_json(
        outdir / "source_spec.json",
        {
            "source_hypothesis": "STATE_LOAD_FEEDBACK",
            "enabled_source": "S_state",
            "disabled_sources": ["R_relax", "L_lock", "P_threshold"],
            "S0_authoritative_TG_S": S0_AUTHORITATIVE,
            "casewise_normalization": False,
        },
    )

    phi, prof = b1s.solve_qball(cfg)
    op = b1s.build_kg(cfg["N"], cfg["L"], cfg["c"], cfg["m"], cfg["dt"])
    refs = b1s.ref_source_norms(phi, cfg, op)
    base = b1s.baseline_contract(cfg, phi, prof)
    write_csv(outdir / "baseline_node_contract.csv", [base])
    if base["status"] != "TG_SOURCE_NODE_BASELINE_CLOSED":
        status = "TG_SOURCE_NODE_NOT_STATIONARY"
        write_docs(outdir, status, [status], {"max_periods_completed": 0})
        write_csv(outdir / "artifact_hashes.csv", b1s.artifact_hashes(outdir))
        print(json.dumps({"status": status, "outdir": str(outdir)}, indent=2))
        return

    # LT0 translated modal-reference correction.
    trans_old_off, trans_old_full, trans_old_off_rows, trans_old_full_rows, trans_old_delta = run_pair("LT0_translated_oldref_2P", 2.0, 1.0, cfg, phi, refs, outdir, kind="translated", mode_kind="base")
    trans_new_off, trans_new_full, trans_new_off_rows, trans_new_full_rows, trans_new_delta = run_pair("LT0_translated_newref_2P", 2.0, 1.0, cfg, phi, refs, outdir, kind="translated", mode_kind="translated")
    translated_rows = [
        {
            "reference": "old_fixed_stationary_mode",
            "full_modal_leakage_max": trans_old_full["modal_leakage_max"],
            "full_modal_leakage_final": trans_old_full["modal_leakage_final"],
            **trans_old_delta,
        },
        {
            "reference": "translated_initial_grid_mode",
            "full_modal_leakage_max": trans_new_full["modal_leakage_max"],
            "full_modal_leakage_final": trans_new_full["modal_leakage_final"],
            **trans_new_delta,
        },
    ]
    write_csv(outdir / "translated_modal_reference.csv", translated_rows)

    long_specs = [(25.0, 1.0), (50.0, 1.0)]
    if args.matrix == "closure":
        long_specs.extend([(25.0, 0.5), (25.0, 2.0)])
    if args.matrix == "quick":
        long_specs = [(2.0, 1.0)]

    long_rows: list[dict[str, Any]] = []
    phase_fit_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    shell_rows: list[dict[str, Any]] = []
    for periods, lam in long_specs:
        label = f"LT1_{periods:g}P_lam{lam:g}"
        off, full, off_rows, full_rows, delta = run_pair(label, periods, lam, cfg, phi, refs, outdir, write_trajectory=(lam == 1.0 and periods in (25.0, 50.0)))
        t, drift = phase_drift(full_rows, off_rows)
        phase_fit_rows.extend(fit_phase_models(label, t, drift))
        T_class, T_slope = classify_field_boundedness(full_rows, "T_peak")
        G_class, G_slope = classify_field_boundedness(full_rows, "G_peak")
        cls = classify_pair(full, full_rows, delta, gates)
        row = {
            "run_id": label,
            "periods": periods,
            "lambda_fb": lam,
            "class": cls,
            **delta,
            "profile_overlap": full["profile_overlap"],
            "modal_leakage_max": full["modal_leakage_max"],
            "modal_leakage_final": full["modal_leakage_final"],
            "ledger_residual_abs": full["ledger_residual_abs"],
            "core_amp_final": full["core_amp_final"],
            "node_width_final": full["node_width_final"],
            "shell_flux_max": full["shell_flux_max"],
            "T_peak": full["T_peak"],
            "G_peak": full["G_peak"],
            "T_class": T_class,
            "G_class": G_class,
            "T_peak_slope": T_slope,
            "G_peak_slope": G_slope,
        }
        long_rows.append(row)
        field_rows.append({"run_id": label, "periods": periods, "lambda_fb": lam, "T_class": T_class, "G_class": G_class, "T_peak_slope": T_slope, "G_peak_slope": G_slope, "T_peak": full["T_peak"], "G_peak": full["G_peak"]})
        ledger_rows.append({"run_id": label, "periods": periods, "lambda_fb": lam, "E_phi_final": full["E_phi_final"], "E_T_final": full["E_T_final"], "E_G_final": full["E_G_final"], "dissipated_E": full["dissipated_E"], "boundary_E": full["boundary_E"], "ledger_residual_abs": full["ledger_residual_abs"]})
        shell_rows.append({"run_id": label, "periods": periods, "lambda_fb": lam, "shell_flux_max": full["shell_flux_max"], "delta_shell_flux_max": delta["delta_shell_flux_max"]})
        # Stream artifacts after every long pair.
        write_csv(outdir / "long_time_pairs.csv", long_rows)
        write_csv(outdir / "phase_drift_fits.csv", phase_fit_rows)
        write_csv(outdir / "field_boundedness.csv", field_rows)
        write_csv(outdir / "energy_ledger.csv", ledger_rows)
        write_csv(outdir / "shell_flux.csv", shell_rows)
        write_csv(outdir / "attractor_classification.csv", long_rows)

    basin_specs = [
        ("amplitude_perturb", "amplitude_perturb", None, 0.0, 0.0),
        ("width_perturb", "width_perturb", None, 0.0, 0.0),
        ("global_phase", "global_phase", None, 0.0, 0.0),
        ("translated", "translated", "translated", 0.0, 0.0),
        ("initial_T_seed", "base", None, 1.0e-5, 0.0),
        ("initial_G_seed", "base", None, 0.0, 1.0e-5),
    ]
    if args.matrix in ("quick", "primary"):
        basin_specs = basin_specs[:2]
        args.basin_periods = 2.0
    if args.matrix == "primary":
        basin_specs = []
    basin_rows: list[dict[str, Any]] = []
    for name, kind, mode_kind, t_seed, g_seed in basin_specs:
        label = f"LT4_basin_{name}_{args.basin_periods:g}P"
        off, full, off_rows, full_rows, delta = run_pair(label, args.basin_periods, 1.0, cfg, phi, refs, outdir, kind=kind, mode_kind=mode_kind, t_seed=t_seed, g_seed=g_seed)
        cls = classify_pair(full, full_rows, delta, gates)
        passed = cls in ("STABLE_SHIFTED_NODE", "DAMPED_TO_SHIFTED_NODE", "BOUNDED_PERIODIC_FEEDBACK") and full["profile_overlap"] > gates["profile_overlap_min"] and full["modal_leakage_max"] < gates["modal_leakage_max"]
        basin_rows.append({"case": name, "periods": args.basin_periods, "class": cls, "pass": passed, "profile_overlap": full["profile_overlap"], "modal_leakage_max": full["modal_leakage_max"], **delta})
        write_csv(outdir / "basin_tests.csv", basin_rows)

    validation_specs = [
        ("dt_half", {"dt": cfg["dt"] / 2.0}),
        ("grid_refined", {"N": 56}),
        ("larger_box", {"N": 56, "L": 12.0}),
    ]
    if args.matrix in ("quick", "primary"):
        validation_specs = []
    validation_rows: list[dict[str, Any]] = []
    base25 = next((r for r in long_rows if abs(float(r["periods"]) - 25.0) < 1e-9 and abs(float(r["lambda_fb"]) - 1.0) < 1e-9), None)
    for name, overrides in validation_specs:
        vcfg = dict(cfg)
        vcfg.update(overrides)
        vphi = phi
        if vcfg["N"] != cfg["N"] or vcfg["L"] != cfg["L"]:
            vphi, _ = b1s.solve_qball(vcfg)
        label = f"LT5_validation_{name}_{args.validation_periods:g}P"
        off, full, off_rows, full_rows, delta = run_pair(label, args.validation_periods, 1.0, vcfg, vphi, refs, outdir)
        cls = classify_pair(full, full_rows, delta, gates)
        slope_ref = float(base25["delta_modal_phase_slope"]) if base25 is not None else float("nan")
        slope = float(delta["delta_modal_phase_slope"])
        rel = abs(slope - slope_ref) / max(abs(slope_ref), 1e-30) if np.isfinite(slope_ref) else float("nan")
        passed = cls in ("STABLE_SHIFTED_NODE", "DAMPED_TO_SHIFTED_NODE", "BOUNDED_PERIODIC_FEEDBACK") and full["profile_overlap"] > gates["profile_overlap_min"] and full["modal_leakage_max"] < gates["modal_leakage_max"] and (not np.isfinite(rel) or rel < gates["phase_slope_relative_tolerance"])
        validation_rows.append({"case": name, "periods": args.validation_periods, "class": cls, "pass": passed, "phase_slope_relative_difference": rel, "profile_overlap": full["profile_overlap"], "modal_leakage_max": full["modal_leakage_max"], **delta})
        write_csv(outdir / "numerical_validation.csv", validation_rows)

    write_csv(outdir / "long_time_pairs.csv", long_rows)
    write_csv(outdir / "phase_drift_fits.csv", phase_fit_rows)
    write_csv(outdir / "field_boundedness.csv", field_rows)
    write_csv(outdir / "energy_ledger.csv", ledger_rows)
    write_csv(outdir / "shell_flux.csv", shell_rows)
    write_csv(outdir / "basin_tests.csv", basin_rows)
    write_csv(outdir / "numerical_validation.csv", validation_rows)
    write_csv(outdir / "attractor_classification.csv", long_rows)

    status, labels, falsification_rows = classify_final(long_rows, basin_rows, validation_rows, field_rows, gates)
    write_csv(outdir / "falsification_results.csv", falsification_rows)
    p50 = next((r for r in long_rows if abs(float(r["periods"]) - 50.0) < 1e-9 and abs(float(r["lambda_fb"]) - 1.0) < 1e-9), {})
    key = {
        "max_periods_completed": max(float(r["periods"]) for r in long_rows) if long_rows else 0.0,
        "period50_class": p50.get("class"),
        "period50_modal_leakage_max": p50.get("modal_leakage_max"),
        "period50_profile_overlap": p50.get("profile_overlap"),
        "period50_delta_modal_phase_final": p50.get("delta_modal_phase_final"),
        "period50_delta_modal_phase_slope": p50.get("delta_modal_phase_slope"),
        "bounded_feedback_promoted": status == "TG_STATE_LOAD_BOUNDED_FEEDBACK_SUPPORTED",
        "gates": gates,
    }
    write_docs(outdir, status, labels, key)
    (outdir / "TECHNICAL_HANDOFF.md").write_text(
        "\n".join(
            [
                "# TG-B1S-LT Technical Handoff",
                "",
                f"Status: `{status}`.",
                f"Run directory: `{outdir.as_posix()}`.",
                "",
                "## Labels",
                "",
                *(f"- `{label}`" for label in labels),
                "",
                "## Rule",
                "",
                "State-load only; no R_relax, L_lock, P_threshold, photon, radiative, gravity, objective-time, geodesic or production claim.",
            ]
        ),
        encoding="utf-8",
    )
    (outdir / "OPEN_QUESTIONS.md").write_text(
        "# TG-B1S-LT Open Questions\n\n- If bounded feedback is supported, should TG-B1R phase-tension relaxation begin as a separate branch?\n- If validation fails, which numerical discrepancy dominates the 25-period gate?\n",
        encoding="utf-8",
    )
    write_csv(outdir / "artifact_hashes.csv", b1s.artifact_hashes(outdir))
    print(json.dumps({"status": status, "labels": labels, "outdir": str(outdir)}, indent=2))


if __name__ == "__main__":
    main()
