"""TG-B1S-D drift decomposition audit.

This standalone GPU audit keeps the TG-B1S state-load model frozen and asks
whether the 50-period CONTINUOUS_DRIFT label is really a stable frequency
shift, a transient to a shifted state, structural drift, or numerical
accumulation.

Frozen scope:
- STATE_LOAD_FEEDBACK only.
- R_relax, L_lock and P_threshold remain disabled.
- Q-ball, S_state, S0, T/G equations, damping/coupling signs, coefficient
  maps, absorber and G->phi feedback operator are not changed.
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
from jax_scout import gravity_TG_B1S_long_time_gpu as lt  # noqa: E402
from jax_scout import gravity_TG_B1S_state_load_feedback_gpu as b1s  # noqa: E402


from jax_scout.provenance import write_json  # noqa: E402  (shared: stamps run metadata)

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


def make_coordinates(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(cfg["N"])
    L = float(cfg["L"])
    x = np.linspace(-0.5 * L, 0.5 * L, n, endpoint=False)
    return np.meshgrid(x, x, x, indexing="ij")


def norm2(phi: np.ndarray, dV: float) -> float:
    return float(np.sum(np.abs(phi) ** 2) * dV)


def center_of_mass(phi: np.ndarray, coords: tuple[np.ndarray, np.ndarray, np.ndarray], dV: float) -> np.ndarray:
    rho = np.abs(phi) ** 2
    mass = float(np.sum(rho) * dV) + 1e-30
    return np.asarray([float(np.sum(c * rho) * dV / mass) for c in coords])


def orbital_metrics(
    phi_full: np.ndarray,
    phi_off: np.ndarray,
    mode: np.ndarray,
    cfg: dict[str, Any],
    coords: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, float]:
    dx = float(cfg["L"]) / int(cfg["N"])
    dV = dx**3
    n_off = math.sqrt(norm2(phi_off, dV)) + 1e-30
    n_full = math.sqrt(norm2(phi_full, dV)) + 1e-30
    inner = np.sum(np.conj(phi_off) * phi_full) * dV
    alpha = float(np.angle(inner))
    unaligned = math.sqrt(norm2(phi_full - phi_off, dV)) / n_off
    phase_aligned = math.sqrt(norm2(phi_full - np.exp(1j * alpha) * phi_off, dV)) / n_off
    overlap = float(abs(inner) / (n_off * n_full + 1e-30))

    com_full = center_of_mass(phi_full, coords, dV)
    com_off = center_of_mass(phi_off, coords, dV)
    shift_cells = np.rint((com_full - com_off) / dx).astype(int)
    # Keep this a diagnostic, not a search over arbitrary translations.
    shift_cells = np.clip(shift_cells, -2, 2)
    shifted = np.roll(phi_off, tuple(int(x) for x in shift_cells), axis=(0, 1, 2))
    inner_t = np.sum(np.conj(shifted) * phi_full) * dV
    alpha_t = float(np.angle(inner_t))
    phase_translation = math.sqrt(norm2(phi_full - np.exp(1j * alpha_t) * shifted, dV)) / n_off
    shifted_norm = math.sqrt(norm2(shifted, dV)) + 1e-30
    amp_scale = n_full / shifted_norm
    amp_phase_translation = math.sqrt(norm2(phi_full - amp_scale * np.exp(1j * alpha_t) * shifted, dV)) / n_full

    def ref_metrics(phi: np.ndarray, prefix: str) -> dict[str, float]:
        n_phi = math.sqrt(norm2(phi, dV)) + 1e-30
        n_mode = math.sqrt(norm2(mode, dV)) + 1e-30
        inn = np.sum(np.conj(mode) * phi) * dV
        a = float(np.angle(inn))
        phase_dist = math.sqrt(norm2(phi - np.exp(1j * a) * mode, dV)) / n_mode
        amp = n_phi / n_mode
        amp_dist = math.sqrt(norm2(phi - amp * np.exp(1j * a) * mode, dV)) / n_phi
        return {
            f"{prefix}_reference_phase_distance": phase_dist,
            f"{prefix}_reference_amp_phase_distance": amp_dist,
            f"{prefix}_reference_overlap": float(abs(inn) / (n_phi * n_mode + 1e-30)),
        }

    out = {
        "alpha_star": alpha,
        "alpha_star_translation": alpha_t,
        "unaligned_distance": unaligned,
        "phase_aligned_distance": phase_aligned,
        "phase_translation_aligned_distance": phase_translation,
        "amp_phase_translation_aligned_distance": amp_phase_translation,
        "full_off_modal_overlap": overlap,
        "translation_shift_x": int(shift_cells[0]),
        "translation_shift_y": int(shift_cells[1]),
        "translation_shift_z": int(shift_cells[2]),
        "com_delta_x": float(com_full[0] - com_off[0]),
        "com_delta_y": float(com_full[1] - com_off[1]),
        "com_delta_z": float(com_full[2] - com_off[2]),
    }
    out.update(ref_metrics(phi_full, "full"))
    out.update(ref_metrics(phi_off, "off"))
    return out


@jax.jit
def orbital_metrics_jax(
    phi_full: jnp.ndarray,
    phi_off: jnp.ndarray,
    mode: jnp.ndarray,
    g: dict[str, jnp.ndarray],
) -> dict[str, jnp.ndarray]:
    """GPU-resident full-field orbital comparison.

    The live D4/D5 runs must not pull full 3D fields back to NumPy at every
    checkpoint. This mirrors the scalar diagnostics from ``orbital_metrics``
    while keeping all reductions on the selected JAX device.
    """
    dV = g["dx"] ** 3
    eps = jnp.asarray(1e-30, dtype=jnp.float64)

    def norm(phi: jnp.ndarray) -> jnp.ndarray:
        return jnp.sqrt(jnp.sum(jnp.abs(phi) ** 2) * dV) + eps

    def dist(phi_a: jnp.ndarray, phi_b: jnp.ndarray) -> jnp.ndarray:
        return jnp.sqrt(jnp.sum(jnp.abs(phi_a - phi_b) ** 2) * dV)

    def com(phi: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        rho = jnp.abs(phi) ** 2
        mass = jnp.sum(rho) * dV + eps
        return (
            jnp.sum(g["X"] * rho) * dV / mass,
            jnp.sum(g["Y"] * rho) * dV / mass,
            jnp.sum(g["Z"] * rho) * dV / mass,
        )

    n_off = norm(phi_off)
    n_full = norm(phi_full)
    inner = jnp.sum(jnp.conj(phi_off) * phi_full) * dV
    alpha = jnp.angle(inner)
    phase = jnp.exp(1j * alpha)
    unaligned = dist(phi_full, phi_off) / n_off
    phase_aligned = dist(phi_full, phase * phi_off) / n_off
    overlap = jnp.abs(inner) / (n_off * n_full + eps)

    # Full/off comparisons are same-origin in the closure rows, including the
    # translated-node basin row. Keep the translation diagnostic conservative:
    # report relative COM displacement, but do not perform a CPU-side roll/search
    # in the live loop.
    com_full = com(phi_full)
    com_off = com(phi_off)
    phase_translation = phase_aligned
    amp_scale = n_full / n_off
    amp_phase_translation = dist(phi_full, amp_scale * phase * phi_off) / n_full

    n_mode = norm(mode)

    def ref_metrics(phi: jnp.ndarray, prefix: str) -> dict[str, jnp.ndarray]:
        n_phi = norm(phi)
        inn = jnp.sum(jnp.conj(mode) * phi) * dV
        a = jnp.angle(inn)
        phase_dist = dist(phi, jnp.exp(1j * a) * mode) / n_mode
        amp = n_phi / n_mode
        amp_dist = dist(phi, amp * jnp.exp(1j * a) * mode) / n_phi
        return {
            f"{prefix}_reference_phase_distance": phase_dist,
            f"{prefix}_reference_amp_phase_distance": amp_dist,
            f"{prefix}_reference_overlap": jnp.abs(inn) / (n_phi * n_mode + eps),
        }

    zero = jnp.asarray(0.0, dtype=jnp.float64)
    out = {
        "alpha_star": alpha,
        "alpha_star_translation": alpha,
        "unaligned_distance": unaligned,
        "phase_aligned_distance": phase_aligned,
        "phase_translation_aligned_distance": phase_translation,
        "amp_phase_translation_aligned_distance": amp_phase_translation,
        "full_off_modal_overlap": overlap,
        "translation_shift_x": zero,
        "translation_shift_y": zero,
        "translation_shift_z": zero,
        "com_delta_x": com_full[0] - com_off[0],
        "com_delta_y": com_full[1] - com_off[1],
        "com_delta_z": com_full[2] - com_off[2],
    }
    out.update(ref_metrics(phi_full, "full"))
    out.update(ref_metrics(phi_off, "off"))
    return out


def as_float_dict(payload: dict[str, Any]) -> dict[str, float]:
    return {k: float(np.asarray(v)) for k, v in payload.items()}


def run_pair_decomposition(
    run_id: str,
    periods: float,
    lambda_fb: float,
    cfg: dict[str, Any],
    phi: np.ndarray,
    refs: dict[str, float],
    sample_dt: float,
    kind: str = "base",
    mode_kind: str | None = None,
    t_seed: float = 0.0,
    g_seed: float = 0.0,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, float]], list[dict[str, float]]]:
    duration = periods * period_from_cfg(cfg)
    op = b1s.build_kg(cfg["N"], cfg["L"], cfg["c"], cfg["m"], cfg["dt"])
    g = b1s.make_grid(op)
    cfgv = b1s.cfg_array(cfg)
    refv = b1s.refs_array(refs)
    mode_source = lt.mode_for_kind(mode_kind or kind, phi, cfg)
    mode = jnp.asarray(mode_source.astype(np.complex128))
    flags = jnp.asarray(br.arm_flags("full_loop"), dtype=jnp.float64)
    off_state = lt.make_initial_state_lt(phi, cfg, kind=kind, t_seed=t_seed, g_seed=g_seed)
    full_state = lt.make_initial_state_lt(phi, cfg, kind=kind, t_seed=t_seed, g_seed=g_seed)
    initial_phi = np.asarray(full_state[0])
    every = max(1, int(round(sample_dt / cfg["dt"])))
    steps = int(round(duration / cfg["dt"]))
    chunks = max(1, steps // every)

    pair_rows: list[dict[str, Any]] = []
    off_rows: list[dict[str, float]] = []
    full_rows: list[dict[str, float]] = []

    start = time.time()
    for idx in range(chunks + 1):
        t = idx * every * cfg["dt"]
        off_d = lt.br.modal_diagnostics(off_state, cfgv, refv, g, mode)
        full_d = lt.br.modal_diagnostics(full_state, cfgv, refv, g, mode)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), off_d)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), full_d)
        off_dict = {"t": t, **as_float_dict(off_d)}
        full_dict = {"t": t, **as_float_dict(full_d)}
        off_rows.append(off_dict)
        full_rows.append(full_dict)
        metrics_jax = orbital_metrics_jax(full_state[0], off_state[0], mode, g)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics_jax)
        metrics = as_float_dict(metrics_jax)
        row: dict[str, Any] = {
            "run_id": run_id,
            "t": t,
            "period": t / period_from_cfg(cfg),
            "lambda_fb": lambda_fb,
            **metrics,
            "delta_core_amp": full_dict["core_amp"] - off_dict["core_amp"],
            "delta_width": full_dict["node_width"] - off_dict["node_width"],
            "delta_core_energy": full_dict["core_energy"] - off_dict["core_energy"],
            "delta_charge": full_dict["charge"] - off_dict["charge"],
            "delta_modal_leakage": full_dict["modal_leakage"] - off_dict["modal_leakage"],
            "full_modal_leakage": full_dict["modal_leakage"],
            "off_modal_leakage": off_dict["modal_leakage"],
            "T_peak": full_dict["T_peak"],
            "G_peak": full_dict["G_peak"],
            "shell_flux_proxy": full_dict["shell_flux_proxy"],
        }
        pair_rows.append(row)
        if idx < chunks:
            off_state = br.evolve_n_lambda(off_state, cfgv, refv, g, flags, jnp.asarray(0.0, dtype=jnp.float64), every)
            full_state = br.evolve_n_lambda(full_state, cfgv, refv, g, flags, jnp.asarray(lambda_fb, dtype=jnp.float64), every)

    # Add unwrapped phase and instantaneous frequency after collection.
    pf = np.unwrap(np.asarray([r["modal_phase"] for r in full_rows], dtype=float))
    po = np.unwrap(np.asarray([r["modal_phase"] for r in off_rows], dtype=float))
    delta_theta = (pf - pf[0]) - (po - po[0])
    tvals = np.asarray([r["t"] for r in pair_rows], dtype=float)
    if len(tvals) > 3:
        delta_omega = np.gradient(delta_theta, tvals)
    else:
        delta_omega = np.full_like(delta_theta, np.nan)
    for i, row in enumerate(pair_rows):
        row["delta_theta"] = float(delta_theta[i])
        row["delta_omega_inst"] = float(delta_omega[i])

    full_summary = br.summarize_rows(f"{run_id}_full", "full_loop", lambda_fb, cfg, full_rows, full_state, initial_phi, duration, periods)
    off_summary = br.summarize_rows(f"{run_id}_off", "feedback_off", 0.0, cfg, off_rows, off_state, initial_phi, duration, periods)
    delta = br.delta_between(full_summary, off_summary, full_rows, off_rows)
    summary = {
        "run_id": run_id,
        "periods": periods,
        "duration": duration,
        "lambda_fb": lambda_fb,
        "kind": kind,
        "mode_kind": mode_kind or kind,
        "execution_time_s": time.time() - start,
        **delta,
        "phase_aligned_distance_final": pair_rows[-1]["phase_aligned_distance"],
        "phase_aligned_distance_max": float(max(r["phase_aligned_distance"] for r in pair_rows)),
        "phase_translation_aligned_distance_final": pair_rows[-1]["phase_translation_aligned_distance"],
        "phase_translation_aligned_distance_max": float(max(r["phase_translation_aligned_distance"] for r in pair_rows)),
        "orbital_distance_final": pair_rows[-1]["phase_translation_aligned_distance"],
        "full_modal_leakage_max": full_summary["modal_leakage_max"],
        "full_profile_overlap": full_summary["profile_overlap"],
        "ledger_residual_abs": full_summary["ledger_residual_abs"],
        "T_peak": full_summary["T_peak"],
        "G_peak": full_summary["G_peak"],
        "config_hash": b1s.config_hash({**cfg, "run_id": run_id, "periods": periods, "lambda_fb": lambda_fb, "kind": kind, "mode_kind": mode_kind or kind}),
    }
    return summary, pair_rows, full_rows, off_rows


def fit_series(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    span = float(x[-1] - x[0]) if len(x) else 0.0
    amp = float(np.max(y) - np.min(y)) if len(y) else 0.0
    floor = max(1e-10, 1e-7 * max(float(np.max(np.abs(y))) if len(y) else 1.0, 1.0))
    if len(x) < 6 or amp < floor:
        return {"classification": "BELOW_NUMERICAL_FLOOR", "slope": 0.0, "rmse_linear": 0.0, "rmse_constant": 0.0}

    const = np.full_like(y, float(np.mean(y)))
    rmse_const = float(np.sqrt(np.mean((y - const) ** 2)))
    coef1 = np.polyfit(x, y, 1)
    pred1 = np.polyval(coef1, x)
    rmse1 = float(np.sqrt(np.mean((y - pred1) ** 2)))
    coef2 = np.polyfit(x, y, 2)
    pred2 = np.polyval(coef2, x)
    rmse2 = float(np.sqrt(np.mean((y - pred2) ** 2)))

    # Saturating fit by a fixed tau grid.
    best_sat = {"rmse": float("inf"), "tau": float("nan"), "limit": float("nan")}
    if span > 0:
        z = x - x[0]
        for tau in np.geomspace(max(span / 200.0, 1e-8), max(span * 10.0, 1e-7), 40):
            basis = 1.0 - np.exp(-z / tau)
            A = np.column_stack([basis, np.ones_like(basis)])
            beta, *_ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ beta
            rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
            if rmse < best_sat["rmse"]:
                best_sat = {"rmse": rmse, "tau": float(tau), "limit": float(beta[0] + beta[1])}

    # One-frequency bounded oscillator with offset.
    best_osc = {"rmse": float("inf"), "omega": float("nan")}
    if span > 0:
        for freq in np.linspace(1.0 / span, 8.0 / span, 32):
            omega = 2.0 * math.pi * freq
            A = np.column_stack([np.sin(omega * x), np.cos(omega * x), np.ones_like(x)])
            beta, *_ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ beta
            rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
            if rmse < best_osc["rmse"]:
                best_osc = {"rmse": rmse, "omega": float(omega)}

    slope_effect = abs(float(coef1[0])) * max(span, 1e-30)
    if rmse2 < 0.65 * rmse1 and abs(float(coef2[0])) * span * span > 0.25 * amp:
        label = "QUADRATIC_OR_ACCELERATING"
    elif best_sat["rmse"] < 0.70 * rmse1:
        label = "SATURATING"
    elif best_osc["rmse"] < 0.55 * rmse1 and slope_effect < 0.5 * amp:
        label = "BOUNDED_OSCILLATORY"
    elif rmse1 < 0.75 * rmse_const and slope_effect > 0.35 * amp:
        label = "LINEAR_SECULAR"
    elif rmse_const <= 1.10 * rmse1:
        label = "CONSTANT_OFFSET"
    else:
        label = "LINEAR_SECULAR" if slope_effect > 0.25 * amp else "CONSTANT_OFFSET"

    return {
        "classification": label,
        "slope": float(coef1[0]),
        "intercept": float(coef1[1]),
        "quadratic": float(coef2[0]),
        "rmse_constant": rmse_const,
        "rmse_linear": rmse1,
        "rmse_quadratic": rmse2,
        "rmse_saturating": best_sat["rmse"],
        "tau_saturating": best_sat["tau"],
        "limit_saturating": best_sat["limit"],
        "rmse_oscillatory": best_osc["rmse"],
        "omega_oscillatory": best_osc["omega"],
        "range": amp,
    }


def drift_channel_rows(run_id: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    t = np.asarray([r["t"] for r in rows], dtype=float)
    channels = {
        "delta_theta": np.asarray([r["delta_theta"] for r in rows], dtype=float),
        "delta_omega": np.asarray([r["delta_omega_inst"] for r in rows], dtype=float),
        "delta_A_core": np.asarray([r["delta_core_amp"] for r in rows], dtype=float),
        "delta_width": np.asarray([r["delta_width"] for r in rows], dtype=float),
        "delta_E_core": np.asarray([r["delta_core_energy"] for r in rows], dtype=float),
        "delta_Q": np.asarray([r["delta_charge"] for r in rows], dtype=float),
        "d_orbital": np.asarray([r["phase_translation_aligned_distance"] for r in rows], dtype=float),
        "modal_leakage": np.asarray([r["full_modal_leakage"] for r in rows], dtype=float),
    }
    out = []
    for name, series in channels.items():
        fit = fit_series(t, series)
        out.append({"run_id": run_id, "channel": name, "final_value": float(series[-1]), **fit})
    return out


def asymptotic_frequency_row(run_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    t = np.asarray([r["t"] for r in rows], dtype=float)
    theta = np.asarray([r["delta_theta"] for r in rows], dtype=float)
    x = t - t[0]
    if len(t) < 8:
        return {"run_id": run_id, "status": "INSUFFICIENT_SAMPLES"}
    coef = np.polyfit(x, theta, 1)
    pred = np.polyval(coef, x)
    half = len(x) // 2
    early = np.polyfit(x[:half], theta[:half], 1)[0]
    late = np.polyfit(x[half:], theta[half:], 1)[0]
    inst = np.asarray([r["delta_omega_inst"] for r in rows], dtype=float)
    predicted_final = float(coef[0] * x[-1] + coef[1])
    measured_final = float(theta[-1])
    return {
        "run_id": run_id,
        "delta_omega_infty": float(coef[0]),
        "theta0": float(coef[1]),
        "fit_rmse": float(np.sqrt(np.mean((theta - pred) ** 2))),
        "fit_max_abs_residual": float(np.max(np.abs(theta - pred))),
        "early_slope": float(early),
        "late_slope": float(late),
        "early_late_relative_difference": float(abs(late - early) / max(abs(late), abs(early), 1e-30)),
        "instantaneous_frequency_variance": float(np.var(inst[np.isfinite(inst)])),
        "predicted_final_phase": predicted_final,
        "measured_final_phase": measured_final,
        "prediction_error": float(measured_final - predicted_final),
        "constant_slope_supported": bool(abs(late - early) / max(abs(late), abs(early), 1e-30) < 0.35),
    }


def classify_interpretation(
    run_rows: dict[str, list[dict[str, Any]]],
    drift_rows: list[dict[str, Any]],
    asym_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    basin_rows: list[dict[str, Any]],
) -> tuple[str, list[str], list[dict[str, Any]]]:
    target_id = "D3_100P_lam1" if "D3_100P_lam1" in run_rows else "D2_50P_lam1"
    drows = [r for r in drift_rows if r["run_id"] == target_id]
    by_channel = {r["channel"]: r for r in drows}
    phase_ok = by_channel.get("delta_theta", {}).get("classification") == "LINEAR_SECULAR"
    freq_row = next((r for r in asym_rows if r.get("run_id") == target_id), {})
    constant_slope = bool(freq_row.get("constant_slope_supported", False))
    structural_channels = [
        "delta_A_core",
        "delta_width",
        "delta_E_core",
        "delta_Q",
        "d_orbital",
        "modal_leakage",
    ]
    structural = [
        ch
        for ch in structural_channels
        if by_channel.get(ch, {}).get("classification") in ("LINEAR_SECULAR", "QUADRATIC_OR_ACCELERATING")
    ]
    validation_present = bool(validation_rows)
    validation_ok = validation_present and all(str(r.get("pass", "")).lower() == "true" for r in validation_rows)
    basin_present = bool(basin_rows)
    basin_ok = basin_present and all(str(r.get("pass", "")).lower() == "true" for r in basin_rows)
    numerical_shrink = validation_present and any(str(r.get("numerical_accumulation_suspected", "")).lower() == "true" for r in validation_rows)

    tests = [
        {"test": "phase_linear_constant_frequency", "pass": phase_ok and constant_slope, "target_run": target_id},
        {"test": "phase_aligned_structure_bounded", "pass": not structural, "structural_channels": ",".join(structural)},
        {"test": "numerical_validation_present", "pass": validation_present},
        {"test": "numerical_validation_agrees", "pass": validation_ok},
        {"test": "basin_orbital_stability_present", "pass": basin_present},
        {"test": "basin_orbital_stability_agrees", "pass": basin_ok},
    ]

    if numerical_shrink:
        return "TG_STATE_LOAD_DRIFT_NUMERICALLY_INDUCED", ["TG_STATE_LOAD_DRIFT_NUMERICALLY_INDUCED"], tests
    if structural:
        return "TG_STATE_LOAD_STRUCTURAL_SLOW_DRIFT_CONFIRMED", ["TG_STATE_LOAD_STRUCTURAL_SLOW_DRIFT_CONFIRMED"], tests
    if phase_ok and constant_slope and validation_ok and basin_ok:
        return (
            "TG_STATE_LOAD_ORBITALLY_BOUNDED_FREQUENCY_SHIFT_SUPPORTED",
            ["TG_STATE_LOAD_ORBITALLY_BOUNDED_FREQUENCY_SHIFT_SUPPORTED"],
            tests,
        )
    if phase_ok and constant_slope and not structural:
        return "TG_STATE_LOAD_LONG_TIME_DRIFT_UNRESOLVED", ["TG_STATE_LOAD_LONG_TIME_DRIFT_UNRESOLVED"], tests
    return "TG_STATE_LOAD_LONG_TIME_DRIFT_UNRESOLVED", ["TG_STATE_LOAD_LONG_TIME_DRIFT_UNRESOLVED"], tests


def write_docs(outdir: Path, status: str, labels: list[str], key: dict[str, Any]) -> None:
    docdir = ROOT / "docs" / "gravity_maturity"
    docdir.mkdir(parents=True, exist_ok=True)
    results = [
        "# TG-B1S-D Drift Decomposition Results",
        "",
        "Timestamp: 2026-07-14.",
        f"Run directory: `{outdir.as_posix()}`.",
        f"Status: `{status}`.",
        "",
        "## Labels",
        "",
        *(f"- `{label}`" for label in labels),
        "",
        "## Preserved Prior Labels",
        "",
        "- `TG_STATE_LOAD_BACKREACTION_ROBUST`",
        "- `TG_STATE_LOAD_BACKREACTION_ROBUST_BUT_WEAK`",
        "- `TG_STATE_LOAD_CONTINUOUS_SLOW_DRIFT`",
        "",
        "## Key Metrics",
        "",
        f"- Target run: `{key.get('target_run')}`.",
        f"- Final phase-aligned orbital distance: `{key.get('final_orbital_distance')}`.",
        f"- Final phase drift: `{key.get('final_delta_theta')}`.",
        f"- Asymptotic frequency slope: `{key.get('delta_omega_infty')}`.",
        f"- Structural channels: `{key.get('structural_channels')}`.",
        "",
        "## Interpretation",
        "",
        key.get("interpretation", ""),
        "",
        "No damping, coupling, source normalization or field map was changed. No bounded-feedback, gravity, photon, objective-time, geodesic, universal-free-fall, production or IRER-validation claim is made unless explicitly listed above.",
    ]
    (docdir / "TG_B1S_D_RESULTS.md").write_text("\n".join(results), encoding="utf-8")
    write_json(docdir / "TG_B1S_D_SUMMARY.json", {"status": status, "labels": labels, "run_directory": str(outdir), **key})
    inputs = [
        "# TG-B1S-D Documentation Inputs",
        "",
        f"- Status: `{status}`.",
        f"- Labels: `{', '.join(labels)}`.",
        f"- Artifacts: `{outdir.as_posix()}`.",
        "- Source: `STATE_LOAD_FEEDBACK` only.",
        "- Frozen model: no coefficient, damping, coupling, source or absorber redesign.",
        "",
        "## Strongest Tables",
        "",
        "- `orbital_distance.csv`",
        "- `phase_alignment.csv`",
        "- `drift_channel_fits.csv`",
        "- `asymptotic_frequency.csv`",
        "- `long_time_100_period.csv`",
        "- `numerical_validation.csv`",
        "- `basin_orbital_stability.csv`",
        "- `falsification_results.csv`",
        "",
        "## Caveats",
        "",
        "- Relative phase is allowed to grow under a stable frequency shift.",
        "- Phase-aligned profile distance, amplitude, width and leakage determine whether the drift is structural.",
        "- `TG_STATE_LOAD_BOUNDED_FEEDBACK_SUPPORTED` remains unpromoted unless validation and basin gates close.",
    ]
    (docdir / "TG_B1S_D_DOCUMENTATION_INPUTS.md").write_text("\n".join(inputs), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--matrix", choices=("quick", "primary", "closure"), default="primary")
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
    ap.add_argument("--sample-dt", type=float, default=1.0)
    ap.add_argument("--validation-sample-dt", type=float, default=1.5)
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
    outdir = Path(args.out) if args.out else ROOT / "sweep_runs" / f"TG_B1S_DRIFT_DECOMPOSITION_GPU_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "git_state_before.txt").write_text(b1s.git_state(), encoding="utf-8")
    pf = b1s.preflight(outdir)
    cfg = default_config(args)
    cfg["config_hash"] = b1s.config_hash(cfg)
    write_json(outdir / "environment_versions.json", {**pf, "stage": "TG-B1S-D_DRIFT_DECOMPOSITION"})
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
    write_json(
        outdir / "preregistered_matrix.json",
        {
            "matrix": args.matrix,
            "D0_phase_aligned_profile_comparison": True,
            "D1_separate_drift_channels": True,
            "D2_asymptotic_frequency_contract": True,
            "D3_100_period_run": args.matrix in ("primary", "closure"),
            "D4_numerical_validation": args.matrix == "closure",
            "D5_basin_test": args.matrix == "closure",
            "state_load_only": True,
        },
    )

    phi, prof = b1s.solve_qball(cfg)
    op = b1s.build_kg(cfg["N"], cfg["L"], cfg["c"], cfg["m"], cfg["dt"])
    refs = b1s.ref_source_norms(phi, cfg, op)
    base = b1s.baseline_contract(cfg, phi, prof)
    write_csv(outdir / "baseline_node_contract.csv", [base])
    if base["status"] != "TG_SOURCE_NODE_BASELINE_CLOSED":
        status = "TG_SOURCE_NODE_NOT_STATIONARY"
        write_csv(outdir / "falsification_results.csv", [{"test": "baseline_node_contract", "pass": False, "status": base["status"]}])
        write_docs(outdir, status, [status], {"target_run": None, "interpretation": "Baseline node contract failed; no drift decomposition was run."})
        write_csv(outdir / "artifact_hashes.csv", b1s.artifact_hashes(outdir))
        print(json.dumps({"status": status, "outdir": str(outdir)}, indent=2))
        return

    if args.matrix == "quick":
        run_specs = [("D0_2P_lam1", 2.0, 1.0)]
    else:
        run_specs = [("D2_50P_lam1", 50.0, 1.0), ("D3_100P_lam1", 100.0, 1.0)]

    all_orbital_rows: list[dict[str, Any]] = []
    all_phase_rows: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []
    asym_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    energy_rows: list[dict[str, Any]] = []
    run_rows: dict[str, list[dict[str, Any]]] = {}

    for run_id, periods, lam in run_specs:
        summary, pair_rows, full_rows, off_rows = run_pair_decomposition(run_id, periods, lam, cfg, phi, refs, args.sample_dt)
        run_rows[run_id] = pair_rows
        all_orbital_rows.extend(pair_rows)
        all_phase_rows.extend(
            {
                "run_id": r["run_id"],
                "t": r["t"],
                "period": r["period"],
                "alpha_star": r["alpha_star"],
                "alpha_star_translation": r["alpha_star_translation"],
                "delta_theta": r["delta_theta"],
                "delta_omega_inst": r["delta_omega_inst"],
            }
            for r in pair_rows
        )
        dr = drift_channel_rows(run_id, pair_rows)
        ar = asymptotic_frequency_row(run_id, pair_rows)
        drift_rows.extend(dr)
        asym_rows.append(ar)
        summary["structural_channels"] = ",".join(
            r["channel"]
            for r in dr
            if r["channel"] not in ("delta_theta", "delta_omega")
            and r["classification"] in ("LINEAR_SECULAR", "QUADRATIC_OR_ACCELERATING")
        )
        long_rows.append(summary)
        energy_rows.append(
            {
                "run_id": run_id,
                "periods": periods,
                "lambda_fb": lam,
                "ledger_residual_abs": summary["ledger_residual_abs"],
                "T_peak": summary["T_peak"],
                "G_peak": summary["G_peak"],
                "full_modal_leakage_max": summary["full_modal_leakage_max"],
            }
        )
        write_csv(outdir / "orbital_distance.csv", all_orbital_rows)
        write_csv(outdir / "phase_alignment.csv", all_phase_rows)
        write_csv(outdir / "drift_channel_fits.csv", drift_rows)
        write_csv(outdir / "asymptotic_frequency.csv", asym_rows)
        write_csv(outdir / "long_time_100_period.csv", long_rows)
        write_csv(outdir / "energy_ledger.csv", energy_rows)

    validation_rows: list[dict[str, Any]] = []
    basin_rows: list[dict[str, Any]] = []

    if args.matrix == "closure":
        base50 = next((r for r in long_rows if r["run_id"] == "D2_50P_lam1"), None)
        validation_specs = [
            ("dt_half", {"dt": cfg["dt"] / 2.0}),
            ("grid_refined", {"N": 56}),
            ("larger_box", {"N": 56, "L": 12.0}),
            ("absorber_wider", {"absorb_width": cfg["absorb_width"] * 1.25}),
        ]
        for name, overrides in validation_specs:
            vcfg = dict(cfg)
            vcfg.update(overrides)
            vphi = phi
            if vcfg["N"] != cfg["N"] or vcfg["L"] != cfg["L"]:
                vphi, _ = b1s.solve_qball(vcfg)
            run_id = f"D4_{name}_50P_lam1"
            summary, rows, _full_rows, _off_rows = run_pair_decomposition(run_id, 50.0, 1.0, vcfg, vphi, refs, args.validation_sample_dt)
            fit = asymptotic_frequency_row(run_id, rows)
            base_slope = float(base50["delta_modal_phase_slope"]) if base50 else float("nan")
            rel = abs(float(fit.get("delta_omega_infty", np.nan)) - base_slope) / max(abs(base_slope), 1e-30)
            shrink = summary["phase_translation_aligned_distance_final"] < 0.5 * float(base50["phase_translation_aligned_distance_final"]) if base50 else False
            passed = (
                summary["full_profile_overlap"] > 0.999
                and summary["full_modal_leakage_max"] < 1e-3
                and rel < 0.50
                and summary["ledger_residual_abs"] < 1e-4
            )
            validation_rows.append(
                {
                    "case": name,
                    "run_id": run_id,
                    "pass": passed,
                    "delta_omega_infty": fit.get("delta_omega_infty"),
                    "relative_slope_difference": rel,
                    "orbital_distance_final": summary["phase_translation_aligned_distance_final"],
                    "modal_leakage_max": summary["full_modal_leakage_max"],
                    "ledger_residual_abs": summary["ledger_residual_abs"],
                    "numerical_accumulation_suspected": bool(shrink),
                }
            )
            write_csv(outdir / "numerical_validation.csv", validation_rows)

        basin_specs = [
            ("amplitude_perturb", "amplitude_perturb", None, 0.0, 0.0),
            ("width_perturb", "width_perturb", None, 0.0, 0.0),
            ("global_phase", "global_phase", None, 0.0, 0.0),
            ("translated", "translated", "translated", 0.0, 0.0),
            ("initial_T_seed", "base", None, 1.0e-5, 0.0),
            ("initial_G_seed", "base", None, 0.0, 1.0e-5),
        ]
        base50_dist = float(base50["phase_translation_aligned_distance_final"]) if base50 else float("nan")
        for name, kind, mode_kind, t_seed, g_seed in basin_specs:
            run_id = f"D5_{name}_25P_lam1"
            summary, rows, _full_rows, _off_rows = run_pair_decomposition(run_id, 25.0, 1.0, cfg, phi, refs, args.validation_sample_dt, kind=kind, mode_kind=mode_kind, t_seed=t_seed, g_seed=g_seed)
            same_orbital = summary["full_profile_overlap"] > 0.999 and summary["full_modal_leakage_max"] < 1e-3
            basin_rows.append(
                {
                    "case": name,
                    "run_id": run_id,
                    "pass": same_orbital,
                    "orbital_distance_final": summary["phase_translation_aligned_distance_final"],
                    "reference_50P_orbital_distance": base50_dist,
                    "modal_leakage_max": summary["full_modal_leakage_max"],
                    "profile_overlap": summary["full_profile_overlap"],
                }
            )
            write_csv(outdir / "basin_orbital_stability.csv", basin_rows)

    write_csv(outdir / "numerical_validation.csv", validation_rows)
    write_csv(outdir / "basin_orbital_stability.csv", basin_rows)
    status, labels, falsification_rows = classify_interpretation(run_rows, drift_rows, asym_rows, validation_rows, basin_rows)
    write_csv(outdir / "falsification_results.csv", falsification_rows)
    target_run = "D3_100P_lam1" if "D3_100P_lam1" in run_rows else "D2_50P_lam1" if "D2_50P_lam1" in run_rows else run_specs[-1][0]
    target_rows = run_rows[target_run]
    target_freq = next((r for r in asym_rows if r.get("run_id") == target_run), {})
    structural = ",".join(
        r["channel"]
        for r in drift_rows
        if r["run_id"] == target_run
        and r["channel"] not in ("delta_theta", "delta_omega")
        and r["classification"] in ("LINEAR_SECULAR", "QUADRATIC_OR_ACCELERATING")
    )
    if status == "TG_STATE_LOAD_STRUCTURAL_SLOW_DRIFT_CONFIRMED":
        interp = "After removing global phase, the full-loop node still shows secular or accelerating change in shape/modal channels. The frozen state-load model remains localized but does not show an orbitally bounded shifted node."
    elif status == "TG_STATE_LOAD_ORBITALLY_BOUNDED_FREQUENCY_SHIFT_SUPPORTED":
        interp = "The relative phase is explained by an approximately constant frequency shift while phase-aligned profile distance and shape channels remain bounded across validation and basin checks."
    else:
        interp = "The audit separates phase drift from profile drift, but the available gates do not yet distinguish stable frequency shift, structural drift and numerical accumulation robustly enough for promotion."
    key = {
        "target_run": target_run,
        "final_orbital_distance": target_rows[-1]["phase_translation_aligned_distance"],
        "final_phase_aligned_distance": target_rows[-1]["phase_aligned_distance"],
        "final_delta_theta": target_rows[-1]["delta_theta"],
        "delta_omega_infty": target_freq.get("delta_omega_infty"),
        "structural_channels": structural,
        "bounded_feedback_promoted": False,
        "preserved_prior_labels": [
            "TG_STATE_LOAD_BACKREACTION_ROBUST",
            "TG_STATE_LOAD_BACKREACTION_ROBUST_BUT_WEAK",
            "TG_STATE_LOAD_CONTINUOUS_SLOW_DRIFT",
        ],
        "interpretation": interp,
    }
    write_docs(outdir, status, labels, key)
    (outdir / "TECHNICAL_HANDOFF.md").write_text(
        "\n".join(
            [
                "# TG-B1S-D Technical Handoff",
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
                "Frozen state-load model only; no R_relax, L_lock, P_threshold, photon, radiative, gravity, objective-time, geodesic or production claim.",
                "",
                "## Decision",
                "",
                interp,
            ]
        ),
        encoding="utf-8",
    )
    write_csv(outdir / "artifact_hashes.csv", b1s.artifact_hashes(outdir))
    print(json.dumps({"status": status, "labels": labels, "outdir": str(outdir)}, indent=2))


if __name__ == "__main__":
    main()
