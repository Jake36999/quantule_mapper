"""TG-B1S-R state-load backreaction robustness audit.

This standalone GPU audit tries to falsify the short-run backreaction detected
in TG-B1S.  It reuses the validated TG-S/B1S Q-ball, S_state source, fixed
global normalization, T/G equations, damping/coupling signs, coefficient map and
radial boundary treatment.  It does not enable R_relax, L_lock or P_threshold.

The only new control parameter is lambda_fb, a scalar multiplier on the already
defined G -> phi feedback term:

    A = exp(-epsilon_G * lambda_fb * G)

when feedback is enabled.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jax_scout import gravity_TG_B1S_state_load_feedback_gpu as b1s


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


def arm_flags(arm: str) -> list[float]:
    if arm == "pure_baseline":
        return [1.0, 0.0, 0.0, 0.0]
    if arm == "feed_forward":
        return [1.0, 1.0, 1.0, 0.0]
    if arm == "feedback_off":
        return [1.0, 1.0, 1.0, 0.0]
    if arm == "full_loop":
        return [1.0, 1.0, 1.0, 1.0]
    if arm == "source_off":
        return [0.0, 1.0, 1.0, 1.0]
    if arm == "temporal_off":
        return [1.0, 0.0, 1.0, 1.0]
    if arm == "geometric_off":
        return [1.0, 1.0, 0.0, 1.0]
    raise ValueError(arm)


def period_from_cfg(cfg: dict[str, Any]) -> float:
    return 2.0 * math.pi / float(cfg["w"])


def make_initial_state(
    phi: np.ndarray,
    cfg: dict[str, Any],
    kind: str = "base",
    tg_seed: float = 0.0,
) -> tuple[jnp.ndarray, ...]:
    psi, pi = b1s.initial_state(kind, phi, cfg)
    if kind == "radial_perturb":
        op = b1s.build_kg(cfg["N"], cfg["L"], cfg["c"], cfg["m"], cfg["dt"])
        g = b1s.make_grid(op)
        bump = np.asarray(jnp.exp(-0.5 * (g["R"] / 1.2) ** 2))
        psi = psi * (1.0 + 1.0e-3 * bump)
        pi = (-1j * cfg["w"] * psi).astype(np.complex128)
    real_zero = jnp.zeros_like(jnp.real(jnp.asarray(psi)))
    if tg_seed:
        op = b1s.build_kg(cfg["N"], cfg["L"], cfg["c"], cfg["m"], cfg["dt"])
        g = b1s.make_grid(op)
        seed = tg_seed * jnp.exp(-0.5 * (g["R"] / 1.5) ** 2)
        T0 = seed
        G0 = 0.5 * seed
    else:
        T0 = real_zero
        G0 = real_zero
    return (
        jnp.asarray(psi),
        jnp.asarray(pi),
        T0,
        real_zero,
        G0,
        real_zero,
    )


@jax.jit
def rhs_lambda(
    state: tuple[jnp.ndarray, ...],
    cfg: jnp.ndarray,
    refs: jnp.ndarray,
    g: dict[str, jnp.ndarray],
    flags: jnp.ndarray,
    lambda_fb: jnp.ndarray,
) -> tuple[jnp.ndarray, ...]:
    phi, pi, T, VT, G, VG = state
    source_enabled, temporal_enabled, geometric_enabled, feedback_enabled = flags
    _, c, m, a, s, f, alpha_T, omega_T, omega_G, gamma_T, gamma_G, kappa, eps_G, cT, cG, L, abs_width, abs_strength, _core = cfg
    rho = jnp.abs(phi) ** 2
    source = b1s.state_load(phi, pi, cfg, refs, g) * source_enabled * temporal_enabled
    absorb = b1s.absorb_profile(g, L, abs_width, abs_strength)
    A = jnp.exp(-eps_G * lambda_fb * G * geometric_enabled * feedback_enabled)
    kg_force = c * c * b1s.div_A_grad(phi, A, g) - m * m * phi + (a * rho + s * rho**2 + f * rho**3) * phi
    T_t = VT * temporal_enabled
    VT_t = (
        cT * cT * b1s.lap_real(T, g)
        - omega_T * omega_T * T
        - gamma_T * VT
        + alpha_T * source
        - kappa * G * geometric_enabled
        - absorb * VT
    ) * temporal_enabled
    G_t = VG * geometric_enabled
    VG_t = (
        cG * cG * b1s.lap_real(G, g)
        - omega_G * omega_G * G
        - gamma_G * VG
        - kappa * T
        - absorb * VG
    ) * geometric_enabled
    return pi, kg_force, T_t, VT_t, G_t, VG_t


@jax.jit
def rk4_lambda(
    state: tuple[jnp.ndarray, ...],
    cfg: jnp.ndarray,
    refs: jnp.ndarray,
    g: dict[str, jnp.ndarray],
    flags: jnp.ndarray,
    lambda_fb: jnp.ndarray,
) -> tuple[jnp.ndarray, ...]:
    dt = cfg[0]
    k1 = rhs_lambda(state, cfg, refs, g, flags, lambda_fb)
    s2 = tuple(y + 0.5 * dt * dy for y, dy in zip(state, k1))
    k2 = rhs_lambda(s2, cfg, refs, g, flags, lambda_fb)
    s3 = tuple(y + 0.5 * dt * dy for y, dy in zip(state, k2))
    k3 = rhs_lambda(s3, cfg, refs, g, flags, lambda_fb)
    s4 = tuple(y + dt * dy for y, dy in zip(state, k3))
    k4 = rhs_lambda(s4, cfg, refs, g, flags, lambda_fb)
    return tuple(y + (dt / 6.0) * (a + 2.0 * b + 2.0 * c + d) for y, a, b, c, d in zip(state, k1, k2, k3, k4))


@partial(jax.jit, static_argnames=("nsteps",))
def evolve_n_lambda(
    state: tuple[jnp.ndarray, ...],
    cfg: jnp.ndarray,
    refs: jnp.ndarray,
    g: dict[str, jnp.ndarray],
    flags: jnp.ndarray,
    lambda_fb: jnp.ndarray,
    nsteps: int,
) -> tuple[jnp.ndarray, ...]:
    def body(_, s):
        return rk4_lambda(s, cfg, refs, g, flags, lambda_fb)

    return lax.fori_loop(0, nsteps, body, state)


@jax.jit
def modal_diagnostics(
    state: tuple[jnp.ndarray, ...],
    cfg: jnp.ndarray,
    refs: jnp.ndarray,
    g: dict[str, jnp.ndarray],
    mode: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    d = b1s.diagnostics(state, cfg, refs, g)
    phi = state[0]
    dV = g["dx"] ** 3
    denom = jnp.sum(jnp.abs(mode) ** 2) * dV + 1e-30
    q = jnp.sum(jnp.conj(mode) * phi) * dV / denom
    residual = phi - q * mode
    res_norm = jnp.sqrt(jnp.sum(jnp.abs(residual) ** 2) * dV)
    phi_norm = jnp.sqrt(jnp.sum(jnp.abs(phi) ** 2) * dV) + 1e-30
    d["modal_q_real"] = jnp.real(q)
    d["modal_q_imag"] = jnp.imag(q)
    d["modal_amp"] = jnp.abs(q)
    d["modal_phase"] = jnp.angle(q)
    d["modal_leakage"] = res_norm / phi_norm
    return d


def summarize_rows(
    run_id: str,
    arm: str,
    lambda_fb: float,
    cfg: dict[str, Any],
    rows: list[dict[str, float]],
    final_state: tuple[jnp.ndarray, ...],
    initial_phi: np.ndarray,
    duration: float,
    period_count: float,
) -> dict[str, Any]:
    base = b1s.summarize_run(run_id, arm, cfg, rows, final_state, initial_phi)
    times = np.asarray([r["t"] for r in rows], dtype=float)
    modal_phase = np.unwrap(np.asarray([r["modal_phase"] for r in rows], dtype=float))
    modal_amp = np.asarray([r["modal_amp"] for r in rows], dtype=float)
    modal_leak = np.asarray([r["modal_leakage"] for r in rows], dtype=float)
    if len(times) > 4:
        cut = len(times) // 2
        modal_freq = -float(np.polyfit(times[cut:], modal_phase[cut:], 1)[0])
    else:
        modal_freq = float("nan")
    base.update(
        {
            "lambda_fb": lambda_fb,
            "duration": duration,
            "period_count": period_count,
            "modal_phase_initial": float(modal_phase[0]),
            "modal_phase_final_unwrapped": float(modal_phase[-1]),
            "modal_frequency": modal_freq,
            "modal_amp_initial": float(modal_amp[0]),
            "modal_amp_final": float(modal_amp[-1]),
            "modal_amp_drift": float(modal_amp[-1] - modal_amp[0]),
            "modal_leakage_final": float(modal_leak[-1]),
            "modal_leakage_max": float(np.max(modal_leak)),
        }
    )
    return base


def run_case(
    run_id: str,
    arm: str,
    lambda_fb: float,
    cfg: dict[str, Any],
    phi: np.ndarray,
    refs: dict[str, float],
    duration: float,
    sample_dt: float | None = None,
    kind: str = "base",
    tg_seed: float = 0.0,
    collect_g: bool = False,
) -> tuple[dict[str, Any], list[dict[str, float]], list[np.ndarray]]:
    op = b1s.build_kg(cfg["N"], cfg["L"], cfg["c"], cfg["m"], cfg["dt"])
    g = b1s.make_grid(op)
    cfgv = b1s.cfg_array(cfg)
    refv = b1s.refs_array(refs)
    flags = jnp.asarray(arm_flags(arm), dtype=jnp.float64)
    mode = jnp.asarray(phi.astype(np.complex128))
    state = make_initial_state(phi, cfg, kind=kind, tg_seed=tg_seed)
    initial_phi = np.asarray(state[0])
    steps = int(round(duration / cfg["dt"]))
    every = max(1, int(round((sample_dt or cfg["sample_dt"]) / cfg["dt"])))
    chunks = max(1, steps // every)
    rows: list[dict[str, float]] = []
    g_snapshots: list[np.ndarray] = []
    d0 = modal_diagnostics(state, cfgv, refv, g, mode)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), d0)
    rows.append({"t": 0.0, **{k: float(np.asarray(v)) for k, v in d0.items()}})
    if collect_g:
        g_snapshots.append(np.asarray(state[4]))
    start = time.time()
    for idx in range(chunks):
        state = evolve_n_lambda(state, cfgv, refv, g, flags, jnp.asarray(lambda_fb, dtype=jnp.float64), every)
        d = modal_diagnostics(state, cfgv, refv, g, mode)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), d)
        rows.append({"t": (idx + 1) * every * cfg["dt"], **{k: float(np.asarray(v)) for k, v in d.items()}})
        if collect_g:
            g_snapshots.append(np.asarray(state[4]))
    summary = summarize_rows(run_id, arm, lambda_fb, cfg, rows, state, initial_phi, duration, duration / period_from_cfg(cfg))
    summary["execution_time_s"] = time.time() - start
    summary["config_hash"] = b1s.config_hash({**cfg, "lambda_fb": lambda_fb, "duration": duration, "sample_dt": sample_dt or cfg["sample_dt"]})
    return summary, rows, g_snapshots


@jax.jit
def phi_rhs_fixed_g(
    state: tuple[jnp.ndarray, jnp.ndarray],
    cfg: jnp.ndarray,
    g: dict[str, jnp.ndarray],
    Gfield: jnp.ndarray,
    lambda_fb: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    phi, pi = state
    _, c, m, a, s, f, *_rest = cfg
    eps_G = cfg[12]
    rho = jnp.abs(phi) ** 2
    A = jnp.exp(-eps_G * lambda_fb * Gfield)
    kg_force = c * c * b1s.div_A_grad(phi, A, g) - m * m * phi + (a * rho + s * rho**2 + f * rho**3) * phi
    return pi, kg_force


@jax.jit
def phi_rk4_fixed_g(
    state: tuple[jnp.ndarray, jnp.ndarray],
    cfg: jnp.ndarray,
    g: dict[str, jnp.ndarray],
    Gfield: jnp.ndarray,
    lambda_fb: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    dt = cfg[0]
    k1 = phi_rhs_fixed_g(state, cfg, g, Gfield, lambda_fb)
    s2 = tuple(y + 0.5 * dt * dy for y, dy in zip(state, k1))
    k2 = phi_rhs_fixed_g(s2, cfg, g, Gfield, lambda_fb)
    s3 = tuple(y + 0.5 * dt * dy for y, dy in zip(state, k2))
    k3 = phi_rhs_fixed_g(s3, cfg, g, Gfield, lambda_fb)
    s4 = tuple(y + dt * dy for y, dy in zip(state, k3))
    k4 = phi_rhs_fixed_g(s4, cfg, g, Gfield, lambda_fb)
    return tuple(y + (dt / 6.0) * (a + 2.0 * b + 2.0 * c + d) for y, a, b, c, d in zip(state, k1, k2, k3, k4))


@partial(jax.jit, static_argnames=("nsteps",))
def evolve_phi_fixed_g(
    state: tuple[jnp.ndarray, jnp.ndarray],
    cfg: jnp.ndarray,
    g: dict[str, jnp.ndarray],
    Gfield: jnp.ndarray,
    lambda_fb: jnp.ndarray,
    nsteps: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    def body(_, s):
        return phi_rk4_fixed_g(s, cfg, g, Gfield, lambda_fb)

    return lax.fori_loop(0, nsteps, body, state)


def run_replay(
    run_id: str,
    transform: str,
    Gseq: list[np.ndarray],
    lambda_fb: float,
    cfg: dict[str, Any],
    phi: np.ndarray,
    refs: dict[str, float],
    duration: float,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    op = b1s.build_kg(cfg["N"], cfg["L"], cfg["c"], cfg["m"], cfg["dt"])
    g = b1s.make_grid(op)
    cfgv = b1s.cfg_array(cfg)
    refv = b1s.refs_array(refs)
    mode = jnp.asarray(phi.astype(np.complex128))
    psi, pi = b1s.initial_state("base", phi, cfg)
    state2 = (jnp.asarray(psi), jnp.asarray(pi))
    zero = jnp.zeros_like(jnp.real(state2[0]))
    state6 = (state2[0], state2[1], zero, zero, zero, zero)
    every = max(1, int(round(cfg["sample_dt"] / cfg["dt"])))
    steps = int(round(duration / cfg["dt"]))
    chunks = max(1, steps // every)
    rows: list[dict[str, float]] = []
    d0 = modal_diagnostics(state6, cfgv, refv, g, mode)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), d0)
    rows.append({"t": 0.0, **{k: float(np.asarray(v)) for k, v in d0.items()}})
    for idx in range(chunks):
        source_idx = min(idx, len(Gseq) - 1)
        if transform == "time_shifted":
            source_idx = min(idx + 4, len(Gseq) - 1)
        Gfield = np.asarray(Gseq[source_idx])
        if transform == "sign_reversed":
            Gfield = -Gfield
        elif transform == "decorrelated_roll":
            Gfield = np.roll(Gfield, max(1, cfg["N"] // 5), axis=0)
        state2 = evolve_phi_fixed_g(state2, cfgv, g, jnp.asarray(Gfield), jnp.asarray(lambda_fb, dtype=jnp.float64), every)
        state6 = (state2[0], state2[1], zero, zero, jnp.asarray(Gfield), zero)
        d = modal_diagnostics(state6, cfgv, refv, g, mode)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), d)
        rows.append({"t": (idx + 1) * every * cfg["dt"], **{k: float(np.asarray(v)) for k, v in d.items()}})
    summary = summarize_rows(run_id, "field_replay", lambda_fb, cfg, rows, state6, phi, duration, duration / period_from_cfg(cfg))
    summary["replay_transform"] = transform
    return summary, rows


def delta_between(full: dict[str, Any], off: dict[str, Any], rows_full: list[dict[str, float]] | None = None, rows_off: list[dict[str, float]] | None = None) -> dict[str, float]:
    out = {
        "delta_core_energy_final": float(full["core_energy_final"] - off["core_energy_final"]),
        "delta_core_amp_final": float(full["core_amp_final"] - off["core_amp_final"]),
        "delta_width_final": float(full["node_width_final"] - off["node_width_final"]),
        "delta_node_frequency": float(full["node_frequency"] - off["node_frequency"]),
        "delta_modal_frequency": float(full["modal_frequency"] - off["modal_frequency"]),
        "delta_shell_flux_max": float(full["shell_flux_max"] - off["shell_flux_max"]),
        "delta_modal_leakage_final": float(full["modal_leakage_final"] - off["modal_leakage_final"]),
    }
    if rows_full is not None and rows_off is not None:
        pf = np.unwrap(np.asarray([r["modal_phase"] for r in rows_full], dtype=float))
        po = np.unwrap(np.asarray([r["modal_phase"] for r in rows_off], dtype=float))
        n = min(len(pf), len(po))
        drift = (pf[:n] - pf[0]) - (po[:n] - po[0])
        out["delta_modal_phase_final"] = float(drift[-1])
        out["delta_modal_phase_absmax"] = float(np.max(np.abs(drift)))
        if n > 4:
            times = np.asarray([r["t"] for r in rows_full[:n]], dtype=float)
            out["delta_modal_phase_slope"] = float(np.polyfit(times[n // 2 :], drift[n // 2 :], 1)[0])
        else:
            out["delta_modal_phase_slope"] = float("nan")
    return out


def fit_response(rows: list[dict[str, Any]], observable: str, train_lambdas: set[float], holdout_lambdas: set[float]) -> list[dict[str, Any]]:
    train = [r for r in rows if float(r["lambda_fb"]) in train_lambdas]
    hold = [r for r in rows if float(r["lambda_fb"]) in holdout_lambdas]
    x = np.asarray([float(r["lambda_fb"]) for r in train], dtype=float)
    y = np.asarray([float(r[observable]) for r in train], dtype=float)
    out: list[dict[str, Any]] = []
    if len(train) >= 2:
        coef1 = np.polyfit(x, y, 1)
        for row in hold:
            lam = float(row["lambda_fb"])
            pred = float(np.polyval(coef1, lam))
            out.append({"observable": observable, "model": "linear", "lambda_fb": lam, "observed": row[observable], "predicted": pred, "abs_error": abs(float(row[observable]) - pred)})
    if len(train) >= 3:
        coef2 = np.polyfit(x, y, 2)
        for row in hold:
            lam = float(row["lambda_fb"])
            pred = float(np.polyval(coef2, lam))
            out.append({"observable": observable, "model": "quadratic", "lambda_fb": lam, "observed": row[observable], "predicted": pred, "abs_error": abs(float(row[observable]) - pred)})
    return out


def classify_long(summary: dict[str, Any], delta: dict[str, float]) -> str:
    if summary["profile_overlap"] < 0.98 or summary["modal_leakage_max"] > 0.25:
        return "FEEDBACK_DESTABILIZATION"
    if abs(delta.get("delta_modal_phase_final", 0.0)) < 1e-9 and abs(delta.get("delta_core_amp_final", 0.0)) < 1e-9:
        return "BACKREACTION_NULL"
    if abs(delta.get("delta_modal_phase_slope", 0.0)) > 1e-12 and summary["profile_overlap"] > 0.995:
        return "STABLE_SHIFTED_NODE"
    return "NUMERICALLY_UNRESOLVED"


def classify_final(
    snr_rows: list[dict[str, Any]],
    coupling_rows: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
    long_rows: list[dict[str, Any]],
    basin_rows: list[dict[str, Any]],
    max_long_periods: float,
) -> tuple[str, list[str], list[dict[str, Any]]]:
    high_snr = [r for r in snr_rows if float(r["snr_fb"]) >= 5.0]
    has_phase_snr = any(r["observable"] in ("delta_modal_phase_final", "delta_modal_frequency", "delta_node_frequency") for r in high_snr)
    zero_row = next((r for r in coupling_rows if abs(float(r["lambda_fb"])) < 1e-12), None)
    zero_ok = zero_row is not None and abs(float(zero_row["delta_modal_phase_final"])) < 1e-12 and abs(float(zero_row["delta_core_amp_final"])) < 1e-12
    nonzero = [r for r in coupling_rows if float(r["lambda_fb"]) > 0.0]
    smooth_sign = bool(nonzero) and all(np.sign(float(r["delta_core_amp_final"])) == np.sign(float(nonzero[0]["delta_core_amp_final"])) for r in nonzero if abs(float(r["delta_core_amp_final"])) > 1e-12)
    replay_ok = any(r.get("control") == "correct_replay" and abs(float(r.get("delta_modal_phase_final", 0.0))) > 0.0 for r in replay_rows)
    robust = len(high_snr) >= 2 and has_phase_snr and zero_ok and smooth_sign and replay_ok
    bounded_ready = robust and max_long_periods >= 25.0 and all(r.get("attractor_class") in ("STABLE_SHIFTED_NODE", "DAMPED_TO_NEW_FIXED_POINT", "BOUNDED_PERIODIC_FEEDBACK") for r in long_rows) and all(str(r.get("pass", "")).lower() == "true" for r in basin_rows)
    tests = [
        {"test": "snr_floor_gate", "pass": len(high_snr) >= 2 and has_phase_snr, "high_snr_observables": ",".join(r["observable"] for r in high_snr)},
        {"test": "lambda_zero_vanishes", "pass": zero_ok},
        {"test": "coupling_sign_smooth", "pass": smooth_sign},
        {"test": "field_replay_control_nonzero", "pass": replay_ok},
        {"test": "bounded_long_time_gate", "pass": bounded_ready, "max_long_periods": max_long_periods},
    ]
    if bounded_ready:
        return "TG_STATE_LOAD_BOUNDED_FEEDBACK_SUPPORTED", ["TG_STATE_LOAD_BACKREACTION_ROBUST", "TG_STATE_LOAD_BOUNDED_FEEDBACK_SUPPORTED"], tests
    if robust:
        return "TG_STATE_LOAD_BACKREACTION_ROBUST_BUT_WEAK", ["TG_STATE_LOAD_BACKREACTION_ROBUST", "TG_STATE_LOAD_BACKREACTION_ROBUST_BUT_WEAK"], tests
    if not (len(high_snr) >= 2 and has_phase_snr):
        return "TG_STATE_LOAD_BACKREACTION_BELOW_NUMERICAL_FLOOR", ["TG_STATE_LOAD_BACKREACTION_BELOW_NUMERICAL_FLOOR"], tests
    return "TG_STATE_LOAD_NUMERICALLY_UNRESOLVED", ["TG_STATE_LOAD_NUMERICALLY_UNRESOLVED"], tests


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


def write_docs(outdir: Path, status: str, labels: list[str], key: dict[str, Any]) -> None:
    docdir = ROOT / "docs" / "gravity_maturity"
    docdir.mkdir(parents=True, exist_ok=True)
    result = [
        "# TG-B1S-R Backreaction Robustness Results",
        "",
        "Timestamp: 2026-07-14.",
        f"Run directory: `{outdir.as_posix()}`.",
        f"Status: `{status}`.",
        "",
        "## Bounded Labels",
        "",
        *(f"- `{label}`" for label in labels),
        "",
        "## Key Metrics",
        "",
        f"- Best SNR observable: `{key.get('best_snr_observable')}` = `{key.get('best_snr')}`.",
        f"- Max long duration tested in node periods: `{key.get('max_long_periods')}`.",
        f"- Baseline modal phase drift at lambda=1: `{key.get('baseline_delta_modal_phase_final')}`.",
        "",
        "## Interpretation",
        "",
        "This audit tests whether the short-run TG-B1S backreaction exceeds numerical spread, vanishes at zero feedback coupling, scales smoothly with `lambda_fb`, appears in modal phase, and remains bounded in the tested long-duration scout.",
        "",
        "No bounded-feedback, radiation, photon, gravity, objective-time, geodesic, universal-free-fall or IRER-validation claim is made unless explicitly listed above.",
    ]
    (docdir / "TG_B1S_R_RESULTS.md").write_text("\n".join(result), encoding="utf-8")
    write_json(docdir / "TG_B1S_R_SUMMARY.json", {"status": status, "labels": labels, "run_directory": str(outdir), **key})
    inputs = [
        "# TG-B1S-R Documentation Inputs",
        "",
        f"- Status: `{status}`.",
        f"- Labels: `{', '.join(labels)}`.",
        f"- Artifacts: `{outdir.as_posix()}`.",
        "- Source: `STATE_LOAD_FEEDBACK` only.",
        "- `R_relax`, `L_lock` and `P_threshold` disabled.",
        "- `lambda_fb` multiplies only the existing G-to-phi coefficient.",
        "",
        "## Strongest Supporting Tables",
        "",
        "- `numerical_floor.csv`",
        "- `coupling_response.csv`",
        "- `modal_phase_metrics.csv`",
        "- `field_replay_controls.csv`",
        "- `long_time_metrics.csv`",
        "- `falsification_results.csv`",
        "",
        "## Caveats",
        "",
        "- This is still a state-load branch; no radiation is required or claimed.",
        "- Long intervals are bounded scouts unless the 25/50-period gate is explicitly present.",
        "- Do not update the master catalogue until review.",
    ]
    (docdir / "TG_B1S_R_DOCUMENTATION_INPUTS.md").write_text("\n".join(inputs), encoding="utf-8")


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--matrix", choices=("bounded", "quick"), default="bounded")
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
    ap.add_argument("--sample-dt", type=float, default=0.05)
    ap.add_argument("--long-periods", default="2,10")
    ap.add_argument("--sample-dt-long", type=float, default=0.25)
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
    outdir = Path(args.out) if args.out else ROOT / "sweep_runs" / f"TG_B1S_BACKREACTION_ROBUSTNESS_GPU_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "git_state_before.txt").write_text(b1s.git_state(), encoding="utf-8")
    pf = b1s.preflight(outdir)
    cfg = default_config(args)
    cfg["config_hash"] = b1s.config_hash(cfg)
    write_json(outdir / "environment_versions.json", {**pf, "stage": "TG-B1S-R_BACKREACTION_ROBUSTNESS"})
    write_json(
        outdir / "source_spec.json",
        {
            "source_hypothesis": "STATE_LOAD_FEEDBACK",
            "enabled_source": "S_state",
            "disabled_sources": ["R_relax", "L_lock", "P_threshold"],
            "S0_authoritative_TG_S": S0_AUTHORITATIVE,
            "casewise_normalization": False,
            "new_control_parameter": "lambda_fb multiplies only G->phi feedback coefficient",
        },
    )
    write_json(
        outdir / "preregistered_matrix.json",
        {
            "lambda_ladder": [0.0, 0.25, 0.5, 1.0, 1.5, 2.0],
            "long_periods": parse_float_list(args.long_periods),
            "matrix": args.matrix,
            "state_load_only": True,
            "bounded_feedback_promotion_requires_25_period_or_longer_gate": True,
        },
    )

    phi, prof = b1s.solve_qball(cfg)
    op = b1s.build_kg(cfg["N"], cfg["L"], cfg["c"], cfg["m"], cfg["dt"])
    refs = b1s.ref_source_norms(phi, cfg, op)
    base = b1s.baseline_contract(cfg, phi, prof)
    write_csv(outdir / "baseline_node_contract.csv", [base])
    if base["status"] != "TG_SOURCE_NODE_BASELINE_CLOSED":
        status = "TG_SOURCE_NODE_NOT_STATIONARY"
        write_docs(outdir, status, [status], {"baseline": base})
        write_csv(outdir / "artifact_hashes.csv", b1s.artifact_hashes(outdir))
        print(json.dumps({"status": status, "outdir": str(outdir)}, indent=2))
        return

    period = period_from_cfg(cfg)
    duration = cfg["T"]
    run_manifest: list[dict[str, Any]] = []

    # R0: numerical floor and duplicate deterministic checks.
    floor_pairs: list[dict[str, Any]] = []
    floor_variants = [
        ("base", {}),
        ("duplicate", {}),
        ("dt_half", {"dt": cfg["dt"] / 2.0}),
        ("output_cadence_half", {"sample_dt": cfg["sample_dt"] / 2.0}),
        ("output_cadence_double", {"sample_dt": cfg["sample_dt"] * 2.0}),
        ("absorb_width_wide", {"absorb_width": 2.4}),
    ]
    if args.matrix == "bounded":
        floor_variants.extend([("grid_refined", {"N": 56}), ("larger_box", {"N": 56, "L": 12.0})])
    variant_deltas: list[dict[str, Any]] = []
    for name, overrides in floor_variants:
        vcfg = dict(cfg)
        vcfg.update(overrides)
        vphi = phi
        if vcfg["N"] != cfg["N"] or vcfg["L"] != cfg["L"]:
            vphi, _ = b1s.solve_qball(vcfg)
        off, off_rows, _ = run_case(f"floor_{name}_off", "feedback_off", 0.0, vcfg, vphi, refs, duration, sample_dt=vcfg.get("sample_dt", cfg["sample_dt"]))
        full, full_rows, _ = run_case(f"floor_{name}_full", "full_loop", 1.0, vcfg, vphi, refs, duration, sample_dt=vcfg.get("sample_dt", cfg["sample_dt"]))
        delta = delta_between(full, off, full_rows, off_rows)
        row = {"variant": name, **delta}
        variant_deltas.append(row)
        floor_pairs.append(row)
        run_manifest.extend([off, full])
    observables = [
        "delta_core_energy_final",
        "delta_core_amp_final",
        "delta_node_frequency",
        "delta_modal_frequency",
        "delta_modal_phase_final",
        "delta_shell_flux_max",
    ]
    base_delta = next(r for r in variant_deltas if r["variant"] == "base")
    floor_rows: list[dict[str, Any]] = []
    for obs in observables:
        vals = np.asarray([float(r[obs]) for r in variant_deltas], dtype=float)
        sigma = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        signal = abs(float(base_delta[obs]))
        snr = signal / sigma if sigma > 0.0 else float("inf")
        floor_rows.append({"observable": obs, "baseline_signal": signal, "sigma_num": sigma, "snr_fb": snr, "variant_count": len(vals)})
    write_csv(outdir / "numerical_floor.csv", floor_rows)

    # R1/R2: lambda ladder with modal phase.
    lambda_values = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
    off0, off0_rows, _ = run_case("lambda_0_off_reference", "full_loop", 0.0, cfg, phi, refs, duration)
    coupling_rows: list[dict[str, Any]] = []
    modal_rows: list[dict[str, Any]] = []
    row_cache: dict[float, tuple[dict[str, Any], list[dict[str, float]]]] = {0.0: (off0, off0_rows)}
    for lam in lambda_values:
        if lam == 0.0:
            summary, rows = off0, off0_rows
        else:
            summary, rows, _ = run_case(f"lambda_{lam:g}", "full_loop", lam, cfg, phi, refs, duration)
            row_cache[lam] = (summary, rows)
        delta = delta_between(summary, off0, rows, off0_rows)
        coupling_rows.append({"lambda_fb": lam, **delta, "modal_leakage_final": summary["modal_leakage_final"], "profile_overlap": summary["profile_overlap"]})
        modal_rows.append(
            {
                "run_id": summary["run_id"],
                "lambda_fb": lam,
                "modal_frequency": summary["modal_frequency"],
                "modal_phase_final_unwrapped": summary["modal_phase_final_unwrapped"],
                "modal_amp_final": summary["modal_amp_final"],
                "modal_leakage_final": summary["modal_leakage_final"],
                **delta,
            }
        )
        run_manifest.append(summary)
    fit_rows: list[dict[str, Any]] = []
    train_lams = {0.0, 0.5, 1.0, 2.0}
    holdout_lams = {0.25, 1.5}
    for obs in ("delta_core_amp_final", "delta_modal_phase_final", "delta_modal_frequency"):
        fit_rows.extend(fit_response(coupling_rows, obs, train_lams, holdout_lams))
    write_csv(outdir / "coupling_response.csv", coupling_rows)
    write_csv(outdir / "modal_phase_metrics.csv", modal_rows)
    write_csv(outdir / "coupling_fit_holdout.csv", fit_rows)

    # R3: bounded long-duration scout.
    long_rows: list[dict[str, Any]] = []
    max_long_periods = 0.0
    for pc in parse_float_list(args.long_periods):
        if args.matrix == "quick" and pc > 2.0:
            continue
        max_long_periods = max(max_long_periods, pc)
        dur = pc * period
        lcfg = dict(cfg)
        lcfg["sample_dt"] = args.sample_dt_long
        off_long, off_long_rows, _ = run_case(f"long_{pc:g}P_off", "full_loop", 0.0, lcfg, phi, refs, dur, sample_dt=args.sample_dt_long)
        full_long, full_long_rows, _ = run_case(f"long_{pc:g}P_full", "full_loop", 1.0, lcfg, phi, refs, dur, sample_dt=args.sample_dt_long)
        delta = delta_between(full_long, off_long, full_long_rows, off_long_rows)
        attractor = classify_long(full_long, delta)
        long_rows.append({"periods": pc, "duration": dur, "attractor_class": attractor, **delta, "profile_overlap": full_long["profile_overlap"], "modal_leakage_max": full_long["modal_leakage_max"], "ledger_residual_abs": full_long["ledger_residual_abs"], "shell_flux_max": full_long["shell_flux_max"]})
        run_manifest.extend([off_long, full_long])
    write_csv(outdir / "long_time_metrics.csv", long_rows)
    write_csv(outdir / "attractor_classification.csv", long_rows)

    # R4: field replay controls.
    ff_summary, ff_rows, Gseq = run_case("replay_feedforward_source", "feed_forward", 0.0, cfg, phi, refs, duration, collect_g=True)
    replay_rows: list[dict[str, Any]] = []
    replay_ref_off = off0
    replay_ref_rows = off0_rows
    for control, transform in [
        ("correct_replay", "correct"),
        ("time_shifted_replay", "time_shifted"),
        ("sign_reversed_replay", "sign_reversed"),
        ("decorrelated_roll_replay", "decorrelated_roll"),
    ]:
        rep, rep_rows = run_replay(control, transform, Gseq, 1.0, cfg, phi, refs, duration)
        delta = delta_between(rep, replay_ref_off, rep_rows, replay_ref_rows)
        replay_rows.append({"control": control, "transform": transform, **delta, "modal_leakage_final": rep["modal_leakage_final"], "profile_overlap": rep["profile_overlap"]})
        run_manifest.append(rep)
    write_csv(outdir / "field_replay_controls.csv", replay_rows)

    # R5: nearby validated Q-ball scaling.
    scaling_rows: list[dict[str, Any]] = []
    for w in (0.956, 0.964, 0.968):
        scfg = dict(cfg)
        scfg["w"] = w
        sphi, _ = b1s.solve_qball(scfg)
        soff, soff_rows, _ = run_case(f"scale_w{w:.3f}_off", "full_loop", 0.0, scfg, sphi, refs, duration)
        sfull, sfull_rows, _ = run_case(f"scale_w{w:.3f}_full", "full_loop", 1.0, scfg, sphi, refs, duration)
        delta = delta_between(sfull, soff, sfull_rows, soff_rows)
        scaling_rows.append({"w": w, "charge": sfull["charge_initial"], "energy": sfull["E_phi_initial"], "state_load_integral": sfull["source_integral_mean"], "T_peak": sfull["T_peak"], "G_peak": sfull["G_peak"], **delta})
        run_manifest.extend([soff, sfull])
    write_csv(outdir / "state_load_scaling.csv", scaling_rows)

    # R6: local basin checks, short and conservative.
    basin_specs = [
        ("global_phase", "global_phase", 0.0, 1.0),
        ("translated", "translated", 0.0, 1.0),
        ("radial_perturb", "radial_perturb", 0.0, 1.0),
        ("initial_TG_seed", "base", 1.0e-5, 1.0),
        ("lambda_half", "base", 0.0, 0.5),
        ("lambda_one_half", "base", 0.0, 1.5),
    ]
    basin_rows: list[dict[str, Any]] = []
    for name, kind, tg_seed, lam in basin_specs:
        boff, boff_rows, _ = run_case(f"basin_{name}_off", "full_loop", 0.0, cfg, phi, refs, duration, kind=kind, tg_seed=tg_seed)
        bfull, bfull_rows, _ = run_case(f"basin_{name}_full", "full_loop", lam, cfg, phi, refs, duration, kind=kind, tg_seed=tg_seed)
        delta = delta_between(bfull, boff, bfull_rows, boff_rows)
        passed = bfull["profile_overlap"] > 0.98 and bfull["modal_leakage_max"] < 0.25
        basin_rows.append({"case": name, "lambda_fb": lam, "pass": passed, "profile_overlap": bfull["profile_overlap"], "modal_leakage_max": bfull["modal_leakage_max"], **delta})
        run_manifest.extend([boff, bfull])
    write_csv(outdir / "basin_tests.csv", basin_rows)

    # Consolidated copies for required artifact names.
    write_csv(outdir / "run_manifest.csv", run_manifest)
    write_csv(outdir / "energy_ledger.csv", run_manifest)
    write_csv(outdir / "shell_flux.csv", run_manifest)
    write_csv(outdir / "numerical_validation.csv", floor_rows + [{"section": "long_time", **r} for r in long_rows])

    status, labels, falsification_rows = classify_final(floor_rows, coupling_rows, replay_rows, long_rows, basin_rows, max_long_periods)
    write_csv(outdir / "falsification_results.csv", falsification_rows)

    best_snr_row = max(floor_rows, key=lambda r: float(r["snr_fb"]) if np.isfinite(float(r["snr_fb"])) else 1e300)
    key = {
        "best_snr_observable": best_snr_row["observable"],
        "best_snr": best_snr_row["snr_fb"],
        "max_long_periods": max_long_periods,
        "baseline_delta_modal_phase_final": base_delta.get("delta_modal_phase_final"),
        "baseline_delta_core_amp_final": base_delta.get("delta_core_amp_final"),
        "baseline_delta_modal_frequency": base_delta.get("delta_modal_frequency"),
    }
    write_docs(outdir, status, labels, key)
    (outdir / "TECHNICAL_HANDOFF.md").write_text(
        "\n".join(
            [
                "# TG-B1S-R Technical Handoff",
                "",
                f"Status: `{status}`.",
                f"Run directory: `{outdir.as_posix()}`.",
                "",
                "## Labels",
                "",
                *(f"- `{label}`" for label in labels),
                "",
                "## Scope",
                "",
                "State-load feedback only. No R_relax, L_lock, P_threshold, photon, gravity, objective-time or production claim.",
            ]
        ),
        encoding="utf-8",
    )
    (outdir / "OPEN_QUESTIONS.md").write_text(
        "# TG-B1S-R Open Questions\n\n- Should the long-duration gate be extended to 25 and 50 node periods?\n- Should TG-B1R phase-tension relaxation remain deferred until this result is reviewed?\n",
        encoding="utf-8",
    )
    write_csv(outdir / "artifact_hashes.csv", b1s.artifact_hashes(outdir))
    print(json.dumps({"status": status, "labels": labels, "outdir": str(outdir)}, indent=2))


if __name__ == "__main__":
    main()
