"""TG-B2 two-node inter-node force scout (A-WELL, theory-faithful sign).

New explicitly-labelled branch, per the SIGN-1 design contract + Jake's ruling (2026-07-15):
the frozen TG-B1S uses A = exp(-eps_G*G) -> A-HILL near a node (dense node = faster propagation), the
*anti*-throttling polarity. IRER throttling (dense -> slower resolution -> lower lapse) wants an A-WELL:

    A_well = exp(+eps_G*G)          (a_sign = +1)   <-- theory-faithful; predicted ATTRACTION
    A_hill = exp(-eps_G*G)          (a_sign = -1)   <-- frozen TG-B1S; predicted REPULSION  (sign control)

Only the G->A polarity is flipped; the validated S_state -> T -> G sector is reused UNCHANGED from
`gravity_TG_B1S_state_load_feedback_gpu` (imported helpers). Force law (Gravity-D, verified):
d<P>/dt = -c^2 int grad(A)|grad phi|^2 dV  -> a node is pushed toward SMALLER A.

Experiment: two Q-balls at x = +-q, relative phase Delta_phi (default pi/2, where the bare KS/Gordon force ~ 0).
Isolate the loop force as F_TG = F_full - F_off (off = feedback_enabled=0 -> A=1 -> bare KG two-body force).
Observable: node separation d(t) = x_R - x_L (on-device COM), started from rest; initial curvature a = d''(0).
  a_full < a_off (separation shrinks faster with the loop) -> ATTRACTION.
Sign control: run BOTH a_sign=+1 (well) and -1 (hill); F_TG must REVERSE.

GPU (WSL/JAX). Mirror-only; production closed; frozen TG-B1S untouched; no gravity/UFF/IRER claim.
  <wsl python> jax_scout/gravity_TG_B2_two_node_awell.py --N 64 --L 16 --sep 3.5 --T 8 --dphi 1.5707963
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

from jax_scout import gravity_TG_B1S_state_load_feedback_gpu as b1s  # noqa: E402
from jax_scout.phase_d_c3_wave import build_kg  # noqa: E402


def preflight():
    backend = jax.default_backend()
    devs = jax.devices()
    ok = backend == "gpu" and any(d.platform == "gpu" for d in devs)
    return {"backend": backend, "devices": [str(d) for d in devs], "gpu_ok": bool(ok),
            "x64": bool(jax.config.jax_enable_x64)}


@partial(jax.jit, static_argnames=())
def rhs_2n(state, cfg, refs, g, flags, a_sign):
    """TG-B1S dynamics with the G->A polarity parametrized by a_sign (+1 well, -1 hill=frozen)."""
    phi, pi, T, VT, G, VG = state
    src_en, temp_en, geom_en, fb_en = flags
    c, m, a, s, f = cfg[1], cfg[2], cfg[3], cfg[4], cfg[5]
    alpha_T, omega_T, omega_G = cfg[6], cfg[7], cfg[8]
    gamma_T, gamma_G, kappa, eps_G, cT, cG = cfg[9], cfg[10], cfg[11], cfg[12], cfg[13], cfg[14]
    L, abs_w, abs_s = cfg[15], cfg[16], cfg[17]
    rho = jnp.abs(phi) ** 2
    source = b1s.state_load(phi, pi, cfg, refs, g) * src_en * temp_en
    absorb = b1s.absorb_profile(g, L, abs_w, abs_s)
    A = jnp.exp(a_sign * eps_G * G * geom_en * fb_en)          # <-- ONLY change vs frozen b1s.rhs
    kg_force = c * c * b1s.div_A_grad(phi, A, g) - m * m * phi + (a * rho + s * rho ** 2 + f * rho ** 3) * phi
    T_t = VT * temp_en
    VT_t = (cT * cT * b1s.lap_real(T, g) - omega_T * omega_T * T - gamma_T * VT
            + alpha_T * source - kappa * G * geom_en - absorb * VT) * temp_en
    G_t = VG * geom_en
    VG_t = (cG * cG * b1s.lap_real(G, g) - omega_G * omega_G * G - gamma_G * VG
            - kappa * T - absorb * VG) * geom_en
    return pi, kg_force, T_t, VT_t, G_t, VG_t


@partial(jax.jit, static_argnames=())
def rk4_2n(state, cfg, refs, g, flags, a_sign):
    dt = cfg[0]
    k1 = rhs_2n(state, cfg, refs, g, flags, a_sign)
    s2 = tuple(y + 0.5 * dt * dy for y, dy in zip(state, k1))
    k2 = rhs_2n(s2, cfg, refs, g, flags, a_sign)
    s3 = tuple(y + 0.5 * dt * dy for y, dy in zip(state, k2))
    k3 = rhs_2n(s3, cfg, refs, g, flags, a_sign)
    s4 = tuple(y + dt * dy for y, dy in zip(state, k3))
    k4 = rhs_2n(s4, cfg, refs, g, flags, a_sign)
    return tuple(y + (dt / 6.0) * (p + 2 * q + 2 * r + w) for y, p, q, r, w in zip(state, k1, k2, k3, k4))


@partial(jax.jit, static_argnames=("nsteps",))
def evolve_2n(state, cfg, refs, g, flags, a_sign, nsteps):
    return lax.fori_loop(0, nsteps, lambda _, sst: rk4_2n(sst, cfg, refs, g, flags, a_sign), state)


@jax.jit
def node_diag(state, g):
    """On-device per-node COM + distinctness scalars (masks split the box at x=0)."""
    phi = state[0]
    dV = g["dx"] ** 3
    rho = jnp.abs(phi) ** 2
    X = g["X"]
    rp = jnp.where(X > 0, rho, 0.0)
    rl = jnp.where(X < 0, rho, 0.0)
    mp = jnp.sum(rp) * dV + 1e-30
    ml = jnp.sum(rl) * dV + 1e-30
    x_R = jnp.sum(X * rp) * dV / mp
    x_L = jnp.sum(X * rl) * dV / ml
    return {"x_R": x_R, "x_L": x_L, "sep": x_R - x_L, "mass_R": mp, "mass_L": ml,
            "mass_total": jnp.sum(rho) * dV, "amp_max": jnp.max(jnp.abs(phi)),
            "G_min": jnp.min(state[4]), "G_max": jnp.max(state[4]), "T_peak": jnp.max(jnp.abs(state[2]))}


def place_two(phi_np, cfg, sep, dphi):
    """Two Q-balls at x = +-sep/2, relative phase dphi (right node carries +dphi)."""
    sc = int(round((sep / 2.0 / cfg["L"]) * cfg["N"]))
    left = np.roll(phi_np, -sc, axis=0)
    right = np.roll(phi_np, +sc, axis=0) * np.exp(1j * dphi)
    psi = (left + right).astype(np.complex128)
    pi = (-1j * cfg["w"] * psi).astype(np.complex128)
    return psi, pi, sc


def to_dev(psi, pi, g):
    z = jnp.zeros_like(jnp.asarray(psi.real))
    return (jnp.asarray(psi), jnp.asarray(pi), z, z, z, z)


def run_arm(name, psi, pi, cfgv, refsv, g, flags, a_sign, nchunk, chunk_steps, sample_dt):
    state = to_dev(psi, pi, g)
    traj = []
    d0 = node_diag(state, g)
    t = 0.0
    traj.append({"t": t, **{k: float(v) for k, v in d0.items()}})
    for _ in range(nchunk):
        state = evolve_2n(state, cfgv, refsv, g, jnp.asarray(flags, dtype=jnp.float64),
                          jnp.asarray(a_sign, dtype=jnp.float64), chunk_steps)
        t += chunk_steps * float(cfgv[0])
        d = node_diag(state, g)
        traj.append({"t": t, **{k: float(v) for k, v in d.items()}})
    return traj


def fit_curvature(traj):
    """d(t)-d0 ~ v0 t + 0.5 a t^2 (started from rest, v0~0); return a and the early/late split."""
    t = np.array([r["t"] for r in traj])
    d = np.array([r["sep"] for r in traj])
    d0 = d[0]
    # quadratic fit through the window
    coef = np.polyfit(t, d - d0, 2)   # coef[0] = 0.5 a
    a = 2.0 * coef[0]
    v0 = coef[1]
    return {"a_curv": float(a), "v0": float(v0), "d0": float(d0), "d_final": float(d[-1]),
            "sep_min": float(np.min(d)), "sep_max": float(np.max(d))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--L", type=float, default=16.0)
    ap.add_argument("--sep", type=float, default=3.5)      # center-to-center separation
    ap.add_argument("--dphi", type=float, default=np.pi / 2)
    ap.add_argument("--T", type=float, default=8.0)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--sample-dt", type=float, default=0.4)
    # frozen TG-B1S physics defaults
    ap.add_argument("--c", type=float, default=0.5477); ap.add_argument("--m", type=float, default=1.0)
    ap.add_argument("--a", type=float, default=0.8); ap.add_argument("--s", type=float, default=-0.5)
    ap.add_argument("--f", type=float, default=-0.1); ap.add_argument("--w", type=float, default=0.964)
    ap.add_argument("--alpha-T", type=float, default=0.35); ap.add_argument("--omega-T", type=float, default=1.25)
    ap.add_argument("--omega-G", type=float, default=0.85); ap.add_argument("--gamma-T", type=float, default=0.08)
    ap.add_argument("--gamma-G", type=float, default=0.06); ap.add_argument("--kappa-TG", type=float, default=0.55)
    ap.add_argument("--epsilon-G", type=float, default=0.06); ap.add_argument("--cT", type=float, default=0.7)
    ap.add_argument("--cG", type=float, default=0.55); ap.add_argument("--absorb-width", type=float, default=1.6)
    ap.add_argument("--absorb-strength", type=float, default=0.02); ap.add_argument("--core-radius", type=float, default=2.0)
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else ROOT / "sweep_runs" / f"TG_B2_TWO_NODE_AWELL_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    pf = preflight()
    print(f"=== TG-B2 two-node A-well scout | {pf['backend']} {pf['devices']} x64={pf['x64']} | out={out} ===", flush=True)
    if not pf["gpu_ok"]:
        print("WARNING: not on GPU backend; refusing heavy run.", flush=True)
        json.dump({"status": "GPU_PREFLIGHT_FAILED", **pf}, open(out / "summary.json", "w"), indent=2)
        return

    cfg = vars(args).copy()
    cfg["kappa_TG"] = args.kappa_TG; cfg["epsilon_G"] = args.epsilon_G
    phi, prof = b1s.solve_qball(cfg)
    op = build_kg(cfg["N"], cfg["L"], cfg["c"], cfg["m"], cfg["dt"])
    refs = b1s.ref_source_norms(phi, cfg, op)
    g = b1s.make_grid(op)
    cfgv = b1s.cfg_array(cfg)
    refsv = b1s.refs_array(refs)
    psi, pi, sc = place_two(phi, cfg, args.sep, args.dphi)
    print(f"[setup] qball resid={prof['residual']:.1e} amp={prof['amp']:.3f}; sep={args.sep} (cells {sc}) "
          f"dphi={args.dphi:.4f}; S0={refs['source_global_norm_S0']}", flush=True)

    chunk_steps = max(1, int(round(args.sample_dt / args.dt)))
    nchunk = max(2, int(round(args.T / args.dt / chunk_steps)))
    FULL = [1.0, 1.0, 1.0, 1.0]
    OFF = [1.0, 1.0, 1.0, 0.0]   # feedback off -> A=1 -> bare KG two-body force (same for both signs)

    arms = {}
    print("[run] OFF (bare force, feedback_enabled=0) ...", flush=True)
    arms["off"] = run_arm("off", psi, pi, cfgv, refsv, g, OFF, -1.0, nchunk, chunk_steps, args.sample_dt)
    print("[run] WELL full (a_sign=+1, theory-faithful) ...", flush=True)
    arms["well_full"] = run_arm("well_full", psi, pi, cfgv, refsv, g, FULL, +1.0, nchunk, chunk_steps, args.sample_dt)
    print("[run] HILL full (a_sign=-1, frozen sign, control) ...", flush=True)
    arms["hill_full"] = run_arm("hill_full", psi, pi, cfgv, refsv, g, FULL, -1.0, nchunk, chunk_steps, args.sample_dt)

    fits = {k: fit_curvature(v) for k, v in arms.items()}
    a_off = fits["off"]["a_curv"]
    a_well = fits["well_full"]["a_curv"]
    a_hill = fits["hill_full"]["a_curv"]
    F_TG_well = a_well - a_off
    F_TG_hill = a_hill - a_off
    for k in ("off", "well_full", "hill_full"):
        print(f"[fit] {k:10s}: a_curv={fits[k]['a_curv']:+.4e}  v0={fits[k]['v0']:+.3e}  "
              f"d0={fits[k]['d0']:.3f} d_final={fits[k]['d_final']:.3f} sep_min={fits[k]['sep_min']:.3f}", flush=True)
    print(f"\n[F_TG] well (a_sign=+1): a_full-a_off = {F_TG_well:+.4e}   ({'INWARD/attract' if F_TG_well<0 else 'OUTWARD/repel'})", flush=True)
    print(f"[F_TG] hill (a_sign=-1): a_full-a_off = {F_TG_hill:+.4e}   ({'INWARD/attract' if F_TG_hill<0 else 'OUTWARD/repel'})", flush=True)
    sign_reverses = (F_TG_well * F_TG_hill) < 0
    well_attracts = F_TG_well < 0
    print(f"[control] sign-flip reverses force: {sign_reverses}", flush=True)

    if sign_reverses and well_attracts:
        verdict = "TG_B2_AWELL_ATTRACTION_SCOUT_POSITIVE"
    elif sign_reverses and not well_attracts:
        verdict = "TG_B2_AWELL_REPULSION_HILL_ATTRACTION_UNEXPECTED"
    elif abs(F_TG_well) < 0.2 * abs(a_off + 1e-30) and abs(F_TG_well) < 1e-6:
        verdict = "TG_B2_LOOP_FORCE_NULL_OR_BELOW_FLOOR"
    else:
        verdict = "TG_B2_INCONCLUSIVE_SEE_TRAJECTORIES"

    summary = {"verdict": verdict, "preflight": pf, "config": cfg,
               "qball_residual": float(prof["residual"]), "sep_cells": sc, "S0": refs["source_global_norm_S0"],
               "a_off": a_off, "a_well_full": a_well, "a_hill_full": a_hill,
               "F_TG_well": F_TG_well, "F_TG_hill": F_TG_hill,
               "sign_flip_reverses": bool(sign_reverses), "well_attracts": bool(well_attracts),
               "fits": fits,
               "note": "F_TG = a_full - a_off; a<0 = separation shrinking = attraction. off = feedback-off bare KG force.",
               "caveats": ["scout: single separation, single phase, short window, initial-curvature force proxy",
                           "short-range Yukawa mediation -> NOT gravity/1-over-r^2; long-range = separate LR-1 fork",
                           "capture/merger check: inspect sep_min vs d0 and mass_R/mass_L balance in trajectories.json"],
               "boundary": "mirror-only two-node force scout; frozen TG-B1S untouched; no gravity/UFF/IRER claim"}
    json.dump(summary, open(out / "summary.json", "w"), indent=2, default=float)
    json.dump(arms, open(out / "trajectories.json", "w"), indent=2, default=float)
    print(f"\n=== {verdict} ===", flush=True)


if __name__ == "__main__":
    main()
