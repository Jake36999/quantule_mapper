"""TG-B2 two-node DYNAMICAL force confirmation (momentum observable).

The static calc (`gravity_TG_B2_static_force.py`) showed A-well -> attraction. The COM-evolution scout failed because
a masked-COM is contaminated by A-induced internal density RESHAPING (symmetric reshaping moves the COM without the
node translating), which is why its Delta_phi=0 differential even contradicted the static force.

Fix: measure FIELD MOMENTUM, the true conjugate to force. Symmetric internal reshaping imparts ZERO net momentum, so
the half-space momentum P_R,x isolates the actual inter-node push. KG momentum density (phi_t = pi):

    p_x = -2 Re( conj(pi) d_x phi )        (>0 = +x momentum; right-mover e^{i(kx-wt)} gives +2 w k |phi|^2 > 0)
    P_R = Integral_{x>0} p_x dV            dP_R/dt = force on the right node

Loop force isolated by same-geometry full/off:  J(t) = P_R(full) - P_R(off).  Delta_phi = 0 (symmetric, no net
current; the common in-phase transient cancels in J).  Attraction => right node gains -x momentum => P_R more
negative in the loop arm than off => J < 0 (in this convention; calibrated by the OFF arm, whose bare in-phase force
is attractive).  Sign control: run BOTH A-well (+) and A-hill (-); J must reverse.

Reuses the frozen TG-B1S dynamics via `gravity_TG_B2_two_node_awell` (A = exp(a_sign*eps_G*G)); frozen model untouched.
GPU (WSL/JAX). Mirror-only; production closed; no gravity/UFF/IRER claim.
  <wsl python> jax_scout/gravity_TG_B2_dynamical_force.py --N 64 --L 16 --T 12
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from jax_scout import gravity_TG_B1S_state_load_feedback_gpu as b1s  # noqa: E402
from jax_scout import gravity_TG_B2_two_node_awell as b2  # noqa: E402
from jax_scout.phase_d_c3_wave import build_kg  # noqa: E402


@jax.jit
def mom_diag(state, g):
    phi, pi = state[0], state[1]
    dV = g["dx"] ** 3
    dxphi = jnp.fft.ifftn(g["ikx"] * jnp.fft.fftn(phi))
    px = -2.0 * jnp.real(jnp.conj(pi) * dxphi)          # +x momentum density
    X = g["X"]
    rho = jnp.abs(phi) ** 2
    return {"P_R": jnp.sum(jnp.where(X > 0, px, 0.0)) * dV,
            "P_L": jnp.sum(jnp.where(X < 0, px, 0.0)) * dV,
            "P_tot": jnp.sum(px) * dV,
            "massR": jnp.sum(jnp.where(X > 0, rho, 0.0)) * dV,
            "massL": jnp.sum(jnp.where(X < 0, rho, 0.0)) * dV,
            "amp": jnp.max(jnp.abs(phi))}


def run_arm(psi, pi, cfgv, refsv, g, flags, a_sign, nchunk, chunk_steps, dt):
    state = b2.to_dev(psi, pi, g)
    traj = []
    d0 = mom_diag(state, g)
    traj.append({"t": 0.0, **{k: float(v) for k, v in d0.items()}})
    t = 0.0
    for _ in range(nchunk):
        state = b2.evolve_2n(state, cfgv, refsv, g, jnp.asarray(flags, dtype=jnp.float64),
                             jnp.asarray(a_sign, dtype=jnp.float64), chunk_steps)
        t += chunk_steps * dt
        d = mom_diag(state, g)
        traj.append({"t": t, **{k: float(v) for k, v in d.items()}})
    return traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--L", type=float, default=16.0)
    ap.add_argument("--sep", type=float, default=3.5)
    ap.add_argument("--T", type=float, default=12.0)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--sample-dt", type=float, default=0.5)
    for k, v in dict(c=0.5477, m=1.0, a=0.8, s=-0.5, f=-0.1, w=0.964, alpha_T=0.35, omega_T=1.25, omega_G=0.85,
                     gamma_T=0.08, gamma_G=0.06, kappa_TG=0.55, epsilon_G=0.06, cT=0.7, cG=0.55,
                     absorb_width=1.6, absorb_strength=0.02, core_radius=2.0).items():
        ap.add_argument(f"--{k.replace('_','-')}", type=float, default=v)
    args = ap.parse_args()

    out = Path(args.out) if args.out else ROOT / "sweep_runs" / f"TG_B2_DYNAMICAL_FORCE_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    pf = b2.preflight()
    print(f"=== TG-B2 dynamical force (momentum) | {pf['backend']} {pf['devices']} x64={pf['x64']} | out={out} ===", flush=True)
    if not pf["gpu_ok"]:
        json.dump({"status": "GPU_PREFLIGHT_FAILED", **pf}, open(out / "summary.json", "w"), indent=2)
        print("GPU_PREFLIGHT_FAILED", flush=True); return

    cfg = vars(args).copy()
    phi, prof = b1s.solve_qball(cfg)
    op = build_kg(cfg["N"], cfg["L"], cfg["c"], cfg["m"], cfg["dt"])
    refs = b1s.ref_source_norms(phi, cfg, op)
    g = b1s.make_grid(op)
    cfgv = b1s.cfg_array(cfg); refsv = b1s.refs_array(refs)
    psi, pi, sc = b2.place_two(phi, cfg, args.sep, 0.0)   # dphi = 0 (symmetric, no net current)
    chunk_steps = max(1, int(round(args.sample_dt / args.dt)))
    nchunk = max(4, int(round(args.T / args.dt / chunk_steps)))
    print(f"[setup] resid={prof['residual']:.1e} sep={args.sep} (cells {sc}) dphi=0 T={args.T} samples={nchunk}", flush=True)

    FULL = [1.0, 1.0, 1.0, 1.0]; OFF = [1.0, 1.0, 1.0, 0.0]
    arms = {}
    print("[run] OFF (bare) ...", flush=True)
    arms["off"] = run_arm(psi, pi, cfgv, refsv, g, OFF, -1.0, nchunk, chunk_steps, args.dt)
    print("[run] WELL full (a_sign=+1) ...", flush=True)
    arms["well"] = run_arm(psi, pi, cfgv, refsv, g, FULL, +1.0, nchunk, chunk_steps, args.dt)
    print("[run] HILL full (a_sign=-1) ...", flush=True)
    arms["hill"] = run_arm(psi, pi, cfgv, refsv, g, FULL, -1.0, nchunk, chunk_steps, args.dt)

    t = np.array([r["t"] for r in arms["off"]])
    PR = {k: np.array([r["P_R"] for r in v]) for k, v in arms.items()}
    Ptot = {k: np.array([r["P_tot"] for r in v]) for k, v in arms.items()}
    massbal = {k: float(np.max(np.abs([r["massR"] - r["massL"] for r in v]))) for k, v in arms.items()}
    # loop impulse differentials (P_R already ~0 at t=0)
    J_well = PR["well"] - PR["off"]
    J_hill = PR["hill"] - PR["off"]
    # inward direction calibration: OFF bare in-phase force is attractive -> P_R,off drifts inward (sign = inward)
    inward_sign = float(np.sign(PR["off"][-1])) if abs(PR["off"][-1]) > 0 else -1.0
    Jw_end, Jh_end = float(J_well[-1]), float(J_hill[-1])
    well_inward = (Jw_end * inward_sign) > 0        # J_well same direction as bare inward drift
    sign_reverses = (Jw_end * Jh_end) < 0
    Ptot_max = max(float(np.max(np.abs(v))) for v in Ptot.values())

    for k in ("off", "well", "hill"):
        print(f"[{k:4s}] P_R(0)={PR[k][0]:+.3e} P_R(T)={PR[k][-1]:+.3e}  massImbal_max={massbal[k]:.2e}", flush=True)
    print(f"\n[inward] bare OFF P_R(T)={PR['off'][-1]:+.3e} -> inward sign = {inward_sign:+.0f}", flush=True)
    print(f"[J] loop impulse well: J(T)={Jw_end:+.4e}  ({'INWARD/attract' if well_inward else 'OUTWARD/repel'})", flush=True)
    print(f"[J] loop impulse hill: J(T)={Jh_end:+.4e}", flush=True)
    print(f"[control] sign-flip reverses: {sign_reverses};  P_tot conservation max={Ptot_max:.2e}", flush=True)

    if well_inward and sign_reverses:
        verdict = "TG_B2_DYNAMICAL_AWELL_ATTRACTION_CONFIRMED"
    elif (not well_inward) and sign_reverses:
        verdict = "TG_B2_DYNAMICAL_AWELL_REPULSION"
    else:
        verdict = "TG_B2_DYNAMICAL_UNCLEAR_SEE_TRAJECTORIES"

    summary = {"verdict": verdict, "sep": args.sep, "dphi": 0.0, "config": cfg, "preflight": pf,
               "P_R_off_end": float(PR["off"][-1]), "inward_sign": inward_sign,
               "J_well_end": Jw_end, "J_hill_end": Jh_end,
               "well_inward": bool(well_inward), "sign_flip_reverses": bool(sign_reverses),
               "P_total_conservation_max": Ptot_max, "mass_imbalance_max": massbal,
               "note": "J = P_R(full)-P_R(off); inward calibrated from bare OFF drift. Momentum observable (not COM).",
               "caveats": ["short-range Yukawa -> NOT gravity/1-over-r^2 (long-range = LR-1)",
                           "weak effect (eps_G=0.06); near-field separations",
                           "half-space momentum: dP_R/dt = body force + midplane stress flux; differential isolates loop"],
               "boundary": "mirror-only dynamical two-node force; frozen TG-B1S untouched; no gravity/UFF/IRER claim"}
    json.dump(summary, open(out / "summary.json", "w"), indent=2, default=float)
    json.dump({k: v for k, v in arms.items()}, open(out / "trajectories.json", "w"), indent=2, default=float)
    print(f"\n=== {verdict} ===", flush=True)


if __name__ == "__main__":
    main()
