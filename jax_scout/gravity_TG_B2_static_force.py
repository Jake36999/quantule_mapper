"""TG-B2 two-node STATIC inter-node force (A-well vs A-hill) — clean instrument.

Supersedes the COM-evolution scout (`gravity_TG_B2_two_node_awell.py`), which failed: the COM-separation observable
is dominated by phase-dependent transients (momentum-current sloshing at dphi=pi/2; in-phase tail interference at
dphi=0) and gave opposite signs at the two phases. That is an observable failure, not a physical force.

This computes the inter-node force the same transient-free, analytic way FC-1 computed the frequency shift:
  1. two in-phase Q-balls at x = +-q  (dphi=0: symmetric, no momentum current);
  2. two-node S_state -> static screened T/G (exact Fourier solve, as FC-1/box-dependence);
  3. A_well = exp(+eps_G G)  (theory-faithful),  A_hill = exp(-eps_G G)  (frozen sign, control);
  4. force on the RIGHT node via the VERIFIED Gravity-D momentum law (force-contract residual 3.6e-17):
         F_R,x = - c^2 * Integral_{x>0} (d_x A) |grad phi|^2 dV
     F_R,x < 0  => right node pushed toward center => ATTRACTION;  > 0 => REPULSION.
  5. sweep half-separation q -> short-range (Yukawa) falloff.

Self-force cancels by each node's symmetry; F_R is the inter-node (cross) force. Bare control A=1 -> F_R=0.
CPU-only; read-only; frozen TG-B1S untouched; no gravity/UFF/IRER claim.
  python jax_scout/gravity_TG_B2_static_force.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from jax_scout.phase_d_c3_wave import build_kg, qball_petviashvili  # noqa: E402

BASE = dict(N=64, L=16.0, c=0.5477, m=1.0, a=0.8, s=-0.5, f=-0.1, w=0.964, dt=0.002,
            alpha_T=0.35, omega_T=1.25, omega_G=0.85, gamma_T=0.08, gamma_G=0.06,
            kappa_TG=0.55, epsilon_G=0.06, cT=0.7, cG=0.55, core_radius=2.0)
S0 = 135.6862187684289
SEPARATIONS = [2.5, 3.0, 3.5, 4.0, 5.0]   # center-to-center


def solve_single(N, L, w):
    op = build_kg(N, L, BASE["c"], BASE["m"], BASE["dt"])
    mu = BASE["m"] ** 2 - w ** 2
    for sig in (1.5, 1.2, 1.8, 2.0):
        phi, prof = qball_petviashvili(op, BASE["a"], BASE["s"], BASE["f"], mu, sig=sig)
        if phi is not None and prof["residual"] < 1e-6 and prof["occ"] < 0.5:
            return np.asarray(phi, dtype=np.complex128), prof
    raise RuntimeError("Q-ball solve failed")


def grids(N, L):
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    k1 = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing="ij")
    return X, Y, Z, KX, KY, KZ, (L / N) ** 3


def dx_spec(f, KX):
    return np.real(np.fft.ifftn(1j * KX * np.fft.fftn(f)))


def grad2(phi, KX, KY, KZ):
    pk = np.fft.fftn(phi)
    gx = np.fft.ifftn(1j * KX * pk); gy = np.fft.ifftn(1j * KY * pk); gz = np.fft.ifftn(1j * KZ * pk)
    return np.abs(gx) ** 2 + np.abs(gy) ** 2 + np.abs(gz) ** 2


def s_state(phi, KX, KY, KZ):
    w = BASE["w"]; pi = -1j * w * phi
    rho = np.abs(phi) ** 2; g2 = grad2(phi, KX, KY, KZ)
    Gpot = BASE["a"] * rho ** 2 / 2 + BASE["s"] * rho ** 3 / 3 + BASE["f"] * rho ** 4 / 4
    energy = np.abs(pi) ** 2 + BASE["c"] ** 2 * g2 + BASE["m"] ** 2 * rho - Gpot
    charge = np.abs(np.imag(np.conj(phi) * pi))
    e_ref = float(np.max(np.abs(energy))); q_ref = float(np.max(charge))
    raw = 0.5 * np.maximum(energy, 0.0) / (e_ref + 1e-12) + 0.5 * charge / (q_ref + 1e-12)
    return raw / S0


def static_G(S, KX, KY, KZ):
    k2 = KX ** 2 + KY ** 2 + KZ ** 2
    Sk = np.fft.fftn(S)
    dT = BASE["cT"] ** 2 * k2 + BASE["omega_T"] ** 2
    dG = BASE["cG"] ** 2 * k2 + BASE["omega_G"] ** 2
    D = dT * dG - BASE["kappa_TG"] ** 2
    Gk = -BASE["kappa_TG"] * BASE["alpha_T"] * Sk / D
    return np.real(np.fft.ifftn(Gk))


def force_on_right(A, g2, X, KX, dV, c):
    """F_R,x = -c^2 Int_{x>0} (d_x A) |grad phi|^2 dV. <0 => toward center => attraction."""
    dxA = dx_spec(A, KX)
    integrand = dxA * g2
    return -c ** 2 * float(np.sum(np.where(X > 0, integrand, 0.0)) * dV)


def main():
    out = ROOT / "sweep_runs" / f"TG_B2_STATIC_FORCE_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"=== TG-B2 static two-node force | CPU-only | out={out} ===", flush=True)

    N, L, c, epsG = BASE["N"], BASE["L"], BASE["c"], BASE["epsilon_G"]
    phi, prof = solve_single(N, L, BASE["w"])
    X, Y, Z, KX, KY, KZ, dV = grids(N, L)
    print(f"[qball] resid={prof['residual']:.1e} amp={prof['amp']:.3f}", flush=True)

    rows = []
    for sep in SEPARATIONS:
        sc = int(round((sep / 2.0 / L) * N))
        phi2 = (np.roll(phi, -sc, axis=0) + np.roll(phi, +sc, axis=0)).astype(np.complex128)  # dphi=0
        S = s_state(phi2, KX, KY, KZ)
        G = static_G(S, KX, KY, KZ)
        g2 = grad2(phi2, KX, KY, KZ)
        A_well = np.exp(+epsG * G)
        A_hill = np.exp(-epsG * G)
        F_well = force_on_right(A_well, g2, X, KX, dV, c)
        F_hill = force_on_right(A_hill, g2, X, KX, dV, c)
        F_bare = force_on_right(np.ones_like(G), g2, X, KX, dV, c)   # must be ~0
        G_mid = float(np.min(G))
        row = {"sep": sep, "sep_cells": sc, "F_R_well": F_well, "F_R_hill": F_hill, "F_R_bare": F_bare,
               "G_min": G_mid, "A_well_min": float(np.min(A_well)), "A_hill_max": float(np.max(A_hill))}
        rows.append(row)
        rd = "ATTRACT" if F_well < 0 else "REPEL"
        print(f"[sep={sep:.1f} cells={sc}] F_R_well={F_well:+.4e} ({rd})  F_R_hill={F_hill:+.4e}  "
              f"F_bare={F_bare:+.2e}  G_min={G_mid:+.3e}", flush=True)

    # verdict from the closest well-resolved separation (sep=3.5, mid of range)
    ref = next(r for r in rows if abs(r["sep"] - 3.5) < 1e-9)
    well_attracts = ref["F_R_well"] < 0
    sign_reverses = ref["F_R_well"] * ref["F_R_hill"] < 0
    bare_clean = all(abs(r["F_R_bare"]) < 0.05 * max(abs(r["F_R_well"]), 1e-30) for r in rows)
    # falloff: is |F_well| decreasing with separation (short-range)?
    fw = [abs(r["F_R_well"]) for r in rows]
    monotone_falloff = all(fw[i] >= fw[i + 1] for i in range(len(fw) - 1))
    if well_attracts and sign_reverses:
        verdict = "TG_B2_STATIC_AWELL_ATTRACTION_SUPPORTED"
    elif (not well_attracts) and sign_reverses:
        verdict = "TG_B2_STATIC_AWELL_REPULSION"
    else:
        verdict = "TG_B2_STATIC_FORCE_UNCLEAR"

    summary = {"verdict": verdict, "well_attracts_at_3.5": bool(well_attracts),
               "sign_flip_reverses": bool(sign_reverses), "bare_force_negligible": bool(bare_clean),
               "short_range_monotone_falloff": bool(monotone_falloff),
               "rows": rows, "config": BASE, "S0": S0,
               "method": "F_R = -c^2 Int_{x>0} d_x A |grad phi|^2 dV (verified Gravity-D law); dphi=0 in-phase pair",
               "caveats": ["quasi-static T/G (Fourier screened solve; absorber neglected; steady-state)",
                           "first order in (A-1); force-law form from single-probe Gravity-D applied to the pair",
                           "short-range Yukawa mediation -> NOT gravity/1-over-r^2 (long-range = LR-1 fork)"],
               "boundary": "mirror-only static two-node force; frozen TG-B1S untouched; no gravity/UFF/IRER claim"}
    json.dump(summary, open(out / "summary.json", "w"), indent=2, default=float)
    print(f"\n[falloff] |F_R_well| vs sep: {[f'{x:.2e}' for x in fw]}  monotone_falloff={monotone_falloff}", flush=True)
    print(f"[control] bare A=1 force negligible: {bare_clean}; sign-flip reverses: {sign_reverses}", flush=True)
    print(f"\n=== {verdict} ===", flush=True)
    print("(quasi-static analytic force via the verified Gravity-D law; no gravity/IRER claim)", flush=True)


if __name__ == "__main__":
    main()
