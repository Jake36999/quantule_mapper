"""TG-B2 MC-3 alignment sweep — does the loop-mediated two-node force track relative phase (Payan alignment)?

Recovered-concept MC-3 (CROSSMAP): relative phase Delta_phi between two coherent cores is the "Payan/parallel
alignment" variable; the bare two-body law has a pi/2 crossover (attract below, repel above). Question here: does the
A-well LOOP-mediated force F_R also depend on alignment, or is it alignment-independent? If it tracks Delta_phi, the
"non-universality" of these forces is (candidate) alignment-modulated coupling, not noise — a testable reframe, not a
confirmation.

Method: reuse the FROZEN static-force machinery (gravity_TG_B2_static_force: quasi-static screened T/G + verified
Gravity-D body force), placing the two nodes with relative phase Delta_phi, sweeping Delta_phi in [0, pi]. Report
F_R_well(dphi), F_R_hill(dphi) (sign-flip control), the S_state integral and A-well depth per dphi (how alignment
changes the mediation), and a grad-phi cross-alignment readout. CPU-only; frozen modules imported unchanged; mirror-
only; no gravity/IRER claim.
  python jax_scout/gravity_TG_B2_alignment_sweep.py --sep 3.0
"""
from __future__ import annotations

import argparse
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

from jax_scout import gravity_TG_B2_static_force as sf  # noqa: E402 (frozen registry module, imported unchanged)


def place_two_phase(phi_np, N, L, sep, dphi):
    sc = int(round((sep / 2.0 / L) * N))
    left = np.roll(phi_np, -sc, axis=0)
    right = np.roll(phi_np, +sc, axis=0) * np.exp(1j * dphi)
    return (left + right).astype(np.complex128), sc


def grad_cross_alignment(phi_L, phi_R, KX, KY, KZ, dV):
    """Re<grad phi_L*, grad phi_R> / |..| — a grad-phi alignment readout between the two node fields (in [-1,1])."""
    def g(f):
        pk = np.fft.fftn(f)
        return (np.fft.ifftn(1j * KX * pk), np.fft.ifftn(1j * KY * pk), np.fft.ifftn(1j * KZ * pk))
    gL = g(phi_L); gR = g(phi_R)
    num = np.real(np.sum(sum(np.conj(a) * b for a, b in zip(gL, gR))) * dV)
    nL = np.sqrt(np.real(np.sum(sum(np.abs(a) ** 2 for a in gL)) * dV))
    nR = np.sqrt(np.real(np.sum(sum(np.abs(a) ** 2 for a in gR)) * dV))
    return float(num / (nL * nR + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sep", type=float, default=3.0)
    ap.add_argument("--n-dphi", type=int, default=9)   # 0..pi inclusive
    args = ap.parse_args()

    out = ROOT / "sweep_runs" / f"TG_B2_ALIGNMENT_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"=== TG-B2 MC-3 alignment sweep | CPU | sep={args.sep} | out={out} ===", flush=True)

    N, L, c, epsG = sf.BASE["N"], sf.BASE["L"], sf.BASE["c"], sf.BASE["epsilon_G"]
    phi, prof = sf.solve_single(N, L, sf.BASE["w"])
    X, Y, Z, KX, KY, KZ, dV = sf.grids(N, L)
    print(f"[qball] resid={prof['residual']:.1e}", flush=True)

    dphis = np.linspace(0.0, np.pi, args.n_dphi)
    rows = []
    for dphi in dphis:
        phi2, sc = place_two_phase(phi, N, L, args.sep, dphi)
        S = sf.s_state(phi2, KX, KY, KZ)
        G = sf.static_G(S, KX, KY, KZ)
        g2 = sf.grad2(phi2, KX, KY, KZ)
        A_well = np.exp(+epsG * G); A_hill = np.exp(-epsG * G)
        F_well = sf.force_on_right(A_well, g2, X, KX, dV, c)
        F_hill = sf.force_on_right(A_hill, g2, X, KX, dV, c)
        phi_L = np.roll(phi, -sc, axis=0); phi_R = np.roll(phi, +sc, axis=0) * np.exp(1j * dphi)
        align = grad_cross_alignment(phi_L, phi_R, KX, KY, KZ, dV)
        row = {"dphi": float(dphi), "dphi_over_pi": float(dphi / np.pi),
               "F_R_well": F_well, "F_R_hill": F_hill,
               "S_integral": float(np.sum(S) * dV), "G_min": float(np.min(G)),
               "A_well_min": float(np.min(A_well)), "grad_alignment": align}
        rows.append(row)
        rd = "ATTRACT" if F_well < 0 else "REPEL"
        print(f"[dphi={dphi/np.pi:.3f}pi] F_R_well={F_well:+.4e} ({rd})  F_R_hill={F_hill:+.4e}  "
              f"S_int={row['S_integral']:.3f}  A_well_min={row['A_well_min']:.5f}  grad_align={align:+.3f}", flush=True)

    Fw = np.array([r["F_R_well"] for r in rows])
    # does the loop force track alignment? magnitude variation across dphi, and any sign crossover
    rel_var = float((Fw.max() - Fw.min()) / (abs(Fw).mean() + 1e-30))
    sign_flip = bool(np.any(np.sign(Fw[:-1]) != np.sign(Fw[1:])))
    always_attract = bool(np.all(Fw < 0))
    # correlation of F_R_well with the alignment readout
    al = np.array([r["grad_alignment"] for r in rows])
    corr = float(np.corrcoef(Fw, al)[0, 1]) if np.std(Fw) > 0 and np.std(al) > 0 else float("nan")
    if sign_flip:
        verdict = "TG_B2_ALIGNMENT_LOOP_FORCE_HAS_CROSSOVER"      # loop force changes sign with alignment (MC-3 strong)
    elif rel_var > 0.15:
        verdict = "TG_B2_ALIGNMENT_LOOP_FORCE_MODULATED_NO_CROSSOVER"  # magnitude tracks alignment, no sign change
    else:
        verdict = "TG_B2_ALIGNMENT_LOOP_FORCE_ALIGNMENT_INDEPENDENT"   # ~flat -> loop force is not alignment-governed
    summary = {"verdict": verdict, "sep": args.sep, "rows": rows,
               "F_well_range_rel": rel_var, "sign_flip_across_dphi": sign_flip,
               "always_attract": always_attract, "corr_F_vs_gradalign": corr,
               "note": "MC-3 test: does the A-well LOOP-mediated force track relative phase (Payan alignment)? Candidate reframe, NOT a confirmation; bare two-body pi/2 law is a separate (dynamical) observable.",
               "boundary": "quasi-static; mirror-only; frozen modules; no gravity/IRER claim"}
    json.dump(summary, open(out / "summary.json", "w"), indent=2, default=float)
    print(f"\n[trend] F_well range/mean={rel_var:.3f}  sign_flip={sign_flip}  always_attract={always_attract}  "
          f"corr(F,grad_align)={corr:+.3f}", flush=True)
    print(f"=== {verdict} ===", flush=True)


if __name__ == "__main__":
    main()
