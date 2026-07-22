"""TG-B2 far-field kernel & mass-scaling (Gate 3) — is the near-field M^0/gentle-falloff a near-field artifact?

The P1 charge audit showed the near-field (sep=3) M^0 force is a ~1.7-power near-field-kernel cancellation, NOT a
source-normalization effect, and is therefore SEPARATION-SPECIFIC. Gate 3 (reviews, 2026-07-18): measure the
single-source T/G->A response kernel and convolve it with the measured source/receiver profiles to predict F(sep,M)
with NO free exponent, in a box large enough to reach the far field. The static body force IS that convolution (it is
computed from the actual screened fields), so this is the honest Gate-3 instrument — and static has no breathing noise
floor, which the dynamical far field would hit.

Outputs:
  (K) single-source kernel: G(r), A(r) radial profiles; fitted screening length of the G tail.
  (S) far-field F(sep) sweep in a LARGE box: does the gentle near field steepen into the screened exponential tail?
      local power/exp slopes per interval; surface-gap parameterization h = sep - 2*R_support.
  (M) far-field F(mass) sweep at a far separation: intrinsic mass exponent where the monopole picture applies +
      whether Q_source*Q_receiver now predicts F (i.e. the near-field cancellation is gone).

Reuses frozen `gravity_TG_B2_static_force` (screened T/G solve + verified Gravity-D body force), imported unchanged.
Per-w pi (matches the run). CPU-first (large FFTs are fine); no time evolution; mirror-only; no gravity/IRER claim.
  python jax_scout/gravity_TG_B2_farfield_kernel.py
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

from jax_scout import gravity_TG_B2_static_force as sf  # noqa: E402

BIG = dict(sf.BASE); BIG["N"] = 112; BIG["L"] = 28.0     # dx=0.25, holds sep up to ~10 with margin
SEPS = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
W_FAMILY = [0.945, 0.955, 0.964, 0.972, 0.980]
SEP_FAR = 8.0


def s_state_w(phi, w, KX, KY, KZ):
    """Normalized S_state with the CORRECT per-w pi (sf.s_state hardcodes BASE['w']; audit-matched here)."""
    c, m, a, s, f = BIG["c"], BIG["m"], BIG["a"], BIG["s"], BIG["f"]
    pi = -1j * w * phi
    rho = np.abs(phi) ** 2
    g2 = sf.grad2(phi, KX, KY, KZ)
    Gpot = a * rho ** 2 / 2 + s * rho ** 3 / 3 + f * rho ** 4 / 4
    energy = np.abs(pi) ** 2 + c ** 2 * g2 + m ** 2 * rho - Gpot
    charge = np.abs(np.imag(np.conj(phi) * pi))
    e_ref = float(np.max(np.abs(energy))); q_ref = float(np.max(charge))
    raw = 0.5 * np.maximum(energy, 0.0) / (e_ref + 1e-12) + 0.5 * charge / (q_ref + 1e-12)
    return raw / sf.S0


def support_radius(phi, X, Y, Z, dV, frac=0.99):
    rho = np.abs(phi) ** 2; tot = np.sum(rho) * dV
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2).ravel(); w = (rho * dV).ravel()
    order = np.argsort(r); c = np.cumsum(w[order])
    return float(r[order][np.searchsorted(c, frac * tot)])


def force_at(phi, w, sep, N, L, X, Y, Z, KX, KY, KZ, dV, c, epsG):
    sc = int(round((sep / 2.0 / L) * N))
    phi2 = (np.roll(phi, -sc, axis=0) + np.roll(phi, +sc, axis=0)).astype(np.complex128)  # in-phase
    S = s_state_w(phi2, w, KX, KY, KZ)
    G = sf.static_G(S, KX, KY, KZ)
    g2 = sf.grad2(phi2, KX, KY, KZ)
    A_well = np.exp(+epsG * G)
    F = sf.force_on_right(A_well, g2, X, KX, dV, c)
    return F, float(np.sum(S) * dV), float(np.sum(g2) * dV)   # F, Q_source, Q_recv


def main():
    out = ROOT / "sweep_runs" / f"TG_B2_FARFIELD_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"=== TG-B2 far-field kernel & mass-scaling | CPU | N={BIG['N']} L={BIG['L']} | out={out} ===", flush=True)
    N, L, c, epsG = BIG["N"], BIG["L"], BIG["c"], BIG["epsilon_G"]
    X, Y, Z, KX, KY, KZ, dV = sf.grids(N, L)
    rr = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)

    # baseline node (w=0.964)
    phi0, prof0 = sf.solve_single(N, L, 0.964)
    Rsup = support_radius(phi0, X, Y, Z, dV)
    print(f"[qball w=0.964] resid={prof0['residual']:.1e} support_radius(99%)={Rsup:.2f}", flush=True)

    # (K) single-source kernel: G(r), A(r), screening length
    S1 = s_state_w(phi0, 0.964, KX, KY, KZ)
    G1 = sf.static_G(S1, KX, KY, KZ)
    A1 = np.exp(+epsG * G1)
    r = rr.ravel(); g = np.abs(G1).ravel()
    band = (r > 3.0) & (r < 0.42 * L) & (g > 1e-15)
    lam = float(-1.0 / np.polyfit(r[band], np.log(g[band]), 1)[0]) if band.sum() > 20 else None
    print(f"[kernel] G screening-length fit (tail r=3..{0.42*L:.0f}) lambda={lam}", flush=True)

    # (S) far-field F(sep) at w=0.964
    sep_rows = []
    for sep in SEPS:
        F, Qs, Qr = force_at(phi0, 0.964, sep, N, L, X, Y, Z, KX, KY, KZ, dV, c, epsG)
        h = sep - 2 * Rsup
        sep_rows.append({"sep": sep, "surface_gap_h": h, "F_R": F, "Q_source": Qs, "Q_recv": Qr})
        print(f"[F(sep) sep={sep:.0f} h={h:+.1f}] F_R={F:+.4e}", flush=True)

    # local slopes (power and exp) per interval
    s_arr = np.array([x["sep"] for x in sep_rows]); F_arr = np.abs([x["F_R"] for x in sep_rows])
    intervals = []
    for i in range(len(s_arr) - 1):
        p = np.log(F_arr[i + 1] / F_arr[i]) / np.log(s_arr[i + 1] / s_arr[i])
        lam_i = (s_arr[i] - s_arr[i + 1]) / np.log(F_arr[i + 1] / F_arr[i]) if F_arr[i + 1] != F_arr[i] else np.inf
        intervals.append({"from": float(s_arr[i]), "to": float(s_arr[i + 1]), "power_slope": float(p),
                          "exp_lambda": float(lam_i)})

    # (M) far-field F(mass) at SEP_FAR
    mass_rows = []
    for w in W_FAMILY:
        phi, prof = sf.solve_single(N, L, w)
        Mnode = float(np.sum(np.abs(phi) ** 2) * dV)
        F, Qs, Qr = force_at(phi, w, SEP_FAR, N, L, X, Y, Z, KX, KY, KZ, dV, c, epsG)
        mass_rows.append({"w": w, "mass": Mnode, "F_R": F, "Q_source": Qs, "Q_recv": Qr})
        print(f"[F(mass) w={w} M={Mnode:.2f} @sep={SEP_FAR:.0f}] F_R={F:+.4e} Qs={Qs:.3f} Qr={Qr:.3f}", flush=True)

    def pfit(x, y):
        x = np.asarray(x, float); y = np.abs(np.asarray(y, float)); m = (x > 0) & (y > 0)
        return float(np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0]) if m.sum() >= 3 else float("nan")

    Mv = [x["mass"] for x in mass_rows]
    p_F_far = pfit(Mv, [x["F_R"] for x in mass_rows])
    p_Qs_far = pfit(Mv, [x["Q_source"] for x in mass_rows])
    p_Qr_far = pfit(Mv, [x["Q_recv"] for x in mass_rows])
    # does Q_source*Q_recv predict F at far field? (near field it did NOT: 1.69 vs 0)
    p_pred = p_Qs_far + p_Qr_far

    # sep-falloff summary: near (h<0, overlapping) vs far (h>0, separated) power
    far = [x for x in sep_rows if x["surface_gap_h"] > 0]
    p_far_sep = pfit([x["sep"] for x in far], [x["F_R"] for x in far]) if len(far) >= 3 else float("nan")

    summary = {"box": {"N": N, "L": L}, "support_radius_99": Rsup, "kernel_screening_length": lam,
               "sep_rows": sep_rows, "sep_local_intervals": intervals, "far_field_sep_power": p_far_sep,
               "mass_rows": mass_rows, "SEP_FAR": SEP_FAR,
               "farfield_exponents_vs_M": {"F ~ M^p": p_F_far, "Q_source ~ M^p": p_Qs_far,
                                           "Q_recv ~ M^p": p_Qr_far, "Q_source*Q_recv predicted p": p_pred},
               "note": "Gate-3: static body force = kernel convolution (no free exponent). Compare far-field F~M^p to the near-field M^0 (which the P1 audit showed was a near-field kernel cancellation). If far-field p != 0 and matches Q_source*Q_recv, the monopole picture is recovered.",
               "boundary": "static/mirror-only; frozen modules; no gravity/UFF/IRER claim"}
    json.dump(summary, open(out / "summary.json", "w"), indent=2, default=float)
    print("\n=== far-field intervals (power slope / exp lambda) ===", flush=True)
    for iv in intervals:
        print(f"  sep {iv['from']:.0f}->{iv['to']:.0f}: power={iv['power_slope']:+.2f}  exp_lambda={iv['exp_lambda']:.2f}", flush=True)
    print(f"\n[far-field F(sep) power over h>0]: {p_far_sep:+.2f}", flush=True)
    print(f"[far-field @sep={SEP_FAR:.0f}] F~M^{p_F_far:+.2f}  Qs~M^{p_Qs_far:+.2f}  Qr~M^{p_Qr_far:+.2f}  "
          f"-> Qs*Qr predicts M^{p_pred:+.2f}  (near-field was F~M^0 vs predicted M^1.69)", flush=True)
    verdict = ("TG_B2_FARFIELD_MASS_SCALING_RECOVERED" if abs(p_F_far - p_pred) < 0.5 and abs(p_F_far) > 0.4
               else "TG_B2_FARFIELD_STILL_FLAT_OR_KERNEL_DOMINATED" if abs(p_F_far) < 0.4
               else "TG_B2_FARFIELD_PARTIAL")
    print(f"\n=== {verdict} ===", flush=True)


if __name__ == "__main__":
    main()
