"""TG-B1S box-dependence discriminator (SCREEN-1 + FC-box) for the D4 larger-box failure.

D4 fact: baseline (N48/L10) and grid_refined (N56/L10) PASS with delta_omega ~ -2.16e-6; larger_box (N56/L12)
FAILS with delta_omega = -8.52e-7 (~40% of ref). The ONLY variable unique to the failing row is L (10->12) --
grid_refined already cleared N=56 at L=10. Question: does the LOCAL screened stiffening mechanism (FC-1) predict a
~60% drop when the box grows, or is the measured drop something the local model does NOT contain (fixed-reference
comparability / absorber-coupling / coarse-grid under-resolution)?

Method: recompute the FC-1 analytic fixed-profile shift
    d_omega_fp = c^2 Int (A-1)|grad phi0|^2 dV / (2 w0 Int rho0 dV)
at several geometries, holding ALL physics params + frozen S0 fixed, changing only (N, L):
    baseline        N=48 L=10  (dx 0.2083)   measured PASS
    grid_refined    N=56 L=10  (dx 0.1786)   measured PASS
    larger_box      N=56 L=12  (dx 0.2143)   measured FAIL  <-- reproduce?
    larger_box_fine N=68 L=12  (dx 0.1765)   resolution control (separates L from dx)
    baseline_fine   N=68 L=10  (dx 0.1471)   resolution control at L=10
Discriminator:
  - analytic drops ~60% at L=12  -> the local mechanism IS box-size dependent (physical)
  - analytic ~flat across L       -> measured drop is comparability/absorber/resolution, NOT the local mechanism
Also fit the G(r) Yukawa screening length (SCREEN-1) and compare to the light coupled-mode range.

CPU-only; read-only; no model/label/production change.
  python jax_scout/gravity_TG_B1S_box_dependence.py
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

BASE = dict(c=0.5477, m=1.0, a=0.8, s=-0.5, f=-0.1, w=0.964, dt=0.002,
            alpha_T=0.35, omega_T=1.25, omega_G=0.85, gamma_T=0.08, gamma_G=0.06,
            kappa_TG=0.55, epsilon_G=0.06, cT=0.7, cG=0.55, core_radius=2.0, lambda_fb=1.0)
S0 = 135.6862187684289
MEASURED = {"baseline": -2.162695983559886e-06, "grid_refined": -2.1626959888870264e-06,
            "larger_box": -8.524302969263542e-07}
REF = -2.150936882840006e-06

GEOMETRIES = [  # (name, N, L)
    ("baseline", 48, 10.0),
    ("grid_refined", 56, 10.0),
    ("larger_box", 56, 12.0),
    ("larger_box_fine", 68, 12.0),
    ("baseline_fine", 68, 10.0),
]


def solve_qball(N, L, w):
    op = build_kg(N, L, BASE["c"], BASE["m"], BASE["dt"])
    mu = BASE["m"] ** 2 - w ** 2
    for sig in (1.5, 1.2, 1.8, 2.0):
        phi, prof = qball_petviashvili(op, BASE["a"], BASE["s"], BASE["f"], mu, sig=sig)
        if phi is not None and prof["residual"] < 1e-6 and prof["occ"] < 0.5:
            return np.asarray(phi, dtype=np.complex128), prof
    raise RuntimeError(f"Q-ball solve failed at N={N} L={L} w={w}")


def grids(N, L):
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    k1 = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing="ij")
    R = np.sqrt(X * X + Y * Y + Z * Z)
    return X, Y, Z, R, KX, KY, KZ, (L / N) ** 3


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


def static_TG(S, KX, KY, KZ):
    k2 = KX ** 2 + KY ** 2 + KZ ** 2
    Sk = np.fft.fftn(S)
    dT = BASE["cT"] ** 2 * k2 + BASE["omega_T"] ** 2
    dG = BASE["cG"] ** 2 * k2 + BASE["omega_G"] ** 2
    D = dT * dG - BASE["kappa_TG"] ** 2
    Tk = BASE["alpha_T"] * Sk * dG / D
    Gk = -BASE["kappa_TG"] * BASE["alpha_T"] * Sk / D
    return np.real(np.fft.ifftn(Tk)), np.real(np.fft.ifftn(Gk))


def leg1_shift(N, L):
    """Analytic fixed-profile Rayleigh shift + the integrals that build it, at geometry (N,L)."""
    w0 = BASE["w"]
    phi, prof = solve_qball(N, L, w0)
    X, Y, Z, R, KX, KY, KZ, dV = grids(N, L)
    rho = np.abs(phi) ** 2
    S = s_state(phi, KX, KY, KZ)
    T, G = static_TG(S, KX, KY, KZ)
    A = np.exp(-BASE["epsilon_G"] * BASE["lambda_fb"] * G)
    g2 = grad2(phi, KX, KY, KZ)
    num = BASE["c"] ** 2 * float(np.sum((A - 1.0) * g2) * dV)
    den = 2.0 * w0 * float(np.sum(rho) * dV)
    d_omega = num / den
    return {
        "N": N, "L": L, "dx": L / N, "residual": float(prof["residual"]), "amp": float(prof["amp"]),
        "int_rho": float(np.sum(rho) * dV), "int_grad2": float(np.sum(g2) * dV),
        "int_Am1_grad2": float(np.sum((A - 1.0) * g2) * dV),
        "A_minus_1_max": float(np.max(A - 1.0)), "G_peak": float(np.max(np.abs(G))),
        "d_omega_fp": d_omega, "R": R, "G": G, "dV": dV,
    }


def screening_fit(R, G, L):
    """Fit |G(r)| radial tail to a exp(-r/lambda); return lambda. Tail band avoids the core and the box edge."""
    r = R.ravel(); g = np.abs(G).ravel()
    lo, hi = 1.5, 0.30 * L
    m = (r > lo) & (r < hi) & (g > 1e-14)
    if m.sum() < 20:
        return None
    coef = np.polyfit(r[m], np.log(g[m]), 1)
    return -1.0 / coef[0] if coef[0] < 0 else None


def main():
    out = ROOT / "sweep_runs" / f"TG_B1S_BOX_DEPENDENCE_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"=== TG-B1S box-dependence | CPU-only | out={out} ===", flush=True)

    # light coupled-mode range prediction (G tail dominated by the lighter screened mode)
    wT2, wG2, k = BASE["omega_T"] ** 2, BASE["omega_G"] ** 2, BASE["kappa_TG"]
    # effective screening from the 2x2 mass matrix eigenvalues, scaled by cG (G-sector speed)
    lam_minus = 0.5 * ((wT2 + wG2) - np.sqrt((wT2 - wG2) ** 2 + 4 * k * k))  # lighter eigen-mass^2
    range_light = BASE["cG"] / np.sqrt(max(lam_minus, 1e-30))
    range_naive_G = BASE["cG"] / BASE["omega_G"]
    print(f"[screen] naive c_G/omega_G range={range_naive_G:.4f}; light coupled-mode range~={range_light:.4f}", flush=True)

    rows = []
    base_val = None
    for name, N, L in GEOMETRIES:
        r = leg1_shift(N, L)
        lam = screening_fit(r["R"], r["G"], L)
        r["screening_length_fit"] = lam
        r.pop("R"); r.pop("G"); r.pop("dV")
        r["name"] = name
        r["measured_d_omega"] = MEASURED.get(name)
        rows.append(r)
        if name == "baseline":
            base_val = r["d_omega_fp"]
        print(f"[{name:16s}] N={N} L={L:.0f} dx={L/N:.4f}  resid={r['residual']:.1e}  "
              f"int(A-1)grad2={r['int_Am1_grad2']:.4e}  int_rho={r['int_rho']:.4f}  "
              f"d_omega_fp={r['d_omega_fp']:+.4e}  lambda_screen={lam if lam is None else round(lam,3)}", flush=True)

    # ratios vs baseline (analytic) and the measured drop we are trying to explain
    print("\n[compare] analytic fixed-profile d_omega vs baseline, and vs measured:", flush=True)
    meas_drop = MEASURED["larger_box"] / MEASURED["baseline"]  # ~0.394
    for r in rows:
        ratio = r["d_omega_fp"] / base_val
        r["analytic_ratio_to_baseline"] = ratio
        tag = ""
        if r["name"] == "larger_box":
            tag = f"   <-- measured larger_box/baseline = {meas_drop:.3f}"
        print(f"  {r['name']:16s}: analytic/baseline = {ratio:.3f}{tag}", flush=True)

    lb = next(r for r in rows if r["name"] == "larger_box")
    lbf = next(r for r in rows if r["name"] == "larger_box_fine")
    analytic_lb_ratio = lb["analytic_ratio_to_baseline"]
    # Does the LOCAL mechanism reproduce the ~0.39 measured drop?
    reproduces = abs(analytic_lb_ratio - meas_drop) < 0.15
    # Is the analytic prediction ~flat across L (drop < 15%)?
    flat = abs(1.0 - analytic_lb_ratio) < 0.15
    # dx vs L: does going fine at L=12 restore the baseline value?
    dx_effect = lbf["analytic_ratio_to_baseline"] - lb["analytic_ratio_to_baseline"]

    if reproduces:
        verdict = "BOX_DEPENDENCE_PHYSICAL_LOCAL_MECHANISM_REPRODUCES_DROP"
    elif flat:
        verdict = "BOX_DEPENDENCE_NOT_IN_LOCAL_MECHANISM__MEASURED_DROP_IS_COMPARABILITY_OR_ABSORBER_OR_RESOLUTION"
    else:
        verdict = "BOX_DEPENDENCE_PARTIAL__LOCAL_MECHANISM_EXPLAINS_SOME_NOT_ALL"

    summary = {
        "geometries": rows,
        "measured": {**MEASURED, "reference": REF, "measured_larger_box_over_baseline": meas_drop},
        "analytic_larger_box_over_baseline": analytic_lb_ratio,
        "analytic_flat_across_L": bool(flat),
        "local_mechanism_reproduces_measured_drop": bool(reproduces),
        "dx_effect_larger_box_fine_minus_coarse": dx_effect,
        "screening": {"naive_cG_over_omegaG": range_naive_G, "light_coupled_mode_range": float(range_light)},
        "verdict": verdict,
        "caveats": ["fixed-profile (Leg 1) analytic shift only; absorber neglected in the static T/G solve",
                    "first order in (A-1); quasi-static (ignores T/G transient/oscillatory dynamics)",
                    "the D4 measurement compares each row vs a FIXED N48/L10 reference (not a same-geometry off-baseline)"],
    }
    json.dump(summary, open(out / "summary.json", "w"), indent=2, default=float)
    print(f"\n[dx-vs-L] larger_box_fine(N68,L12) analytic/baseline = {lbf['analytic_ratio_to_baseline']:.3f} "
          f"(coarse L12 was {analytic_lb_ratio:.3f}); dx_effect={dx_effect:+.3f}", flush=True)
    print(f"\n=== {verdict} ===", flush=True)
    print("(analytic-mechanism discriminator only; no gravity/time-dilation/IRER claim; complements CX D4 review)", flush=True)


if __name__ == "__main__":
    main()
