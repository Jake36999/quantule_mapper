"""TG-B1S frequency-contract check (Claude equation-audit finding 3).

Question: is the measured D3/D4 modal-frequency shift explained by the quasi-static geometric response A(x)?

Method (no perturbed eigensolve needed; CPU-only so the live D4 GPU job is untouched):
  1. Rebuild the frozen Q-ball (validated Petviashvili, exact D3 config) and its S_state (exact b1s normalization).
  2. Solve the STATIC screened T/G response analytically in Fourier space (exact for the linear T/G sector):
         (c_T^2 k^2 + w_T^2) T_k = alpha_T S_k - kappa G_k
         (c_G^2 k^2 + w_G^2) G_k = -kappa T_k
     ->  G_k = -kappa alpha_T S_k / D_k,  T_k = alpha_T S_k (c_G^2 k^2 + w_G^2)/D_k,
         D_k = (c_T^2 k^2 + w_T^2)(c_G^2 k^2 + w_G^2) - kappa^2   (>0 since kappa^2 < w_T^2 w_G^2).
     Then A(x) = exp(-eps_G * lambda_fb * G(x)).
  3. Leg 1 (fixed profile, Rayleigh):   d_omega_fp = c^2 Int (A-1)|grad phi0|^2 dV / (2 w0 Int rho0 dV).
  4. Leg 2 (fixed Q, envelope theorem): perturbed family energy E'(Q) = E(Q) + dE(Q) with
         dE(omega) = c^2 Int (A-1)|grad phi_omega|^2 dV   on the UNPERTURBED Petviashvili family,
     so  d_omega_Q = [d(dE)/d_omega] / [dQ/d_omega]      (central differences over omega0 +/- delta).
  5. Convention conversion: the D3 observable is delta_omega_infty = slope of (theta_full - theta_off) with
     theta ~ -omega t  =>  measured PHYSICAL shift = -delta_omega_infty = +2.150936882840006e-06.

Scientific boundary: an analytic-mechanism check for the frozen phenomenological model only. No gravity /
time-dilation / IRER claim. Read-only; no model, label, or production change.

  python jax_scout/gravity_TG_B1S_frequency_contract.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")           # do NOT touch the GPU (D4 campaign is live)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from jax_scout.phase_d_c3_wave import build_kg, qball_petviashvili  # noqa: E402

# Frozen D3 configuration (colab_jobs/baselines/tg_b1s_d_20260714_184736_baseline_manifest.json)
CFG = dict(N=48, L=10.0, c=0.5477, m=1.0, a=0.8, s=-0.5, f=-0.1, w=0.964, dt=0.002,
           alpha_T=0.35, omega_T=1.25, omega_G=0.85, gamma_T=0.08, gamma_G=0.06,
           kappa_TG=0.55, epsilon_G=0.06, cT=0.7, cG=0.55, core_radius=2.0, lambda_fb=1.0)
S0_AUTHORITATIVE = 135.6862187684289
MEASURED_DELTA_OMEGA_INFTY = -2.150936882840006e-06     # D3 convention (slope of delta_theta)
MEASURED_PHYSICAL_SHIFT = -MEASURED_DELTA_OMEGA_INFTY   # omega_full - omega_off in physical convention


def solve_family_member(w: float):
    op = build_kg(CFG["N"], CFG["L"], CFG["c"], CFG["m"], CFG["dt"])
    mu = CFG["m"] ** 2 - w ** 2
    for sig in (1.5, 1.2, 1.8, 2.0):
        phi, prof = qball_petviashvili(op, CFG["a"], CFG["s"], CFG["f"], mu, sig=sig)
        if phi is not None and prof["residual"] < 1e-6 and prof["occ"] < 0.5:
            return np.asarray(phi, dtype=np.complex128), prof, op
    raise RuntimeError(f"Q-ball solve failed at w={w}")


def grids(op):
    N, L = CFG["N"], CFG["L"]
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    k1 = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing="ij")
    return X, Y, Z, KX, KY, KZ, (L / N) ** 3


def grad2_of(phi, KX, KY, KZ):
    pk = np.fft.fftn(phi)
    gx = np.fft.ifftn(1j * KX * pk); gy = np.fft.ifftn(1j * KY * pk); gz = np.fft.ifftn(1j * KZ * pk)
    return np.abs(gx) ** 2 + np.abs(gy) ** 2 + np.abs(gz) ** 2


def s_state_field(phi, KX, KY, KZ):
    """Exact b1s S_state: [0.5 max(E,0)/e_ref + 0.5 |q|/q_ref]/S0, refs = per-field maxima (b1s ref_source_norms)."""
    w = CFG["w"]
    pi = -1j * w * phi
    rho = np.abs(phi) ** 2
    g2 = grad2_of(phi, KX, KY, KZ)
    Gpot = CFG["a"] * rho ** 2 / 2 + CFG["s"] * rho ** 3 / 3 + CFG["f"] * rho ** 4 / 4
    energy = np.abs(pi) ** 2 + CFG["c"] ** 2 * g2 + CFG["m"] ** 2 * rho - Gpot
    charge = np.abs(np.imag(np.conj(phi) * pi))
    e_ref = float(np.max(np.abs(energy))); q_ref = float(np.max(charge))
    raw = 0.5 * np.maximum(energy, 0.0) / (e_ref + 1e-12) + 0.5 * charge / (q_ref + 1e-12)
    return raw / S0_AUTHORITATIVE


def static_TG_response(S, KX, KY, KZ):
    """Exact static screened solve of the linear T/G sector (absorber neglected: fields screened well inside)."""
    k2 = KX ** 2 + KY ** 2 + KZ ** 2
    Sk = np.fft.fftn(S)
    dT = CFG["cT"] ** 2 * k2 + CFG["omega_T"] ** 2
    dG = CFG["cG"] ** 2 * k2 + CFG["omega_G"] ** 2
    D = dT * dG - CFG["kappa_TG"] ** 2                      # >0 for all k (stability margin 3.7x)
    Tk = CFG["alpha_T"] * Sk * dG / D
    Gk = -CFG["kappa_TG"] * CFG["alpha_T"] * Sk / D
    return np.real(np.fft.ifftn(Tk)), np.real(np.fft.ifftn(Gk))


def main():
    out = ROOT / "sweep_runs" / f"TG_B1S_FREQ_CONTRACT_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"=== TG-B1S frequency contract | CPU-only (JAX_PLATFORMS={os.environ['JAX_PLATFORMS']}) | out={out} ===", flush=True)

    w0 = CFG["w"]
    phi0, prof0, op = solve_family_member(w0)
    X, Y, Z, KX, KY, KZ, dV = grids(op)
    rho0 = np.abs(phi0) ** 2
    print(f"[qball] w0={w0} residual={prof0['residual']:.2e} amp={prof0['amp']:.4f} occ={prof0['occ']:.4f}", flush=True)

    # --- static screened response + A field ---
    S = s_state_field(phi0, KX, KY, KZ)
    T, G = static_TG_response(S, KX, KY, KZ)
    A = np.exp(-CFG["epsilon_G"] * CFG["lambda_fb"] * G)
    rr2 = X ** 2 + Y ** 2 + Z ** 2
    core = np.exp(-0.5 * rr2 / CFG["core_radius"] ** 2)
    cnorm = float(np.sum(core) * dV)
    T_node = float(np.sum(core * T) * dV / cnorm); G_node = float(np.sum(core * G) * dV / cnorm)
    print(f"[static] T_peak={np.max(np.abs(T)):.6e}  G_peak={np.max(np.abs(G)):.6e}  "
          f"T_node={T_node:.6e}  G_node={G_node:.6e}", flush=True)
    print(f"[static] sign check: G<0 near node = {bool(G_node < 0)};  A-1 in core: "
          f"max={np.max(A-1):.3e} min={np.min(A-1):.3e}  (A>1 at node = A-hill, per audit finding 2)", flush=True)

    # --- Leg 1: fixed-profile Rayleigh shift ---
    g2_0 = grad2_of(phi0, KX, KY, KZ)
    num = CFG["c"] ** 2 * float(np.sum((A - 1.0) * g2_0) * dV)
    den = 2.0 * w0 * float(np.sum(rho0) * dV)
    d_omega_fp = num / den
    print(f"[leg1] fixed-profile d_omega = {d_omega_fp:+.6e}  (direct stiffening channel)", flush=True)

    # --- Leg 2: fixed-Q envelope shift over the unperturbed family ---
    dw = 0.004
    dE = {}; Q = {}
    for w in (w0 - dw, w0, w0 + dw):
        phw, prw, _ = solve_family_member(w)
        rw = np.abs(phw) ** 2
        Q[w] = w * float(np.sum(rw) * dV)                     # Q = Im int phi* pi = w int rho
        dE[w] = CFG["c"] ** 2 * float(np.sum((A - 1.0) * grad2_of(phw, KX, KY, KZ)) * dV)
        print(f"[leg2] w={w:.4f}: residual={prw['residual']:.1e}  Q={Q[w]:.6f}  dE={dE[w]:+.6e}", flush=True)
    ddE_dw = (dE[w0 + dw] - dE[w0 - dw]) / (2 * dw)
    dQ_dw = (Q[w0 + dw] - Q[w0 - dw]) / (2 * dw)
    d_omega_Q = ddE_dw / dQ_dw
    print(f"[leg2] d(dE)/dw={ddE_dw:+.6e}  dQ/dw={dQ_dw:+.4f} (VK sign: {'stable dQ/dw<0' if dQ_dw<0 else 'dQ/dw>0'})  "
          f"-> fixed-Q d_omega = {d_omega_Q:+.6e}", flush=True)

    # --- comparison in BOTH conventions ---
    meas_phys = MEASURED_PHYSICAL_SHIFT
    print(f"\n[compare] measured D3 delta_omega_infty (slope of delta_theta) = {MEASURED_DELTA_OMEGA_INFTY:+.6e}")
    print(f"[compare] measured PHYSICAL shift (omega_full - omega_off)      = {meas_phys:+.6e}")
    print(f"[compare] predicted physical shift, fixed-profile (leg1)        = {d_omega_fp:+.6e}  "
          f"ratio={d_omega_fp/meas_phys:+.3f}")
    print(f"[compare] predicted physical shift, fixed-Q (leg2)              = {d_omega_Q:+.6e}  "
          f"ratio={d_omega_Q/meas_phys:+.3f}", flush=True)
    best = d_omega_Q if abs(d_omega_Q / meas_phys - 1) < abs(d_omega_fp / meas_phys - 1) else d_omega_fp
    sign_ok = best * meas_phys > 0
    mag_ok = 0.5 < abs(best / meas_phys) < 2.0
    verdict = ("TG_B1S_FREQUENCY_CONTRACT_SIGN_AND_MAGNITUDE_MATCH" if (sign_ok and mag_ok)
               else "TG_B1S_FREQUENCY_CONTRACT_SIGN_MATCH_ONLY" if sign_ok
               else "TG_B1S_FREQUENCY_CONTRACT_MISMATCH")
    res = {"config": CFG, "S0": S0_AUTHORITATIVE, "qball_residual": prof0["residual"],
           "T_peak": float(np.max(np.abs(T))), "G_peak": float(np.max(np.abs(G))),
           "T_node": T_node, "G_node": G_node, "A_minus_1_max": float(np.max(A - 1)), "A_minus_1_min": float(np.min(A - 1)),
           "leg1_fixed_profile_d_omega": d_omega_fp, "leg2_fixed_Q_d_omega": d_omega_Q,
           "dQ_dw": dQ_dw, "ddE_dw": ddE_dw,
           "measured_delta_omega_infty_D3_convention": MEASURED_DELTA_OMEGA_INFTY,
           "measured_physical_shift": meas_phys,
           "convention_note": "D3 delta_omega_infty is the slope of (theta_full - theta_off); theta ~ -omega t, so physical omega shift = -delta_omega_infty",
           "caveats": ["quasi-static (ignores T/G transient + oscillatory dynamics)",
                       "first-order in (A-1)", "absorber neglected in the static solve (fields screened)",
                       "steady S from unperturbed node (second-order neglected)"],
           "verdict": verdict}
    json.dump(res, open(out / "summary.json", "w"), indent=2, default=float)
    print(f"\n=== {verdict} | analytic-mechanism check only; no gravity/time-dilation/IRER claim ===", flush=True)


if __name__ == "__main__":
    main()
