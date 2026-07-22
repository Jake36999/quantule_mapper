"""Gravity Audit A — R_int relational-rate identification diagnostic (Stage A).

Tests whether the interaction-conditioned resolution rate
    R_int(x) = || F_full[psi] - F_isolated[psi] ||^2      (state change attributable to environmental coupling)
is a clean RELATIONAL observable, as the theory requires. On the true-flat NLS substrate (geom_off) the linear
(D∇²) part of F=∂_tψ is additive and CANCELS in the counterfactual subtraction, so R_int is EXACTLY the nonlinear
interaction term (pure field algebra; no solver step, no FFT):

    R_int_field = | g(rho_full)·psi_full - g(rho_L)·psi_L - g(rho_R)·psi_R |^2,   g(rho)=a·rho+s·rho^2+f·rho^3
    psi_full = psi_L + psi_R,  rho = |psi|^2

Stage-A checks (see docs/IRER_GRAVITY_REENTRY_AUDIT_AND_VALIDATION_PLAN.md §2A/§3):
  A1  distance/isolation : R_int -> 0 as separation grows (isolated); grows on approach.
  A2  self-vs-relational : at large separation the SELF nonlinear activity is large but R_int ~ 0 (a breathing/
      phase-rotating ISOLATED soliton has |∂_tψ|^2 > 0 yet R_int = 0 — R_int excludes self-activity).
  A3  density-decoupling : an ISOLATED soliton has R_int = 0 at ANY amplitude/density; and R_int varies with
      RELATIVE PHASE at fixed separation -> raw density cannot explain R_int.
  A4  determinism/convergence : identical config -> identical R_int (bit-exact); Fourier-refined grid -> R_int stable.

Read-only analysis. No production/solver/Hunter/config change; no gravity claim. Gravity sector remains PAUSED;
this only validates the SOURCE observable (Audit A), independent of any metric/force question.

  python jax_scout/gravity_A_rint_diagnostic.py [--phi-iso <run>/phi_iso.npy] [--out DIR]
"""
import os, sys, json, argparse, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAM = {"a": 0.8, "s": -0.5, "f": -0.1, "D": 0.3}   # matches C2.9/C2.10 Galilean family
L_DEFAULT = 20.0


def g(rho):
    return FAM["a"] * rho + FAM["s"] * rho ** 2 + FAM["f"] * rho ** 3


def nonlin(psi):
    return g(np.abs(psi) ** 2) * psi


def R_int_field(psiL, psiR):
    full = psiL + psiR
    return np.abs(nonlin(full) - nonlin(psiL) - nonlin(psiR)) ** 2


def place(phi, x0, N, L):
    """EXACT sub-cell shift by x0 along axis 0 via spectral phase (band-limited, grid-independent) — so a given
    physical separation is identical across grids (integer-cell rolling would make 'sep' grid-dependent and
    spuriously break grid-convergence for a separation-steep quantity like R_int)."""
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    F = np.fft.fft(phi, axis=0) * np.exp(-1j * k * x0)[:, None, None]
    return np.fft.ifft(F, axis=0)


def integ(field, N, L):
    return float(field.sum()) * (L / N) ** 3          # ∫ dV


def self_activity(psiL, psiR, N, L):
    return integ(np.abs(nonlin(psiL)) ** 2, N, L) + integ(np.abs(nonlin(psiR)) ** 2, N, L)


def R_int(psiL, psiR, N, L):
    return integ(R_int_field(psiL, psiR), N, L)


def fourier_refine(phi, N2):
    """Exact (band-limited) resample of a periodic field N^3 -> N2^3 by k-space zero-padding."""
    N = phi.shape[0]
    F = np.fft.fftn(phi)
    Fs = np.fft.fftshift(F)
    out = np.zeros((N2, N2, N2), dtype=complex)
    lo = (N2 - N) // 2
    out[lo:lo + N, lo:lo + N, lo:lo + N] = Fs
    return np.fft.ifftn(np.fft.ifftshift(out)) * (N2 / N) ** 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi-iso", default="sweep_runs/PHASE_D_C2_8_TWONODE_CODEX_20260709_123611/phi_iso.npy")
    ap.add_argument("--L", type=float, default=L_DEFAULT)
    ap.add_argument("--out", default=None)
    A = ap.parse_args()
    out = A.out or os.path.join(ROOT, "sweep_runs", f"GRAVITY_A_RINT_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    p = A.phi_iso if os.path.isabs(A.phi_iso) else os.path.join(ROOT, A.phi_iso)
    phi = np.load(p).astype(np.complex128)
    N = phi.shape[0]; L = A.L
    amp = float(np.abs(phi).max()); mass1 = integ(np.abs(phi) ** 2, N, L)
    print(f"=== Gravity Audit A — R_int diagnostic | {FAM} | N={N} L={L} | phi_iso amp={amp:.3f} mass={mass1:.1f} ===", flush=True)
    res = {"family": FAM, "N": N, "L": L, "phi_iso": p, "amp_single": amp, "mass_single": mass1}

    # --- A1: distance / isolation (in-phase pair) ---
    seps = [12.0, 10.0, 8.0, 6.0, 4.0, 2.0, 1.0]
    a1 = []
    for s in seps:
        psiL = place(phi, -s / 2, N, L); psiR = place(phi, +s / 2, N, L)
        r = R_int(psiL, psiR, N, L); sa = self_activity(psiL, psiR, N, L)
        a1.append({"sep": s, "R_int": r, "self_activity": sa, "relational_frac": r / (sa + 1e-30)})
        print(f"[A1] sep={s:5.1f}  R_int={r:.4e}  self_act={sa:.4e}  relational_frac={r/(sa+1e-30):.3e}", flush=True)
    res["A1_distance"] = a1
    r_far, r_close = a1[0]["R_int"], a1[-2]["R_int"]           # sep=12 vs sep=2
    a1_pass = (r_far / (r_close + 1e-30) < 0.05) and (r_close > 0)

    # --- A2: self-vs-relational at large separation (the breather point) ---
    psiL = place(phi, -6.0, N, L); psiR = place(phi, +6.0, N, L)   # sep=12, ~isolated
    r_far2 = R_int(psiL, psiR, N, L); sa_far = self_activity(psiL, psiR, N, L)
    a2_pass = (sa_far > 1e-6) and (r_far2 / (sa_far + 1e-30) < 1e-2)
    print(f"[A2] large-sep: SELF nonlinear activity={sa_far:.4e} (>0), R_int={r_far2:.4e} -> relational fraction "
          f"{r_far2/(sa_far+1e-30):.2e} (R_int excludes self-activity: {'PASS' if a2_pass else 'FAIL'})", flush=True)
    res["A2_self_vs_relational"] = {"self_activity": sa_far, "R_int": r_far2, "relational_frac": r_far2 / (sa_far + 1e-30), "pass": a2_pass}

    # --- A3a: isolated soliton has R_int=0 at ANY density ---
    a3iso = []
    for scale in (1.0, 1.5, 2.0):
        psi_iso = scale * phi
        r_iso = R_int(psi_iso, np.zeros_like(psi_iso), N, L)      # no partner
        pk = float((np.abs(psi_iso) ** 2).max())
        a3iso.append({"amp_scale": scale, "peak_rho": pk, "R_int": r_iso})
        print(f"[A3a] isolated soliton amp×{scale}: peak_rho={pk:.3f}  R_int={r_iso:.3e} (should be ~0)", flush=True)
    # --- A3b: R_int varies with RELATIVE PHASE at fixed separation -> beyond density ---
    a3ph = []
    s_fix = 4.0
    for dphi in (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi):
        psiL = place(phi, -s_fix / 2, N, L); psiR = np.exp(1j * dphi) * place(phi, +s_fix / 2, N, L)
        r = R_int(psiL, psiR, N, L); m = integ(np.abs(psiL + psiR) ** 2, N, L)
        a3ph.append({"dphi": dphi, "R_int": r, "total_mass": m})
        print(f"[A3b] sep={s_fix} dphi={dphi:.3f}: R_int={r:.4e}  total_mass={m:.2f}", flush=True)
    rphys = [x["R_int"] for x in a3ph]
    a3_pass = (max(a["R_int"] for a in a3iso) < 1e-12) and ((max(rphys) - min(rphys)) / (max(rphys) + 1e-30) > 0.1)
    print(f"[A3] isolated max R_int={max(a['R_int'] for a in a3iso):.2e} (~0); phase-spread of R_int="
          f"{(max(rphys)-min(rphys))/(max(rphys)+1e-30):.2f} -> density cannot explain R_int: "
          f"{'PASS' if a3_pass else 'FAIL'}", flush=True)
    res["A3_density_decoupling"] = {"isolated": a3iso, "phase": a3ph, "pass": a3_pass}

    # --- A4: determinism + grid convergence (Fourier refine to N2) ---
    psiL = place(phi, -2.0, N, L); psiR = place(phi, +2.0, N, L)  # sep=4
    r_a = R_int(psiL, psiR, N, L); r_b = R_int(psiL, psiR, N, L)
    determ = (r_a == r_b)
    N2 = int(round(N * 4 / 3)) if N % 3 == 0 else N + 32          # a finer band-limited grid
    phi2 = fourier_refine(phi, N2)
    psiL2 = place(phi2, -2.0, N2, L); psiR2 = place(phi2, +2.0, N2, L)
    r_ref = R_int(psiL2, psiR2, N2, L)
    conv_err = abs(r_ref - r_a) / (abs(r_a) + 1e-30)
    a4_pass = determ and (conv_err < 0.05)
    print(f"[A4] determinism: identical={'YES' if determ else 'NO'}; grid N={N}->{N2} R_int {r_a:.4e}->{r_ref:.4e} "
          f"(rel err {conv_err:.2e}) -> {'PASS' if a4_pass else 'FAIL'}", flush=True)
    res["A4_determinism_convergence"] = {"identical": bool(determ), "N": N, "N2": N2, "R_int_N": r_a, "R_int_N2": r_ref, "conv_err": conv_err, "pass": a4_pass}

    # ================= A.1 HARDENING (per reviewer): name/normalize, density control, decomposition ================
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)

    # --- A5: intensity vs RATE, and the LOCAL field (gravity/clock need the local field, not a global norm) ---
    psiL = place(phi, -2.0, N, L); psiR = place(phi, +2.0, N, L); full = psiL + psiR   # sep=4 in-phase
    dF = nonlin(full) - nonlin(psiL) - nonlin(psiR)              # interaction forcing (i-stripped)
    I_int = integ(np.abs(dF) ** 2, N, L)                         # INTENSITY / power, units (field/time)^2
    gamma = float(np.sqrt(integ(np.abs(dF) ** 2, N, L) / (integ(np.abs(full) ** 2, N, L) + 1e-30)))  # RATE ~ 1/time
    gamma_field_x = (np.abs(dF) / (np.abs(full) + 1e-6)).sum(axis=(1, 2))
    I_int_field_x = (np.abs(dF) ** 2).sum(axis=(1, 2))
    np.savez_compressed(os.path.join(out, "rint_local_sep4.npz"), x=x, I_int_x=I_int_field_x, gamma_x=gamma_field_x)
    print(f"[A5] naming: I_int (intensity, (field/t)^2)={I_int:.4e}  gamma=||dF||/||psi|| (rate, 1/t)={gamma:.4e}; "
          f"local fields saved (x-proj). Gravity/clock must use the LOCAL field.", flush=True)
    res["A5_naming"] = {"I_int_intensity": I_int, "gamma_rate": gamma, "note": "I_int is interaction INTENSITY, not yet a completed-resolution rate"}

    # --- A6: decomposition robustness — spectral vs spatial-mask components (ordering must survive) ---
    a6 = []
    for s in (6.0, 4.0, 2.0):
        pL = place(phi, -s / 2, N, L); pR = place(phi, +s / 2, N, L); fl = pL + pR
        I_spec = integ(np.abs(nonlin(fl) - nonlin(pL) - nonlin(pR)) ** 2, N, L)
        wl = 0.5 * (1.0 - np.tanh(x / 0.6))[:, None, None]      # smooth midplane split of the SAME total field
        mL = fl * wl; mR = fl * (1.0 - wl)
        I_mask = integ(np.abs(nonlin(fl) - nonlin(mL) - nonlin(mR)) ** 2, N, L)
        a6.append({"sep": s, "I_spectral": I_spec, "I_mask": I_mask})
        print(f"[A6] sep={s}: I_int spectral={I_spec:.4e}  mask={I_mask:.4e}", flush=True)
    ord_spec = [r["I_spectral"] for r in a6]; ord_mask = [r["I_mask"] for r in a6]
    a6_pass = all(np.diff(ord_spec) > 0) and all(np.diff(ord_mask) > 0)   # both grow as sep shrinks
    print(f"[A6] ordering preserved under decomposition (grows on approach): "
          f"spectral {'Y' if all(np.diff(ord_spec)>0) else 'N'}, mask {'Y' if all(np.diff(ord_mask)>0) else 'N'} "
          f"-> {'PASS' if a6_pass else 'FAIL'}", flush=True)
    res["A6_decomposition_robustness"] = {"rows": a6, "pass": bool(a6_pass)}

    # --- A7: fixed-total-mass phase sweep (partial DENSITY control) — does I_int vary at held ∫ρ? ---
    s_fix = 4.0; M0 = mass1 * 2.0                                # target total mass
    a7 = []
    for dphi in (0.0, np.pi / 2, np.pi):
        pL = place(phi, -s_fix / 2, N, L); pR = np.exp(1j * dphi) * place(phi, +s_fix / 2, N, L)
        m = integ(np.abs(pL + pR) ** 2, N, L); sc = np.sqrt(M0 / (m + 1e-30))
        pLn, pRn = sc * pL, sc * pR
        r = integ(np.abs(nonlin(pLn + pRn) - nonlin(pLn) - nonlin(pRn)) ** 2, N, L)
        a7.append({"dphi": dphi, "I_int_fixed_mass": r, "mass": integ(np.abs(pLn + pRn) ** 2, N, L)})
        print(f"[A7] dphi={dphi:.3f} @ fixed total_mass={M0:.1f}: I_int={r:.4e}", flush=True)
    rr = [a["I_int_fixed_mass"] for a in a7]
    a7_spread = (max(rr) - min(rr)) / (max(rr) + 1e-30)
    print(f"[A7] I_int spread at FIXED total mass = {a7_spread:.2f} (>0 => I_int carries info beyond total mass; "
          f"caveat: pointwise interference profile still varies)", flush=True)
    res["A7_fixed_mass_phase"] = {"rows": a7, "spread": a7_spread,
                                  "caveat": "holds total mass fixed, not the full pointwise density profile"}

    passes = {"A1_distance_isolation": bool(a1_pass), "A2_excludes_self_activity": bool(a2_pass),
              "A3_density_decoupling_partial": bool(a3_pass), "A4_determinism_convergence": bool(a4_pass),
              "A6_decomposition_robust": bool(a6_pass)}
    # NOT promoted to "resolution-rate source": this validates an interaction-INTENSITY diagnostic (candidate I_int),
    # not the completed-resolution rate R_res=G(I_int) that defines emergent time. Matched-POINTWISE-density control
    # and causal-ordering (partner-ablation predicts subsequent divergence) remain for A.1 completion.
    verdict = "INTERACTION_INTENSITY_DIAGNOSTIC_VALIDATED" if all(passes.values()) else (
        "INTERACTION_INTENSITY_DIAGNOSTIC_PARTIAL" if any(passes.values()) else "DIAGNOSTIC_FAIL")
    res["stage_a_passes"] = passes; res["verdict"] = verdict
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)
    print(f"\n=== {verdict} | {passes} | out={out} ===", flush=True)


if __name__ == "__main__":
    main()
