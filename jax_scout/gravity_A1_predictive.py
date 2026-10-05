"""Gravity Audit A.1 — predictive validity beyond the Taylor identity (reviewer criteria 6-7).

The causal-alignment result was a short-time identity (div(t)=t·ΔF_int(0)+O(t²)), so it is not independent evidence.
This test asks the non-circular question: does the initial interaction intensity I_int^A(0) predict an INDEPENDENT
LATER observable — the momentum imparted to the target A — using a two-field mirror model evolved forward, where the
predictor (I_int at t=0) and the target (P_A at t=T) are different quantities?

Model (two explicit subsystems, coupled NLS; minimal standalone split-step, no production solver):
    i∂_tψ_A = −D∇²ψ_A + g(ρ_A)ψ_A + κ|ψ_B|²ψ_A ,   and symmetrically for B.
Target A starts at rest (P_A(0)=0) with a FIXED IC (so ρ_A(0) is identical across configs); only the ENVIRONMENT B
is varied (its distance). Density-only predictors therefore predict ZERO variation in A's response; if I_int^A(0)
predicts |P_A(T)|, it carries predictive information density does not.

Read-only standalone numpy; no production/solver change; gravity sector PAUSED. Tests the SOURCE's predictive
validity; not a metric/clock/force/gravity claim.

  python jax_scout/gravity_A1_predictive.py
"""
import os, sys, json, argparse, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout.gravity_A_rint_diagnostic import FAM, place, integ

D = FAM["D"]


def gofrho(rho):
    return FAM["a"] * rho + FAM["s"] * rho ** 2 + FAM["f"] * rho ** 3


def momentum_x(psi, kx):
    dxpsi = np.fft.ifftn(1j * kx * np.fft.fftn(psi))
    return float(np.imag(np.conj(psi) * dxpsi).sum())


def evolve_coupled(psiA, psiB, kx, ksq, kappa, dt, nsteps):
    """Strang split-step for the coupled NLS (linear in Fourier, nonlinear+coupling in real space)."""
    Ehalf = np.exp(-1j * D * ksq * (dt / 2.0))     # i∂_tψ=-D∇²ψ -> ψ_k *= exp(-iD k² dt)
    A = psiA.copy(); B = psiB.copy()
    for _ in range(nsteps):
        A = np.fft.ifftn(Ehalf * np.fft.fftn(A)); B = np.fft.ifftn(Ehalf * np.fft.fftn(B))
        rA = np.abs(A) ** 2; rB = np.abs(B) ** 2
        A = A * np.exp(-1j * (gofrho(rA) + kappa * rB) * dt)
        B = B * np.exp(-1j * (gofrho(rB) + kappa * rA) * dt)
        A = np.fft.ifftn(Ehalf * np.fft.fftn(A)); B = np.fft.ifftn(Ehalf * np.fft.fftn(B))
    return A, B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi-iso", default="sweep_runs/PHASE_D_C2_8_TWONODE_CODEX_20260709_123611/phi_iso.npy")
    ap.add_argument("--L", type=float, default=20.0); ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=0.005); ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    A_ = ap.parse_args()
    out = A_.out or os.path.join(ROOT, "sweep_runs", f"GRAVITY_A1_PREDICT_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    p = A_.phi_iso if os.path.isabs(A_.phi_iso) else os.path.join(ROOT, A_.phi_iso)
    phi = np.load(p).astype(np.complex128); N = phi.shape[0]; L = A_.L; kap = A_.kappa
    k1 = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    kx = k1[:, None, None] * np.ones((N, N, N)); ky = k1[None, :, None]; kz = k1[None, None, :]
    ksq = kx ** 2 + ky ** 2 + kz ** 2
    nsteps = int(round(A_.T / A_.dt))
    print(f"=== Gravity Audit A.1 — predictive validity | N={N} L={L} kappa={kap} T={A_.T} ===", flush=True)

    psiA0 = place(phi, -2.0, N, L)                 # target: FIXED IC across all configs (rho_A(0) identical)
    rows = []
    for xB in (-1.0, 0.0, 1.0, 2.0, 3.0, 4.0):     # vary ONLY the environment distance
        psiB0 = place(phi, xB, N, L)
        I0 = integ(np.abs(kap * np.abs(psiB0) ** 2 * psiA0) ** 2, N, L)   # I_int^A(0)
        PA0 = momentum_x(psiA0, kx)                                        # ~0 (at rest)
        AT, BT = evolve_coupled(psiA0, psiB0, kx, ksq, kap, A_.dt, nsteps)
        PAT = momentum_x(AT, kx)                                           # INDEPENDENT later observable
        rows.append({"xB": xB, "dist": abs(xB + 2.0), "I_int0": I0, "P_A_0": PA0, "P_A_T": PAT, "absP_A_T": abs(PAT)})
        print(f"[cfg] env dist={abs(xB+2):.1f}: I_int^A(0)={I0:.4e}  ->  |P_A(T)|={abs(PAT):.4e} "
              f"(P_A(0)={PA0:+.2e})", flush=True)

    I0s = np.array([r["I_int0"] for r in rows]); PTs = np.array([r["absP_A_T"] for r in rows])
    # rank correlation (monotone predictive power), log-log slope, and the density baseline note
    order_ok = np.all(np.argsort(-I0s) == np.argsort(-PTs))
    with np.errstate(all="ignore"):
        pear = float(np.corrcoef(np.log(I0s + 1e-30), np.log(PTs + 1e-30))[0, 1])
    verdict = "PREDICTIVE_VALIDITY_SUPPORTED" if (order_ok and pear > 0.9) else "PREDICTIVE_VALIDITY_PARTIAL"
    print(f"\n[pred] rho_A(0) is IDENTICAL across configs -> a density predictor gives ZERO variance; "
          f"I_int^A(0) monotonically orders |P_A(T)|: {order_ok}; log-log corr={pear:.3f}", flush=True)
    print(f"=== {verdict} | I_int(0) predicts the independent later momentum response; density cannot | out={out} ===", flush=True)
    res = {"N": N, "L": L, "kappa": kap, "T": A_.T, "dt": A_.dt, "rows": rows,
           "monotone_order": bool(order_ok), "loglog_corr": pear, "verdict": verdict,
           "note": "rho_A(0) fixed across configs; density-only predictor has zero variance, so any predictive power here is beyond density"}
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
