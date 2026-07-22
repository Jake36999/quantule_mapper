"""Gravity Audit A.2b — force-contract closure (resolve the ~100x mismatch) + second target family.

The A.2 P3 "100x" was a normalization bug, not physics: momentum_x() returns a bare grid SUM (no dV), while integ()
includes dV=(L/N)^3. With a dV-consistent momentum, the NLS Ehrenfest law is EXACT:
    dP_A/dt = -kappa ∫ rho_A ∇rho_B dV     (external potential V_B=kappa rho_B; self-term integrates to 0)
This script derives/checks that quantitatively (analytic vs finite-difference under dt refinement), and re-runs the
predictor comparison for a SECOND target family to confirm I_int's advantage is not profile-specific.

Framing corrections carried in (reviewer): the coupling C=kappa rho_B psi_A is DENSITY coupling used as a controlled
harness — NOT "the natural IRER gravity coupling". I_int is the interaction-LOAD diagnostic computed FROM it. The
FORCE law uses ∇rho_B (the coupling-potential gradient), NOT ∇I_int.

Standalone numpy; no production/solver change; gravity PAUSED.

  python jax_scout/gravity_A2b_force_contract.py
"""
import os, sys, json, argparse, time, itertools
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout.gravity_A_rint_diagnostic import FAM, place, integ
from jax_scout.gravity_A1_predictive import evolve_coupled, momentum_x

D = FAM["D"]


def dx_field(f, kx):
    return np.real(np.fft.ifftn(1j * kx * np.fft.fftn(f)))


def P_dV(psi, kx, dV):
    """dV-consistent momentum integral P = ∫ Im(psi* d_x psi) dV."""
    return momentum_x(psi, kx) * dV


def rankdata(a):
    return np.argsort(np.argsort(a)).astype(float)


def spearman(a, b):
    return float(np.corrcoef(rankdata(np.asarray(a)), rankdata(np.asarray(b)))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi-iso", default="sweep_runs/PHASE_D_C2_8_TWONODE_CODEX_20260709_123611/phi_iso.npy")
    ap.add_argument("--L", type=float, default=20.0); ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=0.005); ap.add_argument("--out", default=None)
    A_ = ap.parse_args()
    out = A_.out or os.path.join(ROOT, "sweep_runs", f"GRAVITY_A2B_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    p = A_.phi_iso if os.path.isabs(A_.phi_iso) else os.path.join(ROOT, A_.phi_iso)
    phi = np.load(p).astype(np.complex128); N = phi.shape[0]; L = A_.L; kap = A_.kappa
    dV = (L / N) ** 3
    k1 = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    kx = k1[:, None, None] * np.ones((N, N, N)); ky = k1[None, :, None]; kz = k1[None, None, :]
    ksq = kx ** 2 + ky ** 2 + kz ** 2
    print(f"=== Gravity Audit A.2b | N={N} L={L} kappa={kap} | dV={dV:.6e} (1/dV={1/dV:.2f}) ===", flush=True)
    res = {"N": N, "L": L, "kappa": kap, "dV": dV}

    # ---------- Force contract: analytic Ehrenfest vs finite-difference (dV-consistent) ----------
    pA = place(phi, -2.0, N, L); pB = place(phi, +1.0, N, L)
    rA, rB = np.abs(pA) ** 2, np.abs(pB) ** 2
    F_analytic = -kap * integ(rA * dx_field(rB, kx), N, L)          # includes dV
    rows = []
    for dt in (0.02, 0.01, 0.005):
        nst = 4
        AT, BT = evolve_coupled(pA, pB, kx, ksq, kap, dt, nst)
        F_fd = P_dV(AT, kx, dV) / (nst * dt)                       # dV-consistent dP_A/dt
        rows.append({"dt": dt, "F_fd": F_fd, "rel_err_vs_analytic": abs(F_fd - F_analytic) / (abs(F_analytic) + 1e-30)})
        print(f"[force] dt={dt:.3f}: F_FD(dV)={F_fd:+.5e}  vs  F_analytic={F_analytic:+.5e}  "
              f"rel_err={rows[-1]['rel_err_vs_analytic']:.2e}", flush=True)
    force_ok = rows[-1]["rel_err_vs_analytic"] < 0.05
    print(f"[force] Ehrenfest law dP_A/dt = -kappa∫rho_A ∇rho_B holds under dt refine (100x was the missing dV) "
          f"-> {'PASS' if force_ok else 'FAIL'}", flush=True)
    res["force_contract"] = {"F_analytic": F_analytic, "rows": rows, "pass": bool(force_ok)}

    # ---------- Second target family: Gaussian blob target (env stays the soliton) ----------
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    def gauss(x0, sig=1.1, amp=1.0):
        return (amp * np.exp(-((X - x0) ** 2 + Y ** 2 + Z ** 2) / (2 * sig ** 2))).astype(np.complex128)
    seps = [2.0, 3.0, 4.0, 5.0]; phases = [0.0, np.pi / 2, np.pi]; amps = [0.8, 1.2]
    fam_rows = []
    for (s, dphi, amp) in itertools.product(seps, phases, amps):
        tA = gauss(-s / 2)                                          # DIFFERENT target morphology (Gaussian)
        tB = amp * np.exp(1j * dphi) * place(phi, +s / 2, N, L)
        rAt, rBt = np.abs(tA) ** 2, np.abs(tB) ** 2
        pred = {"neg_sep": -s, "overlap": integ(rAt * rBt, N, L),
                "I_int": integ(np.abs(kap * rBt * tA) ** 2, N, L)}
        AT, _ = evolve_coupled(tA, tB, kx, ksq, kap, A_.dt, int(round(0.5 / A_.dt)))
        fam_rows.append({"sep": s, "dphi": dphi, "amp": amp, **pred, "absP": abs(P_dV(AT, kx, dV))})
    y = [r["absP"] for r in fam_rows]
    sp = {pn: spearman([r[pn] for r in fam_rows], y) for pn in ("neg_sep", "overlap", "I_int")}
    fam_ok = abs(sp["I_int"]) >= abs(sp["overlap"]) - 1e-9
    print(f"[fam2] Gaussian target family, Spearman vs |P_A(T)|: "
          f"I_int={sp['I_int']:+.3f}  overlap={sp['overlap']:+.3f}  sep={sp['neg_sep']:+.3f} "
          f"-> I_int>=overlap: {fam_ok}", flush=True)
    res["second_family"] = {"spearman": sp, "I_int_ge_overlap": bool(fam_ok)}

    verdict = "A2B_FORCE_CONTRACT_CLOSED" if (force_ok and fam_ok) else "A2B_PARTIAL"
    res["verdict"] = verdict
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)
    print(f"\n=== {verdict} | force_law={force_ok} second_family_I_int>=overlap={fam_ok} | out={out} ===", flush=True)


if __name__ == "__main__":
    main()
