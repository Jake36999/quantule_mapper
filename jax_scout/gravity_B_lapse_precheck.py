"""Gravity Audit B — static lapse algebra pre-check (design deliverable; NOT the clock, which is Audit C).

Verifies the H-G- throttling lapse N(x)=1/(1+beta*I_hat(x)) is well-posed BEFORE any dynamical clock:
  (1) beta=0 -> N == 1 exactly (the essential null);
  (2) N is bounded in (0,1] for beta>0 (no cliff — 1/(1+x) is naturally bounded, unlike production Omega^2(rho));
  (3) flat background: where I_int -> 0 (vacuum), N -> 1;
  (4) DISTINGUISHABILITY from the overlap control: I_int ~ rho_A*rho_B^2 vs overlap ~ rho_A*rho_B differ by a factor
      rho_B, so N(I_int) and N(overlap) have DIFFERENT spatial profiles. If they were identical, I_int would add
      nothing over plain overlap. (Whether that difference is measurable in a clock is Audit C's job.)

Standalone numpy; no production/solver change; gravity ladder CLOSED. Design-only.

  python jax_scout/gravity_B_lapse_precheck.py
"""
import os, sys, json, argparse, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout.gravity_A_rint_diagnostic import FAM, place, integ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi-iso", default="sweep_runs/PHASE_D_C2_8_TWONODE_CODEX_20260709_123611/phi_iso.npy")
    ap.add_argument("--L", type=float, default=20.0); ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    A = ap.parse_args()
    out = A.out or os.path.join(ROOT, "sweep_runs", f"GRAVITY_B_PRECHECK_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    p = A.phi_iso if os.path.isabs(A.phi_iso) else os.path.join(ROOT, A.phi_iso)
    phi = np.load(p).astype(np.complex128); N = phi.shape[0]; L = A.L; kap = A.kappa
    print(f"=== Gravity Audit B — lapse pre-check | N={N} L={L} kappa={kap} ===", flush=True)

    psiA = place(phi, -2.0, N, L); psiB = place(phi, +1.0, N, L)
    rA, rB = np.abs(psiA) ** 2, np.abs(psiB) ** 2
    I_field = (kap ** 2) * rA * rB ** 2           # local interaction-load field  (~ rho_A rho_B^2)
    O_field = rA * rB                              # local plain-overlap field     (~ rho_A rho_B)
    I0 = integ(I_field, N, L) + 1e-30              # fixed reference scales (NOT per-run-fitted to any clock)
    O0 = integ(O_field, N, L) + 1e-30
    Ihat = I_field / I0 * integ(np.ones_like(rA), N, L)   # dimensionless O(1)-scaled load
    Ohat = O_field / O0 * integ(np.ones_like(rA), N, L)
    res = {"N": N, "L": L, "kappa": kap, "betas": {}}

    def lapse(hat, beta):
        return 1.0 / (1.0 + beta * hat)

    ok = {"beta0_null": True, "bounded": True, "flat_background": True, "distinguishable": True}
    for beta in (0.0, 0.5, 1.0, 2.0):
        Nf = lapse(Ihat, beta); NfO = lapse(Ohat, beta)
        nmin, nmax = float(Nf.min()), float(Nf.max())
        # background: cells where the load is negligible -> N should be ~1
        bg = Ihat < 1e-6
        bg_dev = float(np.max(np.abs(Nf[bg] - 1.0))) if bg.any() else 0.0
        # distinguishability from overlap lapse (spatial pattern), only meaningful for beta>0
        core = Ihat > 0.01 * Ihat.max()
        corr = float(np.corrcoef(Nf[core].ravel(), NfO[core].ravel())[0, 1]) if beta > 0 and core.any() else 1.0
        maxdiff = float(np.max(np.abs(Nf[core] - NfO[core]))) if beta > 0 and core.any() else 0.0
        res["betas"][beta] = {"N_min": nmin, "N_max": nmax, "bg_max_dev_from_1": bg_dev,
                              "vs_overlap_corr_in_core": corr, "vs_overlap_maxdiff": maxdiff}
        print(f"[beta={beta:.1f}] N in [{nmin:.4f}, {nmax:.4f}]  bg|N-1|max={bg_dev:.2e}  "
              f"vs-overlap: corr={corr:.4f} maxdiff={maxdiff:.4f}", flush=True)
        if beta == 0.0 and (abs(nmin - 1.0) > 1e-12 or abs(nmax - 1.0) > 1e-12):
            ok["beta0_null"] = False
        if beta > 0.0:
            if not (0.0 < nmin <= nmax <= 1.0 + 1e-12):
                ok["bounded"] = False
            if bg_dev > 1e-4:
                ok["flat_background"] = False
            if maxdiff < 1e-3:            # N(I_int) must NOT be identical to N(overlap)
                ok["distinguishable"] = False

    verdict = "B_LAPSE_PRECHECK_PASS" if all(ok.values()) else "B_LAPSE_PRECHECK_PARTIAL"
    res["checks"] = ok; res["verdict"] = verdict
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)
    print(f"\n=== {verdict} | {ok} | out={out} ===", flush=True)
    print("   (design pre-check only; the CLOCK effect + overlap control run dynamically in Audit C)", flush=True)


if __name__ == "__main__":
    main()
