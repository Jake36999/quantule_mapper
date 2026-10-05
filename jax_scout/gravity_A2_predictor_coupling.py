"""Gravity Audit A.2 — coupling-faithfulness + predictor generality + direct force-gradient test.

Addresses three reviewer gaps before any lapse/clock:
  P1  COUPLING FAITHFULNESS: the explicit coupling C(A,B)=kappa|psi_B|^2 psi_A must behave lawfully — lambda=0 null,
      A<->B exchange symmetry, equal-and-opposite momentum transfer (P_A+P_B conserved), global-phase covariance.
  P2  PREDICTOR GENERALITY: on a BROADER sweep (separation x phase x env-amplitude), does I_int^A(0) predict the
      independent later |P_A(T)| BETTER than simple relational baselines — separation d_AB, overlap ∫rho_A rho_B,
      cross-energy, grad-overlap? (The 0.967 distance-only correlation may be a shared dependence on separation.)
      Reported: Spearman rank correlation per predictor + leave-one-phase-out held-out check.
  P3  FORCE vs GRADIENT: is the force on A the GRADIENT of the coupling potential (kappa*rho_B), distinct from the
      scalar intensity I_int? Compare predicted F_x=-kappa∫rho_A ∂_x rho_B dV to measured dP_A/dt, with nulls:
      centred/symmetric env -> ~0; reflected env -> sign flips; lambda=0 -> 0.

Minimal standalone coupled-NLS split-step (numpy); no production/solver change; gravity PAUSED. Read-only audit.

  python jax_scout/gravity_A2_predictor_coupling.py
"""
import os, sys, json, argparse, time, itertools
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout.gravity_A_rint_diagnostic import FAM, place, integ
from jax_scout.gravity_A1_predictive import evolve_coupled, momentum_x, gofrho

D = FAM["D"]


def rankdata(a):
    return np.argsort(np.argsort(a)).astype(float)


def spearman(a, b):
    ra, rb = rankdata(np.asarray(a)), rankdata(np.asarray(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def dx_field(f, kx):
    return np.real(np.fft.ifftn(1j * kx * np.fft.fftn(f)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi-iso", default="sweep_runs/PHASE_D_C2_8_TWONODE_CODEX_20260709_123611/phi_iso.npy")
    ap.add_argument("--L", type=float, default=20.0); ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=0.005); ap.add_argument("--T", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    A_ = ap.parse_args()
    out = A_.out or os.path.join(ROOT, "sweep_runs", f"GRAVITY_A2_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    p = A_.phi_iso if os.path.isabs(A_.phi_iso) else os.path.join(ROOT, A_.phi_iso)
    phi = np.load(p).astype(np.complex128); N = phi.shape[0]; L = A_.L; kap = A_.kappa
    k1 = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    kx = k1[:, None, None] * np.ones((N, N, N)); ky = k1[None, :, None]; kz = k1[None, None, :]
    ksq = kx ** 2 + ky ** 2 + kz ** 2
    ns = int(round(A_.T / A_.dt))
    print(f"=== Gravity Audit A.2 | N={N} L={L} kappa={kap} T={A_.T} ===", flush=True)
    res = {"N": N, "L": L, "kappa": kap, "T": A_.T}

    # ---------- P1: coupling faithfulness ----------
    pA = place(phi, -2.0, N, L); pB = place(phi, +1.0, N, L)
    # lambda=0 null
    A0, B0 = evolve_coupled(pA, pB, kx, ksq, 0.0, A_.dt, ns)
    p1_null = abs(momentum_x(A0, kx))
    # lambda=1: momentum conservation (equal-and-opposite) + exchange symmetry
    A1, B1 = evolve_coupled(pA, pB, kx, ksq, kap, A_.dt, ns)
    PA, PB = momentum_x(A1, kx), momentum_x(B1, kx)
    p_tot = abs(PA + PB)
    Asw, Bsw = evolve_coupled(pB, pA, kx, ksq, kap, A_.dt, ns)     # swap A<->B
    exch = abs(momentum_x(Asw, kx) - PB) / (abs(PB) + 1e-30)       # swapped-A momentum should equal original-B
    Aph, Bph = evolve_coupled(np.exp(1j * 0.7) * pA, pB, kx, ksq, kap, A_.dt, ns)
    phase_cov = abs(abs(momentum_x(Aph, kx)) - abs(PA)) / (abs(PA) + 1e-30)
    p1 = {"lambda0_null_PA": p1_null, "PA": PA, "PB": PB, "PA_plus_PB": p_tot,
          "exchange_rel_err": exch, "global_phase_cov_rel_err": phase_cov}
    p1_pass = (p1_null < 1e-6) and (p_tot < 0.05 * (abs(PA) + abs(PB) + 1e-30)) and (exch < 1e-6) and (phase_cov < 1e-6)
    print(f"[P1] lambda=0 null |P_A|={p1_null:.2e}; P_A={PA:+.3e} P_B={PB:+.3e} |P_A+P_B|={p_tot:.2e} "
          f"(equal&opposite); exchange err={exch:.2e}; phase-cov err={phase_cov:.2e} -> {'PASS' if p1_pass else 'FAIL'}", flush=True)
    res["P1_coupling_faithfulness"] = {**p1, "pass": bool(p1_pass)}

    # ---------- P2: predictor generality vs baselines ----------
    seps = [2.0, 3.0, 4.0, 5.0]; phases = [0.0, np.pi / 2, np.pi]; amps = [0.8, 1.2]
    cfgs = list(itertools.product(seps, phases, amps))
    rho_A = np.abs(pA) ** 2  # target fixed template density (target placed per-config below at -s/2 though)
    rowsP = []
    for (s, dphi, amp) in cfgs:
        tA = place(phi, -s / 2, N, L)
        tB = amp * np.exp(1j * dphi) * place(phi, +s / 2, N, L)
        rA, rB = np.abs(tA) ** 2, np.abs(tB) ** 2
        pred = {
            "sep": s,
            "neg_sep": -s,
            "overlap": integ(rA * rB, N, L),
            "cross_energy": integ(np.real(np.conj(tA) * tB), N, L),
            "grad_overlap": integ(np.abs(dx_field(tA, kx)) * np.abs(dx_field(tB, kx)), N, L),
            "I_int": integ(np.abs(kap * rB * tA) ** 2, N, L),
        }
        AT, BT = evolve_coupled(tA, tB, kx, ksq, kap, A_.dt, ns)
        absP = abs(momentum_x(AT, kx))
        rowsP.append({"sep": s, "dphi": dphi, "amp": amp, **{f"pred_{k}": v for k, v in pred.items()}, "absP_A_T": absP})
    y = np.array([r["absP_A_T"] for r in rowsP])
    predictors = ["neg_sep", "overlap", "cross_energy", "grad_overlap", "I_int"]
    sp = {pn: spearman([r[f"pred_{pn}"] for r in rowsP], y) for pn in predictors}
    # leave-one-phase-out held-out: fit rank order on 2 phases, test on the third (report worst held-out spearman)
    hold = {}
    for pn in predictors:
        vals = []
        for hp in phases:
            tr = [r for r in rowsP if r["dphi"] != hp]; te = [r for r in rowsP if r["dphi"] == hp]
            if len(te) >= 3:
                vals.append(spearman([r[f"pred_{pn}"] for r in te], [r["absP_A_T"] for r in te]))
        hold[pn] = float(np.min(vals)) if vals else float("nan")
    best = max(sp, key=lambda k: abs(sp[k]))
    p2_pass = (abs(sp["I_int"]) >= max(abs(sp[k]) for k in predictors if k != "I_int") - 1e-9)
    print(f"[P2] Spearman(predictor, |P_A(T)|) over {len(cfgs)} configs (sep×phase×amp):", flush=True)
    for pn in predictors:
        print(f"      {pn:14s} spearman={sp[pn]:+.3f}   worst-held-out-phase={hold[pn]:+.3f}", flush=True)
    print(f"[P2] best overall predictor = {best}; I_int best-or-tied = {p2_pass}", flush=True)
    res["P2_predictor_generality"] = {"spearman": sp, "held_out_worst": hold, "best": best, "I_int_best_or_tied": bool(p2_pass), "n_configs": len(cfgs)}

    # ---------- P3: force vs gradient of coupling potential ----------
    def measured_force(tA, tB):
        A1, _ = evolve_coupled(tA, tB, kx, ksq, kap, A_.dt, 20)     # short: P_A(20 dt)/ (20 dt)
        return momentum_x(A1, kx) / (20 * A_.dt)

    def predicted_force(tA, tB):
        rA = np.abs(tA) ** 2; rB = np.abs(tB) ** 2
        return -kap * integ(rA * dx_field(rB, kx), N, L)            # F = -∫ rho_A ∇(kappa rho_B)

    p3 = {}
    # offset env (asymmetric) -> nonzero force, predicted ~ measured in sign
    tA = place(phi, -2.0, N, L); tB = place(phi, +1.0, N, L)
    fm, fp = measured_force(tA, tB), predicted_force(tA, tB)
    p3["offset"] = {"F_measured": fm, "F_predicted": fp, "same_sign": bool(fm * fp > 0)}
    # centred/symmetric env straddling A (two half-blobs equidistant) -> net force ~0
    tBs = place(phi, -2.0 - 2.5, N, L) + place(phi, -2.0 + 2.5, N, L)
    fm_c, fp_c = measured_force(tA, tBs), predicted_force(tA, tBs)
    p3["centred_symmetric"] = {"F_measured": fm_c, "F_predicted": fp_c}
    # reflected env -> force reverses
    tBr = place(phi, -5.0, N, L)                                    # env on the OTHER side of A(-2): at -5
    fm_r, fp_r = measured_force(tA, tBr), predicted_force(tA, tBr)
    p3["reflected"] = {"F_measured": fm_r, "F_predicted": fp_r, "reversed_vs_offset": bool(fm_r * fm < 0)}
    p3_pass = p3["offset"]["same_sign"] and (abs(fm_c) < 0.3 * abs(fm)) and p3["reflected"]["reversed_vs_offset"]
    print(f"[P3] offset: F_meas={fm:+.3e} F_pred={fp:+.3e} same-sign={p3['offset']['same_sign']}", flush=True)
    print(f"[P3] centred-symmetric: F_meas={fm_c:+.3e} (should be ~0 vs |{fm:.2e}|)", flush=True)
    print(f"[P3] reflected: F_meas={fm_r:+.3e} (should reverse sign vs offset)  -> {'PASS' if p3_pass else 'PARTIAL'}", flush=True)
    res["P3_force_gradient"] = {**p3, "pass": bool(p3_pass)}

    verdict = ("A2_PASS" if (p1_pass and p2_pass and p3_pass) else "A2_PARTIAL")
    res["verdict"] = verdict
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)
    print(f"\n=== {verdict} | P1_coupling={p1_pass} P2_I_int_best={p2_pass}(best={best}) P3_gradient_force={p3_pass} | out={out} ===", flush=True)


if __name__ == "__main__":
    main()
