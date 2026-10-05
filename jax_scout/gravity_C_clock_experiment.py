"""Gravity Audit C — first clock experiment (throttling-lapse internal-consistency + source-distinguishability).

Tests the H-G- throttling lapse N=1/(1+beta*I_hat) as a CLOCK-RATE modulator, with the reviewer's guards built in:
  * GLOBAL preregistered scale I_* (fixed reference config; NOT per-run max/mean) so beta is comparable and the
    effect is not normalized into existence;
  * WEAK-FIELD beta ladder chosen by target N_min in {1.0(null), 0.95, 0.8, 0.5} (no 1000x slowdowns);
  * MATCHED-STRENGTH overlap control: I_int-sourced and overlap-sourced lapse fields built to the SAME N_min, so any
    clock difference is spatial ORGANIZATION, not field strength;
  * OBJECTIVE vs RELATIONAL fork: I_int contains the probe density (rho_A rho_B^2), so different clocks see different
    N -> we measure BOTH clocks and report whether they slow universally (objective) or probe-specifically (relational).

Clock model (static effective rate): a clock probe of density rho_c ticks at the profile-weighted local lapse
    r = <N>_c = (int rho_c N dV)/(int rho_c dV),   fractional slowing = 1 - r.
This is the algebraic consequence of the lapse (a first consistency/organization check, NOT a proof of time
dilation; H-G- remains a postulate). Dynamical proper-time-coupled clocks are a later step.

Standalone numpy; no production/solver change; gravity ladder CLOSED. Design-only, gravity PAUSED.

  python jax_scout/gravity_C_clock_experiment.py
"""
import os, sys, json, argparse, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout.gravity_A_rint_diagnostic import FAM, place, integ

KAP = 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi-iso", default="sweep_runs/PHASE_D_C2_8_TWONODE_CODEX_20260709_123611/phi_iso.npy")
    ap.add_argument("--L", type=float, default=20.0); ap.add_argument("--out", default=None)
    A = ap.parse_args()
    out = A.out or os.path.join(ROOT, "sweep_runs", f"GRAVITY_C_CLOCK_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    p = A.phi_iso if os.path.isabs(A.phi_iso) else os.path.join(ROOT, A.phi_iso)
    phi = np.load(p).astype(np.complex128); N = phi.shape[0]; L = A.L
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    def gauss(x0, sig):
        return np.exp(-((X - x0) ** 2 + Y ** 2 + Z ** 2) / (2 * sig ** 2))

    env = place(phi, 0.0, N, L); rB = np.abs(env) ** 2          # environment/load at origin
    clk_narrow = gauss(0.0, 0.8); clk_wide = gauss(0.0, 1.6)    # two structurally different clock morphologies

    def cavg(field, rho_c):
        return float(integ(rho_c * field, N, L) / (integ(rho_c, N, L) + 1e-30))

    # ---- GLOBAL preregistered scales from a NAMED reference config (narrow clock centred at x=1.5 in env) ----
    ref_c = gauss(1.5, 0.8)
    I_star = float(np.max((KAP ** 2) * ref_c * rB ** 2))         # fixed global load scale
    S_star = float(np.max(rB ** 2))                              # fixed global objective (env-only) scale
    O_star = float(np.max(ref_c * rB))                           # fixed global overlap scale
    print(f"=== Gravity Audit C — clock experiment | N={N} L={L} | I*={I_star:.4e} S*={S_star:.4e} O*={O_star:.4e} ===", flush=True)
    res = {"N": N, "L": L, "I_star": I_star, "S_star": S_star, "O_star": O_star, "ladder": []}

    # clock centred at x_c near the load (peak overlap with env); reference far clock ~ N=1
    xc = 1.2
    cN = gauss(xc, 0.8); cW = gauss(xc, 1.6)
    I_field_N = (KAP ** 2) * cN * rB ** 2                        # relational load fields (contain the probe)
    I_field_W = (KAP ** 2) * cW * rB ** 2
    S_field = rB ** 2                                            # objective (env-only) field
    O_field_N = cN * rB                                          # plain overlap fields
    targets = [1.0, 0.95, 0.8, 0.5]                              # weak-field N_min ladder
    for t in targets:
        # choose beta so the RELATIONAL narrow-clock field hits N_min=t at its peak (weak-field calibration)
        peakI = float(np.max(I_field_N)) / I_star
        beta = (1.0 / t - 1.0) / (peakI + 1e-30)
        # relational clocks (each from its OWN I_int) -> objective vs relational fork
        NrelN = 1.0 / (1.0 + beta * I_field_N / I_star)
        NrelW = 1.0 / (1.0 + beta * I_field_W / I_star)
        r_relN, r_relW = cavg(NrelN, cN), cavg(NrelW, cW)
        # objective lapse (common env-only field) -> both clocks read the SAME N
        betaS = (1.0 / t - 1.0) / (float(np.max(S_field)) / S_star + 1e-30)
        Nobj = 1.0 / (1.0 + betaS * S_field / S_star)
        r_objN, r_objW = cavg(Nobj, cN), cavg(Nobj, cW)
        # MATCHED-STRENGTH overlap control: same N_min as the relational narrow field
        betaO = (1.0 / t - 1.0) / (float(np.max(O_field_N)) / O_star + 1e-30)
        Nov = 1.0 / (1.0 + betaO * O_field_N / O_star)
        r_ov = cavg(Nov, cN)
        row = {"target_Nmin": t, "beta": beta,
               "relational_narrow_rate": r_relN, "relational_wide_rate": r_relW,
               "objective_narrow_rate": r_objN, "objective_wide_rate": r_objW,
               "matched_overlap_narrow_rate": r_ov,
               "rel_narrow_vs_wide_gap": abs(r_relN - r_relW),
               "obj_narrow_vs_wide_gap": abs(r_objN - r_objW),
               "Iint_vs_matched_overlap_gap": abs(r_relN - r_ov)}
        res["ladder"].append(row)
        print(f"[Nmin={t:.2f} beta={beta:.3f}] relational: narrow={r_relN:.4f} wide={r_relW:.4f} (gap {abs(r_relN-r_relW):.4f}); "
              f"objective: narrow={r_objN:.4f} wide={r_objW:.4f} (gap {abs(r_objN-r_objW):.4f}); "
              f"I_int={r_relN:.4f} vs matched-overlap={r_ov:.4f} (gap {abs(r_relN-r_ov):.4f})", flush=True)

    L0 = res["ladder"][0]
    null_ok = abs(L0["relational_narrow_rate"] - 1.0) < 1e-9 and abs(L0["objective_narrow_rate"] - 1.0) < 1e-9
    monotone = all(res["ladder"][i]["relational_narrow_rate"] > res["ladder"][i + 1]["relational_narrow_rate"] for i in range(len(targets) - 1))
    relational_signature = res["ladder"][-1]["rel_narrow_vs_wide_gap"] > 10 * res["ladder"][-1]["obj_narrow_vs_wide_gap"]
    overlap_distinct = res["ladder"][-1]["Iint_vs_matched_overlap_gap"] > 1e-3
    res["findings"] = {"beta0_null": bool(null_ok), "monotone_with_beta": bool(monotone),
                       "relational_gt_objective_clock_gap": bool(relational_signature),
                       "Iint_distinct_from_matched_overlap": bool(overlap_distinct)}
    verdict = "C_CLOCK_FIRST_PASS" if (null_ok and monotone) else "C_CLOCK_PARTIAL"
    res["verdict"] = verdict
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)
    print(f"\n[findings] beta0-null={null_ok}; monotone={monotone}; "
          f"relational(probe-specific) clock gap >> objective gap: {relational_signature} "
          f"(-> lapse is RELATIONAL under I_int, so 'clock universality' does NOT apply — the B-R branch); "
          f"I_int distinct from MATCHED overlap: {overlap_distinct} (gap {res['ladder'][-1]['Iint_vs_matched_overlap_gap']:.4f})", flush=True)
    print(f"=== {verdict} | consistency check; NOT a time-dilation proof (H-G- is a postulate under test) | out={out} ===", flush=True)


if __name__ == "__main__":
    main()
