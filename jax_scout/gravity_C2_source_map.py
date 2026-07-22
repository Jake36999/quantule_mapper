"""Gravity Audit C.2 — source-discrimination map + REAL source-layer B-O emergence test.

Fixes two C.1 overclaims:
  (1) B-O emergence must be SOURCE-LAYER: each clock generates its OWN N_{A|E} from its OWN interaction; check whether
      two equally-normalized compact clocks (different widths) produce CONVERGENT <N> as width -> 0. (C.1 supplied one
      shared N -> only clock-response consistency.)
  (2) The I_int~overlap "0.2%" was one config. Map Dr(x) = r_Iint(x) - r_overlap(x) across positions x widths x
      environment profiles, with sources matched by background(N=1) + INTEGRATED DEFICIT, to see whether the
      degeneracy is GENERAL or config-specific.

Clock rate = profile-weighted <N> over the compact envelope (frozen). Consistency/organization map only; NOT time
dilation/gravity. Standalone numpy; production gravity ladder CLOSED.

  python jax_scout/gravity_C2_source_map.py
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
    out = A.out or os.path.join(ROOT, "sweep_runs", f"GRAVITY_C2_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    p = A.phi_iso if os.path.isabs(A.phi_iso) else os.path.join(ROOT, A.phi_iso)
    phi = np.load(p).astype(np.complex128); N = phi.shape[0]; L = A.L
    x = np.linspace(-L / 2, L / 2, N, endpoint=False); X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    gauss = lambda x0, s: np.exp(-((X - x0) ** 2 + Y ** 2 + Z ** 2) / (2 * s ** 2))

    # two environment profiles (a soliton, and a broad Gaussian blob)
    envs = {"soliton": np.abs(place(phi, 0.0, N, L)) ** 2, "blob": gauss(0.0, 2.0) ** 2}
    deficit = lambda Nf: float(integ(1.0 - Nf, N, L))

    def beta_for_deficit(field, target):
        lo, hi = 0.0, 1e7
        for _ in range(70):
            mid = 0.5 * (lo + hi)
            if deficit(1.0 / (1.0 + mid * field)) < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def clock_rate(Nf, rc):
        return float(integ(rc * Nf, N, L) / (integ(rc, N, L) + 1e-30))

    res = {"N": N, "L": L, "emergence": {}, "map": []}

    # ---------- (1) REAL source-layer emergence: own-N convergence as width -> 0 ----------
    rE = envs["soliton"]; xc = 1.2
    beta = 5.0
    em_rows = []
    for w in (1.6, 1.2, 0.8, 0.5, 0.3):
        rc = gauss(xc, w) ** 2; rc = rc / (integ(rc, N, L) + 1e-30)     # equally NORMALIZED (∫rho=1)
        I_hat = (KAP ** 2) * rc * rE ** 2                               # this clock's OWN relational load field
        Nown = 1.0 / (1.0 + beta * I_hat)
        em_rows.append({"width": w, "own_clock_rate": clock_rate(Nown, rc), "N_at_center": float(1.0 / (1.0 + beta * I_hat[np.argmin((x-xc)**2), N//2, N//2]))})
        print(f"[emergence] width={w:.2f}: own <N>={em_rows[-1]['own_clock_rate']:.5f}", flush=True)
    rates = [r["own_clock_rate"] for r in em_rows]
    # convergence: successive differences shrink as width shrinks (probes converge to a common point value)
    conv = all(abs(rates[i+1] - rates[i]) <= abs(rates[i] - rates[i-1]) + 1e-9 for i in range(1, len(rates)-1))
    spread = max(rates) - min(rates)
    res["emergence"] = {"rows": em_rows, "converging_toward_point_value": bool(conv), "rate_spread": spread,
                        "note": "own-N per width; as width->0 each clock samples N at the point -> convergent = B-O emerges in compact limit"}
    print(f"[emergence] own-N spread over widths = {spread:.4f}; monotone-converging = {conv} "
          f"(compact-limit convergence => objective lapse emerges from relational source)", flush=True)

    # ---------- (2) source-discrimination map Dr(x) across position x width x env, matched deficit ----------
    ref = gauss(1.5, 0.8) ** 2
    max_dr = 0.0
    for ename, rEe in envs.items():
        I_star = float(np.max((KAP ** 2) * ref * rEe ** 2)) + 1e-30
        O_star = float(np.max(ref * rEe)) + 1e-30
        for w in (0.8, 0.4):
            for xc in (0.6, 1.2, 2.0, 3.0):
                rc = gauss(xc, w) ** 2; rc = rc / (integ(rc, N, L) + 1e-30)
                If = (KAP ** 2) * rc * rEe ** 2 / I_star; Of = rc * rEe / O_star
                # match by integrated deficit to a fixed budget
                budget = 0.02
                NI = 1.0 / (1.0 + beta_for_deficit(If, budget) * If)
                NO = 1.0 / (1.0 + beta_for_deficit(Of, budget) * Of)
                dr = abs(clock_rate(NI, rc) - clock_rate(NO, rc))
                max_dr = max(max_dr, dr)
                res["map"].append({"env": ename, "width": w, "xc": xc, "delta_r_Iint_overlap": dr})
    for m in res["map"]:
        print(f"[map] env={m['env']:8s} w={m['width']:.1f} xc={m['xc']:.1f}: |Dr(I_int,overlap)|={m['delta_r_Iint_overlap']:.4f}", flush=True)
    if max_dr < 2e-3:
        concl = "GENERAL_DEGENERACY: overlap is an adequate reduced source (I_int adds no clock-level structure here)"
    elif max_dr < 2e-2:
        concl = "WEAK_LOCAL_DISTINCTION: I_int differs from overlap only modestly / in narrow regions"
    else:
        concl = "I_int_CARRIES_CLOCK_LEVEL_STRUCTURE beyond overlap across profiles"
    res["max_delta_r"] = max_dr; res["map_conclusion"] = concl
    verdict = "C2_SOURCE_MAP_DONE"
    res["verdict"] = verdict
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)
    print(f"\n[map] max |Dr| across all configs = {max_dr:.4f} -> {concl}", flush=True)
    print(f"=== {verdict} | out={out} ===", flush=True)


if __name__ == "__main__":
    main()
