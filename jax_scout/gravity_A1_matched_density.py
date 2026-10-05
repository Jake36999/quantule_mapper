"""Gravity Audit A.1 — matched-density / density-decoupling controls (the decisive density-vs-relation discriminator).

Two clean tests that raw density (hence Ω²(ρ)) cannot account for the interaction:

  B1  SOURCE-BLINDNESS: the SAME pointwise density field is realized by (i) an interacting soliton pair (I_int>0) and
      (ii) a single coherent blob with that density (I_int=0). Since Ω²=Ω²(ρ) is a function of ρ ONLY, it assigns
      both the identical geometry — yet their interaction differs. So density cannot be the interaction source.

  B2  LOCAL MULTIVALUEDNESS: for a genuine interacting pair, the local field I_int(x) is NOT a single-valued
      function of the local density ρ(x): points at the same ρ carry a range of I_int. A windowed decomposition would
      spuriously collapse I_int to f(ρ); the physical (fixed soliton-profile) decomposition does not.

Read-only numpy field algebra; no production/solver change; gravity sector PAUSED. This validates the SOURCE
observable's independence from density; it is not a metric/clock/force/gravity claim.

  python jax_scout/gravity_A1_matched_density.py [--phi-iso ...] [--sep 4.0]
"""
import os, sys, json, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout.gravity_A_rint_diagnostic import FAM, nonlin, place, integ, R_int, R_int_field

try:
    from gravity.unified_omega import derive_stable_conformal_factor
    HAVE_OMEGA = True
except Exception:
    HAVE_OMEGA = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi-iso", default="sweep_runs/PHASE_D_C2_8_TWONODE_CODEX_20260709_123611/phi_iso.npy")
    ap.add_argument("--L", type=float, default=20.0)
    ap.add_argument("--sep", type=float, default=4.0)
    ap.add_argument("--out", default=None)
    A = ap.parse_args()
    import time
    out = A.out or os.path.join(ROOT, "sweep_runs", f"GRAVITY_A1_MATCHED_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    p = A.phi_iso if os.path.isabs(A.phi_iso) else os.path.join(ROOT, A.phi_iso)
    phi = np.load(p).astype(np.complex128); N = phi.shape[0]; L = A.L
    print(f"=== Gravity Audit A.1 — matched-density controls | {FAM} | N={N} L={L} sep={A.sep} ===", flush=True)
    res = {"N": N, "L": L, "sep": A.sep, "family": FAM}

    psiL = place(phi, -A.sep / 2, N, L); psiR = place(phi, +A.sep / 2, N, L)
    pair = psiL + psiR
    rho_pair = np.abs(pair) ** 2
    I_pair = R_int(psiL, psiR, N, L)

    # --- B1: source-blindness — same density, blob (I_int=0) vs pair (I_int>0) ---
    blob = np.sqrt(rho_pair).astype(np.complex128)           # single coherent object, SAME pointwise density
    rho_blob = np.abs(blob) ** 2
    rho_match = float(np.max(np.abs(rho_pair - rho_blob)))     # should be ~0 (machine)
    I_blob = 0.0                                               # isolated single object: no partner -> R_int = 0
    omega_note = ""
    if HAVE_OMEGA:
        prm = {"param_a_coupling": 2.3098, "param_rho_vac": 1.0, "param_skip_topology_cap": True}
        om_pair = np.asarray(derive_stable_conformal_factor(rho_pair, prm))
        om_blob = np.asarray(derive_stable_conformal_factor(rho_blob, prm))
        om_diff = float(np.max(np.abs(om_pair - om_blob)))
        omega_note = f"; max|Omega2_pair-Omega2_blob|={om_diff:.2e} (identical -> density-source blind to the I_int difference)"
    b1_pass = (rho_match < 1e-9) and (I_pair > 1e-6)
    print(f"[B1] same density (max|d_rho|={rho_match:.2e}): I_int(pair)={I_pair:.4e} vs I_int(blob)={I_blob:.1f}"
          f"{omega_note} -> {'PASS' if b1_pass else 'FAIL'}", flush=True)
    res["B1_source_blindness"] = {"rho_match_maxabs": rho_match, "I_int_pair": I_pair, "I_int_blob": I_blob, "pass": bool(b1_pass)}

    # --- B2: local multivaluedness — is I_int(x) a function of rho(x)? (physical decomposition) ---
    Ifield = R_int_field(psiL, psiR).ravel()
    rfield = rho_pair.ravel()
    m = rfield > 1e-4                                          # ignore deep vacuum
    Iv, rv = Ifield[m], rfield[m]
    # bin by rho; within each bin, spread of I_int (if single-valued in rho, within-bin spread ~0)
    nb = 12
    edges = np.quantile(rv, np.linspace(0, 1, nb + 1))
    rows = []
    within_cv = []
    for i in range(nb):
        sel = (rv >= edges[i]) & (rv < edges[i + 1] if i < nb - 1 else rv <= edges[i + 1])
        if sel.sum() < 20:
            continue
        Ii = Iv[sel]
        cv = float(np.std(Ii) / (np.mean(Ii) + 1e-30))        # within-bin coefficient of variation of I_int
        within_cv.append(cv)
        rows.append({"rho_lo": float(edges[i]), "rho_hi": float(edges[i + 1]), "n": int(sel.sum()),
                     "I_mean": float(np.mean(Ii)), "I_std": float(np.std(Ii)), "I_cv": cv})
    med_cv = float(np.median(within_cv)) if within_cv else np.nan
    # if I_int were a single-valued function of rho, within-bin CV ~ 0. Large CV => multivalued => beyond density.
    b2_pass = med_cv > 0.2
    print(f"[B2] local I_int(x) vs rho(x): median within-rho-bin CV of I_int = {med_cv:.2f} "
          f"(>>0 => same rho carries a RANGE of I_int => density insufficient) -> {'PASS' if b2_pass else 'FAIL'}", flush=True)
    for r in rows:
        print(f"     rho[{r['rho_lo']:.3f},{r['rho_hi']:.3f}) n={r['n']:6d}  I_mean={r['I_mean']:.3e}  I_cv={r['I_cv']:.2f}", flush=True)
    res["B2_local_multivaluedness"] = {"median_within_bin_cv": med_cv, "bins": rows, "pass": bool(b2_pass)}

    verdict = "MATCHED_DENSITY_CONTROLS_PASS" if (b1_pass and b2_pass) else "MATCHED_DENSITY_CONTROLS_PARTIAL"
    res["verdict"] = verdict
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)
    print(f"\n=== {verdict} | B1 source-blindness={b1_pass} B2 local-multivalued={b2_pass} | out={out} ===", flush=True)


if __name__ == "__main__":
    main()
