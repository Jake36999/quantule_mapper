"""Gravity ladder — DE-SATURATION PILOT (Option 1, diagnostic candidate geometry, NOT accepted physics).
Post-processing only: recompute the geometry (Omega^2) response of the SAVED Rung A+D load under (a) the production
soft-clip vs (b) Codex's de-saturated candidate soft-clip (beta~0.75, wider window). Question: does a de-saturated
(graded, cap_fraction=0) conformal soft-clip turn the load's Omega^2 response from a SATURATION CLIFF into a
distance-graded WELL (potential-like)? If yes, a de-saturated geometry is a viable candidate contract for rung B;
if not, the blocker is deeper. No evolution, no default change, no gravity claim.

  wsl:  python jax_scout/gravity_desaturation_pilot.py --load sweep_runs/GRAVITY_AD_bg0/load_relaxed.npy
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jax_scout import core_saturation_search as css
import gravity.unified_omega as uo

L = css.L_


def geom_variants():
    base = {"param_rho_vac": float(css.FEB.get("param_rho_vac", 1.1866)),
            "param_a_coupling": float(css.FEB.get("param_a_coupling", 2.3098)),
            "param_skip_topology_cap": True}
    return {
        "production": {**base, "param_conformal_softclip_beta": 3.0,
                       "param_omega_sq_min": 1e-9, "param_omega_sq_max": 1e6},
        "desat_b0.75_w1e8": {**base, "param_conformal_softclip_beta": 0.75,
                             "param_omega_sq_min": 1e-6, "param_omega_sq_max": 1e8},
    }


def radial(field, N, center, nbins=24):
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    d = [((x - x[center[a]]) - L * np.round((x - x[center[a]]) / L)) for a in range(3)]
    R = np.sqrt(d[0][:, None, None] ** 2 + d[1][None, :, None] ** 2 + d[2][None, None, :] ** 2)
    edges = np.linspace(0, L / 2, nbins + 1); rc, pr = [], []
    for i in range(nbins):
        m = (R >= edges[i]) & (R < edges[i + 1])
        if m.any():
            rc.append(0.5 * (edges[i] + edges[i + 1])); pr.append(float(field[m].mean()))
    return np.array(rc), np.array(pr)


def characterize(rho, params, N):
    om = np.asarray(uo.derive_stable_conformal_factor(rho, params))
    cen = np.unravel_index(int(rho.argmax()), rho.shape)
    rc, omp = radial(om, N, cen)
    om_far = float(np.mean(omp[-3:])); om_ctr = float(om[cen])
    cap = float(np.mean((om >= 0.99 * params["param_omega_sq_max"]) | (om <= 1.01 * params["param_omega_sq_min"])))
    # graded well test: is the radial profile monotone-ish and smooth (no >5x single-bin jump)?
    ratios = omp[1:] / np.maximum(omp[:-1], 1e-30)
    max_jump = float(np.max(np.maximum(ratios, 1.0 / np.maximum(ratios, 1e-30))))
    # falloff of |omp-om_far| e-folding
    dprof = np.abs(omp - om_far); fall = np.nan
    if dprof[0] > 1e-3 * (abs(om_far) + 1e-30):
        norm = dprof / (dprof[0] + 1e-30); g = norm > 1e-3
        if g.sum() >= 4:
            c = np.polyfit(rc[g], np.log(norm[g] + 1e-30), 1); fall = float(-1 / c[0]) if c[0] < 0 else np.inf
    graded = cap < 0.02 and max_jump < 3.0
    return {"omega_center": om_ctr, "omega_far": om_far, "well": om_far - om_ctr,
            "cap_fraction": cap, "max_radial_jump": max_jump, "falloff_len": fall,
            "graded_well": bool(graded), "radial_r": [round(v, 2) for v in rc],
            "radial_omega": [round(v, 3) for v in omp]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", default="sweep_runs/GRAVITY_AD_bg0/load_relaxed.npy")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    psi = np.load(os.path.join(root, a.load) if not os.path.isabs(a.load) else a.load)
    rho = np.abs(psi) ** 2; N = rho.shape[0]
    out = a.out or os.path.join(root, "sweep_runs", "GRAVITY_DESAT_PILOT")
    os.makedirs(out, exist_ok=True)
    print(f"=== GRAVITY DE-SATURATION PILOT | load {a.load} N={N} rho[{rho.min():.2e},{rho.max():.2f}] ===", flush=True)
    res = {}
    for name, params in geom_variants().items():
        c = characterize(rho, params, N)
        res[name] = c
        print(f"\n[{name}] beta={params['param_conformal_softclip_beta']} window=[{params['param_omega_sq_min']:.0e},"
              f"{params['param_omega_sq_max']:.0e}]", flush=True)
        print(f"    Omega^2: center={c['omega_center']:.3f} far={c['omega_far']:.3f} well={c['well']:.3f} "
              f"cap_frac={c['cap_fraction']:.3f} max_radial_jump={c['max_radial_jump']:.1f} "
              f"falloff_len={c['falloff_len']} graded_well={c['graded_well']}", flush=True)
        print(f"    radial Omega^2: " + " ".join(f"{r:.1f}:{o:.2f}" for r, o in
              zip(c['radial_r'][::3], c['radial_omega'][::3])), flush=True)
    prod, desat = res["production"], res["desat_b0.75_w1e8"]
    verdict = ("DESAT_GIVES_GRADED_WELL" if desat["graded_well"] and not prod["graded_well"] else
               "DESAT_STILL_NOT_GRADED" if not desat["graded_well"] else
               "PRODUCTION_ALREADY_GRADED")
    res["verdict"] = verdict
    print(f"\n=== {verdict} ===", flush=True)
    json.dump(res, open(os.path.join(out, "desat_pilot.json"), "w"), indent=2, default=float)
    print(f"DESAT_PILOT_DONE {out}", flush=True)


if __name__ == "__main__":
    main()
