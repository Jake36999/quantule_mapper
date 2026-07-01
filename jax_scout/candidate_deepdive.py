"""
Candidate deep-dive: mutual-support test (ablation + isolated baseline) + passive
emergent-geometry/SDG diagnostic on the top promotable multiseed stable_multinode
candidates.

For each candidate (params from the multiseed sweep):
  INTACT    : 6-seed IC (== sweep multiseed IC)
  ABLATION  : 5-seed IC (one node removed)         -> do the others destabilize?
  ISOLATED  : 1-seed IC (lone node, same params)    -> does a lone node persist as well?
Mutual support  = cluster persists AND (lone node decays OR ablation disrupts the rest).
Geometry verdict = geometry_diag (IRER-SDG-DIAG-v1, passive) on the intact field.
Promote-to-CuPy = mutual support AND geometry follows RD with bounded curvature.

Run in WSL2 jax venv:
  python /mnt/f/quantule_mapper/jax_scout/candidate_deepdive.py --k 6 --steps 800
"""
import os, sys, csv, json, glob, argparse, time
import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import physics, geometry_diag as gd, stable_collapse as scl

SEED, K_SEEDS = 20260619, 6


def seed_ic_components(N, L):
    rng = np.random.default_rng(SEED)
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    w = L / 12.0
    bumps = []
    for _ in range(K_SEEDS):
        cx, cy, cz = rng.uniform(-L / 2, L / 2, 3)
        bumps.append(np.exp(-((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2) / (2 * w ** 2)))
    noise = 0.01 * (rng.standard_normal((N, N, N)) + 1j * rng.standard_normal((N, N, N)))
    return bumps, noise


def build_ic(bumps, k, noise):
    psi0 = np.zeros(bumps[0].shape, np.complex128)
    for b in bumps[:k]:
        psi0 = psi0 + b
    return (psi0 + noise).astype(np.complex128)


def run(pvec, ic, N, L, dt, steps):
    pm, pf, en, am, fin = physics.probe_one(jnp.asarray(pvec), jnp.asarray(ic), N, L, dt, steps,
                                            jnp.float64, jnp.complex128)
    pm, pf, en, am = np.asarray(pm), np.asarray(pf), np.asarray(en), np.asarray(am)
    e0 = float(en[0]) if en[0] > 0 else 1e-30
    return {"psi_mid": pm, "psi_final": pf, "er": float(en[-1] / e0), "amp_max": float(am.max()),
            "finite": bool(fin), "nodes": scl.node_count(np.abs(pf) ** 2), "coh": scl.phase_coherence(pf)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--N", type=int, default=48)
    ap.add_argument("--L", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--steps", type=int, default=800)
    args = ap.parse_args()
    dx = args.L / args.N

    sd = sorted(glob.glob(os.path.join(ROOT, "sweep_runs", "STABLE_COLLAPSE_multiseed_*")))[-1]
    rows = [r for r in csv.DictReader(open(os.path.join(sd, "stable_collapse_results.csv")))
            if len(r) > 5]
    prom = [r for r in rows if float(r["coherence"]) > 0.2 and int(r["nodes_final"]) >= 2
            and 0.3 <= float(r["energy_ratio"]) <= 5.0]
    prom.sort(key=lambda r: -float(r["coherence"]))
    panel = prom[: args.k]
    order = physics.SWEEP_PARAM_ORDER
    bumps, noise = seed_ic_components(args.N, args.L)

    outdir = os.path.join(sd, "deepdive")
    os.makedirs(outdir, exist_ok=True)
    print(f"DEEP-DIVE: {len(panel)} candidates from {os.path.basename(sd)}  "
          f"(N={args.N}, steps={args.steps}, diag={gd.DIAG_CONTRACT_VERSION})\n")

    results, t0 = [], time.time()
    for r in panel:
        params = {k: float(r[k]) for k in order}
        pvec = [params[k] for k in order]
        intact = run(pvec, build_ic(bumps, 6, noise), args.N, args.L, args.dt, args.steps)
        ablated = run(pvec, build_ic(bumps, 5, noise), args.N, args.L, args.dt, args.steps)
        isolated = run(pvec, build_ic(bumps, 1, noise), args.N, args.L, args.dt, args.steps)

        dF = gd.diagnose(intact["psi_final"], params, dx)
        dM = gd.diagnose(intact["psi_mid"], params, dx)
        geo = gd.geometry_verdict(dF)

        cluster_persists = intact["finite"] and intact["nodes"] >= 2 and 0.3 <= intact["er"] <= 5.0
        isolated_decays = (isolated["er"] < 0.3) or (isolated["nodes"] == 0)
        ablation_disrupts = (ablated["nodes"] < intact["nodes"] - 1) or (ablated["er"] < 0.5 * intact["er"])
        mutual = bool(cluster_persists and (isolated_decays or ablation_disrupts))
        curv_stable = abs(dF["curvature_l2"] - dM["curvature_l2"]) / (dM["curvature_l2"] + 1e-12) < 2.0
        promote = bool(mutual and geo == "geometry_follows_RD_bounded" and curv_stable)

        rec = {"idx": r["idx"], **{k: round(params[k], 4) for k in order},
               "intact_nodes": intact["nodes"], "intact_er": round(intact["er"], 3), "intact_coh": round(intact["coh"], 3),
               "ablated_nodes": ablated["nodes"], "ablated_er": round(ablated["er"], 3),
               "isolated_nodes": isolated["nodes"], "isolated_er": round(isolated["er"], 3),
               "isolated_decays": isolated_decays, "ablation_disrupts": ablation_disrupts,
               "mutual_support": mutual, "geometry_verdict": geo, "curv_stable": curv_stable,
               "curvature_max": round(dF["curvature_max"], 4), "curvature_node_corr": round(dF["curvature_node_correlation"], 3),
               "phase_coh_nodes": round(dF["phase_coherence_nodes"], 3) if dF["phase_coherence_nodes"] == dF["phase_coherence_nodes"] else None,
               "current_circ": round(dF["current_circulation_l2"], 4), "sdg_h_norm": round(dF["sdg_h_norm_l2"], 1),
               "node_omega_contrast": round(dF["node_omega_contrast"], 4),
               "PROMOTE_TO_CUPY": promote}
        results.append(rec)
        json.dump({"params": params, "diag_final": dF, "diag_mid": dM, "verdicts": {
            "mutual_support": mutual, "geometry": geo, "promote": promote},
            "diag_contract_version": gd.DIAG_CONTRACT_VERSION},
            open(os.path.join(outdir, f"deepdive_{r['idx']}.json"), "w"), indent=2)
        print(f"[{r['idx']}] intact(n={intact['nodes']},er={intact['er']:.2f}) "
              f"abl(n={ablated['nodes']},er={ablated['er']:.2f}) iso(n={isolated['nodes']},er={isolated['er']:.2f}) "
              f"| mutual={mutual} geo={geo} curv_max={dF['curvature_max']:.3f} -> PROMOTE={promote}", flush=True)

    with open(os.path.join(outdir, "deepdive_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys())); w.writeheader(); w.writerows(results)
    npro = sum(1 for r in results if r["PROMOTE_TO_CUPY"])
    nmut = sum(1 for r in results if r["mutual_support"])
    print(f"\nDONE {len(panel)} candidates in {(time.time()-t0)/60:.1f} min")
    print(f"mutual_support: {nmut}/{len(panel)}   PROMOTE_TO_CUPY: {npro}/{len(panel)}")
    print(f"-> {outdir}/deepdive_summary.csv")


if __name__ == "__main__":
    main()
