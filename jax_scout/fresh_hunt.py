"""
Fresh corrected-physics mutual-support hunt (NOT legacy-seeded, NOT prime-SSE-ranked).

Search target: bounded, persistent, MUTUALLY-SUPPORTING node configurations where the
cluster survives better than isolated nodes, ablation disrupts the remaining structure,
and geometry stays bounded. Headline = stable-collapse / mutual-support; prime-SSE is
NOT used here.

Method: LHS over the full parameter space (gain_bounds: D,eta,rho_vac,omega0,a_coupling,
s,f,a). For each config, evolve 4 shared multiseed ICs in batched vmap sweeps:
  intact(6 seeds) / isolated(1) / ablation(5) / phase_scrambled(6 random-phase).
Then compute per-config mutual-support metrics + passive SDG geometry diagnostic.

JAX scout first; CuPy only for finalists that clear the escalation gate.

Run in WSL2 jax venv:
  python /mnt/f/quantule_mapper/jax_scout/fresh_hunt.py --size 64 --batch 16 --N 48 --steps 800
"""
import os, sys, csv, json, time, argparse
import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import physics, geometry_diag as gd, stable_collapse as scl

SEED, K_SEEDS = 20260619, 6
CURV_BOUND = 1.0   # curvature_max below this = bounded (validated condensate ref ~0.03)


def seed_components(N, L):
    rng = np.random.default_rng(SEED)
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    w = L / 12.0
    bumps, phases = [], []
    for _ in range(K_SEEDS):
        cx, cy, cz = rng.uniform(-L / 2, L / 2, 3)
        bumps.append(np.exp(-((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2) / (2 * w ** 2)))
        phases.append(rng.uniform(0, 2 * np.pi))
    noise = 0.01 * (rng.standard_normal((N, N, N)) + 1j * rng.standard_normal((N, N, N)))
    return bumps, phases, noise


def build_ic(bumps, phases, noise, kind):
    psi = np.zeros(bumps[0].shape, np.complex128)
    if kind == "intact":
        for b in bumps: psi += b
    elif kind == "ablation":
        for b in bumps[:-1]: psi += b
    elif kind == "isolated":
        psi += bumps[0]
    elif kind == "phase_scrambled":
        for b, ph in zip(bumps, phases): psi += b * np.exp(1j * ph)
    return (psi + noise).astype(np.complex128)


def batch_run(params, psi0, N, L, dt, steps):
    pm, pf, en, am, fin = physics.sweep_probe(jnp.asarray(params), jnp.asarray(psi0), N, L, dt, steps,
                                              jnp.float64, jnp.complex128)
    return np.asarray(pm), np.asarray(pf), np.asarray(en), np.asarray(am), np.asarray(fin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--N", type=int, default=48)
    ap.add_argument("--L", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--bounds-file", default="/mnt/f/quantule_mapper/jax_scout/gain_bounds.json")
    ap.add_argument("--outdir", default="/mnt/f/quantule_mapper/sweep_runs")
    args = ap.parse_args()
    dx = args.L / args.N

    bounds = json.load(open(args.bounds_file))
    order = physics.SWEEP_PARAM_ORDER
    lo = np.array([bounds[k][0] for k in order]); hi = np.array([bounds[k][1] for k in order])
    from scipy.stats import qmc
    params = lo + qmc.LatinHypercube(d=len(order), seed=SEED).random(args.size) * (hi - lo)

    bumps, phases, noise = seed_components(args.N, args.L)
    ics = {k: build_ic(bumps, phases, noise, k) for k in ("intact", "isolated", "ablation", "phase_scrambled")}
    outdir = os.path.join(args.outdir, f"FRESH_HUNT_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(outdir, exist_ok=True)
    print(f"FRESH corrected-physics mutual-support hunt (NOT legacy-seeded)\n"
          f"size={args.size} N={args.N} steps={args.steps} axes={order}\n")

    rows, t0 = [], time.time()
    for b0 in range(0, args.size, args.batch):
        pb = params[b0:b0 + args.batch]
        out = {k: batch_run(pb, ics[k], args.N, args.L, args.dt, args.steps) for k in ics}
        for j in range(pb.shape[0]):
            def summ(kind):
                pm, pf, en, am, fin = out[kind]
                e0 = float(en[j][0]) if en[j][0] > 0 else 1e-30
                return {"finite": bool(fin[j]), "er": float(en[j][-1] / e0), "amp": float(am[j].max()),
                        "nodes": scl.node_count(np.abs(pf[j]) ** 2), "coh": scl.phase_coherence(pf[j]),
                        "pf": pf[j]}
            I, S, B, P = summ("intact"), summ("isolated"), summ("ablation"), summ("phase_scrambled")
            par = {k: float(pb[j][i]) for i, k in enumerate(order)}
            dia = gd.diagnose(I["pf"], par, dx) if I["finite"] else None
            geo = gd.geometry_verdict(dia) if dia else "nonfinite"

            cluster_persists = I["finite"] and I["nodes"] >= 2 and 0.3 <= I["er"] <= 5.0 and I["amp"] < 1e3
            iso_surv = S["er"] / max(I["er"], 1e-9)
            abl_node_loss = max(0.0, (I["nodes"] - 1) - B["nodes"]) / max(I["nodes"], 1)
            abl_e_loss = max(0.0, (I["er"] - B["er"]) / max(I["er"], 1e-9))
            abl_sens = abl_node_loss + abl_e_loss + (1.0 if not B["finite"] else 0.0)
            phase_dep = max(0.0, (I["nodes"] - P["nodes"]) / max(I["nodes"], 1)) + \
                        max(0.0, (I["coh"] - P["coh"]) / max(I["coh"], 1e-9))
            curv_bounded = bool(dia and dia["curvature_max"] < CURV_BOUND)
            support_index = (np.clip(1 - iso_surv, 0, 1) + np.clip(abl_sens, 0, 2) + np.clip(phase_dep, 0, 2)) \
                if cluster_persists else 0.0
            escalate = bool(cluster_persists and iso_surv < 0.7 and abl_sens > 0.2 and curv_bounded
                            and geo == "geometry_follows_RD_bounded")
            rows.append({"idx": b0 + j, **{k: round(par[k], 4) for k in order},
                         "support_index": round(float(support_index), 3),
                         "isolated_survival_ratio": round(iso_surv, 3), "ablation_sensitivity": round(float(abl_sens), 3),
                         "phase_lock_dependency": round(float(phase_dep), 3),
                         "energy_retention_ratio": round(I["er"], 3), "node_survival_fraction": round(I["nodes"] / K_SEEDS, 3),
                         "intact_nodes": I["nodes"], "iso_nodes": S["nodes"], "abl_nodes": B["nodes"], "scr_nodes": P["nodes"],
                         "curvature_max": round(dia["curvature_max"], 3) if dia else None,
                         "curvature_boundedness": curv_bounded,
                         "omega_node_correlation": round(dia["omega_node_correlation"], 3) if dia else None,
                         "sdg_h_norm_l2": round(dia["sdg_h_norm_l2"], 1) if dia else None,
                         "geometry": geo, "ESCALATE": escalate})
        print(f"  batch {b0//args.batch+1}/{-(-args.size//args.batch)} [{time.time()-t0:.0f}s]", flush=True)

    rows.sort(key=lambda r: -r["support_index"])
    with open(os.path.join(outdir, "fresh_hunt_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump({"size": args.size, "N": args.N, "steps": args.steps, "axes": order, "bounds": bounds,
               "headline": "stable_collapse/mutual_support", "prime_sse": "auxiliary_only",
               "legacy_seeded": False}, open(os.path.join(outdir, "meta.json"), "w"), indent=2)

    nesc = sum(1 for r in rows if r["ESCALATE"])
    print(f"\nDONE {args.size} configs in {(time.time()-t0)/60:.1f} min   ESCALATE: {nesc}/{len(rows)}")
    print("top 8 by support_index:")
    for r in rows[:8]:
        print(f"  idx={r['idx']:>4} support={r['support_index']:.2f} iso_surv={r['isolated_survival_ratio']:.2f} "
              f"abl_sens={r['ablation_sensitivity']:.2f} phase_dep={r['phase_lock_dependency']:.2f} "
              f"nodes={r['intact_nodes']} curv={r['curvature_max']} geo={r['geometry']} ESC={r['ESCALATE']}")
    print(f"-> {outdir}/fresh_hunt_results.csv")


if __name__ == "__main__":
    main()
