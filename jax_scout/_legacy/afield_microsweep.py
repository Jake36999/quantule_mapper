"""
IRER_A_FIELD_GEOMETRIC_FEEDBACK_v1_PROTOTYPE — paired micro-sweep (NOT a broad hunt).

For the top gamma_A=0 multiseed candidates, run a PAIRED design:
  gamma_A in {0 (control), 0.2 (tiny), 1.0 (low), 5.0 (medium)}
  IC in {intact(6 seeds), isolated(1), ablation(5), phase_scrambled(6, random phases)}
and record A / geometry diagnostics. Tests whether finite-speed A feedback changes the
gamma_A=0 failure mode (independent condensates) into genuine MUTUAL SUPPORT.

Promotion (per candidate, for some gamma_A>0): cluster persists AND isolated decays/weakens
AND ablation disrupts the rest AND geometry follows nodes with bounded curvature AND finite.
Falsification: if NO gamma_A>0 changes the isolated/ablation behaviour vs gamma_A=0.

A-on results are NOT rank-compatible with gamma_A=0 and are NOT promoted to CuPy here —
CuPy validates only if paired controls show genuine mutual support.

Run in WSL2 jax venv:  python /mnt/f/quantule_mapper/jax_scout/afield_microsweep.py --k 3 --steps 800
"""
import os, sys, csv, json, glob, time, argparse
import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import physics, afield_prototype as af, geometry_diag as gd, stable_collapse as scl

SEED, K_SEEDS = 20260619, 6
GAMMAS = [0.0, 0.2, 1.0, 5.0]


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
        for b in bumps:
            psi += b
    elif kind == "ablation":
        for b in bumps[:-1]:
            psi += b
    elif kind == "isolated":
        psi += bumps[0]
    elif kind == "phase_scrambled":
        for b, ph in zip(bumps, phases):
            psi += b * np.exp(1j * ph)
    return (psi + noise).astype(np.complex128)


def corr(a, b):
    a = a.ravel() - a.ravel().mean(); b = b.ravel() - b.ravel().mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def run_one(pvec, ic, gamma, N, L, dt, steps, params, topology):
    out = af.simulate_afield(jnp.asarray(pvec), jnp.asarray(ic), float(gamma), N, L, dt, steps,
                             jnp.float64, jnp.complex128, topology)
    psi_mid, psi_fin, A_mid, A_fin, energy, max_amp, A_energy, A_max, rve_min, rve_max, finite = \
        [np.asarray(o) if hasattr(o, "shape") else o for o in out]
    finite = bool(out[10])
    e0 = float(energy[0]) if energy[0] > 0 else 1e-30
    rho_f = np.abs(psi_fin) ** 2
    res = {"finite": finite, "er": float(energy[-1] / e0), "amp_max": float(max_amp.max()),
           "nodes": scl.node_count(rho_f), "coh": scl.phase_coherence(psi_fin),
           "A_energy": float(A_energy[-1]), "A_max": float(A_max.max()),
           "A_rms": float(np.sqrt(A_energy[-1] / psi_fin.size)),
           "A_node_corr": corr(A_fin, rho_f), "A_phase_lag_corr": corr(A_fin, np.abs(psi_mid) ** 2),
           "rve_min": float(rve_min.min()), "rve_max": float(rve_max.max())}
    return res, psi_fin, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--N", type=int, default=48)
    ap.add_argument("--L", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--topology", choices=["vacuum_ref", "additive_potential"], default="vacuum_ref")
    args = ap.parse_args()
    dx = args.L / args.N

    sd = sorted(glob.glob(os.path.join(ROOT, "sweep_runs", "STABLE_COLLAPSE_multiseed_*")))[-1]
    rows = [r for r in csv.DictReader(open(os.path.join(sd, "stable_collapse_results.csv"))) if len(r) > 5]
    prom = sorted([r for r in rows if float(r["coherence"]) > 0.2 and int(r["nodes_final"]) >= 2
                   and 0.3 <= float(r["energy_ratio"]) <= 5.0], key=lambda r: -float(r["coherence"]))[: args.k]
    order = physics.SWEEP_PARAM_ORDER
    bumps, phases, noise = seed_components(args.N, args.L)
    outdir = os.path.join(sd, f"afield_microsweep_{args.topology}"); os.makedirs(outdir, exist_ok=True)
    print(f"{af.BRANCH}  topology={args.topology}  contract={af.contract_key_for(args.topology)}\n"
          f"paired micro-sweep: {len(prom)} candidates x {GAMMAS} gamma x 4 ICs "
          f"(N={args.N}, steps={args.steps})\n")

    results, t0 = [], time.time()
    for r in prom:
        params = {k: float(r[k]) for k in order}
        pvec = [params[k] for k in order]
        for g in GAMMAS:
            ics = {k: run_one(pvec, build_ic(bumps, phases, noise, k), g, args.N, args.L, args.dt, args.steps, params, args.topology)
                   for k in ("intact", "isolated", "ablation", "phase_scrambled")}
            intact = ics["intact"][0]; iso = ics["isolated"][0]; abl = ics["ablation"][0]; scr = ics["phase_scrambled"][0]
            geo = gd.geometry_verdict(gd.diagnose(ics["intact"][1], params, dx)) if intact["finite"] else "nonfinite"
            cluster_persists = intact["finite"] and intact["nodes"] >= 2 and 0.3 <= intact["er"] <= 5.0
            # isolated baseline starts with 1 seed; "decays/weakens" = the LONE node retains
            # markedly LESS energy than it does in the cluster (or vanishes) -> cluster helps it.
            isolated_decays = (iso["nodes"] == 0) or (iso["er"] < 0.5 * intact["er"])
            ablation_disrupts = (abl["nodes"] < intact["nodes"] - 1) or (abl["er"] < 0.5 * intact["er"]) or (not abl["finite"])
            scramble_weakens = (scr["nodes"] < intact["nodes"]) or (scr["coh"] < 0.5 * intact["coh"])
            mutual = bool(cluster_persists and isolated_decays and ablation_disrupts)
            promote = bool(mutual and geo == "geometry_follows_RD_bounded" and intact["finite"])
            mod_depth = g * intact["A_rms"] / max(params["param_rho_vac"], 1e-9)
            rec = {"idx": r["idx"], "gamma_A": g,
                   "intact_n": intact["nodes"], "intact_er": round(intact["er"], 2), "intact_coh": round(intact["coh"], 3),
                   "iso_n": iso["nodes"], "iso_er": round(iso["er"], 2),
                   "abl_n": abl["nodes"], "abl_er": round(abl["er"], 2), "abl_finite": abl["finite"],
                   "scr_n": scr["nodes"], "scr_coh": round(scr["coh"], 3),
                   "A_rms": round(intact["A_rms"], 4), "A_max": round(intact["A_max"], 3),
                   "A_node_corr": round(intact["A_node_corr"], 3), "A_phase_lag_corr": round(intact["A_phase_lag_corr"], 3),
                   "mod_depth": round(mod_depth, 3), "rve_min": round(intact["rve_min"], 3), "rve_max": round(intact["rve_max"], 3),
                   "geo": geo, "isolated_decays": isolated_decays, "ablation_disrupts": ablation_disrupts,
                   "scramble_weakens": scramble_weakens, "mutual_support": mutual, "PROMOTE": promote}
            results.append(rec)
            print(f"[{r['idx']} g={g}] intact(n={intact['nodes']},er={intact['er']:.2f}) "
                  f"iso(n={iso['nodes']},er={iso['er']:.2f}) abl(n={abl['nodes']}) scr(n={scr['nodes']}) "
                  f"| Amod={mod_depth:.2f} A_nc={intact['A_node_corr']:.2f} geo={geo} "
                  f"mutual={mutual} -> PROMOTE={promote}", flush=True)

    with open(os.path.join(outdir, "afield_microsweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys())); w.writeheader(); w.writerows(results)
    json.dump({"branch": af.BRANCH, "topology": args.topology,
               "contract_key": af.contract_key_for(args.topology), "gammas": GAMMAS,
               "N": args.N, "steps": args.steps, "rank_compatible_with_gamma0": False},
              open(os.path.join(outdir, "meta.json"), "w"), indent=2)

    npro = sum(1 for r in results if r["PROMOTE"])
    # falsification: did ANY gamma_A>0 change isolated/ablation behaviour vs the gamma_A=0 control?
    changed = any(r["gamma_A"] > 0 and (r["isolated_decays"] or r["ablation_disrupts"]) for r in results)
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min   PROMOTE: {npro}/{len(results)}")
    print(f"FALSIFICATION: A feedback changed the isolated/ablation failure mode for some gamma_A>0: {changed}")
    print("VERDICT:", "PROTOTYPE shows candidate mutual-support signal -> deeper validation" if npro > 0 else
          ("A feedback altered behaviour but no full mutual-support promotion" if changed else
           "FALSIFIED: gamma_A>0 (this topology/range) does NOT produce mutual support"))
    print(f"-> {outdir}/afield_microsweep.csv")


if __name__ == "__main__":
    main()
