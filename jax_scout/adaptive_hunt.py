"""
Adaptive corrected-physics mutual-support hunt (mission-control timeboxed).

Headline objective = STABLE-COLLAPSE / MUTUAL-SUPPORT (prime-SSE NOT used). Evolutionary
search on the JAX scout: each individual = (8 params, ic_family in {coherent, randphase}).
Fitness = support core = (1 - isolated_survival_ratio) + ablation_sensitivity, gated by
hard-rejection. Elites -> Gaussian mutation + crossover + diversity injection (basin
exploration). A separate broadband-NOISE scan each generation covers that IC family
(classification only; no ablation possible). Timeboxed; checkpoints every generation.
NOT legacy-seeded. gamma_A=0 (A-field branches contract-separated, not used here).

Run (WSL2 jax venv), e.g. 4.5h:
  python /mnt/f/quantule_mapper/jax_scout/adaptive_hunt.py --hours 4.5 --pop 48 --N 40 --steps 500
"""
import os, sys, csv, json, time, argparse
import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import physics, geometry_diag as gd, stable_collapse as scl

SEED, K_SEEDS = 20260619, 6
CURV_BOUND = 1.0     # max|conformal curvature| above this = runaway geometry (reject)
AMP_BOUND = 1e3
ER_MIN, ER_MAX = 0.1, 5.0  # intact/ablated energy RETENTION band: outside = runaway/collapse,
                           # NOT bounded stable collapse (closes the energy-runaway-at-bounded-amp loophole)
ISO_GATE = 0.7       # isolated must survive < this fraction of the cluster
ABL_GATE = 0.3       # bounded ablation must disrupt survivors by at least this
FAMILIES = ["coherent", "randphase"]


def family_ics(family, N, L, base_seed):
    rng = np.random.default_rng(base_seed)
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    w = L / 12.0
    bumps, phases = [], []
    for _ in range(K_SEEDS):
        cx, cy, cz = rng.uniform(-L / 2, L / 2, 3)
        bumps.append(np.exp(-((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2) / (2 * w ** 2)))
        phases.append(rng.uniform(0, 2 * np.pi) if family == "randphase" else 0.0)
    noise = 0.01 * (rng.standard_normal((N, N, N)) + 1j * rng.standard_normal((N, N, N)))
    out = {}
    for kind, ks in (("intact", range(6)), ("ablation", range(5)), ("isolated", range(1))):
        psi = np.zeros((N, N, N), np.complex128)
        for i in ks:
            psi += bumps[i] * np.exp(1j * phases[i])
        out[kind] = (psi + noise).astype(np.complex128)
    return out


def noise_ic(N, L, seed):
    rng = np.random.default_rng(seed)
    return (0.3 * (rng.standard_normal((N, N, N)) + 1j * rng.standard_normal((N, N, N)))).astype(np.complex128)


def batch(params, psi0, N, L, dt, steps):
    pm, pf, en, am, fin = physics.sweep_probe(jnp.asarray(params), jnp.asarray(psi0), N, L, dt, steps,
                                              jnp.float64, jnp.complex128)
    return np.asarray(pm), np.asarray(pf), np.asarray(en), np.asarray(am), np.asarray(fin)


def reject_class(I, curv):
    if not I["finite"] or I["amp"] > AMP_BOUND:
        return "unstable_runaway"
    if I["er"] > ER_MAX or I["er"] < ER_MIN:
        return "energy_runaway"   # collective-gain growth or collapse; not bounded retention
    if curv >= CURV_BOUND:
        return "geometry_runaway"
    if I["nodes"] < 2:
        return "single_or_dissipative"
    if I["coh"] < 0.05:
        return "incoherent_speckle"
    return None


def evaluate(pop, ICS, N, L, dt, steps, bsz, dx, order):
    """pop: list of {params(8-vec), family}. Returns per-individual metric dicts with the
    PATCHED objective: bounded_ablation_sensitivity counts only bounded reorganisation of the
    survivors (0 unless BOTH intact and ablated are valid bounded trajectories)."""
    res = [None] * len(pop)
    for fam in FAMILIES:
        idxs = [i for i, p in enumerate(pop) if p["family"] == fam]
        if not idxs:
            continue
        P = np.array([pop[i]["params"] for i in idxs])
        out = {}
        for kind in ("intact", "isolated", "ablation"):
            mm = []
            for b0 in range(0, len(idxs), bsz):
                pb = P[b0:b0 + bsz]
                pm, pf, en, am, fin = batch(pb, ICS[fam][kind], N, L, dt, steps)
                for j in range(pb.shape[0]):
                    e0 = float(en[j][0]) if en[j][0] > 0 else 1e-30
                    mm.append({"finite": bool(fin[j]), "er": float(en[j][-1] / e0), "amp": float(am[j].max()),
                               "nodes": scl.node_count(np.abs(pf[j]) ** 2), "coh": scl.phase_coherence(pf[j]),
                               "pf": pf[j] if kind in ("intact", "ablation") else None})
            out[kind] = mm
        for k, i in enumerate(idxs):
            I, S, B = out["intact"][k], out["isolated"][k], out["ablation"][k]
            par = {kk: float(P[k][m]) for m, kk in enumerate(order)}
            I_curv = gd.curvature_max_only(I["pf"], par, dx) if I["finite"] else float("inf")
            B_curv = gd.curvature_max_only(B["pf"], par, dx) if B["finite"] else float("inf")
            rej = reject_class(I, I_curv)
            iso_surv = S["er"] / max(I["er"], 1e-9)
            I_bounded = (I["finite"] and I["amp"] < AMP_BOUND and ER_MIN <= I["er"] <= ER_MAX
                         and I["nodes"] >= 2 and I_curv < CURV_BOUND)
            B_bounded = (B["finite"] and B["amp"] < AMP_BOUND and ER_MIN <= B["er"] <= ER_MAX
                         and B_curv < CURV_BOUND)
            # RAW (old, loophole) metric — logged only, never used for selection
            raw_abl = (max(0.0, (I["nodes"] - 1 - B["nodes"]) / max(I["nodes"], 1))
                       + max(0.0, (I["er"] - B["er"]) / max(I["er"], 1e-9)) + (1.0 if not B["finite"] else 0.0))
            # BOUNDED metric: only bounded reorganisation of the survivors counts; 0 if either
            # trajectory is invalid (kills the ablation-instability exploit).
            if I_bounded and B_bounded:
                node_disrupt = max(0.0, (I["nodes"] - 1 - B["nodes"]) / max(I["nodes"], 1))
                e_change = abs(I["er"] - B["er"]) / max(I["er"], 1e-9)
                coh_change = abs(I["coh"] - B["coh"]) / max(I["coh"], 1e-9)
                bounded_abl = node_disrupt + 0.5 * e_change + 0.5 * coh_change
            else:
                bounded_abl = 0.0
            support_legs = bool(I_bounded and iso_surv < ISO_GATE and bounded_abl > ABL_GATE)
            core = (np.clip(1 - iso_surv, 0, 1) + np.clip(bounded_abl, 0, 2)) if I_bounded else 0.0
            res[i] = {"family": fam, "reject": rej, "core": float(core), "iso_surv": iso_surv,
                      "raw_abl_sens": float(raw_abl), "bounded_abl_sens": float(bounded_abl),
                      "intact_nodes": I["nodes"], "abl_nodes": B["nodes"], "iso_nodes": S["nodes"],
                      "er": I["er"], "amp": I["amp"], "coh": I["coh"], "curv": float(I_curv),
                      "abl_bounded": B_bounded, "support_legs": support_legs, "pf": I["pf"]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=4.5)
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--N", type=int, default=40)
    ap.add_argument("--L", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--bsz", type=int, default=16)
    ap.add_argument("--elite", type=int, default=10)
    ap.add_argument("--bounds-file", default="/mnt/f/quantule_mapper/jax_scout/gain_bounds.json")
    ap.add_argument("--outdir", default="/mnt/f/quantule_mapper/sweep_runs")
    args = ap.parse_args()
    dx = args.L / args.N
    bounds = json.load(open(args.bounds_file)); order = physics.SWEEP_PARAM_ORDER
    lo = np.array([bounds[k][0] for k in order]); hi = np.array([bounds[k][1] for k in order])
    rng = np.random.default_rng(SEED)
    ICS = {fam: family_ics(fam, args.N, args.L, SEED) for fam in FAMILIES}

    outdir = os.path.join(args.outdir, f"ADAPTIVE_HUNT_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(outdir, exist_ok=True)
    log = open(os.path.join(outdir, "all_evals.csv"), "w", newline="")
    cols = ["gen", "family", "core", "iso_surv", "raw_abl_sens", "bounded_abl_sens", "abl_bounded",
            "support_legs", "curv", "intact_nodes", "abl_nodes", "er", "coh", "reject", *order]
    cw = csv.DictWriter(log, fieldnames=cols); cw.writeheader()

    def sample(n):
        u = rng.random((n, len(order)))
        return [{"params": (lo + u[i] * (hi - lo)), "family": FAMILIES[i % 2]} for i in range(n)]

    pop = sample(args.pop)
    t0 = time.time(); deadline = t0 + args.hours * 3600
    gen = 0
    n_eval = 0; n_support = 0; reject_counts = {}; fam_counts = {f: 0 for f in FAMILIES}; noise_counts = {}
    best = []   # global elites: (core, gen, individual-metrics)
    basins = {}  # rounded-param -> count
    while time.time() < deadline:
        gen += 1
        res = evaluate(pop, ICS, args.N, args.L, args.dt, args.steps, args.bsz, dx, order)
        for ind, r in zip(pop, res):
            if r is None:
                continue
            n_eval += 1; fam_counts[r["family"]] += 1
            reject_counts[r["reject"] or "accepted"] = reject_counts.get(r["reject"] or "accepted", 0) + 1
            if r["support_legs"]:
                n_support += 1
            cw.writerow({"gen": gen, **{k: round(float(ind["params"][i]), 4) for i, k in enumerate(order)},
                         **{k: r[k] for k in ("family", "core", "iso_surv", "raw_abl_sens", "bounded_abl_sens",
                                              "abl_bounded", "support_legs", "curv", "intact_nodes",
                                              "abl_nodes", "er", "coh", "reject")}})
            if r["reject"] is None and r["core"] > 0.05:
                key = tuple(np.round(ind["params"], 1))
                basins[key] = basins.get(key, 0) + 1
                best.append((r["core"], r["bounded_abl_sens"], gen, dict(params=ind["params"].tolist(),
                            family=r["family"], iso_surv=r["iso_surv"], bounded_abl_sens=r["bounded_abl_sens"],
                            raw_abl_sens=r["raw_abl_sens"], support_legs=r["support_legs"],
                            intact_nodes=r["intact_nodes"], er=r["er"], coh=r["coh"], curv=r["curv"])))
        log.flush()
        best.sort(key=lambda x: -x[0]); best = best[:30]
        # broadband-noise exploration scan (classification only; logged separately)
        npop = sample(args.bsz // 2)
        pm, pf, en, am, fin = batch(np.array([n["params"] for n in npop]),
                                    noise_ic(args.N, args.L, SEED + gen), args.N, args.L, args.dt, args.steps)
        for j in range(len(npop)):
            o = scl.observe(pm[j], pf[j], en[j], am[j], fin[j])
            noise_counts[o["class"]] = noise_counts.get(o["class"], 0) + 1

        # next generation: elites (mutate+crossover) + diversity injection
        elites = [b[3] for b in best[:args.elite]] if best else []
        newpop = []
        if elites:
            for _ in range(args.pop - args.pop // 4):
                pa = elites[rng.integers(len(elites))]
                pb = elites[rng.integers(len(elites))]
                cx = np.where(rng.random(len(order)) < 0.5, np.array(pa["params"]), np.array(pb["params"]))
                mut = cx + rng.normal(0, 0.12, len(order)) * (hi - lo)
                newpop.append({"params": np.clip(mut, lo, hi),
                               "family": pa["family"] if rng.random() < 0.8 else FAMILIES[rng.integers(2)]})
        newpop += sample(args.pop - len(newpop))   # diversity / fresh basins
        pop = newpop

        elapsed = (time.time() - t0) / 3600
        bc = best[0] if best else (0, 0, 0, {})
        print(f"[gen {gen} t={elapsed:.2f}h] evals={n_eval} support_legs={n_support} best_core={bc[0]:.3f} "
              f"best_boundedAbl={bc[1]:.3f} basins={len(basins)} "
              f"reject={({k:v for k,v in reject_counts.items() if k!='accepted'})}", flush=True)

        json.dump({"gen": gen, "elapsed_h": elapsed, "n_eval": n_eval, "n_support_legs": n_support,
                   "fam_counts": fam_counts, "reject_counts": reject_counts, "noise_counts": noise_counts,
                   "n_basins": len(basins), "objective": "bounded_mutual_support (3-leg gated)",
                   "best": [{"core": b[0], "bounded_abl_sens": b[1], "gen": b[2], **b[3]} for b in best[:10]],
                   "legacy_seeded": False, "gamma_A": 0.0},
                  open(os.path.join(outdir, "status.json"), "w"), indent=2)

    log.close()
    print(f"\n=== TIMEBOX DONE: {gen} gens, {n_eval} evals in {(time.time()-t0)/3600:.2f}h ===")
    print(f"families: {fam_counts}")
    print(f"reject classes: {reject_counts}")
    print(f"noise-IC classes: {noise_counts}")
    print(f"distinct basins (core>0.05): {len(basins)}")
    print(f"support_legs candidates (intact survives + isolated weaker + BOUNDED ablation disrupt): {n_support}")
    if best:
        print(f"best bounded_abl_sens: {max(b[1] for b in best):.3f}; "
              f"best raw_abl_sens: {max(b[3]['raw_abl_sens'] for b in best):.3f} (raw = pre-gate, may be instability)")
        print("top 5 by core:")
        for b in best[:5]:
            print(f"  core={b[0]:.3f} bounded_abl={b[1]:.3f} raw_abl={b[3]['raw_abl_sens']:.3f} "
                  f"iso_surv={b[3]['iso_surv']:.3f} nodes={b[3]['intact_nodes']} legs={b[3]['support_legs']} "
                  f"curv={b[3]['curv']:.2f} fam={b[3]['family']}")
    has_support = any(b[3]["support_legs"] for b in best)
    verdict = ("SUPPORT CANDIDATES (all 3 legs) found -> Phase C full-fidelity reproduction"
               if has_support else
               "NO bounded mutual-support: no candidate passed all 3 legs (intact survives + isolated "
               "weaker + BOUNDED ablation disruption). Any raw ablation signal was instability only.")
    print("VERDICT:", verdict)
    print(f"-> {outdir}")


if __name__ == "__main__":
    main()
