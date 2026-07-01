"""
FMIA bridge-hunting adaptive hunt — objective = fmia_transfer_score (NOT destructive ablation).

The transfer re-analysis (docs/FMIA_TRANSFER_DIAGNOSTIC_FINDING.md) showed the prior bounded
population has NO Omega^2 corridors / density bridges (conductance ~ 0) — the structural
precondition for FMIA Informational-Parallel transfer. That objective only rewarded node
survival, so the search isolated nodes in voids. This hunt explicitly rewards the structural
precondition AND null-referenced transfer along it.

STAGED objective (cheap -> expensive), all gated against instability gaming:
  Stage 1 (batched sweep_probe, cheap): hard gates finite/amp/energy-band/curvature/multinode;
          reject unstable_runaway/energy_runaway/geometry_runaway/single_or_dissipative/speckle.
  Stage 2 (final field only, cheap): density-bridge conductance, Omega^2 corridor, J_info bridge
          flux, interference overlap between node centroids — ALL snapshot metrics from pf.
          This is the new structural reward; most no-bridge configs stop here cheaply.
  Stage 3 (trajectory re-run, expensive): only for structurally-promising bridge-formers — full
          null-referenced temporal transfer (energy-exchange / phase-coupling-above-floor /
          action-rate excess via transfer_diag v2).

  fmia_transfer_score = geom_bounded * ( node_stability + 2*max_corridor_conductance
        + mean_corridor_conductance + J_info_bridge_flux + interference_excess
        + 1.5*phase_coupling_excess_ABOVE_FLOOR + energy_exchange_excess + action_rate_excess )
  Transfer terms are 0 unless a bridge triggers Stage 3; phase coupling credited only ABOVE the
  0.73 independence floor; instability is rejected (score 0) -> cannot be gamed.

Reclassify (NOT discard) into: stable_independent_condensates, collective_density_threshold,
no_corridor_stable_nodes, marginal_phase_thread, candidate_transfer_seed (+ reject classes).

Run (WSL2 jax venv):
  calibration:  python jax_scout/bridge_hunt.py --calibrate
  full hunt:    python jax_scout/bridge_hunt.py --hours 3 --pop 48 --N 48 --steps 800
"""
import os, sys, csv, json, time, argparse, glob, gc
import numpy as np
# GPU memory on this 8GB card is SHARED with the Windows display and FRAGMENTED under WSL2:
# a single large contiguous pool grab fails (PREALLOCATE=true at 0.55-0.75 -> RESOURCE_EXHAUSTED
# at startup), while PREALLOCATE=false's BFC pool GROWS per capture and never releases
# (cumulative OOM at gen 11). The 'platform' allocator sidesteps both: it cudaMalloc/cudaFree
# on demand per op (no big contiguous grab) and RELEASES memory back when arrays are freed (no
# cumulative growth). Slower per op, but the working set is small (~0.3GB) and robustness wins.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import physics, geometry_diag as gd, stable_collapse as scl, transfer_diag as td

SEED, K_SEEDS = 20260619, 6
CURV_BOUND, AMP_BOUND = 1.0, 1e3
ER_MIN, ER_MAX = 0.1, 5.0
FAMILIES = ["coherent", "randphase"]
FLOOR_P95 = td.THR_PHASECOUP            # 0.73 phase-coupling independence floor
STAGE3_CObd = 0.05                      # max corridor conductance to TRIGGER the temporal stage
STAGE3_TOPK = 10                        # cap full trajectory transfer to the top-K bridge-formers
                                        # PER GENERATION (bounds GPU-memory churn -> prevents the
                                        # gen-3 OOM/abort that killed the first 3h launch)
N_SNAP = 30                             # in-hunt snapshots (smaller capture; validation uses 40)


def family_ic(family, N, L, base_seed):
    rng = np.random.default_rng(base_seed)
    x = np.linspace(-L/2, L/2, N, endpoint=False); X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    w = L/12.0; psi = np.zeros((N, N, N), np.complex128)
    for _ in range(K_SEEDS):
        cx, cy, cz = rng.uniform(-L/2, L/2, 3)
        bump = np.exp(-((X-cx)**2+(Y-cy)**2+(Z-cz)**2)/(2*w**2))
        ph = rng.uniform(0, 2*np.pi) if family == "randphase" else 0.0
        psi += bump*np.exp(1j*ph)
    noise = 0.01*(rng.standard_normal((N, N, N))+1j*rng.standard_normal((N, N, N)))
    return (psi+noise).astype(np.complex128)


def reject_class(fin, er, amp, curv, nodes, coh):
    if not fin or amp > AMP_BOUND:
        return "unstable_runaway"
    if er > ER_MAX or er < ER_MIN:
        return "energy_runaway"
    if curv >= CURV_BOUND:
        return "geometry_runaway"
    if nodes < 2:
        return "single_or_dissipative"
    if coh < 0.05:
        return "incoherent_speckle"
    return None


def structural_from_field(pf, par, dx, N, rng):
    """Stage 2 — snapshot bridge/corridor metrics from the final field only (cheap)."""
    nodes = td.detect_nodes(pf, dx)
    if len(nodes) < 2:
        return {"nodes": len(nodes), "max_cond": 0.0, "mean_cond": 0.0,
                "mean_jflux": 0.0, "mean_align": 0.0, "interf_excess": 0.0}
    cents = [nd["centroid"] for nd in nodes]
    geo = td.geometry_fields(pf, par, dx)
    conds, jflux, aligns, interf = [], [], [], []
    for i in range(len(cents)):
        for j in range(i+1, len(cents)):
            cm = td.corridor_pair_metrics(geo, cents[i], cents[j], N, dx)
            conds.append(cm["conductance"]); jflux.append(abs(cm["J_flux"])); aligns.append(cm["path_align"])
            ie, _ = td.interference_overlap(pf, cents[i], cents[j], N, rng)
            interf.append(ie)
    return {"nodes": len(nodes), "max_cond": float(np.max(conds)), "mean_cond": float(np.mean(conds)),
            "mean_jflux": float(np.mean(jflux)), "mean_align": float(np.mean(aligns)),
            "interf_excess": float(np.mean(interf))}


def _node_stability(nodes):
    """Reward bounded FEW-node structure (~3-5); penalise fragmentation (space-filling
    high-node-count fields, which trivially inflate max-pair conductance)."""
    up = np.clip((nodes - 1) / 2.0, 0, 1)        # ramps 2->4 nodes
    down = np.clip((9 - nodes) / 3.0, 0, 1)      # ramps down 6->9 nodes
    return float(up * down)


def _energy_clean(er):
    """Stable collapse is ENERGY-CONSERVATIVE (er~1): trap energy in a standing structure,
    neither dissipate (er<<1) nor grow (er>>1). Peaked at er=1; discounts the FMIA bonus for
    dissipating/growing configs (whose 'exchange' can be differential decay, not transfer)."""
    return float(np.exp(-(np.log(max(er, 1e-9)) ** 2) / (2 * 0.5 ** 2)))


def fmia_score(geom_bounded, st, temporal, er):
    """node_stability (bounded existence base) + energy-conservation-weighted FMIA bonus.
    The FMIA bonus is driven by NULL-REFERENCED transfer + NETWORK-wide selective bridging:
    max_cond (single lucky pair, inflated by node count / space-filling) is down-weighted;
    mean_cond (naturally low for space-filling) carries the structural reward; J_flux supporting."""
    if not geom_bounded:
        return 0.0
    phase_excess = max(0.0, temporal.get("phase_coupling_score", 0.0) - FLOOR_P95)
    fmia_bonus = (
        1.0 * st["mean_cond"] + 0.5 * st["max_cond"]         # network bridging dominates max-pair
        + 0.5 * st["interf_excess"]
        + 2.0 * phase_excess                                  # null-referenced, dominant
        + 1.5 * temporal.get("energy_exchange_index", 0.0)    # null-referenced
        + 0.5 * temporal.get("action_rate_coherence", 0.0)
        + 0.3 * st["mean_jflux"]                              # supporting evidence only
    )
    return float(_node_stability(st["nodes"]) + _energy_clean(er) * fmia_bonus)


def classify(rej, geom_bounded, st, temporal, iso_surv=np.nan):
    if rej is not None:
        return rej
    if not geom_bounded or st["nodes"] < 2:
        return "single_or_dissipative"
    has_bridge = st["max_cond"] > STAGE3_CObd
    # transfer must be NULL-REFERENCED: above-floor phase coupling or energy-exchange excess.
    # J_flux is NOT a valid trigger (generically nonzero, not null-referenced).
    above_floor = temporal.get("phase_coupling_score", 0.0) > FLOOR_P95
    transfer = above_floor or temporal.get("energy_exchange_index", 0.0) > td.THR_EXCHANGE
    if has_bridge and transfer:
        return "candidate_transfer_seed"
    if above_floor and not has_bridge:
        return "marginal_phase_thread"
    if has_bridge:
        return "bridge_no_transfer"          # structure present, transfer not above null (partial seed)
    if not np.isnan(iso_surv) and iso_surv < 0.5:
        return "collective_density_threshold"
    return "no_corridor_stable_nodes"


def evaluate(pop, ICS, N, L, dt, steps, bsz, dx, order, rng):
    """Pass 1: cheap probe + snapshot structural for ALL. Pass 2: run the expensive Stage-3
    trajectory transfer only for the GLOBAL top-STAGE3_TOPK bridge-formers (bounds GPU-memory
    churn — the cause of the gen-3 OOM/abort — and focuses the analysis on the best bridges)."""
    res = [None] * len(pop)
    cheap = [None] * len(pop)
    # --- Pass 1: cheap, batched ---
    for fam in FAMILIES:
        idxs = [i for i, p in enumerate(pop) if p["family"] == fam]
        if not idxs:
            continue
        P = np.array([pop[i]["params"] for i in idxs])
        for b0 in range(0, len(idxs), bsz):
            pb = P[b0:b0+bsz]
            pm, pf, en, am, fin = physics.sweep_probe(jnp.asarray(pb), jnp.asarray(ICS[fam]["intact"]),
                                                      N, L, dt, steps, jnp.float64, jnp.complex128)
            pf = np.asarray(pf); en = np.asarray(en); am = np.asarray(am); fin = np.asarray(fin)
            for j in range(pb.shape[0]):
                i = idxs[b0+j]; par = {kk: float(pb[j][m]) for m, kk in enumerate(order)}
                e0 = float(en[j][0]) if en[j][0] > 0 else 1e-30
                I = {"finite": bool(fin[j]), "er": float(en[j][-1]/e0), "amp": float(am[j].max()),
                     "nodes": scl.node_count(np.abs(pf[j])**2), "coh": scl.phase_coherence(pf[j])}
                try:
                    curv = gd.curvature_max_only(pf[j], par, dx) if I["finite"] else float("inf")
                    rej = reject_class(I["finite"], I["er"], I["amp"], curv, I["nodes"], I["coh"])
                    geom_bounded = (rej is None)
                    st = (structural_from_field(pf[j], par, dx, N, rng) if geom_bounded
                          else {"nodes": I["nodes"], "max_cond": 0.0, "mean_cond": 0.0,
                                "mean_jflux": 0.0, "mean_align": 0.0, "interf_excess": 0.0})
                except Exception as e:                              # never let one config kill the gen
                    curv, rej, geom_bounded = float("inf"), "analysis_error", False
                    st = {"nodes": I["nodes"], "max_cond": 0.0, "mean_cond": 0.0,
                          "mean_jflux": 0.0, "mean_align": 0.0, "interf_excess": 0.0}
                cheap[i] = {"fam": fam, "par": par, "params": list(pb[j]), "I": I,
                            "curv": float(curv), "rej": rej, "geom_bounded": geom_bounded, "st": st}
    # --- Pass 2: global top-K bridge-formers get Stage 3 ---
    bridge = [i for i in range(len(pop)) if cheap[i] and cheap[i]["geom_bounded"]
              and cheap[i]["st"]["max_cond"] > STAGE3_CObd]
    bridge.sort(key=lambda i: -(cheap[i]["st"]["mean_cond"] + 0.5*cheap[i]["st"]["max_cond"]
                                + 0.3*cheap[i]["st"]["mean_jflux"]))
    stage3_set = set(bridge[:STAGE3_TOPK])
    # --- Pass 3: assemble (Stage 3 only for the selected, guarded) ---
    for i in range(len(pop)):
        c = cheap[i]
        if c is None:
            continue
        temporal = {}; stage3 = i in stage3_set
        if stage3:
            try:
                tr = td.analyze_candidate(c["params"], c["par"], ICS[c["fam"]]["intact"], N, L, dt,
                                          steps, N_SNAP, bounded_abl_sens=0.0, iso_surv=np.nan)
                temporal = {kk: tr.get(kk, 0.0) for kk in
                            ("phase_coupling_score", "action_rate_coherence", "energy_exchange_index")}
            except Exception:
                temporal = {}; stage3 = False
        st = c["st"]; I = c["I"]
        score = fmia_score(c["geom_bounded"], st, temporal, I["er"])
        klass = classify(c["rej"], c["geom_bounded"], st, temporal)
        res[i] = {"family": c["fam"], "reject": c["rej"], "fmia_score": float(score), "klass": klass,
                  "nodes": st["nodes"], "max_cond": st["max_cond"], "mean_cond": st["mean_cond"],
                  "mean_jflux": st["mean_jflux"], "interf_excess": st["interf_excess"],
                  "phase_coup": float(temporal.get("phase_coupling_score", 0.0)),
                  "arc": float(temporal.get("action_rate_coherence", 0.0)),
                  "exch": float(temporal.get("energy_exchange_index", 0.0)),
                  "er": I["er"], "amp": I["amp"], "curv": c["curv"], "stage3": stage3,
                  "params": c["params"]}
    return res


def load_seed_population(order):
    """Calibration seeds: the 12 known bounded (no-bridge) configs as NEGATIVE controls +
    gen15 as the marginal thread."""
    d = sorted(glob.glob(os.path.join(ROOT, "sweep_runs", "ADAPTIVE_HUNT_2026062*")))[-1]
    rows = list(csv.DictReader(open(os.path.join(d, "all_evals.csv"))))
    def f(r, k):
        try: return float(r[k])
        except: return float("nan")
    bnd = [r for r in rows if r["reject"] == "" and 2 <= f(r, "intact_nodes") <= 20
           and f(r, "curv") < 1.0 and 0.1 <= f(r, "er") <= 5.0]
    bnd.sort(key=lambda r: f(r, "iso_surv"))
    seeds = []
    for r in bnd[:12]:
        seeds.append({"params": np.array([f(r, k) for k in order]), "family": "coherent",
                      "tag": "neg_control_bounded"})
    return seeds


def main():
    global STAGE3_TOPK
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--N", type=int, default=48)
    ap.add_argument("--L", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--bsz", type=int, default=12)
    ap.add_argument("--elite", type=int, default=10)
    ap.add_argument("--stage3-topk", type=int, default=STAGE3_TOPK,
                    help="override Stage-3 cap (raise it to stress-test the GPU-memory fix)")
    ap.add_argument("--calibrate", action="store_true",
                    help="2-gen calibration seeded with known bounded configs + LHS; no timebox")
    ap.add_argument("--bounds-file", default=os.path.join(ROOT, "jax_scout", "gain_bounds.json"))
    ap.add_argument("--outdir", default=os.path.join(ROOT, "sweep_runs"))
    args = ap.parse_args()
    STAGE3_TOPK = args.stage3_topk
    dx = args.L / args.N
    bounds = json.load(open(args.bounds_file)); order = physics.SWEEP_PARAM_ORDER
    lo = np.array([bounds[k][0] for k in order]); hi = np.array([bounds[k][1] for k in order])
    rng = np.random.default_rng(SEED)
    ICS = {fam: {"intact": family_ic(fam, args.N, args.L, SEED)} for fam in FAMILIES}

    def sample(n):
        u = rng.random((n, len(order)))
        return [{"params": (lo+u[i]*(hi-lo)), "family": FAMILIES[i % 2]} for i in range(n)]

    tag = "BRIDGE_CALIB" if args.calibrate else "BRIDGE_HUNT"
    outdir = os.path.join(args.outdir, f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(outdir, exist_ok=True)
    log = open(os.path.join(outdir, "all_evals.csv"), "w", newline="")
    cols = ["gen", "family", "klass", "fmia_score", "nodes", "max_cond", "mean_cond", "mean_jflux",
            "interf_excess", "phase_coup", "arc", "exch", "er", "curv", "stage3", "reject", *order]
    cw = csv.DictWriter(log, fieldnames=cols, extrasaction="ignore"); cw.writeheader()

    if args.calibrate:
        seeds = load_seed_population(order)
        pop = seeds + sample(args.pop - len(seeds))
        max_gen, deadline = 2, float("inf")
    else:
        pop = sample(args.pop)
        max_gen, deadline = 10**9, time.time() + args.hours*3600

    t0 = time.time(); gen = 0; n_eval = 0; reject_counts = {}; klass_counts = {}; best = []
    crashed = None
    while time.time() < deadline and gen < max_gen:
        gen += 1
        try:
            res = evaluate(pop, ICS, args.N, args.L, args.dt, args.steps, args.bsz, dx, order, rng)
        except Exception:                                   # never lose the run on one bad generation
            import traceback
            crashed = traceback.format_exc()
            with open(os.path.join(outdir, "CRASH.txt"), "w") as fh:
                fh.write(f"gen={gen}\n{crashed}")
            print(f"[gen {gen}] EVALUATE CRASHED — logged to CRASH.txt, stopping cleanly:\n{crashed}",
                  flush=True)
            break
        for ind, r in zip(pop, res):
            if r is None:
                continue
            n_eval += 1
            reject_counts[r["reject"] or "accepted"] = reject_counts.get(r["reject"] or "accepted", 0)+1
            klass_counts[r["klass"]] = klass_counts.get(r["klass"], 0)+1
            cw.writerow({"gen": gen, **{k: round(float(ind["params"][i]), 4) for i, k in enumerate(order)},
                         **{k: r[k] for k in ("family", "klass", "fmia_score", "nodes", "max_cond",
                            "mean_cond", "mean_jflux", "interf_excess", "phase_coup", "arc", "exch",
                            "er", "curv", "stage3", "reject")}})
            if r["reject"] is None and r["fmia_score"] > 0.05:
                best.append((r["fmia_score"], gen, r))
        log.flush(); best.sort(key=lambda x: -x[0]); best = best[:30]
        # next gen
        elites = [b[2] for b in best[:args.elite]] if best else []
        newpop = []
        if elites and not args.calibrate:
            for _ in range(args.pop - args.pop//4):
                pa = elites[rng.integers(len(elites))]; pb = elites[rng.integers(len(elites))]
                cx = np.where(rng.random(len(order)) < 0.5, np.array(pa["params"]), np.array(pb["params"]))
                mut = np.clip(cx + rng.normal(0, 0.12, len(order))*(hi-lo), lo, hi)
                newpop.append({"params": mut, "family": pa["family"] if rng.random() < 0.8 else FAMILIES[rng.integers(2)]})
        newpop += sample(args.pop - len(newpop))
        pop = newpop
        elapsed = (time.time()-t0)/3600
        bc = best[0] if best else (0, 0, {})
        print(f"[gen {gen} t={elapsed:.2f}h] evals={n_eval} best_fmia={bc[0]:.3f} "
              f"klass={klass_counts} reject={({k:v for k,v in reject_counts.items() if k!='accepted'})}",
              flush=True)
        json.dump({"gen": gen, "elapsed_h": elapsed, "n_eval": n_eval, "objective": "fmia_transfer_score",
                   "klass_counts": klass_counts, "reject_counts": reject_counts,
                   "best": [{"fmia_score": b[0], "gen": b[1], **{k: b[2][k] for k in
                            ("klass", "nodes", "max_cond", "mean_cond", "mean_jflux", "phase_coup",
                             "arc", "exch", "er", "curv", "stage3", "family")}, "params": b[2]["params"]}
                            for b in best[:10]]},
                  open(os.path.join(outdir, "status.json"), "w"), indent=2, default=float)
        gc.collect()                                        # free captured device arrays each gen

    log.close()
    print(f"\n=== {tag} DONE: {gen} gens, {n_eval} evals in {(time.time()-t0)/3600:.2f}h ===")
    print(f"reject classes: {reject_counts}")
    print(f"transfer classes: {klass_counts}")
    n_seed = klass_counts.get("candidate_transfer_seed", 0)
    print(f"candidate_transfer_seed (bridge + above-null transfer): {n_seed}")
    if best:
        print("top 5 by fmia_score:")
        for b in best[:5]:
            r = b[2]
            print(f"  fmia={b[0]:.3f} {r['klass']:28} nodes={r['nodes']} maxCond={r['max_cond']:.3f} "
                  f"Jflux={r['mean_jflux']:.3f} pcoup={r['phase_coup']:.3f} stage3={r['stage3']} "
                  f"er={r['er']:.2f} curv={r['curv']:.2f}")
    print(f"-> {outdir}")


if __name__ == "__main__":
    main()
