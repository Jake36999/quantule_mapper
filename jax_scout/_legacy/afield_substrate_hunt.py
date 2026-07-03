"""
FRESH SUBSTRATE-QUALITY hunt (stable scout path; NOT a tensor Stage C hunt, NOT a Payan hunt).

Goal: find MORE unique bounded strong-bridge substrates so the corrected routing gate can be applied
to a larger pool (the prior pool had only N=4 unique strong bridges). Expands N before the Payan pivot.

OBJECTIVE = SUBSTRATE QUALITY ONLY (no routing/coupling metric in the evolutionary loop -- routing is
expensive and denominator-vulnerable; it is applied ONLY as a validation gate on finalists):
  * bounded energy at the SETTLE (~800) AND at the ROUTING HORIZON (2800 = settle 800 + cont 2000),
  * stable few-node structure (2-8) with the count preserved settle->horizon (no fragmentation),
  * STRONG non-saturated bridge conductance MEASURED AT THE SETTLE TIME (0.15 < cond_settle < 0.85,
    reward rising to ~0.5) -- the bridge the routing gate will actually see,
  * finite/bounded all the way to the routing horizon (so finalists are ROUTING-EVALUABLE).
HARD-REJECT (cannot be gamed): nonfinite, amp runaway, energy drift (settle or horizon), curvature
  runaway, <2 or >8 nodes, node-count drift >2 (fragmentation), bridge saturation (>0.85).
NOT rewarded: bridge/void routing ratio, phase coupling, energy growth, saturation, space-filling.

Preserves ALL unique strong-bridge substrates (cond_settle >= 0.3, accepted) each generation to
frozen_substrates.json, even though routing is NOT evaluated during the hunt.

CAUTION: validated current-coupled A machinery (IRER-SNCGL-CURRENT-COUPLED-AFFECT-ETDRK4-v1); JAX
scout; not for CuPy/Hunter promotion. WSL2 jax venv:
  python /mnt/f/quantule_mapper/jax_scout/afield_substrate_hunt.py --hours 4
"""
import os, sys, csv, json, time, argparse, hashlib
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
import jax
jax.config.update("jax_enable_x64", True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import physics, transfer_diag as td, geometry_diag as gd
from jax_scout import afield_current_coupled as cc

SEED = 20260619
CONTRACT_KEY = cc.CONTRACT_KEY
order = physics.SWEEP_PARAM_ORDER
A_ORDER = ["gamma_A", "kappa", "c_A"]
A_BOUNDS = {"gamma_A": [0.0, 0.3], "kappa": [0.5, 3.0], "c_A": [0.5, 3.0]}
SETTLE, HORIZON, NSNAP, PRE_STEPS = 800, 2800, 40, 400   # horizon = routing settle(800)+cont(2000)
ER_LO, ER_HI, CURV_MAX, AMP_MAX = 0.5, 2.0, 1.0, 1e3
NODE_LO, NODE_HI, BRIDGE_LO, BRIDGE_HI, STRONG = 2, 8, 0.15, 0.85, 0.3
L_ = cc.L


def _bridge_cond(psi, par, dx, N):
    nodes = td.detect_nodes(psi, dx)
    if len(nodes) < 2:
        return 0.0, len(nodes)
    geo = td.geometry_fields(psi, par, dx); best = 0.0
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            c = td.corridor_pair_metrics(geo, nodes[i]["centroid"], nodes[j]["centroid"], N, dx)["conductance"]
            best = max(best, c)
    return float(best), len(nodes)


def evaluate_one(par, g, kap, cA, N=48):
    dx = L_ / N; ic = cc.multiseed_ic(N, SEED)
    ps, _, fin = cc.capture_cc(par, ic, g, N, PRE_STEPS, 8, kappa=kap, c_A=cA)
    if not fin or float(np.max(np.abs(ps[-1]))) > AMP_MAX:
        return {"reject": "pre_runaway", "score": 0.0, "klass": "RUNAWAY_REJECT"}
    snaps, _, fin = cc.capture_cc(par, ic, g, N, HORIZON, NSNAP, kappa=kap, c_A=cA)
    amp = float(np.max(np.abs(snaps[-1]))) if fin else float("inf")
    if not fin or amp > AMP_MAX:
        return {"reject": "nonfinite_or_runaway", "score": 0.0, "klass": "NONFINITE_REJECT"}
    e0 = float(np.sum(np.abs(snaps[0]) ** 2)) + 1e-30
    i_settle = max(1, round(SETTLE / HORIZON * NSNAP))
    psi_s, psi_e = snaps[i_settle], snaps[-1]
    er_s = float(np.sum(np.abs(psi_s) ** 2) / e0); er_e = float(np.sum(np.abs(psi_e) ** 2) / e0)
    curv = float(gd.curvature_max_only(psi_e, par, dx))
    bridge_s, n_s = _bridge_cond(psi_s, par, dx, N)
    n_e = len(td.detect_nodes(psi_e, dx))
    rej = None
    if not (ER_LO <= er_s <= ER_HI and ER_LO <= er_e <= ER_HI): rej = "energy_drift"
    elif curv >= CURV_MAX: rej = "curvature_runaway"
    elif n_s < NODE_LO: rej = "single_or_dissipative"
    elif n_s > NODE_HI or n_e > NODE_HI: rej = "fragmented"
    elif abs(n_e - n_s) > 2: rej = "node_count_unstable"
    elif bridge_s > BRIDGE_HI: rej = "bridge_saturated"
    base = {"er_s": er_s, "er_e": er_e, "curv": curv, "bridge_s": bridge_s, "n_s": n_s, "n_e": n_e, "amp": amp}
    if rej:
        return {"reject": rej, "score": 0.0, "klass": "SUBSTRATE_REJECT", **base}
    # substrate-quality score (NO routing/coupling)
    e_clean = float(np.exp(-(np.log(max(er_s, 1e-9)) ** 2) / (2 * 0.4 ** 2))
                    * np.exp(-(np.log(max(er_e, 1e-9)) ** 2) / (2 * 0.4 ** 2)))
    node_stab = float(np.clip((n_s - 1) / 2, 0, 1) * np.clip((9 - n_s) / 3, 0, 1) * (1.0 if abs(n_e - n_s) <= 1 else 0.5))
    bridge_q = float(np.clip((bridge_s - BRIDGE_LO) / (0.5 - BRIDGE_LO), 0, 1))   # full reward by cond~0.5
    score = e_clean * (node_stab + 2.0 * bridge_q)
    klass = ("STRONG_BRIDGE_SUBSTRATE" if bridge_s >= STRONG else
             "BOUNDED_WEB" if bridge_s >= BRIDGE_LO else "BOUNDED_NO_BRIDGE")
    return {"reject": None, "score": float(score), "klass": klass, **base}


def cfg_hash(vec):
    return hashlib.md5(np.round(np.asarray(vec), 4).tobytes()).hexdigest()[:8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--N", type=int, default=48)
    ap.add_argument("--elite", type=int, default=6)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--bounds-file", default=os.path.join(ROOT, "jax_scout", "gain_bounds.json"))
    ap.add_argument("--outdir", default=os.path.join(ROOT, "sweep_runs"))
    args = ap.parse_args()
    bounds = json.load(open(args.bounds_file))
    lo = np.array([bounds[k][0] for k in order] + [A_BOUNDS[k][0] for k in A_ORDER])
    hi = np.array([bounds[k][1] for k in order] + [A_BOUNDS[k][1] for k in A_ORDER])
    D = len(order) + len(A_ORDER); rng = np.random.default_rng(SEED)
    sample = lambda n: [lo + rng.random(D) * (hi - lo) for _ in range(n)]

    def split(vec):
        par = {k: float(vec[i]) for i, k in enumerate(order)}
        g, kap, cA = (float(vec[len(order) + i]) for i in range(3))
        return par, g, kap, cA

    tag = "SUBSTRATE_CALIB" if args.calibrate else "SUBSTRATE_HUNT"
    outdir = os.path.join(args.outdir, f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(outdir, exist_ok=True)
    log = open(os.path.join(outdir, "all_evals.csv"), "w", newline="")
    cols = ["gen", "hash", "klass", "score", "reject", "er_s", "er_e", "curv", "bridge_s", "n_s", "n_e", "amp", *order, *A_ORDER]
    cw = csv.DictWriter(log, fieldnames=cols, extrasaction="ignore"); cw.writeheader()

    pop = sample(args.pop)
    max_gen = 2 if args.calibrate else 10 ** 9
    deadline = float("inf") if args.calibrate else time.time() + args.hours * 3600
    t0 = time.time(); gen = 0; n_eval = 0; klass_counts = {}; reject_counts = {}; best = []
    frozen = {}   # hash -> unique STRONG-bridge substrate record (preserved across gens)
    while time.time() < deadline and gen < max_gen:
        gen += 1
        for vec in pop:
            par, g, kap, cA = split(vec); h = cfg_hash(vec)
            try:
                r = evaluate_one(par, g, kap, cA, args.N)
            except Exception:
                r = {"reject": "eval_error", "score": 0.0, "klass": "NONFINITE_REJECT"}
            n_eval += 1
            klass_counts[r["klass"]] = klass_counts.get(r["klass"], 0) + 1
            reject_counts[r.get("reject") or "accepted"] = reject_counts.get(r.get("reject") or "accepted", 0) + 1
            row = {"gen": gen, "hash": h, "klass": r["klass"], "score": round(r["score"], 4), "reject": r.get("reject") or "",
                   **{k: round(float(vec[i]), 4) for i, k in enumerate(order)},
                   **{k: round(float(vec[len(order) + i]), 4) for i, k in enumerate(A_ORDER)}}
            for k in ("er_s", "er_e", "curv", "bridge_s", "n_s", "n_e", "amp"):
                if k in r: row[k] = round(r[k], 4) if isinstance(r[k], float) else r[k]
            cw.writerow(row)
            if not r.get("reject") and r["score"] > 0.05:
                best.append((r["score"], gen, vec, r))
            if not r.get("reject") and r.get("bridge_s", 0) >= STRONG and h not in frozen:
                frozen[h] = {"hash": h, "gen": gen, "score": r["score"], "bridge_s": r["bridge_s"],
                             "er_s": r["er_s"], "er_e": r["er_e"], "curv": r["curv"], "n_s": r["n_s"], "n_e": r["n_e"],
                             "params": {k: float(vec[i]) for i, k in enumerate(order)},
                             "gamma_A": float(vec[len(order)]), "kappa": float(vec[len(order) + 1]), "c_A": float(vec[len(order) + 2])}
        log.flush(); best.sort(key=lambda x: -x[0]); best = best[:25]
        elites = [b[2] for b in best[:args.elite]]; newpop = []
        if elites and not args.calibrate:
            for _ in range(args.pop - args.pop // 3):
                pa = elites[rng.integers(len(elites))]; pb = elites[rng.integers(len(elites))]
                cx = np.where(rng.random(D) < 0.5, pa, pb)
                newpop.append(np.clip(cx + rng.normal(0, 0.12, D) * (hi - lo), lo, hi))
        newpop += sample(args.pop - len(newpop)); pop = newpop
        el = (time.time() - t0) / 3600; bc = best[0] if best else (0, 0, None, {})
        print(f"[gen {gen} t={el:.2f}h] evals={n_eval} best={bc[0]:.3f} frozen_strong={len(frozen)} "
              f"klass={klass_counts} reject={({k: v for k, v in reject_counts.items() if k != 'accepted'})}", flush=True)
        json.dump({"gen": gen, "elapsed_h": el, "n_eval": n_eval, "contract": CONTRACT_KEY, "settle": SETTLE,
                   "horizon": HORIZON, "n_frozen_strong": len(frozen), "klass_counts": klass_counts,
                   "reject_counts": reject_counts,
                   "best": [{"score": b[0], "gen": b[1], **b[3], "hash": cfg_hash(b[2]),
                             **{k: float(b[2][i]) for i, k in enumerate(order)},
                             **{k: float(b[2][len(order) + i]) for i, k in enumerate(A_ORDER)}} for b in best[:10]]},
                  open(os.path.join(outdir, "status.json"), "w"), indent=2, default=float)
        json.dump({"contract": CONTRACT_KEY, "settle": SETTLE, "horizon": HORIZON, "strong_threshold": STRONG,
                   "n_unique_strong": len(frozen), "substrates": sorted(frozen.values(), key=lambda x: -x["bridge_s"])},
                  open(os.path.join(outdir, "frozen_substrates.json"), "w"), indent=2, default=float)

    log.close()
    print(f"\n=== {tag} DONE: {gen} gens, {n_eval} evals in {(time.time()-t0)/3600:.2f}h ===")
    print(f"klass: {klass_counts}\nreject: {reject_counts}")
    print(f"UNIQUE STRONG-BRIDGE SUBSTRATES frozen (cond_settle>={STRONG}, bounded, routing-horizon-stable): {len(frozen)}")
    print(f"-> {outdir}\n   next: validate with afield_routing_validate_pool.py --pool {os.path.join(outdir, 'frozen_substrates.json')}")


if __name__ == "__main__":
    main()
