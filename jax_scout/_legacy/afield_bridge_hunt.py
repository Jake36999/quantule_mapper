"""
STAGE 2 — A-coupled bridge hunt (current-coupled A-field IN from the start).

Stage 1 showed the gamma_A=0 finalists are the WRONG SUBSTRATE: they are energy-unstable by 1600
steps (er~3) regardless of A, so the A-field rescue cannot be tested cleanly on them. The right
candidates for gamma_A>0 likely live in a different basin. This hunt searches the baseline params
PLUS the A-field params (gamma_A, kappa, c_A) from the start, and selects directly for a GOOD
substrate that maintains directed transfer to 1600 steps under A-on.

OBJECTIVE rewards (cheap proxies from one A-coupled 1600-step trajectory):
  * energy-conserved AT 1600 (er in [0.5,2]) -- the substrate fix, NOT 800-step-only;
  * stable few-node structure (2-8 nodes, not space-filling);
  * a real but NON-SATURATED bridge (0.15 < maxCond < 0.85);
  * phase coupling that PERSISTS to 1600 above the 0.73 floor (not a transient);
  * A localized along the bridge (A_bridge_loc > 1), A-energy bounded (not ballooning).
HARD-REJECT (cannot be gamed): non-finite, amp/A runaway, energy growth/collapse, curvature
  runaway, bridge saturation (>0.85), fragmentation (>8 nodes).
NOT rewarded: raw J_flux, global-web response, energy growth, bridge saturation, space-filling,
  800-step-only coupling, aggregate phase coupling without persistence.
The EXPENSIVE web->wires metric (global_mode_fraction / routing) is applied at the validation
tier to finalists, NOT in the inner loop.

CAUTION: ACTIVE A-coupled branch (IRER-SNCGL-CURRENT-COUPLED-AFFECT-ETDRK4-v1). Default-off
elsewhere, contract-stamped, segregated from gamma_A=0 rankings. JAX scout; CuPy has no
current-coupled term -> NOT for CuPy/Hunter promotion. Not proof.

Run (WSL2 jax venv):
  calibration:  python jax_scout/afield_bridge_hunt.py --calibrate
  full hunt:    python jax_scout/afield_bridge_hunt.py --hours 3
"""
import os, sys, csv, json, time, argparse
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
FLOOR = td.THR_PHASECOUP
order = physics.SWEEP_PARAM_ORDER
A_ORDER = ["gamma_A", "kappa", "c_A"]
# A-field search bounds (gamma_A>0.3 distorts; kappa>4 + slow c_A runs away; from Stage 1 map)
A_BOUNDS = {"gamma_A": [0.0, 0.3], "kappa": [0.5, 3.0], "c_A": [0.5, 3.0]}
# gates
ER_LO, ER_HI, CURV_MAX, AMP_MAX, A_E_MAX = 0.5, 2.0, 1.0, 1e3, 5e2
NODE_LO, NODE_HI, BRIDGE_LO, BRIDGE_HI = 2, 8, 0.15, 0.85
STEPS, NSNAP, PRE_STEPS, PRE_NSNAP = 1600, 40, 400, 10


L_ = cc.L
from jax_scout._legacy.afield_current_tune import a_bridge_localization as cc_a_bridge_loc  # noqa: E402


def evaluate_one(par, g, kap, cA, N=48):
    ic = cc.multiseed_ic(N, SEED)
    # cheap pre-gate (short run): reject obvious non-finite/blow-up/dissipation
    ps, As, fin = cc.capture_cc(par, ic, g, N, PRE_STEPS, PRE_NSNAP, kappa=kap, c_A=cA)
    if not fin or float(np.max(np.abs(ps[-1]))) > AMP_MAX:
        return {"reject": "pre_runaway", "score": 0.0, "klass": "A_RUNAWAY_REJECT"}
    # full eval
    snaps, Asnaps, fin = cc.capture_cc(par, ic, g, N, STEPS, NSNAP, kappa=kap, c_A=cA)
    amp = float(np.max(np.abs(snaps[-1]))) if fin else float("inf")
    if not fin or amp > AMP_MAX:
        return {"reject": "runaway", "score": 0.0, "klass": "A_RUNAWAY_REJECT"}
    er = float(np.sum(np.abs(snaps[-1])**2)/(np.sum(np.abs(snaps[0])**2)+1e-30))
    curv = gd.curvature_max_only(snaps[-1], par, L_/N)
    A_E = float(np.sum(Asnaps[-1]))
    a = cc.analyze(snaps, par, N)
    nodes = a["n_persistent_nodes"]; bridge = a["max_cond"]; pcoup = a["phase_coupling_score"]; exch = a["energy_exchange_index"]
    A_loc = cc_a_bridge_loc(Asnaps[-1], snaps[-1], par, N)
    # hard gates
    rej = None
    if not (ER_LO <= er <= ER_HI): rej = "energy_unbounded"
    elif curv >= CURV_MAX: rej = "curvature_runaway"
    elif A_E > A_E_MAX or not np.isfinite(A_E): rej = "A_runaway"
    elif nodes < NODE_LO: rej = "single_or_dissipative"
    elif nodes > NODE_HI: rej = "fragmented_spacefilling"
    elif bridge > BRIDGE_HI: rej = "bridge_saturated"
    if rej:
        return {"reject": rej, "score": 0.0, "klass": "A_DISTORT_REJECT", "er": er, "curv": float(curv),
                "nodes": nodes, "bridge": bridge, "pcoup": pcoup, "exch": exch, "A_E": A_E, "A_loc": A_loc}
    # score (only for gated-clean configs)
    e_clean = float(np.exp(-(np.log(max(er, 1e-9))**2)/(2*0.4**2)))     # peak er=1
    node_stab = float(np.clip((nodes-1)/2, 0, 1)*np.clip((9-nodes)/3, 0, 1))
    bridge_q = float(np.clip((bridge-BRIDGE_LO)/(0.5-BRIDGE_LO), 0, 1))  # reward up to ~0.5, not saturation
    persist = max(0.0, pcoup - FLOOR)                                    # transfer PERSISTS to 1600 above floor
    a_loc_bonus = float(np.clip((A_loc-1.0), 0, 2)) if np.isfinite(A_loc) else 0.0
    score = e_clean*(node_stab + bridge_q + 3.0*persist + 0.5*a_loc_bonus + 0.5*exch)
    klass = ("A_PERSISTENT_TRANSFER_CANDIDATE" if pcoup > FLOOR else
             "A_BOUNDED_WEB" if bridge >= BRIDGE_LO else "A_BOUNDED_NO_BRIDGE")
    return {"reject": None, "score": float(score), "klass": klass, "er": er, "curv": float(curv),
            "nodes": nodes, "bridge": bridge, "pcoup": pcoup, "exch": exch, "A_E": A_E, "A_loc": float(A_loc) if np.isfinite(A_loc) else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--N", type=int, default=48)
    ap.add_argument("--elite", type=int, default=6)
    ap.add_argument("--calibrate", action="store_true", help="2 gens, no timebox")
    ap.add_argument("--bounds-file", default=os.path.join(ROOT, "jax_scout", "gain_bounds.json"))
    ap.add_argument("--outdir", default=os.path.join(ROOT, "sweep_runs"))
    args = ap.parse_args()
    bounds = json.load(open(args.bounds_file))
    lo = np.array([bounds[k][0] for k in order] + [A_BOUNDS[k][0] for k in A_ORDER])
    hi = np.array([bounds[k][1] for k in order] + [A_BOUNDS[k][1] for k in A_ORDER])
    D = len(order) + len(A_ORDER)
    rng = np.random.default_rng(SEED)

    def sample(n):
        u = rng.random((n, D)); return [lo + u[i]*(hi-lo) for i in range(n)]

    def split(vec):
        par = {k: float(vec[i]) for i, k in enumerate(order)}
        g, kap, cA = (float(vec[len(order)+i]) for i in range(3))
        return par, g, kap, cA

    tag = "AF_BRIDGE_CALIB" if args.calibrate else "AF_BRIDGE_HUNT"
    outdir = os.path.join(args.outdir, f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(outdir, exist_ok=True)
    log = open(os.path.join(outdir, "all_evals.csv"), "w", newline="")
    cols = ["gen", "klass", "score", "reject", "er", "curv", "nodes", "bridge", "pcoup", "exch",
            "A_E", "A_loc", *order, *A_ORDER]
    cw = csv.DictWriter(log, fieldnames=cols, extrasaction="ignore"); cw.writeheader()

    pop = sample(args.pop)
    max_gen = 2 if args.calibrate else 10**9
    deadline = float("inf") if args.calibrate else time.time()+args.hours*3600
    t0 = time.time(); gen = 0; n_eval = 0; klass_counts = {}; reject_counts = {}; best = []
    while time.time() < deadline and gen < max_gen:
        gen += 1
        for vec in pop:
            par, g, kap, cA = split(vec)
            try:
                r = evaluate_one(par, g, kap, cA, args.N)
            except Exception as e:
                r = {"reject": "eval_error", "score": 0.0, "klass": "A_RUNAWAY_REJECT"}
            n_eval += 1
            klass_counts[r["klass"]] = klass_counts.get(r["klass"], 0)+1
            reject_counts[r.get("reject") or "accepted"] = reject_counts.get(r.get("reject") or "accepted", 0)+1
            row = {"gen": gen, "klass": r["klass"], "score": r["score"], "reject": r.get("reject") or "",
                   **{k: round(v, 4) for k, v in zip(order, vec)},
                   **{k: round(float(vec[len(order)+i]), 4) for i, k in enumerate(A_ORDER)}}
            for k in ("er", "curv", "nodes", "bridge", "pcoup", "exch", "A_E", "A_loc"):
                if k in r: row[k] = r[k]
            cw.writerow(row)
            if not r.get("reject") and r["score"] > 0.05:
                best.append((r["score"], gen, vec, r))
        log.flush(); best.sort(key=lambda x: -x[0]); best = best[:20]
        # next gen
        elites = [b[2] for b in best[:args.elite]]
        newpop = []
        if elites and not args.calibrate:
            for _ in range(args.pop - args.pop//3):
                pa = elites[rng.integers(len(elites))]; pb = elites[rng.integers(len(elites))]
                cx = np.where(rng.random(D) < 0.5, pa, pb)
                newpop.append(np.clip(cx + rng.normal(0, 0.12, D)*(hi-lo), lo, hi))
        newpop += sample(args.pop - len(newpop)); pop = newpop
        el = (time.time()-t0)/3600; bc = best[0] if best else (0, 0, None, {})
        print(f"[gen {gen} t={el:.2f}h] evals={n_eval} best_score={bc[0]:.3f} klass={klass_counts} "
              f"reject={({k: v for k, v in reject_counts.items() if k != 'accepted'})}", flush=True)
        json.dump({"gen": gen, "elapsed_h": el, "n_eval": n_eval, "contract": CONTRACT_KEY,
                   "klass_counts": klass_counts, "reject_counts": reject_counts,
                   "best": [{"score": b[0], "gen": b[1], **b[3],
                             **{k: float(b[2][i]) for i, k in enumerate(order)},
                             **{k: float(b[2][len(order)+i]) for i, k in enumerate(A_ORDER)}} for b in best[:10]]},
                  open(os.path.join(outdir, "status.json"), "w"), indent=2, default=float)

    log.close()
    print(f"\n=== {tag} DONE: {gen} gens, {n_eval} evals in {(time.time()-t0)/3600:.2f}h ===")
    print(f"klass: {klass_counts}\nreject: {reject_counts}")
    npc = klass_counts.get("A_PERSISTENT_TRANSFER_CANDIDATE", 0)
    print(f"A_PERSISTENT_TRANSFER_CANDIDATE (energy-bounded@1600 + phase coupling >floor under A-on): {npc}")
    if best:
        print("top 5:")
        for b in best[:5]:
            r = b[3]
            print(f"  score={b[0]:.3f} {r['klass']:34} er={r['er']:.2f} nodes={r['nodes']} bridge={r['bridge']:.3f} "
                  f"pcoup={r['pcoup']:.3f} A_loc={r['A_loc']:.2f} gA={b[2][len(order)]:.3f} kap={b[2][len(order)+1]:.2f} cA={b[2][len(order)+2]:.2f}")
    print(f"-> {outdir}")


if __name__ == "__main__":
    main()
