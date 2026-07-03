"""
VALIDATION TIER after the fresh substrate hunt: apply the corrected denominator-safe routing gate
(afield_routing_gate) to the EXPANDED pool of unique strong-bridge substrates (frozen_substrates.json)
+ weak/no-bridge controls. Single-seed screen at CONT=2000, escalate any screen-pass to a 3-seed
robustness verdict. This is the N-expanded test of whether ANY strong geometric bridge routes.

Outcome:
  * 0 candidates pass + controls clean -> STRONG_BRIDGE_ROUTING_NO_SUPPORT_CORRECTED_GATE_N_EXPANDED
    (the Payan RFC becomes the justified next active direction)
  * any candidate passes -> VALID_ROUTING_CANDIDATE_UNDER_MECHANISM_VALIDATION (do NOT jump to Payan;
    run the deeper mechanism-validation tier: exact rerun, altered seeds, longer window,
    bridge/node/void perturbation map, phase scramble, density-preserved phase-randomized control,
    bridge weakening/removal, parameter-basin test, resolution/timestep scaling).

WSL2 jax venv:
  python /mnt/f/quantule_mapper/jax_scout/afield_routing_validate_pool.py --pool <SUBSTRATE_HUNT_dir>/frozen_substrates.json --top 15
"""
import os, sys, json, glob, time, argparse, csv
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
import jax
jax.config.update("jax_enable_x64", True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import physics
import jax_scout.afield_routing_gate as G

order = physics.SWEEP_PARAM_ORDER


def load_pool(pool_path, top):
    data = json.load(open(pool_path))
    subs = data.get("substrates", [])
    subs = sorted(subs, key=lambda s: -s.get("bridge_s", 0.0))[:top]
    out = []
    for s in subs:
        par = {k: float(s["params"][k]) for k in order}
        out.append((par, float(s["gamma_A"]), float(s["kappa"]), float(s["c_A"]),
                    f"{s.get('hash','?')}_br{s.get('bridge_s',0):.2f}", "candidate"))
    return out


def load_pool_from_csv(csv_path, top):
    """Fallback when the frozen-strong pool is empty: take the strongest AVAILABLE bounded substrates
    (highest bridge_s at the routing settle) from the substrate hunt's all_evals.csv, deduped."""
    rows = list(csv.DictReader(open(csv_path)))
    def f(r, k):
        try: return float(r[k])
        except: return float("nan")
    acc = [r for r in rows if r.get("reject", "") == "" and 2 <= f(r, "n_s") <= 8 and 0.5 <= f(r, "er_s") <= 2.0]
    acc = sorted(acc, key=lambda r: -f(r, "bridge_s"))
    seen, out = set(), []
    for r in acc:
        h = r.get("hash", "")
        if h in seen:
            continue
        seen.add(h)
        par = {k: f(r, k) for k in order}
        out.append((par, f(r, "gamma_A"), f(r, "kappa"), f(r, "c_A"),
                    f"{h}_br{f(r,'bridge_s'):.2f}", "candidate"))
        if len(out) >= top:
            break
    return out


def load_controls():
    """Reuse the established weak + no-bridge controls from the prior bridge hunt CSV."""
    d = sorted(glob.glob(os.path.join(ROOT, "sweep_runs", "AF_BRIDGE_HUNT_2026*")))[-1]
    known = G.pick_known(d, n_strong=0)
    return [k for k in known if k[5] in ("weak_control", "no_bridge_control")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", help="frozen_substrates.json from afield_substrate_hunt")
    ap.add_argument("--csv", help="all_evals.csv (fallback: strongest available substrates by bridge_s)")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    if args.csv:
        cand = load_pool_from_csv(args.csv, args.top)
        pool_src = args.csv
    else:
        cand = load_pool(args.pool, args.top)
        pool_src = args.pool
    pool = cand + load_controls()
    n_cand = sum(1 for *_, role in pool if role == "candidate")
    print(f"=== ROUTING VALIDATION (N-EXPANDED) — corrected gate cont={G.HUNT_CONT}, "
          f"abs_bridge>={G.ABS_BRIDGE_FLOOR}, void_denom>={G.VOID_DENOM_FLOOR}, resolved={G.REQUIRE_RESOLVED}, "
          f"bridge/node>={G.BRIDGE_NODE_THR} ===")
    print(f"pool: {n_cand} unique strong-bridge substrates + weak/no-bridge controls "
          f"(single-seed screen -> escalate passers to 3 seeds)\n")
    results = []; reasons = {}; routing = []; control_leak = []
    for par, g, kap, cA, label, role in pool:
        t0 = time.time()
        m = G.measure(par, g, kap, cA, seed=G.SEED, cont=G.HUNT_CONT)
        p, why = G.gate_one(m)
        reasons[why] = reasons.get(why, 0) + 1
        row = {"label": label, "role": role, "screen_pass": bool(p), "reason": why,
               **{k: m.get(k) for k in ("bridge_cond", "bridge_amp", "node_amp", "void_amp",
                                        "bridge_resolved", "er", "n_nodes", "n_nodes_end", "status")}}
        if p:
            cl = G.classify(par, g, kap, cA, role=role, seeds=G.SEEDS, cont=G.HUNT_CONT)
            row["multiseed"] = cl; row["klass"] = cl["klass"]
            if role == "candidate" and cl["klass"] == "ROUTING_CANDIDATE_WITH_VALID_DENOMINATOR":
                routing.append(label)
            if role != "candidate" and cl["klass"] == "CONTROL_LEAK_REJECT":
                control_leak.append(label)
        else:
            row["klass"] = "CONTROL_CLEAN" if role != "candidate" else why + "_REJECT"
        results.append(row)
        amp = (f"bridge={m.get('bridge_amp'):.4f} node={m.get('node_amp'):.4f} void={m.get('void_amp'):.5f} "
               f"res={m.get('bridge_resolved')}") if m.get("status") == "ok" else m.get("status")
        print(f"[{label}] role={role} cond={m.get('bridge_cond')} {amp} -> {row['klass']}  ({time.time()-t0:.0f}s)", flush=True)

    n_routing = len(routing)
    if n_routing > 0:
        verdict = "VALID_ROUTING_CANDIDATE_UNDER_MECHANISM_VALIDATION"
    elif not control_leak:
        verdict = "STRONG_BRIDGE_ROUTING_NO_SUPPORT_CORRECTED_GATE_N_EXPANDED"
    else:
        verdict = "METRIC_LEAK_INCONCLUSIVE"
    outdir = os.path.dirname(os.path.abspath(pool_src))
    out = {"contract_key": G.cc.CONTRACT_KEY, "pool": pool_src, "n_candidates": n_cand,
           "n_routing_candidates": n_routing, "routing": routing, "control_leak": control_leak,
           "screen_reason_histogram": reasons, "verdict": verdict, "results": results}
    od = os.path.join(outdir, "routing_validation_expanded.json")
    json.dump(out, open(od, "w"), indent=2, default=float)
    print(f"\n=== {n_cand} substrates validated; ROUTING_CANDIDATE: {n_routing}; control leaks: {len(control_leak)} ===")
    print(f"screen reasons: {reasons}\nVERDICT: {verdict}\nwrote {od}")
    if verdict == "STRONG_BRIDGE_ROUTING_NO_SUPPORT_CORRECTED_GATE_N_EXPANDED":
        print("N-expanded confirms: strong geometric bridges do not route -> Payan RFC is the justified next direction.")
    elif verdict == "VALID_ROUTING_CANDIDATE_UNDER_MECHANISM_VALIDATION":
        print("A candidate PASSED -> do NOT jump to Payan; run the deeper mechanism-validation tier (see module docstring).")


if __name__ == "__main__":
    main()
