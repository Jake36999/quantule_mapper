"""
CORRECTED substrate/routing hunt (denominator-safe gate, validated current-coupled A; NO tensor).

The window scan established a window-robust NULL on the 4 known configs: no resolved, bridge-specific,
above-noise routing exists, and the bridge/void ratio is intrinsically invalid (no-bridge control
b/v=949). This confirms the finding across a BROADER pool of bounded strong-bridge substrates to make
the 'repeatedly: N strong bridges, 0 valid routing' statement (the evidence base for a Payan RFC).

Reuses the prior hunt's discovered substrates (sweep_runs/AF_BRIDGE_HUNT_*/all_evals.csv) rather than
re-running a multi-hour evolutionary search -- the bridges are already found; the question is whether
ANY of them route under the corrected gate (afield_routing_gate). Single-seed broad screen at
cont=2000; any screen-pass is escalated to a 3-seed robustness verdict. Controls (weak/no-bridge) run
through the SAME gate and must stay clean.

Classes (afield_routing_gate): ROUTING_CANDIDATE_WITH_VALID_DENOMINATOR / BOUNDED_STRONG_BRIDGE_SUBSTRATE
/ DENOMINATOR_COLLAPSE_REJECT / NOT_RESOLVED / FRAGMENTATION_REJECT / ENERGY_DRIFT_REJECT /
SEED_FRAGILE_REJECT / CONTROL_LEAK_REJECT. Outcome if 0 routing candidates + controls clean ->
PROMISING_FOR_PAYAN_PHASE_ALIGNMENT.

WSL2 jax venv:  python /mnt/f/quantule_mapper/jax_scout/afield_routing_hunt.py --n 18
"""
import os, sys, csv, json, glob, time, argparse
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
SEED = G.SEED


def F(r, k):
    try: return float(r[k])
    except: return float("nan")


def select_pool(d, n_strong):
    rows = list(csv.DictReader(open(os.path.join(d, "all_evals.csv"))))
    bnd = [r for r in rows if r["reject"] == "" and 0.5 <= F(r, "er") <= 2.0 and 2 <= F(r, "nodes") <= 8]
    # unique gens, strongest bounded bridges first
    strong = sorted([r for r in bnd if 0.3 < F(r, "bridge") < 0.85], key=lambda r: -F(r, "bridge"))
    seen, cand = set(), []
    for r in strong:
        if r["gen"] in seen:
            continue
        seen.add(r["gen"]); cand.append(r)
        if len(cand) >= n_strong:
            break
    weak = sorted([r for r in bnd if 0.1 <= F(r, "bridge") <= 0.25], key=lambda r: F(r, "bridge"))[:1]
    none = sorted([r for r in bnd if F(r, "bridge") < 0.05], key=lambda r: F(r, "bridge"))[:1]
    def pack(r, role):
        return ({k: F(r, k) for k in order}, F(r, "gamma_A"), F(r, "kappa"), F(r, "c_A"),
                f"gen{r['gen']}_br{F(r,'bridge'):.2f}", role)
    return ([pack(r, "candidate") for r in cand]
            + [pack(r, "weak_control") for r in weak] + [pack(r, "no_bridge_control") for r in none])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=18, help="number of strong-bridge substrate candidates")
    args = ap.parse_args()
    d = sorted(glob.glob(os.path.join(ROOT, "sweep_runs", "AF_BRIDGE_HUNT_2026*")))[-1]
    pool = select_pool(d, args.n)
    n_cand = sum(1 for *_, role in pool if role == "candidate")
    print(f"=== CORRECTED ROUTING HUNT (gate cont={G.HUNT_CONT}, abs_bridge>={G.ABS_BRIDGE_FLOOR}, "
          f"void_denom>={G.VOID_DENOM_FLOOR}, resolved={G.REQUIRE_RESOLVED}, bridge/node>={G.BRIDGE_NODE_THR}) ===")
    print(f"pool: {n_cand} bounded strong-bridge substrates + weak + no-bridge controls "
          f"(single-seed screen, escalate passers to 3 seeds)\n")
    results = []; reasons = {}; routing = []; control_leak = []
    for par, g, kap, cA, label, role in pool:
        t0 = time.time()
        m = G.measure(par, g, kap, cA, seed=SEED, cont=G.HUNT_CONT)
        p, why = G.gate_one(m)
        reasons[why] = reasons.get(why, 0) + 1
        amp = (f"bridge={m.get('bridge_amp', float('nan')):.4f} node={m.get('node_amp', float('nan')):.4f} "
               f"void={m.get('void_amp', float('nan')):.5f} res={m.get('bridge_resolved')}") if m.get("status") == "ok" else m.get("status")
        row = {"label": label, "role": role, "screen_pass": bool(p), "reason": why,
               "bridge_cond": m.get("bridge_cond"), **{k: m.get(k) for k in
               ("bridge_amp", "node_amp", "void_amp", "bridge_resolved", "er", "n_nodes", "n_nodes_end", "status")}}
        # escalate screen-passers to multi-seed robustness
        if p:
            cl = G.classify(par, g, kap, cA, role=role, seeds=G.SEEDS, cont=G.HUNT_CONT)
            row["multiseed"] = cl
            row["klass"] = cl["klass"]
            if role == "candidate" and cl["klass"] == "ROUTING_CANDIDATE_WITH_VALID_DENOMINATOR":
                routing.append(label)
            if role != "candidate" and cl["klass"] == "CONTROL_LEAK_REJECT":
                control_leak.append(label)
        else:
            row["klass"] = ("CONTROL_CLEAN" if role != "candidate" else
                            {"DENOMINATOR_COLLAPSE": "DENOMINATOR_COLLAPSE_REJECT",
                             "WEAK_ABS_BRIDGE": "DENOMINATOR_COLLAPSE_REJECT",
                             "NOT_RESOLVED": "NOT_RESOLVED_REJECT", "FRAGMENTATION": "FRAGMENTATION_REJECT",
                             "CURVATURE_RUNAWAY": "CURVATURE_RUNAWAY_REJECT", "ENERGY_DRIFT": "ENERGY_DRIFT_REJECT",
                             "WEAK_BRIDGE_VOID": "WEAK_BRIDGE_VOID_REJECT", "WEAK_BRIDGE_NODE": "WEAK_BRIDGE_NODE_REJECT"}
                            .get(why, "SUBSTRATE_UNUSABLE"))
        results.append(row)
        print(f"[{label}] role={role} cond={m.get('bridge_cond')} {amp} -> {row['klass']} "
              f"(screen={why})  ({time.time()-t0:.0f}s)", flush=True)

    n_routing = len(routing)
    verdict = ("ROUTING_FOUND" if n_routing > 0 else
               "PROMISING_FOR_PAYAN_PHASE_ALIGNMENT" if not control_leak else
               "METRIC_LEAK_INCONCLUSIVE")
    out = {"contract_key": G.cc.CONTRACT_KEY, "gate": {"cont": G.HUNT_CONT, "abs_bridge_floor": G.ABS_BRIDGE_FLOOR,
           "void_denom_floor": G.VOID_DENOM_FLOOR, "require_resolved": G.REQUIRE_RESOLVED,
           "bridge_void_thr": G.BRIDGE_VOID_THR, "bridge_node_thr": G.BRIDGE_NODE_THR},
           "n_candidates": n_cand, "n_routing_candidates": n_routing, "routing": routing,
           "control_leak": control_leak, "screen_reason_histogram": reasons, "verdict": verdict, "results": results}
    od = os.path.join(d, "afield_routing_hunt.json")
    json.dump(out, open(od, "w"), indent=2, default=float)
    print(f"\n=== {n_cand} strong-bridge substrates screened; ROUTING_CANDIDATE_WITH_VALID_DENOMINATOR: "
          f"{n_routing}; control leaks: {len(control_leak)} ===")
    print(f"screen reasons: {reasons}")
    print(f"VERDICT: {verdict}")
    print(f"wrote {od}")
    if verdict == "PROMISING_FOR_PAYAN_PHASE_ALIGNMENT":
        print("Strong geometric bridges, no valid selective routing under the denominator-safe gate, "
              "controls clean -> evidence base for the Payan-state / phase-alignment RFC.")


if __name__ == "__main__":
    main()
