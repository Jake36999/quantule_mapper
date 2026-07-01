"""feb param-basin edge confirmation — seed-robustness of the sensitive-axis window boundaries.

Before committing the joint (param_a, param_eta, param_rho_vac) grid ranges, confirm the OAT window
edges are seed-robust. Re-runs the edge cells at 2 NEW seeds (20260620, 20260621) and compares to the
seed-20260619 OAT reference. Fully brackets the critical param_a window (both TRUE edges + both reject
neighbors) and checks the eta / rho_vac TRUE edges.

Fixed feb params except the one perturbed axis; K=6 / per-blob / N=96 / T=12000 / classifier v3.
Read-only w.r.t. physics (css.run_probe/classify). Resumable (--out skips done) + deadline-aware.
WSL2 jax venv:  python jax_scout/feb_param_edge_confirm.py [--out DIR] [--deadline-hours 4.5]
"""
import os, sys, csv, json, time, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import core_saturation_search as css

N, T = 96, 12000
SEEDS = [20260620, 20260621]
# (param, factor, OAT seed-20260619 reference class)
EDGE_CELLS = [
    ("param_a", 0.75, "SPIN_DOWN_REJECT"),       # critical window: lower reject neighbor
    ("param_a", 0.9,  "TRUE_SATURATED_BOUND_STATE"),  # critical window: lower TRUE edge
    ("param_a", 1.1,  "TRUE_SATURATED_BOUND_STATE"),  # critical window: upper TRUE edge
    ("param_a", 1.25, "TRANSIENT_GROWER_REJECT"),     # critical window: upper reject neighbor
    ("param_eta", 0.75, "TRUE_SATURATED_BOUND_STATE"),
    ("param_eta", 1.25, "TRUE_SATURATED_BOUND_STATE"),
    ("param_rho_vac", 0.75, "TRUE_SATURATED_BOUND_STATE"),
]
COLS = ["key", "param", "factor", "seed", "ref_class_seed619", "klass", "match", "er_fin", "er_max",
        "late_drift", "floor_ratio", "bounded_breathing", "n_fin", "held_mass", "wallclock_min", "error"]


def configs():
    cfgs = []
    for p, f, ref in EDGE_CELLS:
        for seed in SEEDS:
            pp = dict(css.FEB); pp[p] = float(css.FEB[p]) * f
            cfgs.append({"key": f"{p}_x{f}_s{seed}", "param": p, "factor": f, "seed": seed,
                         "ref": ref, "params": pp})
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--deadline-hours", type=float, default=4.5)
    args = ap.parse_args()
    out = args.out or os.path.join(ROOT, "sweep_runs", f"FEB_PARAM_EDGE_CONFIRM_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)

    rows, done = [], set()
    csv_path = os.path.join(out, "feb_param_edge_confirm_results.csv")
    if os.path.exists(csv_path):
        for r in csv.DictReader(open(csv_path, newline="")):
            if r.get("klass") and r["klass"] != "ERROR":
                rows.append(r); done.add(r["key"])
        if done:
            print(f"RESUME: skipping {len(done)} done", flush=True)

    def flush():
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
        with open(os.path.join(out, "feb_param_edge_confirm_summary.json"), "w") as fh:
            json.dump({"N": N, "T": T, "seeds": SEEDS, "feb_params": css.FEB,
                       "classifier": css.classifier_spec(), "rows": rows}, fh, indent=2, default=float)

    todo = [c for c in configs() if c["key"] not in done]
    start = time.time(); deadline = start + args.deadline_hours * 3600
    print(f"=== FEB PARAM EDGE CONFIRM | classifier={css.classifier_spec()['version']} | {len(todo)} todo ===", flush=True)
    for i, c in enumerate(todo, 1):
        if time.time() + (T / 1000.0 + 1.5) * 60 > deadline:
            print(f"[{i}/{len(todo)}] SKIP {c['key']} (deadline)", flush=True); continue
        t0 = time.time()
        try:
            r = css.run_probe(c["params"], N, T, 6, seed=c["seed"], ic_norm=css.IC_NORM_PER_BLOB_FIXED)
            m = r["metrics"]; pf = np.asarray(r["psi_fin"])
            held = float(np.sum(np.abs(pf) ** 2)) if np.isfinite(pf).all() else float("inf")
            match = (r["klass"] == c["ref"])
            row = {"key": c["key"], "param": c["param"], "factor": c["factor"], "seed": c["seed"],
                   "ref_class_seed619": c["ref"], "klass": r["klass"], "match": match,
                   "er_fin": m["er_fin"], "er_max": m["er_max"], "late_drift": m.get("late_drift"),
                   "floor_ratio": m.get("floor_ratio"), "bounded_breathing": m.get("bounded_breathing"),
                   "n_fin": m["n_fin"], "held_mass": held, "wallclock_min": round((time.time() - t0) / 60.0, 1)}
            print(f"[{i}/{len(todo)}] {c['key']:28} -> {r['klass']:26} match={match} (ref {c['ref'][:9]}) ({row['wallclock_min']}m)", flush=True)
        except Exception as exc:
            row = {"key": c["key"], "param": c["param"], "factor": c["factor"], "seed": c["seed"],
                   "ref_class_seed619": c["ref"], "klass": "ERROR", "error": str(exc)[:200]}
            print(f"[{i}/{len(todo)}] {c['key']} ERROR {exc}", flush=True)
        rows.append(row); flush()
    ok = [r for r in rows if r.get("klass") not in ("ERROR", None)]
    matches = sum(1 for r in ok if str(r.get("match")) == "True")
    print(f"=== DONE: {matches}/{len(ok)} match seed-619 reference, {round((time.time()-start)/3600,2)}h ===", flush=True)
    print(out)


if __name__ == "__main__":
    main()
