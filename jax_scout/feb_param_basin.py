"""feb parameter-neighborhood basin map + node-family long-T revalidation (DO NOT auto-run).

Maps how wide feb's bound-state basin is in PARAMETER space (the finding was: bound-state-ness is a
parameter-regime property), and revalidates the discovered node-count family at T=24000 under the
breathing-aware v3 gate. Anchor = feb params (css.FEB); K=6 / per-blob-fixed unless noted; N=96.
Classifier v3. Read-only w.r.t. physics: uses css.run_probe/classify unchanged — no PDE/solver/geometry
change. Resumable (--out skips completed keys) and deadline-bounded (daytime-teardown safe).

  P1  revalidate node family  : K in {3,4,5,6,8}, feb params, per-blob, T=24000  (long-T under v3)
  P2  parameter OAT basin map : each of feb's 8 params perturbed one-at-a-time, K6/per-blob, T=12000

Plan: docs/PHASE_C_PARAM_BASIN_BATTERY_PLAN.md.
WSL2 jax venv:  python jax_scout/feb_param_basin.py [--out DIR] [--deadline-hours 11.5]
"""
import os, sys, csv, json, time, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import core_saturation_search as css

N, SEED = 96, 20260619
FACTORS = [0.5, 0.75, 0.9, 1.1, 1.25, 1.5]
OMEGA0_SET = [0.05, 0.1, 0.2, 0.4]          # feb omega0 = 0; multiplicative is degenerate -> absolute steps
COLS = ["key", "phase", "param", "value", "K", "T", "klass", "er_fin", "er_max", "er_min",
        "floor_ratio", "late_drift", "bounded_breathing", "n_fin", "core_fin", "held_mass", "wallclock_min", "error"]


def configs():
    cfgs = []
    # P1 — node-family long-T revalidation under v3
    for K in [3, 4, 5, 6, 8]:
        cfgs.append({"key": f"revalid_K{K}_T24000", "phase": "P1_revalidate", "param": "K", "value": K,
                     "params": dict(css.FEB), "K": K, "T": 24000})
    # P2 — parameter OAT basin map (T=12000, K6)
    cfgs.append({"key": "center_feb_T12000", "phase": "P2_center", "param": "(none)", "value": 1.0,
                 "params": dict(css.FEB), "K": 6, "T": 12000})
    for p in css.order:
        if p == "param_omega0":
            for v in OMEGA0_SET:
                pp = dict(css.FEB); pp[p] = float(v)
                cfgs.append({"key": f"OAT_{p}_set{v}", "phase": "P2_OAT", "param": p, "value": float(v),
                             "params": pp, "K": 6, "T": 12000})
        else:
            for f in FACTORS:
                pp = dict(css.FEB); pp[p] = float(css.FEB[p]) * f
                cfgs.append({"key": f"OAT_{p}_x{f}", "phase": "P2_OAT", "param": p, "value": float(css.FEB[p]) * f,
                             "params": pp, "K": 6, "T": 12000})
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--deadline-hours", type=float, default=11.5)
    args = ap.parse_args()
    out = args.out or os.path.join(ROOT, "sweep_runs", f"FEB_PARAM_BASIN_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)

    rows, done = [], set()
    csv_path = os.path.join(out, "feb_param_basin_results.csv")
    if os.path.exists(csv_path):
        for r in csv.DictReader(open(csv_path, newline="")):
            if r.get("klass") and r["klass"] != "ERROR":
                rows.append(r); done.add(r["key"])
        if done:
            print(f"RESUME: skipping {len(done)} done", flush=True)

    def flush():
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
        with open(os.path.join(out, "feb_param_basin_summary.json"), "w") as fh:
            json.dump({"N": N, "seed": SEED, "feb_params": css.FEB, "factors": FACTORS,
                       "classifier": css.classifier_spec(), "rows": rows}, fh, indent=2, default=float)

    todo = [c for c in configs() if c["key"] not in done]
    start = time.time(); deadline = start + args.deadline_hours * 3600
    print(f"=== FEB PARAM BASIN | classifier={css.classifier_spec()['version']} | {len(todo)} todo | "
          f"deadline {args.deadline_hours}h ===", flush=True)
    for i, c in enumerate(todo, 1):
        est = c["T"] / 1000.0 + 1.5
        if time.time() + est * 60 > deadline:
            print(f"[{i}/{len(todo)}] SKIP {c['key']} (deadline)", flush=True)
            continue
        t0 = time.time()
        try:
            r = css.run_probe(c["params"], N, c["T"], c["K"], seed=SEED, ic_norm=css.IC_NORM_PER_BLOB_FIXED)
            m = r["metrics"]; pf = np.asarray(r["psi_fin"])
            held = float(np.sum(np.abs(pf) ** 2)) if np.isfinite(pf).all() else float("inf")
            np.savez_compressed(os.path.join(out, c["key"] + "_probe.npz"), psi_fin=pf, er=r["er"])
            row = {"key": c["key"], "phase": c["phase"], "param": c["param"], "value": c["value"], "K": c["K"],
                   "T": c["T"], "klass": r["klass"], "er_fin": m["er_fin"], "er_max": m["er_max"],
                   "er_min": m.get("er_min"), "floor_ratio": m.get("floor_ratio"), "late_drift": m.get("late_drift"),
                   "bounded_breathing": m.get("bounded_breathing"), "n_fin": m["n_fin"], "core_fin": m["core_fin"],
                   "held_mass": held, "wallclock_min": round((time.time() - t0) / 60.0, 1)}
            print(f"[{i}/{len(todo)}] {c['key']:30} -> {r['klass']:26} n_fin={m['n_fin']} "
                  f"drift={m.get('late_drift'):+.3f} breath={m.get('bounded_breathing')} ({row['wallclock_min']}m)", flush=True)
        except Exception as exc:
            row = {"key": c["key"], "phase": c["phase"], "param": c["param"], "value": c["value"],
                   "K": c["K"], "T": c["T"], "klass": "ERROR", "error": str(exc)[:200]}
            print(f"[{i}/{len(todo)}] {c['key']} ERROR {exc}", flush=True)
        rows.append(row); flush()
    print(f"=== DONE: {sum(1 for r in rows if r.get('klass') not in ('ERROR', None))} ok, "
          f"{round((time.time()-start)/3600,2)}h ===", flush=True)
    print(out)


if __name__ == "__main__":
    main()
