"""feb-basin confirmation — seed-robustness + T=24000 rigor for the node-count family.

Confirms the docs/PHASE_C_FEB_BASIN_RESULTS.md finding (feb's param regime is a node-count bound-state
basin) with:
  * 2 extra seeds (20260620, 20260621) x K in {3,4,5,6,8} per-blob @ T=12000  (seed robustness)
  * K in {3,6} @ T=24000, seed 20260619                                       (long-time rigor)

Fixed feb params (css.FEB), per-blob-fixed norm, N=96, classifier v2 (drift gate). Read-only w.r.t.
physics. Resumable: --out into an existing dir skips completed keys. Saves psi_fin for topology render.
WSL2 jax venv:  python jax_scout/feb_basin_confirm.py [--out DIR]
"""
import os, sys, csv, json, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import core_saturation_search as css

N = 96
COLS = ["key", "K", "seed", "T", "klass", "er_fin", "er_max", "late_slope", "late_drift",
        "n_fin", "n_mid", "core_fin", "initial_mass", "held_mass", "wallclock_min", "error"]


def configs():
    cfgs = []
    for seed in [20260620, 20260621]:
        for K in [3, 4, 5, 6, 8]:
            cfgs.append({"key": f"K{K}_s{seed}_T12000", "K": K, "seed": seed, "T": 12000})
    for K in [3, 6]:
        cfgs.append({"key": f"K{K}_s20260619_T24000", "K": K, "seed": 20260619, "T": 24000})
    return cfgs


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(ROOT, "sweep_runs", f"FEB_BASIN_CONFIRM_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)

    rows, done = [], set()
    csv_path = os.path.join(out, "feb_basin_confirm_results.csv")
    if os.path.exists(csv_path):
        for r in csv.DictReader(open(csv_path, newline="")):
            if r.get("klass") and r["klass"] != "ERROR":
                rows.append(r); done.add(r["key"])
        if done:
            print(f"RESUME: skipping {len(done)} done: {sorted(done)}", flush=True)

    def flush():
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
        with open(os.path.join(out, "feb_basin_confirm_summary.json"), "w") as fh:
            json.dump({"N": N, "params": css.FEB, "classifier": css.classifier_spec(), "rows": rows}, fh, indent=2, default=float)

    todo = [c for c in configs() if c["key"] not in done]
    print(f"=== FEB BASIN CONFIRM | classifier={css.classifier_spec()['version']} | {len(todo)} configs ===", flush=True)
    for i, c in enumerate(todo, 1):
        t0 = time.time()
        try:
            r = css.run_probe(css.FEB, N, c["T"], c["K"], seed=c["seed"], ic_norm=css.IC_NORM_PER_BLOB_FIXED)
            m = r["metrics"]; pf = np.asarray(r["psi_fin"])
            held = float(np.sum(np.abs(pf) ** 2)) if np.isfinite(pf).all() else float("inf")
            np.savez_compressed(os.path.join(out, c["key"] + "_probe.npz"),
                                psi0=r["psi0"], psi_mid=r["psi_mid"], psi_fin=pf, energy=r["energy"], er=r["er"])
            row = {"key": c["key"], "K": c["K"], "seed": c["seed"], "T": c["T"], "klass": r["klass"],
                   "er_fin": m["er_fin"], "er_max": m["er_max"], "late_slope": m["late_slope"],
                   "late_drift": m.get("late_drift"), "n_fin": m["n_fin"], "n_mid": m["n_mid"],
                   "core_fin": m["core_fin"], "initial_mass": r["ic_stats"]["initial_mass"], "held_mass": held,
                   "wallclock_min": round((time.time() - t0) / 60.0, 1)}
            print(f"[{i}/{len(todo)}] {c['key']:24} -> {r['klass']:26} drift={m.get('late_drift'):+.3f} "
                  f"n_fin={m['n_fin']} er_fin={m['er_fin']:.3f} ({row['wallclock_min']}m)", flush=True)
        except Exception as exc:
            row = {"key": c["key"], "K": c["K"], "seed": c["seed"], "T": c["T"], "klass": "ERROR", "error": str(exc)[:200]}
            print(f"[{i}/{len(todo)}] {c['key']} ERROR {exc}", flush=True)
        rows.append(row); flush()
    print("=== CONFIRM DONE ===", flush=True); print(out)


if __name__ == "__main__":
    main()
