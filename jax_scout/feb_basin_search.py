"""feb56dc7 basin focused search — node-count + IC-norm grid at feb's parameters.

Characterizes the basin around the one validated long-time bound state (feb56dc7) instead of a broad
hunt. Fixed feb parameters (css.FEB); only the IC family is varied. All runs at N=96, T=12000 so the
v2 long-time stability gate (normalized late-half energy drift) decides promotion. See
docs/PHASE_C_FEB_BASIN_SEARCH_PLAN.md.

Grid A (node-count basin): K in {3,4,5,6,8}, per-blob-fixed norm.
Grid B (norm/mass probe @ K=6): total-mass-fixed at {0.5,1,2}x the per-blob-natural mass.

Read-only w.r.t. physics: uses css.run_probe / css.classify unchanged; no PDE/solver/geometry change.
WSL2 jax venv:  python /mnt/f/quantule_mapper/jax_scout/feb_basin_search.py
"""
import os, sys, csv, json, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import core_saturation_search as css

N, T, SEED = 96, 12000, 20260619


def natural_mass(K):
    _, stats = css.build_ic(N, K, SEED, ic_norm=css.IC_NORM_PER_BLOB_FIXED)
    return float(stats["initial_mass"])


COLS = ["key", "K", "ic_norm", "target_mass", "klass", "er_fin", "er_max", "late_slope", "late_drift",
        "n_fin", "n_mid", "core_fin", "initial_mass", "held_mass", "wallclock_min", "error"]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="Resume into this existing dir (skip completed keys); default = new timestamped dir.")
    args = ap.parse_args()
    global OUT
    OUT = args.out or os.path.join(ROOT, "sweep_runs", f"FEB_BASIN_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(OUT, exist_ok=True)

    m6 = natural_mass(6)
    grid = []
    for K in [3, 4, 5, 6, 8]:
        grid.append({"key": f"K{K}_perblob", "K": K, "ic_norm": css.IC_NORM_PER_BLOB_FIXED, "mass": None})
    for fac in [0.5, 1.0, 2.0]:
        grid.append({"key": f"K6_totalmass_{fac:g}x", "K": 6, "ic_norm": css.IC_NORM_TOTAL_MASS_FIXED, "mass": fac * m6})

    # resume: keep completed rows, skip their keys
    rows = []
    done = set()
    existing_csv = os.path.join(OUT, "feb_basin_results.csv")
    if os.path.exists(existing_csv):
        with open(existing_csv, newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("klass") and r["klass"] != "ERROR":
                    rows.append(r); done.add(r["key"])
        if done:
            print(f"RESUME: skipping {len(done)} completed configs: {sorted(done)}", flush=True)
    grid = [g for g in grid if g["key"] not in done]

    def flush():
        with open(os.path.join(OUT, "feb_basin_results.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
        with open(os.path.join(OUT, "feb_basin_summary.json"), "w") as fh:
            json.dump({"N": N, "T": T, "seed": SEED, "params": css.FEB, "natural_K6_mass": m6,
                       "classifier": css.classifier_spec(), "rows": rows}, fh, indent=2, default=float)

    print(f"=== FEB BASIN N={N} T={T} | feb params | natural K6 mass={m6:.1f} | "
          f"classifier={css.classifier_spec()['version']} | {len(grid)} configs ===", flush=True)
    for i, g in enumerate(grid, 1):
        t0 = time.time()
        try:
            r = css.run_probe(css.FEB, N, T, g["K"], seed=SEED, ic_norm=g["ic_norm"], target_initial_mass=g["mass"])
            m = r["metrics"]; pf = np.asarray(r["psi_fin"])
            held = float(np.sum(np.abs(pf) ** 2)) if np.isfinite(pf).all() else float("inf")
            np.savez_compressed(os.path.join(OUT, g["key"] + "_probe.npz"),
                                psi0=r["psi0"], psi_mid=r["psi_mid"], psi_fin=pf, energy=r["energy"], er=r["er"])
            row = {"key": g["key"], "K": g["K"], "ic_norm": g["ic_norm"], "target_mass": g["mass"],
                   "klass": r["klass"], "er_fin": m["er_fin"], "er_max": m["er_max"], "late_slope": m["late_slope"],
                   "late_drift": m.get("late_drift"), "n_fin": m["n_fin"], "n_mid": m["n_mid"],
                   "core_fin": m["core_fin"], "initial_mass": r["ic_stats"]["initial_mass"], "held_mass": held,
                   "wallclock_min": round((time.time() - t0) / 60.0, 1)}
            print(f"[{i}/{len(grid)}] {g['key']:20} -> {r['klass']:26} drift={m.get('late_drift'):+.3f} "
                  f"er_fin={m['er_fin']:.3f} n_fin={m['n_fin']} held={held:.1f} ({row['wallclock_min']}m)", flush=True)
        except Exception as exc:
            row = {"key": g["key"], "K": g["K"], "ic_norm": g["ic_norm"], "target_mass": g["mass"],
                   "klass": "ERROR", "error": str(exc)[:200]}
            print(f"[{i}/{len(grid)}] {g['key']} ERROR {exc}", flush=True)
        rows.append(row); flush()

    print("=== FEB BASIN DONE ===", flush=True)
    print(OUT)


if __name__ == "__main__":
    main()
