"""feb-center resolution check — N=128 at the validation T bars (v3 gate).

Highest-priority falsifier: the feb-basin family is all N=96. This re-runs feb-center (css.FEB, K=6,
per-blob, seed 20260619 — the same config as the OAT `center_feb` and the `--ref feb56dc7` control) at
**N=128** at T=12000 (gate bar) and T=24000 (long-time bar), v3-gated, and compares to the N=96 result
(TRUE at both: T12000 drift -0.080; T24000 drift -0.177 bounded-breathing). Read-only w.r.t. physics.
Resumable (--out skips completed). Verdict-first (no trace -> memory-safe at N=128).
WSL2 jax venv:  python jax_scout/feb_center_resolution.py [--out DIR]
"""
import os, sys, csv, json, time, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import core_saturation_search as css

N = 128
SEED = css.SEED  # 20260619
T_BARS = [12000, 24000]
COLS = ["key", "N", "T", "klass", "er_fin", "er_max", "er_min", "floor_ratio", "late_drift",
        "bounded_breathing", "n_fin", "core_fin", "held_mass", "wallclock_min", "error"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(ROOT, "sweep_runs", f"FEB_CENTER_RESOLUTION_N128_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    csv_path = os.path.join(out, "resolution_results.csv")

    rows, done = [], set()
    if os.path.exists(csv_path):
        for r in csv.DictReader(open(csv_path, newline="")):
            if r.get("klass") and r["klass"] != "ERROR":
                rows.append(r); done.add(r["key"])

    def flush():
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
        json.dump({"N": N, "seed": SEED, "feb_params": css.FEB, "classifier": css.classifier_spec(), "rows": rows},
                  open(os.path.join(out, "resolution_summary.json"), "w"), indent=2, default=float)

    print(f"=== FEB-CENTER RESOLUTION N={N} | classifier={css.classifier_spec()['version']} | out={out} ===", flush=True)
    for T in T_BARS:
        key = f"feb_center_N128_T{T}"
        if key in done:
            print(f"skip {key} (done)", flush=True); continue
        t0 = time.time()
        try:
            r = css.run_probe(css.FEB, N, T, 6, seed=SEED, ic_norm=css.IC_NORM_PER_BLOB_FIXED)
            m = r["metrics"]; pf = np.asarray(r["psi_fin"])
            held = float(np.sum(np.abs(pf) ** 2)) if np.isfinite(pf).all() else float("inf")
            np.savez_compressed(os.path.join(out, key + "_probe.npz"), psi_fin=pf, er=r["er"])
            row = {"key": key, "N": N, "T": T, "klass": r["klass"], "er_fin": m["er_fin"], "er_max": m["er_max"],
                   "er_min": m.get("er_min"), "floor_ratio": m.get("floor_ratio"), "late_drift": m.get("late_drift"),
                   "bounded_breathing": m.get("bounded_breathing"), "n_fin": m["n_fin"], "core_fin": m["core_fin"],
                   "held_mass": held, "wallclock_min": round((time.time() - t0) / 60.0, 1)}
            print(f"{key} -> {r['klass']} drift={m.get('late_drift'):+.3f} breath={m.get('bounded_breathing')} "
                  f"er_fin={m['er_fin']:.3f} n_fin={m['n_fin']} ({row['wallclock_min']}m)", flush=True)
        except Exception as exc:
            row = {"key": key, "N": N, "T": T, "klass": "ERROR", "error": str(exc)[:200]}
            print(f"{key} ERROR {exc}", flush=True)
        rows.append(row); flush()
    print(f"=== DONE {out} ===", flush=True)


if __name__ == "__main__":
    main()
