"""Long-T breathing characterisation — is feb a periodic breather, a relaxing steady state, or a slow decay?

feb's er(t) over T=24000 shows a single rise (peak ~t4000) then a slow decline with no second oscillation,
and is still declining at t=24000. This run extends two well-interior T24000-core cells (feb-center and
a1.05,e1.0,r1.0) to T=72000 to disambiguate:
  - periodic breather  -> er oscillates (a trough then a second rise)
  - relaxing steady    -> er flattens to a plateau
  - slow decay         -> er keeps declining toward the spin-down floor (would weaken the long-time claim)

v3, K=6/per-blob/N=96/seed 20260619; all params at feb except param_a (one cell ×1.05). Verdict-first
(full er saved for spectral analysis; no trace -> memory-safe at long T). Read-only physics. Resumable.
WSL2 jax venv:  python jax_scout/feb_breathing_longt.py [--out DIR] [--T 72000]
"""
import os, sys, csv, json, time, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import core_saturation_search as css

N, K, SEED = 96, 6, 20260619
CELLS = [("feb_center", 1.0), ("a1.05_core", 1.05)]  # (label, param_a factor); eta,rho at feb
COLS = ["key", "a_factor", "T", "klass", "er_fin", "er_max", "er_min", "floor_ratio", "late_drift",
        "bounded_breathing", "n_fin", "held_mass", "wallclock_min", "error"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--T", type=int, default=72000)
    args = ap.parse_args()
    T = args.T
    out = args.out or os.path.join(ROOT, "sweep_runs", f"FEB_BREATHING_LONGT_T{T}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    csv_path = os.path.join(out, "feb_breathing_longt_results.csv")

    rows, done = [], set()
    if os.path.exists(csv_path):
        for r in csv.DictReader(open(csv_path, newline="")):
            if r.get("klass") and r["klass"] != "ERROR":
                rows.append(r); done.add(r["key"])
        if done:
            print(f"RESUME: skipping {len(done)} done", flush=True)

    def flush():
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
        json.dump({"N": N, "T": T, "K": K, "seed": SEED, "feb_params": css.FEB,
                   "classifier": css.classifier_spec(), "rows": rows},
                  open(os.path.join(out, "feb_breathing_longt_summary.json"), "w"), indent=2, default=float)

    print(f"=== FEB BREATHING LONG-T (T={T}) | classifier={css.classifier_spec()['version']} | out={out} ===", flush=True)
    for label, fa in CELLS:
        key = f"{label}_T{T}"
        if key in done:
            print(f"skip {key} (done)", flush=True); continue
        pp = dict(css.FEB); pp["param_a"] = float(css.FEB["param_a"]) * fa
        t0 = time.time()
        try:
            r = css.run_probe(pp, N, T, K, seed=SEED, ic_norm=css.IC_NORM_PER_BLOB_FIXED)
            m = r["metrics"]; pf = np.asarray(r["psi_fin"]); er = np.asarray(r["er"], dtype=np.float32)
            held = float(np.sum(np.abs(pf) ** 2)) if np.isfinite(pf).all() else float("inf")
            np.savez_compressed(os.path.join(out, key + "_probe.npz"), psi_fin=pf, er=er)
            row = {"key": key, "a_factor": fa, "T": T, "klass": r["klass"], "er_fin": m["er_fin"],
                   "er_max": m["er_max"], "er_min": m.get("er_min"), "floor_ratio": m.get("floor_ratio"),
                   "late_drift": m.get("late_drift"), "bounded_breathing": m.get("bounded_breathing"),
                   "n_fin": m["n_fin"], "held_mass": held, "wallclock_min": round((time.time() - t0) / 60.0, 1)}
            print(f"{key} -> {r['klass']} er_max={m['er_max']:.3f} er_fin={m['er_fin']:.3f} drift={m.get('late_drift'):+.3f} "
                  f"breath={m.get('bounded_breathing')} ({row['wallclock_min']}m)", flush=True)
        except Exception as exc:
            row = {"key": key, "a_factor": fa, "T": T, "klass": "ERROR", "error": str(exc)[:200]}
            print(f"{key} ERROR {exc}", flush=True)
        rows.append(row); flush()
    print(f"=== DONE {out} ===", flush=True)


if __name__ == "__main__":
    main()
