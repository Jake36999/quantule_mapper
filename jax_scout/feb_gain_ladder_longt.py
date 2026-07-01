"""Long-T gain-ladder — does ANY param_a give a genuine long-time stationary state, or is the core a slow-decaying transient?

Follow-up to feb_breathing_longt.py (T=72000). That run FALSIFIED the "bounded breather" reading:
BOTH feb-center (a x1.0) and the interior core (a x1.05) slowly DECAY to T=72000 -- single early peak
(~t4000) then a monotone decline with er_min == er_fin (still falling at the end, no plateau, no rebound;
FFT shows no periodic component). The T=24000 "core" verdicts were a window artifact (er had not yet fallen
below er0), exactly paralleling the earlier T6000 -> T24000 overturn.

Crucially the decay is (weakly) GAIN-DEPENDENT: a x1.05 decays slower (er_fin 0.812) than a x1.0
(er_fin 0.416). The T=24000 delineation showed er_fin rises monotonically with param_a (0.52 -> 2.31 across
the eta x1.0 plane). So the true long-time balance -- if one exists -- sits at a gain ABOVE feb-center.

This run climbs a param_a ladder at feb (eta x1.0, rho x1.0), all else = feb, to T=72000, and reads the
LATE-WINDOW SLOPE of er(t) to disambiguate:
  - some a* gives late_slope ~ 0 with er_fin in band     -> genuine long-time stationary state at a* (REFINE)
  - cells jump decay (slope<0) -> growth (slope>0) with   -> no stationary gain in the family; the core is a
    no zero-slope member (or growers blow up)                long-lived transient / metastable (DOWNGRADE)
Gain-dependence of the slope also separates PHYSICAL decay (slope changes with a) from a pure numerical-
accumulation artifact (slope ~ a-independent) over the ~15M FP64 steps at T=72000.

v3, K=6/per-blob/N=96/seed 20260619; verdict-first, full er(t) saved, resumable. Read-only physics.
WSL2 jax venv:  python jax_scout/feb_gain_ladder_longt.py [--out DIR] [--T 72000]
"""
import os, sys, csv, json, time, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import core_saturation_search as css

N, K, SEED = 96, 6, 20260619
# param_a factors relative to feb. 1.0 and 1.05 already run (both DECAY) in feb_breathing_longt; this ladder
# climbs above them to locate a late-stationary gain (slope->0) or prove the decay->growth transition skips it.
CELLS = [("a1.075_ladder", 1.075), ("a1.10_ladder", 1.10),
         ("a1.125_ladder", 1.125), ("a1.15_ladder", 1.15)]
COLS = ["key", "a_factor", "T", "klass", "er0", "er_fin", "er_max", "er_min", "floor_ratio", "late_drift",
        "late_slope_10pct_per1k", "late_slope_50pct_per1k", "peak_frac", "crossed_er0_frac",
        "bounded_breathing", "n_fin", "held_mass", "wallclock_min", "error"]


def late_metrics(er):
    """Late-window linear-fit slopes (fractional er change per 1000 steps) + peak / er0-crossing locations."""
    er = np.asarray(er, dtype=float); n = er.size
    t = np.arange(n)
    er0 = float(er[0]); imax = int(np.argmax(er))
    def slope_per1k(lo):
        s = slice(int(lo * n), n)
        return float(np.polyfit(t[s], er[s], 1)[0] * 1000.0)
    # first post-peak index where er dips below its starting value (None if it never does)
    post = np.where(er[imax:] < er0)[0]
    crossed = float((imax + post[0]) / n) if post.size else None
    return {"er0": er0, "peak_frac": imax / n,
            "late_slope_10pct_per1k": slope_per1k(0.90),
            "late_slope_50pct_per1k": slope_per1k(0.50),
            "crossed_er0_frac": crossed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--T", type=int, default=72000)
    args = ap.parse_args()
    T = args.T
    out = args.out or os.path.join(ROOT, "sweep_runs", f"FEB_GAIN_LADDER_LONGT_T{T}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    csv_path = os.path.join(out, "feb_gain_ladder_longt_results.csv")

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
                  open(os.path.join(out, "feb_gain_ladder_longt_summary.json"), "w"), indent=2, default=float)

    print(f"=== FEB GAIN LADDER LONG-T (T={T}) | classifier={css.classifier_spec()['version']} | out={out} ===", flush=True)
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
            lm = late_metrics(er)
            np.savez_compressed(os.path.join(out, key + "_probe.npz"), psi_fin=pf, er=er)
            row = {"key": key, "a_factor": fa, "T": T, "klass": r["klass"], "er_fin": m["er_fin"],
                   "er_max": m["er_max"], "er_min": m.get("er_min"), "floor_ratio": m.get("floor_ratio"),
                   "late_drift": m.get("late_drift"), "bounded_breathing": m.get("bounded_breathing"),
                   "n_fin": m["n_fin"], "held_mass": held, "wallclock_min": round((time.time() - t0) / 60.0, 1),
                   **lm}
            print(f"{key} -> {r['klass']} er_max={m['er_max']:.3f} er_fin={m['er_fin']:.3f} "
                  f"late_slope/1k={lm['late_slope_10pct_per1k']:+.4f} "
                  f"drift={m.get('late_drift'):+.3f} ({row['wallclock_min']}m)", flush=True)
        except Exception as exc:
            row = {"key": key, "a_factor": fa, "T": T, "klass": "ERROR", "error": str(exc)[:200]}
            print(f"{key} ERROR {exc}", flush=True)
        rows.append(row); flush()
    print(f"=== DONE {out} ===", flush=True)


if __name__ == "__main__":
    main()
