"""Confirm a* — the long-time-stationary gain found by feb_gain_ladder_longt (a ~ x1.15, param_a ~ 0.55).

The gain ladder (FEB_GAIN_LADDER_LONGT_T72000, 2026-07-01) found the late-window er slope crosses ~0 at
a x1.15: er rises to ~2.07 then holds (slope -0.0006/1k over the last half), a genuine T=72000 stationary
bound state, while lower gains slowly decay (slope proportional to distance below a*). See
docs/PHASE_C_GAIN_LADDER_RESULTS.md. This overnight run firms up that single-seed / single-T / single-point
result three ways:

  (1) LONGER-T   a x1.15 and a x1.125 at T=144000 -- does x1.15 stay flat (true fixed point), and does the
                 sub-a* survivor x1.125 (slope -0.0047) reveal itself as a slow decayer (cross er0)?
  (2) SEED-ROBUST a x1.15 at seeds 20260620 / 20260621 (T72000) -- does the stationary state hold across ICs?
  (3) BRACKET a*  a x1.16 / x1.175 / x1.20 (T72000) -- locate the zero-crossing precisely and confirm the
                 late slope goes POSITIVE (growth) above a*, i.e. a* is a crossing/knife-edge, not a floor.

~9.6 h wallclock (T72000 ~ 64 min/cell; T144000 ~ 128 min). N96 / K6 / per-blob / v3, geometry frozen
e8d6a78ea. Verdict-first, full er(t) saved, resumable (per-cell CSV). Read-only physics.
WSL2 jax venv:  python jax_scout/feb_astar_confirm.py [--out DIR]
"""
import os, sys, csv, json, time, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import core_saturation_search as css

N, K = 96, 6
S619, S620, S621 = 20260619, 20260620, 20260621
# (label, param_a factor, T, seed)
CELLS = [
    ("a1.15_longT",   1.15,  144000, S619),   # (1) longer-T confirm a* stays flat
    ("a1.125_longT",  1.125, 144000, S619),   # (1) sub-a* longer-T -> does it decay across er0?
    ("a1.15_seed620", 1.15,  72000,  S620),   # (2) seed-robust
    ("a1.15_seed621", 1.15,  72000,  S621),   # (2) seed-robust
    ("a1.16",         1.16,  72000,  S619),   # (3) refine zero-crossing
    ("a1.175",        1.175, 72000,  S619),   # (3) just above a*
    ("a1.20",         1.20,  72000,  S619),   # (3) growth edge (slope positive?)
]
COLS = ["key", "a_factor", "T", "seed", "klass", "er0", "er_fin", "er_max", "er_min", "floor_ratio",
        "late_drift", "late_slope_10pct_per1k", "late_slope_50pct_per1k", "peak_frac", "crossed_er0_frac",
        "bounded_breathing", "n_fin", "held_mass", "wallclock_min", "error"]


def late_metrics(er):
    """Late-window linear-fit slopes (fractional er change per 1000 steps) + peak / er0-crossing locations."""
    er = np.asarray(er, dtype=float); n = er.size
    t = np.arange(n)
    er0 = float(er[0]); imax = int(np.argmax(er))
    def slope_per1k(lo):
        s = slice(int(lo * n), n)
        if s.stop - s.start < 3:
            return float("nan")
        return float(np.polyfit(t[s], er[s], 1)[0] * 1000.0)
    post = np.where(er[imax:] < er0)[0]
    crossed = float((imax + post[0]) / n) if post.size else None
    return {"er0": er0, "peak_frac": imax / n,
            "late_slope_10pct_per1k": slope_per1k(0.90),
            "late_slope_50pct_per1k": slope_per1k(0.50),
            "crossed_er0_frac": crossed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(ROOT, "sweep_runs", f"FEB_ASTAR_CONFIRM_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    csv_path = os.path.join(out, "feb_astar_confirm_results.csv")

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
        json.dump({"N": N, "K": K, "cells": [list(c) for c in CELLS], "feb_params": css.FEB,
                   "classifier": css.classifier_spec(), "rows": rows},
                  open(os.path.join(out, "feb_astar_confirm_summary.json"), "w"), indent=2, default=float)

    print(f"=== FEB A* CONFIRM | classifier={css.classifier_spec()['version']} | out={out} ===", flush=True)
    for label, fa, T, seed in CELLS:
        key = label
        if key in done:
            print(f"skip {key} (done)", flush=True); continue
        pp = dict(css.FEB); pp["param_a"] = float(css.FEB["param_a"]) * fa
        t0 = time.time()
        try:
            r = css.run_probe(pp, N, T, K, seed=seed, ic_norm=css.IC_NORM_PER_BLOB_FIXED)
            m = r["metrics"]; pf = np.asarray(r["psi_fin"]); er = np.asarray(r["er"], dtype=np.float32)
            held = float(np.sum(np.abs(pf) ** 2)) if np.isfinite(pf).all() else float("inf")
            lm = late_metrics(er)
            np.savez_compressed(os.path.join(out, key + "_probe.npz"), psi_fin=pf, er=er)
            row = {"key": key, "a_factor": fa, "T": T, "seed": seed, "klass": r["klass"], "er_fin": m["er_fin"],
                   "er_max": m["er_max"], "er_min": m.get("er_min"), "floor_ratio": m.get("floor_ratio"),
                   "late_drift": m.get("late_drift"), "bounded_breathing": m.get("bounded_breathing"),
                   "n_fin": m["n_fin"], "held_mass": held, "wallclock_min": round((time.time() - t0) / 60.0, 1),
                   **lm}
            print(f"{key} (a{fa} T{T} s{seed}) -> {r['klass']} er_max={m['er_max']:.3f} er_fin={m['er_fin']:.3f} "
                  f"late_slope/1k={lm['late_slope_50pct_per1k']:+.4f} drift={m.get('late_drift'):+.3f} "
                  f"({row['wallclock_min']}m)", flush=True)
        except Exception as exc:
            row = {"key": key, "a_factor": fa, "T": T, "seed": seed, "klass": "ERROR", "error": str(exc)[:200]}
            print(f"{key} ERROR {exc}", flush=True)
        rows.append(row); flush()
    print(f"=== DONE {out} ===", flush=True)


if __name__ == "__main__":
    main()
