"""
Track 2: re-run a selected panel of low-SSE configs through the CURRENT corrected solver,
score with corrected spectral SSE, and classify by the promotion success condition.

Headline:   corrected_sse = sum(matched_errors) = log_prime_sse - missing_penalty - noise_penalty
Auxiliary:  missing_penalty, noise_penalty kept as separate columns (for the ASTE hunter).
Stability:  max|psi|, energy first/last, NaN/Inf, sentinel -> classification.

The old emergent-gravity tensor/fluid metrics (omega_sq, A-field, N-stages in the H5) are
deliberately NOT folded into the headline; only the spectral prime-SSE is used.

Classification (success condition: strong corrected hits under current solver, stable, no
reliance on the unstable saturation/gravity layer):
  PROMOTE                 - n_peaks>=2, corrected_sse < STRONG_SSE, bounded & not collapsed
  saturated_hit_review    - strong hit but saturated (relies on unstable layer) -> review only
  weak_hit                - peaks present but corrected_sse not strong
  historical_only_artifact- no/one peak under current solver (old-solver structure didn't survive)
  unstable_nan / run_failed

Usage: python tools/rerun_lowsse.py --panel <panel.csv> [--reuse]
Run from repo root with the .venv.
"""
import os
import sys
import csv
import json
import time
import argparse
import subprocess

import numpy as np
import h5py

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CFG_DIR = r"E:\Development_back_up_folder_2026\long_run data back up\input_configs"
OUT_DIR = r"E:\Development_back_up_folder_2026\lowsse_rerun_2026-06-18"
DATA_DIR = os.path.join(OUT_DIR, "data")
REPORT = os.path.join(OUT_DIR, "track2_report.csv")

STRONG_SSE = 1.0      # corrected_sse below this counts as a strong spectral hit
SATURATED_AMP = 1e3   # max|psi| above this = saturated (the unstable regime)
COLLAPSE_RATIO = 1e-3  # energy_last/energy_first below this = collapsed to vacuum

FIELDS = ["config_hash", "tags", "old_log_prime_sse", "new_total_sse", "corrected_sse",
          "best_single_error", "missing_peak_penalty", "noise_penalty", "n_peaks_found_main",
          "max_amp_peak", "energy_first", "energy_last", "nan_inf", "sentinel",
          "worker_status", "classification", "note"]


def score_field(h5_path):
    import quantulemapper_real as qm
    with h5py.File(h5_path, "r") as f:
        psi = f["psi_final"][:]
        e0 = e1 = None
        if "telemetry" in f and "energy" in f["telemetry"] and len(f["telemetry"]["energy"]) > 0:
            en = f["telemetry"]["energy"][:]
            e0, e1 = float(en[0]), float(en[-1])
        sentinel = float(f["sentinel_code"][0]) if "sentinel_code" in f else None
    nan_inf = bool(np.isnan(psi).any() or np.isinf(psi).any())
    rho = np.abs(psi) ** 2
    diag = qm.prime_log_sse(rho)
    new_total = float(diag["log_prime_sse"])
    miss = float(diag["missing_peak_penalty"])
    noise = float(diag["noise_penalty"])
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass
    return {
        "new_total_sse": new_total, "corrected_sse": new_total - miss - noise,
        "best_single_error": float(diag["best_single_error"]), "missing_peak_penalty": miss,
        "noise_penalty": noise, "n_peaks_found_main": int(diag["n_peaks_found_main"]),
        "max_amp_peak": float(np.max(np.abs(psi))), "energy_first": e0, "energy_last": e1,
        "nan_inf": nan_inf, "sentinel": sentinel,
    }


def classify(r):
    if r["worker_status"] in ("TIMEOUT", "NO_OUTPUT"):
        return "run_failed"
    if r.get("nan_inf"):
        return "unstable_nan"
    n = r.get("n_peaks_found_main")
    amp = r.get("max_amp_peak")
    e0, e1 = r.get("energy_first"), r.get("energy_last")
    collapsed = (e0 and e1 is not None and e0 > 0 and (e1 / e0) < COLLAPSE_RATIO)
    if n is None or n < 2 or collapsed:
        return "historical_only_artifact"
    corrected = r.get("corrected_sse")
    if corrected is not None and corrected < STRONG_SSE:
        return "saturated_hit_review" if (amp and amp > SATURATED_AMP) else "PROMOTE"
    return "weak_hit"


def load_panel(panel_csv):
    with open(panel_csv, newline="") as f:
        return [(r["config_hash"], r.get("tags", "")) for r in csv.DictReader(f)]


def already_done():
    if not os.path.exists(REPORT):
        return set()
    with open(REPORT, newline="") as f:
        return {r["config_hash"] for r in csv.DictReader(f)}


def append_row(row):
    new = not os.path.exists(REPORT)
    with open(REPORT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--reuse", action="store_true", help="skip worker if H5 already exists")
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    panel = load_panel(args.panel)
    done = already_done()
    todo = [(h, t) for h, t in panel if h not in done]
    print(f"panel: {len(panel)}  done: {len(panel)-len(todo)}  to run: {len(todo)}  -> {REPORT}\n")

    t_start = time.time()
    for i, (h, tags) in enumerate(todo):
        cfg = os.path.join(CFG_DIR, f"config_{h}.json")
        out_h5 = os.path.join(DATA_DIR, f"rho_{h}.h5")
        t0 = time.time()
        row = {k: "" for k in FIELDS}
        row.update({"config_hash": h, "tags": tags, "worker_status": "UNKNOWN"})

        if args.reuse and os.path.exists(out_h5):
            row["worker_status"] = "REUSED"
        else:
            print(f"[{i+1}/{len(todo)}] {h[:10]} [{tags}] running worker...", flush=True)
            try:
                proc = subprocess.run(
                    [sys.executable, "worker_cupy.py", "--params", cfg, "--output", out_h5],
                    cwd=ROOT, capture_output=True, text=True, timeout=1800)
                row["worker_status"] = "RAN" if os.path.exists(out_h5) else "NO_OUTPUT"
                if not os.path.exists(out_h5):
                    row["note"] = (proc.stderr or "")[-300:]
            except subprocess.TimeoutExpired:
                row["worker_status"] = "TIMEOUT"

        if os.path.exists(out_h5):
            try:
                row.update(score_field(out_h5))
            except Exception as exc:
                row["note"] = f"score_error: {exc}"[:300]
        row["classification"] = classify(row)
        append_row(row)
        cs = row["corrected_sse"]
        print(f"[{i+1}/{len(todo)}] {h[:10]} [{tags}] -> {row['classification']} "
              f"corrected={cs if cs=='' else round(cs,4)} n_peaks={row['n_peaks_found_main']} "
              f"max_amp={row['max_amp_peak'] if row['max_amp_peak']=='' else round(row['max_amp_peak'],2)} "
              f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\nTrack 2 done: {len(todo)} runs in {(time.time()-t_start)/60:.1f} min -> {REPORT}")


if __name__ == "__main__":
    main()
