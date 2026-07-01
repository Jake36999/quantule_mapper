"""
Escalate top JAX-scout candidates to 128^3 CuPy source-of-truth reproduction.

Selection is by the SCOUT structure proxy (n_local_max, prominence) among stable
configs -- NOT the 48^3 canonical score, which is unreliable for peak detection at
scout resolution. Each candidate is run through the production worker_cupy at 128^3
/1200 and scored with the canonical prime_log_sse (corrected headline = matched
errors only; penalties kept separate). This is the definitive test of whether the
corrected solver produces real prime-spectrum peaks anywhere in these regions.

Usage (native .venv): python jax_scout/escalate.py [--sweepdir <dir>] [--n 6]
"""
import os
import sys
import csv
import json
import glob
import time
import argparse
import subprocess
import numpy as np
import h5py

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
ORDER = ["param_D", "param_eta", "param_rho_vac", "param_omega0", "param_a_coupling", "param_s", "param_f"]


def latest_sweepdir():
    dirs = sorted(glob.glob(os.path.join(ROOT, "sweep_runs", "CORRECTED_PHYSICS_JAX_SCOUT_*")))
    return dirs[-1]


def score_h5(h5_path):
    import quantulemapper_real as qm
    with h5py.File(h5_path, "r") as f:
        psi = f["psi_final"][:]
        e = f["telemetry"]["energy"][:] if "telemetry" in f and "energy" in f["telemetry"] else np.array([np.nan])
    rho = np.abs(psi) ** 2
    d = qm.prime_log_sse(rho)
    corrected = float(d["log_prime_sse"]) - float(d["missing_peak_penalty"]) - float(d["noise_penalty"])
    try:
        import cupy as cp; cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass
    return {"corrected_sse": round(corrected, 5), "best_single_error": round(float(d["best_single_error"]), 5),
            "n_peaks": int(d["n_peaks_found_main"]), "missing_penalty": float(d["missing_peak_penalty"]),
            "noise_penalty": round(float(d["noise_penalty"]), 4), "max_amp": round(float(np.max(np.abs(psi))), 4),
            "energy_first": round(float(e[0]), 4), "energy_last": round(float(e[-1]), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweepdir", default=None)
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()
    sd = args.sweepdir or latest_sweepdir()
    order = list(json.load(open(os.path.join(sd, "sweep_meta.json")))["param_order"])
    rows = list(csv.DictReader(open(os.path.join(sd, "sweep_results.csv"))))
    for r in rows:
        for k in r:
            if k not in ("finite",):
                try: r[k] = float(r[k])
                except Exception: pass
    stable = [r for r in rows if str(r["finite"]) == "True" and r["energy_ratio"] > 1e-3 and r["max_amp"] < 1e3]
    stable.sort(key=lambda r: (r["n_local_max"], r["peak_to_median"]), reverse=True)
    panel = stable[:args.n]

    cfgdir = os.path.join(sd, "escalation_configs"); os.makedirs(cfgdir, exist_ok=True)
    datadir = os.path.join(sd, "escalation_data"); os.makedirs(datadir, exist_ok=True)
    report = os.path.join(sd, "escalation_128_report.csv")
    print(f"escalating top {len(panel)} scout candidates to 128^3 CuPy from {sd}\n")

    out_rows, t0 = [], time.time()
    for i, r in enumerate(panel):
        sid = int(r["idx"])
        cfg = {k: r[k] for k in order}
        cfg["param_splash_coupling"] = cfg.pop("param_s")
        cfg["param_splash_fraction"] = cfg.pop("param_f")
        cfg.update({"global_seed": 42, "config_hash": f"scout_{sid}",
                    "simulation": {"N_grid": 128, "L_domain": 10.0, "T_steps": 1200, "dt": 0.005,
                                   "collapse_threshold": 1e10}})
        cfgpath = os.path.join(cfgdir, f"config_scout_{sid}.json")
        json.dump(cfg, open(cfgpath, "w"), indent=2)
        h5 = os.path.join(datadir, f"rho_scout_{sid}.h5")
        ts = time.time()
        print(f"[{i+1}/{len(panel)}] scout_{sid} (nloc={int(r['n_local_max'])}) running 128^3...", flush=True)
        try:
            subprocess.run([sys.executable, "worker_cupy.py", "--params", cfgpath, "--output", h5],
                           cwd=ROOT, capture_output=True, text=True, timeout=1800)
        except subprocess.TimeoutExpired:
            pass
        row = {"scout_idx": sid, **{k: r[k] for k in order}, "scout_nloc": int(r["n_local_max"])}
        if os.path.exists(h5):
            row.update(score_h5(h5))
            row["promote"] = bool(row["n_peaks"] >= 2 and row["corrected_sse"] < 1.0 and row["max_amp"] < 1e3)
        else:
            row["note"] = "no_output"; row["promote"] = False
        out_rows.append(row)
        print(f"      -> n_peaks={row.get('n_peaks')} corrected={row.get('corrected_sse')} "
              f"promote={row.get('promote')} ({time.time()-ts:.0f}s)", flush=True)

    with open(report, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys())); w.writeheader(); w.writerows(out_rows)
    npro = sum(1 for r in out_rows if r.get("promote"))
    print(f"\nDONE {len(panel)} escalations in {(time.time()-t0)/60:.1f} min -> {report}")
    print(f"PROMOTED (>=2 real prime peaks, corrected<1.0, stable @128^3): {npro}")
    if npro == 0:
        print("Corrected solver produced no real prime-spectrum peaks at 128^3 in the top scout regions.")


if __name__ == "__main__":
    main()
