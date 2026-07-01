"""
Canonical scoring of JAX-scout sweep candidates (native Windows .venv, cupy).

Loads the top fields saved by sweep_run.py and scores them with the PRODUCTION
prime_log_sse (quantulemapper_real) using corrected matched-SSE (penalties excluded
from headline, kept as columns). Flags which candidates to escalate to full 128^3
CuPy source-of-truth reproduction.

These scores are at SCOUT resolution (N=48) -> still NOT final evidence. They only
rank candidates for CuPy validation.

Usage (native): python jax_scout/score_sweep.py [--sweepdir <dir>]
"""
import os
import sys
import csv
import json
import glob
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PROMOTE_SSE = 1.0   # corrected matched-SSE below this (with >=2 peaks, stable) -> escalate


def latest_sweepdir():
    base = os.path.join(ROOT, "sweep_runs")
    dirs = sorted(glob.glob(os.path.join(base, "CORRECTED_PHYSICS_JAX_SCOUT_*")))
    if not dirs:
        sys.exit("no sweep dirs found")
    return dirs[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweepdir", default=None)
    args = ap.parse_args()
    sweepdir = args.sweepdir or latest_sweepdir()
    print(f"scoring candidates in: {sweepdir}")

    meta = json.load(open(os.path.join(sweepdir, "sweep_meta.json")))
    npz = np.load(os.path.join(sweepdir, "sweep_fields_top.npz"), allow_pickle=True)
    idx, params, fields = npz["idx"], npz["params"], npz["fields"]
    order = list(meta["param_order"])

    import quantulemapper_real as qm

    rows = []
    for i in range(len(idx)):
        psi = fields[i]
        rho = np.abs(psi) ** 2
        diag = qm.prime_log_sse(rho)
        total = float(diag["log_prime_sse"])
        miss = float(diag["missing_peak_penalty"])
        noise = float(diag["noise_penalty"])
        corrected = total - miss - noise
        npk = int(diag["n_peaks_found_main"])
        max_amp = float(np.max(np.abs(psi)))
        stable = bool(np.all(np.isfinite(psi))) and max_amp < 1e3
        escalate = stable and npk >= 2 and corrected < PROMOTE_SSE
        row = {"scout_idx": int(idx[i]),
               **{k: float(params[i][j]) for j, k in enumerate(order)},
               "corrected_sse": round(corrected, 5), "best_single_error": round(float(diag["best_single_error"]), 5),
               "n_peaks": npk, "missing_penalty": miss, "noise_penalty": round(noise, 4),
               "max_amp": round(max_amp, 4), "stable": stable, "escalate": escalate}
        rows.append(row)
        try:
            import cupy as cp; cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    rows.sort(key=lambda r: (not r["escalate"], r["corrected_sse"]))
    out = os.path.join(sweepdir, "candidate_scores.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    esc = [r for r in rows if r["escalate"]]
    print(f"\n{meta['label']} / {meta['evidence']}  (scout N={meta['N']}^3, NOT final evidence)")
    print(f"scored {len(rows)} top candidates -> {out}")
    print(f"escalation candidates (>=2 peaks, corrected<{PROMOTE_SSE}, stable): {len(esc)}\n")
    print(f"{'scout_idx':>9} {'corrected':>10} {'n_pk':>5} {'max_amp':>9} {'escalate':>9}")
    for r in rows[:12]:
        print(f"{r['scout_idx']:>9} {r['corrected_sse']:10.4f} {r['n_peaks']:>5} {r['max_amp']:>9.3f} {str(r['escalate']):>9}")
    if esc:
        # write escalation config JSONs at production resolution for CuPy reproduction
        escdir = os.path.join(sweepdir, "escalation_configs")
        os.makedirs(escdir, exist_ok=True)
        for r in esc:
            cfg = {k: r[k] for k in order}
            cfg["param_splash_coupling"] = cfg.pop("param_s")
            cfg["param_splash_fraction"] = cfg.pop("param_f")
            cfg.update({"global_seed": 42, "config_hash": f"scout_{r['scout_idx']}",
                        "simulation": {"N_grid": 128, "L_domain": 10.0, "T_steps": 1200, "dt": 0.005,
                                       "collapse_threshold": 1e10}})
            json.dump(cfg, open(os.path.join(escdir, f"config_scout_{r['scout_idx']}.json"), "w"), indent=2)
        print(f"\nwrote {len(esc)} escalation configs (128^3) -> {escdir}")
    else:
        print("\nNo escalation candidates: corrected solver produced no stable prime-matching")
        print("structure in this sweep. (Consistent with Track-2 dissipative behaviour.)")


if __name__ == "__main__":
    main()
