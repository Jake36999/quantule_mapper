"""Delineate the T=24000-stable core — eta x1.0 plane (param_a x param_rho_vac) at T=24000.

Stage 2 showed the T24000-validated interior is narrower than the T12000 basin (the low-drive corner
a0.9,e1.0,r0.85 decays by T24000). This maps the full central spine plane (param_eta x1.0; param_a x
param_rho_vac) at T=24000 to draw the long-time core contour: all 15 cells are TRUE at T12000, so any that
reject at T24000 are the window-marginal shell. v3, K=6/per-blob/N=96/seed 20260619; all params except
param_a and param_rho_vac at feb. Read-only physics. Resumable + deadline-aware.
WSL2 jax venv:  python jax_scout/feb_core_delineation.py [--out DIR] [--deadline-hours 7.0]
"""
import os, sys, csv, json, time, argparse, itertools
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import core_saturation_search as css

N, T, K, SEED = 96, 24000, 6, 20260619
A_FACTORS = [0.9, 0.95, 1.0, 1.05, 1.1]
RHO_FACTORS = [0.85, 1.0, 1.25]   # eta fixed at feb (x1.0)
COLS = ["key", "a_factor", "rho_factor", "param_a", "param_rho_vac", "T", "klass", "er_fin", "er_max",
        "er_min", "floor_ratio", "late_drift", "bounded_breathing", "n_fin", "held_mass", "wallclock_min", "error"]


def configs():
    cfgs = []
    for fa, fr in itertools.product(A_FACTORS, RHO_FACTORS):
        pp = dict(css.FEB)
        pp["param_a"] = float(css.FEB["param_a"]) * fa
        pp["param_rho_vac"] = float(css.FEB["param_rho_vac"]) * fr
        cfgs.append({"key": f"a{fa}_e1.0_r{fr}_T24000", "a": fa, "r": fr, "params": pp})
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--deadline-hours", type=float, default=7.0)
    args = ap.parse_args()
    out = args.out or os.path.join(ROOT, "sweep_runs", f"FEB_CORE_DELINEATION_T24000_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    csv_path = os.path.join(out, "feb_core_delineation_results.csv")

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
                   "a_factors": A_FACTORS, "rho_factors": RHO_FACTORS, "eta": "feb x1.0",
                   "classifier": css.classifier_spec(), "rows": rows},
                  open(os.path.join(out, "feb_core_delineation_summary.json"), "w"), indent=2, default=float)

    todo = [c for c in configs() if c["key"] not in done]
    start = time.time(); deadline = start + args.deadline_hours * 3600
    print(f"=== FEB CORE DELINEATION (eta x1.0, T={T}) | classifier={css.classifier_spec()['version']} | "
          f"{len(todo)} todo | out={out} ===", flush=True)
    for i, c in enumerate(todo, 1):
        if time.time() + (T / 1000.0 + 1.5) * 60 > deadline:
            print(f"[{i}/{len(todo)}] SKIP {c['key']} (deadline)", flush=True); continue
        t0 = time.time()
        try:
            r = css.run_probe(c["params"], N, T, K, seed=SEED, ic_norm=css.IC_NORM_PER_BLOB_FIXED)
            m = r["metrics"]; pf = np.asarray(r["psi_fin"])
            held = float(np.sum(np.abs(pf) ** 2)) if np.isfinite(pf).all() else float("inf")
            np.savez_compressed(os.path.join(out, c["key"] + "_probe.npz"), psi_fin=pf, er=r["er"])
            row = {"key": c["key"], "a_factor": c["a"], "rho_factor": c["r"],
                   "param_a": c["params"]["param_a"], "param_rho_vac": c["params"]["param_rho_vac"], "T": T,
                   "klass": r["klass"], "er_fin": m["er_fin"], "er_max": m["er_max"], "er_min": m.get("er_min"),
                   "floor_ratio": m.get("floor_ratio"), "late_drift": m.get("late_drift"),
                   "bounded_breathing": m.get("bounded_breathing"), "n_fin": m["n_fin"], "held_mass": held,
                   "wallclock_min": round((time.time() - t0) / 60.0, 1)}
            print(f"[{i}/{len(todo)}] {c['key']:26} -> {r['klass']:26} n_fin={m['n_fin']} drift={m.get('late_drift'):+.3f} "
                  f"breath={m.get('bounded_breathing')} er_fin={m['er_fin']:.2f} ({row['wallclock_min']}m)", flush=True)
        except Exception as exc:
            row = {"key": c["key"], "a_factor": c["a"], "rho_factor": c["r"], "T": T, "klass": "ERROR", "error": str(exc)[:200]}
            print(f"[{i}/{len(todo)}] {c['key']} ERROR {exc}", flush=True)
        rows.append(row); flush()
    ok = [r for r in rows if r.get("klass") not in ("ERROR", None)]
    import collections
    core = sum(1 for r in ok if r["klass"] == "TRUE_SATURATED_BOUND_STATE")
    print(f"=== DONE: {len(ok)} ok, {round((time.time()-start)/3600,2)}h | core(TRUE@T24000)={core}/{len(ok)} | "
          f"{dict(collections.Counter(r['klass'] for r in ok))} ===", flush=True)
    print(out)


if __name__ == "__main__":
    main()
