"""feb joint (param_a, param_eta, param_rho_vac) basin map — Stage 1 (coupled boundary).

Tests whether the OAT sensitivity story survives COUPLED perturbation of the three sensitive axes.
5 x 3 x 3 = 45 configs, multiplicative factors of feb's values; all other params (incl. param_omega0=0,
param_D, param_a_coupling, param_s, param_f) held at feb. K=6 / per-blob / seed 20260619 / N=96 / T=12000 /
classifier v3. Read-only w.r.t. physics (css.run_probe/classify) — no PDE/solver/geometry change.
Resumable (--out skips done) + deadline-aware. Scientific rejects ARE the coupled basin boundary, not
script failures. See docs/PHASE_C_FEB_PARAMETER_BASIN_CONSOLIDATED_ANALYSIS.md (revised joint-grid plan).
WSL2 jax venv:  python jax_scout/feb_joint_basin.py [--out DIR] [--deadline-hours 11.0]
"""
import os, sys, csv, json, time, argparse, itertools
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import core_saturation_search as css

N, T, K, SEED = 96, 12000, 6, 20260619
A_FACTORS = [0.9, 0.95, 1.0, 1.05, 1.1]
ETA_FACTORS = [0.8, 1.0, 1.2]
RHO_FACTORS = [0.85, 1.0, 1.25]
COLS = ["key", "a_factor", "eta_factor", "rho_factor", "param_a", "param_eta", "param_rho_vac",
        "klass", "er_fin", "er_max", "er_min", "floor_ratio", "late_drift", "bounded_breathing",
        "n_fin", "core_fin", "held_mass", "wallclock_min", "error"]


def configs():
    cfgs = []
    for fa, fe, fr in itertools.product(A_FACTORS, ETA_FACTORS, RHO_FACTORS):
        pp = dict(css.FEB)
        pp["param_a"] = float(css.FEB["param_a"]) * fa
        pp["param_eta"] = float(css.FEB["param_eta"]) * fe
        pp["param_rho_vac"] = float(css.FEB["param_rho_vac"]) * fr
        cfgs.append({"key": f"a{fa}_e{fe}_r{fr}", "a_factor": fa, "eta_factor": fe, "rho_factor": fr, "params": pp})
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--deadline-hours", type=float, default=11.0)
    args = ap.parse_args()
    out = args.out or os.path.join(ROOT, "sweep_runs", f"FEB_JOINT_BASIN_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    csv_path = os.path.join(out, "feb_joint_basin_results.csv")

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
                   "a_factors": A_FACTORS, "eta_factors": ETA_FACTORS, "rho_factors": RHO_FACTORS,
                   "classifier": css.classifier_spec(), "rows": rows},
                  open(os.path.join(out, "feb_joint_basin_summary.json"), "w"), indent=2, default=float)

    todo = [c for c in configs() if c["key"] not in done]
    start = time.time(); deadline = start + args.deadline_hours * 3600
    print(f"=== FEB JOINT BASIN | classifier={css.classifier_spec()['version']} | {len(todo)} todo | "
          f"deadline {args.deadline_hours}h | out={out} ===", flush=True)
    for i, c in enumerate(todo, 1):
        if time.time() + (T / 1000.0 + 1.5) * 60 > deadline:
            print(f"[{i}/{len(todo)}] SKIP {c['key']} (deadline)", flush=True); continue
        t0 = time.time()
        try:
            r = css.run_probe(c["params"], N, T, K, seed=SEED, ic_norm=css.IC_NORM_PER_BLOB_FIXED)
            m = r["metrics"]; pf = np.asarray(r["psi_fin"])
            held = float(np.sum(np.abs(pf) ** 2)) if np.isfinite(pf).all() else float("inf")
            np.savez_compressed(os.path.join(out, c["key"] + "_probe.npz"), psi_fin=pf, er=r["er"])
            row = {"key": c["key"], "a_factor": c["a_factor"], "eta_factor": c["eta_factor"], "rho_factor": c["rho_factor"],
                   "param_a": c["params"]["param_a"], "param_eta": c["params"]["param_eta"], "param_rho_vac": c["params"]["param_rho_vac"],
                   "klass": r["klass"], "er_fin": m["er_fin"], "er_max": m["er_max"], "er_min": m.get("er_min"),
                   "floor_ratio": m.get("floor_ratio"), "late_drift": m.get("late_drift"), "bounded_breathing": m.get("bounded_breathing"),
                   "n_fin": m["n_fin"], "core_fin": m["core_fin"], "held_mass": held, "wallclock_min": round((time.time() - t0) / 60.0, 1)}
            print(f"[{i}/{len(todo)}] {c['key']:22} -> {r['klass']:26} n_fin={m['n_fin']} drift={m.get('late_drift'):+.3f} "
                  f"breath={m.get('bounded_breathing')} ({row['wallclock_min']}m)", flush=True)
        except Exception as exc:
            row = {"key": c["key"], "a_factor": c["a_factor"], "eta_factor": c["eta_factor"], "rho_factor": c["rho_factor"],
                   "klass": "ERROR", "error": str(exc)[:200]}
            print(f"[{i}/{len(todo)}] {c['key']} ERROR {exc}", flush=True)
        rows.append(row); flush()
    ok = [r for r in rows if r.get("klass") not in ("ERROR", None)]
    import collections
    print(f"=== DONE: {len(ok)} ok, {round((time.time()-start)/3600,2)}h | {dict(collections.Counter(r['klass'] for r in ok))} ===", flush=True)
    print(out)


if __name__ == "__main__":
    main()
