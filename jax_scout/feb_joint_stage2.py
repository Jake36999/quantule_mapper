"""feb joint basin Stage 2 — boundary seed-repeat + interior T=24000 confirmation.

Follows FEB_JOINT_BASIN Stage 1 (docs/PHASE_C_JOINT_PARAM_BASIN_RESULTS.md). Two parts, classifier v3,
K=6/per-blob/N=96, feb params except the 3 perturbed axes (param_a, param_eta, param_rho_vac as factors
of feb); all other params incl param_omega0=0 at feb. Read-only w.r.t. physics. Resumable + deadline-aware.

  boundary seed-repeat (T=12000): the Stage-1 TRUE<->reject transition cells re-run at seeds 20260620/621,
      to test whether the coupled boundary is seed-robust (Stage-1 upper edges were seed-sensitive).
  interior T=24000 (seed 20260619): deep interior cells, to confirm the coupled interior is long-time
      stable (not a T=12000 artifact).

WSL2 jax venv:  python jax_scout/feb_joint_stage2.py [--out DIR] [--deadline-hours 6.0]
"""
import os, sys, csv, json, time, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import core_saturation_search as css

N, K = 96, 6
SEEDS = [20260620, 20260621]
TAG2CLASS = {"T": "TRUE_SATURATED_BOUND_STATE", "S": "SPIN_DOWN_REJECT", "G": "TRANSIENT_GROWER_REJECT"}
# Stage-1 transition cells: (a_factor, eta_factor, rho_factor, stage1_seed619_tag)
BOUNDARY_CELLS = [
    (1.0, 1.2, 0.85, "S"), (1.05, 1.2, 0.85, "T"),   # a-rescue at high loss (low rho)
    (0.95, 1.2, 1.0, "S"), (1.0, 1.2, 1.0, "T"),     # a-rescue at high loss (mid rho)
    (0.9, 1.2, 1.0, "S"), (0.9, 1.2, 1.25, "T"),     # rho-rescue at high loss
    (1.0, 0.8, 1.25, "T"), (1.05, 0.8, 1.25, "G"),   # grower onset
]
INTERIOR_CELLS = [(1.1, 1.0, 1.25), (0.9, 1.0, 0.85), (1.05, 1.0, 1.0)]  # T=24000
COLS = ["key", "kind", "a_factor", "eta_factor", "rho_factor", "seed", "T", "stage1_ref", "klass", "match",
        "er_fin", "er_max", "floor_ratio", "late_drift", "bounded_breathing", "n_fin", "held_mass", "wallclock_min", "error"]


def _params(fa, fe, fr):
    pp = dict(css.FEB)
    pp["param_a"] = float(css.FEB["param_a"]) * fa
    pp["param_eta"] = float(css.FEB["param_eta"]) * fe
    pp["param_rho_vac"] = float(css.FEB["param_rho_vac"]) * fr
    return pp


def configs():
    cfgs = []
    for fa, fe, fr, tag in BOUNDARY_CELLS:
        for seed in SEEDS:
            cfgs.append({"key": f"bnd_a{fa}_e{fe}_r{fr}_s{seed}", "kind": "boundary_seed",
                         "a": fa, "e": fe, "r": fr, "seed": seed, "T": 12000, "ref": TAG2CLASS[tag]})
    for fa, fe, fr in INTERIOR_CELLS:
        cfgs.append({"key": f"int_a{fa}_e{fe}_r{fr}_T24000", "kind": "interior_T24000",
                     "a": fa, "e": fe, "r": fr, "seed": 20260619, "T": 24000, "ref": "TRUE_SATURATED_BOUND_STATE"})
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--deadline-hours", type=float, default=6.0)
    args = ap.parse_args()
    out = args.out or os.path.join(ROOT, "sweep_runs", f"FEB_JOINT_STAGE2_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    csv_path = os.path.join(out, "feb_joint_stage2_results.csv")

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
        json.dump({"N": N, "K": K, "feb_params": css.FEB, "classifier": css.classifier_spec(), "rows": rows},
                  open(os.path.join(out, "feb_joint_stage2_summary.json"), "w"), indent=2, default=float)

    todo = [c for c in configs() if c["key"] not in done]
    start = time.time(); deadline = start + args.deadline_hours * 3600
    print(f"=== FEB JOINT STAGE2 | classifier={css.classifier_spec()['version']} | {len(todo)} todo | out={out} ===", flush=True)
    for i, c in enumerate(todo, 1):
        if time.time() + (c["T"] / 1000.0 + 1.5) * 60 > deadline:
            print(f"[{i}/{len(todo)}] SKIP {c['key']} (deadline)", flush=True); continue
        t0 = time.time()
        try:
            r = css.run_probe(_params(c["a"], c["e"], c["r"]), N, c["T"], K, seed=c["seed"], ic_norm=css.IC_NORM_PER_BLOB_FIXED)
            m = r["metrics"]; pf = np.asarray(r["psi_fin"])
            held = float(np.sum(np.abs(pf) ** 2)) if np.isfinite(pf).all() else float("inf")
            match = (r["klass"] == c["ref"])
            np.savez_compressed(os.path.join(out, c["key"] + "_probe.npz"), psi_fin=pf, er=r["er"])
            row = {"key": c["key"], "kind": c["kind"], "a_factor": c["a"], "eta_factor": c["e"], "rho_factor": c["r"],
                   "seed": c["seed"], "T": c["T"], "stage1_ref": c["ref"], "klass": r["klass"], "match": match,
                   "er_fin": m["er_fin"], "er_max": m["er_max"], "floor_ratio": m.get("floor_ratio"),
                   "late_drift": m.get("late_drift"), "bounded_breathing": m.get("bounded_breathing"),
                   "n_fin": m["n_fin"], "held_mass": held, "wallclock_min": round((time.time() - t0) / 60.0, 1)}
            print(f"[{i}/{len(todo)}] {c['key']:30} -> {r['klass']:26} match={match} (ref {c['ref'][:9]}) "
                  f"drift={m.get('late_drift'):+.3f} ({row['wallclock_min']}m)", flush=True)
        except Exception as exc:
            row = {"key": c["key"], "kind": c["kind"], "a_factor": c["a"], "eta_factor": c["e"], "rho_factor": c["r"],
                   "seed": c["seed"], "T": c["T"], "stage1_ref": c["ref"], "klass": "ERROR", "error": str(exc)[:200]}
            print(f"[{i}/{len(todo)}] {c['key']} ERROR {exc}", flush=True)
        rows.append(row); flush()
    ok = [r for r in rows if r.get("klass") not in ("ERROR", None)]
    bnd = [r for r in ok if r["kind"] == "boundary_seed"]
    print(f"=== DONE: {len(ok)} ok, {round((time.time()-start)/3600,2)}h | boundary matches: "
          f"{sum(1 for r in bnd if str(r.get('match'))=='True')}/{len(bnd)} ===", flush=True)
    print(out)


if __name__ == "__main__":
    main()
