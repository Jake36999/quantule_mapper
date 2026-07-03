"""A5 -- production H7 re-validation harness (runs locally in the repo .venv; no separate machine).

Question (same as the jax_scout re-discovery, but on the PRODUCTION CuPy path): with production `stability_metrics`
now flowing worker -> HDF5 -> validation_pipeline -> provenance -> Hunter (A3/A4/A4b), does the re-aimed objective
re-find a*≈×1.15 as the certifiable long-time attractor, rank the matched controls below it, and REFUSE to promote a
short-window (T=12000) run? This does NOT change physics/geometry/gate; it replays a fixed param set and scores it.

Two commands:
  build-configs --out DIR    # writes worker_cupy `--params` JSON per cell (pure python, no CuPy)
  evaluate --provenance-dir DIR   # scores the resulting provenance reports and prints the A5 verdict (no CuPy)

Between them, in the repo .venv (.venv/Scripts/python.exe; cupy 14.0.1, local), each cell runs through
worker_cupy + validation_pipeline (see docs/PRODUCTION_H7_REVALIDATION_RUNBOOK.md). This driver is jax-free.
"""
import os, sys, json, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config_utils import generate_canonical_hash
from orchestrator.run_identity import resolve_provenance_report
from tools.stability_objective import stability_score, MIN_STABILITY_T

# feb params (frozen; == jax_scout core_saturation_search.FEB). a* = param_a * 1.15. Production solver reads
# param_s/param_f directly (solver/core.py:71-72), so css.FEB maps verbatim.
FEB = {"param_D": 2.7329, "param_eta": 0.0704, "param_rho_vac": 1.1866, "param_omega0": 0.0,
       "param_a_coupling": 2.3098, "param_s": 0.0129, "param_f": -0.4861, "param_a": 0.4802}
N_GRID, DT, L_DOMAIN, SEED = 96, 0.005, 10.0, 20260619
LONG_T, SHORT_T = 36000, 12000     # LONG_T >= MIN_STABILITY_T (certifiable); SHORT_T is the window-artifact probe

# (name, param_a factor, T_steps, role). a* + two matched controls at LONG_T + a short-window a* non-promotion probe.
CELLS = [
    ("astar_longT",   1.15, LONG_T,  "astar"),               # target: certifiable, top-ranked
    ("decayer_longT", 1.05, LONG_T,  "control_decayer"),     # below a*
    ("grower_longT",  1.25, LONG_T,  "control_grower"),      # below a* (band penalty / reject)
    ("astar_shortT",  1.15, SHORT_T, "window_artifact"),     # MUST NOT certify (window gate)
]


def cell_config(name, a_factor, t_steps):
    """Full worker_cupy `--params` config for a cell, with a deterministic config_hash (== what the worker
    stamps into /identity and what validation_pipeline names the provenance by)."""
    params = dict(FEB); params["param_a"] = round(FEB["param_a"] * a_factor, 10)
    cfg = {**params,
           "simulation": {"n_grid": N_GRID, "t_steps": int(t_steps), "dt": DT, "l_domain": L_DOMAIN},
           "global_seed": SEED}
    cfg["config_hash"] = generate_canonical_hash(cfg)
    return cfg


def build_configs(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    index = []
    for (name, af, t, role) in CELLS:
        cfg = cell_config(name, af, t)
        path = os.path.join(out_dir, f"{name}.params.json")
        json.dump(cfg, open(path, "w"), indent=2)
        index.append({"name": name, "role": role, "a_factor": af, "T": t,
                      "config_hash": cfg["config_hash"], "params_file": path})
        print(f"  {name:14s} role={role:16s} a×{af} param_a={cfg['param_a']} T={t} -> {cfg['config_hash'][:12]}")
    json.dump(index, open(os.path.join(out_dir, "cells_index.json"), "w"), indent=2)
    print(f"=== wrote {len(CELLS)} configs + cells_index.json to {out_dir} ===")
    return index


def evaluate(provenance_dir):
    rows = []
    for (name, af, t, role) in CELLS:
        cfg = cell_config(name, af, t); ch = cfg["config_hash"]
        prov_path = resolve_provenance_report(provenance_dir, ch)
        row = {"name": name, "role": role, "a_factor": af, "T": t, "config_hash": ch,
               "provenance": prov_path, "score": None, "certifiable": None, "reject": None, "status": "MISSING"}
        if prov_path and os.path.exists(prov_path):
            try:
                prov = json.load(open(prov_path))
                sm = prov.get("stability_metrics")
                if sm:
                    ss = stability_score(sm)
                    row.update(score=ss["score"], certifiable=ss["certifiable"], reject=ss["reject"],
                               status="SCORED")
                else:
                    row["status"] = "NO_STABILITY_METRICS"
            except Exception as exc:
                row["status"] = f"ERROR:{str(exc)[:60]}"
        rows.append(row)

    by = {r["name"]: r for r in rows}
    astar, decay, grow, short = by["astar_longT"], by["decayer_longT"], by["grower_longT"], by["astar_shortT"]
    checks, reasons = {}, []

    def scored(r): return r["status"] == "SCORED" and r["score"] is not None
    if not all(scored(r) for r in (astar, decay, grow, short)):
        missing = [r["name"] for r in rows if not scored(r)]
        return {"verdict": "INCOMPLETE", "reason": f"cells not scored: {missing}", "rows": rows}

    checks["astar_certifiable"] = bool(astar["certifiable"])
    checks["astar_top_of_longT"] = astar["score"] >= max(decay["score"], grow["score"])
    checks["decayer_below_astar"] = decay["score"] < astar["score"]
    checks["grower_below_astar"] = grow["score"] < astar["score"]
    checks["shortT_not_certifiable"] = not short["certifiable"]     # window gate holds -> no artifact promotion
    for k, ok in checks.items():
        if not ok:
            reasons.append(k)
    verdict = "PRODUCTION_H7_REVALIDATION_PASS" if all(checks.values()) else "PRODUCTION_H7_REVALIDATION_REVIEW"
    return {"verdict": verdict, "checks": checks, "failed_checks": reasons, "rows": rows,
            "note": "css.classify / production gate remains the certifier; this scores the objective's re-find. "
                    "Production IC = single Gaussian (solver/run.py) differs from jax_scout multiseed -> a genuine "
                    "cross-IC re-validation of a*."}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-configs"); b.add_argument("--out", required=True)
    e = sub.add_parser("evaluate"); e.add_argument("--provenance-dir", required=True)
    e.add_argument("--json-out", default=None)
    args = ap.parse_args()
    if args.cmd == "build-configs":
        build_configs(args.out)
    else:
        res = evaluate(args.provenance_dir)
        print(f"\n{'name':16s} {'role':16s} {'score':>7} {'cert':>5} {'status'}")
        for r in res["rows"]:
            sc = f"{r['score']:.3f}" if r["score"] is not None else "  -  "
            print(f"{r['name']:16s} {r['role']:16s} {sc:>7} {str(r['certifiable']):>5} {r['status']}")
        print(f"\n=== {res['verdict']} ===")
        if res.get("failed_checks"):
            print(f"failed checks: {res['failed_checks']}")
        if args.json_out:
            json.dump(res, open(args.json_out, "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
