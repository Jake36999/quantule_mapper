"""H7 RE-VALIDATION — can the re-aimed stability objective re-discover the known a* basin? (NOT a new hunt.)

Searches the validated sensitive axes (param_a × param_eta × param_rho_vac) around the feb basin with jax_scout,
scores each candidate with tools.stability_objective (NO prime-SSE steering), and records the css.classify
verdict as certification. The question is narrow and pre-registered: does the objective rank the known
a*≈×1.15 / eta×1.0 balance region highest, with known decayers/growers/failures below?

Staged: Stage B = cheap-filter run at a moderate T (ranks, does NOT certify — window gate). Stage C (certifiable
T≥24000) is done separately on the top region (or via the existing joint-basin T24000 data). Small,
re-validation-focused — anchored to "can it re-find a*", not "can it find something new". jax_scout only, no cupy.
WSL2 jax venv:  python jax_scout/hunter_reaim_rediscovery.py [--T 8000] [--out DIR]
"""
import os, sys, csv, json, time, argparse, itertools
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
import jax
jax.config.update("jax_enable_x64", True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import core_saturation_search as css
from tools.stability_objective import stability_score

N, K, SEED = 96, 6, 20260619
# search grid on the validated sensitive axes (factors x feb); center (1.15, 1.0, 1.0) = the known a*.
A_FACTORS = [1.05, 1.15, 1.25]
ETA_FACTORS = [0.85, 1.0, 1.15]
RHO_FACTORS = [1.0]
ASTAR = (1.15, 1.0, 1.0)   # the pre-registered target
COLS = ["a_factor", "eta_factor", "rho_factor", "is_astar", "klass", "score", "certifiable", "reject",
        "c_slope", "c_band", "late_slope_50pct_per1k", "er_fin", "er_max", "floor_ratio", "n_fin", "wallclock_min"]


def late_slope_per1k(er):
    er = np.asarray(er, float); n = er.size
    s = slice(n // 2, n)
    return float(np.polyfit(np.arange(n)[s], er[s], 1)[0] * 1000.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=8000, help="Stage-B cheap-filter window (ranks, not certifies)")
    ap.add_argument("--a-factors", default=None, help="comma list overriding A_FACTORS (Stage-C narrow grid)")
    ap.add_argument("--eta-factors", default=None, help="comma list overriding ETA_FACTORS")
    ap.add_argument("--rho-factors", default=None, help="comma list overriding RHO_FACTORS")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    afs = [float(x) for x in args.a_factors.split(",")] if args.a_factors else A_FACTORS
    efs = [float(x) for x in args.eta_factors.split(",")] if args.eta_factors else ETA_FACTORS
    rfs = [float(x) for x in args.rho_factors.split(",")] if args.rho_factors else RHO_FACTORS
    out = args.out or os.path.join(ROOT, "sweep_runs", f"HUNTER_REAIM_REDISCOVERY_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    csv_path = os.path.join(out, "rediscovery_results.csv")

    grid = list(itertools.product(afs, efs, rfs))
    print(f"=== HUNTER RE-AIM RE-DISCOVERY (H7 re-validation) | grid={len(grid)} cells | T={args.T} (Stage-B "
          f"cheap filter) | scorer=stability_objective (NO prime-SSE) | out={out} ===", flush=True)
    rows = []
    for (af, ef, rf) in grid:
        pp = dict(css.FEB)
        pp["param_a"] = float(css.FEB["param_a"]) * af
        pp["param_eta"] = float(css.FEB["param_eta"]) * ef
        pp["param_rho_vac"] = float(css.FEB["param_rho_vac"]) * rf
        t0 = time.time()
        try:
            r = css.run_probe(pp, N, args.T, K, seed=SEED, ic_norm=css.IC_NORM_PER_BLOB_FIXED)
            m = r["metrics"]; er = np.asarray(r["er"], float)
            metrics = {"er_fin": m["er_fin"], "er_max": m["er_max"], "floor_ratio": m.get("floor_ratio", 0.0),
                       "late_slope_50pct_per1k": late_slope_per1k(er), "late_drift": m.get("late_drift"), "T": args.T}
            ss = stability_score(metrics)
            row = {"a_factor": af, "eta_factor": ef, "rho_factor": rf, "is_astar": (af, ef, rf) == ASTAR,
                   "klass": r["klass"], "score": ss["score"], "certifiable": ss["certifiable"], "reject": ss["reject"],
                   "c_slope": ss["components"].get("slope"), "c_band": ss["components"].get("band"),
                   "late_slope_50pct_per1k": round(metrics["late_slope_50pct_per1k"], 5),
                   "er_fin": round(m["er_fin"], 3), "er_max": round(m["er_max"], 3),
                   "floor_ratio": round(m.get("floor_ratio", 0.0), 3), "n_fin": m["n_fin"],
                   "wallclock_min": round((time.time() - t0) / 60, 1)}
        except Exception as exc:
            row = {"a_factor": af, "eta_factor": ef, "rho_factor": rf, "klass": "ERROR", "score": -9, "reject": str(exc)[:120]}
        rows.append(row)
        print(f"  a×{af} eta×{ef} rho×{rf} {'[A*]' if row.get('is_astar') else '    '} -> {row.get('klass')} "
              f"score={row.get('score')} slope={row.get('late_slope_50pct_per1k')} er_fin={row.get('er_fin')} "
              f"({row.get('wallclock_min')}m)", flush=True)
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

    ranked = sorted([r for r in rows if r.get("score") is not None], key=lambda r: r["score"], reverse=True)
    top = ranked[0] if ranked else {}
    astar_rank = next((i for i, r in enumerate(ranked) if r.get("is_astar")), None)
    verdict = "REDISCOVERY_PASS" if (astar_rank is not None and astar_rank <= 1) else "REDISCOVERY_REVIEW"
    print(f"=== ranked top: a×{top.get('a_factor')} eta×{top.get('eta_factor')} score={top.get('score')} | "
          f"a* rank = {astar_rank} (0-indexed) -> {verdict} ===", flush=True)
    json.dump({"grid": grid, "astar": ASTAR, "T": args.T, "verdict": verdict, "astar_rank": astar_rank,
               "ranked": ranked, "note": "Stage-B cheap filter: ranks, does not certify (window gate); certifiable "
               "confirmation via existing joint-basin T24000 or a Stage-C rerun."},
              open(os.path.join(out, "rediscovery_summary.json"), "w"), indent=2, default=float)
    print(f"=== DONE {out} ===", flush=True)


if __name__ == "__main__":
    main()
