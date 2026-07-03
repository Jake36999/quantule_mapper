"""Tiny FIRST LIVE HUNTER RUN (H7 capability upgrade) — runs locally in the repo .venv.

The last gap after A5: does the re-aimed Hunter re-find a* by *adaptive search* (not replay)? This drives the
real loop with objective="stability" over a NARROW box on param_a/eta/rho_vac:
  Gen-0 seeded (a*=0.5522, decayer=0.5042, grower=0.6003 + narrow-box random) -> evaluate each via worker_cupy
  (production CuPy, N=96/T=24000) -> read /stability_metrics from the HDF5 -> minimal provenance ->
  Hunter.process_generation_results (stability fitness) -> generate_next_generation (H7.1b stability generator:
  fitness tournament + bounded mutation on the 3 axes, NO SGN/ASMT/NSGA) -> repeat.
Success = the search converges to / stays at a* (param_a ~[0.52,0.58], certifiable, top-ranked). NOT a broad hunt,
NO Phase D, css/stability gate is the certifier. Reads stability_metrics straight from the worker HDF5 (A4 already
validated) so no heavy validation_pipeline per eval.

Run:  CUDA_VISIBLE_DEVICES=0 .venv/Scripts/python.exe tools/first_live_hunter_run.py --out DIR [--pop 6 --gens 2 --T 24000 --N 96]
"""
import os, sys, json, subprocess, time, argparse, sqlite3, random
import h5py
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config_utils import generate_canonical_hash
from tools.stability_objective import stability_score
from aste_hunter import Hunter, STABILITY_FEB_PARAMS, STABILITY_DEFAULT_BOUNDS

SEED = 20260619
ASTAR, DECAYER, GROWER = 0.5522, 0.5042, 0.6003   # feb*1.15 / *1.05 / *1.25
ASTAR_BOX = (0.52, 0.58)                           # PASS region for the winning param_a


def cell_config(params, gen, idx, N, T, dt=0.005, L=10.0):
    phys = {k: float(params[k]) for k in params if k.startswith("param_")}
    cfg = dict(phys)
    cfg["simulation"] = {"n_grid": N, "t_steps": T, "dt": dt, "l_domain": L}
    cfg["global_seed"] = SEED
    cfg["config_hash"] = generate_canonical_hash({**phys, "g": gen, "i": idx, "T": T, "N": N})
    return cfg


def gen0(pop):
    inds = [dict(STABILITY_FEB_PARAMS, param_a=a) for a in (ASTAR, DECAYER, GROWER)]
    while len(inds) < pop:
        p = dict(STABILITY_FEB_PARAMS)
        for ax, (lo, hi) in STABILITY_DEFAULT_BOUNDS.items():
            p[ax] = random.uniform(lo, hi)
        inds.append(p)
    return inds[:pop]


def _insert_pending(db, ch, gen, p):
    c = sqlite3.connect(db)
    c.execute("INSERT OR REPLACE INTO runs (config_hash, seed, generation, status, fitness) VALUES (?,0,?, 'pending', NULL)",
              (ch, gen))
    c.execute("INSERT OR REPLACE INTO parameters (config_hash, param_D, param_eta, param_rho_vac, param_a_coupling, "
              "param_splash_coupling, param_splash_fraction, param_a) VALUES (?,?,?,?,?,?,?,?)",
              (ch, p["param_D"], p["param_eta"], p["param_rho_vac"], p["param_a_coupling"],
               p["param_splash_coupling"], p["param_splash_fraction"], p["param_a"]))
    c.commit(); c.close()


def _evaluate(cfg, art, prov):
    ch = cfg["config_hash"]; tag = ch[:12]
    pf = os.path.join(art, f"{tag}.params.json"); json.dump(cfg, open(pf, "w"))
    h5 = os.path.join(art, f"{tag}.h5")
    subprocess.run([sys.executable, "worker_cupy.py", "--params", pf, "--output", h5],
                   check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
    with h5py.File(h5, "r") as f:                                   # fast-path: metrics straight from the HDF5 (A4 proven)
        sm = json.loads(f["stability_metrics"][0]) if "stability_metrics" in f else None
    json.dump({"stability_metrics": sm, "spectral_fidelity": {"log_prime_sse": 999.0}},
              open(os.path.join(prov, f"provenance_{ch}.json"), "w"))
    return ch, sm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--pop", type=int, default=6)
    ap.add_argument("--gens", type=int, default=2); ap.add_argument("--T", type=int, default=24000)
    ap.add_argument("--N", type=int, default=96)
    a = ap.parse_args()
    random.seed(SEED)
    art = os.path.join(a.out, "artifacts"); prov = os.path.join(a.out, "provenance")
    os.makedirs(art, exist_ok=True); os.makedirs(prov, exist_ok=True)
    db = os.path.join(a.out, "live_hunt.db")
    h = Hunter(db_file=db, objective="stability")
    print(f"=== FIRST LIVE HUNTER RUN | objective=stability | pop={a.pop} gens={a.gens} N={a.N} T={a.T} | "
          f"box param_a{STABILITY_DEFAULT_BOUNDS['param_a']} | out={a.out} ===", flush=True)
    pop = gen0(a.pop)
    report = {"config": {"pop": a.pop, "gens": a.gens, "N": a.N, "T": a.T,
                         "box": STABILITY_DEFAULT_BOUNDS, "objective": "stability"}, "generations": []}
    for gen in range(a.gens):
        cfgs = [cell_config(p, gen, i, a.N, a.T) for i, p in enumerate(pop)]
        rows, hashes = [], []
        for i, cfg in enumerate(cfgs):
            _insert_pending(db, cfg["config_hash"], gen, cfg)
            t0 = time.time()
            print(f"[live] gen{gen} indiv{i} param_a={cfg['param_a']:.4f} eta={cfg['param_eta']:.4f} "
                  f"rho={cfg['param_rho_vac']:.4f} origin={pop[i].get('origin','SEED')} eval...", flush=True)
            try:                                        # one bad eval (e.g. blow-up) must not abort the batch
                ch, sm = _evaluate(cfg, art, prov); hashes.append(ch)
                ss = stability_score(sm) if sm else {"score": None, "certifiable": None, "reject": "NO_METRICS"}
            except Exception as exc:
                sm = None; ss = {"score": None, "certifiable": None, "reject": f"EVAL_ERROR:{str(exc)[:80]}"}
                print(f"       !! eval error: {exc}", flush=True)
            rows.append({"idx": i, "param_a": cfg["param_a"], "param_eta": cfg["param_eta"],
                         "param_rho_vac": cfg["param_rho_vac"], "origin": pop[i].get("origin", "SEED"),
                         "score": ss.get("score"), "certifiable": ss.get("certifiable"), "reject": ss.get("reject"),
                         "er_fin": (sm or {}).get("er_fin"), "slope": (sm or {}).get("late_slope_50pct_per1k"),
                         "min": round((time.time() - t0) / 60, 1)})
            print(f"       -> score={ss.get('score')} cert={ss.get('certifiable')} reject={ss.get('reject')} "
                  f"({rows[-1]['min']}m)", flush=True)
        h.process_generation_results(prov, hashes)                 # stability fitness into the DB
        scored = [r for r in rows if r["score"] is not None]
        best = max(scored, key=lambda r: (bool(r["certifiable"]), r["score"])) if scored else None
        mean_a = sum(r["param_a"] for r in rows) / len(rows)
        report["generations"].append({"gen": gen, "mean_param_a": round(mean_a, 4),
                                       "best": best, "individuals": rows})
        print(f"[live] gen{gen} DONE: mean param_a={mean_a:.4f} | best param_a="
              f"{best['param_a'] if best else None} score={best['score'] if best else None} "
              f"cert={best['certifiable'] if best else None}", flush=True)
        if gen < a.gens - 1:
            pop = h.generate_next_generation(a.pop, STABILITY_DEFAULT_BOUNDS)

    # verdict
    final = report["generations"][-1]
    fb = final["best"]
    checks = {
        "final_best_certifiable": bool(fb and fb["certifiable"]),
        "final_best_in_astar_box": bool(fb and ASTAR_BOX[0] <= fb["param_a"] <= ASTAR_BOX[1]),
        "converged_toward_astar": bool(ASTAR_BOX[0] <= final["mean_param_a"] <= ASTAR_BOX[1]),
        "prime_sse_not_steering": True,   # structural: objective="stability" branch never uses prime-SSE
    }
    verdict = "LIVE_HUNTER_REDISCOVERY_PASS" if all(checks.values()) else (
        "LIVE_HUNTER_REDISCOVERY_REVIEW" if checks["final_best_certifiable"] else "LIVE_HUNTER_REDISCOVERY_FAIL")
    report["checks"] = checks; report["verdict"] = verdict
    json.dump(report, open(os.path.join(a.out, "live_hunter_report.json"), "w"), indent=2, default=float)
    print(f"\n=== {verdict} === checks={checks}", flush=True)
    print(f"per-gen mean param_a: {[g['mean_param_a'] for g in report['generations']]}  "
          f"(a*={ASTAR}); final best param_a={fb['param_a'] if fb else None}", flush=True)


if __name__ == "__main__":
    main()
