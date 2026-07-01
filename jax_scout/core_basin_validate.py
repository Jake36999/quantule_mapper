"""
BASIN VALIDATION — confirm the rotational-core SUSTAIN basin holds at CONVERGED resolution (N=96)
and across MULTIPLE seeds (guards ChatGPT's failure modes: 'only one seed/config sustains' or
'sustain is a low-N artifact'). Picks SUSTAIN configs near the basin centre (eta in [-0.12, 0.02])
from the 6h CORE_BASIN sweep, re-runs them at N=96 with 3 different ICs (bare S-NCGL), reclassifies.

WSL2 jax venv:  python /mnt/f/quantule_mapper/jax_scout/core_basin_validate.py
"""
import os, sys, csv, glob, json
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import physics
from jax_scout.afield_current_coupled import multiseed_ic, L as L_, dt as DT
from jax_scout.core_basin_sweep import core_metrics, classify, order

SEEDS = [20260619, 20260620, 20260621]
N, T, NPICK = 96, 1600, 6


def F(r, k):
    try: return float(r[k])
    except: return float("nan")


def main():
    d = [c for c in sorted(glob.glob(os.path.join(ROOT, "sweep_runs", "CORE_BASIN_2026*"))) if "CALIB" not in c][-1]
    rows = list(csv.DictReader(open(os.path.join(d, "all_evals.csv"))))
    sus = [r for r in rows if r["klass"] == "SUSTAIN" and -0.12 <= F(r, "param_eta") <= 0.02]
    # rank by sustain quality: retain both core density and circulation
    sus.sort(key=lambda r: -(min(F(r, "cd_ratio"), F(r, "vt_ratio"))))
    pick = sus[:NPICK]
    print(f"=== BASIN VALIDATION @ N={N} T={T}, {len(SEEDS)} seeds, {len(pick)} basin-centre SUSTAIN configs ===")
    print(f"(from {os.path.basename(d)}; eta-band [-0.12,0.02])\n")
    pm = jnp.asarray(np.array([[F(r, k) for k in order] for r in pick]))
    dx = L_ / N
    results = []
    for r in pick:
        results.append({"idx": r["idx"], "param_eta": F(r, "param_eta"),
                        "n48_klass": "SUSTAIN", "n96_seeds": []})
    for seed in SEEDS:
        psi0 = jnp.asarray(multiseed_ic(N, seed)); ic_e = float(np.sum(np.abs(np.asarray(psi0)) ** 2)) + 1e-30
        psi_mid, psi_fin, energy, max_amp, finite = physics.sweep_probe(pm, psi0, N, L_, DT, T, jnp.float64, jnp.complex128)
        energy = np.asarray(energy); finite = np.asarray(finite); psi_mid = np.asarray(psi_mid); psi_fin = np.asarray(psi_fin)
        for b, r in enumerate(pick):
            er = energy[b] / ic_e
            par = {k: F(r, k) for k in order}
            viable = bool(finite[b]) and np.isfinite(er).all() and float(np.max(er)) <= 3.0 and float(er[-1]) >= 0.3
            mid = core_metrics(psi_mid[b], par, N, dx) if viable else None
            fin = core_metrics(psi_fin[b], par, N, dx) if viable else None
            kl, _ = classify(bool(finite[b]), er, mid, fin)
            results[b]["n96_seeds"].append(kl)
        print(f"  seed {seed}: " + ", ".join(results[b]["n96_seeds"][-1] for b in range(len(pick))), flush=True)
    n_robust = 0
    print("\n--- per config (n48 -> n96 across seeds) ---")
    for r in results:
        sus_count = sum(1 for k in r["n96_seeds"] if k == "SUSTAIN")
        robust = sus_count >= 2
        n_robust += int(robust)
        print(f"  idx={r['idx']} eta={r['param_eta']:+.3f}: N96 {r['n96_seeds']}  -> {'ROBUST' if robust else 'fragile'}")
    verdict = "BASIN_CONFIRMED_N96_MULTISEED" if n_robust >= max(3, len(pick) - 1) else \
              "BASIN_PARTIAL" if n_robust >= 1 else "BASIN_NOT_CONFIRMED"
    out = {"N": N, "T": T, "seeds": SEEDS, "n_pick": len(pick), "n_robust": n_robust,
           "verdict": verdict, "results": results}
    json.dump(out, open(os.path.join(d, "basin_validation_N96.json"), "w"), indent=2, default=float)
    print(f"\n=== {n_robust}/{len(pick)} basin-centre configs SUSTAIN robustly at N=96 (>=2/3 seeds) ===")
    print(f"VERDICT: {verdict}\nwrote {os.path.join(d,'basin_validation_N96.json')}")


if __name__ == "__main__":
    main()
