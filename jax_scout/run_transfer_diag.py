"""
Re-analyse the existing stable bounded-node candidate population with the FMIA
transfer / interaction-rate diagnostic (jax_scout/transfer_diag.py) and re-classify.

This is the next step the user specified: NOT another raw hunt. We take the bounded
multi-node configs already found (ADAPTIVE_HUNT_20260620_082624, the clean doubly-gated
run), re-run each capturing a time-resolved trajectory, and measure whether the nodes
exchange energy / phase-current / route information through Omega^2 corridors / share an
interference lattice -- the IRER-FMIA signature -- rather than only testing destructive
ablation.

WSL2 jax venv:  python /mnt/f/quantule_mapper/jax_scout/run_transfer_diag.py
"""
import os, sys, csv, glob, json, time
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")
import jax
jax.config.update("jax_enable_x64", True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import physics, transfer_diag as td

SEED, K = 20260619, 6
N, L, dt, STEPS, N_SNAP = 48, 10.0, 0.005, 800, 40
order = physics.SWEEP_PARAM_ORDER
N_CAND = int(os.environ.get("TD_NCAND", "12"))


def intact_ic(N, L):
    """Same 6-seed multiseed intact IC as validate_candidates.py (reproducible)."""
    rng = np.random.default_rng(SEED)
    x = np.linspace(-L/2, L/2, N, endpoint=False); X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    w = L/12.0; psi = np.zeros((N, N, N), np.complex128)
    for _ in range(K):
        cx, cy, cz = rng.uniform(-L/2, L/2, 3)
        psi += np.exp(-((X-cx)**2+(Y-cy)**2+(Z-cz)**2)/(2*w**2))
    noise = 0.01*(rng.standard_normal((N, N, N))+1j*rng.standard_normal((N, N, N)))
    return (psi + noise).astype(np.complex128)


def load_candidates():
    d = sorted(glob.glob(os.path.join(ROOT, "sweep_runs", "ADAPTIVE_HUNT_2026062*")))[-1]
    rows = list(csv.DictReader(open(os.path.join(d, "all_evals.csv"))))
    def f(r, k, dv=float("nan")):
        try: return float(r[k])
        except: return dv
    bnd = [r for r in rows if r["reject"] == "" and 2 <= f(r, "intact_nodes") <= 20
           and f(r, "curv") < 1.0 and 0.1 <= f(r, "er") <= 5.0]
    bnd.sort(key=lambda r: f(r, "iso_surv"))      # most collective first
    return os.path.basename(d), bnd[:N_CAND]


METRICS = ["n_persistent_nodes", "interaction_graph_density", "mean_transfer_strength",
           "max_transfer_strength", "energy_exchange_index", "phase_coupling_score",
           "geometric_path_alignment", "omega_corridor_conductance",
           "interference_lattice_overlap", "action_rate_coherence",
           "raw_phase_lock", "raw_E_xcorr", "raw_interference"]


def main():
    src, cands = load_candidates()
    ic = intact_ic(N, L)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(ROOT, "sweep_runs", f"TRANSFER_DIAG_{stamp}")
    os.makedirs(outdir, exist_ok=True)
    print(f"FMIA transfer diagnostic [{td.TRANSFER_DIAG_CONTRACT_VERSION}]")
    print(f"source={src}  candidates={len(cands)}  N={N}/{STEPS} snaps={N_SNAP}\n")

    results = []
    for n, r in enumerate(cands):
        pvec = [float(r[k]) for k in order]
        par = {k: float(r[k]) for k in order}
        iso = float(r.get("iso_surv", "nan") or "nan")
        ba = float(r.get("bounded_abl_sens", "0") or "0")
        t0 = time.time()
        res = td.analyze_candidate(pvec, par, ic, N, L, dt, STEPS, N_SNAP,
                                   bounded_abl_sens=ba, iso_surv=iso)
        res["gen"] = r["gen"]; res["family"] = r["family"]
        res.update({k: par[k] for k in order})
        results.append(res)
        dtt = time.time() - t0
        print(f"[{n+1:2d}/{len(cands)}] gen{r['gen']:>2} iso={iso:.2f} bAbl={ba:.2f} "
              f"nP={res.get('n_persistent_nodes',0)} "
              f"Jflux={res.get('mean_transfer_strength',0):.3f} "
              f"align={res.get('geometric_path_alignment',0):.2f} "
              f"cond={res.get('omega_corridor_conductance',0):.2f} "
              f"|exch={res.get('energy_exchange_index',0):.3f} "
              f"pcoup={res.get('phase_coupling_score',0):.3f} "
              f"arc={res.get('action_rate_coherence',0):.3f} "
              f"interf={res.get('interference_lattice_overlap',0):.3f} "
              f"-> {res['klass']}  ({dtt:.1f}s)")

    # write CSV
    cols = (["gen", "family", "klass", "finite", "amp_final", "iso_surv", "bounded_abl_sens"]
            + METRICS + order)
    with open(os.path.join(outdir, "transfer_evals.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for res in results:
            w.writerow(res)
    json.dump({"contract": td.TRANSFER_DIAG_CONTRACT_VERSION, "source": src,
               "N": N, "steps": STEPS, "n_snap": N_SNAP, "n_candidates": len(cands),
               "results": results}, open(os.path.join(outdir, "transfer_diag.json"), "w"),
              indent=2, default=float)

    # summary
    print("\n=== classification tally ===")
    tally = {}
    for res in results:
        tally[res["klass"]] = tally.get(res["klass"], 0) + 1
    for k, v in sorted(tally.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    print("\n=== metric ranges across population (for threshold calibration) ===")
    for k in METRICS:
        vals = [res.get(k, 0.0) for res in results if res.get("finite")]
        if vals:
            print(f"  {k:30s} min={min(vals):.3f} med={np.median(vals):.3f} max={max(vals):.3f}")
    print(f"\nwrote {outdir}")


if __name__ == "__main__":
    main()
