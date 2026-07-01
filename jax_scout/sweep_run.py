"""
CORRECTED_PHYSICS_JAX_SCOUT  /  NOT_CUPY_FINAL_EVIDENCE

JAX vmap scout sweep over the corrected-physics parameter space (bounds from
burn_in_config.json). Evolves a batch of configs in parallel on GPU, computes a
cheap host-side spectral-structure proxy, and saves the top candidates' final
fields for canonical CuPy scoring (jax_scout/score_sweep.py).

The JAX scout only FINDS candidates. CuPy (source of truth) validates them.

Run in WSL2 jax venv:
  python /mnt/f/quantule_mapper/jax_scout/sweep_run.py --size 256 --batch 16 --N 48 --steps 600
"""
import os
import sys
import csv
import json
import time
import argparse
import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import physics

LABEL = "CORRECTED_PHYSICS_JAX_SCOUT"
EVIDENCE = "NOT_CUPY_FINAL_EVIDENCE"
SEED = 20260619


def radial_structure(psi_final):
    """Cheap spectral-structure proxy on the density field (numpy, host-side)."""
    rho = np.abs(psi_final) ** 2
    power = np.abs(np.fft.fftn(rho)) ** 2
    N = rho.shape[0]
    kx = np.fft.fftfreq(N) * N
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
    kr = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2).astype(int)
    prof = np.bincount(kr.ravel(), weights=power.ravel()) / np.maximum(np.bincount(kr.ravel()), 1)
    p = prof[2:N // 2]  # skip DC + first bin, up to Nyquist
    if p.size < 4 or np.median(p) <= 0:
        return 0.0, 0
    peak_to_median = float(np.max(p) / np.median(p))
    # count interior local maxima above 3x median
    thr = 3.0 * np.median(p)
    nloc = int(np.sum((p[1:-1] > p[:-2]) & (p[1:-1] > p[2:]) & (p[1:-1] > thr)))
    return peak_to_median, nloc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--N", type=int, default=48)
    ap.add_argument("--L", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--bounds-file", type=str, default=None,
                    help="JSON of {param: [lo,hi]} overriding burn_in_config bounds (e.g. gain_bounds.json)")
    ap.add_argument("--outdir", type=str,
                    default=r"/mnt/f/quantule_mapper/sweep_runs")
    args = ap.parse_args()

    # bounds: explicit file (e.g. edge-of-stability gain_bounds.json) or production config
    if args.bounds_file:
        bounds = json.load(open(args.bounds_file))
    else:
        bounds = json.load(open(os.path.join(ROOT, "burn_in_config.json")))["bounds"]
    lo = np.array([bounds[k][0] for k in physics.SWEEP_PARAM_ORDER], dtype=np.float64)
    hi = np.array([bounds[k][1] for k in physics.SWEEP_PARAM_ORDER], dtype=np.float64)

    # seed policy: deterministic Latin-Hypercube over the 7-d box
    from scipy.stats import qmc
    sampler = qmc.LatinHypercube(d=len(physics.SWEEP_PARAM_ORDER), seed=SEED)
    unit = sampler.random(args.size)
    params = lo + unit * (hi - lo)  # [size, n_params]

    # shared IC (one Gaussian+noise field; differences are purely parametric)
    rng = np.random.default_rng(SEED)
    x = np.linspace(-args.L / 2, args.L / 2, args.N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    psi0 = np.exp(-(X ** 2 + Y ** 2 + Z ** 2) / 2.0).astype(np.complex128)
    psi0 += 0.01 * (rng.standard_normal(psi0.shape) + 1j * rng.standard_normal(psi0.shape))
    psi0_j = jnp.asarray(psi0)

    outdir = os.path.join(args.outdir, f"{LABEL}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(outdir, exist_ok=True)
    print(f"{LABEL} / {EVIDENCE}")
    print(f"backend={jax.default_backend()} dtype=complex128(FP64) N={args.N}^3 dt={args.dt} "
          f"steps={args.steps} size={args.size} batch={args.batch} seed={SEED}")
    print(f"bounds: {dict(zip(physics.SWEEP_PARAM_ORDER, zip(lo.tolist(), hi.tolist())))}")
    print(f"out -> {outdir}\n")

    rows, fields, t0 = [], {}, time.time()
    for b0 in range(0, args.size, args.batch):
        pb = params[b0:b0 + args.batch]
        pbj = jnp.asarray(pb)
        flds, max_amp, e0, e1, finite = physics.sweep(
            pbj, psi0_j, args.N, args.L, args.dt, args.steps,
            jnp.float64, jnp.complex128)
        flds = np.asarray(flds); max_amp = np.asarray(max_amp)
        e0 = np.asarray(e0); e1 = np.asarray(e1); finite = np.asarray(finite)
        for j in range(pb.shape[0]):
            idx = b0 + j
            # gain regime can blow up -> NaN/Inf; don't feed those to the proxy
            ptm, nloc = (radial_structure(flds[j]) if bool(finite[j]) else (0.0, 0))
            er = float(e1[j] / e0[j]) if (e0[j] > 0 and np.isfinite(e1[j])) else 0.0
            row = {"idx": idx, **{k: float(pb[j][i]) for i, k in enumerate(physics.SWEEP_PARAM_ORDER)},
                   "max_amp": float(max_amp[j]), "energy_ratio": er,
                   "finite": bool(finite[j]), "peak_to_median": ptm, "n_local_max": nloc}
            rows.append(row)
            fields[idx] = flds[j]
        print(f"  batch {b0//args.batch+1}/{-(-args.size//args.batch)} done "
              f"({b0+pb.shape[0]}/{args.size})  [{time.time()-t0:.0f}s]", flush=True)

    # rank by REAL spectral peaks (interior local maxima), then prominence; require
    # stability (finite, not collapsed, not saturated). peak_to_median alone is high
    # even for smooth decaying fields, so n_local_max is the true discriminator.
    def cand_key(r):
        stable = r["finite"] and r["energy_ratio"] > 1e-3 and r["max_amp"] < 1e3
        return (stable, r["n_local_max"], r["peak_to_median"])
    rows.sort(key=cand_key, reverse=True)

    with open(os.path.join(outdir, "sweep_results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    top = rows[:args.topk]
    np.savez_compressed(
        os.path.join(outdir, "sweep_fields_top.npz"),
        idx=np.array([r["idx"] for r in top]),
        params=np.array([[r[k] for k in physics.SWEEP_PARAM_ORDER] for r in top]),
        fields=np.stack([fields[r["idx"]] for r in top]),
        param_order=np.array(physics.SWEEP_PARAM_ORDER))
    meta = {"label": LABEL, "evidence": EVIDENCE, "seed": SEED, "size": args.size,
            "N": args.N, "L": args.L, "dt": args.dt, "steps": args.steps, "dtype": "complex128",
            "param_order": physics.SWEEP_PARAM_ORDER,
            "bounds": {k: bounds[k] for k in physics.SWEEP_PARAM_ORDER},
            "runtime_s": time.time() - t0}
    json.dump(meta, open(os.path.join(outdir, "sweep_meta.json"), "w"), indent=2)

    npk = sum(1 for r in rows if r["n_local_max"] >= 1 and r["finite"]
              and r["energy_ratio"] > 1e-3 and r["max_amp"] < 1e3)
    print(f"\nDONE {args.size} configs in {(time.time()-t0)/60:.1f} min")
    print(f"peak-producing (>=1 local max, stable): {npk}")
    print("top 5 by structure proxy:")
    for r in rows[:5]:
        print(f"  idx={r['idx']:4d} p2m={r['peak_to_median']:7.2f} nloc={r['n_local_max']} "
              f"max_amp={r['max_amp']:.2e} e_ratio={r['energy_ratio']:.3f}")
    print(f"\nfields of top {args.topk} saved for CuPy scoring -> {outdir}/sweep_fields_top.npz")


if __name__ == "__main__":
    main()
