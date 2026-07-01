"""
FOCUSED ETA-BAND REFINEMENT — turn the 'thin sustain band' into an empirical stability CURVE
P(SUSTAIN | eta) at converged resolution (N=96). Tests whether eta is the master knob INDEPENDENT
of the other GL params (ChatGPT failure mode: 'band depends on hidden a/s/f/D interactions').

Design: eta grid over [-0.08, 0.02]; for each eta, evaluate over a fixed set of L permissive
'background' samples of the OTHER params (a,s,f,D,a_coupling,rho_vac,omega0) x S seeds (ICs).
P(class|eta) = fraction over (L x S) samples. Same backgrounds reused across all eta & seeds so the
curve isolates eta. Bare S-NCGL (no A, no coupling, no solver mod), engine physics.sweep_probe.

WSL2 jax venv:  python /mnt/f/quantule_mapper/jax_scout/core_basin_refine.py
"""
import os, sys, csv, glob, json, time
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

N, T = 96, 1600
ETA_GRID = [round(x, 4) for x in np.linspace(-0.08, 0.02, 11)]
SEEDS = [20260619, 20260620, 20260621]
L_BG = 8                       # permissive background samples of the OTHER params
CHUNK = 8                      # sweep_probe batch (memory-safe at N=96)
ETA_IDX = order.index("param_eta")
CLASSES = ["SUSTAIN", "SPIN_DOWN", "COLLAPSE", "BLOWUP", "FRAGMENT", "VIABLE_NO_NODES"]


def main():
    bounds = json.load(open(os.path.join(ROOT, "jax_scout", "gain_bounds.json")))
    lo = np.array([bounds[k][0] for k in order]); hi = np.array([bounds[k][1] for k in order])
    rng = np.random.default_rng(20260622)
    # fixed permissive backgrounds (other params); eta column overwritten per grid point
    bg = np.array([lo + rng.random(len(order)) * (hi - lo) for _ in range(L_BG)])
    outdir = os.path.join(ROOT, "sweep_runs", f"CORE_BASIN_REFINE_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(outdir, exist_ok=True)
    cols = ["eta", "seed", "bg", "klass", "er_fin", "core_fin", "vt_fin", *order]
    log = open(os.path.join(outdir, "all_evals.csv"), "w", newline=""); cw = csv.DictWriter(log, fieldnames=cols, extrasaction="ignore"); cw.writeheader()
    dx = L_ / N
    # build full config list: (eta, seed, bg) -> param vec
    counts = {e: {c: 0 for c in CLASSES} for e in ETA_GRID}
    t0 = time.time()
    print(f"=== ETA-BAND REFINE N={N} T={T} | eta {ETA_GRID} | {len(SEEDS)} seeds x {L_BG} backgrounds ===", flush=True)
    for seed in SEEDS:
        psi0 = jnp.asarray(multiseed_ic(N, seed)); ic_e = float(np.sum(np.abs(np.asarray(psi0)) ** 2)) + 1e-30
        # configs for this seed: all (eta x bg)
        vecs, meta = [], []
        for e in ETA_GRID:
            for j in range(L_BG):
                v = bg[j].copy(); v[ETA_IDX] = e
                vecs.append(v); meta.append((e, j))
        vecs = np.array(vecs)
        for s in range(0, len(vecs), CHUNK):
            chunk = vecs[s:s + CHUNK]
            pad = CHUNK - len(chunk)
            if pad:
                chunk = np.vstack([chunk, np.repeat(chunk[-1:], pad, 0)])
            pm = jnp.asarray(chunk)
            psi_mid, psi_fin, energy, max_amp, finite = physics.sweep_probe(pm, psi0, N, L_, DT, T, jnp.float64, jnp.complex128)
            energy = np.asarray(energy); finite = np.asarray(finite); psi_mid = np.asarray(psi_mid); psi_fin = np.asarray(psi_fin)
            for b in range(len(chunk) - pad):
                e, j = meta[s + b]; er = energy[b] / ic_e
                par = {k: float(chunk[b][i]) for i, k in enumerate(order)}
                viable = bool(finite[b]) and np.isfinite(er).all() and float(np.max(er)) <= 3.0 and float(er[-1]) >= 0.3
                mid = core_metrics(psi_mid[b], par, N, dx) if viable else None
                fin = core_metrics(psi_fin[b], par, N, dx) if viable else None
                kl, m = classify(bool(finite[b]), er, mid, fin)
                counts[e][kl] = counts[e].get(kl, 0) + 1
                cw.writerow({"eta": e, "seed": seed, "bg": j, "klass": kl,
                             "er_fin": m.get("er_fin"), "core_fin": m.get("core_fin"), "vt_fin": m.get("vt_fin"),
                             **{k: round(float(chunk[b][i]), 4) for i, k in enumerate(order)}})
            log.flush()
        print(f"  seed {seed} done ({(time.time()-t0)/60:.1f} min elapsed)", flush=True)
    log.close()
    # P(class|eta)
    ntot = len(SEEDS) * L_BG
    curve = {"N": N, "T": T, "eta_grid": ETA_GRID, "n_per_eta": ntot, "seeds": SEEDS, "L_bg": L_BG,
             "P": {c: [counts[e].get(c, 0) / ntot for e in ETA_GRID] for c in CLASSES}, "counts": {str(e): counts[e] for e in ETA_GRID}}
    json.dump(curve, open(os.path.join(outdir, "refine_curve.json"), "w"), indent=2)
    print(f"\n=== P(class | eta)  ({ntot} samples/eta) ===")
    print("  eta     SUSTAIN  SPIN_DN  COLLAPSE  BLOWUP")
    for e in ETA_GRID:
        c = counts[e]
        print(f"  {e:+.3f}   {c.get('SUSTAIN',0)/ntot:5.2f}    {c.get('SPIN_DOWN',0)/ntot:5.2f}    "
              f"{c.get('COLLAPSE',0)/ntot:5.2f}     {c.get('BLOWUP',0)/ntot:5.2f}")
    best = max(ETA_GRID, key=lambda e: counts[e].get("SUSTAIN", 0))
    print(f"\npeak P(SUSTAIN) at eta={best:+.3f} = {counts[best].get('SUSTAIN',0)/ntot:.2f}")
    print(f"wrote {outdir}/refine_curve.json + all_evals.csv  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
