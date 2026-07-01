"""
Stable-collapse observable + classifier for CORRECTED_PHYSICS_JAX_SCOUT.

Implements the target search: "find bounded, persistent, mutually-supporting node/mode
configurations that avoid both dissipative collapse and blow-up." Uses independent
first-principles observables (NO prime_log_sse, NO k^2 weighting). Classifies each
trajectory into:
    unstable        - blow-up / NaN / clipping / runaway saturation
    dissipative     - loses structure, relaxes toward smooth low-energy vacuum
    single_node     - bounded, energy retained, ONE concentrated node (trivial condensate)
    transient       - multi-structure forms mid-run then vanishes / merges by the end
    stable_multinode- bounded, energy retained, PERSISTENT multi-node/mode structure (TARGET)

Auxiliary differentiators recorded (not headline): mode incommensurability (distance of
dominant mode-ratios from low-order rationals), phase coherence, node survival, prime-log
left for a separate auxiliary pass.

Run in WSL2 jax venv:
  python /mnt/f/quantule_mapper/jax_scout/stable_collapse.py --size 256 --batch 16 --N 48 --steps 800
"""
import os
import sys
import csv
import json
import time
import argparse
from collections import Counter
import numpy as np
import scipy.ndimage as ndi
import scipy.signal

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import physics

LABEL, EVIDENCE, SEED = "CORRECTED_PHYSICS_JAX_SCOUT", "NOT_CUPY_FINAL_EVIDENCE", 20260619

# --- classifier thresholds (first-pass, tunable) ---
AMP_BLOWUP = 1e3      # max|psi| above this = unstable (clipping/blow-up)
DISSIPATE_ER = 0.5    # final/initial energy below this AND smooth = dissipative
NODE_SIGMA = 3.0      # density node = connected region above mean + NODE_SIGMA*std
CONC_SMOOTH = 4.0     # max(rho)/mean(rho) below this = "smooth" (no real concentration)
COH_FLOOR = 0.05      # phase coherence above this = organized (random-phase floor ~1/sqrt(Nvox))


def node_count(rho):
    thr = rho.mean() + NODE_SIGMA * rho.std()
    lbl, n = ndi.label(rho > thr)
    if n == 0:
        return 0
    sizes = np.bincount(lbl.ravel())[1:]
    return int(np.sum(sizes >= 3))  # ignore single-voxel specks


def spectral_modes(rho):
    """Independent radial power profile (no k^2 weighting) + relative-prominence peaks."""
    power = np.abs(np.fft.fftn(rho - rho.mean())) ** 2
    N = rho.shape[0]
    kx = np.fft.fftfreq(N) * N
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
    kr = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2).astype(int)
    prof = np.bincount(kr.ravel(), weights=power.ravel()) / np.maximum(np.bincount(kr.ravel()), 1)
    prof = prof[1:N // 2]
    if prof.size < 4 or prof.max() <= 0:
        return 0, []
    sm = ndi.gaussian_filter1d(prof, 1.0)
    rng = sm.max() - np.median(sm)
    pk, _ = scipy.signal.find_peaks(sm, prominence=0.05 * rng, height=np.median(sm) + 0.1 * rng, distance=2)
    return len(pk), [int(p) + 1 for p in pk]  # +1 for the [1:] offset


def incommensurability(ks):
    """Min distance of pairwise mode ratios from a low-order rational p/q (q<=8). High=incommensurate."""
    ks = sorted([k for k in ks if k > 0])
    if len(ks) < 2:
        return None
    dists = []
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            r = ks[j] / ks[i]
            best = min(abs(r - p / q) for q in range(1, 9) for p in range(1, int(r * q) + 2))
            dists.append(best)
    return float(np.mean(dists))


def phase_coherence(psi):
    a = np.abs(psi)
    s = a.sum()
    return float(np.abs(psi.sum()) / s) if s > 0 else 0.0


def classify(obs):
    if not obs["finite"] or obs["amp_max"] > AMP_BLOWUP:
        return "unstable"
    smooth = obs["conc_final"] < CONC_SMOOTH
    if smooth and obs["energy_ratio"] < DISSIPATE_ER:
        return "dissipative"
    nf, nm, mf = obs["nodes_final"], obs["nodes_mid"], obs["modes_final"]
    # ORGANIZATION gate (mutual support vs random speckle) is COHERENCE, not periodicity.
    # Random NOISE -> coherence ~ 1/sqrt(Nvox) (~0.007); a coupled/coherent multi-node
    # configuration stays phase-coherent (~0.9). A periodic lattice (modes>=1) is a
    # SPECIAL CASE, not a requirement -- aperiodic coherent clusters still count.
    coherent = obs["coherence"] > COH_FLOOR
    if nf >= 2 and nm >= 2:
        return "stable_multinode" if coherent else "incoherent_multinode"
    if nm >= 2 and nf < 2:
        return "transient"
    if smooth:
        return "dissipative"
    return "single_node"


def observe(psi_mid, psi_final, energy, max_amp, finite):
    rho_m = np.abs(psi_mid) ** 2
    rho_f = np.abs(psi_final) ** 2
    e0 = float(energy[0]) if energy[0] > 0 else 1e-30
    modes_final, peak_ks = spectral_modes(rho_f)
    obs = {
        "finite": bool(finite), "amp_max": float(np.max(max_amp)), "amp_final": float(max_amp[-1]),
        "energy_ratio": float(energy[-1] / e0), "energy_growth": float(np.max(energy) / e0),
        "conc_final": float(rho_f.max() / max(rho_f.mean(), 1e-30)),
        "nodes_mid": node_count(rho_m), "nodes_final": node_count(rho_f),
        "modes_final": modes_final, "peak_ks": peak_ks,
        "incommens": incommensurability(peak_ks), "coherence": phase_coherence(psi_final),
    }
    obs["class"] = classify(obs)
    return obs


def make_ic(kind, N, L, seed):
    """Shared initial condition (same across all configs in a sweep -> differences are
    purely parametric). 'gaussian' single seed; 'noise' broadband; 'multiseed' K bumps."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    if kind == "gaussian":
        psi0 = np.exp(-(X ** 2 + Y ** 2 + Z ** 2) / 2.0).astype(np.complex128)
        psi0 += 0.01 * (rng.standard_normal(psi0.shape) + 1j * rng.standard_normal(psi0.shape))
    elif kind == "noise":
        psi0 = (0.3 * (rng.standard_normal((N, N, N)) + 1j * rng.standard_normal((N, N, N)))).astype(np.complex128)
    elif kind == "multiseed":
        K, w = 6, L / 12.0
        psi0 = np.zeros((N, N, N), np.complex128)
        for _ in range(K):
            cx, cy, cz = rng.uniform(-L / 2, L / 2, 3)
            psi0 += np.exp(-((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2) / (2 * w ** 2))
        psi0 += 0.01 * (rng.standard_normal(psi0.shape) + 1j * rng.standard_normal(psi0.shape))
    else:
        raise ValueError(f"unknown ic kind: {kind}")
    return psi0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ic", choices=["gaussian", "noise", "multiseed"], default="gaussian")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--N", type=int, default=48)
    ap.add_argument("--L", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--bounds-file", default="/mnt/f/quantule_mapper/jax_scout/gain_bounds.json")
    ap.add_argument("--outdir", default="/mnt/f/quantule_mapper/sweep_runs")
    args = ap.parse_args()

    bounds = json.load(open(args.bounds_file))
    order = physics.SWEEP_PARAM_ORDER
    lo = np.array([bounds[k][0] for k in order]); hi = np.array([bounds[k][1] for k in order])
    from scipy.stats import qmc
    unit = qmc.LatinHypercube(d=len(order), seed=SEED).random(args.size)
    params = lo + unit * (hi - lo)

    psi0 = make_ic(args.ic, args.N, args.L, SEED)
    psi0_j = jnp.asarray(psi0)

    outdir = os.path.join(args.outdir, f"STABLE_COLLAPSE_{args.ic}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(outdir, exist_ok=True)
    print(f"{LABEL} / {EVIDENCE}  STABLE-COLLAPSE classification  IC={args.ic}")
    print(f"N={args.N}^3 dt={args.dt} steps={args.steps} size={args.size} bounds={os.path.basename(args.bounds_file)}\n")

    rows, t0 = [], time.time()
    for b0 in range(0, args.size, args.batch):
        pb = params[b0:b0 + args.batch]
        pm, pf, en, am, fin = physics.sweep_probe(jnp.asarray(pb), psi0_j, args.N, args.L,
                                                  args.dt, args.steps, jnp.float64, jnp.complex128)
        pm = np.asarray(pm); pf = np.asarray(pf); en = np.asarray(en); am = np.asarray(am); fin = np.asarray(fin)
        for j in range(pb.shape[0]):
            o = observe(pm[j], pf[j], en[j], am[j], fin[j])
            o["idx"] = b0 + j
            for i, k in enumerate(order):
                o[k] = float(pb[j][i])
            rows.append(o)
        print(f"  batch {b0//args.batch+1}/{-(-args.size//args.batch)} [{time.time()-t0:.0f}s]", flush=True)

    tally = Counter(r["class"] for r in rows)
    cols = ["idx", "class", *order, "energy_ratio", "energy_growth", "conc_final",
            "nodes_mid", "nodes_final", "modes_final", "incommens", "coherence", "amp_max", "peak_ks"]
    with open(os.path.join(outdir, "stable_collapse_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    json.dump({"label": LABEL, "ic": args.ic, "N": args.N, "dt": args.dt, "steps": args.steps,
               "size": args.size, "bounds": bounds, "param_order": order, "tally": dict(tally)},
              open(os.path.join(outdir, "meta.json"), "w"), indent=2)

    print(f"\nDONE {args.size} configs in {(time.time()-t0)/60:.1f} min")
    print("class tally:", dict(tally))
    cand = sorted([r for r in rows if r["class"] == "stable_multinode"],
                  key=lambda r: -(r["nodes_final"] + r["modes_final"]))
    print(f"\nstable_multinode candidates: {len(cand)}")
    for r in cand[:10]:
        print(f"  idx={r['idx']:4d} nodes={r['nodes_final']} modes={r['modes_final']} "
              f"e_ratio={r['energy_ratio']:.2f} incommens={r['incommens']} coh={r['coherence']:.3f} "
              f"peak_ks={r['peak_ks']}")
    print(f"\n-> {outdir}/stable_collapse_results.csv")


if __name__ == "__main__":
    main()
