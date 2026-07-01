"""
Phase-C validation of corrected-hunt candidates at full fidelity (N=48/800).
Tests whether the 'support' signal is GENUINE bounded mutual support or a collective-gain /
energy-runaway artifact (the second loophole: energy can run away ~85x while amp & curvature
stay below their gates). Genuine = intact stays ENERGY-BOUNDED + clean few nodes, isolated
markedly weaker, ablation bounded-disrupts, geometry bounded.

WSL2 jax venv: python /mnt/f/quantule_mapper/jax_scout/validate_candidates.py
"""
import os, sys, csv, glob
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import physics, geometry_diag as gd, stable_collapse as scl

SEED, K = 20260619, 6
N, L, dt, STEPS = 48, 10.0, 0.005, 800
order = physics.SWEEP_PARAM_ORDER


def ics(N, L):
    rng = np.random.default_rng(SEED)
    x = np.linspace(-L/2, L/2, N, endpoint=False); X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    w = L/12.0; bumps, ph = [], []
    for _ in range(K):
        cx, cy, cz = rng.uniform(-L/2, L/2, 3)
        bumps.append(np.exp(-((X-cx)**2+(Y-cy)**2+(Z-cz)**2)/(2*w**2))); ph.append(rng.uniform(0, 2*np.pi))
    noise = 0.01*(rng.standard_normal((N,N,N))+1j*rng.standard_normal((N,N,N)))
    out = {}
    for kind, ks, phs in (("intact", range(6), [0]*6), ("ablation", range(5), [0]*6), ("isolated", range(1), [0]*6)):
        psi = np.zeros((N,N,N), np.complex128)
        for i in ks: psi += bumps[i]*np.exp(1j*phs[i])
        out[kind] = (psi+noise).astype(np.complex128)
    return out


def run(pvec, ic):
    pm, pf, en, am, fin = physics.probe_one(jnp.asarray(pvec), jnp.asarray(ic), N, L, dt, STEPS)
    pf = np.asarray(pf); en = np.asarray(en); am = np.asarray(am)
    e0 = float(en[0]) if en[0] > 0 else 1e-30
    return {"finite": bool(fin), "er": float(en[-1]/e0), "amp": float(am.max()),
            "nodes": scl.node_count(np.abs(pf)**2), "coh": scl.phase_coherence(pf), "pf": pf}


d = sorted(glob.glob(os.path.join(ROOT, "sweep_runs", "ADAPTIVE_HUNT_2026062*")))[-1]
rows = list(csv.DictReader(open(os.path.join(d, "all_evals.csv"))))
for r in rows:
    for k in ("iso_surv", "bounded_abl_sens", "core", "curv", "intact_nodes", "er"):
        try: r[k] = float(r[k])
        except: r[k] = np.nan
cand = [r for r in rows if r["support_legs"] == "True"]
cand += sorted([r for r in rows if r["reject"] == "" and r["iso_surv"] < 0.2
                and 2 <= r["intact_nodes"] <= 20 and r["curv"] < 1.0 and r["bounded_abl_sens"] > 0.2],
               key=lambda r: r["iso_surv"])[:4]
seen = set(); cand = [c for c in cand if not (id(c) in seen or seen.add(id(c)))]
IC = ics(N, L)
print(f"Phase-C @N={N}/{STEPS} on {len(cand)} candidates from {os.path.basename(d)}\n")
for r in cand:
    pvec = [float(r[k]) for k in order]; par = {k: float(r[k]) for k in order}
    I = run(pvec, IC["intact"]); S = run(pvec, IC["isolated"]); B = run(pvec, IC["ablation"])
    curv = gd.curvature_max_only(I["pf"], par, L/N) if I["finite"] else float("inf")
    iso_ratio = S["er"]/max(I["er"], 1e-9)
    energy_bounded = I["finite"] and 0.2 <= I["er"] <= 5.0 and I["amp"] < 1e3
    clean_nodes = 2 <= I["nodes"] <= 20
    abl_ok = B["finite"] and B["amp"] < 1e3 and (B["nodes"] < I["nodes"]-1 or B["er"] < 0.7*I["er"])
    genuine = energy_bounded and clean_nodes and iso_ratio < 0.5 and abl_ok and curv < 1.0
    verdict = "GENUINE_BOUNDED_SUPPORT" if genuine else (
        "ENERGY_RUNAWAY" if (I["finite"] and I["er"] > 5) else
        "FRAGMENTED/NODES" if I["nodes"] > 20 else
        "UNSTABLE" if not I["finite"] else "WEAK/ambiguous")
    print(f"scout(iso_surv={r['iso_surv']:.3f} bnd_abl={r['bounded_abl_sens']:.2f} n={int(r['intact_nodes'])} "
          f"er={r['er']:.1f} legs={r['support_legs']}) D={par['param_D']:.2f} s={par['param_s']:.2f}")
    print(f"  @48/800 intact(fin={I['finite']},n={I['nodes']},er={I['er']:.2f},amp={I['amp']:.1f},curv={curv:.2f}) "
          f"iso(er={S['er']:.2f}->ratio={iso_ratio:.3f}) abl(fin={B['finite']},n={B['nodes']},er={B['er']:.2f})")
    print(f"  -> {verdict}\n")
