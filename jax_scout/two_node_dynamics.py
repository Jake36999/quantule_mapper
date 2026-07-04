"""Phase D.5 — two-node dynamics: measure the node-node interaction law (READ-ONLY re the solver; pure Phase C
physics, D_imag=0). Places TWO a*-profile nodes at a controlled initial separation, evolves at the validated a*
params, and tracks separation d(t), node count, phase-difference, and total mass -> does the pair attract / merge /
hold / recede across the ~0.5-box coupling radius found in D.4? No solver/gate/physics change (D_imag=0 = frozen
baseline); this is an IC + measurement experiment.

  wsl:  python jax_scout/two_node_dynamics.py [--spacings 0.2,0.35,0.5,0.7 --T 12000 --out DIR]
"""
import os, sys, csv, json, time, argparse
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import core_saturation_search as css, physics
from jax_scout import transfer_diag as td
from jax_scout.phase_d_c1_transport import _evolve_chunk    # jitted lax.scan of physics.step

N = 96; L = css.L_; DT = css.DT; DX = L / N
A_FACTOR = 1.15
W = L / 12.0                                                  # multiseed_ic blob width


def two_node_ic(d0_box, seed=20260704):
    """Two unit-amplitude L/12 Gaussians separated by d0_box (box units) along x, + small noise."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    s = d0_box * L
    psi = (np.exp(-((X + s / 2) ** 2 + Y ** 2 + Z ** 2) / (2 * W ** 2))
           + np.exp(-((X - s / 2) ** 2 + Y ** 2 + Z ** 2) / (2 * W ** 2)))
    psi = psi + 0.01 * (rng.standard_normal((N, N, N)) + 1j * rng.standard_normal((N, N, N)))
    return psi.astype(np.complex128)


def _sep_phase(psi):
    """Track the two biggest nodes: (n_nodes, separation_box, |Δφ|, mass_ratio_of_two)."""
    nodes = td.detect_nodes(psi, DX)
    n = len(nodes)
    if n == 0:
        return n, np.nan, np.nan, np.nan
    nodes = sorted(nodes, key=lambda nd: -nd["M"])
    if n == 1:
        return n, 0.0, 0.0, float(nodes[0]["M"])
    ci = np.asarray(nodes[0]["centroid"], float); cj = np.asarray(nodes[1]["centroid"], float)
    d = ci - cj; d = d - N * np.round(d / N)
    sep = float(np.linalg.norm(d)) / N
    dphi = float(np.abs(np.angle(np.exp(1j * (nodes[0]["phase"] - nodes[1]["phase"])))))
    return n, sep, dphi, float(nodes[0]["M"] + nodes[1]["M"])


def run_one(d0, params, T, dt_chunk):
    ops = physics.build_operators(N, L, DT, params)
    psi0 = two_node_ic(d0)
    M0 = float(np.sum(np.abs(psi0) ** 2))
    psi_k = physics.initial_psi_k(jnp.asarray(psi0), ops)
    traj = {"t": [], "n": [], "sep": [], "dphi": [], "mass_ratio": []}
    n0, sep0, _, _ = _sep_phase(psi0)
    for c in range(T // dt_chunk):
        psi_k = _evolve_chunk(psi_k, ops, dt_chunk)
        cur = np.asarray(jnp.fft.ifftn(psi_k))
        n, sep, dphi, m2 = _sep_phase(cur)
        traj["t"].append((c + 1) * dt_chunk * DT); traj["n"].append(n)
        traj["sep"].append(sep); traj["dphi"].append(dphi)
        traj["mass_ratio"].append(float(np.sum(np.abs(cur) ** 2)) / M0)
        if not np.isfinite(cur).all():
            break
    seps = np.array([s for s in traj["sep"] if np.isfinite(s)])
    ns = np.array(traj["n"])
    merged = bool((ns[len(ns) // 2:] <= 1).any()) if len(ns) else False
    sep_fin = float(traj["sep"][-1]) if traj["sep"] else np.nan
    dsep = sep_fin - sep0 if np.isfinite(sep_fin) else np.nan
    if merged:
        verdict = "MERGE"
    elif np.isfinite(dsep) and dsep < -0.03:
        verdict = "APPROACH"
    elif np.isfinite(dsep) and dsep > 0.03:
        verdict = "RECEDE"
    else:
        verdict = "HOLD"
    return {"d0": d0, "sep0": sep0, "sep_fin": sep_fin, "dsep": dsep, "n_fin": int(ns[-1]) if len(ns) else 0,
            "merged": merged, "dphi_fin": float(traj["dphi"][-1]) if traj["dphi"] else np.nan,
            "mass_ratio_fin": traj["mass_ratio"][-1] if traj["mass_ratio"] else np.nan,
            "verdict": verdict, "traj": traj}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacings", default="0.2,0.3,0.4,0.5,0.7")
    ap.add_argument("--T", type=int, default=12000); ap.add_argument("--dtchunk", type=int, default=1000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    spac = [float(x) for x in a.spacings.split(",")]
    out = a.out or os.path.join(ROOT, "sweep_runs", f"PHASE_D_TWONODE_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    params = dict(css.FEB); params["param_a"] = float(css.FEB["param_a"]) * A_FACTOR
    print(f"=== PHASE D.5 TWO-NODE DYNAMICS (a*×{A_FACTOR}, D_imag=0) | spacings={spac} T={a.T} N={N} | out={out} ===", flush=True)
    rows = []
    for d0 in spac:
        t0 = time.time()
        r = run_one(d0, params, a.T, a.dtchunk)
        r["min"] = round((time.time() - t0) / 60, 1); traj = r.pop("traj")
        json.dump(traj, open(os.path.join(out, f"traj_d{d0}.json"), "w"), default=float)
        rows.append(r)
        print(f"  d0={d0:.2f} sep {r['sep0']:.3f}->{r['sep_fin'] if np.isfinite(r['sep_fin']) else float('nan'):.3f} "
              f"(dsep={r['dsep']:+.3f}) n_fin={r['n_fin']} merged={r['merged']} dphi={r['dphi_fin']:.3f} "
              f"mass={r['mass_ratio_fin']:.2f} -> {r['verdict']} ({r['min']}m)", flush=True)
        with open(os.path.join(out, "two_node_results.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=[k for k in rows[0] if k != "traj"], extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
    verdicts = {r["d0"]: r["verdict"] for r in rows}
    print(f"\n=== interaction law (d0 -> verdict): {verdicts} ===", flush=True)
    json.dump(rows, open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
