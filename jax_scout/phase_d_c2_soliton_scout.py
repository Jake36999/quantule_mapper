"""Phase D / C2.1 — native conservative-soliton SCOUT (jax_scout mirror, kinetic_mode='conservative', dt small).
Does the conservative/NLS substrate have its OWN stable localized structures (distinct from the dissipative a*)?
Scans single-Gaussian ICs over amplitude x width, evolves conservatively, classifies disperse/collapse/radiate/
fragment/localized. NOT Hunter; a filter grid. No solver default/gate change; no clipping. NOT reusing a*.

  wsl:  python jax_scout/phase_d_c2_soliton_scout.py [--N 48 --T 4000 --dt 0.001 --dtchunk 500 --out DIR]
"""
import os, sys, csv, json, time, argparse, itertools
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import core_saturation_search as css, physics
from jax_scout import transfer_diag as td
from jax_scout.phase_d_c1_transport import _evolve_chunk

L = css.L_
AMPS = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
SIGMAS = [0.04, 0.06, 0.083, 0.11, 0.15]     # width / L
COLS = ["A", "sigma", "klass", "mass_ret", "amp_ret", "occ_ratio", "n_fin", "amp_fin", "min"]


def gaussian_ic(A, sigma_box, N, seed=20260704):
    rng = np.random.default_rng(seed)
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    sig = sigma_box * L
    psi = A * np.exp(-(X ** 2 + Y ** 2 + Z ** 2) / (2 * sig ** 2))
    psi = psi + 0.005 * (rng.standard_normal((N, N, N)) + 1j * rng.standard_normal((N, N, N)))
    return psi.astype(np.complex128)


def occ(rho):
    """participation fraction (Sum rho)^2/(Sum rho^2)/Ncells; small = localized, ~1 = spread/uniform."""
    s1 = float(rho.sum()); s2 = float((rho ** 2).sum())
    return (s1 ** 2 / (s2 + 1e-30)) / rho.size


def evolve(A, sigma, N, dt, T, dt_chunk):
    dx = L / N
    ops = physics.build_operators(N, L, dt, {**css.FEB, "param_a": float(css.FEB["param_a"]) * 1.15,
                                             "kinetic_mode": "conservative"})
    psi0 = gaussian_ic(A, sigma, N)
    M0 = float(np.sum(np.abs(psi0) ** 2)); amp0 = float(np.max(np.abs(psi0))); occ0 = occ(np.abs(psi0) ** 2)
    psi_k = physics.initial_psi_k(jnp.asarray(psi0), ops); cur = psi0
    for c in range(T // dt_chunk):
        psi_k = _evolve_chunk(psi_k, ops, dt_chunk)
        cur = np.asarray(jnp.fft.ifftn(psi_k))
        if not np.isfinite(cur).all():
            return {"klass": "COLLAPSE", "mass_ret": np.nan, "amp_ret": np.nan, "occ_ratio": np.nan, "n_fin": 0, "amp_fin": np.nan}
    rho = np.abs(cur) ** 2
    mass_ret = float(rho.sum()) / M0; amp_ret = float(np.max(np.abs(cur))) / (amp0 + 1e-30)
    occ_ratio = occ(rho) / (occ0 + 1e-30); n = len(td.detect_nodes(cur, dx)); amp_fin = float(np.max(np.abs(cur)))
    if amp_ret > 5 or amp_fin > 50:
        k = "COLLAPSE"
    elif n > 3:
        k = "FRAGMENT"
    elif occ_ratio > 3.0 or amp_ret < 0.3:
        k = "DISPERSE"
    elif mass_ret < 0.5:
        k = "RADIATE"
    elif mass_ret >= 0.6 and 0.4 <= amp_ret <= 3.0 and occ_ratio < 2.0 and 1 <= n <= 3:
        k = "LOCALIZED"
    else:
        k = "MARGINAL"
    return {"klass": k, "mass_ret": mass_ret, "amp_ret": amp_ret, "occ_ratio": occ_ratio, "n_fin": n, "amp_fin": amp_fin}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=48); ap.add_argument("--T", type=int, default=4000)
    ap.add_argument("--dt", type=float, default=0.001); ap.add_argument("--dtchunk", type=int, default=500)
    ap.add_argument("--amps", default=None); ap.add_argument("--sigmas", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    amps = [float(x) for x in a.amps.split(",")] if a.amps else AMPS
    sigmas = [float(x) for x in a.sigmas.split(",")] if a.sigmas else SIGMAS
    out = a.out or os.path.join(ROOT, "sweep_runs", f"PHASE_D_C2_SCOUT_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True); csv_path = os.path.join(out, "c2_scout.csv")
    grid = list(itertools.product(amps, sigmas))
    print(f"=== C2.1 NATIVE-SOLITON SCOUT | N={a.N} T={a.T} dt={a.dt} | {len(grid)} cells | out={out} ===", flush=True)
    rows = []
    for (A, sig) in grid:
        t0 = time.time()
        try:
            r = evolve(A, sig, a.N, a.dt, a.T, a.dtchunk)
        except Exception as exc:
            r = {"klass": f"ERR:{str(exc)[:40]}"}
        r.update({"A": A, "sigma": sig, "min": round((time.time() - t0) / 60, 1)})
        rows.append(r)
        print(f"  A={A} sig={sig} -> {r.get('klass'):10s} mass_ret={r.get('mass_ret')} amp_ret={r.get('amp_ret')} "
              f"occ={r.get('occ_ratio')} n={r.get('n_fin')} ({r.get('min')}m)", flush=True)
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    loc = [r for r in rows if r.get("klass") == "LOCALIZED"]
    marg = [r for r in rows if r.get("klass") == "MARGINAL"]
    import collections
    hist = collections.Counter(r.get("klass") for r in rows)
    verdict = "C2_LOCALIZED_CANDIDATES_FOUND" if loc else ("C2_MARGINAL_ONLY" if marg else "C2_NATIVE_SOLITON_NOT_FOUND_IN_INITIAL_GRID")
    print(f"\n=== klass histogram: {dict(hist)} ===", flush=True)
    print(f"=== {verdict} | {len(loc)} LOCALIZED, {len(marg)} MARGINAL ===", flush=True)
    for r in sorted(loc + marg, key=lambda r: -(r.get('mass_ret') or 0))[:10]:
        print(f"    A={r['A']} sig={r['sigma']} {r['klass']} mass_ret={r['mass_ret']:.2f} amp_ret={r['amp_ret']:.2f} "
              f"occ={r['occ_ratio']:.2f} n={r['n_fin']}", flush=True)
    json.dump({"verdict": verdict, "hist": dict(hist), "localized": loc, "marginal": marg, "N": a.N, "T": a.T, "dt": a.dt},
              open(os.path.join(out, "summary.json"), "w"), indent=2, default=float)
    print(f"C2_SCOUT_DONE {out}", flush=True)


if __name__ == "__main__":
    main()
