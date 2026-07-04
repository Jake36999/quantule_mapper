"""Phase D / C1 transport probe: does the dispersive channel (param_D_imag>0) give the site-pinned a* attractor a
MOBILITY the real-diffusive baseline structurally lacked? Re-runs the Phase-C kick test on the build_operators path
(which threads param_D_imag) over a small (D_imag, kick) grid. D_imag=0 must reproduce the Phase C null (v~0);
D_imag>0 is the test. jax_scout mirror only (WSL jax); NO production/CuPy change.

  wsl:  python jax_scout/phase_d_c1_transport.py [--tkick 8000 --out DIR]
"""
import os, sys, csv, json, time, argparse
from functools import partial
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import core_saturation_search as css, physics
from jax_scout import transfer_diag as td
from jax_scout.feb_kick_inertia import circ_angle, velocity, _X, DEFAULT_STATE, N, L, DT, DX

A_FACTOR = 1.15
# coherent range only: the stability diagnostic (no kick, 2000 steps) showed D_imag>=0.005 grows mass + fragments
# a* (D=0.005 -> mass 1.29/nodes 7; D=0.01 -> nodes 388; D=0.02 -> mass 9.6). a* stays coherent only for D<=~0.002.
D_IMAGS = [0.0, 0.001, 0.002]
KICKS = [0, 1, 2]
COLS = ["D_imag", "kick_n", "k", "v_x", "mobility_v_over_k", "disp_x_boxes", "com_fit_r2",
        "mass_ratio_fin", "peak_ratio_fin", "n_start", "n_end", "min"]


@partial(jax.jit, static_argnames=("n_steps",))
def _evolve_chunk(psi_k, ops, n_steps):
    def body(pk, _):
        return physics.step(pk, ops), None
    out, _ = jax.lax.scan(body, psi_k, None, length=n_steps)
    return out


def evolve_kick_c1(psi_settle, params, n_kick, t_kick, dt_chunk, D_imag):
    ops = physics.build_operators(N, L, DT, {**params, "param_D_imag": float(D_imag)})
    k = 2 * np.pi * n_kick / L
    psi = (np.asarray(psi_settle) * np.exp(1j * k * _X[0])).astype(np.complex128)
    M0 = float(np.sum(np.abs(psi) ** 2)); p0 = float(np.max(np.abs(psi) ** 2))
    n_start = len(td.detect_nodes(psi, DX))
    psi_k = physics.initial_psi_k(jnp.asarray(psi), ops)
    ang = {0: [], 1: [], 2: []}; t_phys = []; mass_r = []; peak_r = []
    cur = psi
    for c in range(t_kick // dt_chunk):
        psi_k = _evolve_chunk(psi_k, ops, dt_chunk)
        cur = np.asarray(jnp.fft.ifftn(psi_k)); rho = np.abs(cur) ** 2
        for ax in (0, 1, 2):
            ang[ax].append(circ_angle(rho, ax))
        t_phys.append((c + 1) * dt_chunk * DT)
        mass_r.append(float(np.sum(rho)) / M0); peak_r.append(float(np.max(rho)) / p0)
        if not np.isfinite(cur).all():
            break
    n_end = len(td.detect_nodes(cur, DX))
    vx, r2x, posx = velocity(ang[0], t_phys)
    return {"D_imag": D_imag, "kick_n": n_kick, "k": k, "v_x": vx,
            "mobility_v_over_k": (vx / k) if k != 0 else 0.0,
            "disp_x_boxes": float(posx[-1] - posx[0]) / L, "com_fit_r2": r2x,
            "mass_ratio_fin": mass_r[-1], "peak_ratio_fin": peak_r[-1], "n_start": n_start, "n_end": n_end}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=DEFAULT_STATE)
    ap.add_argument("--tkick", type=int, default=8000); ap.add_argument("--dtchunk", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(ROOT, "sweep_runs", f"PHASE_D_C1_TRANSPORT_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    csv_path = os.path.join(out, "c1_transport_results.csv")
    psi_settle = np.load(args.state)["psi_fin"].astype(np.complex128)
    params = dict(css.FEB); params["param_a"] = float(css.FEB["param_a"]) * A_FACTOR
    print(f"=== PHASE D C1 TRANSPORT PROBE (a*×{A_FACTOR}) | D_imag={D_IMAGS} kicks={KICKS} tkick={args.tkick} "
          f"N={N} | out={out} ===", flush=True)
    print(f"    settled a* mass={float(np.sum(np.abs(psi_settle)**2)):.1f} nodes={len(td.detect_nodes(psi_settle,DX))}", flush=True)
    rows = []
    for D_imag in D_IMAGS:
        for n in KICKS:
            t0 = time.time()
            r = evolve_kick_c1(psi_settle, params, n, args.tkick, args.dtchunk, D_imag)
            r["min"] = round((time.time() - t0) / 60, 1)
            rows.append({c: r.get(c) for c in COLS})
            print(f"  D_imag={D_imag} n={n} k={r['k']:.3f} -> v_x={r['v_x']:+.5f} (v/k={r['mobility_v_over_k']:+.5f}) "
                  f"disp={r['disp_x_boxes']:+.3f}box r2={r['com_fit_r2']:.2f} mass={r['mass_ratio_fin']:.3f} "
                  f"peak={r['peak_ratio_fin']:.3f} nodes {r['n_start']}->{r['n_end']} ({r['min']}m)", flush=True)
            with open(csv_path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

    # mobility mu = slope of v_x vs k at each D_imag; coherence = mass & peak preserved
    summary = {"D_imags": D_IMAGS, "kicks": KICKS, "tkick": args.tkick, "by_D_imag": {}}
    for D_imag in D_IMAGS:
        sub = [r for r in rows if r["D_imag"] == D_imag]
        kk = np.array([r["k"] for r in sub]); vv = np.array([r["v_x"] for r in sub])
        mu = float(np.polyfit(kk, vv, 1)[0]) if len(kk) >= 2 else float("nan")
        coh = all(0.7 <= r["mass_ratio_fin"] <= 1.3 and r["n_end"] >= 1 for r in sub)
        summary["by_D_imag"][D_imag] = {"mobility_mu": mu, "coherent": coh,
                                        "max_disp_box": max(abs(r["disp_x_boxes"]) for r in sub)}
        print(f"  [mu] D_imag={D_imag}: mobility mu=dv/dk={mu:+.5f}  coherent={coh}  "
              f"max|disp|={summary['by_D_imag'][D_imag]['max_disp_box']:.3f}box", flush=True)
    mu0 = abs(summary["by_D_imag"][D_IMAGS[0]]["mobility_mu"])
    mu_hi = abs(summary["by_D_imag"][D_IMAGS[-1]]["mobility_mu"])
    coh_hi = summary["by_D_imag"][D_IMAGS[-1]]["coherent"]
    if mu_hi > 5 * max(mu0, 1e-4) and coh_hi:
        verdict = "C1_TRANSPORT_DETECTED"      # dispersion gives coherent mobility the baseline lacked
    elif mu_hi > 5 * max(mu0, 1e-4):
        verdict = "C1_MOBILITY_BUT_INCOHERENT"  # moves but a* structure not preserved
    else:
        verdict = "C1_STILL_PINNED"            # dispersion did NOT confer mobility (site-pinning deeper than kinetic)
    summary["verdict"] = verdict
    summary["mu_baseline"] = mu0; summary["mu_dispersive"] = mu_hi
    json.dump(summary, open(os.path.join(out, "c1_transport_summary.json"), "w"), indent=2, default=float)
    print(f"\n=== {verdict} === mu(D=0)={mu0:.5f} vs mu(D={D_IMAGS[-1]})={mu_hi:.5f}, coherent={coh_hi}", flush=True)


if __name__ == "__main__":
    main()
