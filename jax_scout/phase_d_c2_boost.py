"""Phase D / C2 — conservative Galilean-boost mobility test (mirror). Does a phase ramp exp(i k x) translate a node
in the CONSERVATIVE substrate (v proportional to k), where the dissipative substrate dissipated it (C1/kick null)?
dt small (Schrodinger CFL). No solver default/gate change.

  wsl:  python jax_scout/phase_d_c2_boost.py [--dt 0.001 --kicks 0,2 --chunks 6 --chunk 700]
"""
import os, sys, argparse
import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import core_saturation_search as css, physics, transfer_diag as td
from jax_scout.phase_d_c1_transport import _evolve_chunk
from jax_scout.feb_kick_inertia import circ_angle, velocity, _X

N, L, DX = 96, css.L_, css.L_ / 96
ASTATE = os.path.join(ROOT, "sweep_runs", "FEB_GAIN_LADDER_LONGT_T72000_20260701_175708", "a1.15_ladder_T72000_probe.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=0.001); ap.add_argument("--kicks", default="0,2")
    ap.add_argument("--chunks", type=int, default=6); ap.add_argument("--chunk", type=int, default=700)
    a = ap.parse_args()
    kicks = [int(x) for x in a.kicks.split(",")]
    params = dict(css.FEB); params["param_a"] = float(css.FEB["param_a"]) * 1.15
    opsC = physics.build_operators(N, L, a.dt, {**params, "kinetic_mode": "conservative"})
    psi_a = np.load(ASTATE)["psi_fin"].astype(np.complex128)
    print(f"=== C2 conservative boost (dt={a.dt}, kicks={kicks}, chunks={a.chunks}x{a.chunk}) ===", flush=True)
    rows = []
    for n in kicks:
        k = 2 * np.pi * n / L
        psi = (psi_a * np.exp(1j * k * _X[0])).astype(np.complex128)
        M0 = float(np.sum(np.abs(psi) ** 2))
        pk = physics.initial_psi_k(jnp.asarray(psi), opsC); ang = []; tt = []
        cur = psi
        for c in range(a.chunks):
            pk = _evolve_chunk(pk, opsC, a.chunk); cur = np.asarray(jnp.fft.ifftn(pk))
            if not np.isfinite(cur).all():
                print(f"  n={n}: NON-FINITE @ {(c+1)*a.chunk}", flush=True); break
            ang.append(circ_angle(np.abs(cur) ** 2, 0)); tt.append((c + 1) * a.chunk * a.dt)
        if len(tt) >= 3:
            v, r2, pos = velocity(ang, tt); disp = (pos[-1] - pos[0]) / L
            nn = len(td.detect_nodes(cur, DX)); mr = float(np.sum(np.abs(cur) ** 2)) / M0
            rows.append((n, k, v, disp, r2, nn, mr))
            print(f"  n={n} k={k:.3f} -> v_x={v:+.4f} (v/k={v/k if k else 0:+.4f}) disp={disp:+.4f}box "
                  f"r2={r2:.2f} n_nodes={nn} mass={mr:.2f}", flush=True)
    if len(rows) >= 2:
        kk = np.array([r[1] for r in rows]); vv = np.array([r[2] for r in rows])
        mu = float(np.polyfit(kk, vv, 1)[0]) if kk.std() > 0 else float("nan")
        print(f"=== mobility mu=dv/dk = {mu:+.4f} (>>0 = ballistic transport; ~0 = still pinned) ===", flush=True)
    print("C2_BOOST_DONE", flush=True)


if __name__ == "__main__":
    main()
