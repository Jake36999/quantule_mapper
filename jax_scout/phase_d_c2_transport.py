"""Phase D / C2 conservative-substrate test (jax_scout mirror). Parity (dissipative default = frozen baseline
byte-for-byte) + conservative single-node behaviour (norm conservation / soliton / disperse / collapse) + conservative
two-node dynamics (motion vs the dissipative D.5 merge-or-hold baseline). No solver default/gate change.

  wsl:  python jax_scout/phase_d_c2_transport.py [--T 3000]
"""
import os, sys, argparse
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
from jax_scout.two_node_dynamics import two_node_ic, _sep_phase

N, L, DT, DX = 96, css.L_, css.DT, css.L_ / 96
ASTATE = os.path.join(ROOT, "sweep_runs", "FEB_GAIN_LADDER_LONGT_T72000_20260701_175708", "a1.15_ladder_T72000_probe.npz")


def md(a, b): return float(jnp.max(jnp.abs(jnp.asarray(a) - jnp.asarray(b))))


def track(psi_k, ops, T, dt_chunk, label, sep=False):
    psi0 = np.asarray(jnp.fft.ifftn(psi_k)); M0 = float(np.sum(np.abs(psi0) ** 2))
    print(f"  [{label}] evolving T={T} ...", flush=True)
    for c in range(T // dt_chunk):
        psi_k = _evolve_chunk(psi_k, ops, dt_chunk)
        cur = np.asarray(jnp.fft.ifftn(psi_k))
        if not np.isfinite(cur).all():
            print(f"    step~{(c+1)*dt_chunk}: NON-FINITE (collapse)", flush=True); return "COLLAPSE", None
        mr = float(np.sum(np.abs(cur) ** 2)) / M0; mx = float(np.max(np.abs(cur)))
        n, s, dphi, _ = _sep_phase(cur)
        print(f"    t={(c+1)*dt_chunk*DT:.1f}: mass={mr:.3f} max|psi|={mx:.3f} n_nodes={n}"
              + (f" sep={s:.3f}" if sep else ""), flush=True)
    return "OK", (mr, mx, n, s if sep else None)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--T", type=int, default=3000); ap.add_argument("--dtchunk", type=int, default=1000)
    a = ap.parse_args()
    params = dict(css.FEB); params["param_a"] = float(css.FEB["param_a"]) * 1.15

    print("=== C2 PARITY (dissipative default == frozen baseline) ===", flush=True)
    opsA = physics.build_operators(N, L, DT, params)                                   # default
    opsB = physics.build_operators(N, L, DT, {**params, "kinetic_mode": "dissipative"})
    opsC = physics.build_operators(N, L, DT, {**params, "kinetic_mode": "conservative"})
    for nm in ("L_k", "E", "E2", "Q", "f1", "f2", "f3"):
        print(f"   max|Δ {nm}| dissipative(default vs explicit) = {md(getattr(opsA, nm), getattr(opsB, nm)):.3e}", flush=True)
    print(f"   kfac: dissipative={complex(opsA.kfac)}  conservative={complex(opsC.kfac)}", flush=True)
    print(f"   conservative L_k: max|Re|={float(jnp.max(jnp.abs(jnp.real(opsC.L_k)))):.3e} (expect ~0, pure imaginary) "
          f"Im range=[{float(jnp.min(jnp.imag(opsC.L_k))):.1f},{float(jnp.max(jnp.imag(opsC.L_k))):.1f}]", flush=True)

    print("\n=== C2 CONSERVATIVE single-node (a* state) — norm conservation / stability ===", flush=True)
    psi = np.load(ASTATE)["psi_fin"].astype(np.complex128)
    pk = physics.initial_psi_k(jnp.asarray(psi), opsC)
    st, _ = track(pk, opsC, a.T, a.dtchunk, "conservative a*")
    print(f"   -> {st}", flush=True)

    if st == "OK":
        print("\n=== C2 CONSERVATIVE two-node (sep 0.4; held in the dissipative substrate) — motion? ===", flush=True)
        psi0 = two_node_ic(0.4)
        pk2 = physics.initial_psi_k(jnp.asarray(psi0), opsC)
        track(pk2, opsC, a.T, a.dtchunk, "conservative 2-node", sep=True)


if __name__ == "__main__":
    main()
