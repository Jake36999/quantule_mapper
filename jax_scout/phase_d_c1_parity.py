"""Phase D / C1 parity gate: prove param_D_imag=0 preserves the frozen Phase C baseline BYTE-FOR-BYTE, and that
D_imag>0 actually opens the dispersive channel. Runs on the jax_scout mirror (WSL jax). No production/CuPy change.

  wsl:  source ~/jax_irer/bin/activate && python jax_scout/phase_d_c1_parity.py
"""
import os, sys
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jax_scout import physics, core_saturation_search as css

params = dict(css.FEB); params["param_a"] = float(css.FEB["param_a"]) * 1.15   # a*
N, L, dt = 48, css.L_, css.DT


def md(a, b):
    return float(jnp.max(jnp.abs(jnp.asarray(a) - jnp.asarray(b))))


ops_base = physics.build_operators(N, L, dt, params)                              # no param_D_imag key
ops_zero = physics.build_operators(N, L, dt, {**params, "param_D_imag": 0.0})     # explicit 0.0
ops_disp = physics.build_operators(N, L, dt, {**params, "param_D_imag": 0.05})    # dispersive

print("=== Phase D C1 parity gate (N=48, a*) ===")
print("[D_imag=0 vs baseline] byte-identity of the ETDRK4 operators (expect 0.0):")
for name in ("L_k", "E", "E2", "Q", "f1", "f2", "f3"):
    print(f"   max|Δ {name}| = {md(getattr(ops_base, name), getattr(ops_zero, name)):.3e}")

print("\n[D_imag=0.05] the dispersive channel is actually present:")
print(f"   max|Δ L_k vs baseline| = {md(ops_base.L_k, ops_disp.L_k):.3e}  (expect > 0)")
print(f"   L_k imag range base   = [{float(jnp.min(jnp.imag(ops_base.L_k))):.3f}, {float(jnp.max(jnp.imag(ops_base.L_k))):.3f}]")
print(f"   L_k imag range D=0.05 = [{float(jnp.min(jnp.imag(ops_disp.L_k))):.3f}, {float(jnp.max(jnp.imag(ops_disp.L_k))):.3f}]  (i*(-D_imag*k^2))")
print(f"   L_k REAL parts identical (dispersion is imaginary-only): max|Δ Re L_k| = {md(jnp.real(ops_base.L_k), jnp.real(ops_disp.L_k)):.3e}")

print("\n[50-step evolution] D_imag=0 reproduces the baseline trajectory (expect 0.0):")
psi0, _ = css.build_ic(N, 6, css.SEED)
pk_b = physics.initial_psi_k(jnp.asarray(psi0), ops_base)
pk_0 = physics.initial_psi_k(jnp.asarray(psi0), ops_zero)
pk_d = physics.initial_psi_k(jnp.asarray(psi0), ops_disp)
for _ in range(50):
    pk_b = physics.step(pk_b, ops_base)
    pk_0 = physics.step(pk_0, ops_zero)
    pk_d = physics.step(pk_d, ops_disp)
print(f"   max|Δ psi_k(50), D_imag=0 vs baseline| = {md(pk_b, pk_0):.3e}")
print(f"   max|Δ psi_k(50), D_imag=0.05 vs baseline| = {md(pk_b, pk_d):.3e}  (expect > 0: dispersion changes evolution)")

ok = md(ops_base.L_k, ops_zero.L_k) == 0.0 and md(pk_b, pk_0) == 0.0 and md(ops_base.L_k, ops_disp.L_k) > 0.0
print(f"\n=== {'C1_PARITY_PASS (D_imag=0 preserves Phase C byte-for-byte; D_imag>0 opens dispersion)' if ok else 'C1_PARITY_FAIL'} ===")
