"""Stepper order gates — every active time-stepper must converge at its designed order.

WHY THIS FILE EXISTS. In October 2026 two ETDRK4 bugs made the Phase C / C1 / C2 integrator converge
at about order 0.6 instead of 4 (docs/instrument_integrity/ETDRK4_INTEGRATOR_BUGS_2026-10.md). Both
passed every identity test, because identities check what the equations CONSERVE, not how ACCURATELY
they are solved. Both also passed CuPy<->JAX parity, because the two copies shared the bug. A dt-halving
test catches both in seconds. This file runs one for every stepper on an active research path.

Two kinds of gate:
  * ORDER: run the same IC at dt, dt/2, dt/4 against a fine reference; the error ratio gives the order.
  * MMS (method of manufactured solutions): pick an exact smooth psi*(t), add the source term
    S = d(psi*)/dt - L psi* - N(psi*) to the stepper's nonlinear term, and check the stepper reproduces
    psi* at the designed order. This needs no reference run, so it cannot share a bug with one.

DESIGN RULE (same as tests/test_physics_identities.py): exercise the harness code path and never
re-implement the stepper. MMS wraps only the nonlinear/RHS function, via monkeypatch, with a
stage-time counter; the stepper itself runs unmodified, eagerly (jax.disable_jit), so the patched
global is the one it calls.

Gated steppers (expected order):
  ETDRK4      jax_scout/physics.py:step                          (dissipative + conservative)  4
  KG Strang   jax_scout/phase_d_c3_wave.py:kg_evolve                                          2
  TG RK4      jax_scout/gravity_TG_B1S_state_load_feedback_gpu.py:rk4_step                    4
  TG-B2 RK4   jax_scout/gravity_TG_B2_two_node_awell.py:rk4_2n                                4
  Gravity-D   jax_scout/gravity_D_neutral_probe_gpu.py:rk4_step                               4
The CuPy ETDRK4 production solver is gated separately in tests/test_etdrk4_order.py.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

jax = pytest.importorskip("jax", reason="JAX not installed in this environment")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from jax_scout import physics  # noqa: E402
from jax_scout.phase_d_c3_wave import build_kg, kg_evolve, invariants  # noqa: E402
from jax_scout import gravity_TG_B1S_state_load_feedback_gpu as b1s  # noqa: E402
from jax_scout import gravity_TG_B2_two_node_awell as b2  # noqa: E402
from jax_scout import gravity_D_neutral_probe_gpu as gd  # noqa: E402


# ------------------------------------------------------------------ helpers

def observed_orders(run, dts, ref=None, ref_div=16):
    """log2 error ratios along a halving dt ladder. `ref` is the exact answer if known, else a fine run."""
    target = ref if ref is not None else run(dts[-1] / ref_div)
    errs = [float(jnp.max(jnp.abs(run(dt) - target))) for dt in dts]
    return [float(np.log2(errs[i] / errs[i + 1])) for i in range(len(errs) - 1)], errs


def _gauss(N, L, shift=0.0):
    x = jnp.linspace(-L / 2, L / 2, N, endpoint=False)
    X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
    return X, Y, Z, jnp.exp(-((X - shift) ** 2 + Y ** 2 + Z ** 2) / 2).astype(jnp.complex128)


FEB = {"param_D": 2.7329, "param_eta": 0.0704, "param_rho_vac": 1.1866, "param_a_coupling": 2.3098,
       "param_splash_coupling": 0.0129, "param_splash_fraction": -0.4861, "param_a": 0.5522}
CONSERVATIVE = {"param_D": 1.0, "param_a": 0.8, "param_s": -0.2, "kinetic_mode": "conservative",
                "param_geom_off": True}


# ------------------------------------------------------------------ ETDRK4 (physics.step)

def _etdrk4_run(params, N, L, T, psi0):
    def run(dt):
        ops = physics.build_operators(N, L, dt, params)
        pk = physics.initial_psi_k(psi0, ops)
        pk = jax.lax.fori_loop(0, int(round(T / dt)), lambda i, p: physics.step(p, ops), pk)
        return jnp.fft.ifftn(pk)
    return run


def test_etdrk4_dissipative_feb_is_fourth_order():
    """Full Phase C physics (geometry on, feb params at a*). Pre-fix this measured ~0.6."""
    *_, psi0 = _gauss(16, 10.0)
    orders, errs = observed_orders(_etdrk4_run(FEB, 16, 10.0, 0.5, psi0), [0.02, 0.01, 0.005])
    assert min(orders) > 3.7, (orders, errs)


def test_etdrk4_conservative_branch_is_fourth_order():
    """C2 NLS branch, L = -i D k^2: the coefficient bug was largest here (f1 off by up to 95%)."""
    *_, psi0 = _gauss(16, 10.0)
    orders, errs = observed_orders(_etdrk4_run(CONSERVATIVE, 16, 10.0, 2.0, psi0), [0.04, 0.02, 0.01])
    assert min(orders) > 3.7, (orders, errs)


def _mms_etdrk4(params, N, L, T, dts, monkeypatch):
    """Manufactured solution psi*(x,t) = (1 + 0.2 sin t) e^{i 0.7 t} (1 + 0.3 cos k x cos k y),
    band-limited (one Fourier mode), so dealiasing is exact and the only error is time-stepping."""
    X, Y, Z, _ = _gauss(N, L)
    k1 = 2 * jnp.pi / L
    shape = (1.0 + 0.3 * jnp.cos(k1 * X) * jnp.cos(k1 * Y)).astype(jnp.complex128)

    def amp(t):
        return (1 + 0.2 * np.sin(t)) * np.exp(0.7j * t)

    def damp(t):
        return (0.2 * np.cos(t) + 0.7j * (1 + 0.2 * np.sin(t))) * np.exp(0.7j * t)

    orig = physics.n_op
    results = []
    for dt in dts:
        ops = physics.build_operators(N, L, dt, params)
        stage = {"i": 0}
        c = (0.0, 0.5, 0.5, 1.0)

        def source_k(t):
            psi_k = jnp.fft.fftn(amp(t) * shape) * ops.dealias_mask
            dpsi_k = jnp.fft.fftn(damp(t) * shape) * ops.dealias_mask
            return dpsi_k - ops.L_k * psi_k - orig(psi_k, ops)

        def n_op_mms(psi_k, ops_, *a, **kw):
            i = stage["i"]; stage["i"] += 1
            t = (i // 4) * dt + c[i % 4] * dt
            return orig(psi_k, ops_, *a, **kw) + source_k(t) * ops_.dealias_mask

        monkeypatch.setattr(physics, "n_op", n_op_mms)
        pk = jnp.fft.fftn(amp(0.0) * shape) * ops.dealias_mask
        with jax.disable_jit():
            for _ in range(int(round(T / dt))):
                pk = physics.step(pk, ops)
        monkeypatch.setattr(physics, "n_op", orig)
        results.append(float(jnp.max(jnp.abs(jnp.fft.ifftn(pk) - amp(T) * shape))))
    return [float(np.log2(results[i] / results[i + 1])) for i in range(len(results) - 1)], results


def test_etdrk4_mms_dissipative(monkeypatch):
    orders, errs = _mms_etdrk4(FEB, 8, 10.0, 0.4, [0.1, 0.05, 0.025], monkeypatch)
    assert min(orders) > 3.7, (orders, errs)


def test_etdrk4_mms_conservative(monkeypatch):
    orders, errs = _mms_etdrk4(CONSERVATIVE, 8, 10.0, 0.4, [0.1, 0.05, 0.025], monkeypatch)
    assert min(orders) > 3.7, (orders, errs)


# ------------------------------------------------------------------ KG Strang (phase_d_c3_wave)

def _kg_run(N, L, T, a=0.8, s=-0.5, f=-0.1):
    *_, psi0 = _gauss(N, L)

    def run(dt, return_state=False):
        op = build_kg(N, L, 1.0, 1.0, dt)
        pk = jnp.fft.fftn(psi0) * op["mask"]
        qk = -0.9j * pk
        pk, qk = kg_evolve(pk, qk, op, a, s, f, int(round(T / dt)))
        return (pk, qk, op) if return_state else jnp.fft.ifftn(pk)
    return run


def test_kg_strang_is_second_order():
    orders, errs = observed_orders(_kg_run(16, 10.0, 1.0), [0.04, 0.02, 0.01])
    assert min(orders) > 1.9, (orders, errs)


def test_kg_strang_energy_drift_scales_as_dt_squared():
    """The splitting error is the only energy error, so the drift must shrink ~4x per dt halving."""
    run = _kg_run(16, 10.0, 1.0)
    drifts = []
    for dt in (0.04, 0.02):
        pk, qk, op = run(dt, return_state=True)
        op0 = build_kg(16, 10.0, 1.0, 1.0, dt)
        *_, psi0 = _gauss(16, 10.0)
        p0 = jnp.fft.fftn(psi0) * op0["mask"]
        e0 = invariants(p0, -0.9j * p0, op0, 0.8, -0.5, -0.1)
        e1 = invariants(pk, qk, op, 0.8, -0.5, -0.1)
        key = "E" if "E" in e0 else sorted(e0)[0]
        drifts.append(abs(float(e1[key]) - float(e0[key])))
    assert drifts[0] / max(drifts[1], 1e-300) > 3.0, drifts


# ------------------------------------------------------------------ TG RK4 (B1S + B2)

TG_BASE = dict(dt=0.002, c=0.5477, m=1.0, a=0.8, s=-0.5, f=-0.1, alpha_T=0.35, omega_T=1.25,
               omega_G=0.85, gamma_T=0.08, gamma_G=0.06, kappa_TG=0.55, epsilon_G=0.06, cT=0.7,
               cG=0.55, L=8.0, absorb_width=1.6, absorb_strength=0.02, core_radius=2.0)
TG_REFS = {"reference_energy_max": 1.0, "reference_charge_max": 1.0, "source_global_norm_S0": 1.0}


def _tg_setup(N):
    grid = b1s.make_grid(build_kg(N, 8.0, TG_BASE["c"], TG_BASE["m"], 0.002))
    X, Y, Z = grid["X"], grid["Y"], grid["Z"]
    rr = (X - 1) ** 2 + Y ** 2 + Z ** 2
    rl = (X + 1) ** 2 + Y ** 2 + Z ** 2
    phi = (jnp.exp(-rr) + jnp.exp(-rl)).astype(jnp.complex128)
    G = -0.01 * jnp.exp(-0.5 * rr)
    z = jnp.zeros_like(G)
    return grid, (phi, -0.964j * phi, z, z, G, z)


def _tg_run(stepper, N, T):
    grid, st = _tg_setup(N)
    refs = b1s.refs_array(TG_REFS)
    fl = jnp.asarray([1.0, 1.0, 1.0, 1.0])

    def run(dt):
        cfg = b1s.cfg_array(dict(TG_BASE, dt=dt))
        s = jax.lax.fori_loop(0, int(round(T / dt)), lambda i, s: stepper(s, cfg, refs, grid, fl), st)
        return jnp.concatenate([a.ravel() for a in s])
    return run


def test_tg_b1s_rk4_is_fourth_order():
    orders, errs = observed_orders(_tg_run(b1s.rk4_step, 16, 0.5), [0.02, 0.01, 0.005])
    assert min(orders) > 3.7, (orders, errs)


def test_tg_b2_rk4_is_fourth_order():
    stepper = lambda s, c, r, g, f: b2.rk4_2n(s, c, r, g, f, jnp.asarray(1.0))  # noqa: E731
    orders, errs = observed_orders(_tg_run(stepper, 16, 0.5), [0.02, 0.01, 0.005])
    assert min(orders) > 3.7, (orders, errs)


def test_tg_b1s_rk4_mms(monkeypatch):
    """Manufactured solution: every field follows its t=0 shape times (1 + 0.1 sin t), with the
    source added inside rhs, which is the function rk4_step calls."""
    N, T = 8, 0.4
    grid, st0 = _tg_setup(N)
    refs = b1s.refs_array(TG_REFS)
    fl = jnp.asarray([1.0, 1.0, 1.0, 1.0])
    orig = b1s.rhs

    def star(t):
        return tuple(y * (1 + 0.1 * np.sin(t)) for y in st0)

    def dstar(t):
        return tuple(y * (0.1 * np.cos(t)) for y in st0)

    errs = []
    dts = [0.1, 0.05, 0.025]
    for dt in dts:
        cfg = b1s.cfg_array(dict(TG_BASE, dt=dt))
        stage = {"i": 0}
        c = (0.0, 0.5, 0.5, 1.0)

        def rhs_mms(state, cfg_, refs_, g_, flags_):
            i = stage["i"]; stage["i"] += 1
            t = (i // 4) * dt + c[i % 4] * dt
            base = orig(state, cfg_, refs_, g_, flags_)
            exact = orig(star(t), cfg_, refs_, g_, flags_)
            return tuple(b + (d - e) for b, d, e in zip(base, dstar(t), exact))

        monkeypatch.setattr(b1s, "rhs", rhs_mms)
        s = st0
        with jax.disable_jit():
            for _ in range(int(round(T / dt))):
                s = b1s.rk4_step(s, cfg, refs, grid, fl)
        monkeypatch.setattr(b1s, "rhs", orig)
        errs.append(max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(s, star(T))))
    orders = [float(np.log2(errs[i] / errs[i + 1])) for i in range(len(errs) - 1)]
    assert min(orders) > 3.7, (orders, errs)


# ------------------------------------------------------------------ Gravity-D RK4

def test_gravity_d_rk4_is_fourth_order():
    N, L, T = 16, 10.0, 0.5
    grid = gd.build_grid(N, L)
    X, Y, Z = grid["X"], grid["Y"], grid["Z"]
    psi0 = (jnp.exp(-((X - 0.5) ** 2 + Y ** 2 + Z ** 2) / 2) * jnp.exp(1j * 0.6 * X)).astype(jnp.complex128)
    Nf = 1.0 + 0.2 * jnp.exp(-(X ** 2 + Y ** 2 + Z ** 2) / 4)

    def run(dt):
        return gd.evolve_n(psi0, Nf, grid, jnp.asarray(dt), int(round(T / dt)))

    orders, errs = observed_orders(run, [0.02, 0.01, 0.005])
    assert min(orders) > 3.7, (orders, errs)
