"""jax_scout/continuation.py (Phase F3) against a closed-form answer on the DISSIPATIVE substrate.

The uniform state of the S-NCGL operator, psi = sqrt(rho*) e^{i omega0 t} with g(rho*) = eta
(g = a rho + s rho^2 + f rho^3), is a relative equilibrium: the return map rotates it by theta = omega0 T.
On the stable (upper) root Newton-Krylov must converge QUADRATICALLY to rho* and theta to round-off.
(The conservative NLS soliton is NOT a valid target: solitons form a continuous family in mu, so the
bordered Jacobian is singular along it -- documented in docs/research_infrastructure/BASIN_MAPPING.md.)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
jax = pytest.importorskip("jax", reason="continuation needs JAX")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jax_scout import continuation as C  # noqa: E402

ETA, A, S, F, W0 = 0.0704, 0.55223, 0.0129, -0.4861, 1.1866
P = {"param_D": 2.7329, "param_eta": ETA, "param_rho_vac": 1.1866, "param_omega0": W0,
     "param_a_coupling": 2.3098, "param_s": S, "param_f": F, "param_a": A}


def test_newton_krylov_recovers_the_uniform_state_quadratically():
    rho_star = max(r.real for r in np.roots([F, S, A, -ETA]) if abs(r.imag) < 1e-12 and r.real > 0)
    N, T = 8, 0.25
    prob = C.RelEqProblem(N=N, L=10.0, dt=0.005, T=T, params=P, translations=False)
    ref = jnp.full((N, N, N), np.sqrt(rho_star), dtype=jnp.complex128)
    r = C.newton_krylov(prob, prob.pack(ref * 1.02, 0.0, jnp.zeros(3)), psi_ref=ref, tol=1e-12,
                        max_iter=10, gmres_tol=1e-12, restart=30, maxiter=3)
    psi, theta, _, _ = prob.unpack(jnp.asarray(r.x))
    assert r.converged, r.residuals
    assert abs(float(jnp.mean(jnp.abs(psi) ** 2)) - rho_star) / rho_star < 1e-9
    assert abs(float(theta) - W0 * T) < 1e-10
    # quadratic tail: each of the last steps roughly squares the residual
    tail = [x for x in r.residuals if x > 1e-14][-3:]
    assert np.log10(tail[-1]) < 1.6 * np.log10(tail[-2]) or tail[-1] < 1e-12, r.residuals
