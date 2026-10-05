"""Validate jax_scout/continuation.py on a case with a known answer: the exact NLS soliton of the C2
conservative branch is a relative equilibrium, psi(T) = e^{i w T} phi, with |w| = mu (Petviashvili)."""
import sys, time, numpy as np
sys.path.insert(0, "/mnt/f/quantule_mapper_dev")
import jax, jax.numpy as jnp
from jax_scout import physics, continuation as C
from jax_scout.phase_d_c2_3_exact_soliton import petviashvili

N, L, dt, mu = (int(sys.argv[1]) if len(sys.argv) > 1 else 24), 10.0, 0.005, 0.2
T = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
P = {"param_D": 1.0, "param_eta": 0.0704, "param_rho_vac": 1.1866, "param_omega0": 0.0, "param_a_coupling": 0.0,
     "param_s": -0.2, "param_f": 0.0, "param_a": 0.8, "kinetic_mode": "conservative", "param_geom_off": True}
ops = physics.build_operators(N, L, dt, P)
phi, prof = petviashvili(1.0, 0.08, ops, N, mu)
print("petviashvili residual", prof["residual"], "amp", prof["amp"])
phi = jnp.asarray(phi)
prob = C.RelEqProblem(N=N, L=L, dt=dt, T=T, params=P)
# where does the exact soliton go? measure theta directly
out = jnp.fft.ifftn(prob.flow(phi, None))
theta_true = float(jnp.angle(jnp.sum(jnp.conj(phi) * out)))
print("measured rotation over T: theta =", theta_true, " mu*T =", mu * T)
rng = np.random.default_rng(0)
pert = phi * (1 + 0.05 * rng.standard_normal(phi.shape)) + 0.01 * jnp.roll(phi, 1, axis=0)
x0 = prob.pack(pert, 0.0, jnp.zeros(3))
t0 = time.time()
r = C.newton_krylov(prob, x0, psi_ref=phi, tol=1e-9, max_iter=10, gmres_tol=1e-10, restart=100, maxiter=10, log=print)
psi, th, a, _ = prob.unpack(jnp.asarray(r.x))
err = float(jnp.max(jnp.abs(jnp.abs(psi) - jnp.abs(phi)))) / float(jnp.max(jnp.abs(phi)))
print("converged", r.converged, "iters", r.iters, "theta", float(th), "shift", np.asarray(a),
      "max rel |psi|-|phi|", err, "%.1fs" % (time.time() - t0))
mult = C.floquet_multipliers(prob, r.x, p=None, k=6)
print("leading |Floquet multipliers|:", np.round(mult, 6))
