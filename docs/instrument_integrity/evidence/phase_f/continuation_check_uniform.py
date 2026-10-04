"""Known-answer check for jax_scout/continuation.py on the DISSIPATIVE substrate (isolated solutions).
Uniform state of the S-NCGL operator: psi = sqrt(rho*) e^{i w t} with -eta + g(rho*) = 0,
g = a rho + s rho^2 + f rho^3, and the phase rotates at omega0 (L_k at k=0 = -eta + i omega0).
Translations are degenerate for a uniform state, so they are switched off."""
import sys, time, numpy as np
sys.path.insert(0, "/mnt/f/quantule_mapper_dev")
import jax, jax.numpy as jnp
from jax_scout import continuation as C

N, L, dt, T = 16, 10.0, 0.005, float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
eta, a, s, f, w0 = 0.0704, 0.55223, 0.0129, -0.4861, 1.1866
P = {"param_D": 2.7329, "param_eta": eta, "param_rho_vac": 1.1866, "param_omega0": w0,
     "param_a_coupling": 2.3098, "param_s": s, "param_f": f, "param_a": a}
roots = sorted(r.real for r in np.roots([f, s, a, -eta]) if abs(r.imag) < 1e-12 and r.real > 0)
gp = lambda r: a + 2 * s * r + 3 * f * r * r
prob = C.RelEqProblem(N=N, L=L, dt=dt, T=T, params=P, translations=False)
for rho_star in roots:
    mu_amp = float(np.exp(2 * rho_star * gp(rho_star) * T))
    print("\nroot rho* = %.6f  theta* = %.4f  analytic amplitude multiplier = %.6f" % (rho_star, w0 * T, mu_amp), flush=True)
    ref = jnp.full((N, N, N), np.sqrt(rho_star), dtype=jnp.complex128)
    x0 = prob.pack(ref * 1.02, 0.0, jnp.zeros(3))
    t0 = time.time()
    r = C.newton_krylov(prob, x0, psi_ref=ref, tol=1e-10, max_iter=10, gmres_tol=1e-10, restart=40, maxiter=4, log=print)
    psi, th, _, _ = prob.unpack(jnp.asarray(r.x))
    rho = float(jnp.mean(jnp.abs(psi) ** 2))
    print("converged", r.converged, "iters", r.iters, "rho", rho, "rel err %.2e" % (abs(rho - rho_star) / rho_star),
          "theta %.10f err %.2e" % (float(th), abs(float(th) - w0 * T)), "%.1fs" % (time.time() - t0), flush=True)
    m = C.floquet_multipliers(prob, r.x, k=6)
    print("leading |mu|:", np.round(m, 6), " closest to analytic: %.2e" % float(np.min(np.abs(m - mu_amp))), flush=True)
