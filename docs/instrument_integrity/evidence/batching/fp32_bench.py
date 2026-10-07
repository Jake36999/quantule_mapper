"""fp32 vs fp64 ETDRK4 on the GTX 1080: speed, and how far fp32 drifts from fp64 over a long a* run."""
import sys, time, numpy as np
sys.path.insert(0, "/mnt/f/quantule_mapper")
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax_scout import physics, core_saturation_search as css
P = dict(css.FEB); P["param_a"] = css.FEB["param_a"] * 1.15
for N in (48, 96):
    psi0, _ = css.build_ic(N, 6, seed=20260619)
    res = {}
    for name, rd, cd in (("fp64", jnp.float64, jnp.complex128), ("fp32", jnp.float32, jnp.complex64)):
        ops = physics.build_operators(N, 10.0, 0.005, P, rd, cd)
        adv = jax.jit(lambda pk, n: jax.lax.fori_loop(0, n, lambda i, q: physics.step(q, ops), pk), static_argnums=1)
        pk = physics.initial_psi_k(jnp.asarray(psi0, dtype=cd), ops)
        adv(pk, 10).block_until_ready()
        t0 = time.time(); out = adv(pk, 2000); out.block_until_ready(); dt = (time.time() - t0) / 2000
        pk = physics.initial_psi_k(jnp.asarray(psi0, dtype=cd), ops)
        traj = []
        for _ in range(36):                    # T = 36 * 2000 * 0.005 = 360 (the a* replay length)
            pk = adv(pk, 2000); traj.append(float(jnp.sum(jnp.abs(pk) ** 2)))
        res[name] = (dt, np.array(traj), np.asarray(jnp.fft.ifftn(pk)))
        print(f"N={N} {name}: {dt*1e3:.2f} ms/step", flush=True)
    e64, e32 = res["fp64"][1], res["fp32"][1]
    p64, p32 = res["fp64"][2], res["fp32"][2]
    print(f"N={N} speed-up fp32/fp64: {res['fp64'][0]/res['fp32'][0]:.1f}x | energy ratio rel diff over T=360: "
          f"max {np.max(np.abs(e32/e64-1)):.2e}, final |psi|^2 field rel L2 diff {np.linalg.norm(abs(p32)**2-abs(p64)**2)/np.linalg.norm(abs(p64)**2):.2e}", flush=True)
