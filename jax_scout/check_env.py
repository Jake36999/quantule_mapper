"""
jax_scout/check_env.py - import/compile health check + backend confirmation.

Run in the WSL2 jax venv from the F: repo root:
    wsl -d Ubuntu -- bash -lc '. ~/jax_irer/bin/activate && python /mnt/f/quantule_mapper/jax_scout/check_env.py'
Confirms: x64 FP64 active, GPU backend visible, physics module imports, one step compiles+runs.
"""
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jax_scout import physics

print("jax            :", jax.__version__)
print("backend        :", jax.default_backend())
print("devices        :", jax.devices())
print("x64 enabled    :", jax.config.read("jax_enable_x64"))

N = 16
params = {"param_D": 1.0, "param_eta": 0.2, "param_rho_vac": 1.33, "param_omega0": 1.0,
          "param_a": 0.1, "param_s": 0.05, "param_f": 0.02, "param_a_coupling": 1.0}
ops = physics.build_operators(N, 10.0, 0.005, params,
                              real_dtype=jnp.float64, complex_dtype=jnp.complex128)
key = jax.random.PRNGKey(0)
psi0 = (jax.random.normal(key, (N, N, N), jnp.float64)
        + 1j * jax.random.normal(jax.random.PRNGKey(1), (N, N, N), jnp.float64)).astype(jnp.complex128)
psi_k0, psi_k_final, traj = physics.simulate(psi0, ops, 5)
psi_k_final.block_until_ready()

print("compile+run    : OK  (5 steps @ %d^3, dtype=%s)" % (N, psi_k_final.dtype))
print("final finite   :", bool(jnp.all(jnp.isfinite(jnp.abs(psi_k_final)))))
print("\nENV CHECK PASS" if str(jax.default_backend()) == "gpu" else
      "\nENV CHECK WARN: backend is not gpu")
