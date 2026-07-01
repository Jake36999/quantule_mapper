"""
Trajectory from the JAX port (FP64, x64 enabled) — runs on GPU in WSL2.
Loads the SAME psi0.npy/cfg.json the CuPy reference used.
"""
import os
import sys
import json
import numpy as np

# Be a polite tenant on the 8 GB desktop GPU: don't grab 75% up front (avoids
# spurious RESOURCE_EXHAUSTED probing on a card shared with Windows/CuPy).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import jax
jax.config.update("jax_enable_x64", True)   # MUST precede any array creation for complex128
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # repo root
ART = os.path.join(HERE, "artifacts")
sys.path.insert(0, ROOT)

from jax_scout import physics

with open(os.path.join(ART, "cfg.json")) as fh:
    CFG = json.load(fh)
psi0 = np.load(os.path.join(ART, "psi0.npy"))

N, L, dt = CFG["N"], CFG["L"], CFG["dt"]
n_steps, every = CFG["n_steps"], CFG["collect_every"]
params = CFG["params"]

print("jax", jax.__version__, "| backend:", jax.default_backend(),
      "| x64:", jax.config.read("jax_enable_x64"))

ops = physics.build_operators(N, L, dt, params,
                              real_dtype=jnp.float64, complex_dtype=jnp.complex128)
psi0_j = jnp.asarray(psi0, dtype=jnp.complex128)
psi_k0, psi_k_final, traj = physics.simulate(psi0_j, ops, n_steps)
psi_k0 = np.asarray(psi_k0)
traj = np.asarray(traj)   # (n_steps, N, N, N); traj[k] is state AFTER step k+1

steps = [i + 1 for i in range(n_steps) if (i + 1) % every == 0]
sel = traj[[s - 1 for s in steps]]

out = os.path.join(ART, "jax_traj.npz")
np.savez_compressed(out, psi_k0=psi_k0, traj=sel, steps=np.array(steps))
print(f"[jax]  wrote {out}  traj={sel.shape} dtype={sel.dtype}  steps={steps}")
