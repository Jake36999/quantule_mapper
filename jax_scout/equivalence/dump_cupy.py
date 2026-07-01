"""
Reference trajectory from the CuPy ASTE backbone (the source of truth).
Drives the *production* solver/core.py:ETDRK4Solver.step directly in FP64.
Run with the native-Windows .venv that has cupy installed.
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))   # repo root
ART = os.path.join(HERE, "artifacts")
sys.path.insert(0, ROOT)

import cupy as cp
from solver.core import ETDRK4Solver

with open(os.path.join(ART, "cfg.json")) as fh:
    CFG = json.load(fh)
psi0 = np.load(os.path.join(ART, "psi0.npy"))

N, L, dt = CFG["N"], CFG["L"], CFG["dt"]
n_steps, every = CFG["n_steps"], CFG["collect_every"]
params = CFG["params"]

solver = ETDRK4Solver(N, L, dt, params)
psi0_cp = cp.asarray(psi0)
psi_k = solver.fft_single(psi0_cp) * solver.dealias_mask
psi_k0 = cp.asnumpy(psi_k)

snaps, steps = [], []
for i in range(n_steps):
    psi_k = solver.step(psi_k)
    if (i + 1) % every == 0:
        snaps.append(cp.asnumpy(psi_k))
        steps.append(i + 1)

out = os.path.join(ART, "cupy_traj.npz")
np.savez_compressed(out, psi_k0=psi_k0, traj=np.stack(snaps), steps=np.array(steps))
print(f"[cupy] wrote {out}  traj={np.stack(snaps).shape} dtype={snaps[0].dtype}  steps={steps}")
