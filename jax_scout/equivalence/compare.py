"""
Compare the CuPy backbone vs JAX port trajectories and render a verdict.

At FP64, bit-identity across cuFFT(CuPy) and XLA-FFT(JAX) plus differing reduction
orders is NOT expected; the meaningful bar is agreement to FP64 round-off that does
not blow up over steps. PASS threshold: relative L2 error < 1e-8 on every snapshot.
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
PASS_TOL = 1e-8

c = np.load(os.path.join(ART, "cupy_traj.npz"))
j = np.load(os.path.join(ART, "jax_traj.npz"))

with open(os.path.join(ART, "cfg.json")) as fh:
    CFG = json.load(fh)


def rel_l2(a, b):
    denom = np.linalg.norm(b.ravel())
    return float(np.linalg.norm((a - b).ravel()) / denom) if denom > 0 else float("nan")


def maxabs(a, b):
    return float(np.max(np.abs(a - b)))


assert np.array_equal(c["steps"], j["steps"]), f"step mismatch: {c['steps']} vs {j['steps']}"
steps = c["steps"]

print(f"config: N={CFG['N']}^3  dt={CFG['dt']}  n_steps={CFG['n_steps']}")
print(f"params: {CFG['params']}\n")

print(f"{'step':>6} | {'rel_L2(psi_k)':>16} | {'max_abs_diff':>14}")
print("-" * 44)
e0 = rel_l2(j["psi_k0"], c["psi_k0"])
print(f"{'init':>6} | {e0:16.3e} | {maxabs(j['psi_k0'], c['psi_k0']):14.3e}")

worst = e0
for idx, s in enumerate(steps):
    e = rel_l2(j["traj"][idx], c["traj"][idx])
    m = maxabs(j["traj"][idx], c["traj"][idx])
    worst = max(worst, e)
    print(f"{int(s):6d} | {e:16.3e} | {m:14.3e}")

print("-" * 44)
verdict = "PASS" if worst < PASS_TOL else "FAIL"
print(f"\nworst rel_L2 = {worst:.3e}   (threshold {PASS_TOL:.0e})   ->   {verdict}")
print("\nJAX port and CuPy backbone are numerically equivalent in FP64."
      if verdict == "PASS" else
      "\nDivergence exceeds FP64 round-off -- investigate the port.")
