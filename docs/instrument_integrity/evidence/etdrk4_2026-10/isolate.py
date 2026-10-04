"""Isolate the order-limiting piece: real ETDRK4Solver.step with (A) simple cubic N, (B) full N_op but
fused_process_omega replaced by identity, (C) full N_op."""
import sys, json, os, numpy as np, cupy as cp
sys.path.insert(0, r"F:\quantule_mapper")
import solver.core as core
from solver.core import ETDRK4Solver
from solver.run import initialize_psi
FEB = {"param_D": 2.7329, "param_eta": 0.0704, "param_rho_vac": 1.1866, "param_a_coupling": 2.3098,
       "param_splash_coupling": 0.0129, "param_splash_fraction": -0.4861, "param_a": 0.5522}
FEB.update(json.loads(os.environ.get("OVR", "{}")))
N, L, T = 32, 10.0, 2.0
mode = sys.argv[1]
if mode == "B":
    core.fused_process_omega = lambda o, lo, hi: (o, cp.sqrt(o))
    core.fused_scale_derivative = lambda o, d, lo, hi: d
def run(dt):
    s = ETDRK4Solver(N, L, dt, dict(FEB))
    if mode == "A":
        def nop(psi_k):
            p = s.ifft_single(psi_k); return s.fft_single(-0.5 * cp.abs(p)**2 * p) * s.dealias_mask
        s.N_op = nop
    psi0 = initialize_psi(N, L, 42)
    if os.environ.get("SMOOTH"): x = cp.linspace(-L/2, L/2, N, endpoint=False); X,Y,Z = cp.meshgrid(x,x,x,indexing="ij"); psi0 = cp.exp(-(X**2+Y**2+Z**2)/2).astype(cp.complex128)
    psi_k = s.fft_single(psi0) * s.dealias_mask
    for _ in range(int(round(T/dt))): psi_k = s.step(psi_k)
    return s.ifft_single(psi_k)
ref = run(T/3200); nr = float(cp.linalg.norm(ref)); prev = None
print(f"mode {mode} ovr={os.environ.get('OVR','{}')}")
for dt in [0.04, 0.02, 0.01, 0.005]:
    e = float(cp.linalg.norm(run(dt)-ref))/nr
    print(f"  dt={dt:.4f} err={e:.3e} order={'-' if prev is None else f'{np.log2(prev/e):.2f}'}"); prev = e
