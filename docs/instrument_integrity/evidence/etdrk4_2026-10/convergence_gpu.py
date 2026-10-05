"""dt-convergence of the real CuPy ETDRK4Solver, OLD (half-circle+real) vs FIXED (full-circle complex) coefficients."""
import sys, time, numpy as np, cupy as cp
sys.path.insert(0, r"F:\quantule_mapper")
from solver.core import ETDRK4Solver
from solver.run import initialize_psi

FEB = {"param_D": 2.7329, "param_eta": 0.0704, "param_rho_vac": 1.1866, "param_a_coupling": 2.3098,
       "param_splash_coupling": 0.0129, "param_splash_fraction": -0.4861, "param_a": 0.5522}
import os, json
FEB.update(json.loads(os.environ.get("OVR","{}")))
DTS=json.loads(os.environ.get("DTS","[0.04,0.02,0.01,0.005,0.0025]")); REFN=int(os.environ.get("REFN","6400"))
N, L, T = int(sys.argv[1]) if len(sys.argv) > 1 else 32, 10.0, float(sys.argv[2]) if len(sys.argv) > 2 else 2.0

def old_coeffs(s, dt):
    M = 64; th = cp.exp(1j*cp.pi*(cp.arange(1, M+1)-0.5)/M); w = s.L_k*dt
    acc = [cp.zeros_like(w) for _ in range(4)]
    for i in range(M):
        we = w + th[i]; ew = cp.exp(we)
        acc[0] += (cp.exp(we/2)-1)/we; acc[1] += (-4-we+ew*(4-3*we+we**2))/we**3
        acc[2] += (2+we+ew*(we-2))/we**3; acc[3] += (-4-3*we-we**2+ew*(4-we))/we**3
    s.Q, s.f1, s.f2, s.f3 = [dt*cp.real(a/M) for a in acc]

def run(dt, old):
    s = ETDRK4Solver(N, L, dt, dict(FEB))
    if old:
        old_coeffs(s, dt)
        def old_step(psi_k):  # original stage c used N(a) instead of N(u_n)
            Nn = s.N_op(psi_k); a = s.E2*psi_k + s.Q*Nn; Na = s.N_op(a); b = s.E2*psi_k + s.Q*Na; Nb = s.N_op(b)
            c = s.E2*a + s.Q*(2*Nb - Na); Nc = s.N_op(c)
            return (s.E*psi_k + s.f1*Nn + 2*s.f2*(Na+Nb) + s.f3*Nc) * s.dealias_mask
        s.step = old_step
    psi_k = s.fft_single(initialize_psi(N, L, 42)) * s.dealias_mask
    for _ in range(int(round(T/dt))): psi_k = s.step(psi_k)
    cp.cuda.Stream.null.synchronize()
    return s.ifft_single(psi_k)

t0 = time.time(); ref = run(T/REFN, False); nref = float(cp.linalg.norm(ref))
print(f"N={N} T={T} feb params (a=a*), reference dt={T/REFN:.6f} ovr={os.environ.get('OVR','{}')}  |psi_ref|={nref:.4f}  ({time.time()-t0:.1f}s)")
print("    dt      OLD rel err   order     FIXED rel err  order")
prev = None
for dt in DTS:
    eo = float(cp.linalg.norm(run(dt, True)-ref))/nref; ef = float(cp.linalg.norm(run(dt, False)-ref))/nref
    oo = f"{np.log2(prev[0]/eo):5.2f}" if prev else "  -  "; of = f"{np.log2(prev[1]/ef):5.2f}" if prev else "  -  "
    print(f"{dt:8.4f}   {eo:.3e}    {oo}     {ef:.3e}     {of}"); prev = (eo, ef)
