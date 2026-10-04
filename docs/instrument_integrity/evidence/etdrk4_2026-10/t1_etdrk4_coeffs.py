"""Test 1: the solver's Kassam-Trefethen coefficients take real() of an upper-half-circle mean.
That trick is only valid when L is real. Here L = -D k^2 - eta + i*omega0 is complex."""
import numpy as np
M=64; dt=0.005
def kt_solver(w):   # exactly as solver/core.py and jax_scout/physics.py
    th=np.exp(1j*np.pi*(np.arange(1,M+1)-0.5)/M); we=w[...,None]+th; ew=np.exp(we)
    Q=((np.exp(we/2)-1)/we).mean(-1); f1=((-4-we+ew*(4-3*we+we**2))/we**3).mean(-1)
    f2=((2+we+ew*(we-2))/we**3).mean(-1); f3=((-4-3*we-we**2+ew*(4-we))/we**3).mean(-1)
    return [dt*np.real(x) for x in (Q,f1,f2,f3)]
def kt_full(w):     # correct for complex w: full circle, no real()
    th=np.exp(2j*np.pi*(np.arange(1,2*M+1)-0.5)/(2*M)); we=w[...,None]+th; ew=np.exp(we)
    Q=((np.exp(we/2)-1)/we).mean(-1); f1=((-4-we+ew*(4-3*we+we**2))/we**3).mean(-1)
    f2=((2+we+ew*(we-2))/we**3).mean(-1); f3=((-4-3*we-we**2+ew*(4-we))/we**3).mean(-1)
    return [dt*x for x in (Q,f1,f2,f3)]
D,eta,om=2.7329,0.0704,1.1866
k=2*np.pi*np.fft.fftfreq(64,d=10/64); k2=np.unique(np.add.outer(k**2,k**2).ravel())[:200]
for label,L in [("Phase C baseline (omega0=rho_vac)",-D*k2-eta+1j*om),
                ("NLS-like (i*D*k^2)",-1j*1.0*k2),
                ("real L (omega0=0, control)",-D*k2-eta+0j)]:
    a=kt_solver(L*dt); b=kt_full(L*dt)
    rel=[np.max(np.abs(x-y))/np.max(np.abs(y)) for x,y in zip(a,b)]
    print(f"{label:38s} max rel err Q,f1,f2,f3 = "+", ".join(f"{r:.2e}" for r in rel))
# ODE check: du/dt = L u + c (constant forcing), exact u = e^{Lt}u0 + (e^{Lt}-1)/L c
def step(u,L,c,coef):
    Q,f1,f2,f3=coef; E=np.exp(L*dt); E2=np.exp(L*dt/2); N=lambda u:c
    Nu=N(u); a=E2*u+Q*Nu; Na=N(a); b=E2*u+Q*Na; Nb=N(b); cc=E2*a+Q*(2*Nb-Nu); Nc=N(cc)
    return E*u+f1*Nu+2*f2*(Na+Nb)+f3*Nc
for name,L in [("L=-eta+i*omega0 (k=0 mode)",np.array([-eta+1j*om])),("L=-D k^2-eta+i*om, k=2",np.array([-D*4-eta+1j*om])),("NLS L=-i k^2, k=3",np.array([-9j]))]:
    c=1.0+0.5j; T=10.0; n=int(T/dt)
    ex=(np.exp(L*T)-1)/L*c
    for tag,coef in [("solver real()",kt_solver(L*dt)),("full complex",kt_full(L*dt))]:
        u=np.zeros(1,complex)
        for _ in range(n): u=step(u,L,c,coef)
        print(f"{name:30s} {tag:14s} |u-exact|/|exact| at t=10: {abs(u-ex)[0]/abs(ex)[0]:.3e}")
