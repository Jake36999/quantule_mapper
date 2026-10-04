import numpy as np, importlib.util
spec=importlib.util.spec_from_file_location("t1","t1_etdrk4_coeffs.py")
src=open("t1_etdrk4_coeffs.py").read().split("D,eta,om=")[0]
g={}; exec(src.replace("dt=0.005","dt=None"),g)
eta,om=0.0704,1.1866; L=np.array([-eta+1j*om]); c=1+0.5j; T=10.0
# nonlinear test: du/dt = L u - |u|^2 u ; reference = full-complex coeffs at tiny dt
def run(dt,full):
    g['dt']=dt; coef=(g['kt_full'] if full else g['kt_solver'])(L*dt); Q,f1,f2,f3=coef
    E=np.exp(L*dt);E2=np.exp(L*dt/2);N=lambda u:-abs(u)**2*u+c
    u=np.array([0.3+0j])
    for _ in range(int(round(T/dt))):
        Nu=N(u);a=E2*u+Q*Nu;Na=N(a);b=E2*u+Q*Na;Nb=N(b);cc=E2*a+Q*(2*Nb-Nu);Nc=N(cc)
        u=E*u+f1*Nu+2*f2*(Na+Nb)+f3*Nc
    return u[0]
ref=run(0.0005,True)
print(" dt      err(solver real())   err(full complex)")
for dt in [0.04,0.02,0.01,0.005]:
    print(f"{dt:6.3f}   {abs(run(dt,False)-ref):.3e}            {abs(run(dt,True)-ref):.3e}")
