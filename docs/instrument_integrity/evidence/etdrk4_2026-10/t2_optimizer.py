import numpy as np, random, warnings
from quantulemapper_real import calculate_bipartite_sse, fit_scale_factor, TARGET_LN_PRIMES
rng=np.random.default_rng(0)
print("== T2a: 'target-shuffle' null vs main metric ==")
same=0
for i in range(1000):
    peaks=sorted(rng.uniform(0.05,1,rng.integers(1,12))); s=fit_scale_factor(peaks); sp=[p*s for p in peaks]
    m=calculate_bipartite_sse(sp,TARGET_LN_PRIMES)["total_sse"]; n=calculate_bipartite_sse(sp,rng.permutation(TARGET_LN_PRIMES))["total_sse"]
    same+= (m==n)
print(f"null == main in {same}/1000 random cases (bipartite_sse sorts targets, so the shuffle is undone)")

print("\n== T2b: what log-prime SSE does PURE NOISE get? ==")
res=[]
for i in range(20000):
    k=7; peaks=sorted(rng.uniform(0.03,1,k)); s=fit_scale_factor(peaks)
    r=calculate_bipartite_sse([p*s for p in peaks],TARGET_LN_PRIMES); res.append((r["total_sse"],r["primary_harmonic_error"]))
res=np.array(res)
print(f"7 uniform-random peaks: median SSE={np.median(res[:,0]):.3f}, 5th pct={np.percentile(res[:,0],5):.3f}, min={res[:,0].min():.4f}")
print(f"  primary_harmonic_error (best single match): median={np.median(res[:,1]):.2e}, frac<0.01 = {np.mean(res[:,1]<0.01):.2f}")
print("  (scale clamp 0.01..5 can reach ln17=2.83 from k<=1 peaks; 'primary_harmonic_error<0.01' = predator-lock trigger)")

print("\n== T2c: Hunter crossover (70% per-gene 50/50 blend) on a FLAT landscape ==")
pop=np.random.default_rng(1).uniform(-2,2,(40,7)); r=random.Random(1)
for gen in range(31):
    if gen in (0,5,10,20,30): print(f"gen {gen:2d}: mean per-param std = {pop.std(0).mean():.4f}")
    new=[pop[0].copy()]
    while len(new)<40:
        p1,p2=pop[r.randrange(40)],pop[r.randrange(40)]
        c=np.where(np.array([r.random()<0.7 for _ in range(7)]),0.5*p1+0.5*p2,p1)
        mut=np.array([r.random()<0.1 for _ in range(7)]); f=np.array([r.uniform(-.05*.3,.05*.3) for _ in range(7)])
        c=c+mut*c*f; new.append(c)
    pop=np.array(new)
print("  -> diversity collapses ~geometrically with no selection pressure at all (mutation is multiplicative, ~1%)")

print("\n== T2d: SGN step size ==")
for g in [0.1,1,10,100]:
    lr=0.01/(1+g); print(f"|grad|={g:5}: step = {lr*g:.4f}  (a_coupling range 4.0, splash range 10.0)")

print("\n== T2e: FSS quadratic fit with constant ell column ==")
from scipy.optimize import curve_fit
def poly(X,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9):
    a,s,lp=X; return c0+c1*a+c2*s+c3*lp+c4*a**2+c5*s**2+c6*lp**2+c7*a*s+c8*a*lp+c9*s*lp
a=rng.uniform(0,1,15); s=rng.uniform(0,1,15); lp=np.full(15,np.log(3)); y=(a-.4)**2+(s-.6)**2+0.01*rng.normal(size=15)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always"); popt,pcov=curve_fit(poly,(a,s,lp),y,maxfev=10000)
print("pcov diag:",np.diag(pcov)[:4],"...  confidence=",1/(1+np.mean(np.sqrt(np.diag(pcov)))), "| warnings:",[str(x.message)[:50] for x in w])
J=np.column_stack([np.ones(15),a,s,lp,a*a,s*s,lp*lp,a*s,a*lp,s*lp]); print("design matrix rank:",np.linalg.matrix_rank(J),"of 10")
