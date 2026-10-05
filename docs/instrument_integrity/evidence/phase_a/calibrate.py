import os, sys, numpy as np
os.environ.setdefault("JAX_ENABLE_X64","1")
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
sys.path.insert(0, "/mnt/f/quantule_mapper")
from jax_scout import physics
from jax_scout.phase_d_c3_wave import build_kg, kg_evolve
from jax_scout import gravity_TG_B1S_state_load_feedback_gpu as b1s
from jax_scout import gravity_TG_B2_two_node_awell as b2
from jax_scout import gravity_D_neutral_probe_gpu as gd

def orders(run, dts, ref_div=16):
    ref = run(dts[-1]/ref_div); errs=[float(jnp.max(jnp.abs(run(dt)-ref))) for dt in dts]
    return errs, [np.log2(errs[i]/errs[i+1]) for i in range(len(errs)-1)]

N, L = 16, 10.0
x = jnp.linspace(-L/2, L/2, N, endpoint=False); X,Y,Z = jnp.meshgrid(x,x,x,indexing="ij")
g0 = jnp.exp(-(X**2+Y**2+Z**2)/2).astype(jnp.complex128)
FEB = {"param_D":2.7329,"param_eta":0.0704,"param_rho_vac":1.1866,"param_a_coupling":2.3098,
       "param_splash_coupling":0.0129,"param_splash_fraction":-0.4861,"param_a":0.5522}
def etd(params, T):
    def run(dt):
        ops = physics.build_operators(N, L, dt, params); pk = physics.initial_psi_k(g0, ops)
        n = int(round(T/dt)); pk = jax.lax.fori_loop(0, n, lambda i,p: physics.step(p, ops), pk)
        return jnp.fft.ifftn(pk)
    return run
print("ETDRK4 dissipative feb:", orders(etd(FEB, 0.5), [0.02,0.01,0.005,0.0025]))
print("ETDRK4 conservative:", orders(etd({"param_D":1.0,"param_a":0.8,"param_s":-0.2,"kinetic_mode":"conservative","param_geom_off":True}, 0.5), [0.004,0.002,0.001,0.0005]))

op0 = build_kg(N, L, 1.0, 1.0, 0.01)
def kg(T):
    def run(dt):
        op = build_kg(N, L, 1.0, 1.0, dt); pk = jnp.fft.fftn(g0)*op["mask"]; qk = -1j*0.9*pk
        pk, qk = kg_evolve(pk, qk, op, 0.8, -0.5, -0.1, int(round(T/dt))); return jnp.fft.ifftn(pk)
    return run
print("KG strang:", orders(kg(1.0), [0.04,0.02,0.01,0.005]))

BASE = dict(dt=0.002, c=0.5477, m=1.0, a=0.8, s=-0.5, f=-0.1, alpha_T=0.35, omega_T=1.25, omega_G=0.85, gamma_T=0.08, gamma_G=0.06,
            kappa_TG=0.55, epsilon_G=0.06, cT=0.7, cG=0.55, L=8.0, absorb_width=1.6, absorb_strength=0.02, core_radius=2.0)
grid = b1s.make_grid(build_kg(N, 8.0, BASE["c"], BASE["m"], 0.002))
refs = b1s.refs_array({"reference_energy_max":1.0,"reference_charge_max":1.0,"source_global_norm_S0":1.0})
fl = jnp.asarray([1.,1.,1.,1.])
Xg,Yg,Zg = grid["X"],grid["Y"],grid["Z"]
phi = (jnp.exp(-((Xg-1)**2+Yg**2+Zg**2)) + jnp.exp(-((Xg+1)**2+Yg**2+Zg**2))).astype(jnp.complex128)
G = -0.01*jnp.exp(-0.5*((Xg-1)**2+Yg**2+Zg**2)); z = jnp.zeros_like(G)
st = (phi, -1j*0.964*phi, z, z, G, z)
def tg(stepper, T):
    def run(dt):
        cfg = b1s.cfg_array(dict(BASE, dt=dt)); n=int(round(T/dt)); s=st
        s = jax.lax.fori_loop(0, n, lambda i, s: stepper(s, cfg), s)
        return jnp.concatenate([a.ravel() for a in s])
    return run
print("TG B1S rk4:", orders(tg(lambda s,c: b1s.rk4_step(s,c,refs,grid,fl), 0.5), [0.04,0.02,0.01,0.005]))
print("TG B2 rk4_2n:", orders(tg(lambda s,c: b2.rk4_2n(s,c,refs,grid,fl,jnp.asarray(1.0)), 0.5), [0.04,0.02,0.01,0.005]))
print([k for k in dir(gd) if k.startswith("make") or k=="D" or "grid" in k.lower()])
