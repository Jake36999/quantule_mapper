import os, sys
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jax_scout import core_saturation_search as css, physics
from jax_scout.phase_d_c1_transport import _evolve_chunk
from jax_scout.phase_d_c2_5_family_scout import _ops_family, N, DT
from jax_scout.phase_d_c2_3_exact_soliton import petviashvili

L = css.L_
D = 1.0
k1 = 2 * np.pi / L
ops = _ops_family(0.8, -0.2, 0.0, D)
print("geom_fac =", float(np.asarray(ops.geom_fac)))
x = np.linspace(-L / 2, L / 2, N, endpoint=False)
X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

# 2. plane-wave one-step phase
psi = (1e-6 * np.exp(1j * k1 * X)).astype(np.complex128)
pk = physics.initial_psi_k(jnp.asarray(psi), ops)
r = complex(np.asarray(physics.step(pk, ops))[1, 0, 0] / np.asarray(pk)[1, 0, 0])
print(f"2. one-step phase: {np.angle(r):+.6e}  expect {-D * k1**2 * DT:+.6e}")

# 3. boosted linear packet translation
psi = (0.01 * np.exp(-(X**2 + Y**2 + Z**2) / (2 * 1.5**2)) * np.exp(1j * k1 * X)).astype(np.complex128)
pk = physics.initial_psi_k(jnp.asarray(psi), ops)
ixs = [24]
for i in range(5):
    pk = _evolve_chunk(pk, ops, 100)
    rho = np.abs(np.asarray(jnp.fft.ifftn(pk))) ** 2
    ixs.append(int(np.unravel_index(rho.argmax(), rho.shape)[0]))
print(f"3. packet peak trajectory {ixs}  (expect 24 -> ~30; v=2Dk={2 * D * k1:.3f})")

# 4. soliton on the TRUE substrate + winding boost
psi_c, prof = petviashvili(1.0, 0.08, ops, N, 0.2)
res = prof.get("residual", np.inf) if isinstance(prof, dict) else np.inf
print(f"4. petviashvili on TRUE pure NLS: {prof if psi_c is None else ''}")
if psi_c is not None:
    print(f"   residual={prof['residual']:.2e} amp={prof['amp']:.3f} occ={prof['occ']:.4f} mass={prof['mass']:.0f}")
if psi_c is not None and prof["residual"] < 1e-6 and prof["occ"] < 0.9:
    pk = physics.initial_psi_k(jnp.asarray((psi_c * np.exp(1j * k1 * X)).astype(np.complex128)), ops)
    rho0 = np.abs(psi_c) ** 2
    ixs = [int(np.unravel_index(rho0.argmax(), rho0.shape)[0])]
    for i in range(5):
        pk = _evolve_chunk(pk, ops, 100)
        cur = np.asarray(jnp.fft.ifftn(pk))
        rho = np.abs(cur) ** 2
        ixs.append(int(np.unravel_index(rho.argmax(), rho.shape)[0]))
    m = float(np.sum(np.abs(cur) ** 2) / np.sum(np.abs(psi_c) ** 2))
    print(f"   boosted soliton peak trajectory {ixs} (expect +6 cells by t=0.5) mass_ret={m:.4f}")
else:
    # mu=0.2 branch may not exist on the TRUE substrate (different D_eff!) -> scan a few mu
    for mu in (0.05, 0.1, 0.15, 0.25):
        psi_c, prof = petviashvili(1.0, 0.08, ops, N, mu)
        if psi_c is not None:
            print(f"   mu={mu}: residual={prof['residual']:.2e} amp={prof['amp']:.3f} occ={prof['occ']:.4f}")
