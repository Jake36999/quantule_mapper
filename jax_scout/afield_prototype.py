"""
IRER_A_FIELD_GEOMETRIC_FEEDBACK_v1_PROTOTYPE

Minimal causal A-field geometric-feedback prototype (JAX scout, ACTIVE experimental
branch — NOT passive diagnostic, NOT rank-compatible with gamma_A=0, NOT for Hunter
promotion or CuPy validation until paired controls pass).

Topology (the smallest A-coupling path):
    A          : finite-speed field-of-affect, integrated in spectral space
                 d^2 A_k/dt^2 = -c^2 k^2 A_k + rho_k   (DC source removed; k=0 pinned)
    A~ (Atilde): A_real = Re(ifft(A_k))
    rho_vac_eff = max(rho_vac + gamma_A * A~, RHO_VAC_EFF_FLOOR)
    Omega^2     = (rho_vac_eff / rho)^a               # via physics.step(..., rho_vac_eff)

Guarantees / safety:
  * gamma_A = 0 -> rho_vac_eff = rho_vac (constant) -> reproduces physics.step exactly.
  * A is NOT a direct psi force term; A does NOT replace rho; old fluid/tensor feedback off.
  * rho_vac_eff floored > 0 (negative gamma_A*A~ cannot drive it non-positive).
  * A is updated ONCE per outer step (held fixed across the 4 ETDRK4 substages), mirroring
    solver/run.py:update_field_of_affect ordering.
  * A-on runs carry a distinct contract key (CONTRACT_KEY) -> not rank-compatible with A-off.
"""
from functools import partial
import jax
import jax.numpy as jnp
from jax import lax

from jax_scout import physics

BRANCH = "IRER_A_FIELD_GEOMETRIC_FEEDBACK_v1_PROTOTYPE"
CONTRACT_KEY = "IRER-SNCGL-CAUSAL-AFFECT-ETDRK4-v1"   # A-on runs (vacuum_ref topology)
RHO_VAC_EFF_FLOOR = 1e-6


def _update_afield(A_k, A_dot_k, psi_k, ops, dt):
    """One causal A-field wave step (mirror solver/core.update_field_of_affect)."""
    psi_real = jnp.fft.ifftn(psi_k)
    rho_real = jnp.maximum(jnp.abs(psi_real) ** 2, ops.rho_floor)
    rho_k = jnp.fft.fftn(rho_real) * ops.dealias_mask
    rho_k = rho_k.at[0, 0, 0].set(0.0)                       # remove DC (total-mass) source
    accel_k = -ops.c_sq_k_sq * A_k + rho_k
    A_dot_k = A_dot_k + accel_k * dt
    A_k = (A_k + A_dot_k * dt) * ops.dealias_mask
    A_k = A_k.at[0, 0, 0].set(0.0)                           # pin gauge zero mode
    A_dot_k = A_dot_k.at[0, 0, 0].set(0.0)
    A_real = jnp.real(jnp.fft.ifftn(A_k))
    return A_k, A_dot_k, A_real


def _simulate_afield(pvec, psi0, gamma_A, N, L, dt, n_steps, rd, cd, topology):
    ops = physics._ops_from_vec(pvec, N, L, dt, rd, cd)
    psi_k = jnp.fft.fftn(psi0) * ops.dealias_mask
    A_k = jnp.zeros((N, N, N), cd)
    A_dot_k = jnp.zeros((N, N, N), cd)

    def body(carry, _):
        psi_k, A_k, A_dot_k = carry
        A_k, A_dot_k, A_real = _update_afield(A_k, A_dot_k, psi_k, ops, dt)
        if topology == "additive_potential":
            # A added to the conformal (Weyl) potential: Omega^2_eff = Omega^2 * exp(gamma_A*A)
            mod = jnp.exp(jnp.clip(gamma_A * A_real, -30.0, 30.0))
            psi_k = physics.step(psi_k, ops, None, mod)
        else:  # vacuum_ref (default): rho_vac_eff = max(rho_vac + gamma_A*A, eps)
            mod = jnp.maximum(ops.rho_vac + gamma_A * A_real, RHO_VAC_EFF_FLOOR)
            psi_k = physics.step(psi_k, ops, mod, None)
        psi = jnp.fft.ifftn(psi_k)
        diag = (jnp.sum(jnp.abs(psi) ** 2), jnp.max(jnp.abs(psi)),
                jnp.sum(A_real ** 2), jnp.max(jnp.abs(A_real)),
                jnp.min(mod), jnp.max(mod))   # mod = rho_vac_eff (vacuum_ref) or Omega^2-mult (additive)
        return (psi_k, A_k, A_dot_k), diag

    half = n_steps // 2
    (psi_k, A_k, A_dot_k), d1 = lax.scan(body, (psi_k, A_k, A_dot_k), None, length=half)
    psi_mid = jnp.fft.ifftn(psi_k); A_mid = jnp.real(jnp.fft.ifftn(A_k))
    (psi_k, A_k, A_dot_k), d2 = lax.scan(body, (psi_k, A_k, A_dot_k), None, length=n_steps - half)
    psi_fin = jnp.fft.ifftn(psi_k); A_fin = jnp.real(jnp.fft.ifftn(A_k))

    cat = lambda i: jnp.concatenate([d1[i], d2[i]])
    energy, max_amp, A_energy, A_max, mod_min, mod_max = (cat(i) for i in range(6))
    finite = jnp.all(jnp.isfinite(jnp.abs(psi_fin))) & jnp.all(jnp.isfinite(A_fin))
    return (psi_mid, psi_fin, A_mid, A_fin, energy, max_amp,
            A_energy, A_max, mod_min, mod_max, finite)


def contract_key_for(topology):
    return ("IRER-SNCGL-ADDITIVE-POT-ETDRK4-v1" if topology == "additive_potential"
            else "IRER-SNCGL-CAUSAL-AFFECT-ETDRK4-v1")


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8, 9))
def simulate_afield(pvec, psi0, gamma_A, N, L, dt, n_steps,
                    real_dtype=jnp.float64, complex_dtype=jnp.complex128,
                    topology="vacuum_ref"):
    """A-coupled trajectory probe. gamma_A TRACED; topology static
    ('vacuum_ref' = rho_vac modulation, 'additive_potential' = Omega^2 * exp(gamma_A*A))."""
    return _simulate_afield(pvec, psi0, gamma_A, N, L, dt, n_steps, real_dtype, complex_dtype, topology)
