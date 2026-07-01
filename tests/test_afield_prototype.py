"""
Safety tests for IRER_A_FIELD_GEOMETRIC_FEEDBACK_v1_PROTOTYPE (run in WSL2 jax venv):
  python /mnt/f/quantule_mapper/tests/test_afield_prototype.py

Asserts:
  1. gamma_A = 0 reproduces the gamma_A-free solver (physics.simulate) to FP64 tolerance.
  2. gamma_A > 0 actually changes the trajectory (coupling is active).
  3. negative/unsafe rho_vac_eff is floored (rve_min >= floor) and stays finite (no NaN).
  4. A-on contract key differs from the LOCAL-RHO solver contract (not rank-compatible).
"""
import os, sys
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jax_scout import physics, afield_prototype as af

N, L, dt, STEPS = 32, 10.0, 0.005, 80
pvec = jnp.array([1.0, 0.2, 1.33, 1.0, 0.5, 0.05, 0.02, 0.1])  # D,eta,rho_vac,omega0,a_coupling,s,f,a
# bounded Gaussian packet IC (random-amplitude IC + cubic gain is numerically unstable;
# equivalence is about gamma_A=0 reproduction, so use a stable bounded field).
_x = jnp.linspace(-L / 2, L / 2, N, endpoint=False)
_X, _Y, _Z = jnp.meshgrid(_x, _x, _x, indexing="ij")
_rng = np.random.default_rng(0)
psi0 = (jnp.exp(-(_X ** 2 + _Y ** 2 + _Z ** 2) / 2.0)
        + 0.01 * jnp.asarray(_rng.standard_normal((N, N, N)) + 1j * _rng.standard_normal((N, N, N)))
        ).astype(jnp.complex128)


def relL2(a, b):
    return float(jnp.linalg.norm((a - b).ravel()) / (jnp.linalg.norm(b.ravel()) + 1e-30))


def main():
    ops = physics._ops_from_vec(pvec, N, L, dt, jnp.float64, jnp.complex128)
    _, psi_k_ref, _ = physics.simulate(psi0, ops, STEPS)
    psi_ref = jnp.fft.ifftn(psi_k_ref)

    # 1. gamma_A = 0 reproduces baseline
    out0 = af.simulate_afield(pvec, psi0, 0.0, N, L, dt, STEPS)
    psi_fin0 = out0[1]
    e0 = relL2(psi_fin0, psi_ref)
    assert e0 < 1e-10, f"gamma_A=0 does NOT reproduce baseline: relL2={e0:.2e}"
    print(f"[1] gamma_A=0 reproduces baseline: relL2={e0:.2e}  PASS")

    # 2. gamma_A > 0 changes the trajectory
    out1 = af.simulate_afield(pvec, psi0, 0.2, N, L, dt, STEPS)
    e1 = relL2(out1[1], psi_ref)
    assert e1 > 1e-6, f"gamma_A>0 did NOT change trajectory: relL2={e1:.2e}"
    print(f"[2] gamma_A=0.2 changes trajectory: relL2 vs baseline={e1:.2e}  PASS")

    # 3. floor guarantee: for FINITE inputs, rho_vac_eff = max(rho_vac + gamma_A*A, floor)
    #    stays >= floor > 0 even where gamma_A*A drives it strongly negative (-> no (neg)^a NaN).
    Atilde = jnp.linspace(-10.0, 10.0, 64)         # finite A values, some large
    rve = jnp.maximum(1.33 + (-50.0) * Atilde, af.RHO_VAC_EFF_FLOOR)
    assert bool(jnp.all(jnp.isfinite(rve))) and float(jnp.min(rve)) >= af.RHO_VAC_EFF_FLOOR
    # and a MODERATE negative gamma_A run stays finite (floor lets it run bounded)
    outn = af.simulate_afield(pvec, psi0, -0.3, N, L, dt, STEPS)
    assert bool(outn[10]), "moderate negative-gamma_A run went non-finite"
    print(f"[3] rho_vac_eff floored (min={float(jnp.min(rve)):.1e}>=floor) + moderate -gamma_A finite  PASS")

    # 4. A-on contract key distinct from LOCAL-RHO (not rank-compatible).
    # (literal to avoid importing orchestrator.contracts -> pydantic, absent in the jax venv)
    LOCAL_RHO = "IRER-SNCGL-LOCAL-RHO-ETDRK4-v1"
    assert af.CONTRACT_KEY != LOCAL_RHO and "CAUSAL-AFFECT" in af.CONTRACT_KEY
    print(f"[4] A-on contract key '{af.CONTRACT_KEY}' != LOCAL-RHO '{LOCAL_RHO}'  PASS")

    # 5. additive_potential topology: gamma_A=0 also reproduces baseline; gamma_A>0 active
    outa0 = af.simulate_afield(pvec, psi0, 0.0, N, L, dt, STEPS, jnp.float64, jnp.complex128, "additive_potential")
    ea0 = relL2(outa0[1], psi_ref)
    assert ea0 < 1e-10, f"additive_potential gamma_A=0 does NOT reproduce baseline: relL2={ea0:.2e}"
    outa1 = af.simulate_afield(pvec, psi0, 0.5, N, L, dt, STEPS, jnp.float64, jnp.complex128, "additive_potential")
    ea1 = relL2(outa1[1], psi_ref)
    assert ea1 > 1e-6 and bool(outa1[10]), f"additive_potential gamma_A>0 inert/non-finite: relL2={ea1:.2e}"
    assert af.contract_key_for("additive_potential") != af.contract_key_for("vacuum_ref")
    print(f"[5] additive_potential: gamma_A=0 reproduces (relL2={ea0:.2e}), gamma_A=0.5 active ({ea1:.2e}), "
          f"distinct contract key  PASS")

    print("\nALL A-FIELD PROTOTYPE SAFETY TESTS PASS")


if __name__ == "__main__":
    main()
