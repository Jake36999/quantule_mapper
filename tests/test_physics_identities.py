"""Physics-identity regression tests — the instrument-integrity guard.

WHY THIS FILE EXISTS. Every bug in the project's instrument-integrity ledger
(docs/IRER_MASTER_HYPOTHESIS_CATALOG.md section 10) was caught the same way: a result
contradicted an identity that must hold regardless of implementation.

  C2.6  a_coupling=0 did not disable geometry -> D_eff = D/151. Caught because a linear
        packet failed to translate, violating a Galilean identity.
  C2.8b peak-tracking gave elasticity e=3.21. Caught because e>1 violates energy conservation.
  C3    boost IC lacked the carrier phase. Caught because v_frac was constant in v.

Those checks were run by hand, occasionally, by whoever remembered. This file runs them on
every commit.

DESIGN RULE: these tests MUST NOT re-implement the physics. A test that recomputes the
operator it is checking can agree with a bug. Every assertion here exercises the SAME code
path the harnesses use, and asserts a mathematical identity that is true independently of how
that code is written.

Fast by construction: synthetic fields, tiny grids, no Q-ball solve, at most a few RK4 steps.
Runs on CPU in seconds. See docs/INFRASTRUCTURE_AND_ORCHESTRATION_ASSESSMENT.md section 2a.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

jax = pytest.importorskip("jax", reason="JAX not installed in this environment")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from jax_scout import gravity_TG_B1S_state_load_feedback_gpu as b1s  # noqa: E402
from jax_scout import gravity_TG_B2_two_node_awell as b2  # noqa: E402
from jax_scout import gravity_TG_B2_midplane_stress_flux as mid  # noqa: E402
from jax_scout.phase_d_c3_wave import build_kg  # noqa: E402

N = 16
L = 8.0
DT = 0.002

BASE = dict(dt=DT, c=0.5477, m=1.0, a=0.8, s=-0.5, f=-0.1,
            alpha_T=0.35, omega_T=1.25, omega_G=0.85, gamma_T=0.08, gamma_G=0.06,
            kappa_TG=0.55, epsilon_G=0.06, cT=0.7, cG=0.55, L=L,
            absorb_width=1.6, absorb_strength=0.02, core_radius=2.0)

FULL = [1.0, 1.0, 1.0, 1.0]     # src, temporal, geometric, feedback
OFF = [1.0, 1.0, 1.0, 0.0]      # feedback disabled -> A must be exactly 1


@pytest.fixture(scope="module")
def grid():
    return b1s.make_grid(build_kg(N, L, BASE["c"], BASE["m"], DT))


@pytest.fixture(scope="module")
def cfgv():
    return b1s.cfg_array(dict(BASE))


def synthetic_state(g, sep=2.0, w=0.964):
    """Two Gaussian blobs plus a non-trivial G field. No Q-ball solve: fast and deterministic."""
    X, Y, Z = g["X"], g["Y"], g["Z"]
    r2r = (X - sep / 2) ** 2 + Y ** 2 + Z ** 2
    r2l = (X + sep / 2) ** 2 + Y ** 2 + Z ** 2
    phi = (jnp.exp(-r2r) + jnp.exp(-r2l)).astype(jnp.complex128)
    pi = -1j * w * phi
    # a deliberately asymmetric G so that d_x A != 0 and the force is non-degenerate
    G = -0.01 * jnp.exp(-0.5 * r2r) - 0.004 * jnp.exp(-0.5 * r2l)
    zero = jnp.zeros_like(G)
    return (phi, pi, zero, zero, G, zero)


# --------------------------------------------------------------------- operators

def test_spectral_derivative_is_exact_for_band_limited_input(grid):
    """deriv_x underlies every gradient, force and energy observable in the project."""
    f = jnp.sin(2 * jnp.pi * grid["X"] / L)
    got = jnp.real(b1s.deriv_x(f, grid))
    want = (2 * jnp.pi / L) * jnp.cos(2 * jnp.pi * grid["X"] / L)
    assert float(jnp.max(jnp.abs(got - want))) < 1e-12


def test_div_A_grad_reduces_to_laplacian_when_A_is_unity(grid):
    """The C2.6 bug class: a geometry operator that does not become the identity when
    geometry is switched off silently rescales the dynamics."""
    f = jnp.sin(2 * jnp.pi * grid["X"] / L) * jnp.cos(2 * jnp.pi * grid["Y"] / L)
    lap = b1s.lap_real(f, grid)
    dag = jnp.real(b1s.div_A_grad(f, jnp.ones_like(f), grid))
    assert float(jnp.max(jnp.abs(lap - dag))) < 1e-12


# --------------------------------------------------------------------- the built-in nulls

def test_feedback_off_gives_unit_A_and_exactly_zero_force(grid, cfgv):
    """The off-arm null. feedback=0 -> A == 1 everywhere -> F_R == 0 EXACTLY (not approximately).

    This is the direct regression test for the C2.6 failure mode: 'off' must really be off.
    """
    st = synthetic_state(grid)
    d = mid.stress_diag(st, cfgv, grid, jnp.asarray(1.0), jnp.asarray(0.0))
    assert float(d["A_min"]) == 1.0
    assert float(d["A_max"]) == 1.0
    assert float(d["F_R"]) == 0.0


def test_feedback_on_gives_non_degenerate_force(grid, cfgv):
    """Guard against the opposite failure: a test suite that passes because nothing happens."""
    st = synthetic_state(grid)
    d = mid.stress_diag(st, cfgv, grid, jnp.asarray(1.0), jnp.asarray(1.0))
    assert float(d["A_min"]) < 1.0
    assert abs(float(d["F_R"])) > 0.0


def test_force_reverses_sign_with_coupling_polarity(grid, cfgv):
    """A-well vs A-hill. In the linear-response regime the response must be odd in a_sign."""
    st = synthetic_state(grid)
    well = float(mid.stress_diag(st, cfgv, grid, jnp.asarray(+1.0), jnp.asarray(1.0))["F_R"])
    hill = float(mid.stress_diag(st, cfgv, grid, jnp.asarray(-1.0), jnp.asarray(1.0))["F_R"])
    assert well * hill < 0.0
    assert abs(well + hill) <= 1e-3 * max(abs(well), abs(hill))


# --------------------------------------------------------------------- observable consistency

def test_energy_decomposition_sums_to_the_total(grid, cfgv):
    """P1-a: E_kin + E_grad + E_mass + E_pot must equal E_R, or the split is meaningless."""
    st = synthetic_state(grid)
    d = mid.stress_diag(st, cfgv, grid, jnp.asarray(1.0), jnp.asarray(1.0))
    parts = sum(float(d[k]) for k in ("E_kin_R", "E_grad_R", "E_mass_R", "E_pot_R"))
    total = float(d["E_R"])
    assert abs(parts - total) <= 1e-10 * max(abs(total), 1.0)


def test_mass_observable_equals_mass_energy_term_at_unit_m(grid, cfgv):
    """M_R = Int rho and E_mass = Int m^2 rho, so at m=1 they must coincide.

    This pins down what F/M_p has actually been dividing by (P1-a section 3).
    """
    assert BASE["m"] == 1.0
    st = synthetic_state(grid)
    d = mid.stress_diag(st, cfgv, grid, jnp.asarray(1.0), jnp.asarray(1.0))
    assert abs(float(d["M_R"]) - float(d["E_mass_R"])) <= 1e-10 * abs(float(d["M_R"]))


def test_two_flux_variants_agree_to_discretisation_order(grid, cfgv):
    """P2 computes the momentum flux two ways: interface-interpolated planes, and the
    spectral divergence over the same mask. They estimate the same quantity, so they must
    agree to the discretisation error and not by orders of magnitude."""
    st = synthetic_state(grid)
    d = mid.stress_diag(st, cfgv, grid, jnp.asarray(1.0), jnp.asarray(1.0))
    a, b = float(d["flux_plane"]), float(d["flux_div"])
    scale = max(abs(a), abs(b), 1e-30)
    assert abs(a - b) / scale < 0.5


# --------------------------------------------------------------------- dynamics

def test_geometry_off_dynamics_are_independent_of_coupling_polarity(grid, cfgv):
    """With feedback off, a_sign cannot matter -- A is 1 either way.

    If this ever fails, an 'off' switch is leaking, which is exactly the C2.6 signature.
    """
    st = synthetic_state(grid)
    refsv = b1s.refs_array({"reference_energy_max": 1.0, "reference_charge_max": 1.0,
                             "source_global_norm_S0": 1.0})
    fl = jnp.asarray(OFF, dtype=jnp.float64)
    r_p = b2.rhs_2n(st, cfgv, refsv, grid, fl, jnp.asarray(+1.0))
    r_m = b2.rhs_2n(st, cfgv, refsv, grid, fl, jnp.asarray(-1.0))
    for x, y in zip(r_p, r_m):
        assert float(jnp.max(jnp.abs(x - y))) == 0.0


def test_u1_charge_is_conserved_under_short_evolution(grid, cfgv):
    """The KG sector carries a conserved U(1) charge. Drift here means the integrator or the
    coupling is not doing what it claims -- the C3 boost-IC bug surfaced this way."""
    st = synthetic_state(grid)
    refsv = b1s.refs_array({"reference_energy_max": 1.0, "reference_charge_max": 1.0,
                             "source_global_norm_S0": 1.0})
    fl = jnp.asarray(FULL, dtype=jnp.float64)

    def charge(s):
        return float(jnp.imag(jnp.sum(jnp.conj(s[0]) * s[1])) * grid["dx"] ** 3)

    q0 = charge(st)
    ev = b2.evolve_2n(st, cfgv, refsv, grid, fl, jnp.asarray(1.0), 25)
    q1 = charge(ev)
    assert abs(q1 - q0) / max(abs(q0), 1e-30) < 1e-6


def test_evolution_stays_finite(grid, cfgv):
    """A cheap NaN/blow-up canary on the frozen dynamics."""
    st = synthetic_state(grid)
    refsv = b1s.refs_array({"reference_energy_max": 1.0, "reference_charge_max": 1.0,
                             "source_global_norm_S0": 1.0})
    ev = b2.evolve_2n(st, cfgv, refsv, grid, jnp.asarray(FULL, dtype=jnp.float64),
                      jnp.asarray(1.0), 50)
    for field in ev:
        assert bool(jnp.all(jnp.isfinite(field)))


# ===================================================================================
# Value-pinning identities (added 2026-09-16, plan item H3)
#
# WHY THESE EXIST. tools/mutation_probe.py injected ten mutations shaped like the bugs in this
# project's own integrity ledger, and SEVEN survived the suite above -- including a global flip of
# the coupling polarity, in a sector whose single largest open problem IS a sign.
#
# The cause was structural, not an oversight. Every test above pins a SYMMETRY: off-is-off,
# antisymmetry in a_sign, charge conservation, finiteness. A symmetry is invariant under exactly
# the mutations that matter here -- flipping a_sign globally swaps the well and hill arms, so
# "well * hill < 0" still holds; scaling the coupling by 1/151 preserves antisymmetry and charge
# conservation both.
#
# These tests pin VALUES against independently-derived expressions, which is the missing half.
# ===================================================================================

GEOM_OFF = [1.0, 1.0, 0.0, 1.0]   # geometry disabled while feedback stays ON -- the C2.6 shape


def test_positive_polarity_is_a_well_and_negative_is_a_hill(grid, cfgv):
    """The ABSOLUTE polarity, which the antisymmetry test above cannot see.

    `A = exp(a_sign * eps_G * G)`, and the module's whole naming rests on which way that bends:
    a_sign=+1 is the A-WELL (theory-faithful, predicts attraction), a_sign=-1 the A-HILL (the
    frozen TG-B1S polarity, the sign control). A well has A <= 1 everywhere and a hill A >= 1,
    so `A_max == 1` and `A_min == 1` are exact statements about which one we built.

    Flipping the coupling globally swaps these and is caught here, where
    `test_force_reverses_sign_with_coupling_polarity` passes regardless because it only ever
    compares the two arms against each other.
    """
    st = synthetic_state(grid)
    well = mid.stress_diag(st, cfgv, grid, jnp.asarray(+1.0), jnp.asarray(1.0))
    hill = mid.stress_diag(st, cfgv, grid, jnp.asarray(-1.0), jnp.asarray(1.0))
    # `<= 1 + tiny` rather than `== 1`: the synthetic G is non-positive but its maximum is a few
    # times 1e-13 below zero, so A_max is 1 - 3e-13 rather than exactly 1. The physical claim is
    # one-sidedness, and the depth assertions below are what make it non-vacuous.
    tiny = 1e-9
    assert float(well["A_max"]) <= 1.0 + tiny, "a_sign=+1 must be a WELL: A never exceeds 1"
    assert float(well["A_min"]) < 1.0 - tiny, "a_sign=+1 must actually dip below 1"
    assert float(hill["A_min"]) >= 1.0 - tiny, "a_sign=-1 must be a HILL: A never falls below 1"
    assert float(hill["A_max"]) > 1.0 + tiny, "a_sign=-1 must actually rise above 1"


def test_geometry_flag_off_keeps_polarity_out_of_the_dynamics(grid, cfgv):
    """The C2.6 shape proper: a flag that does not take effect.

    Every other test that varies the flags uses OFF = [1,1,1,0], which disables FEEDBACK and
    leaves the geometry flag at 1 -- so nothing above ever exercises geom_en = 0, and dropping it
    from the A expression changes no result in the suite. C2.6 was precisely a geometry switch
    that did not switch, and it cost five campaigns of retracted verdicts.
    """
    st = synthetic_state(grid)
    refsv = b1s.refs_array({"reference_energy_max": 1.0, "reference_charge_max": 1.0,
                            "source_global_norm_S0": 1.0})
    fl = jnp.asarray(GEOM_OFF, dtype=jnp.float64)
    r_p = b2.rhs_2n(st, cfgv, refsv, grid, fl, jnp.asarray(+1.0))
    r_m = b2.rhs_2n(st, cfgv, refsv, grid, fl, jnp.asarray(-1.0))
    for x, y in zip(r_p, r_m):
        assert float(jnp.max(jnp.abs(x - y))) == 0.0, (
            "geometry is off, so the coupling polarity cannot reach the dynamics")
    # and with geometry off the G sector must not evolve at all
    _phi_t, _pi_t, _T_t, _VT_t, G_t, VG_t = r_p
    assert float(jnp.max(jnp.abs(G_t))) == 0.0
    assert float(jnp.max(jnp.abs(VG_t))) == 0.0


def test_force_matches_an_independent_linear_response_estimate(grid, cfgv):
    """Pins the MAGNITUDE, which nothing above does.

    `A_well_min ~ 0.99993` means this sector is deep in linear response, so
    `A = exp(a*eps*G) = 1 + a*eps*G` to five digits and therefore

        F_R = -c^2 ∫_{x>0} (dA/dx) |grad phi|^2 dV  ->  -c^2 * a * eps * ∫_{x>0} (dG/dx) |grad phi|^2 dV

    The right-hand side contains no exponential and is evaluated here from the raw fields, so it
    is an independent expression rather than a re-run of the same code -- which is what the design
    rule for this file requires. Agreement is ~0.03%, the size of the neglected quadratic term.

    Any constant rescaling of the coupling (the C2.6 `D_eff = D/151` shape) breaks this by that
    same constant, and a polarity flip breaks the sign.
    """
    st = synthetic_state(grid)
    phi, _pi, _T, _VT, G, _VG = st
    c, eps = BASE["c"], BASE["epsilon_G"]
    dGdx = jnp.real(jnp.fft.ifftn(grid["ikx"] * jnp.fft.fftn(G + 0j)))
    grad2 = sum(jnp.abs(jnp.fft.ifftn(grid[k] * jnp.fft.fftn(phi))) ** 2
                for k in ("ikx", "iky", "ikz"))
    half = grid["X"] > 0
    for a_sign in (+1.0, -1.0):
        got = float(mid.stress_diag(st, cfgv, grid, jnp.asarray(a_sign),
                                    jnp.asarray(1.0))["F_R"])
        want = float(-c * c * a_sign * eps
                     * jnp.sum(jnp.where(half, dGdx * grad2, 0.0)) * grid["dx"] ** 3)
        assert got * want > 0.0, "linear-response estimate disagrees in SIGN"
        assert abs(got - want) <= 1e-2 * abs(want), (
            "F_R=%.6e vs linear response %.6e" % (got, want))


def test_kg_dispersion_and_propagator_match_the_analytic_relation():
    """`build_kg` is used by every harness here and was pinned by nothing.

    The Klein-Gordon dispersion is omega_k^2 = c^2 k^2 + m^2 and the exact free propagator over one
    step is cos(omega dt) -- both closed-form, so this is a statement of the physics rather than a
    second implementation of it. Dropping the mass term or detuning the step passes everything
    above.
    """
    c, m, dt = 0.5477, 1.0, DT
    kg = build_kg(16, 8.0, c, m, dt)
    w = np.asarray(kg["w"])
    k_sq = np.asarray(kg["k_sq"])
    assert np.max(np.abs(w ** 2 - (c * c * k_sq + m * m))) < 1e-12, "omega^2 != c^2 k^2 + m^2"
    assert np.max(np.abs(np.asarray(kg["C"]) - np.cos(w * dt))) < 1e-14, "C != cos(omega dt)"
    assert np.max(np.abs(np.asarray(kg["wS"]) - w * np.sin(w * dt))) < 1e-14
    assert np.max(np.abs(np.asarray(kg["Sw"]) - np.sin(w * dt) / w)) < 1e-14
    # The w -> 0 limit of sin(w dt)/w is dt, and getting it wrong is a silent zero-mode bug.
    # It needs m = 0 to reach: at k = 0 with m = 1 the frequency is 1, not 0, which is why the
    # k=0 entry of the massive grid is sin(dt)/1 and not dt.
    with np.errstate(invalid="ignore", divide="ignore"):
        # np.where evaluates both branches, so the k=0 entry computes 0/0 before being discarded.
        # That is the existing (correct) implementation; the warning is noise, not a defect.
        massless = build_kg(16, 8.0, c, 0.0, dt)
    assert abs(float(np.asarray(massless["Sw"]).ravel()[0]) - dt) < 1e-12
