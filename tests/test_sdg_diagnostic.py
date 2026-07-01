"""
Falsifiability + safety tests for the emergent-geometry/SDG diagnostic layer
(IRER-SDG-DIAG-v1) and the A-field readiness gate.

A diagnostic that cannot be wrong is not science: this asserts the diagnostic
DISCRIMINATES coherent structure from noise (different verdicts + metrics moving the
expected way), is contract-stamped, and confirms the A-field coupling (gamma_A) is
default-OFF so it stays reserved until deliberately enabled.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jax_scout import geometry_diag as gd

PARAMS = {"param_rho_vac": 1.33, "param_a_coupling": 0.5}
N, L = 32, 10.0


def _field(kind):
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    if kind == "condensate":
        return np.exp(-(X ** 2 + Y ** 2 + Z ** 2) / 0.8).astype(np.complex128)
    rng = np.random.default_rng(0)
    return (0.3 * (rng.standard_normal((N, N, N)) + 1j * rng.standard_normal((N, N, N)))).astype(np.complex128)


def test_diagnostic_contract_stamped():
    assert gd.DIAG_CONTRACT_VERSION == "IRER-SDG-DIAG-v1"
    d = gd.diagnose(_field("condensate"), PARAMS, L / N)
    assert d["diag_contract_version"] == "IRER-SDG-DIAG-v1"


def test_diagnostic_is_falsifiable_coherent_vs_noise():
    dc = gd.diagnose(_field("condensate"), PARAMS, L / N)
    dn = gd.diagnose(_field("noise"), PARAMS, L / N)
    # must reach different verdicts (otherwise the diagnostic cannot fail anything)
    assert gd.geometry_verdict(dc) == "geometry_follows_RD_bounded"
    assert gd.geometry_verdict(dn) != "geometry_follows_RD_bounded"
    # discriminating metrics move in the physically expected direction
    assert dc["phase_coherence_nodes"] > dn["phase_coherence_nodes"]
    assert dc["curvature_max"] < dn["curvature_max"]
    assert dc["current_circulation_l2"] < dn["current_circulation_l2"]
    assert dc["sdg_h_norm_l2"] < dn["sdg_h_norm_l2"]


def test_a_field_coupling_default_off():
    """gamma_A must be 0 unless explicitly set -> coupling stays RESERVED by default."""
    import orchestrator.run_identity as ri
    assert ri.affect_strength_for_params({}) == 0.0
    assert ri.affect_strength_for_params({"param_affect_coupling": 0.0}) == 0.0
    assert ri.affect_strength_for_params({"param_affect_coupling": "garbage"}) == 0.0
