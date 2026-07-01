"""
tests/test_run_identity.py

Unit tests for orchestrator/run_identity.py — the canonical, dependency-light
run-identity contract.  These run anywhere (no cupy / GPU required).

Also contains a numpy mirror of solver.core.update_field_of_affect that proves
the k=0 zero-mode projection actually bounds the affect-field secular runaway.
"""
import os
import sys
import tempfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from orchestrator import run_identity as ri  # noqa: E402


# ---------------------------------------------------------------------------
# Variant / topology / strength derivation
# ---------------------------------------------------------------------------

class TestVariantDerivation:
    def test_baseline_is_local_rho(self):
        params = {"param_D": 1.0}  # no affect coupling
        assert ri.affect_strength_for_params(params) == 0.0
        assert ri.affect_topology_for_params(params) == "none"
        assert ri.variant_label_for_params(params) == "LOCAL-RHO"
        assert ri.solver_contract_version_for_params(params) == "IRER-SNCGL-LOCAL-RHO-ETDRK4-v1"

    def test_zero_coupling_forces_baseline_even_if_topology_declared(self):
        # gamma_A == 0 must never be mislabelled as a coupled variant.
        params = {"param_affect_coupling": 0.0, "param_affect_topology": "vacuum_ref"}
        assert ri.affect_topology_for_params(params) == "none"
        assert ri.variant_label_for_params(params) == "LOCAL-RHO"

    def test_vacuum_ref_is_causal_affect(self):
        params = {"param_affect_coupling": 0.3, "param_affect_topology": "vacuum_ref"}
        assert ri.affect_topology_for_params(params) == "vacuum_ref"
        assert ri.variant_label_for_params(params) == "CAUSAL-AFFECT"
        assert ri.solver_contract_version_for_params(params) == "IRER-SNCGL-CAUSAL-AFFECT-ETDRK4-v1"

    def test_default_topology_when_coupling_on_is_vacuum_ref(self):
        params = {"param_affect_coupling": 0.3}  # topology unspecified
        assert ri.affect_topology_for_params(params) == "vacuum_ref"

    def test_additive_potential_is_comparison_arm(self):
        params = {"param_affect_coupling": 0.3, "param_affect_topology": "additive_potential"}
        assert ri.variant_label_for_params(params) == "ADDITIVE-POT"
        assert ri.solver_contract_version_for_params(params) == "IRER-SNCGL-ADDITIVE-POT-ETDRK4-v1"

    def test_non_numeric_coupling_is_safe(self):
        assert ri.affect_strength_for_params({"param_affect_coupling": "garbage"}) == 0.0


# ---------------------------------------------------------------------------
# build_identity
# ---------------------------------------------------------------------------

class TestBuildIdentity:
    def _identity(self, **over):
        base = dict(
            config_hash="abc123", seed=7, generation=3,
            N_grid=64, dt=0.001, T_steps=250, params={"param_D": 1.0},
            run_id="deadbeefcafe", hunt_name="BURN_IN_001",
            utc_start="2026-06-18T12:00:00Z", gpu_backend="numpy",
        )
        base.update(over)
        return ri.build_identity(**base)

    def test_contains_all_required_fields(self):
        ident = self._identity()
        assert ri.missing_identity_fields(ident) == ()

    def test_types_are_canonical(self):
        ident = self._identity()
        assert isinstance(ident["seed"], int)
        assert isinstance(ident["N_grid"], int)
        assert isinstance(ident["dt"], float)
        assert isinstance(ident["affect_strength"], float)

    def test_baseline_identity_labels(self):
        ident = self._identity()
        assert ident["variant_label"] == "LOCAL-RHO"
        assert ident["affect_topology"] == "none"
        assert ident["affect_strength"] == 0.0
        assert ident["solver_contract_version"] == "IRER-SNCGL-LOCAL-RHO-ETDRK4-v1"

    def test_missing_field_detected(self):
        ident = self._identity()
        del ident["seed"]
        assert "seed" in ri.missing_identity_fields(ident)


# ---------------------------------------------------------------------------
# Compatibility gate
# ---------------------------------------------------------------------------

class TestCompatibilityGate:
    def _ident(self, **over):
        base = dict(
            solver_contract_version="IRER-SNCGL-LOCAL-RHO-ETDRK4-v1",
            variant_label="LOCAL-RHO", affect_topology="none", N_grid=64,
        )
        base.update(over)
        return base

    def test_identical_runs_rankable(self):
        assert ri.are_rankable(self._ident(), self._ident())

    def test_different_grid_not_rankable(self):
        assert not ri.are_rankable(self._ident(N_grid=64), self._ident(N_grid=128))

    def test_different_variant_not_rankable(self):
        a = self._ident()
        b = self._ident(variant_label="CAUSAL-AFFECT", affect_topology="vacuum_ref",
                        solver_contract_version="IRER-SNCGL-CAUSAL-AFFECT-ETDRK4-v1")
        assert not ri.are_rankable(a, b)

    def test_baseline_vs_causal_affect_blocked(self):
        baseline = self._ident()
        causal = self._ident(variant_label="CAUSAL-AFFECT", affect_topology="vacuum_ref",
                             solver_contract_version="IRER-SNCGL-CAUSAL-AFFECT-ETDRK4-v1")
        assert not ri.are_rankable(baseline, causal), (
            "LOCAL-RHO baseline and CAUSAL-AFFECT must never be ranked together."
        )


# ---------------------------------------------------------------------------
# HDF5 /identity group round-trip
# ---------------------------------------------------------------------------

class TestIdentityGroupRoundTrip:
    def test_write_then_read(self):
        h5py = pytest.importorskip("h5py")
        ident = ri.build_identity(
            config_hash="abc123", seed=7, generation=3, N_grid=64, dt=0.001,
            T_steps=250, params={"param_affect_coupling": 0.3, "param_affect_topology": "vacuum_ref"},
            run_id="deadbeefcafe1234", hunt_name="HUNT_X",
            utc_start="2026-06-18T12:00:00Z", gpu_backend="numpy",
        )
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "art.h5")
            with h5py.File(path, "w") as f:
                ri.write_identity_group(f, ident)
            with h5py.File(path, "r") as f:
                back = ri.read_identity_group(f)
        assert back["config_hash"] == "abc123"
        assert back["seed"] == 7
        assert back["N_grid"] == 64
        assert abs(back["dt"] - 0.001) < 1e-12
        assert back["variant_label"] == "CAUSAL-AFFECT"
        assert abs(back["affect_strength"] - 0.3) < 1e-12


# ---------------------------------------------------------------------------
# Provenance naming
# ---------------------------------------------------------------------------

class TestProvenanceNaming:
    def test_legacy_fallback_when_no_discriminators(self):
        assert ri.provenance_filename("HASH") == "provenance_HASH.json"

    def test_unique_when_seed_and_run_id(self):
        name = ri.provenance_filename("HASH", seed=7, run_id="deadbeefcafe", utc_date="2026-06-18")
        assert name.startswith("provenance_HASH_seed7_")
        assert "deadbeef" in name
        assert name.endswith(".json")

    def test_two_seeds_differ(self):
        a = ri.provenance_filename("HASH", seed=1, run_id="aaaaaaaa")
        b = ri.provenance_filename("HASH", seed=2, run_id="bbbbbbbb")
        assert a != b

    def test_path_for_artifact_reads_identity(self):
        h5py = pytest.importorskip("h5py")
        ident = ri.build_identity(
            config_hash="HASH", seed=9, generation=0, N_grid=64, dt=0.001,
            T_steps=10, params={}, run_id="feedface0000", utc_start="2026-06-18T00:00:00Z",
            gpu_backend="numpy",
        )
        with tempfile.TemporaryDirectory() as d:
            art = os.path.join(d, "rho_history_HASH.h5")
            with h5py.File(art, "w") as f:
                ri.write_identity_group(f, ident)
            path = ri.provenance_path_for_artifact(d, art, "HASH")
        base = os.path.basename(path)
        assert base.startswith("provenance_HASH_seed9_")
        assert "feedface" in base

    def test_path_for_artifact_falls_back_without_identity(self):
        h5py = pytest.importorskip("h5py")
        with tempfile.TemporaryDirectory() as d:
            art = os.path.join(d, "rho_history_HASH.h5")
            with h5py.File(art, "w") as f:
                f.create_dataset("psi_final", data=np.zeros(2))
            path = ri.provenance_path_for_artifact(d, art, "HASH")
        assert os.path.basename(path) == "provenance_HASH.json"


# ---------------------------------------------------------------------------
# Runtime probes
# ---------------------------------------------------------------------------

class TestRuntimeProbes:
    def test_backend_is_numpy_without_cupy(self):
        # This box has no cupy; on a GPU box this returns 'cupy'.
        assert ri.detect_backend() in ("numpy", "cupy")

    def test_git_commit_returns_string(self):
        commit = ri.git_commit(ROOT)
        assert isinstance(commit, str) and len(commit) > 0

    def test_sha256_file(self):
        with tempfile.NamedTemporaryFile("wb", delete=False) as fh:
            fh.write(b"hello")
            name = fh.name
        try:
            digest = ri.sha256_file(name)
            assert digest == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        finally:
            os.unlink(name)


# ---------------------------------------------------------------------------
# k=0 secular runaway gate — numpy mirror of solver.core.update_field_of_affect
# ---------------------------------------------------------------------------

def _affect_step(A_k, A_dot_k, rho_k, c_sq_k_sq, dt, gate):
    """One symplectic-Euler step of the affect wave equation, matching core.py.
    If gate is True, applies the k=0 zero-mode projection."""
    rho_src = rho_k.copy()
    if gate:
        ri.zero_dc_mode(rho_src)          # project out total-mass DC source
    accel = -c_sq_k_sq * A_k + rho_src
    A_dot_k = A_dot_k + accel * dt
    A_k = A_k + A_dot_k * dt
    if gate:
        ri.zero_dc_mode(A_k)              # pin gauge: zero mode stays zero
        ri.zero_dc_mode(A_dot_k)
    return A_k, A_dot_k


def _run_affect(gate, steps=400, N=8, dt=0.01):
    rng = np.random.default_rng(0)
    # Density-like positive source with a large DC component (total mass).
    rho_real = 1.0 + 0.1 * rng.standard_normal((N, N, N))
    rho_k = np.fft.fftn(rho_real).astype(np.complex128)
    k = np.fft.fftfreq(N) * 2 * np.pi
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    c_sq_k_sq = 1.0 * (kx**2 + ky**2 + kz**2)
    A_k = np.zeros((N, N, N), dtype=np.complex128)
    A_dot_k = np.zeros((N, N, N), dtype=np.complex128)
    dc = []
    for _ in range(steps):
        A_k, A_dot_k = _affect_step(A_k, A_dot_k, rho_k, c_sq_k_sq, dt, gate)
        dc.append(abs(A_k[0, 0, 0]))
    return np.array(dc)


class TestK0Runaway:
    def test_without_gate_dc_mode_runs_away(self):
        dc = _run_affect(gate=False)
        # No restoring force at k=0 -> quadratic growth -> end >> start.
        assert dc[-1] > 100 * (dc[10] + 1e-9), "DC mode should explode without the gate"

    def test_with_gate_dc_mode_bounded(self):
        dc = _run_affect(gate=True)
        assert np.all(dc < 1e-9), "DC mode must stay identically zero with the gate"

    def test_gate_is_pure_function_on_nonzero_modes(self):
        # zero_dc_mode must only touch [0,0,0], leaving every other mode intact.
        arr = np.arange(2 * 3 * 4, dtype=np.complex128).reshape(2, 3, 4) + 1.0
        before = arr.copy()
        ri.zero_dc_mode(arr)
        assert arr[0, 0, 0] == 0
        before[0, 0, 0] = 0
        assert np.array_equal(arr, before)
