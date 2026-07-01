"""
tests/test_data_contract.py

Data contract audit and regression tests.

Covers:
  - HDF5 schema completeness (solver/run.py output)
  - Ledger schema discriminator columns (orchestrator/schema_utils.py)
  - Solver contract correctness (orchestrator/contracts.py)
  - Provenance uniqueness requirement
  - A-coupling topology: default-off, contract-versioned
  - Phase-centering and param_rho_vac degenerate-geometry detection
  - Static regression: LOCAL-RHO path writes expected contract fields
"""
import ast
import json
import os
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SOLVER_RUN = ROOT / "solver" / "run.py"
SOLVER_CORE = ROOT / "solver" / "core.py"
CONTRACTS = ROOT / "orchestrator" / "contracts.py"
SCHEMA_UTILS = ROOT / "orchestrator" / "schema_utils.py"
VALIDATION_PIPELINE = ROOT / "validation_pipeline.py"
BURN_IN_CONFIG = ROOT / "burn_in_config.json"


def _src(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# ===========================================================================
# 1. Solver contract fields
# ===========================================================================

class TestSolverContractFields:
    """solver/run.py must stamp a solver_contract dataset with the required keys."""

    def test_contract_version_constant_imported(self):
        src = _src(SOLVER_RUN)
        assert "from orchestrator.contracts import SOLVER_CONTRACT_VERSION" in src

    def test_solver_contract_json_written(self):
        src = _src(SOLVER_RUN)
        assert '"solver_contract_version"' in src

    def test_contract_declares_geometry_source(self):
        src = _src(SOLVER_RUN)
        assert '"geometry_source"' in src
        assert '"local_stage_rho"' in src

    def test_contract_declares_auxiliary_geometry_false(self):
        src = _src(SOLVER_RUN)
        assert '"auxiliary_geometry": False' in src

    def test_contract_declares_linear_operator(self):
        src = _src(SOLVER_RUN)
        assert '"linear_operator"' in src

    def test_contracts_module_defines_version_constant(self):
        src = _src(CONTRACTS)
        assert 'SOLVER_CONTRACT_VERSION = "IRER-SNCGL-LOCAL-RHO-ETDRK4-v1"' in src

    def test_contracts_module_defines_provisional_threshold(self):
        src = _src(CONTRACTS)
        assert "PROVISIONAL_SSE_THRESHOLD" in src

    def test_result_states_includes_physics_uncertified(self):
        src = _src(CONTRACTS)
        assert '"PHYSICS_UNCERTIFIED"' in src

    def test_result_processor_rejects_wrong_contract_version(self):
        src = _src(ROOT / "orchestrator" / "result_processor.py")
        assert "PHYSICS_UNCERTIFIED" in src
        assert "contract_version != SOLVER_CONTRACT_VERSION" in src


# ===========================================================================
# 2. HDF5 dataset presence (static — checks write calls, not execution)
# ===========================================================================

class TestHDF5DatasetPresence:
    """solver/run.py must create all required datasets on the SUCCESS path."""

    REQUIRED_DATASETS = [
        "psi_final",
        "omega_sq_final",
        "A_final",
        "A_dot_k_final",
        "solver_contract",
        "telemetry",
        "extended_telemetry",
    ]

    def test_all_required_datasets_created(self):
        src = _src(SOLVER_RUN)
        for ds in self.REQUIRED_DATASETS:
            assert f"'{ds}'" in src or f'"{ds}"' in src, \
                f"Dataset '{ds}' not found in solver/run.py"

    def test_fail_path_writes_sentinel_code(self):
        src = _src(SOLVER_RUN)
        assert "'sentinel_code'" in src
        assert "'sentinel_reason'" in src

    def test_energy_sparkline_dataset_present(self):
        src = _src(SOLVER_RUN)
        assert "'energy_sparkline'" in src

    def test_c_invariant_dataset_present(self):
        src = _src(SOLVER_RUN)
        assert "'C_invariant'" in src


# ===========================================================================
# 3. A_dot_final spectral-space label issue (known gap from DATA_CONTRACT.md)
# ===========================================================================

class TestADotFinalLabel:
    """
    The spectral-space affect velocity is now stored as A_dot_k_final so the
    dataset name makes the k-space nature explicit (was the misleading
    A_dot_final).  See DATA_CONTRACT.md §3.2.
    """

    def test_a_dot_dataset_is_k_space_named(self):
        src = _src(SOLVER_RUN)
        assert "A_dot_k_final = solver.A_dot_k" in src
        assert "'A_dot_k_final'" in src
        # The misleading real-space-sounding name must be gone.
        assert "'A_dot_final'" not in src, "Legacy A_dot_final dataset name must be removed."


class TestIdentityGroup:
    """solver/run.py must stamp an /identity group on both SUCCESS and FAIL paths."""

    def test_identity_group_written_via_canonical_helper(self):
        src = _src(SOLVER_RUN)
        assert "from orchestrator.run_identity import write_identity_group" in src
        # Must be invoked on both the FAIL snapshot and the SUCCESS artifact.
        assert src.count("write_identity_group(f, identity)") >= 2, (
            "write_identity_group must be called on both the FAIL and SUCCESS HDF5 paths."
        )

    def test_run_simulation_accepts_identity_param(self):
        src = _src(SOLVER_RUN)
        assert "identity=None," in src

    def test_shim_builds_identity(self):
        src = _src(ROOT / "worker_cupy.py")
        assert "from orchestrator.run_identity import build_identity" in src
        assert "identity=identity," in src


# ===========================================================================
# 4. A-coupling: default-off and contract-versioned
# ===========================================================================

class TestAffectCouplingDefaultOff:
    """
    The A-coupling must be default-off (param_affect_coupling = 0.0 or absent)
    so that γ_A = 0 exactly reproduces the LOCAL-RHO contract.
    """

    def test_n_op_does_not_use_a_real(self):
        src = _src(SOLVER_CORE)
        # After 2026-05-01 stabilisation, A_real must NOT appear in N_op geometry path.
        # The planned A-coupling modulates rho_vac, not rho directly.
        # Check N_op method boundaries:
        n_op_start = src.find("def N_op(")
        step_start = src.find("def step(")
        n_op_body = src[n_op_start:step_start] if n_op_start != -1 and step_start != -1 else ""
        assert "self.A_real" not in n_op_body, (
            "A_real must not appear in N_op body until A-coupling is implemented. "
            "See project_affect_coupling_decision.md for the approved design."
        )

    def test_geometry_source_is_local_rho_only(self):
        src = _src(SOLVER_CORE)
        # N_op must call derive_stable_conformal_factor_with_gradient on self.rho
        assert "derive_stable_conformal_factor_with_gradient(\n            self.rho" in src

    def test_a_field_evolved_each_step(self):
        src = _src(SOLVER_RUN)
        assert "solver.update_field_of_affect(rho_k, dt)" in src

    def test_a_field_update_before_etdrk4_step(self):
        src = _src(SOLVER_RUN)
        update_idx = src.find("solver.update_field_of_affect(rho_k, dt)")
        step_idx = src.find("psi_k = solver.step(psi_k)")
        assert update_idx != -1, "update_field_of_affect call missing"
        assert step_idx != -1, "solver.step call missing"
        assert update_idx < step_idx, "update_field_of_affect must precede solver.step"

    def test_causal_affect_contract_name_defined_or_referenced(self):
        # The new contract version string must be documented somewhere in the codebase.
        # At minimum it must appear in contracts.py or result_processor.py.
        contracts_src = _src(CONTRACTS)
        result_src = _src(ROOT / "orchestrator" / "result_processor.py")
        combined = contracts_src + result_src
        # Either the constant is defined or explicitly referenced in a comment/check.
        # If not yet implemented, this will fail and remind us to add it.
        causal_str = "CAUSAL-AFFECT"
        assert causal_str in combined or causal_str in _src(SOLVER_RUN), (
            f"'{causal_str}' contract version not referenced anywhere. "
            "Add CAUSAL_AFFECT_CONTRACT_VERSION to orchestrator/contracts.py before implementing A-coupling."
        )


# ===========================================================================
# 5. Ledger schema — discriminator columns
# ===========================================================================

class TestLedgerSchemaDiscriminators:
    """
    The ledger must have columns that uniquely identify a run variant.
    These tests check the schema_utils.py CREATE TABLE and _safe_add_columns calls.
    """

    def test_runs_table_has_solver_contract_version_migration(self):
        src = _src(SCHEMA_UTILS)
        # solver_contract_version must be added to metrics (currently there)
        assert '"solver_contract_version"' in src or "'solver_contract_version'" in src

    def test_metrics_table_has_refinement_status(self):
        src = _src(SCHEMA_UTILS)
        assert "refinement_status" in src

    def test_parameters_table_present(self):
        src = _src(SCHEMA_UTILS)
        assert "CREATE TABLE IF NOT EXISTS parameters" in src

    def test_parameters_table_has_rho_vac(self):
        src = _src(SCHEMA_UTILS)
        assert "param_rho_vac" in src

    def test_result_processor_writes_solver_contract_version_to_metrics(self):
        src = _src(ROOT / "orchestrator" / "result_processor.py")
        assert "solver_contract_version" in src
        assert "metrics" in src


# ===========================================================================
# 6. Provenance naming collision (documented gap)
# ===========================================================================

class TestProvenanceNaming:
    """
    The current provenance filename provenance_{config_hash}.json can be
    overwritten if the same params are run again or with a different seed.
    This test documents the gap and will need updating when the naming is fixed.
    """

    def test_provenance_naming_is_collision_free(self):
        # The single-hash naming must be gone from both writer and readers.
        rp_src = _src(ROOT / "orchestrator" / "result_processor.py")
        vp_src = _src(VALIDATION_PIPELINE)
        assert 'f"provenance_{config_hash}.json"' not in rp_src, (
            "result_processor must not hard-code provenance_{config_hash}.json — "
            "use ri.provenance_path_for_artifact / pipeline.provenance_path."
        )
        assert "provenance_path_for_artifact" in rp_src
        # validation_pipeline writer uses the shared collision-free helper.
        assert "provenance_path_for_artifact(self.output_dir, self.input_path" in vp_src

    def test_collision_free_naming_centralized_in_run_identity(self):
        from orchestrator.run_identity import provenance_filename
        a = provenance_filename("HASH", seed=1, run_id="aaaaaaaa")
        b = provenance_filename("HASH", seed=2, run_id="bbbbbbbb")
        assert a != b, "different seeds must yield different provenance filenames"
        assert provenance_filename("HASH") == "provenance_HASH.json", "legacy fallback intact"

    def test_validation_pipeline_schema_version_constant(self):
        src = _src(VALIDATION_PIPELINE)
        assert "SCHEMA_VERSION" in src
        assert "SFP-v" in src


# ===========================================================================
# 7. param_rho_vac degenerate geometry detection
# ===========================================================================

class TestRhoVacDegenerateGeometry:
    """
    param_rho_vac = 0 produces Ω² = 0 everywhere (only saved by conformal floor).
    The current hunt bounds allow [0.0, 2.0] which is dangerous.
    """

    def test_burn_in_config_rho_vac_lower_bound_nondegenerate(self):
        with open(BURN_IN_CONFIG, encoding="utf-8") as f:
            config = json.load(f)
        low = config["bounds"]["param_rho_vac"][0]
        # Degenerate geometry (rho_vac -> 0 => Omega^2 -> 0 everywhere) is now
        # excluded by the search bound. See IRER_MATH_SANITY_CHECK §7.2.
        assert low >= 0.05, f"param_rho_vac lower bound {low} must be >= 0.05 (non-degenerate geometry)"

    def test_burn_in_config_upper_bound_reasonable(self):
        with open(BURN_IN_CONFIG, encoding="utf-8") as f:
            config = json.load(f)
        high = config["bounds"]["param_rho_vac"][1]
        assert 0.0 < high <= 10.0, f"param_rho_vac upper bound {high} looks unreasonable"

    def test_unified_omega_handles_rho_vac_zero(self):
        src = _src(ROOT / "gravity" / "unified_omega.py")
        # Must have a density floor or the division (rho_vac/rho)^a would be 0^a = 0
        assert "maximum" in src or "epsilon" in src.lower() or "floor" in src.lower()

    def test_rho_vac_default_conflict_resolved(self):
        omega_src = _src(ROOT / "gravity" / "unified_omega.py")
        core_src = _src(SOLVER_CORE)
        # Both modules now agree on the canonical default (1.0); core imports it
        # from orchestrator.contracts instead of defaulting to 0.0.
        assert 'param_rho_vac", 1.0' in omega_src
        assert "from orchestrator.contracts import DEFAULT_PARAM_RHO_VAC, DEFAULT_PARAM_OMEGA0" in core_src
        assert "'param_rho_vac', 0.0)" not in core_src, "core must no longer default rho_vac to 0.0"


class TestRhoVacOmega0Split:
    """param_rho_vac (geometry) and param_omega0 (oscillator) are now decoupled."""

    def test_core_uses_omega0_in_linear_operator(self):
        src = _src(SOLVER_CORE)
        assert "1j * self.omega0" in src
        assert "1j * self.rho_vac" not in src, "L_k oscillator term must use omega0, not rho_vac"

    def test_omega0_defaults_to_rho_vac_for_backcompat(self):
        src = _src(SOLVER_CORE)
        assert "self.omega0 = params.get('param_omega0', params.get('param_rho_vac'" in src

    def test_geometry_still_uses_rho_vac(self):
        src = _src(ROOT / "gravity" / "unified_omega.py")
        assert "(rho_vac / rho_capped) ** a" in src

    def test_contracts_define_canonical_defaults(self):
        src = _src(CONTRACTS)
        assert "DEFAULT_PARAM_RHO_VAC = 1.0" in src
        assert "DEFAULT_PARAM_OMEGA0 = 1.0" in src

    def test_burn_in_has_omega0_bounds(self):
        with open(BURN_IN_CONFIG, encoding="utf-8") as f:
            config = json.load(f)
        assert "param_omega0" in config["bounds"]


# ===========================================================================
# 8. Phase centering — documented gap
# ===========================================================================

class TestPhaseCentering:
    """
    solver/run.py applies global phase centering every 50 steps, removing the
    phase accumulated from the i*rho_vac vacuum oscillator term.
    This test documents the behaviour so any removal/change is caught.
    """

    def test_phase_centering_present_in_run_loop(self):
        src = _src(SOLVER_RUN)
        assert "mean_phase" in src
        assert "cp.exp(-1j * mean_phase)" in src

    def test_phase_centering_every_50_steps(self):
        src = _src(SOLVER_RUN)
        assert "step % 50 == 0" in src


# ===========================================================================
# 9. Config hash determinism
# ===========================================================================

class TestConfigHashDeterminism:
    """config_hash must be a deterministic function of params (no random UUID mixing)."""

    def test_config_hash_uses_sha256_of_sorted_json(self):
        src = _src(CONTRACTS)
        assert "sha256" in src
        assert "sort_keys=True" in src

    def test_job_id_is_uuid_prefix(self):
        src = _src(CONTRACTS)
        assert "uuid4" in src or "uuid.uuid4" in src

    def test_job_id_and_config_hash_are_distinct_fields(self):
        src = _src(CONTRACTS)
        assert "job_id" in src
        assert "config_hash" in src
        # Both must coexist — job_id is per-run, config_hash is per-params
        job_idx = src.find("job_id")
        hash_idx = src.find("config_hash")
        assert job_idx != -1 and hash_idx != -1 and job_idx != hash_idx


# ===========================================================================
# 10. Sentinel codes
# ===========================================================================

class TestSentinelCodes:
    """The three sentinel codes must be stable — they're read by validation_pipeline."""

    def test_math_explosion_sentinel_1002(self):
        src = _src(SOLVER_RUN)
        assert "1002" in src
        assert "math_explosion" in src

    def test_physics_drift_sentinel_1003(self):
        src = _src(SOLVER_RUN)
        assert "1003" in src
        assert "physics_drift" in src

    def test_geometry_sanity_sentinel_1004(self):
        src = _src(SOLVER_RUN)
        assert "1004" in src
        assert "geometry_sanity" in src

    def test_sentinel_code_dataset_written_on_fail(self):
        src = _src(SOLVER_RUN)
        assert "'sentinel_code'" in src
        assert "sentinel_code != -1.0" in src or "sentinel_code == -1.0" in src


# ===========================================================================
# 11. K=0 secular runaway in A field (must be fixed before A-coupling)
# ===========================================================================

class TestAFieldK0Runaway:
    """
    The A wave equation has a k=0 (DC) mode that grows secularly as a random walk
    driven by ∫ρ. This must be zeroed each step before A-coupling is enabled.
    Per project_affect_coupling_decision.md this fix is required regardless of coupling path.
    """

    def _update_field_body(self):
        src = _src(SOLVER_CORE)
        method_start = src.find("def update_field_of_affect(")
        next_method = src.find("\n    def ", method_start + 1)
        return src[method_start:next_method] if next_method != -1 else src[method_start:]

    def test_k0_zeroing_present_in_update_field_of_affect(self):
        body = self._update_field_body()
        # The gate must (a) zero the DC source and (b) pin A_k / A_dot_k DC modes.
        assert body.count("[0, 0, 0] = 0") >= 3, (
            "update_field_of_affect must zero the rho_k DC source AND pin "
            "A_k[0,0,0] and A_dot_k[0,0,0]. See orchestrator.run_identity.zero_dc_mode."
        )
        assert "self.A_k[0, 0, 0] = 0" in body
        assert "self.A_dot_k[0, 0, 0] = 0" in body

    def test_k0_gate_value_correctness_proof_exists(self):
        # The boundedness proof is a runnable numpy mirror, GPU-independent.
        proof = (ROOT / "tests" / "test_run_identity.py").read_text(encoding="utf-8")
        assert "class TestK0Runaway" in proof
        assert "test_with_gate_dc_mode_bounded" in proof

    def test_canonical_zero_dc_mode_helper_exists(self):
        from orchestrator.run_identity import zero_dc_mode
        import numpy as np
        arr = np.ones((2, 2, 2), dtype=np.complex128)
        zero_dc_mode(arr)
        assert arr[0, 0, 0] == 0 and arr[1, 1, 1] == 1
