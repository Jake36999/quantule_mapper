import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
# Solver was modularized out of worker_cupy.py into the solver/ package.
CORE = ROOT / "solver" / "core.py"   # ETDRK4Solver (buffer-owning core)
RUN = ROOT / "solver" / "run.py"     # initialize_psi + run_simulation


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_auxiliary_wave_state_buffers_present():
    source = _read(CORE)
    assert "self.A_real = cp.zeros" in source
    assert "self.A_k = cp.zeros" in source
    assert "self.A_dot_k = cp.zeros" in source
    assert "self.c_affect = cp.float64" in source
    assert "self.c_sq_k_sq = (self.c_affect ** 2) * self.k_sq" in source


def test_update_field_of_affect_method_exists():
    tree = ast.parse(_read(CORE))
    method_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    assert "update_field_of_affect" in method_names


def test_run_loop_updates_aux_field_before_step():
    source = _read(RUN)
    update_idx = source.find("solver.update_field_of_affect(rho_k, dt)")
    step_idx = source.find("psi_k = solver.step(psi_k)")
    assert update_idx != -1, "Missing per-step auxiliary field update call"
    assert step_idx != -1, "Missing ETDRK4 step call"
    assert update_idx < step_idx, "Auxiliary field must update before solver.step each timestep"


def test_n_op_uses_local_rho_for_geometry_source():
    # Post-stabilisation (2026-05-01, IRER Stabilisation Baseline) the conformal
    # geometry is sourced from local stage density rho, NOT the affect field A.
    # The planned A-coupling modulates the vacuum reference rho_vac and keeps rho
    # in the denominator, so this assertion will be revisited in the coupling phase.
    source = _read(CORE)
    assert "self.A_real[:] = self.ifft_single(self.A_k).real.astype(cp.float64, copy=False)" in source
    assert (
        "derive_stable_conformal_factor_with_gradient(\n            self.rho, self._simulation_geometry_params"
        in source
    )


def test_artifact_persists_auxiliary_wave_fields():
    source = _read(RUN)
    assert "A_final = solver.A_real" in source
    # DC-v1.0: the spectral-space affect velocity is stored as A_dot_k_final so
    # the dataset name makes its k-space nature explicit (was A_dot_final).
    assert "A_dot_k_final = solver.A_dot_k" in source
    assert "'A_final'" in source
    assert "'A_dot_k_final'" in source
    assert "'A_dot_final'" not in source
