import json
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from validation_pipeline import ArtifactLoader, SpectralFidelityEngine, ValidationPipeline


def _write_params(params_path: Path, config_hash: str) -> None:
    payload = {
        "config_hash": config_hash,
        "simulation": {"L_domain": 10.0, "N_grid": 8},
        "param_D": 1.0,
        "param_eta": 0.65,
        "param_rho_vac": 1.0,
        "param_a_coupling": 0.2,
        "param_splash_coupling": 0.2,
        "param_splash_fraction": -0.5,
    }
    params_path.write_text(json.dumps(payload), encoding="utf-8")


def _write_worker_like_artifact(h5_path: Path, with_legacy_aliases: bool = False) -> None:
    rng = np.random.default_rng(42)
    psi = (rng.standard_normal((8, 8, 8)) + 1j * rng.standard_normal((8, 8, 8))).astype(np.complex64)
    rho = np.abs(psi) ** 2
    omega_sq = (1.0 + 0.05 * rho).astype(np.float32)

    with h5py.File(h5_path, "w") as h5f:
        h5f.create_dataset("psi_final", data=psi)
        h5f.create_dataset("rho_final", data=rho)
        h5f.create_dataset("omega_sq_final", data=omega_sq)

        telem = h5f.create_group("telemetry")
        telem.create_dataset("step", data=np.array([0, 10, 20], dtype=np.int64))
        telem.create_dataset("C_invariant", data=np.array([0.2, 0.25, 0.3], dtype=np.float64))
        telem.create_dataset("energy", data=np.array([1.0, 1.0, 1.0], dtype=np.float64))

        ext = h5f.create_group("extended_telemetry")
        ext.create_dataset("step_count", data=np.array([20], dtype=np.int64))
        ext.create_dataset("sim_time", data=np.array([0.11], dtype=np.float64))
        ext.create_dataset("dt", data=np.array([0.005], dtype=np.float64))
        ext.create_dataset("grid_shape", data=np.array([8, 8, 8], dtype=np.int32))
        ext.create_dataset("params_hash", data=np.array(["abc123"], dtype=h5py.string_dtype(encoding="utf-8")))

        if with_legacy_aliases:
            h5f.create_dataset("phase_coherence_final", data=np.array([0.4, 0.5], dtype=np.float64))
            h5f.create_dataset("grad_phase_var_final", data=np.array([0.1, 0.2], dtype=np.float64))
            h5f.create_dataset("J_info_l2_final", data=np.array([0.9, 1.1], dtype=np.float64))
            h5f.create_dataset("omega_saturation_final", data=np.array([1.2, 1.3], dtype=np.float64))


def test_worker_artifact_has_no_validation_metrics_and_pipeline_derives_canonical_means(tmp_path: Path):
    h5_path = tmp_path / "simulation.h5"
    params_path = tmp_path / "params.json"
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_worker_like_artifact(h5_path, with_legacy_aliases=False)
    _write_params(params_path, config_hash="derived-metrics-hash")

    with h5py.File(h5_path, "r") as h5f:
        ext = h5f["extended_telemetry"]
        forbidden = {"J_info_l2", "grad_phase_var", "phase_coherence", "omega_saturation"}
        assert forbidden.isdisjoint(set(ext.keys()))

    with patch("validation_pipeline.SpectralFidelityEngine.run", return_value={"validation_status": "FAIL: HIGH_SSE", "log_prime_sse": 999.0}), patch(
        "validation_pipeline.ContractEnforcerEngine.enforce", return_value=None
    ):
        pipeline = ValidationPipeline(input_path=str(h5_path), params_path=str(params_path), output_dir=str(output_dir))
        assert pipeline.run() is True

    provenance = json.loads((output_dir / "provenance_derived-metrics-hash.json").read_text(encoding="utf-8"))
    aletheia = provenance.get("aletheia_metrics", {})

    assert aletheia.get("j_info_l2_mean") is not None
    assert aletheia.get("grad_phase_var_mean") is not None
    assert aletheia.get("phase_coherence_mean") is not None
    assert aletheia.get("omega_sat_mean") is not None
    assert aletheia.get("spectral_bandwidth_mean") is not None
    assert aletheia.get("collapse_invariant") is not None


def test_artifact_loader_normalizes_legacy_aliases_to_canonical_means(tmp_path: Path):
    h5_path = tmp_path / "legacy_aliases.h5"
    _write_worker_like_artifact(h5_path, with_legacy_aliases=True)

    _, _, telemetry = ArtifactLoader.load(str(h5_path))

    assert telemetry.get("phase_coherence_mean") is not None
    assert telemetry.get("grad_phase_var_mean") is not None
    assert telemetry.get("j_info_l2_mean") is not None
    assert telemetry.get("omega_sat_mean") is not None


def test_validation_gate_topological_override_keeps_tda_path(tmp_path: Path):
    h5_path = tmp_path / "override_case.h5"
    params_path = tmp_path / "params_override.json"
    output_dir = tmp_path / "out_override"
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(123)
    psi = (
        25.0 * (rng.standard_normal((8, 8, 8)) + 1j * rng.standard_normal((8, 8, 8)))
    ).astype(np.complex64)
    rho = np.abs(psi) ** 2
    with h5py.File(h5_path, "w") as h5f:
        h5f.create_dataset("psi_final", data=psi)
        h5f.create_dataset("rho_final", data=rho)

    payload = {
        "config_hash": "override-hash",
        "simulation": {"L_domain": 10.0, "N_grid": 8},
        "collapse_threshold": 5.0,
        "param_D": 1.0,
        "param_eta": 0.65,
        "param_rho_vac": 1.0,
        "param_a_coupling": 0.2,
    }
    params_path.write_text(json.dumps(payload), encoding="utf-8")

    with patch(
        "validation_pipeline.SpectralFidelityEngine.run",
        return_value={"validation_status": "FAIL: HIGH_SSE", "log_prime_sse": 999.0},
    ), patch("validation_pipeline.ContractEnforcerEngine.enforce", return_value=None), patch(
        "validation_pipeline.TopologyEngine.run_tda", return_value={"status": "tda_ok"}
    ) as tda_mock, patch("validation_pipeline.LOMTelemetryEngine.extract", return_value=3), patch(
        "validation_pipeline.EmpiricalBridgeEngine.run", return_value=(0.0, 0.0)
    ), patch("validation_pipeline.TensorValidationEngine.run", return_value=(0.0, 0.0)), patch(
        "validation_pipeline.StatisticalValidationEngine.run", return_value=(1.0, 999.0)
    ), patch("validation_pipeline.AletheiaMetricsEngine.run", return_value={"pli": 0.0, "ic": 1.0}):
        pipeline = ValidationPipeline(
            input_path=str(h5_path),
            params_path=str(params_path),
            output_dir=str(output_dir),
        )
        assert pipeline.run() is True

    assert tda_mock.called, "Topological override must bypass hard rejection and execute TDA"
    provenance = json.loads((output_dir / "provenance_override-hash.json").read_text(encoding="utf-8"))
    spec = provenance.get("spectral_fidelity", {})
    assert spec.get("validation_status") != "GATE_REJECTED"
    assert spec.get("collapse_event_count") == 3


def test_spectral_fidelity_exception_contract_includes_diagnostics():
    rho = np.ones((4, 4, 4), dtype=np.float64)
    with patch("validation_pipeline.cep_profiler.prime_log_sse", side_effect=RuntimeError("boom")):
        result = SpectralFidelityEngine.run(rho)

    assert result["log_prime_sse"] == 999.0
    assert result["n_peaks_found_main"] == 0
    assert "profiler_exception" in str(result["failure_reason_main"])
    assert "boom" in str(result["failure_reason_main"])


def test_artifact_loader_ingests_optional_causal_field_artifacts(tmp_path: Path):
    h5_path = tmp_path / "causal_fields.h5"
    with h5py.File(h5_path, "w") as h5f:
        h5f.create_dataset("psi_final", data=np.ones((4, 4, 4), dtype=np.complex64))
        h5f.create_dataset("rho_final", data=np.ones((4, 4, 4), dtype=np.float32))
        h5f.create_dataset("A_final", data=np.full((4, 4, 4), 2.0, dtype=np.float64))
        h5f.create_dataset("A_dot_final", data=np.full((4, 4, 4), 3.0 + 1.0j, dtype=np.complex128))

    _, _, telemetry = ArtifactLoader.load(str(h5_path))

    assert "A_final" in telemetry
    assert "A_dot_final" in telemetry
    assert telemetry["A_final"].shape == (4, 4, 4)
    assert telemetry["A_dot_final"].shape == (4, 4, 4)
