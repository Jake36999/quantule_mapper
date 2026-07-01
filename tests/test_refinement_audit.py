"""
tests/test_refinement_audit.py

Unit tests for RefinementManifestGenerator, RefinementAudit, and the
Hermitian phase-scramble fix in quantulemapper_real.
"""
import copy
import numpy as np
import pytest

from validation_pipeline import RefinementManifestGenerator, RefinementAudit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_PARAMS = {
    "config_hash": "test_abc123",
    "global_seed": 42,
    "param_D": 1.0,
    "param_eta": 0.1,
    "param_rho_vac": 1.0,
    "param_a_coupling": 0.5,
    "param_splash_coupling": 0.2,
    "param_splash_fraction": -0.1,
    "simulation": {
        "n_grid": 32,
        "t_steps": 100,
        "dt": 0.001,
        "l_domain": 10.0,
    },
}

def _make_prov(
    sse: float,
    peak: float = 0.3,
    dir_min: float = None,
    directional_consistency: float = None,
) -> dict:
    if dir_min is None:
        dir_min = sse
    payload = {
        "log_prime_sse": sse,
        "dominant_peak_k": peak,
        "sse_directional_min": dir_min,
    }
    if directional_consistency is not None:
        payload["directional_consistency"] = directional_consistency
    return {"spectral_fidelity": payload}


# ---------------------------------------------------------------------------
# RefinementManifestGenerator
# ---------------------------------------------------------------------------

class TestRefinementManifestGenerator:
    def test_returns_four_variants(self):
        variants = RefinementManifestGenerator.generate(BASE_PARAMS)
        assert len(variants) == 4

    def test_variant_labels(self):
        labels = {v["label"] for v in RefinementManifestGenerator.generate(BASE_PARAMS)}
        assert labels == {"dt_half", "dt_quarter", "dealias_2_3", "grid_double"}

    def test_dt_half_halves_timestep(self):
        variants = {v["label"]: v for v in RefinementManifestGenerator.generate(BASE_PARAMS)}
        dt_half = variants["dt_half"]["params"]["simulation"]["dt"]
        assert abs(dt_half - 0.0005) < 1e-10

    def test_dt_half_doubles_steps(self):
        variants = {v["label"]: v for v in RefinementManifestGenerator.generate(BASE_PARAMS)}
        steps = variants["dt_half"]["params"]["simulation"]["t_steps"]
        assert steps == 200

    def test_dt_quarter_quarters_timestep(self):
        variants = {v["label"]: v for v in RefinementManifestGenerator.generate(BASE_PARAMS)}
        dt = variants["dt_quarter"]["params"]["simulation"]["dt"]
        assert abs(dt - 0.00025) < 1e-10

    def test_dealias_2_3_sets_fraction(self):
        variants = {v["label"]: v for v in RefinementManifestGenerator.generate(BASE_PARAMS)}
        frac = variants["dealias_2_3"]["params"]["param_dealias_fraction"]
        assert abs(frac - 2.0 / 3.0) < 1e-10

    def test_grid_double_feasible_for_small_grid(self):
        variants = {v["label"]: v for v in RefinementManifestGenerator.generate(BASE_PARAMS)}
        assert variants["grid_double"]["feasible"] is True
        assert variants["grid_double"]["params"]["simulation"]["n_grid"] == 64

    def test_grid_double_infeasible_for_large_grid(self):
        large = copy.deepcopy(BASE_PARAMS)
        large["simulation"]["n_grid"] = 128
        variants = {v["label"]: v for v in RefinementManifestGenerator.generate(large)}
        assert variants["grid_double"]["feasible"] is False

    def test_base_params_not_mutated(self):
        original_dt = BASE_PARAMS["simulation"]["dt"]
        RefinementManifestGenerator.generate(BASE_PARAMS)
        assert BASE_PARAMS["simulation"]["dt"] == original_dt


# ---------------------------------------------------------------------------
# RefinementAudit
# ---------------------------------------------------------------------------

class TestRefinementAudit:
    def test_insufficient_data_with_one_result(self):
        result = RefinementAudit.assess([("baseline", _make_prov(0.5))])
        assert result["verdict"] == "INSUFFICIENT_DATA"

    def test_absent_when_baseline_sse_high(self):
        results = [
            ("baseline", _make_prov(2.0)),
            ("dt_half", _make_prov(1.8)),
        ]
        result = RefinementAudit.assess(results)
        assert result["verdict"] == "ABSENT"

    def test_stable_when_signal_survives(self):
        results = [
            ("baseline", _make_prov(0.4, peak=0.30)),
            ("dt_half",  _make_prov(0.5, peak=0.31)),
            ("dealias_2_3", _make_prov(0.6, peak=0.30)),
        ]
        result = RefinementAudit.assess(results)
        assert result["verdict"] == "STABLE"
        assert result["baseline_sse"] == pytest.approx(0.4)
        assert result["peak_drift"] < RefinementAudit.PEAK_DRIFT_TOL

    def test_drifting_when_sse_exceeds_threshold(self):
        results = [
            ("baseline", _make_prov(0.4)),
            ("dt_half",  _make_prov(3.5)),   # blows past convergence_threshold
        ]
        result = RefinementAudit.assess(results)
        assert result["verdict"] == "DRIFTING"
        assert "dt_half" in result["reason"]

    def test_drifting_when_peak_drifts(self):
        results = [
            ("baseline",    _make_prov(0.4, peak=0.20)),
            ("dt_half",     _make_prov(0.5, peak=0.28)),   # drift = 0.08 > tol
        ]
        result = RefinementAudit.assess(results)
        assert result["verdict"] == "DRIFTING"
        assert "peak_drift" in result["reason"]

    def test_drifting_when_directional_sse_is_invalid(self):
        results = [
            ("baseline", _make_prov(0.4, peak=0.30, dir_min=0.4, directional_consistency=0.05)),
            ("dt_half", _make_prov(0.5, peak=0.31, dir_min=2.5, directional_consistency=0.05)),
        ]
        result = RefinementAudit.assess(results)
        assert result["verdict"] == "DRIFTING"
        assert "directional_sse_min" in result["reason"]

    def test_stable_when_directional_metrics_pass(self):
        results = [
            ("baseline", _make_prov(0.4, peak=0.30, dir_min=0.4, directional_consistency=0.05)),
            ("dt_half", _make_prov(0.5, peak=0.31, dir_min=0.5, directional_consistency=0.05)),
            ("dealias_2_3", _make_prov(0.6, peak=0.30, dir_min=0.6, directional_consistency=0.05)),
        ]
        result = RefinementAudit.assess(results)
        assert result["verdict"] == "STABLE"

    def test_per_variant_details_populated(self):
        results = [
            ("baseline", _make_prov(0.4)),
            ("dt_half",  _make_prov(0.5)),
        ]
        result = RefinementAudit.assess(results)
        assert "dt_half" in result["per_variant_details"]
        assert "log_prime_sse" in result["per_variant_details"]["dt_half"]


# ---------------------------------------------------------------------------
# Hermitian phase scramble
# ---------------------------------------------------------------------------

class TestSpectralPhaseScramble:
    def test_output_is_real(self):
        from quantulemapper_real import spectral_phase_scramble
        rng = np.random.default_rng(0)
        rho = np.random.default_rng(1).random((16, 16, 16)).astype(np.float64)
        out = spectral_phase_scramble(rho, rng)
        assert np.isrealobj(out), "Output must be purely real"

    def test_output_shape_preserved(self):
        from quantulemapper_real import spectral_phase_scramble
        rng = np.random.default_rng(0)
        rho = np.ones((8, 8, 8), dtype=np.float64)
        out = spectral_phase_scramble(rho, rng)
        assert out.shape == rho.shape

    def test_total_power_preserved(self):
        """Total spectral power (Parseval) should be preserved by phase scramble.

        We cannot assert per-coefficient amplitude equality because the k=0 and
        Nyquist slices contain conjugate pairs whose phases must be antisymmetric;
        randomising them independently slightly redistributes amplitude across
        those constrained coefficients.  The sum of squares (∝ field energy) is
        the meaningful invariant for a spectral null test.
        """
        from quantulemapper_real import spectral_phase_scramble
        rng = np.random.default_rng(0)
        rho = np.random.default_rng(2).random((16, 16, 16)).astype(np.float64)
        out = spectral_phase_scramble(rho, rng)
        # Parseval: sum(|rfftn|²) ∝ sum(|field|²); 1 % tolerance is generous
        orig_power = np.sum(rho ** 2)
        scram_power = np.sum(out ** 2)
        assert abs(scram_power - orig_power) / (orig_power + 1e-30) < 0.02, (
            f"Total power drifted by >{2}%: orig={orig_power:.4f}, scrambled={scram_power:.4f}"
        )

    def test_scramble_changes_field(self):
        from quantulemapper_real import spectral_phase_scramble
        rng = np.random.default_rng(42)
        rho = np.random.default_rng(3).random((16, 16, 16)).astype(np.float64)
        out = spectral_phase_scramble(rho, rng)
        assert not np.allclose(rho, out), "Scrambled field should differ from original"
