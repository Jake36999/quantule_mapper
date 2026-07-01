import unittest

import numpy as np

from jax_scout import core_saturation_search as css


class CoreSaturationSearchTests(unittest.TestCase):
    def test_parser_defaults_preserve_historical_ic_seed(self):
        args = css.build_arg_parser().parse_args([])

        self.assertEqual(args.ic_seed, css.SEED)
        self.assertEqual(args.ic_norm, css.IC_NORM_PER_BLOB_FIXED)

    def test_parser_accepts_ic_seed_override(self):
        args = css.build_arg_parser().parse_args(["--ic-seed", "20260621", "--ic-counts", "1,2,3"])

        self.assertEqual(args.ic_seed, 20260621)
        self.assertEqual(args.ic_counts, "1,2,3")

    def test_build_ic_changes_with_seed_but_preserves_contract(self):
        target = 1000.0
        psi0_a, stats_a = css.build_ic(24, 3, seed=20260619, ic_norm=css.IC_NORM_TOTAL_MASS_FIXED, target_initial_mass=target)
        psi0_b, stats_b = css.build_ic(24, 3, seed=20260621, ic_norm=css.IC_NORM_TOTAL_MASS_FIXED, target_initial_mass=target)

        self.assertFalse(np.allclose(psi0_a, psi0_b))
        self.assertEqual(stats_a["K"], 3)
        self.assertEqual(stats_b["K"], 3)
        self.assertEqual(stats_a["ic_seed"], 20260619)
        self.assertEqual(stats_b["ic_seed"], 20260621)
        self.assertEqual(stats_a["ic_norm"], css.IC_NORM_TOTAL_MASS_FIXED)
        self.assertEqual(stats_b["ic_norm"], css.IC_NORM_TOTAL_MASS_FIXED)
        self.assertAlmostEqual(stats_a["initial_mass"], target, places=6)
        self.assertAlmostEqual(stats_b["initial_mass"], target, places=6)

    def test_ic_descriptor_records_explicit_seed(self):
        info = css.ic_descriptor([1, 2, 3, 4, 6], ic_seed=20260621, ic_norm=css.IC_NORM_TOTAL_MASS_FIXED, target_initial_mass=1200.0)

        self.assertEqual(info["ic_seed"], 20260621)
        self.assertTrue(info["normalized_across_k"])
        self.assertEqual(info["target_initial_mass"], 1200.0)


class StabilityGateTests(unittest.TestCase):
    """v2 long-time stability gate (normalized late-half energy drift)."""

    def _classify(self, er):
        # n_fin/core chosen to pass the structural rejects so only the slope/drift logic decides
        return css.classify(True, np.asarray(er, dtype=float), n_mid=4, n_fin=4, core_fin=0.5)[0]

    def test_flat_saturated_is_true(self):
        er = np.ones(20000)
        self.assertEqual(self._classify(er), "TRUE_SATURATED_BOUND_STATE")

    def test_slow_grower_with_tiny_slope_is_gated_out(self):
        # rises 1.25->1.5 over the late half: per-step slope ~2.5e-5 (<= SAT_SLOPE) but drift ~0.20 (> 0.15)
        er = np.linspace(1.0, 1.5, 20000)
        klass = self._classify(er)
        self.assertEqual(klass, "TRANSIENT_GROWER_REJECT")

    def test_slow_decayer_with_tiny_slope_is_gated_out(self):
        er = np.linspace(1.5, 1.0, 20000)  # drift ~-0.20, er_fin 1.0 (above the 0.3 spin-down floor)
        self.assertEqual(self._classify(er), "SPIN_DOWN_REJECT")

    def test_late_drift_recorded_in_metrics(self):
        _, base = css.classify(True, np.linspace(1.0, 1.5, 20000), n_mid=4, n_fin=4, core_fin=0.5)
        self.assertIn("late_drift", base)
        self.assertGreater(base["late_drift"], 0.15)

    def test_classifier_spec_exposes_gate(self):
        spec = css.classifier_spec()
        self.assertEqual(spec["version"], "PHASE_C_SATURATION_CLASSIFIER_v3")
        self.assertEqual(spec["late_drift_max"], css.LATE_DRIFT_MAX)
        self.assertEqual(spec["breathing_floor_ratio_min"], css.BREATHING_FLOOR_RATIO_MIN)


class BreathingExceptionTests(unittest.TestCase):
    """v3 breathing-aware exception: bounded oscillation accepted despite large drift."""

    def _classify(self, er):
        return css.classify(True, np.asarray(er, dtype=float), n_mid=4, n_fin=4, core_fin=0.5)

    def test_bounded_breathing_downswing_is_true(self):
        # rises 1.0->1.6 (peak mid) then settles to 1.2: never below start, ends below peak; drift ~-0.25
        er = np.concatenate([np.linspace(1.0, 1.6, 10000), np.linspace(1.6, 1.2, 10000)])
        klass, base = self._classify(er)
        self.assertTrue(base["bounded_breathing"])
        self.assertGreater(abs(base["late_drift"]), css.LATE_DRIFT_MAX)
        self.assertEqual(klass, "TRUE_SATURATED_BOUND_STATE")

    def test_floorward_decay_is_not_breathing(self):
        # peaks then falls below start (er_min 0.6 < 0.85*er0): real decay, not breathing
        er = np.concatenate([np.linspace(1.0, 1.5, 10000), np.linspace(1.5, 0.6, 10000)])
        klass, base = self._classify(er)
        self.assertFalse(base["bounded_breathing"])
        self.assertEqual(klass, "SPIN_DOWN_REJECT")

    def test_monotonic_grower_ending_at_peak_not_breathing(self):
        # er_fin == er_max (ends at its peak) -> peak-margin fails -> not breathing -> still rejected
        er = np.linspace(1.0, 1.5, 20000)
        klass, base = self._classify(er)
        self.assertFalse(base["bounded_breathing"])
        self.assertEqual(klass, "TRANSIENT_GROWER_REJECT")

    def test_breathing_metrics_recorded(self):
        _, base = self._classify(np.concatenate([np.linspace(1.0, 1.6, 10000), np.linspace(1.6, 1.2, 10000)]))
        for k in ("er0", "er_min", "floor_ratio", "bounded_breathing"):
            self.assertIn(k, base)


if __name__ == "__main__":
    unittest.main()
