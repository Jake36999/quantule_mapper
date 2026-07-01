import csv
import tempfile
import unittest
from pathlib import Path

from jax_scout import core_saturation_search as css
from jax_scout import core_saturation_replay as csr


def _write_csv(path, rows):
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["idx", "label", "ic_blobs", "ic_norm", "target_initial_mass", "ic_seed", *css.order],
        )
        writer.writeheader()
        writer.writerows(rows)


class CoreSaturationReplayTests(unittest.TestCase):
    def test_classifier_spec_exposes_calibrated_thresholds(self):
        spec = css.classifier_spec()
        self.assertEqual(spec["sat_slope"], css.SAT_SLOPE)
        self.assertEqual(spec["near_slope"], css.NEAR_SLOPE)
        self.assertEqual(spec["energy_true_min"], 0.5)
        self.assertEqual(spec["spin_down_floor"], 0.3)

    def test_ic_descriptor_marks_non_normalized_family(self):
        info = css.ic_descriptor([1, 2, 3, 4, 6])
        self.assertEqual(info["generator"], "multiseed_ic")
        self.assertEqual(info["ic_seed"], css.SEED)
        self.assertFalse(info["normalized_across_k"])
        self.assertEqual(info["blob_width_rule"], "fixed_w=L/12")
        self.assertEqual(info["ic_norm"], "per_blob_fixed")

    def test_build_ic_per_blob_fixed_reports_measured_mass(self):
        psi0, stats = css.build_ic(24, 3, seed=css.SEED, ic_norm="per_blob_fixed")
        measured = float((abs(psi0) ** 2).sum())
        self.assertAlmostEqual(stats["initial_mass"], measured)
        self.assertIsNone(stats["target_initial_mass"])
        self.assertEqual(stats["ic_norm"], "per_blob_fixed")

    def test_build_ic_total_mass_fixed_hits_target(self):
        target = 123.45
        psi0, stats = css.build_ic(24, 3, seed=css.SEED, ic_norm="total_mass_fixed", target_initial_mass=target)
        measured = float((abs(psi0) ** 2).sum())
        self.assertAlmostEqual(measured, target, places=6)
        self.assertAlmostEqual(stats["initial_mass"], target, places=6)
        self.assertAlmostEqual(stats["target_initial_mass"], target, places=6)
        self.assertEqual(stats["ic_norm"], "total_mass_fixed")

    def test_resolve_row_by_idx(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "rows.csv"
            row = {
                "idx": "7",
                "label": "rand",
                "ic_blobs": "3",
                "ic_norm": "total_mass_fixed",
                "target_initial_mass": "42.5",
                "ic_seed": "20260619",
                **{name: str(i + 0.25) for i, name in enumerate(css.order)},
            }
            _write_csv(csv_path, [row])

            resolved = csr.resolve_candidate(csv_path, idx=7)

            self.assertEqual(resolved["idx"], 7)
            self.assertEqual(resolved["label"], "rand")
            self.assertEqual(resolved["ic_blobs"], 3)
            self.assertEqual(resolved["ic_norm"], "total_mass_fixed")
            self.assertEqual(resolved["target_initial_mass"], 42.5)
            self.assertEqual(resolved["ic_seed"], 20260619)
            self.assertEqual(resolved["params"]["param_D"], 0.25)
            self.assertEqual(resolved["params"]["param_a"], 7.25)

    def test_resolve_feb_reference_uses_shared_constants(self):
        resolved = csr.resolve_candidate(None, ref="feb56dc7")
        self.assertEqual(resolved["label"], "ref_feb56dc7")
        self.assertEqual(resolved["ic_blobs"], 6)
        self.assertEqual(resolved["ic_norm"], "per_blob_fixed")
        self.assertEqual(resolved["params"], css.FEB)

    def test_build_resolution_scaled_replay_metadata(self):
        candidate = {
            "target_initial_mass": 1200.0,
            "source": {"kind": "csv_row", "source_resolution_N": 48},
        }

        meta = csr.build_replay_target_metadata(candidate, replay_target_initial_mass=9600.0, replay_resolution_N=96)

        self.assertEqual(meta["saved_target_initial_mass"], 1200.0)
        self.assertEqual(meta["replay_target_initial_mass"], 9600.0)
        self.assertEqual(meta["mass_scaling_mode"], "resolution_scaled_raw_target")
        self.assertEqual(meta["source_resolution_N"], 48)
        self.assertEqual(meta["replay_resolution_N"], 96)
        self.assertAlmostEqual(meta["dx_source"], css.L_ / 48)
        self.assertAlmostEqual(meta["dx_replay"], css.L_ / 96)
        self.assertAlmostEqual(meta["target_integral_mass"], 1200.0 * (css.L_ / 48) ** 3)
        self.assertAlmostEqual(meta["target_raw_mass_replay"], 9600.0)

    def test_build_exact_replay_metadata_without_override(self):
        candidate = {
            "target_initial_mass": 500.0,
            "source": {"kind": "csv_row", "source_resolution_N": 48},
        }

        meta = csr.build_replay_target_metadata(candidate, replay_target_initial_mass=500.0, replay_resolution_N=48)

        self.assertEqual(meta["mass_scaling_mode"], "exact_saved_raw_target")
        self.assertEqual(meta["saved_target_initial_mass"], 500.0)
        self.assertEqual(meta["replay_target_initial_mass"], 500.0)

    def test_cross_resolution_raw_target_requires_explicit_override(self):
        candidate = {
            "target_initial_mass": 1200.0,
            "source": {"kind": "csv_row", "source_resolution_N": 48},
        }

        with self.assertRaisesRegex(ValueError, "Cross-resolution raw-target replay requires --target-initial-mass-override"):
            csr.resolve_replay_target_initial_mass(
                candidate,
                replay_resolution_N=96,
                target_initial_mass=None,
                target_initial_mass_override=None,
            )

    def test_same_resolution_replay_can_use_saved_target(self):
        candidate = {
            "target_initial_mass": 1200.0,
            "source": {"kind": "csv_row", "source_resolution_N": 48},
        }

        target = csr.resolve_replay_target_initial_mass(
            candidate,
            replay_resolution_N=48,
            target_initial_mass=None,
            target_initial_mass_override=None,
        )

        self.assertEqual(target, 1200.0)

    def test_resolve_replay_ic_seed_uses_override_when_present(self):
        candidate = {"ic_seed": 20260619}

        seed = csr.resolve_replay_ic_seed(candidate, ic_seed_override=20260620)

        self.assertEqual(seed, 20260620)

    def test_apply_param_overrides_updates_only_named_fields(self):
        base = dict(css.FEB)

        replay_params, saved_params = csr.resolve_replay_params(
            {"params": base},
            param_overrides={"param_eta": 0.05, "param_a": 0.33},
        )

        self.assertEqual(saved_params, base)
        self.assertEqual(replay_params["param_eta"], 0.05)
        self.assertEqual(replay_params["param_a"], 0.33)
        self.assertEqual(replay_params["param_D"], base["param_D"])


if __name__ == "__main__":
    unittest.main()
