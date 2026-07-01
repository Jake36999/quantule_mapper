import csv
import tempfile
import unittest
from pathlib import Path


def _write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class CoreSaturationCollapseDiagTests(unittest.TestCase):
    def test_label_collapse_like_runaway(self):
        from jax_scout import core_saturation_collapse_diag as diag

        summary = {
            "klass": "LATE_BLOWUP_REJECT",
            "finite_last": False,
            "split_before_blowup": False,
            "rho_peak_growth_ratio": 8.0,
            "core_radius_shrink_ratio": 0.3,
            "compactness_growth_ratio": 5.0,
            "omega2_min_ratio": 0.1,
            "grad_log_omega_growth_ratio": 4.0,
            "high_k_fraction_max": 0.08,
        }
        self.assertEqual(diag.assign_diagnostic_label(summary), "COLLAPSE_LIKE_RUNAWAY")

    def test_label_fragmenting_blowup_when_nodes_split_first(self):
        from jax_scout import core_saturation_collapse_diag as diag

        summary = {
            "klass": "LATE_BLOWUP_REJECT",
            "finite_last": False,
            "split_before_blowup": True,
            "rho_peak_growth_ratio": 6.0,
            "core_radius_shrink_ratio": 0.4,
            "compactness_growth_ratio": 3.0,
            "omega2_min_ratio": 0.2,
            "grad_log_omega_growth_ratio": 2.0,
            "high_k_fraction_max": 0.05,
        }
        self.assertEqual(diag.assign_diagnostic_label(summary), "FRAGMENTING_BLOWUP")

    def test_label_delocalized_growth_without_strong_compaction(self):
        from jax_scout import core_saturation_collapse_diag as diag

        summary = {
            "klass": "LATE_BLOWUP_REJECT",
            "finite_last": True,
            "split_before_blowup": False,
            "rho_peak_growth_ratio": 1.4,
            "core_radius_shrink_ratio": 0.95,
            "compactness_growth_ratio": 1.2,
            "omega2_min_ratio": 0.9,
            "grad_log_omega_growth_ratio": 1.1,
            "high_k_fraction_max": 0.07,
        }
        self.assertEqual(diag.assign_diagnostic_label(summary), "DELOCALIZED_GROWTH")

    def test_label_high_k_artifact_suspect(self):
        from jax_scout import core_saturation_collapse_diag as diag

        summary = {
            "klass": "LATE_BLOWUP_REJECT",
            "finite_last": False,
            "split_before_blowup": False,
            "rho_peak_growth_ratio": 8.0,
            "core_radius_shrink_ratio": 0.25,
            "compactness_growth_ratio": 6.0,
            "omega2_min_ratio": 0.08,
            "grad_log_omega_growth_ratio": 5.0,
            "high_k_fraction_max": 0.42,
        }
        self.assertEqual(diag.assign_diagnostic_label(summary), "HIGH_K_NUMERICAL_ARTIFACT_SUSPECT")

    def test_label_maps_stable_and_spin_down_classes(self):
        from jax_scout import core_saturation_collapse_diag as diag

        sat = {"klass": "TRUE_SATURATED_BOUND_STATE"}
        spin = {"klass": "SPIN_DOWN_REJECT"}
        self.assertEqual(diag.assign_diagnostic_label(sat), "SATURATED_BOUND_STATE")
        self.assertEqual(diag.assign_diagnostic_label(spin), "SPIN_DOWN_DECAY")

    def test_select_required_candidate_groups(self):
        from jax_scout import core_saturation_collapse_diag as diag

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            high = root / "CORE_SAT_HUNT_20260623_123527"
            low = root / "CORE_SAT_HUNT_20260623_120758"
            base = root / "CORE_SAT_HUNT_20260623_113318"
            for path in (high, low, base):
                path.mkdir(parents=True, exist_ok=True)

            fieldnames = ["idx", "ic_blobs", "ic_norm", "target_initial_mass", "klass", "n_fin"]
            _write_csv(
                high / "all_evals.csv",
                [
                    {"idx": "1", "ic_blobs": "1", "ic_norm": "total_mass_fixed", "target_initial_mass": "2050.293702", "klass": "LATE_BLOWUP_REJECT", "n_fin": "0"},
                    {"idx": "2", "ic_blobs": "6", "ic_norm": "total_mass_fixed", "target_initial_mass": "2050.293702", "klass": "TRUE_SATURATED_BOUND_STATE", "n_fin": "4"},
                ],
                fieldnames,
            )
            _write_csv(
                low / "all_evals.csv",
                [
                    {"idx": "3", "ic_blobs": "1", "ic_norm": "total_mass_fixed", "target_initial_mass": "291.882452", "klass": "TRUE_SATURATED_BOUND_STATE", "n_fin": "2"},
                    {"idx": "4", "ic_blobs": "1", "ic_norm": "total_mass_fixed", "target_initial_mass": "291.882452", "klass": "SPIN_DOWN_REJECT", "n_fin": "0"},
                ],
                fieldnames,
            )
            _write_csv(
                base / "all_evals.csv",
                [
                    {"idx": "5", "ic_blobs": "1", "ic_norm": "per_blob_fixed", "target_initial_mass": "", "klass": "TRUE_SATURATED_BOUND_STATE", "n_fin": "2"},
                    {"idx": "6", "ic_blobs": "6", "ic_norm": "per_blob_fixed", "target_initial_mass": "", "klass": "TRUE_SATURATED_BOUND_STATE", "n_fin": "4"},
                ],
                fieldnames,
            )

            groups = diag.select_required_candidates(root)

        group_names = {group["group"] for group in groups}
        labels = {(group["group"], group["idx"]) for group in groups if group.get("idx") is not None}
        self.assertIn("high_target_k1_blowup", group_names)
        self.assertIn("high_target_k6_true", group_names)
        self.assertIn("low_target_k1_true_or_spin", group_names)
        self.assertIn("baseline_k1_k6", group_names)
        self.assertIn("feb56dc7_reference", group_names)
        self.assertIn(("high_target_k1_blowup", 1), labels)
        self.assertIn(("high_target_k6_true", 2), labels)
        self.assertIn(("low_target_k1_true_or_spin", 3), labels)
        self.assertIn(("low_target_k1_true_or_spin", 4), labels)
        self.assertIn(("baseline_k1_k6", 5), labels)
        self.assertIn(("baseline_k1_k6", 6), labels)

    def test_select_high_mass_comparison_candidates(self):
        from jax_scout import core_saturation_collapse_diag as diag

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            high = root / "CORE_SAT_HUNT_20260623_123527"
            high.mkdir(parents=True, exist_ok=True)
            fieldnames = ["idx", "ic_blobs", "ic_norm", "target_initial_mass", "klass", "n_fin"]
            _write_csv(
                high / "all_evals.csv",
                [
                    {"idx": "2", "ic_blobs": "1", "ic_norm": "total_mass_fixed", "target_initial_mass": "2050.293702", "klass": "LATE_BLOWUP_REJECT", "n_fin": "0"},
                    {"idx": "32", "ic_blobs": "6", "ic_norm": "total_mass_fixed", "target_initial_mass": "2050.293702", "klass": "TRUE_SATURATED_BOUND_STATE", "n_fin": "4"},
                    {"idx": "33", "ic_blobs": "6", "ic_norm": "total_mass_fixed", "target_initial_mass": "2050.293702", "klass": "TRUE_SATURATED_BOUND_STATE", "n_fin": "4"},
                    {"idx": "39", "ic_blobs": "6", "ic_norm": "total_mass_fixed", "target_initial_mass": "2050.293702", "klass": "TRUE_SATURATED_BOUND_STATE", "n_fin": "5"},
                ],
                fieldnames,
            )

            candidates = diag.select_high_mass_comparison_candidates(root)

        idxs = {(candidate.get("idx"), candidate.get("group")) for candidate in candidates}
        self.assertIn((2, "high_mass_k1_failure"), idxs)
        self.assertIn((32, "high_mass_k6_stable"), idxs)
        self.assertIn((33, "high_mass_k6_stable"), idxs)
        self.assertIn((39, "high_mass_k6_stable"), idxs)
        self.assertTrue(any(candidate.get("ref") == "feb56dc7" for candidate in candidates))

    def test_interpret_high_mass_comparison_supports_distributed_stabilization(self):
        from jax_scout import core_saturation_collapse_diag as diag

        rows = [
            {"idx": 2, "K": 1, "class": "LATE_BLOWUP_REJECT", "diagnostic_label": "FRAGMENTING_BLOWUP", "compactness_max": 245.0},
            {"idx": 32, "K": 6, "class": "TRUE_SATURATED_BOUND_STATE", "diagnostic_label": "SATURATED_BOUND_STATE", "compactness_max": 45.0},
            {"idx": 33, "K": 6, "class": "TRUE_SATURATED_BOUND_STATE", "diagnostic_label": "SATURATED_BOUND_STATE", "compactness_max": 34.0},
            {"idx": 39, "K": 6, "class": "TRUE_SATURATED_BOUND_STATE", "diagnostic_label": "SATURATED_BOUND_STATE", "compactness_max": 83.0},
        ]

        verdict = diag.interpret_high_mass_comparison(rows)

        self.assertEqual(verdict, "DISTRIBUTED_MASS_STABILIZATION_SUPPORTED")

    def test_select_threshold_pilot_candidates(self):
        from jax_scout import core_saturation_collapse_diag as diag

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fieldnames = ["idx", "ic_blobs", "ic_norm", "target_initial_mass", "klass", "n_fin"]
            for i, run_name in enumerate(diag.THRESHOLD_PILOT_RUN_NAMES):
                run_dir = root / run_name
                run_dir.mkdir(parents=True, exist_ok=True)
                _write_csv(
                    run_dir / "all_evals.csv",
                    [
                        {
                            "idx": str(i),
                            "ic_blobs": "1" if i % 2 == 0 else "6",
                            "ic_norm": "total_mass_fixed",
                            "target_initial_mass": str(100.0 + i),
                            "klass": "TRUE_SATURATED_BOUND_STATE",
                            "n_fin": "4",
                        }
                    ],
                    fieldnames,
                )

            candidates = diag.select_threshold_pilot_candidates(root)

        self.assertEqual(len(candidates), len(diag.THRESHOLD_PILOT_RUN_NAMES))
        self.assertEqual(candidates[0]["group"], "threshold_pilot")
        self.assertEqual(candidates[-1]["idx"], len(diag.THRESHOLD_PILOT_RUN_NAMES) - 1)

    def test_summarize_threshold_results(self):
        from jax_scout import core_saturation_collapse_diag as diag

        rows = [
            {
                "K": 1,
                "target_initial_mass": 500.0,
                "class": "TRUE_SATURATED_BOUND_STATE",
                "diagnostic_label": "SATURATED_BOUND_STATE",
                "node_count_last": 2,
                "compactness_max": 12.0,
                "high_k_fraction_max": 0.08,
                "time_to_failure": None,
                "late_energy_slope": 1.0e-4,
            },
            {
                "K": 1,
                "target_initial_mass": 500.0,
                "class": "LATE_BLOWUP_REJECT",
                "diagnostic_label": "FRAGMENTING_BLOWUP",
                "node_count_last": 4,
                "compactness_max": 30.0,
                "high_k_fraction_max": 0.15,
                "time_to_failure": 2200.0,
                "late_energy_slope": 1.0e-2,
            },
            {
                "K": 6,
                "target_initial_mass": 1200.0,
                "class": "TRUE_SATURATED_BOUND_STATE",
                "diagnostic_label": "SATURATED_BOUND_STATE",
                "node_count_last": 5,
                "compactness_max": 8.0,
                "high_k_fraction_max": 0.05,
                "time_to_failure": None,
                "late_energy_slope": -2.0e-5,
            },
        ]

        summary = diag.summarize_threshold_results(rows)

        self.assertEqual(summary["overall_class_counts"]["TRUE_SATURATED_BOUND_STATE"], 2)
        self.assertEqual(summary["overall_diagnostic_counts"]["SATURATED_BOUND_STATE"], 2)
        self.assertEqual(len(summary["by_K_target_mass"]), 2)
        bucket = next(item for item in summary["by_K_target_mass"] if item["K"] == 1)
        self.assertEqual(bucket["class_counts"]["LATE_BLOWUP_REJECT"], 1)
        self.assertEqual(bucket["final_node_counts"]["2"], 1)
        self.assertEqual(bucket["final_node_counts"]["4"], 1)
        self.assertAlmostEqual(bucket["compactness"]["max"], 30.0)


if __name__ == "__main__":
    unittest.main()
