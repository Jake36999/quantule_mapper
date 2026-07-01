import csv
import json
import tempfile
import unittest
from pathlib import Path

from jax_scout import core_saturation_search as css


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["idx", "label", "ic_blobs", "ic_norm", "target_initial_mass", "ic_seed", *css.order],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path, n=48, t=4000):
    path.write_text(json.dumps({"N": n, "T": t}, indent=2), encoding="utf-8")


def _row(idx, k, target, ic_seed=20260619):
    params = dict(css.FEB)
    params["param_eta"] = 0.0704 + 1.0e-4 * idx
    params["param_a"] = 0.4802 - 1.0e-4 * idx
    return {
        "idx": str(idx),
        "label": "rand",
        "ic_blobs": str(k),
        "ic_norm": "total_mass_fixed",
        "target_initial_mass": str(target),
        "ic_seed": str(ic_seed),
        **{name: str(params[name]) for name in css.order},
    }


class CoreSaturationRobustnessTests(unittest.TestCase):
    def _build_sources(self, root: Path):
        specs = [
            ("CORE_SAT_HUNT_20260623_171758", [_row(4, 1, 500.0)]),
            ("CORE_SAT_HUNT_20260623_173417", [_row(2, 1, 1200.0), _row(10, 6, 1200.0)]),
            ("CORE_SAT_HUNT_20260623_175018", [_row(10, 6, 2050.293702)]),
        ]
        for run_name, rows in specs:
            run_dir = root / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_csv(run_dir / "all_evals.csv", rows)
            _write_summary(run_dir / "summary.json")

    def test_build_manifest_contains_37_rows_and_scaled_cross_resolution_commands(self):
        from jax_scout import core_saturation_robustness as robust

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            sweep_root = tmp / "sweep_runs"
            sweep_root.mkdir()
            self._build_sources(sweep_root)
            outdir = sweep_root / "CORE_SAT_THRESHOLD_BRANCH_ROBUSTNESS_TEST"

            rows = robust.build_manifest(outdir=outdir, sweep_root=sweep_root)

        self.assertEqual(len(rows), 37)
        self.assertEqual(rows[0]["group"], "k1_low_mass_branch")
        self.assertEqual(rows[0]["source_idx"], 4)
        cross_resolution = [row for row in rows if row["kind"] == "scaled_replay"]
        self.assertEqual(len(cross_resolution), 36)
        self.assertTrue(all("--target-initial-mass-override" in row["replay_command"] for row in cross_resolution))
        self.assertTrue(all(row["replay_resolution_N"] == 96 for row in cross_resolution))
        self.assertTrue(all(row["replay_horizon_T"] == 6000 for row in cross_resolution))
        self.assertEqual(rows[-1]["group"], "feb56dc7_control")
        self.assertEqual(rows[-1]["kind"], "reference_control")

    def test_jitter_state_is_deterministic_and_clamped(self):
        from jax_scout import core_saturation_robustness as robust

        base = dict(css.FEB)
        first = robust.build_parameter_state(base, parameter_state="jitter_01", jitter_seed=314159)
        second = robust.build_parameter_state(base, parameter_state="jitter_01", jitter_seed=314159)

        self.assertEqual(first, second)
        self.assertGreaterEqual(first["param_eta"], css.REGIME["param_eta"][0])
        self.assertLessEqual(first["param_eta"], css.REGIME["param_eta"][1])
        self.assertGreaterEqual(first["param_a"], css.REGIME["param_a"][0])
        self.assertLessEqual(first["param_a"], css.REGIME["param_a"][1])

    def test_summarize_robustness_assigns_group_labels(self):
        from jax_scout import core_saturation_robustness as robust

        rows = [
            {"group": "k1_low_mass_branch", "class": "TRUE_SATURATED_BOUND_STATE", "diagnostic_label": "SATURATED_BOUND_STATE", "node_count_last": 2, "late_slope": 1.0e-4, "er_final": 1.0, "compactness_max": 10.0, "high_k_fraction_max": 0.04, "ic_seed": 20260619, "parameter_state": "base", "replay_target_initial_mass": 4000.0},
            {"group": "k1_low_mass_branch", "class": "NEAR_SATURATED_BOUND_STATE", "diagnostic_label": "SATURATED_BOUND_STATE", "node_count_last": 2, "late_slope": 1.5e-4, "er_final": 0.9, "compactness_max": 11.0, "high_k_fraction_max": 0.05, "ic_seed": 20260620, "parameter_state": "jitter_01", "replay_target_initial_mass": 4800.0},
            {"group": "k1_high_mass_failure", "class": "LATE_BLOWUP_REJECT", "diagnostic_label": "FRAGMENTING_BLOWUP", "node_count_last": 4, "late_slope": 1.0e-2, "er_final": 3.1, "compactness_max": 22.0, "high_k_fraction_max": 0.09, "ic_seed": 20260619, "parameter_state": "base", "replay_target_initial_mass": 7200.0},
            {"group": "k1_high_mass_failure", "class": "LATE_BLOWUP_REJECT", "diagnostic_label": "FRAGMENTING_BLOWUP", "node_count_last": 5, "late_slope": 8.0e-3, "er_final": 3.3, "compactness_max": 24.0, "high_k_fraction_max": 0.08, "ic_seed": 20260620, "parameter_state": "jitter_01", "replay_target_initial_mass": 9600.0},
            {"group": "k6_distributed_branch", "class": "TRUE_SATURATED_BOUND_STATE", "diagnostic_label": "SATURATED_BOUND_STATE", "node_count_last": 5, "late_slope": 3.0e-5, "er_final": 1.1, "compactness_max": 8.0, "high_k_fraction_max": 0.03, "ic_seed": 20260619, "parameter_state": "base", "replay_target_initial_mass": 9600.0},
            {"group": "k6_distributed_branch", "class": "TRUE_SATURATED_BOUND_STATE", "diagnostic_label": "SATURATED_BOUND_STATE", "node_count_last": 4, "late_slope": 4.0e-5, "er_final": 1.0, "compactness_max": 9.0, "high_k_fraction_max": 0.04, "ic_seed": 20260620, "parameter_state": "jitter_01", "replay_target_initial_mass": 12800.0},
            {"group": "feb56dc7_control", "class": "TRUE_SATURATED_BOUND_STATE", "diagnostic_label": "SATURATED_BOUND_STATE", "node_count_last": 4, "late_slope": 1.0e-6, "er_final": 1.0, "compactness_max": 7.0, "high_k_fraction_max": 0.02, "ic_seed": 20260619, "parameter_state": "base", "replay_target_initial_mass": None},
        ]

        summary = robust.summarize_results(rows)

        self.assertEqual(summary["group_verdicts"]["k1_low_mass_branch"], "K1_LOW_MASS_BRANCH_ROBUST")
        self.assertEqual(summary["group_verdicts"]["k1_high_mass_failure"], "K1_FAILURE_BOUNDARY_ROBUST")
        self.assertEqual(summary["group_verdicts"]["k6_distributed_branch"], "K6_DISTRIBUTED_BRANCH_ROBUST")
        self.assertEqual(summary["group_verdicts"]["feb56dc7_control"], "FEB_CONTROL_REPRODUCED")


if __name__ == "__main__":
    unittest.main()
