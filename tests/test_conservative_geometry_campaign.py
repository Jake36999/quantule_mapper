import json
import math

import numpy as np


def test_regular_tetrahedron_points_preserve_target_spacing():
    from tools.conservative_geometry_campaign import geometry_points_box, pairwise_periodic_box_distances

    points = geometry_points_box("tetrahedron", spacing_box=0.45)
    distances = pairwise_periodic_box_distances(points)

    assert points.shape == (4, 3)
    assert len(distances) == 6
    assert np.allclose(distances, np.full(6, 0.45), atol=1e-12)


def test_triangular_prism_has_equal_triangle_and_vertical_edges():
    from tools.conservative_geometry_campaign import geometry_points_box, pairwise_periodic_box_distances

    points = geometry_points_box("triangular_prism", spacing_box=0.36)
    distances = np.sort(pairwise_periodic_box_distances(points))

    assert points.shape == (6, 3)
    # Two triangle faces plus three vertical edges should give nine nearest-neighbour edges.
    assert np.allclose(distances[:9], np.full(9, 0.36), atol=1e-12)


def test_ablation_and_perturbation_controls_are_encoded_in_case_config():
    from tools.conservative_geometry_campaign import build_case_config

    cfg = build_case_config(
        case_id="unit_ablate",
        template="ablated_triangle",
        N=32,
        steps=20,
        spacing_box=0.45,
        dt=0.001,
        L=10.0,
        perturbations={"remove_node": 1, "phase_offsets": [0.0, 0.25]},
    )

    assert cfg["phase_d_regime"] == "conservative_c2_geometry_diagnostic_only"
    assert cfg["solver_component"] == "tools.conservative_geometry_campaign.ConservativeCupySolver"
    assert cfg["params"]["kinetic_mode"] == "conservative"
    assert cfg["geometry"]["template"] == "ablated_triangle"
    assert cfg["geometry"]["perturbations"]["remove_node"] == 1
    assert "dissipative" in " ".join(cfg["limitations"]).lower()
    assert len(cfg["config_hash"]) == 64


def test_recommended_manifest_is_conservative_only_medium_campaign(tmp_path):
    from tools.conservative_geometry_campaign import write_recommended_manifest

    path = tmp_path / "manifest.json"
    manifest = write_recommended_manifest(path, N=48, steps=2000)

    templates = [case["template"] for case in manifest["cases"]]
    spacings = [case["spacing_box"] for case in manifest["cases"] if case["template"] == "triangle"]

    assert manifest["regime"] == "conservative_c2_only"
    assert manifest["runtime"]["N"] == 48
    assert manifest["runtime"]["steps"] == 2000
    assert spacings == [0.36, 0.45]
    assert "ablated_triangle" in templates
    assert "tetrahedron" in templates
    assert any(t in templates for t in ("triangular_prism", "octahedron"))
    assert json.loads(path.read_text())["regime"] == "conservative_c2_only"


def test_build_ic_applies_amplitude_phase_and_noise_controls():
    from tools.conservative_geometry_campaign import build_geometry_ic, geometry_points_box

    points = geometry_points_box("triangle", spacing_box=0.36)
    psi_a = build_geometry_ic(
        N=24,
        L=10.0,
        points_box=points,
        width_box=0.083,
        amplitude=1.0,
        amplitude_factors=[1.0, 0.5, 1.0],
        phases=[0.0, math.pi / 2.0, 0.0],
        noise_level=0.0,
        seed=123,
    )
    psi_b = build_geometry_ic(
        N=24,
        L=10.0,
        points_box=points,
        width_box=0.083,
        amplitude=1.0,
        amplitude_factors=[1.0, 0.5, 1.0],
        phases=[0.0, math.pi / 2.0, 0.0],
        noise_level=1e-4,
        seed=123,
    )

    assert psi_a.shape == (24, 24, 24)
    assert psi_a.dtype == np.complex128
    assert not np.allclose(psi_a, psi_b)


def test_arrangement_outcome_labels_planar_volumetric_and_fragmented():
    from tools.conservative_geometry_campaign import arrangement_outcome

    planar = {
        "initial_node_count": 3,
        "final_node_count": 3,
        "z_spread_box": 0.0,
        "planarity_score": 0.0,
        "final_finite": True,
        "fail_reason": "",
    }
    volumetric = {**planar, "initial_node_count": 4, "final_node_count": 4, "z_spread_box": 0.25, "planarity_score": 0.4}
    fragmented = {**planar, "initial_node_count": 3, "final_node_count": 5}

    assert arrangement_outcome(planar) == "stayed_planar"
    assert arrangement_outcome(volumetric) == "stayed_volumetric"
    assert arrangement_outcome(fragmented) == "fragmented"


def test_campaign_output_paths_include_stable_requested_names(tmp_path):
    from tools.conservative_geometry_campaign import campaign_output_paths

    paths = campaign_output_paths(tmp_path, "example_campaign")

    assert paths["stable_csv"].name == "campaign_results.csv"
    assert paths["stable_report"].name == "campaign_report.md"
    assert paths["resolved_manifest"].name == "campaign_manifest_resolved.json"
    assert paths["dashboard"].name == "campaign_dashboard.html"
    assert paths["named_csv"].name == "example_campaign_results.csv"


def test_profile_overlap_detects_same_and_disjoint_profiles():
    from tools.conservative_geometry_campaign import profile_overlap

    rho_a = np.zeros((8, 8, 8), dtype=float)
    rho_a[2, 2, 2] = 1.0
    rho_b = rho_a.copy()
    rho_c = np.zeros_like(rho_a)
    rho_c[5, 5, 5] = 1.0

    assert profile_overlap(rho_a, rho_b) == 1.0
    assert profile_overlap(rho_a, rho_c) == 0.0


def test_threshold_sensitivity_counts_multiple_rho_levels():
    from tools.conservative_geometry_campaign import threshold_sensitivity_counts

    psi = np.zeros((16, 16, 16), dtype=np.complex128)
    psi[3, 3, 3] = 1.0
    psi[10, 10, 10] = 0.75

    counts = threshold_sensitivity_counts(psi, fractions=(0.2, 0.7, 0.9))

    assert counts["thr_0.2"] == 2
    assert counts["thr_0.7"] == 1
    assert counts["thr_0.9"] == 1


def test_numerical_loss_warning_uses_fractional_norm_change():
    from tools.conservative_geometry_campaign import numerical_loss_warning

    assert numerical_loss_warning(100.0, 99.9) == ""
    assert "changed" in numerical_loss_warning(100.0, 95.0)


def test_invariant_audit_paths_include_requested_names(tmp_path):
    from tools.conservative_geometry_campaign import invariant_audit_paths

    paths = invariant_audit_paths(tmp_path)

    assert paths["csv"].name == "conservative_invariant_audit_results.csv"
    assert paths["report"].name == "conservative_invariant_audit_report.md"


def test_steps_for_physical_time_rounds_to_nearest_integer():
    from tools.conservative_geometry_campaign import steps_for_physical_time

    assert steps_for_physical_time(4.0, 0.001) == 4000
    assert steps_for_physical_time(4.0, 0.0005) == 8000
    assert steps_for_physical_time(4.0, 0.00025) == 16000


def test_high_k_fraction_detects_masked_tail_energy():
    from tools.conservative_geometry_campaign import high_k_fraction

    k_sq = np.zeros((4, 4, 4), dtype=float)
    k_sq[0, 0, 0] = 1.0
    k_sq[3, 3, 3] = 100.0
    psi_k = np.zeros((4, 4, 4), dtype=np.complex128)
    psi_k[0, 0, 0] = 3.0
    psi_k[3, 3, 3] = 4.0
    mask = k_sq <= 10.0

    assert np.isclose(high_k_fraction(psi_k, mask), 16.0 / 25.0)


def test_norm_loss_isolation_paths_include_requested_names(tmp_path):
    from tools.conservative_geometry_campaign import norm_loss_isolation_paths

    paths = norm_loss_isolation_paths(tmp_path)

    assert paths["csv"].name == "conservative_norm_loss_isolation_results.csv"
    assert paths["report"].name == "conservative_norm_loss_isolation_report.md"
    assert paths["norm_plot"].name == "conservative_norm_loss_norm_vs_time.png"
    assert paths["rho_plot"].name == "conservative_norm_loss_rho_max_vs_time.png"
