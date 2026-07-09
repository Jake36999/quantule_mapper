import importlib.util
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "triangle_layout_diagnostic.py"
spec = importlib.util.spec_from_file_location("triangle_layout_diagnostic", MODULE_PATH)
tri = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = tri
spec.loader.exec_module(tri)


def test_periodic_pairwise_distances_use_minimal_image():
    points = np.array(
        [
            [1.0, 1.0, 1.0],
            [95.0, 1.0, 1.0],
            [1.0, 49.0, 1.0],
        ]
    )

    distances = tri.pairwise_periodic_distances(points, box_size_vox=96, normalize=True)

    assert np.allclose(sorted(distances), sorted([2 / 96, 48 / 96, math.sqrt(2**2 + 48**2) / 96]))


def test_triangle_shape_classifier_distinguishes_equilateral_stretched_and_line_like():
    assert tri.classify_triangle_shape([0.32, 0.33, 0.34]) == "equilateral-like"
    assert tri.classify_triangle_shape([0.28, 0.31, 0.49]) == "stretched"
    assert tri.classify_triangle_shape([0.1, 0.41, 0.50]) == "line-like"


def test_equilateral_triangle_points_are_centered_on_midplane():
    pts = tri.equilateral_triangle_points(N=96, side_length_box=0.32)

    assert pts.shape == (3, 3)
    assert np.allclose(pts[:, 2], 48.0)
    assert np.allclose(pts.mean(axis=0), [48.0, 48.0, 48.0])
    distances = sorted(tri.pairwise_periodic_distances(pts, box_size_vox=96, normalize=True))
    assert np.allclose(distances, [0.32, 0.32, 0.32], atol=1e-6)


def test_build_triangle_ic_has_three_balanced_peaks_without_noise():
    psi = tri.build_triangle_ic(
        N=64,
        L=10.0,
        side_length_box=0.36,
        width_box=1 / 12,
        amplitude=1.0,
        phases=[0.0, 0.0, 0.0],
        noise_level=0.0,
    )

    metrics = tri.node_geometry_from_psi(psi, L=10.0, expected_nodes=3)

    assert metrics["node_count"] == 3
    assert metrics["shape_class"] == "equilateral-like"
    assert metrics["node_mass_cv"] < 0.05
    assert metrics["rho_peak_cv"] < 0.05
    assert all(abs(d - 0.36) < 0.05 for d in metrics["pairwise_distances_box"])
