import math

import numpy as np

from tools.analyze_gravity_geometry_characterization import (
    classify_curve_region,
    evaluate_geometry_curve,
    production_geometry_params,
    radial_profile,
    score_scan_candidate,
)


def test_production_geometry_curve_is_finite_and_capped():
    params = production_geometry_params()
    rho = np.array([1e-12, 1e-4, 1.0, 2.0], dtype=np.float64)

    rows = evaluate_geometry_curve(rho, params)
    impl = np.array([row["omega_sq_impl"] for row in rows])
    nominal = np.array([row["omega_sq_nominal"] for row in rows])

    assert np.all(np.isfinite(impl))
    assert np.all(impl <= params["omega_sq_max"] * (1.0 + 1e-12))
    assert impl[0] > 0.9 * params["omega_sq_max"]
    assert nominal[0] > impl[0]


def test_classify_curve_region_marks_cap_and_graded():
    params = production_geometry_params()

    assert classify_curve_region(0.95 * params["omega_sq_max"], -0.01, params) == "cap_saturated"
    assert classify_curve_region(1.0, -1.0, params) == "graded"


def test_score_scan_candidate_prefers_graded_over_flat():
    rho = np.logspace(-4, math.log10(2.0), 128)
    graded = score_scan_candidate(
        rho,
        {"omega_sq_impl": rho ** -0.5, "log_slope": np.full_like(rho, -0.5)},
        {"omega_sq_min": 1e-9, "omega_sq_max": 1e6},
    )
    flat = score_scan_candidate(
        rho,
        {"omega_sq_impl": np.full_like(rho, 10.0), "log_slope": np.zeros_like(rho)},
        {"omega_sq_min": 1e-9, "omega_sq_max": 1e6},
    )

    assert graded["score"] > flat["score"]
    assert graded["graded_fraction"] > flat["graded_fraction"]


def test_radial_profile_returns_requested_bin_count():
    field = np.ones((8, 8, 8), dtype=np.float64)
    radii, values = radial_profile(field, center=(4, 4, 4), L=10.0, bins=6)

    assert len(radii) == 6
    assert len(values) == 6
    assert np.all(np.isfinite(values))
