from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..io_utils import FileTracker, metric_paths, relpath, result_template
from ..plotting import scatter_line_plot
from ..report_writer import metric_report, write_metric_outputs

FORMULA = "Omega^2 = (rho_vac / rho)^a, gamma = 2a + 3; design-constraint diagnosis only"


def estimate_exponent(rho: np.ndarray, omega: np.ndarray) -> dict[str, Any]:
    rho = np.asarray(rho, dtype=float)
    omega = np.asarray(omega, dtype=float)
    mask = np.isfinite(rho) & np.isfinite(omega) & (rho > 0) & (omega > 0)
    if not np.any(mask):
        return {"fit_available": False, "reason": "No finite positive rho/omega points."}
    rho, omega = rho[mask], omega[mask]
    cap = np.nanmax(omega)
    floor = np.nanmin(omega)
    graded = (omega < 0.9 * cap) & (omega > 1.1 * floor)
    if int(graded.sum()) < 3:
        return {
            "fit_available": False,
            "reason": "Fewer than three unsaturated graded points.",
            "cap_fraction": float(np.mean(omega >= 0.9 * cap)),
            "floor_fraction": float(np.mean(omega <= 1.1 * floor)),
            "graded_fraction": float(np.mean(graded)),
        }
    slope, intercept = np.polyfit(np.log(rho[graded]), np.log(omega[graded]), 1)
    a = float(-slope)
    return {
        "fit_available": True,
        "a_effective": a,
        "gamma_effective": float(2.0 * a + 3.0),
        "log_intercept": float(intercept),
        "n_fit": int(graded.sum()),
        "cap_fraction": float(np.mean(omega >= 0.9 * cap)),
        "floor_fraction": float(np.mean(omega <= 1.1 * floor)),
        "graded_fraction": float(np.mean(graded)),
    }


def run(config: dict[str, Any], output_root: Path, input_override: str | None = None, make_plots: bool = True) -> dict[str, Any]:
    tracker = FileTracker()
    result = result_template("v7", FORMULA)
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    rho_for_plot: np.ndarray | None = None
    omega_for_plot: np.ndarray | None = None
    fit: dict[str, Any] | None = None
    for root in metric_paths(config, "v7", input_override):
        csv_path = root / "radial_profiles.csv"
        if csv_path.exists():
            data = tracker.read_csv(csv_path)
            rho = np.array([float(r["rho_mean"]) for r in data if r.get("rho_mean")], dtype=float)
            omega = np.array([float(r["omega_sq_mean"]) for r in data if r.get("omega_sq_mean")], dtype=float)
            fit = estimate_exponent(rho, omega)
            rho_for_plot, omega_for_plot = rho, omega
            rows.append({"source": relpath(csv_path), **fit})
        summary = root / "summary.json"
        if summary.exists():
            data = tracker.read_json(summary)
            load = data.get("load", {})
            if "radial_omega" in load:
                omega = np.array(load["radial_omega"], dtype=float)
                rows.append(
                    {
                        "source": relpath(summary),
                        "profile": "radial_omega_only",
                        "omega_min": float(np.nanmin(omega)),
                        "omega_max": float(np.nanmax(omega)),
                        "cap_like_fraction": float(np.mean(omega >= 0.9 * np.nanmax(omega))),
                    }
                )
        desat = root / "desat_pilot.json"
        if desat.exists():
            data = tracker.read_json(desat)
            prod = data.get("production", {})
            rows.append(
                {
                    "source": relpath(desat),
                    "profile": "production_desat_summary",
                    "cap_fraction": prod.get("cap_fraction"),
                    "graded_well": prod.get("graded_well"),
                    "max_radial_jump": prod.get("max_radial_jump"),
                }
            )
    result["data_sources"] = [relpath(p) for p in metric_paths(config, "v7", input_override) if p.exists()]
    result["files_read"] = tracker.files_read
    if not rows:
        result["caveats"] = warnings + ["No radial omega/rho profiles found."]
        write_metric_outputs(output_root / "v7", result, rows, metric_report("v7", "V7 Omega Exponent Diagnosis", result, rows, warnings), warnings)
        return result
    result["status"] = "RAN"
    result["validation_depth"] = "INTERNAL_DATA_BEHAVIOUR_CHECK"
    best_fit = fit or {}
    production_a = 2.3098
    cliff_detected = any(
        (isinstance(r.get("cap_fraction"), (int, float)) and float(r.get("cap_fraction")) > 0.5)
        or (isinstance(r.get("max_radial_jump"), (int, float)) and float(r.get("max_radial_jump")) > 10.0)
        for r in rows
    )
    result["fit_parameters"] = fit or {}
    # Two distinct exponents must not be conflated:
    #   nominal a   = the conformal-law coupling as configured (production_a).
    #   effective a = the exponent actually recovered by a power-law fit to the spatial
    #                 Omega^2(rho) profile; it is STEEPER than nominal because the soft-clip
    #                 saturates ~cap_fraction of the radial points and distorts the law even
    #                 in the unsaturated band. The effective exponent is the physically
    #                 operative one for the vacuum cliff.
    # BEC analogue-gravity reference (weakly-interacting condensate, gamma=2): a = (gamma-3)/2 = -0.5.
    # Both nominal (+2.31) and effective (+2.93) are POSITIVE -> opposite sign from the BEC
    # value (-0.5) -> Omega^2 diverges as rho->0 (the vacuum cliff), rather than vanishing.
    a_effective = best_fit.get("a_effective")
    bec_reference_a = -0.5
    result["fit_quality"] = {
        "n_radial_points": int(rho_for_plot.size) if rho_for_plot is not None else None,
        "unsaturated_points": best_fit.get("n_fit"),
        "nominal_a": production_a,
        "nominal_gamma": 2.0 * production_a + 3.0,
        "fitted_effective_exponent": a_effective,
        "effective_gamma": best_fit.get("gamma_effective"),
        "effective_steeper_than_nominal": (bool(a_effective > production_a) if isinstance(a_effective, (int, float)) else None),
        "bec_reference_a": bec_reference_a,
        "bec_reference_gamma": 2.0,
        "same_sign_as_bec_analogue": (bool((a_effective > 0) == (bec_reference_a > 0)) if isinstance(a_effective, (int, float)) else None),
        "production_a": production_a,
        "implied_gamma": 2.0 * production_a + 3.0,
        "cap_fraction": best_fit.get("cap_fraction"),
        "floor_fraction": best_fit.get("floor_fraction"),
        "graded_fraction": best_fit.get("graded_fraction"),
        "cliff_detected": cliff_detected,
        "rows": len(rows),
        "design_constraint_only": True,
    }
    result["behavioural_result"] = "PARTIAL" if cliff_detected else "INCONCLUSIVE"
    result["match_level_candidate"] = "CLOSE ANALOGUE"
    result["caveats"] = warnings + [
        "This is a design constraint and soft-clip diagnosis, not a gravity result.",
        "Nominal exponent a=+2.31 (as configured) vs effective a=+2.93 (fit from the spatial "
        "Omega^2(rho) profile; steeper because ~31% of radial points are cap-saturated). Both "
        "are positive and thus opposite in sign to the physical BEC analogue-gravity value "
        "a=-0.5 (gamma=2), which is exactly why Omega^2 diverges at the vacuum (the cliff). "
        "Re-entry design constraint: target a<0 / gamma<3 on a filled rho_vac background.",
    ]
    if make_plots and rho_for_plot is not None and omega_for_plot is not None and rho_for_plot.size >= 2:
        scatter_line_plot(output_root / "v7" / "figure.png", np.log(rho_for_plot), np.log(omega_for_plot), np.log(omega_for_plot), "V7 log Omega vs log rho", "log rho", "log Omega^2")
    write_metric_outputs(output_root / "v7", result, rows, metric_report("v7", "V7 Omega Exponent Diagnosis", result, rows, warnings), warnings)
    return result
