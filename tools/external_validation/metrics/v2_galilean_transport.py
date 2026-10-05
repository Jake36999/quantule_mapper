from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..io_utils import FileTracker, metric_paths, relpath, result_template
from ..plotting import scatter_line_plot
from ..report_writer import metric_report, write_metric_outputs

FORMULA = "v = 2Dk for corrected flat-NLS C2 transport"


def fit_line(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": 1.0 if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot),
    }


def run(config: dict[str, Any], output_root: Path, input_override: str | None = None, make_plots: bool = True) -> dict[str, Any]:
    tracker = FileTracker()
    result = result_template("v2", FORMULA)
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for root in metric_paths(config, "v2", input_override):
        path = root / "r3_n96.json"
        if not path.exists():
            warnings.append(f"Missing corrected C2.7 summary: {relpath(path)}")
            continue
        data = tracker.read_json(path)
        D = float(data.get("D", 1.0))
        if "D" not in data:
            warnings.append(f"{relpath(path)} lacks explicit D; using project C2.7 convention D=1.0.")
        for b in data.get("boosts", []):
            rows.append(
                {
                    "source": relpath(path),
                    "n": b.get("n"),
                    "k": b.get("k"),
                    "v_measured": b.get("v"),
                    "v_expected_2Dk": b.get("v_pred", 2.0 * D * float(b.get("k", 0.0))),
                    "v_frac": b.get("v_frac"),
                    "mass_ret": b.get("mass_ret"),
                }
            )
    result["data_sources"] = [relpath(p) for p in metric_paths(config, "v2", input_override) if p.exists()]
    result["files_read"] = tracker.files_read
    if len(rows) < 2:
        result["caveats"] = warnings + ["Need at least two corrected C2.7 boosts for a line fit."]
        write_metric_outputs(output_root / "v2", result, rows, metric_report("v2", "V2 Galilean Transport", result, rows, warnings), warnings)
        return result
    x = np.array([float(r["k"]) for r in rows], dtype=float)
    y = np.array([float(r["v_measured"]) for r in rows], dtype=float)
    fit = fit_line(x, y)
    expected = np.array([float(r["v_expected_2Dk"]) for r in rows], dtype=float)
    fit["expected_slope_2D"] = float(np.mean(expected / x))
    fit["slope_relative_error"] = float(abs(fit["slope"] - fit["expected_slope_2D"]) / abs(fit["expected_slope_2D"]))
    slope_percent_error = 100.0 * fit["slope_relative_error"]
    mass_vals = [float(r["mass_ret"]) for r in rows if r["mass_ret"] is not None]
    result["status"] = "RAN"
    result["validation_depth"] = "INTERNAL_DATA_BEHAVIOUR_CHECK"
    result["fit_parameters"] = fit
    result["fit_quality"] = {
        "n_points": len(rows),
        "D_used": 1.0,
        "source_of_D": "project C2.7 convention or explicit summary field if present",
        "slope": fit["slope"],
        "expected_slope_2D": fit["expected_slope_2D"],
        "slope_percent_error": slope_percent_error,
        "intercept": fit["intercept"],
        "r_squared": fit["r2"],
        "mean_mass_retention": float(np.mean(mass_vals)),
        "mass_retention_min": float(np.min(mass_vals)),
        "n_boosts": len(rows),
    }
    result["behavioural_result"] = (
        "PASS"
        if fit["r2"] > 0.999 and slope_percent_error < 1.0 and result["fit_quality"]["mass_retention_min"] > 0.99
        else "PARTIAL"
    )
    result["match_level_candidate"] = "EXACT EQUATION"
    result["caveats"] = warnings + ["Uses corrected C2.7 data only; old C2.6 bugged geometry-off data are excluded."]
    if make_plots:
        scatter_line_plot(output_root / "v2" / "figure.png", x, y, fit["slope"] * x + fit["intercept"], "V2 v=2Dk transport", "k", "v_measured")
    write_metric_outputs(output_root / "v2", result, rows, metric_report("v2", "V2 Galilean Transport", result, rows, warnings), warnings)
    return result
