from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..io_utils import FileTracker, metric_paths, relpath, result_template
from ..plotting import scatter_line_plot
from ..report_writer import metric_report, write_metric_outputs

FORMULA = "dQ/domega < 0 under the stated C3 Q-ball branch convention"


def run(config: dict[str, Any], output_root: Path, input_override: str | None = None, make_plots: bool = True) -> dict[str, Any]:
    tracker = FileTracker()
    result = result_template("v3", FORMULA)
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    reported_slope: float | None = None
    for root in metric_paths(config, "v3", input_override):
        path = root / "summary.json"
        if not path.exists():
            warnings.append(f"Missing C3 VK summary: {relpath(path)}")
            continue
        data = tracker.read_json(path)
        vk = data.get("G6_vk", {})
        reported_slope = vk.get("dQ_dw")
        for point in vk.get("points", []):
            rows.append(
                {
                    "source": relpath(path),
                    "omega": point.get("w"),
                    "mu": point.get("mu"),
                    "Q": point.get("Q"),
                    "residual": point.get("residual"),
                }
            )
    result["data_sources"] = [relpath(p) for p in metric_paths(config, "v3", input_override) if p.exists()]
    result["files_read"] = tracker.files_read
    if len(rows) < 2:
        result["caveats"] = warnings + ["Need at least two Q(omega) points."]
        write_metric_outputs(output_root / "v3", result, rows, metric_report("v3", "V3 VK Branch", result, rows, warnings), warnings)
        return result
    omega = np.array([float(r["omega"]) for r in rows], dtype=float)
    Q = np.array([float(r["Q"]) for r in rows], dtype=float)
    slope, intercept = np.polyfit(omega, Q, 1)
    diffs = np.diff(Q[np.argsort(omega)])
    monotone = bool(np.all(diffs < 0))
    sign_pass = bool(slope < 0 and monotone)
    result["status"] = "RAN"
    result["validation_depth"] = "INTERNAL_DATA_BEHAVIOUR_CHECK"
    result["fit_parameters"] = {
        "dQ_domega_fit": float(slope),
        "intercept": float(intercept),
        "dQ_domega_reported": reported_slope,
    }
    result["fit_quality"] = {
        "n_points": len(rows),
        "omega_min": float(np.min(omega)),
        "omega_max": float(np.max(omega)),
        "Q_min": float(np.min(Q)),
        "Q_max": float(np.max(Q)),
        "dQ_domega_fit": float(slope),
        "monotonic_decreasing": monotone,
        "sign_pass": sign_pass,
        "monotone_decreasing": monotone,
        "n_points": len(rows),
        "sign": "negative" if slope < 0 else "nonnegative",
    }
    result["behavioural_result"] = "PASS" if sign_pass else "FAIL"
    result["match_level_candidate"] = "SAME FAMILY"
    result["caveats"] = warnings + ["VK sign depends on branch variable convention; this report uses omega/w."]
    if make_plots:
        scatter_line_plot(output_root / "v3" / "figure.png", omega, Q, slope * omega + intercept, "V3 Q(omega) branch", "omega", "Q")
    write_metric_outputs(output_root / "v3", result, rows, metric_report("v3", "V3 VK Branch", result, rows, warnings), warnings)
    return result
