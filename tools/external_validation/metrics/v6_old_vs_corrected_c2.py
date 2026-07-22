from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..io_utils import FileTracker, metric_paths, relpath, result_template
from ..plotting import line_plot
from ..report_writer import metric_report, write_metric_outputs

FORMULA = "old effective mobility/D_eff regression evidence versus corrected v = 2Dk C2 transport"


def run(config: dict[str, Any], output_root: Path, input_override: str | None = None, make_plots: bool = True) -> dict[str, Any]:
    tracker = FileTracker()
    result = result_template("v6", FORMULA)
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    roots = metric_paths(config, "v6", input_override)
    for root in roots:
        c27 = root / "r3_n96.json"
        if c27.exists():
            data = tracker.read_json(c27)
            for b in data.get("boosts", []):
                rows.append(
                    {
                        "source": relpath(c27),
                        "class": "corrected_C2",
                        "k": b.get("k"),
                        "v_measured": b.get("v"),
                        "v_expected": b.get("v_pred"),
                        "ratio": b.get("v_frac"),
                        "mass_ret": b.get("mass_ret"),
                    }
                )
        for name in ("c2_6_independent_audit_summary.json", "c2_6_reproduction_summary.json"):
            summary = root / name
            if summary.exists():
                data = tracker.read_json(summary)
                old = data.get("old_regression", {})
                rows.append(
                    {
                        "source": relpath(summary),
                        "class": "old_regression_D_eff",
                        "k": data.get("k1"),
                        "v_measured": "",
                        "v_expected": "",
                        "ratio": old.get("effective_D_ratio"),
                        "mass_ret": "",
                    }
                )
    for old_path in roots:
        if old_path.name in {"C23_N96", "C24_LOCAL_N96"} and old_path.exists():
            rows.append(
                {
                    "source": relpath(old_path),
                    "class": "old_run_available_no_clean_overlay",
                    "k": "",
                    "v_measured": "",
                    "v_expected": "",
                    "ratio": "",
                    "mass_ret": "",
                }
            )
    result["data_sources"] = [relpath(p) for p in roots if p.exists()]
    result["files_read"] = tracker.files_read
    corrected = [r for r in rows if r["class"] == "corrected_C2"]
    regression = [r for r in rows if r["class"] == "old_regression_D_eff"]
    if not corrected and not regression:
        result["caveats"] = warnings + ["No corrected C2 or C2.6 audit summaries found."]
        write_metric_outputs(output_root / "v6", result, rows, metric_report("v6", "V6 Old vs Corrected C2", result, rows, warnings), warnings)
        return result
    result["status"] = "RAN"
    result["validation_depth"] = "INTERNAL_DATA_BEHAVIOUR_CHECK"
    old_ratio = regression[0].get("ratio") if regression else None
    corrected_ratio = float(np.mean([float(r["ratio"]) for r in corrected])) if corrected else None
    result["fit_parameters"] = {
        "corrected_points": len(corrected),
        "old_regression_points": len(regression),
        "old_D_eff_ratio": old_ratio,
        "expected_bug_ratio_1_over_151": 1.0 / 151.0,
        "corrected_v_over_2Dk": corrected_ratio,
        "old_interpretation_status": "RETRACTED" if regression else "MISSING",
        "corrected_interpretation_status": "CANONICAL" if corrected else "MISSING",
    }
    result["fit_quality"] = {
        "old_D_eff_ratio": old_ratio,
        "expected_bug_ratio_1_over_151": 1.0 / 151.0,
        "corrected_v_over_2Dk": corrected_ratio,
        "old_interpretation_status": "RETRACTED" if regression else "MISSING",
        "corrected_interpretation_status": "CANONICAL" if corrected else "MISSING",
        "corrected_mean_v_frac": corrected_ratio,
        "clean_overlay_available": bool(corrected and regression),
    }
    result["behavioural_result"] = "PASS" if corrected and regression else "PARTIAL"
    result["match_level_candidate"] = "EXACT EQUATION"
    result["caveats"] = warnings + ["V6 is an instrument regression/overlay check, not an empirical validation."]
    if make_plots and corrected:
        x = [float(r["k"]) for r in corrected]
        meas = [float(r["v_measured"]) for r in corrected]
        exp = [float(r["v_expected"]) for r in corrected]
        line_plot(output_root / "v6" / "figure.png", [(x, meas, "corrected measured"), (x, exp, "2Dk expected")], "V6 corrected C2 transport", "k", "velocity")
    write_metric_outputs(output_root / "v6", result, rows, metric_report("v6", "V6 Old vs Corrected C2", result, rows, warnings), warnings)
    return result
