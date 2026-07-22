from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from ..io_utils import FileTracker, metric_paths, relpath, result_template
from ..plotting import heatmap_plot
from ..report_writer import metric_report, write_metric_outputs

FORMULA = "phase x speed collision outcome grid with conservation/radiation telemetry where available"


OUTCOME_CODES = {"CAPTURE": 1.0, "TRANSMIT": 2.0, "SCATTER": 2.0, "BOUND": 1.0, "UNKNOWN": 0.0}


def phase_label(value: Any, folder: Path) -> str:
    if value is not None:
        return f"{float(value):.2f}"
    name = folder.name
    if "ANTIPHASE" in name:
        return f"{math.pi:.2f}"
    if "pi2" in name:
        return f"{math.pi / 2:.2f}"
    if "3pi4" in name:
        return f"{3 * math.pi / 4:.2f}"
    if "7pi8" in name:
        return f"{7 * math.pi / 8:.2f}"
    return "0.00"


def run(config: dict[str, Any], output_root: Path, input_override: str | None = None, make_plots: bool = True) -> dict[str, Any]:
    tracker = FileTracker()
    result = result_template("v5", FORMULA)
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for root in metric_paths(config, "v5", input_override):
        summary = root / "summary.json"
        if not summary.exists():
            warnings.append(f"Missing collision summary: {relpath(summary)}")
            continue
        data = tracker.read_json(summary)
        dphi = data.get("dphi")
        for item in data.get("ladder", []) + data.get("collisions", []):
            rows.append(
                {
                    "source": relpath(summary),
                    "phase": phase_label(item.get("dphi", dphi), root),
                    "vfrac": item.get("vfrac", item.get("v_over_c", item.get("v"))),
                    "speed": item.get("v"),
                    "outcome": item.get("outcome", "UNKNOWN"),
                    "reason": item.get("reason", ""),
                    "radiation_fraction": item.get("rad_frac_end", item.get("rad_frac")),
                    "mass_retention": item.get("mass_ret_end", item.get("mass_end")),
                    "dE_rel_max": item.get("dE_rel_max"),
                    "dQ_rel_max": item.get("dQ_rel_max"),
                    "elasticity": item.get("elasticity_est", item.get("elasticity")),
                    "identity_ambiguous": item.get("identity_ambiguous", ""),
                }
            )
    result["data_sources"] = [relpath(p) for p in metric_paths(config, "v5", input_override) if p.exists()]
    result["files_read"] = tracker.files_read
    if not rows:
        result["caveats"] = warnings + ["No collision ladder rows found."]
        write_metric_outputs(output_root / "v5", result, rows, metric_report("v5", "V5 C3 Collision Phase Grid", result, rows, warnings), warnings)
        return result
    phases = sorted({r["phase"] for r in rows}, key=lambda x: float(x))
    speeds = sorted({str(r["vfrac"]) for r in rows}, key=lambda x: float(x) if x not in {"None", ""} else -1.0)
    missing = 0
    for p in phases:
        for s in speeds:
            if not any(r["phase"] == p and str(r["vfrac"]) == s for r in rows):
                missing += 1
    result["status"] = "RAN"
    result["validation_depth"] = "INTERNAL_DATA_BEHAVIOUR_CHECK"
    outcome_counts: dict[str, int] = {}
    for r in rows:
        key = str(r["outcome"])
        outcome_counts[key] = outcome_counts.get(key, 0) + 1
    antiphase_transmission = [
        r
        for r in rows
        if abs(float(r["phase"]) - math.pi) < 0.15
        and str(r["outcome"]).upper() in {"PASS_THROUGH", "TRANSMIT", "SCATTER"}
    ]
    non_antiphase = [r for r in rows if abs(float(r["phase"]) - math.pi) >= 0.15]
    non_antiphase_capture = [
        r for r in non_antiphase if str(r["outcome"]).upper() in {"CAPTURE", "BOUND"}
    ]
    capture_fraction = float(len(non_antiphase_capture) / len(non_antiphase)) if non_antiphase else None
    result["fit_parameters"] = {"phase_count": len(phases), "speed_count": len(speeds)}
    result["fit_quality"] = {
        "n_cells": len(rows),
        "n_missing_cells": missing,
        "phase_values": phases,
        "speed_values": speeds,
        "outcome_counts": outcome_counts,
        "anti_phase_transmission_cells": len(antiphase_transmission),
        "non_antiphase_capture_fraction": capture_fraction,
        "max_energy_drift": _finite_max([r.get("dE_rel_max") for r in rows]),
        "max_charge_drift": _finite_max([r.get("dQ_rel_max") for r in rows]),
        "rows": len(rows),
        "missing_phase_speed_cells": missing,
        "outcomes": sorted({str(r["outcome"]) for r in rows}),
        "max_dE_rel": _finite_max([r.get("dE_rel_max") for r in rows]),
        "max_dQ_rel": _finite_max([r.get("dQ_rel_max") for r in rows]),
    }
    result["behavioural_result"] = (
        "PASS"
        if antiphase_transmission and capture_fraction is not None and capture_fraction >= 0.7
        else "PARTIAL"
        if outcome_counts
        else "INCONCLUSIVE"
    )
    result["match_level_candidate"] = "CLOSE ANALOGUE"
    result["caveats"] = warnings + ["Uses existing outcome labels only; no collision reinterpretation is performed."]
    if make_plots and len(phases) >= 2 and len(speeds) >= 2:
        grid = np.full((len(phases), len(speeds)), np.nan)
        for r in rows:
            i = phases.index(r["phase"])
            j = speeds.index(str(r["vfrac"]))
            grid[i, j] = OUTCOME_CODES.get(str(r["outcome"]).upper(), 0.0)
        heatmap_plot(output_root / "v5" / "figure.png", grid, speeds, phases, "V5 collision outcome grid")
    write_metric_outputs(output_root / "v5", result, rows, metric_report("v5", "V5 C3 Collision Phase Grid", result, rows, warnings), warnings)
    return result


def _finite_max(values: list[Any]) -> float | None:
    arr = np.array([float(v) for v in values if v is not None and str(v) != "nan"], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.max()) if arr.size else None
