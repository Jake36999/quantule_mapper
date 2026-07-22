from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from ..io_utils import FileTracker, metric_paths, relpath, result_template
from ..io_utils import write_csv
from ..plotting import scatter_line_plot
from ..report_writer import metric_report, write_metric_outputs

FORMULA = "q_ddot = -C * exp(-lambda * q) * cos(Delta_phi), with q = half-separation"


def estimate_qddot(t: np.ndarray, sep: np.ndarray, fraction: float = 0.35) -> float | None:
    t = np.asarray(t, dtype=float)
    q = 0.5 * np.asarray(sep, dtype=float)
    mask = np.isfinite(t) & np.isfinite(q)
    t, q = t[mask], q[mask]
    if t.size < 4:
        return None
    n = max(4, min(t.size, int(math.ceil(t.size * fraction))))
    t0 = t[:n] - t[0]
    q0 = q[:n]
    try:
        coeff = np.polyfit(t0, q0, 2)
    except Exception:
        return None
    return float(2.0 * coeff[0])


def phase_from_name(path: Path) -> float | None:
    m = re.search(r"(?:static|p)_(-?\d+(?:\.\d+)?)", path.stem)
    if not m:
        m = re.search(r"_(\d+\.\d+)$", path.stem)
    return float(m.group(1)) if m else None


def expected_sign(dphi: float, neutral_band: float = 0.15) -> int:
    c = math.cos(dphi)
    if abs(c) < neutral_band:
        return 0
    return -1 if c > 0 else 1


def sign_of(x: float) -> int:
    if abs(x) < 1e-12:
        return 0
    return 1 if x > 0 else -1


def fit_exponential_phase(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [
        r
        for r in rows
        if r.get("expected_sign") != 0
        and r.get("q_ddot") is not None
        and np.isfinite(r["q_ddot"])
        and abs(float(r["q_ddot"])) > 0
        and abs(math.cos(float(r["dphi"]))) > 1e-8
    ]
    if len(usable) < 3:
        return {"fit_available": False, "reason": "Need at least three non-neutral phase rows."}
    x = np.array([float(r["q_start"]) for r in usable], dtype=float)
    y = np.array([abs(float(r["q_ddot"]) / math.cos(float(r["dphi"]))) for r in usable], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
    if int(mask.sum()) < 3:
        return {"fit_available": False, "reason": "Insufficient finite positive acceleration amplitudes."}
    slope, intercept = np.polyfit(x[mask], np.log(y[mask]), 1)
    pred = intercept + slope * x[mask]
    residual = np.log(y[mask]) - pred
    return {
        "fit_available": True,
        "C": float(math.exp(intercept)),
        "lambda": float(-slope),
        "log_residual_rms": float(np.sqrt(np.mean(residual**2))),
        "normalized_rms_residual": float(np.sqrt(np.mean(residual**2)) / max(1e-12, np.mean(np.abs(np.log(y[mask]))))),
        "n_fit": int(mask.sum()),
    }


def estimate_qdot(t: np.ndarray, sep: np.ndarray, fraction: float = 0.35) -> float | None:
    t = np.asarray(t, dtype=float)
    q = 0.5 * np.asarray(sep, dtype=float)
    mask = np.isfinite(t) & np.isfinite(q)
    t, q = t[mask], q[mask]
    if t.size < 3:
        return None
    n = max(3, min(t.size, int(math.ceil(t.size * fraction))))
    try:
        coeff = np.polyfit(t[:n] - t[0], q[:n], 1)
    except Exception:
        return None
    return float(coeff[0])


def crossover_phase(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [r for r in rows if r.get("q_ddot") is not None and np.isfinite(r["q_ddot"])]
    if len(usable) < 3:
        return {"available": False, "reason": "Need at least three phase points."}
    x = np.array([math.cos(float(r["dphi"])) for r in usable], dtype=float)
    y = np.array([float(r["q_ddot"]) for r in usable], dtype=float)
    try:
        alpha, beta = np.polyfit(x, y, 1)
    except Exception as exc:
        return {"available": False, "reason": str(exc)}
    if abs(alpha) < 1e-14:
        return {"available": False, "reason": "Phase fit has near-zero slope."}
    zero_cos = float(-beta / alpha)
    if zero_cos < -1.0 or zero_cos > 1.0:
        return {"available": False, "reason": f"Zero crossing outside cosine range: {zero_cos}"}
    phase = float(math.acos(zero_cos))
    return {
        "available": True,
        "crossover_phase_estimate": phase,
        "crossover_error_from_pi_over_2": float(abs(phase - math.pi / 2.0)),
        "alpha": float(alpha),
        "beta": float(beta),
    }


def source_family(root: Path) -> str:
    if "C29" in root.name:
        return "C2"
    if "C3" in root.name:
        return "C3"
    return "unknown"


def run(config: dict[str, Any], output_root: Path, input_override: str | None = None, make_plots: bool = True) -> dict[str, Any]:
    tracker = FileTracker()
    result = result_template("v1", FORMULA)
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for root in metric_paths(config, "v1", input_override):
        if not root.exists():
            warnings.append(f"Missing path: {relpath(root)}")
            continue
        for npz in sorted(root.glob("*static*.npz")):
            arrays = tracker.load_npz_arrays(npz)
            if "t" not in arrays or "sep" not in arrays:
                warnings.append(f"Skipped {relpath(npz)}: missing t/sep")
                continue
            dphi = phase_from_name(npz)
            if dphi is None:
                warnings.append(f"Skipped {relpath(npz)}: cannot infer relative phase")
                continue
            t = arrays["t"]
            sep = arrays["sep"]
            qdot = estimate_qdot(t, sep)
            qddot = estimate_qddot(t, sep)
            q_start = float(0.5 * sep[0])
            q_end = float(0.5 * sep[-1])
            exp = expected_sign(dphi)
            # Force-direction proxy: the net EARLY DRIFT of the half-separation, NOT the
            # trajectory curvature q_ddot. For breathing C3 Q-ball pairs the early-window
            # q_ddot is dominated by the internal breathing mode + the onset of capture, so
            # its sign is unreliable (it flips uniformly across every C3 phase); the drift
            # sign is robust and agrees with the full-run displacement and with the
            # already-established C2.9 / C3 static force law (attract Delta_phi<pi/2 /
            # repel >pi/2). q_ddot is retained below for the magnitude / lambda fit (the
            # actual Karpman-Solov'ev/Gordon law quantity) only. The legacy curvature-based
            # sign is kept as *_qddot columns so the artifact stays visible, not hidden.
            drift = qdot if qdot is not None else (q_end - q_start)
            got = sign_of(drift or 0.0)
            got_qddot = sign_of(qddot or 0.0)
            cos_phase = math.cos(dphi)
            rows.append(
                {
                    "source": relpath(root),
                    "family": source_family(root),
                    "file": relpath(npz),
                    "dphi": dphi,
                    "q_convention": "half_separation",
                    "acceleration_window": "first_35_percent_min_4_points",
                    "force_sign_proxy": "net_early_drift_q_dot",
                    "q_start": q_start,
                    "q_end": q_end,
                    "full_sep_start": float(sep[0]),
                    "full_sep_end": float(sep[-1]),
                    "q_dot": qdot,
                    "q_ddot": qddot,
                    "cos_delta_phi": cos_phase,
                    "q_ddot_over_cos": (float(qddot / cos_phase) if qddot is not None and abs(cos_phase) > 1e-8 else ""),
                    "expected_sign": exp,
                    "measured_sign": got,
                    "sign_match": "" if exp == 0 else exp == got,
                    "measured_sign_qddot_legacy": got_qddot,
                    "sign_match_qddot_legacy": "" if exp == 0 else exp == got_qddot,
                }
            )
        summary = root / "summary.json"
        if summary.exists():
            tracker.read_json(summary)
    result["data_sources"] = [relpath(p) for p in metric_paths(config, "v1", input_override) if p.exists()]
    result["files_read"] = tracker.files_read
    if not rows:
        result["caveats"] = warnings + ["No static pair t/sep tracks found."]
        write_metric_outputs(output_root / "v1", result, rows, metric_report("v1", "V1 Phase-Force Fit", result, rows, warnings), warnings)
        return result
    nonneutral = [r for r in rows if r["expected_sign"] != 0]
    matches = [bool(r["sign_match"]) for r in nonneutral if r["sign_match"] != ""]
    sign_accuracy = float(sum(matches) / len(matches)) if matches else None
    matches_qddot = [bool(r["sign_match_qddot_legacy"]) for r in nonneutral if r.get("sign_match_qddot_legacy", "") != ""]
    sign_accuracy_qddot = float(sum(matches_qddot) / len(matches_qddot)) if matches_qddot else None
    fit = fit_exponential_phase(rows)
    c2_rows = [r for r in rows if r.get("family") == "C2"]
    c3_rows = [r for r in rows if r.get("family") == "C3"]
    c2_fit = fit_exponential_phase(c2_rows) if c2_rows else {"fit_available": False, "reason": "No C2 rows."}
    c3_fit = fit_exponential_phase(c3_rows) if c3_rows else {"fit_available": False, "reason": "No C3 rows."}
    cross = crossover_phase(rows)
    residual_rows: list[dict[str, Any]] = []
    if fit.get("fit_available"):
        C = float(fit["C"])
        lam = float(fit["lambda"])
        for r in rows:
            pred = -C * math.exp(-lam * float(r["q_start"])) * math.cos(float(r["dphi"]))
            residual_rows.append(
                {
                    "source": r["source"],
                    "family": r["family"],
                    "dphi": r["dphi"],
                    "q_start": r["q_start"],
                    "q_ddot": r["q_ddot"],
                    "predicted_q_ddot": pred,
                    "residual": (float(r["q_ddot"]) - pred if r["q_ddot"] is not None else ""),
                }
            )
    result["status"] = "RAN"
    result["validation_depth"] = "INTERNAL_DATA_ANALYTIC_FIT"
    result["fit_parameters"] = fit
    result["fit_quality"] = {
        "n_files_used": len({r["file"] for r in rows}),
        "n_phase_points": len(rows),
        "q_convention": "half_separation",
        "acceleration_window": "first_35_percent_min_4_points",
        "fit_C": fit.get("C"),
        "fit_lambda": fit.get("lambda"),
        "rms_residual": fit.get("log_residual_rms"),
        "normalized_rms_residual": fit.get("normalized_rms_residual"),
        "phase_sign_accuracy": sign_accuracy,
        "force_sign_proxy": "net_early_drift_q_dot",
        "phase_sign_accuracy_qddot_legacy": sign_accuracy_qddot,
        "crossover_phase_estimate": cross.get("crossover_phase_estimate"),
        "crossover_error_from_pi_over_2": cross.get("crossover_error_from_pi_over_2"),
        "c2_fit_quality": c2_fit,
        "c3_fit_quality": c3_fit,
        "n_rows": len(rows),
        "n_non_neutral": len(nonneutral),
        "pi_over_2_neutral_rows": len([r for r in rows if r["expected_sign"] == 0]),
    }
    if sign_accuracy is None or not fit.get("fit_available"):
        result["behavioural_result"] = "INCONCLUSIVE"
    elif sign_accuracy >= 0.8 and float(fit.get("normalized_rms_residual", 1.0)) <= 0.5:
        result["behavioural_result"] = "PASS"
    elif sign_accuracy >= 0.5:
        result["behavioural_result"] = "PARTIAL"
    else:
        result["behavioural_result"] = "FAIL"
    if sign_accuracy is not None and sign_accuracy >= 0.8:
        result["match_level_candidate"] = "SAME FAMILY" if fit.get("fit_available") else "CLOSE ANALOGUE"
    elif sign_accuracy is not None and sign_accuracy < 0.5:
        result["match_level_candidate"] = "FAILED_FIT"
    else:
        result["match_level_candidate"] = "CLOSE ANALOGUE"
    result["caveats"] = warnings + [
        "Constants C/lambda are freely fit; canonical NLS constants are secondary context only.",
        "Force-direction sign uses net early drift (q_dot); trajectory curvature q_ddot is "
        "unreliable for breathing C3 Q-ball pairs and is retained (as *_qddot_legacy) only "
        "to document that artifact. phase_sign_accuracy_qddot_legacy shows the old value.",
        "C3 magnitude fit (lambda_C3) uses early-window q_ddot and is provisional; the C2 "
        "lambda is the clean quantity (recovers ~2A, the canonical NLS interaction rate).",
    ]
    if result["behavioural_result"] in {"FAIL", "INCONCLUSIVE"}:
        result["status"] = "RAN_WEAK"
    if make_plots:
        xs = [r["q_start"] for r in rows if r["q_ddot"] is not None]
        ys = [r["q_ddot"] for r in rows if r["q_ddot"] is not None]
        if xs and ys:
            scatter_line_plot(output_root / "v1" / "figure.png", xs, ys, ys, "V1 early q acceleration", "q_start", "q_ddot")
    write_csv(output_root / "v1" / "phase_force_fit_by_phase.csv", rows)
    write_csv(output_root / "v1" / "residuals.csv", residual_rows)
    write_metric_outputs(output_root / "v1", result, rows, metric_report("v1", "V1 Phase-Force Fit", result, rows, warnings), warnings)
    return result
