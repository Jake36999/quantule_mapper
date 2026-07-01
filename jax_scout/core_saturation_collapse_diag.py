"""
Post-run collapse/runaway diagnostic pass for selected Phase C candidates.

This is an analysis-only layer:
- loads saved Phase C rows or the feb56dc7 reference
- replays them with the existing solver path / IC family
- computes snapshot-based localization / geometry / spectral diagnostics
- assigns cautious post-run labels without changing the PDE or solver
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CORE_RADIUS_FOR_MASS = 4.0
HIGH_K_FRACTION_ARTIFACT = 0.35
DEFAULT_SNAPS = 40
REF_FEB_N = 96
REF_FEB_T = 6000
HIGH_MASS_COMPARISON_K1_IDX = 2
HIGH_MASS_COMPARISON_K6_IDXS = (32, 33, 39)
THRESHOLD_PILOT_RUN_NAMES = (
    "CORE_SAT_HUNT_20260623_170944",
    "CORE_SAT_HUNT_20260623_171758",
    "CORE_SAT_HUNT_20260623_172609",
    "CORE_SAT_HUNT_20260623_173417",
    "CORE_SAT_HUNT_20260623_174215",
    "CORE_SAT_HUNT_20260623_175018",
)
TRACE_COMPARISON_CASES = (
    ("k1_below_threshold_survivor", "CORE_SAT_HUNT_20260623_171758", 4),
    ("k1_above_threshold_failure", "CORE_SAT_HUNT_20260623_173417", 2),
    ("k6_same_mass_survivor", "CORE_SAT_HUNT_20260623_173417", 10),
)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _to_int(value: object, default: int = -1) -> int:
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value: object, default: float = float("nan")) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _run_paths(root: Path) -> dict[str, Path]:
    return {
        "high": root / "CORE_SAT_HUNT_20260623_123527",
        "low": root / "CORE_SAT_HUNT_20260623_120758",
        "base": root / "CORE_SAT_HUNT_20260623_113318",
    }


def _group_rows(
    run_dir: Path,
    *,
    group: str,
    predicate,
) -> list[dict[str, Any]]:
    rows = _load_csv(run_dir / "all_evals.csv")
    out: list[dict[str, Any]] = []
    for row in rows:
        if not predicate(row):
            continue
        out.append(
            {
                "group": group,
                "run_dir": str(run_dir.resolve()),
                "csv_path": str((run_dir / "all_evals.csv").resolve()),
                "idx": _to_int(row.get("idx")),
                "K": _to_int(row.get("ic_blobs")),
                "ic_norm": row.get("ic_norm") or "per_blob_fixed",
                "target_initial_mass": _to_float(row.get("target_initial_mass"), default=float("nan")),
                "klass": row.get("klass", ""),
                "n_fin": _to_int(row.get("n_fin")),
            }
        )
    return out


def _candidate_from_row(run_dir: Path, row: dict[str, str], *, group: str) -> dict[str, Any]:
    return {
        "group": group,
        "run_dir": str(run_dir.resolve()),
        "csv_path": str((run_dir / "all_evals.csv").resolve()),
        "idx": _to_int(row.get("idx")),
        "K": _to_int(row.get("ic_blobs")),
        "ic_norm": row.get("ic_norm") or "per_blob_fixed",
        "target_initial_mass": _to_float(row.get("target_initial_mass"), default=float("nan")),
        "klass": row.get("klass", ""),
        "n_fin": _to_int(row.get("n_fin")),
    }


def select_required_candidates(root: str | Path) -> list[dict[str, Any]]:
    root_path = Path(root)
    paths = _run_paths(root_path)
    groups: list[dict[str, Any]] = []
    groups.extend(
        _group_rows(
            paths["high"],
            group="high_target_k1_blowup",
            predicate=lambda row: _to_int(row.get("ic_blobs")) == 1 and row.get("klass") == "LATE_BLOWUP_REJECT",
        )
    )
    groups.extend(
        _group_rows(
            paths["high"],
            group="high_target_k6_true",
            predicate=lambda row: _to_int(row.get("ic_blobs")) == 6 and row.get("klass") == "TRUE_SATURATED_BOUND_STATE",
        )
    )
    groups.extend(
        _group_rows(
            paths["low"],
            group="low_target_k1_true_or_spin",
            predicate=lambda row: _to_int(row.get("ic_blobs")) == 1
            and row.get("klass") in {"TRUE_SATURATED_BOUND_STATE", "SPIN_DOWN_REJECT"},
        )
    )
    groups.extend(
        _group_rows(
            paths["base"],
            group="baseline_k1_k6",
            predicate=lambda row: _to_int(row.get("ic_blobs")) in {1, 6},
        )
    )
    groups.append(
        {
            "group": "feb56dc7_reference",
            "run_dir": None,
            "csv_path": None,
            "idx": None,
            "K": 6,
            "ic_norm": "per_blob_fixed",
            "target_initial_mass": None,
            "klass": "TRUE_SATURATED_BOUND_STATE",
            "n_fin": 4,
            "ref": "feb56dc7",
        }
    )
    return groups


def select_high_mass_comparison_candidates(root: str | Path) -> list[dict[str, Any]]:
    root_path = Path(root)
    high = _run_paths(root_path)["high"]
    rows = _load_csv(high / "all_evals.csv")
    selected: list[dict[str, Any]] = []
    for row in rows:
        idx = _to_int(row.get("idx"))
        kval = _to_int(row.get("ic_blobs"))
        klass = row.get("klass", "")
        if idx == HIGH_MASS_COMPARISON_K1_IDX and kval == 1:
            selected.append(
                {
                    "group": "high_mass_k1_failure",
                    "run_dir": str(high.resolve()),
                    "csv_path": str((high / "all_evals.csv").resolve()),
                    "idx": idx,
                    "K": kval,
                    "ic_norm": row.get("ic_norm") or "per_blob_fixed",
                    "target_initial_mass": _to_float(row.get("target_initial_mass"), default=float("nan")),
                    "klass": klass,
                    "n_fin": _to_int(row.get("n_fin")),
                }
            )
        if idx in HIGH_MASS_COMPARISON_K6_IDXS and kval == 6 and klass == "TRUE_SATURATED_BOUND_STATE":
            selected.append(
                {
                    "group": "high_mass_k6_stable",
                    "run_dir": str(high.resolve()),
                    "csv_path": str((high / "all_evals.csv").resolve()),
                    "idx": idx,
                    "K": kval,
                    "ic_norm": row.get("ic_norm") or "per_blob_fixed",
                    "target_initial_mass": _to_float(row.get("target_initial_mass"), default=float("nan")),
                    "klass": klass,
                    "n_fin": _to_int(row.get("n_fin")),
                }
            )
    selected.append(
        {
            "group": "high_mass_reference_control",
            "run_dir": None,
            "csv_path": None,
            "idx": None,
            "K": 6,
            "ic_norm": "per_blob_fixed",
            "target_initial_mass": None,
            "klass": "TRUE_SATURATED_BOUND_STATE",
            "n_fin": 4,
            "ref": "feb56dc7",
        }
    )
    return selected


def select_threshold_pilot_candidates(root: str | Path) -> list[dict[str, Any]]:
    root_path = Path(root)
    candidates: list[dict[str, Any]] = []
    for run_name in THRESHOLD_PILOT_RUN_NAMES:
        run_dir = root_path / run_name
        rows = _load_csv(run_dir / "all_evals.csv")
        for row in rows:
            candidates.append(_candidate_from_row(run_dir, row, group="threshold_pilot"))
    return candidates


def select_trace_comparison_candidates(root: str | Path) -> list[dict[str, Any]]:
    root_path = Path(root)
    out: list[dict[str, Any]] = []
    for group, run_name, idx in TRACE_COMPARISON_CASES:
        run_dir = root_path / run_name
        rows = _load_csv(run_dir / "all_evals.csv")
        match = next((row for row in rows if _to_int(row.get("idx")) == idx), None)
        if match is None:
            raise FileNotFoundError(f"Could not find idx={idx} in {run_dir / 'all_evals.csv'}")
        out.append(_candidate_from_row(run_dir, match, group=group))
    out.append(
        {
            "group": "ref_feb56dc7_control",
            "run_dir": None,
            "csv_path": None,
            "idx": None,
            "K": 6,
            "ic_norm": "per_blob_fixed",
            "target_initial_mass": None,
            "klass": "TRUE_SATURATED_BOUND_STATE",
            "n_fin": 4,
            "ref": "feb56dc7",
        }
    )
    return out


def assign_diagnostic_label(summary: dict[str, Any]) -> str:
    klass = summary.get("klass", "")
    if klass == "TRUE_SATURATED_BOUND_STATE":
        return "SATURATED_BOUND_STATE"
    if klass == "SPIN_DOWN_REJECT":
        return "SPIN_DOWN_DECAY"
    if summary.get("split_before_blowup"):
        return "FRAGMENTING_BLOWUP"
    if _to_float(summary.get("high_k_fraction_max"), 0.0) >= HIGH_K_FRACTION_ARTIFACT:
        return "HIGH_K_NUMERICAL_ARTIFACT_SUSPECT"

    rho_peak_growth = _to_float(summary.get("rho_peak_growth_ratio"), 1.0)
    radius_shrink = _to_float(summary.get("core_radius_shrink_ratio"), 1.0)
    compact_growth = _to_float(summary.get("compactness_growth_ratio"), 1.0)
    omega_drop = _to_float(summary.get("omega2_min_ratio"), 1.0)
    grad_growth = _to_float(summary.get("grad_log_omega_growth_ratio"), 1.0)
    finite_last = bool(summary.get("finite_last", True))

    collapse_like = (
        rho_peak_growth >= 3.0
        and radius_shrink <= 0.65
        and compact_growth >= 2.0
        and (omega_drop <= 0.5 or grad_growth >= 2.5)
        and not finite_last
    )
    if collapse_like:
        return "COLLAPSE_LIKE_RUNAWAY"

    if klass in {"LATE_BLOWUP_REJECT", "TRANSIENT_GROWER_REJECT"}:
        if rho_peak_growth < 2.0 and compact_growth < 1.5 and radius_shrink > 0.8:
            return "DELOCALIZED_GROWTH"
        return "INCONCLUSIVE_FAILURE_TRACE"

    return "INCONCLUSIVE_FAILURE_TRACE"


def interpret_high_mass_comparison(rows: list[dict[str, Any]]) -> str:
    k1_rows = [row for row in rows if _to_int(row.get("K")) == 1]
    k6_rows = [row for row in rows if _to_int(row.get("K")) == 6 and row.get("idx") not in ("", None)]
    if not k1_rows or not k6_rows:
        return "COMPARISON_INCOMPLETE"
    k1_failure = any(row.get("diagnostic_label") != "SATURATED_BOUND_STATE" for row in k1_rows)
    k6_stable = all(row.get("diagnostic_label") == "SATURATED_BOUND_STATE" for row in k6_rows)
    try:
        k1_compactness = max(float(row.get("compactness_max", float("nan"))) for row in k1_rows)
        k6_compactness = max(float(row.get("compactness_max", float("nan"))) for row in k6_rows)
    except (TypeError, ValueError):
        return "COMPARISON_INCOMPLETE"
    if k1_failure and k6_stable and np.isfinite(k1_compactness) and np.isfinite(k6_compactness) and k1_compactness > k6_compactness:
        return "DISTRIBUTED_MASS_STABILIZATION_SUPPORTED"
    return "COMPARISON_INCONCLUSIVE"


def _periodic_displacements(shape: tuple[int, int, int], center: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    axes = [np.arange(n, dtype=float) for n in shape]
    grids = np.meshgrid(*axes, indexing="ij")
    disp = []
    for grid, c, n in zip(grids, center, shape):
        d = ((grid - c + n / 2.0) % n) - n / 2.0
        disp.append(d)
    rr = np.sqrt(disp[0] ** 2 + disp[1] ** 2 + disp[2] ** 2)
    return disp[0], disp[1], disp[2], rr


def _dominant_center(psi: np.ndarray, dx: float) -> np.ndarray:
    from jax_scout import transfer_diag as td

    nodes = sorted(td.detect_nodes(psi, dx), key=lambda node: -node["E"])
    if nodes:
        return np.asarray(nodes[0]["centroid"], dtype=float)
    return np.asarray(np.unravel_index(int(np.argmax(np.abs(psi) ** 2)), psi.shape), dtype=float)


def _half_mass_radius(rho: np.ndarray, rr: np.ndarray) -> float:
    total = float(np.sum(rho))
    if total <= 0:
        return float("nan")
    order = np.argsort(rr.ravel())
    r_sorted = rr.ravel()[order]
    m_sorted = rho.ravel()[order]
    cumulative = np.cumsum(m_sorted)
    idx = int(np.searchsorted(cumulative, 0.5 * total, side="left"))
    idx = min(max(idx, 0), len(r_sorted) - 1)
    return float(r_sorted[idx])


def _participation_ratio(rho: np.ndarray) -> float:
    num = float(np.sum(rho) ** 2)
    den = float(np.sum(rho ** 2)) + 1e-30
    return num / den


def _high_k_fraction(psi: np.ndarray) -> float:
    spectrum = np.abs(np.fft.fftn(psi)) ** 2
    n = psi.shape[0]
    k = np.fft.fftfreq(n)
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    kr = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    kmax = float(np.max(kr)) + 1e-30
    high = kr >= 0.5 * kmax
    total = float(np.sum(spectrum)) + 1e-30
    return float(np.sum(spectrum[high]) / total)


def _snapshot_metrics(psi: np.ndarray, params: dict[str, float], dx: float) -> dict[str, Any]:
    from jax_scout import transfer_diag as td

    finite = bool(np.all(np.isfinite(np.abs(psi))))
    if not finite:
        return {
            "mass": float("nan"),
            "rho_peak": float("nan"),
            "core_radius": float("nan"),
            "mass_inside_radius": float("nan"),
            "compactness": float("nan"),
            "omega2_min": float("nan"),
            "grad_log_omega_max": float("nan"),
            "curvature_proxy_max": float("nan"),
            "node_count": 0,
            "participation_ratio": float("nan"),
            "high_k_fraction": float("nan"),
            "center": None,
            "finite": False,
        }
    rho = np.abs(psi) ** 2
    center = _dominant_center(psi, dx)
    _, _, _, rr = _periodic_displacements(rho.shape, center)
    core_radius = _half_mass_radius(rho, rr)
    inside = rr <= CORE_RADIUS_FOR_MASS
    mass_inside = float(np.sum(rho[inside]))
    compactness = mass_inside / CORE_RADIUS_FOR_MASS
    geo = td.geometry_fields(psi, params, dx)
    omega_sq = np.asarray(geo["omega_sq"], dtype=float)
    grad_x, grad_y, grad_z = td._grad(0.5 * np.log(np.maximum(omega_sq, 1e-30)), dx)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2)
    curvature = np.asarray(geo["R"], dtype=float)
    nodes = td.detect_nodes(psi, dx)
    return {
        "mass": float(np.sum(rho)),
        "rho_peak": float(np.max(rho)),
        "core_radius": core_radius,
        "mass_inside_radius": mass_inside,
        "compactness": compactness,
        "omega2_min": float(np.min(omega_sq)),
        "grad_log_omega_max": float(np.max(np.abs(grad_mag))),
        "curvature_proxy_max": float(np.max(np.abs(curvature))),
        "node_count": len(nodes),
        "participation_ratio": _participation_ratio(rho),
        "high_k_fraction": _high_k_fraction(psi),
        "center": center.tolist(),
        "finite": True,
    }


def _safe_ratio(numer: float, denom: float, default: float = float("nan")) -> float:
    if not np.isfinite(numer) or not np.isfinite(denom) or abs(denom) < 1e-30:
        return default
    return float(numer / denom)


def _late_slope(times: np.ndarray, values: np.ndarray) -> float:
    if len(values) < 3:
        return float("nan")
    start = len(values) // 2
    xs = np.asarray(times[start:], dtype=float)
    ys = np.asarray(values[start:], dtype=float)
    if len(xs) < 3:
        return float("nan")
    return float(np.polyfit(xs - xs[0], ys, 1)[0])


def _first_threshold_time(times: np.ndarray, values: np.ndarray, *, threshold: float, mode: str) -> float | None:
    for t, value in zip(times, values):
        if mode == "ge" and value >= threshold:
            return float(t)
        if mode == "le" and value <= threshold:
            return float(t)
    return None


def _summarize_trace(candidate: dict[str, Any], times: np.ndarray, trace: list[dict[str, Any]], *, finite_last: bool) -> dict[str, Any]:
    masses = np.asarray([row["mass"] for row in trace], dtype=float)
    rho_peak = np.asarray([row["rho_peak"] for row in trace], dtype=float)
    core_radius = np.asarray([row["core_radius"] for row in trace], dtype=float)
    compactness = np.asarray([row["compactness"] for row in trace], dtype=float)
    omega2_min = np.asarray([row["omega2_min"] for row in trace], dtype=float)
    grad_log = np.asarray([row["grad_log_omega_max"] for row in trace], dtype=float)
    curvature = np.asarray([row["curvature_proxy_max"] for row in trace], dtype=float)
    nodes = np.asarray([row["node_count"] for row in trace], dtype=int)
    high_k = np.asarray([row["high_k_fraction"] for row in trace], dtype=float)
    finite_flags = np.asarray([bool(row.get("finite", True)) for row in trace], dtype=bool)

    base_nodes = int(nodes[0]) if len(nodes) else 0
    split_before_blowup = bool(np.max(nodes[:-1] if len(nodes) > 1 else nodes, initial=base_nodes) > max(1, base_nodes))
    er = masses / max(masses[0], 1e-30)
    nonfinite_time = None
    for t, is_finite in zip(times, finite_flags):
        if not is_finite:
            nonfinite_time = float(t)
            break
    blowup_time = _first_threshold_time(times, er, threshold=3.0, mode="ge")
    summary = {
        "klass": candidate["klass"],
        "finite_last": finite_last,
        "time_to_blowup": blowup_time,
        "time_to_failure": nonfinite_time if nonfinite_time is not None else blowup_time,
        "er_final_or_last": float(er[-1]),
        "rho_peak_max": float(np.max(rho_peak)),
        "core_radius_min": float(np.nanmin(core_radius)),
        "compactness_max": float(np.max(compactness)),
        "omega2_min_min": float(np.min(omega2_min)),
        "grad_log_omega_max": float(np.max(grad_log)),
        "curvature_proxy_max": float(np.max(curvature)),
        "node_count_mid": int(nodes[len(nodes) // 2]),
        "node_count_last": int(nodes[-1]),
        "split_before_blowup": split_before_blowup,
        "participation_ratio_min": float(np.min([row["participation_ratio"] for row in trace])),
        "high_k_fraction_max": float(np.max(high_k)),
        "late_energy_slope": _late_slope(times, er),
        "rho_peak_growth_ratio": _safe_ratio(float(np.max(rho_peak)), float(rho_peak[0]), default=1.0),
        "core_radius_shrink_ratio": _safe_ratio(float(np.nanmin(core_radius)), float(core_radius[0]), default=1.0),
        "compactness_growth_ratio": _safe_ratio(float(np.max(compactness)), float(compactness[0]), default=1.0),
        "omega2_min_ratio": _safe_ratio(float(np.min(omega2_min)), float(omega2_min[0]), default=1.0),
        "grad_log_omega_growth_ratio": _safe_ratio(float(np.max(grad_log)), float(grad_log[0]), default=1.0),
        "curvature_growth_ratio": _safe_ratio(float(np.max(curvature)), float(curvature[0]), default=1.0),
    }
    summary["diagnostic_label"] = assign_diagnostic_label(summary)
    return summary


def _candidate_slug(candidate: dict[str, Any]) -> str:
    if candidate.get("ref"):
        return "ref_feb56dc7"
    return f"{Path(candidate['run_dir']).name}_idx_{candidate['idx']}"


def _resolve_replay_contract(
    candidate: dict[str, Any],
    *,
    target_initial_mass_override: float | None = None,
) -> tuple[dict[str, Any], int, int, dict[str, float], np.ndarray, float]:
    from jax_scout import core_saturation_replay as csr
    from jax_scout import core_saturation_search as css

    if candidate.get("ref"):
        replay = csr.resolve_candidate(None, ref="feb56dc7")
        N, T = REF_FEB_N, REF_FEB_T
    else:
        csv_path = candidate["csv_path"]
        replay = csr.resolve_candidate(csv_path, idx=candidate["idx"])
        summary = _load_json(Path(candidate["run_dir"]) / "summary.json")
        N = int(summary["N"])
        T = int(summary["T"])
    if target_initial_mass_override is not None:
        replay["target_initial_mass"] = float(target_initial_mass_override)
    params = replay["params"]
    psi0, ic_stats = css.build_ic(
        N,
        replay["ic_blobs"],
        seed=replay["ic_seed"],
        ic_norm=replay["ic_norm"],
        target_initial_mass=replay["target_initial_mass"],
    )
    return replay, N, T, params, psi0, float(ic_stats["initial_mass"])


def replay_candidate(
    candidate: dict[str, Any],
    *,
    n_snap: int = DEFAULT_SNAPS,
    N_override: int | None = None,
    T_override: int | None = None,
    target_initial_mass_override: float | None = None,
) -> dict[str, Any]:
    from jax_scout import core_saturation_search as css
    from jax_scout import transfer_diag as td

    replay, N, T, params, psi0, initial_mass = _resolve_replay_contract(
        candidate,
        target_initial_mass_override=target_initial_mass_override,
    )
    if N_override is not None:
        N = int(N_override)
    if T_override is not None:
        T = int(T_override)
    if N_override is not None or T_override is not None:
        psi0, ic_stats = css.build_ic(
            N,
            replay["ic_blobs"],
            seed=replay["ic_seed"],
            ic_norm=replay["ic_norm"],
            target_initial_mass=replay["target_initial_mass"],
        )
        initial_mass = float(ic_stats["initial_mass"])
    times = np.linspace(0, T, n_snap + 1)
    snaps, finite_last = td.capture_trajectory([params[k] for k in css.order], psi0, N, css.L_, css.DT, T, n_snap)
    dx = css.L_ / N
    trace = [_snapshot_metrics(np.asarray(psi), params, dx) for psi in snaps]
    summary = _summarize_trace(candidate, times, trace, finite_last=bool(finite_last))
    return {
        "candidate": candidate,
        "replay": {
            "label": replay["label"],
            "ic_blobs": replay["ic_blobs"],
            "ic_seed": replay["ic_seed"],
            "ic_norm": replay["ic_norm"],
            "target_initial_mass": replay["target_initial_mass"],
            "N": N,
            "T": T,
            "dt": css.DT,
            "initial_mass": initial_mass,
        },
        "times": times.tolist(),
        "trace": trace,
        "summary": summary,
        "frames": np.asarray(snaps, dtype=np.complex64),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, default=float)
        handle.write("\n")


def _diagnostic_row(result: dict[str, Any]) -> dict[str, Any]:
    candidate = result["candidate"]
    replay = result["replay"]
    summary = result["summary"]
    return {
        "group": candidate.get("group"),
        "run_dir": candidate.get("run_dir"),
        "idx": candidate.get("idx"),
        "K": candidate.get("K"),
        "ic_norm": replay["ic_norm"],
        "target_initial_mass": replay["target_initial_mass"],
        "class": candidate.get("klass"),
        "N": replay["N"],
        "T": replay["T"],
        "initial_mass": replay["initial_mass"],
        "time_to_failure": summary["time_to_failure"],
        "time_to_blowup": summary["time_to_blowup"],
        "er_final_or_last": summary["er_final_or_last"],
        "late_energy_slope": summary["late_energy_slope"],
        "rho_peak_max": summary["rho_peak_max"],
        "core_radius_min": summary["core_radius_min"],
        "compactness_max": summary["compactness_max"],
        "omega2_min_min": summary["omega2_min_min"],
        "grad_log_omega_max": summary["grad_log_omega_max"],
        "curvature_proxy_max": summary["curvature_proxy_max"],
        "node_count_mid": summary["node_count_mid"],
        "node_count_last": summary["node_count_last"],
        "split_before_blowup": summary["split_before_blowup"],
        "high_k_fraction_max": summary["high_k_fraction_max"],
        "diagnostic_label": summary["diagnostic_label"],
    }


def _float_or_none(value: object) -> float | None:
    out = _to_float(value, default=float("nan"))
    return None if not np.isfinite(out) else float(out)


def _int_or_none(value: object) -> int | None:
    out = _to_int(value, default=-1)
    return None if out < 0 else int(out)


def _compact_metric_summary(values: list[float | None]) -> dict[str, float | None]:
    arr = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    if arr.size == 0:
        return {"min": None, "median": None, "max": None, "mean": None}
    return {
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def summarize_threshold_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    overall_class_counts: dict[str, int] = {}
    overall_diag_counts: dict[str, int] = {}
    by_bucket: dict[str, dict[str, Any]] = {}
    for row in rows:
        klass = str(row.get("class"))
        label = str(row.get("diagnostic_label"))
        overall_class_counts[klass] = overall_class_counts.get(klass, 0) + 1
        overall_diag_counts[label] = overall_diag_counts.get(label, 0) + 1

        k = _to_int(row.get("K"))
        mass = _to_float(row.get("target_initial_mass"))
        key = f"K{k}_M{mass:.6f}"
        bucket = by_bucket.setdefault(
            key,
            {
                "K": k,
                "target_initial_mass": float(mass),
                "rows": [],
                "class_counts": {},
                "diagnostic_counts": {},
                "final_node_counts": {},
            },
        )
        bucket["rows"].append(row)
        bucket["class_counts"][klass] = bucket["class_counts"].get(klass, 0) + 1
        bucket["diagnostic_counts"][label] = bucket["diagnostic_counts"].get(label, 0) + 1
        node_last = _int_or_none(row.get("node_count_last"))
        if node_last is not None:
            node_key = str(node_last)
            bucket["final_node_counts"][node_key] = bucket["final_node_counts"].get(node_key, 0) + 1

    for bucket in by_bucket.values():
        rows_here = bucket.pop("rows")
        bucket["compactness"] = _compact_metric_summary([_float_or_none(row.get("compactness_max")) for row in rows_here])
        bucket["high_k_fraction"] = _compact_metric_summary([_float_or_none(row.get("high_k_fraction_max")) for row in rows_here])
        bucket["time_to_failure"] = _compact_metric_summary([_float_or_none(row.get("time_to_failure")) for row in rows_here])
        bucket["late_energy_slope"] = _compact_metric_summary([_float_or_none(row.get("late_energy_slope")) for row in rows_here])

    ordered = sorted(by_bucket.values(), key=lambda bucket: (bucket["target_initial_mass"], bucket["K"]))
    return {
        "overall_class_counts": overall_class_counts,
        "overall_diagnostic_counts": overall_diag_counts,
        "by_K_target_mass": ordered,
    }


def _trace_arrays(result: dict[str, Any]) -> dict[str, np.ndarray]:
    trace = result["trace"]
    times = np.asarray(result["times"], dtype=float)
    initial_mass = float(result["replay"]["initial_mass"])
    mass = np.asarray([row["mass"] for row in trace], dtype=float)
    return {
        "times": times,
        "energy_mass_proxy": mass / max(initial_mass, 1e-30),
        "rho_peak": np.asarray([row["rho_peak"] for row in trace], dtype=float),
        "compactness": np.asarray([row["compactness"] for row in trace], dtype=float),
        "core_radius": np.asarray([row["core_radius"] for row in trace], dtype=float),
        "omega2_min": np.asarray([row["omega2_min"] for row in trace], dtype=float),
        "grad_log_omega": np.asarray([row["grad_log_omega_max"] for row in trace], dtype=float),
        "high_k_fraction": np.asarray([row["high_k_fraction"] for row in trace], dtype=float),
        "node_count": np.asarray([row["node_count"] for row in trace], dtype=float),
    }


def _plot_trace_panel(result: dict[str, Any], outpath: Path) -> None:
    import matplotlib.pyplot as plt

    arrays = _trace_arrays(result)
    summary = result["summary"]
    replay = result["replay"]
    candidate = result["candidate"]
    fig, axes = plt.subplots(4, 2, figsize=(13, 12), sharex=True)
    panels = [
        ("energy_mass_proxy", "Energy / Mass Proxy", "mass / initial_mass"),
        ("rho_peak", "Peak Density", "max |psi|^2"),
        ("compactness", "Compactness Proxy", "mass_inside_r / r"),
        ("core_radius", "Core Radius", "half-mass radius"),
        ("omega2_min", "Omega^2 Minimum", "min omega^2"),
        ("grad_log_omega", "Geometry-Gradient Proxy", "max |grad log omega|"),
        ("high_k_fraction", "High-k Fraction", "spectral tail"),
        ("node_count", "Node Count", "count"),
    ]
    flat_axes = list(axes.flat)
    for ax, (key, title, ylabel) in zip(flat_axes, panels):
        ax.plot(arrays["times"], arrays[key], lw=1.7)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    flat_axes[-1].set_xlabel("time")
    flat_axes[-2].set_xlabel("time")
    slug = _candidate_slug(candidate)
    time_to_failure = summary.get("time_to_failure")
    failure_text = "none" if time_to_failure is None else f"{float(time_to_failure):.1f}"
    fig.suptitle(
        f"{slug} | class={candidate.get('klass')} | diag={summary['diagnostic_label']} | "
        f"K={candidate.get('K')} | ic_norm={replay['ic_norm']} | "
        f"target_mass={replay['target_initial_mass']} | split={summary['split_before_blowup']} | "
        f"time_to_failure={failure_text}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def _plot_trace_overlay(results: list[dict[str, Any]], outpath: Path) -> None:
    import matplotlib.pyplot as plt

    panels = [
        ("energy_mass_proxy", "Energy / Mass Proxy"),
        ("rho_peak", "Peak Density"),
        ("compactness", "Compactness Proxy"),
        ("core_radius", "Core Radius"),
        ("omega2_min", "Omega^2 Minimum"),
        ("grad_log_omega", "Geometry-Gradient Proxy"),
        ("high_k_fraction", "High-k Fraction"),
        ("node_count", "Node Count"),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(13, 12), sharex=False)
    flat_axes = list(axes.flat)
    for result in results:
        arrays = _trace_arrays(result)
        label = _candidate_slug(result["candidate"])
        for ax, (key, title) in zip(flat_axes, panels):
            ax.plot(arrays["times"], arrays[key], lw=1.5, label=label)
            ax.set_title(title)
            ax.grid(True, alpha=0.25)
    for ax in flat_axes[-2:]:
        ax.set_xlabel("time")
    flat_axes[0].legend(fontsize=7, loc="best")
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def _load_saved_result(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    payload["times"] = payload.get("times", [])
    payload["trace"] = payload.get("trace", [])
    return payload


def render_saved_trace_plots(outdir: str | Path) -> Path:
    outdir_path = Path(outdir)
    results: list[dict[str, Any]] = []
    for summary_path in sorted(outdir_path.glob("*/diagnostic_summary.json")):
        result = _load_saved_result(summary_path)
        _plot_trace_panel(result, summary_path.parent / "trace_panel.png")
        results.append(result)
    if not results:
        raise FileNotFoundError(f"No diagnostic_summary.json files found under {outdir_path}")
    _plot_trace_overlay(results, outdir_path / "trace_overlay.png")
    summary_payload = _load_json(outdir_path / "trace_comparison_summary.json") if (outdir_path / "trace_comparison_summary.json").exists() else {}
    summary_payload["plots_rendered"] = True
    _write_json(outdir_path / "trace_comparison_summary.json", summary_payload)
    return outdir_path


def run_diagnostic(outdir: str | Path, *, root: str | Path | None = None, n_snap: int = DEFAULT_SNAPS, overwrite: bool = False) -> Path:
    root_path = Path(root) if root is not None else (ROOT / "sweep_runs")
    outdir_path = Path(outdir)
    if outdir_path.exists() and any(outdir_path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory already exists and is non-empty: {outdir_path}")
    outdir_path.mkdir(parents=True, exist_ok=True)

    candidates = select_required_candidates(root_path)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        result = replay_candidate(candidate, n_snap=n_snap)
        slug = _candidate_slug(candidate)
        candir = outdir_path / slug
        candir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(candir / "frames.npz", psi=result["frames"], times=np.asarray(result["times"], dtype=float))
        _write_json(candir / "diagnostic_summary.json", {k: v for k, v in result.items() if k != "frames"})
        rows.append(_diagnostic_row(result))
        print(f"{slug}: {result['summary']['diagnostic_label']} | class={candidate['klass']} K={candidate['K']}", flush=True)

    csv_path = outdir_path / "collapse_runaway_diagnostics.csv"
    fieldnames = [
        "run_dir",
        "idx",
        "K",
        "ic_norm",
        "target_initial_mass",
        "class",
        "time_to_blowup",
        "er_final_or_last",
        "rho_peak_max",
        "core_radius_min",
        "compactness_max",
        "omega2_min_min",
        "grad_log_omega_max",
        "curvature_proxy_max",
        "node_count_mid",
        "node_count_last",
        "split_before_blowup",
        "high_k_fraction_max",
        "diagnostic_label",
    ]
    _write_csv(csv_path, rows, fieldnames)
    return outdir_path


def run_high_mass_comparison(
    outdir: str | Path,
    *,
    root: str | Path | None = None,
    overwrite: bool = False,
    N_override: int = REF_FEB_N,
    T_override: int = REF_FEB_T,
    default_snaps: int = 60,
    failure_snaps: int = 120,
) -> Path:
    root_path = Path(root) if root is not None else (ROOT / "sweep_runs")
    outdir_path = Path(outdir)
    if outdir_path.exists() and any(outdir_path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory already exists and is non-empty: {outdir_path}")
    outdir_path.mkdir(parents=True, exist_ok=True)

    candidates = select_high_mass_comparison_candidates(root_path)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        n_snap = failure_snaps if candidate.get("idx") == HIGH_MASS_COMPARISON_K1_IDX else default_snaps
        result = replay_candidate(candidate, n_snap=n_snap, N_override=N_override, T_override=T_override)
        slug = _candidate_slug(candidate)
        candir = outdir_path / slug
        candir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(candir / "frames.npz", psi=result["frames"], times=np.asarray(result["times"], dtype=float))
        _write_json(candir / "diagnostic_summary.json", {k: v for k, v in result.items() if k != "frames"})
        rows.append(_diagnostic_row(result))
        print(f"{slug}: {result['summary']['diagnostic_label']} | class={candidate['klass']} K={candidate['K']} N={N_override} T={T_override}", flush=True)

    interpretation = interpret_high_mass_comparison(rows)
    for row in rows:
        row["interpretation"] = interpretation
    csv_path = outdir_path / "high_mass_k1_k6_comparison.csv"
    fieldnames = [
        "idx",
        "K",
        "ic_norm",
        "target_initial_mass",
        "class",
        "N",
        "T",
        "time_to_failure",
        "time_to_blowup",
        "node_count_last",
        "rho_peak_max",
        "core_radius_min",
        "compactness_max",
        "omega2_min_min",
        "grad_log_omega_max",
        "high_k_fraction_max",
        "diagnostic_label",
        "interpretation",
    ]
    _write_csv(csv_path, rows, fieldnames)
    _write_json(outdir_path / "comparison_summary.json", {"interpretation": interpretation, "rows": rows})
    return outdir_path


def run_threshold_pilot(
    outdir: str | Path,
    *,
    root: str | Path | None = None,
    overwrite: bool = False,
    n_snap: int = DEFAULT_SNAPS,
) -> Path:
    root_path = Path(root) if root is not None else (ROOT / "sweep_runs")
    outdir_path = Path(outdir)
    if outdir_path.exists() and any(outdir_path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory already exists and is non-empty: {outdir_path}")
    outdir_path.mkdir(parents=True, exist_ok=True)

    candidates = select_threshold_pilot_candidates(root_path)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        result = replay_candidate(candidate, n_snap=n_snap)
        slug = _candidate_slug(candidate)
        candir = outdir_path / "replays" / slug
        candir.mkdir(parents=True, exist_ok=True)
        _write_json(candir / "diagnostic_summary.json", {k: v for k, v in result.items() if k != "frames"})
        rows.append(_diagnostic_row(result))
        print(
            f"{slug}: {result['summary']['diagnostic_label']} | class={candidate['klass']} "
            f"K={candidate['K']} target_mass={candidate['target_initial_mass']}",
            flush=True,
        )

    fieldnames = [
        "group",
        "run_dir",
        "idx",
        "K",
        "ic_norm",
        "target_initial_mass",
        "class",
        "N",
        "T",
        "initial_mass",
        "time_to_failure",
        "time_to_blowup",
        "er_final_or_last",
        "late_energy_slope",
        "rho_peak_max",
        "core_radius_min",
        "compactness_max",
        "omega2_min_min",
        "grad_log_omega_max",
        "curvature_proxy_max",
        "node_count_mid",
        "node_count_last",
        "split_before_blowup",
        "high_k_fraction_max",
        "diagnostic_label",
    ]
    _write_csv(outdir_path / "threshold_diagnostics.csv", rows, fieldnames)
    _write_json(outdir_path / "threshold_summary.json", summarize_threshold_results(rows))
    return outdir_path


def run_trace_comparison(
    outdir: str | Path,
    *,
    root: str | Path | None = None,
    overwrite: bool = False,
    default_snaps: int = 60,
    failure_snaps: int = 120,
) -> Path:
    root_path = Path(root) if root is not None else (ROOT / "sweep_runs")
    outdir_path = Path(outdir)
    if outdir_path.exists() and any(outdir_path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory already exists and is non-empty: {outdir_path}")
    outdir_path.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as _plt  # noqa: F401

        can_plot = True
    except ModuleNotFoundError:
        can_plot = False

    candidates = select_trace_comparison_candidates(root_path)
    results: list[dict[str, Any]] = []
    for candidate in candidates:
        n_snap = failure_snaps if candidate.get("group") == "k1_above_threshold_failure" else default_snaps
        result = replay_candidate(candidate, n_snap=n_snap)
        slug = _candidate_slug(candidate)
        candir = outdir_path / slug
        candir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(candir / "frames.npz", psi=result["frames"], times=np.asarray(result["times"], dtype=float))
        _write_json(candir / "diagnostic_summary.json", {k: v for k, v in result.items() if k != "frames"})
        if can_plot:
            _plot_trace_panel(result, candir / "trace_panel.png")
        results.append(result)
        print(f"{slug}: {result['summary']['diagnostic_label']} | class={candidate['klass']} K={candidate['K']}", flush=True)

    if can_plot:
        _plot_trace_overlay(results, outdir_path / "trace_overlay.png")
    rows = [_diagnostic_row(result) for result in results]
    fieldnames = [
        "group",
        "run_dir",
        "idx",
        "K",
        "ic_norm",
        "target_initial_mass",
        "class",
        "N",
        "T",
        "initial_mass",
        "time_to_failure",
        "time_to_blowup",
        "er_final_or_last",
        "late_energy_slope",
        "rho_peak_max",
        "core_radius_min",
        "compactness_max",
        "omega2_min_min",
        "grad_log_omega_max",
        "curvature_proxy_max",
        "node_count_mid",
        "node_count_last",
        "split_before_blowup",
        "high_k_fraction_max",
        "diagnostic_label",
    ]
    _write_csv(outdir_path / "mass_threshold_trace_comparison.csv", rows, fieldnames)
    _write_json(outdir_path / "trace_comparison_summary.json", {"rows": rows, "plots_rendered": can_plot})
    return outdir_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Post-run collapse/runaway diagnostic pass for selected Phase C candidates.")
    parser.add_argument("--outdir", help="Output directory. Defaults to sweep_runs/CORE_SAT_COLLAPSE_DIAG_<timestamp>.")
    parser.add_argument("--root", default=str(ROOT / "sweep_runs"), help="Root sweep_runs directory.")
    parser.add_argument("--n-snap", type=int, default=DEFAULT_SNAPS, help="Number of replay snapshots per candidate.")
    parser.add_argument(
        "--mode",
        choices=["pilot", "high-mass-comparison", "threshold-pilot", "trace-comparison", "render-trace-plots"],
        default="pilot",
        help="Diagnostic preset to run.",
    )
    parser.add_argument("--N-override", type=int, default=REF_FEB_N, help="Replay grid size for high-mass-comparison mode.")
    parser.add_argument("--T-override", type=int, default=REF_FEB_T, help="Replay horizon for high-mass-comparison mode.")
    parser.add_argument("--failure-n-snap", type=int, default=120, help="Dense snapshot count for the K=1 failure trace in high-mass-comparison mode.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "high-mass-comparison":
        outdir = args.outdir or str(ROOT / "sweep_runs" / f"CORE_SAT_HIGH_MASS_COMPARE_{time.strftime('%Y%m%d_%H%M%S')}")
        final_dir = run_high_mass_comparison(
            outdir,
            root=args.root,
            overwrite=bool(args.overwrite),
            N_override=int(args.N_override),
            T_override=int(args.T_override),
            default_snaps=int(args.n_snap),
            failure_snaps=int(args.failure_n_snap),
        )
    elif args.mode == "threshold-pilot":
        outdir = args.outdir or str(ROOT / "sweep_runs" / f"CORE_SAT_THRESHOLD_DIAG_{time.strftime('%Y%m%d_%H%M%S')}")
        final_dir = run_threshold_pilot(
            outdir,
            root=args.root,
            overwrite=bool(args.overwrite),
            n_snap=int(args.n_snap),
        )
    elif args.mode == "trace-comparison":
        outdir = args.outdir or str(ROOT / "sweep_runs" / f"CORE_SAT_TRACE_COMPARE_{time.strftime('%Y%m%d_%H%M%S')}")
        final_dir = run_trace_comparison(
            outdir,
            root=args.root,
            overwrite=bool(args.overwrite),
            default_snaps=int(args.n_snap),
            failure_snaps=int(args.failure_n_snap),
        )
    elif args.mode == "render-trace-plots":
        if not args.outdir:
            raise SystemExit("--outdir is required for render-trace-plots mode")
        final_dir = render_saved_trace_plots(args.outdir)
    else:
        outdir = args.outdir or str(ROOT / "sweep_runs" / f"CORE_SAT_COLLAPSE_DIAG_{time.strftime('%Y%m%d_%H%M%S')}")
        final_dir = run_diagnostic(outdir, root=args.root, n_snap=int(args.n_snap), overwrite=bool(args.overwrite))
    print(final_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
