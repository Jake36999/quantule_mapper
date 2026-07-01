"""
Focused Phase C robustness layer around the restored N96/T6000 threshold branches.

This driver is orchestration only:
- builds a deterministic dry-run manifest
- executes serial row-targeted replays through core_saturation_replay.py
- analyzes saved replay + diagnostic summaries into a compact report pack

It does not duplicate the solver or classifier path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_scout import core_saturation_search as css
from jax_scout import core_saturation_replay as csr
from jax_scout import core_saturation_collapse_diag as diag

REPLAY_N = 96
REPLAY_T = 6000
TRACE_SNAPS = 32
JITTER_SIGMA = 0.01
JITTER_SEED = 314159
IC_SEEDS = (20260619, 20260620)
PARAMETER_STATES = ("base", "jitter_01")

GROUP_SPECS = (
    {
        "group": "k1_low_mass_branch",
        "expected_branch_label": "K1_LOW_MASS_BRANCH_N96_SUPPORTED",
        "source_run": "CORE_SAT_HUNT_20260623_171758",
        "source_idx": 4,
        "K": 1,
        "replay_raw_targets": (4000.0, 4800.0, 5600.0),
    },
    {
        "group": "k1_high_mass_failure",
        "expected_branch_label": "K1_FAILURE_THRESHOLD_N96_SUPPORTED",
        "source_run": "CORE_SAT_HUNT_20260623_173417",
        "source_idx": 2,
        "K": 1,
        "replay_raw_targets": (7200.0, 9600.0),
    },
    {
        "group": "k6_distributed_branch",
        "expected_branch_label": "K6_DISTRIBUTED_BRANCH_N96_SUPPORTED",
        "source_run": "CORE_SAT_HUNT_20260623_173417",
        "source_idx": 10,
        "K": 6,
        "replay_raw_targets": (9600.0, 12800.0),
    },
    {
        "group": "k6_high_mass_branch",
        "expected_branch_label": "K6_DISTRIBUTED_BRANCH_N96_SUPPORTED",
        "source_run": "CORE_SAT_HUNT_20260623_175018",
        "source_idx": 10,
        "K": 6,
        "replay_raw_targets": (12800.0, 16402.349616),
    },
)

REFERENCE_SPEC = {
    "group": "feb56dc7_control",
    "expected_branch_label": "FEB_CONTROL_REPRODUCED",
    "ref": "feb56dc7",
}


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def default_outdir() -> Path:
    return ROOT / "sweep_runs" / f"CORE_SAT_THRESHOLD_BRANCH_ROBUSTNESS_{_timestamp()}"


def default_runtime_log(outdir: Path) -> Path:
    prefix = "CORE_SAT_THRESHOLD_BRANCH_ROBUSTNESS_"
    stamp = outdir.name[len(prefix):] if outdir.name.startswith(prefix) else outdir.name
    return ROOT / "runtime_logs" / f"phase_c_threshold_robustness_{stamp}.log"


def _to_float(value: Any) -> float | None:
    if value in ("", None, "None"):
        return None
    return float(value)


def _dx_weighted_mass(raw_mass: float, n: int) -> float:
    return float(raw_mass) * (css.L_ / float(n)) ** 3


def _output_slug(group: str, source_idx: int | None, target: float | None, ic_seed: int | None, parameter_state: str) -> str:
    if group == "feb56dc7_control":
        return "ref_feb56dc7"
    target_text = "na" if target is None else f"{float(target):.6f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"{group}_idx_{source_idx}_m_{target_text}_seed_{ic_seed}_{parameter_state}"


def build_parameter_state(base_params: dict[str, float], *, parameter_state: str, jitter_seed: int) -> dict[str, float]:
    params = dict(base_params)
    if parameter_state == "base":
        return params
    if parameter_state != "jitter_01":
        raise ValueError(f"Unsupported parameter_state '{parameter_state}'")
    rng = np.random.default_rng(int(jitter_seed))
    for name in css.order:
        params[name] = float(params[name] * (1.0 + rng.normal(0.0, JITTER_SIGMA)))
    params["param_eta"] = float(np.clip(params["param_eta"], css.REGIME["param_eta"][0], css.REGIME["param_eta"][1]))
    params["param_a"] = float(np.clip(params["param_a"], css.REGIME["param_a"][0], css.REGIME["param_a"][1]))
    return params


def _manifest_fieldnames() -> list[str]:
    return [
        "manifest_index",
        "kind",
        "group",
        "expected_branch_label",
        "source_run_dir",
        "source_csv",
        "source_idx",
        "K",
        "source_resolution_N",
        "replay_resolution_N",
        "replay_horizon_T",
        "base_target_mass",
        "replay_raw_target_mass",
        "dx_weighted_target_mass",
        "ic_seed",
        "parameter_state",
        "jitter_seed",
        "trace_snaps",
        "saved_params_json",
        "replay_params_json",
        "output_directory",
        "replay_command",
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, default=float)
        handle.write("\n")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_source_candidate(sweep_root: Path, *, run_name: str, idx: int) -> dict[str, Any]:
    csv_path = sweep_root / run_name / "all_evals.csv"
    return csr.resolve_candidate(csv_path, idx=idx)


def _command_to_string(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _build_replay_command(
    *,
    row: dict[str, Any],
    candidate: dict[str, Any] | None,
    replay_params: dict[str, float] | None,
) -> str:
    parts = ["python", "jax_scout/core_saturation_replay.py"]
    if row["kind"] == "reference_control":
        parts.extend(["--ref", "feb56dc7"])
    else:
        parts.extend(["--csv", row["source_csv"], "--idx", str(row["source_idx"])])
    parts.extend(["--N", str(row["replay_resolution_N"]), "--T", str(row["replay_horizon_T"])])
    parts.extend(["--out", row["output_directory"], "--overwrite"])
    if row["kind"] == "scaled_replay":
        parts.extend(["--target-initial-mass-override", f"{float(row['replay_raw_target_mass']):.12g}"])
        parts.extend(["--ic-seed-override", str(row["ic_seed"])])
        for name in css.order:
            parts.extend(["--param-override", f"{name}={float(replay_params[name]):.12g}"])
    parts.extend(["--trace-snaps", str(row["trace_snaps"])])
    return _command_to_string(parts)


def build_manifest(*, outdir: str | Path, sweep_root: str | Path | None = None) -> list[dict[str, Any]]:
    outdir_path = Path(outdir)
    sweep_root_path = Path(sweep_root) if sweep_root is not None else (ROOT / "sweep_runs")
    rows: list[dict[str, Any]] = []
    manifest_index = 0

    for spec in GROUP_SPECS:
        candidate = _load_source_candidate(sweep_root_path, run_name=spec["source_run"], idx=spec["source_idx"])
        base_params = dict(candidate["params"])
        for replay_raw_target_mass in spec["replay_raw_targets"]:
            for ic_seed in IC_SEEDS:
                for parameter_state in PARAMETER_STATES:
                    jitter_seed = None if parameter_state == "base" else int(JITTER_SEED)
                    replay_params = build_parameter_state(base_params, parameter_state=parameter_state, jitter_seed=JITTER_SEED)
                    slug = _output_slug(spec["group"], spec["source_idx"], replay_raw_target_mass, ic_seed, parameter_state)
                    row = {
                        "manifest_index": manifest_index,
                        "kind": "scaled_replay",
                        "group": spec["group"],
                        "expected_branch_label": spec["expected_branch_label"],
                        "source_run_dir": str((sweep_root_path / spec["source_run"]).resolve()),
                        "source_csv": str((sweep_root_path / spec["source_run"] / "all_evals.csv").resolve()),
                        "source_idx": int(spec["source_idx"]),
                        "K": int(spec["K"]),
                        "source_resolution_N": int(candidate["source"].get("source_resolution_N") or 48),
                        "replay_resolution_N": REPLAY_N,
                        "replay_horizon_T": REPLAY_T,
                        "base_target_mass": _to_float(candidate.get("target_initial_mass")),
                        "replay_raw_target_mass": float(replay_raw_target_mass),
                        "dx_weighted_target_mass": _dx_weighted_mass(float(replay_raw_target_mass), REPLAY_N),
                        "ic_seed": int(ic_seed),
                        "parameter_state": parameter_state,
                        "jitter_seed": jitter_seed,
                        "trace_snaps": TRACE_SNAPS,
                        "saved_params_json": json.dumps(base_params, sort_keys=True),
                        "replay_params_json": json.dumps(replay_params, sort_keys=True),
                        "output_directory": str((outdir_path / slug).resolve()),
                    }
                    row["replay_command"] = _build_replay_command(row=row, candidate=candidate, replay_params=replay_params)
                    rows.append(row)
                    manifest_index += 1

    ref_slug = _output_slug(REFERENCE_SPEC["group"], None, None, None, "base")
    ref_row = {
        "manifest_index": manifest_index,
        "kind": "reference_control",
        "group": REFERENCE_SPEC["group"],
        "expected_branch_label": REFERENCE_SPEC["expected_branch_label"],
        "source_run_dir": None,
        "source_csv": None,
        "source_idx": None,
        "K": 6,
        "source_resolution_N": None,
        "replay_resolution_N": REPLAY_N,
        "replay_horizon_T": REPLAY_T,
        "base_target_mass": None,
        "replay_raw_target_mass": None,
        "dx_weighted_target_mass": None,
        "ic_seed": css.SEED,
        "parameter_state": "base",
        "jitter_seed": None,
        "trace_snaps": TRACE_SNAPS,
        "saved_params_json": json.dumps(css.FEB, sort_keys=True),
        "replay_params_json": json.dumps(css.FEB, sort_keys=True),
        "output_directory": str((outdir_path / ref_slug).resolve()),
    }
    ref_row["replay_command"] = _build_replay_command(row=ref_row, candidate=None, replay_params=None)
    rows.append(ref_row)
    return rows


def write_manifest(*, outdir: str | Path, sweep_root: str | Path | None = None) -> tuple[Path, Path, list[dict[str, Any]]]:
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    rows = build_manifest(outdir=outdir_path, sweep_root=sweep_root)
    csv_path = outdir_path / "robustness_manifest.csv"
    json_path = outdir_path / "robustness_manifest.json"
    _write_csv(csv_path, rows, _manifest_fieldnames())
    _write_json(
        json_path,
        {
            "outdir": str(outdir_path.resolve()),
            "row_count": len(rows),
            "replay_resolution_N": REPLAY_N,
            "replay_horizon_T": REPLAY_T,
            "trace_snaps": TRACE_SNAPS,
            "jitter_sigma": JITTER_SIGMA,
            "jitter_seed": JITTER_SEED,
            "rows": rows,
        },
    )
    return csv_path, json_path, rows


def _load_manifest(outdir: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _read_json(Path(outdir) / "robustness_manifest.json")
    return payload, list(payload["rows"])


def run_manifest(*, outdir: str | Path, sweep_root: str | Path | None = None, runtime_log: str | Path | None = None) -> Path:
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    csv_path = outdir_path / "robustness_manifest.csv"
    json_path = outdir_path / "robustness_manifest.json"
    if not csv_path.exists() or not json_path.exists():
        write_manifest(outdir=outdir_path, sweep_root=sweep_root)
    _, rows = _load_manifest(outdir_path)
    runtime_log_path = Path(runtime_log) if runtime_log is not None else default_runtime_log(outdir_path)
    runtime_log_path.parent.mkdir(parents=True, exist_ok=True)

    status_rows: list[dict[str, Any]] = []
    with runtime_log_path.open("a", encoding="utf-8") as log:
        log.write(f"Launching threshold robustness batch at {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
        log.write(f"Outdir: {outdir_path.resolve()}\n")
        log.write(f"Manifest: {json_path.resolve()}\n")
        log.write(f"Rows: {len(rows)}\n")
        log.flush()
        for row in rows:
            cmd = row["replay_command"]
            out_path = Path(row["output_directory"])
            if (out_path / "summary.json").exists() and (out_path / "diagnostic_summary.json").exists():
                log.write(f"\n=== Row {row['manifest_index']} {row['group']} ===\n")
                log.write("Status: skipped_existing_complete_row\n")
                log.flush()
                status_rows.append(
                    {
                        "manifest_index": row["manifest_index"],
                        "group": row["group"],
                        "output_directory": row["output_directory"],
                        "returncode": 0,
                        "elapsed_seconds": 0.0,
                        "status": "skipped_existing_complete_row",
                    }
                )
                continue
            log.write(f"\n=== Row {row['manifest_index']} {row['group']} ===\n")
            log.write(f"Command: {cmd}\n")
            log.flush()
            started = time.time()
            replay_payload = csr.run_replay(
                csv_path=row["source_csv"],
                idx=row["source_idx"],
                ref="feb56dc7" if row["kind"] == "reference_control" else None,
                N=int(row["replay_resolution_N"]),
                T=int(row["replay_horizon_T"]),
                outdir=row["output_directory"],
                overwrite=True,
                target_initial_mass_override=row["replay_raw_target_mass"],
                ic_seed_override=None if row["kind"] == "reference_control" else int(row["ic_seed"]),
                param_override_specs=[] if row["kind"] == "reference_control" else [
                    f"{name}={json.loads(row['replay_params_json'])[name]}"
                    for name in css.order
                ],
                trace_snaps=int(row["trace_snaps"]),
                command_override=cmd,
            )
            elapsed_s = time.time() - started
            result = replay_payload["result"]
            log.write(
                f"{replay_payload['candidate']['label']} -> {result['klass']} | "
                f"n_fin={result['metrics']['n_fin']} slope={result['metrics']['late_slope']:+.3e} "
                f"er_fin={result['metrics']['er_fin']:.4f}\n"
            )
            log.write(f"{replay_payload['outdir']}\n")
            log.write("Return code: 0\n")
            log.write(f"Elapsed seconds: {elapsed_s:.1f}\n")
            log.flush()
            status_rows.append(
                {
                    "manifest_index": row["manifest_index"],
                    "group": row["group"],
                    "output_directory": row["output_directory"],
                    "returncode": 0,
                    "elapsed_seconds": float(elapsed_s),
                    "status": "ran",
                }
            )
    _write_json(outdir_path / "run_status.json", {"runtime_log": str(runtime_log_path.resolve()), "rows": status_rows})
    return runtime_log_path


def _float_or_none(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _compact_range(values: list[float]) -> dict[str, float | None]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return {"min": None, "max": None}
    return {"min": float(min(clean)), "max": float(max(clean))}


def _load_result_row(row: dict[str, Any]) -> dict[str, Any]:
    outdir = Path(row["output_directory"])
    summary = _read_json(outdir / "summary.json")
    diagnostic = _read_json(outdir / "diagnostic_summary.json") if (outdir / "diagnostic_summary.json").exists() else {}
    diag_summary = diagnostic.get("summary", {})
    metrics = summary.get("metrics", {})
    return {
        "manifest_index": row["manifest_index"],
        "group": row["group"],
        "expected_branch_label": row["expected_branch_label"],
        "source_run_dir": row["source_run_dir"],
        "source_idx": row["source_idx"],
        "K": row["K"],
        "ic_seed": row["ic_seed"],
        "parameter_state": row["parameter_state"],
        "jitter_seed": row["jitter_seed"],
        "base_target_mass": row["base_target_mass"],
        "replay_target_initial_mass": row["replay_raw_target_mass"],
        "dx_weighted_target_mass": row["dx_weighted_target_mass"],
        "class": summary.get("klass"),
        "diagnostic_label": diag_summary.get("diagnostic_label"),
        "final_node_count": metrics.get("n_fin"),
        "late_slope": metrics.get("late_slope"),
        "er_final": metrics.get("er_fin"),
        "er_max": metrics.get("er_max"),
        "compactness_max": diag_summary.get("compactness_max"),
        "high_k_fraction_max": diag_summary.get("high_k_fraction_max"),
        "time_to_failure": diag_summary.get("time_to_failure"),
        "node_count_last": diag_summary.get("node_count_last", metrics.get("n_fin")),
        "node_count_mid": diag_summary.get("node_count_mid", metrics.get("n_mid")),
        "split_before_blowup": diag_summary.get("split_before_blowup"),
        "output_directory": row["output_directory"],
        "replay_kind": summary.get("replay_kind"),
        "mass_scaling_mode": summary.get("mass_scaling_mode"),
    }


def _count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _seed_sensitivity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_seed: dict[str, dict[str, int]] = {}
    for row in rows:
        seed = str(row.get("ic_seed"))
        by_seed.setdefault(seed, {})
        klass = str(row.get("class"))
        by_seed[seed][klass] = by_seed[seed].get(klass, 0) + 1
    return by_seed


def _parameter_sensitivity(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    by_state: dict[str, dict[str, int]] = {}
    for row in rows:
        state = str(row.get("parameter_state"))
        by_state.setdefault(state, {})
        klass = str(row.get("class"))
        by_state[state][klass] = by_state[state].get(klass, 0) + 1
    return by_state


def _mass_sensitivity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        mass = _float_or_none(row.get("replay_target_initial_mass"))
        if mass is None:
            continue
        buckets.setdefault(mass, []).append(row)
    out = []
    for mass in sorted(buckets):
        bucket = buckets[mass]
        out.append(
            {
                "replay_target_initial_mass": float(mass),
                "class_counts": _count_by(bucket, "class"),
                "final_node_counts": _count_by(bucket, "node_count_last"),
            }
        )
    return out


def _group_verdict(group: str, rows: list[dict[str, Any]]) -> str:
    total = len(rows)
    majority = max(1, math.ceil(0.6 * total))
    true_near = sum(1 for row in rows if row.get("class") in {"TRUE_SATURATED_BOUND_STATE", "NEAR_SATURATED_BOUND_STATE"})
    failures = sum(1 for row in rows if row.get("class") not in {"TRUE_SATURATED_BOUND_STATE", "NEAR_SATURATED_BOUND_STATE"})
    low_node_localized = sum(
        1
        for row in rows
        if row.get("class") in {"TRUE_SATURATED_BOUND_STATE", "NEAR_SATURATED_BOUND_STATE"}
        and int(row.get("node_count_last") or 0) <= 3
    )
    distributed = sum(
        1
        for row in rows
        if row.get("class") in {"TRUE_SATURATED_BOUND_STATE", "NEAR_SATURATED_BOUND_STATE"}
        and 4 <= int(row.get("node_count_last") or 0) <= 6
    )
    if group == "k1_low_mass_branch":
        return "K1_LOW_MASS_BRANCH_ROBUST" if true_near >= majority and low_node_localized >= majority else "K1_LOW_MASS_BRANCH_FRAGILE"
    if group == "k1_high_mass_failure":
        failure_majority = max(1, math.ceil(0.75 * total))
        return "K1_FAILURE_BOUNDARY_ROBUST" if failures >= failure_majority and true_near == 0 else "K1_FAILURE_BOUNDARY_INCONCLUSIVE"
    if group in {"k6_distributed_branch", "k6_high_mass_branch"}:
        return "K6_DISTRIBUTED_BRANCH_ROBUST" if true_near >= majority and distributed >= majority else "K6_DISTRIBUTED_BRANCH_FRAGILE"
    if group == "feb56dc7_control":
        return "FEB_CONTROL_REPRODUCED" if total == 1 and rows[0].get("class") == "TRUE_SATURATED_BOUND_STATE" else "BRANCH_ROBUSTNESS_INCONCLUSIVE"
    return "BRANCH_ROBUSTNESS_INCONCLUSIVE"


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["group"]), []).append(row)

    group_summaries: dict[str, Any] = {}
    group_verdicts: dict[str, str] = {}
    for group, group_rows in groups.items():
        group_verdicts[group] = _group_verdict(group, group_rows)
        group_summaries[group] = {
            "total_rows": len(group_rows),
            "class_counts": _count_by(group_rows, "class"),
            "diagnostic_labels": _count_by(group_rows, "diagnostic_label"),
            "final_node_count_distribution": _count_by(group_rows, "node_count_last"),
            "median_late_slope": _median([row["late_slope"] for row in group_rows if _float_or_none(row["late_slope"]) is not None]),
            "median_er_final": _median([row["er_final"] for row in group_rows if _float_or_none(row["er_final"]) is not None]),
            "compactness_range": _compact_range([row["compactness_max"] for row in group_rows if _float_or_none(row["compactness_max"]) is not None]),
            "high_k_range": _compact_range([row["high_k_fraction_max"] for row in group_rows if _float_or_none(row["high_k_fraction_max"]) is not None]),
            "seed_sensitivity": _seed_sensitivity(group_rows),
            "parameter_jitter_sensitivity": _parameter_sensitivity(group_rows),
            "mass_sensitivity": _mass_sensitivity(group_rows),
        }

    return {
        "overall_rows": len(rows),
        "overall_class_counts": _count_by(rows, "class"),
        "overall_diagnostic_labels": _count_by(rows, "diagnostic_label"),
        "group_verdicts": group_verdicts,
        "groups": group_summaries,
    }


def _doc_lines(outdir: Path, summary: dict[str, Any]) -> list[str]:
    lines = [
        "# Phase C Threshold Branch Robustness",
        "",
        "## Purpose",
        "",
        "Test whether the restored N96/T6000 threshold shortlist remains stable under small local perturbations in target mass, IC seed, and nearby parameters without launching a broad search.",
        "",
        "## Method",
        "",
        f"- Resolution-fair replay grid: `N={REPLAY_N}`, `T={REPLAY_T}`",
        f"- Serial replay rows: `{summary['overall_rows']}`",
        f"- Trace capture: `{TRACE_SNAPS}` sparse snapshots per row",
        f"- Parameter jitter: multiplicative Gaussian `sigma={JITTER_SIGMA:.0%}` with deterministic seed `{JITTER_SEED}`",
        "- Cross-resolution rows use explicit raw-target overrides through `core_saturation_replay.py`.",
        "",
        "## Overall Results",
        "",
        f"- Class counts: `{json.dumps(summary['overall_class_counts'], sort_keys=True)}`",
        f"- Diagnostic labels: `{json.dumps(summary['overall_diagnostic_labels'], sort_keys=True)}`",
        "",
        "## Group Verdicts",
        "",
    ]
    for group, verdict in summary["group_verdicts"].items():
        lines.append(f"- `{group}` -> `{verdict}`")
    lines.extend(["", "## By Group", ""])
    for group, payload in summary["groups"].items():
        lines.extend(
            [
                f"### {group}",
                "",
                f"- Total rows: `{payload['total_rows']}`",
                f"- Class counts: `{json.dumps(payload['class_counts'], sort_keys=True)}`",
                f"- Diagnostic labels: `{json.dumps(payload['diagnostic_labels'], sort_keys=True)}`",
                f"- Final node-count distribution: `{json.dumps(payload['final_node_count_distribution'], sort_keys=True)}`",
                f"- Median late slope: `{payload['median_late_slope']}`",
                f"- Median er_final: `{payload['median_er_final']}`",
                f"- Compactness range: `{payload['compactness_range']}`",
                f"- High-k range: `{payload['high_k_range']}`",
                f"- Seed sensitivity: `{json.dumps(payload['seed_sensitivity'], sort_keys=True)}`",
                f"- Parameter-state sensitivity: `{json.dumps(payload['parameter_jitter_sensitivity'], sort_keys=True)}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Caveats",
            "",
            "- This is a focused robustness layer around shortlisted branches, not a population law.",
            "- Interpretation remains branch-local even when a verdict is robust.",
            "- No PDE, solver, or classifier thresholds were changed for this pass.",
            "",
            "## Next Actions",
            "",
            "- If both K=1 and K=6 branches remain locally robust, design the next structured long hunt with explicit K, mass, seed, and provenance separation.",
            "- If the K=1 branch weakens while K=6 remains stable, focus the next discovery layer on distributed high-mass branches plus a few low-K controls.",
            "- If only feb56dc7 stays stable, return to local refinement before any broader search.",
            "",
        ]
    )
    return lines


def analyze_manifest(*, outdir: str | Path) -> tuple[Path, Path]:
    outdir_path = Path(outdir)
    _, manifest_rows = _load_manifest(outdir_path)
    result_rows = [_load_result_row(row) for row in manifest_rows]
    csv_path = outdir_path / "threshold_branch_robustness.csv"
    _write_csv(
        csv_path,
        result_rows,
        [
            "manifest_index",
            "group",
            "expected_branch_label",
            "source_run_dir",
            "source_idx",
            "K",
            "ic_seed",
            "parameter_state",
            "jitter_seed",
            "base_target_mass",
            "replay_target_initial_mass",
            "dx_weighted_target_mass",
            "class",
            "diagnostic_label",
            "final_node_count",
            "late_slope",
            "er_final",
            "er_max",
            "compactness_max",
            "high_k_fraction_max",
            "time_to_failure",
            "node_count_last",
            "node_count_mid",
            "split_before_blowup",
            "output_directory",
            "replay_kind",
            "mass_scaling_mode",
        ],
    )
    summary = summarize_results(result_rows)
    _write_json(outdir_path / "threshold_branch_robustness_summary.json", summary)
    doc_path = ROOT / "docs" / "PHASE_C_THRESHOLD_BRANCH_ROBUSTNESS.md"
    doc_path.write_text("\n".join(_doc_lines(outdir_path, summary)), encoding="utf-8")
    return csv_path, doc_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Focused Phase C robustness batch around restored threshold branches.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--analyze", action="store_true")
    parser.add_argument("--outdir", default=str(default_outdir()), help="Output root for manifest, replay rows, and analysis.")
    parser.add_argument("--sweep-root", default=str(ROOT / "sweep_runs"), help="Source sweep_runs root.")
    parser.add_argument("--runtime-log", default=None, help="Optional explicit runtime log path for --run.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outdir = Path(args.outdir)
    if args.dry_run:
        csv_path, json_path, rows = write_manifest(outdir=outdir, sweep_root=args.sweep_root)
        print(csv_path)
        print(json_path)
        print(f"rows={len(rows)}")
        return 0
    if args.run:
        if not (outdir / "robustness_manifest.json").exists():
            write_manifest(outdir=outdir, sweep_root=args.sweep_root)
        runtime_log = run_manifest(outdir=outdir, sweep_root=args.sweep_root, runtime_log=args.runtime_log)
        print(runtime_log)
        return 0
    csv_path, doc_path = analyze_manifest(outdir=outdir)
    print(csv_path)
    print(doc_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
