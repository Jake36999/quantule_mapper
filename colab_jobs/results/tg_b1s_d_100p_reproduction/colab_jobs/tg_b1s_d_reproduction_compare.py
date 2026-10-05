from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any


VALID_FINAL_STATUSES = {
    "REPRODUCTION_WITHIN_DECLARED_TOLERANCES",
    "REPRODUCTION_OUTSIDE_DECLARED_TOLERANCES",
    "REPRODUCTION_INCOMPLETE",
    "REPRODUCTION_INVALID",
}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_hashes(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    return rows


def parse_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def rel_diff(a: float, b: float) -> float:
    if not math.isfinite(a) or not math.isfinite(b):
        return float("nan")
    return abs(b - a) / max(abs(a), abs(b), 1e-30)


def status_from_bool(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def find_row(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise KeyError(f"Missing row where {key}={value}")


def add_numeric_comparison(
    rows: list[dict[str, Any]],
    *,
    observable: str,
    baseline: float,
    reproduced: float,
    abs_tol: float | None,
    rel_tol: float | None,
    source: str,
    gate: str,
) -> None:
    abs_difference = abs(reproduced - baseline) if math.isfinite(baseline) and math.isfinite(reproduced) else float("nan")
    relative_difference = rel_diff(baseline, reproduced)
    ok = True
    if abs_tol is not None:
        ok = ok and math.isfinite(abs_difference) and abs_difference <= abs_tol
    if rel_tol is not None:
        ok = ok and math.isfinite(relative_difference) and relative_difference <= rel_tol
    if abs_tol is None and rel_tol is None:
        ok = math.isfinite(reproduced)
    rows.append(
        {
            "observable": observable,
            "baseline_value": finite_or_none(baseline),
            "reproduced_value": finite_or_none(reproduced),
            "absolute_difference": finite_or_none(abs_difference),
            "relative_difference": finite_or_none(relative_difference),
            "declared_absolute_tolerance": abs_tol,
            "declared_relative_tolerance": rel_tol,
            "comparison_status": status_from_bool(ok),
            "source_table_field": source,
            "gate_source": gate,
        }
    )


def add_threshold_comparison(
    rows: list[dict[str, Any]],
    *,
    observable: str,
    baseline: float,
    reproduced: float,
    threshold: float,
    direction: str,
    source: str,
    gate: str,
) -> None:
    if direction == "min":
        ok = math.isfinite(reproduced) and reproduced >= threshold
    elif direction == "max":
        ok = math.isfinite(reproduced) and reproduced <= threshold
    else:
        raise ValueError(direction)
    rows.append(
        {
            "observable": observable,
            "baseline_value": finite_or_none(baseline),
            "reproduced_value": finite_or_none(reproduced),
            "absolute_difference": finite_or_none(abs(reproduced - baseline)) if math.isfinite(baseline) and math.isfinite(reproduced) else None,
            "relative_difference": finite_or_none(rel_diff(baseline, reproduced)),
            "declared_absolute_tolerance": None,
            "declared_relative_tolerance": None,
            "declared_threshold": threshold,
            "threshold_direction": direction,
            "comparison_status": status_from_bool(ok),
            "source_table_field": source,
            "gate_source": gate,
        }
    )


def add_exact_comparison(
    rows: list[dict[str, Any]],
    *,
    observable: str,
    baseline: Any,
    reproduced: Any,
    source: str,
    gate: str,
) -> None:
    rows.append(
        {
            "observable": observable,
            "baseline_value": baseline,
            "reproduced_value": reproduced,
            "absolute_difference": None,
            "relative_difference": None,
            "declared_absolute_tolerance": 0,
            "declared_relative_tolerance": 0,
            "comparison_status": status_from_bool(str(baseline) == str(reproduced)),
            "source_table_field": source,
            "gate_source": gate,
        }
    )


def build_reproduction_observables(run_dir: Path, target_run: str) -> dict[str, Any]:
    long_rows = read_csv(run_dir / "long_time_100_period.csv")
    asym_rows = read_csv(run_dir / "asymptotic_frequency.csv")
    drift_rows = read_csv(run_dir / "drift_channel_fits.csv")
    falsification_rows = read_csv(run_dir / "falsification_results.csv")
    long_target = find_row(long_rows, "run_id", target_run)
    asym_target = find_row(asym_rows, "run_id", target_run)
    drift_by_channel = {
        row["channel"]: row
        for row in drift_rows
        if row.get("run_id") == target_run
    }
    falsification_by_test = {row["test"]: row for row in falsification_rows}
    return {
        "status": None,
        "target_run": target_run,
        "delta_omega_infty": parse_float(asym_target.get("delta_omega_infty")),
        "final_delta_theta": parse_float(long_target.get("delta_modal_phase_final")),
        "final_orbital_distance": parse_float(long_target.get("orbital_distance_final")),
        "final_phase_aligned_distance": parse_float(long_target.get("phase_aligned_distance_final")),
        "full_profile_overlap": parse_float(long_target.get("full_profile_overlap")),
        "full_modal_leakage_max": parse_float(long_target.get("full_modal_leakage_max")),
        "ledger_residual_abs": parse_float(long_target.get("ledger_residual_abs")),
        "delta_theta_classification": drift_by_channel.get("delta_theta", {}).get("classification"),
        "d_orbital_classification": drift_by_channel.get("d_orbital", {}).get("classification"),
        "modal_leakage_classification": drift_by_channel.get("modal_leakage", {}).get("classification"),
        "delta_A_core_classification": drift_by_channel.get("delta_A_core", {}).get("classification"),
        "delta_width_classification": drift_by_channel.get("delta_width", {}).get("classification"),
        "delta_E_core_classification": drift_by_channel.get("delta_E_core", {}).get("classification"),
        "delta_Q_classification": drift_by_channel.get("delta_Q", {}).get("classification"),
        "constant_slope_supported": str(asym_target.get("constant_slope_supported")).lower() == "true",
        "early_late_relative_difference": parse_float(asym_target.get("early_late_relative_difference")),
        "structural_channels": falsification_by_test.get("phase_aligned_structure_bounded", {}).get("structural_channels", ""),
    }


def validate_required_outputs(run_dir: Path, expected_files: list[str]) -> list[str]:
    missing = []
    for rel in expected_files:
        if not (run_dir / rel).exists():
            missing.append(rel)
    return missing


def compare(baseline_manifest: dict[str, Any], run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline = baseline_manifest["baseline_observables"]
    tolerances = baseline_manifest["declared_tolerances"]
    target_run = baseline_manifest["target_run"]
    reproduced = build_reproduction_observables(run_dir, target_run)
    rows: list[dict[str, Any]] = []

    add_numeric_comparison(
        rows,
        observable="delta_omega_infty",
        baseline=float(baseline["delta_omega_infty"]),
        reproduced=float(reproduced["delta_omega_infty"]),
        abs_tol=None,
        rel_tol=float(tolerances["delta_omega_relative_difference_max"]["value"]),
        source="asymptotic_frequency.csv:delta_omega_infty",
        gate="D4 validation relative_slope_difference < 0.50",
    )
    add_threshold_comparison(
        rows,
        observable="full_profile_overlap",
        baseline=float(baseline["full_profile_overlap"]),
        reproduced=float(reproduced["full_profile_overlap"]),
        threshold=float(tolerances["profile_overlap_min"]["value"]),
        direction="min",
        source="long_time_100_period.csv:full_profile_overlap",
        gate="D4/D5 full_profile_overlap > 0.999",
    )
    add_threshold_comparison(
        rows,
        observable="full_modal_leakage_max",
        baseline=float(baseline["full_modal_leakage_max"]),
        reproduced=float(reproduced["full_modal_leakage_max"]),
        threshold=float(tolerances["modal_leakage_max"]["value"]),
        direction="max",
        source="long_time_100_period.csv:full_modal_leakage_max",
        gate="D4/D5 full_modal_leakage_max < 1e-3",
    )
    add_threshold_comparison(
        rows,
        observable="ledger_residual_abs",
        baseline=float(baseline["ledger_residual_abs"]),
        reproduced=float(reproduced["ledger_residual_abs"]),
        threshold=float(tolerances["ledger_residual_abs_max"]["value"]),
        direction="max",
        source="long_time_100_period.csv:ledger_residual_abs",
        gate="D4 ledger_residual_abs < 1e-4",
    )
    add_threshold_comparison(
        rows,
        observable="early_late_relative_difference",
        baseline=float(baseline["early_late_relative_difference"]),
        reproduced=float(reproduced["early_late_relative_difference"]),
        threshold=float(tolerances["constant_slope_early_late_relative_difference_max"]["value"]),
        direction="max",
        source="asymptotic_frequency.csv:early_late_relative_difference",
        gate="constant_slope_supported threshold < 0.35",
    )
    add_exact_comparison(
        rows,
        observable="constant_slope_supported",
        baseline=bool(baseline["constant_slope_supported"]),
        reproduced=bool(reproduced["constant_slope_supported"]),
        source="asymptotic_frequency.csv:constant_slope_supported",
        gate="constant_slope_supported exact status",
    )
    add_exact_comparison(
        rows,
        observable="delta_theta_classification",
        baseline=baseline["delta_theta_classification"],
        reproduced=reproduced["delta_theta_classification"],
        source="drift_channel_fits.csv:delta_theta.classification",
        gate="phase channel required classification LINEAR_SECULAR",
    )
    for channel in ("d_orbital", "modal_leakage", "delta_A_core", "delta_width", "delta_E_core", "delta_Q"):
        key = f"{channel}_classification"
        add_exact_comparison(
            rows,
            observable=key,
            baseline=baseline[key],
            reproduced=reproduced[key],
            source=f"drift_channel_fits.csv:{channel}.classification",
            gate="structural channels must not become secular/accelerating",
        )
    add_exact_comparison(
        rows,
        observable="structural_channels",
        baseline=baseline["structural_channels"],
        reproduced=reproduced["structural_channels"],
        source="falsification_results.csv:phase_aligned_structure_bounded.structural_channels",
        gate="no structural secular/accelerating channels",
    )

    for observable in ("final_delta_theta", "final_orbital_distance", "final_phase_aligned_distance"):
        add_numeric_comparison(
            rows,
            observable=observable,
            baseline=float(baseline[observable]),
            reproduced=float(reproduced[observable]),
            abs_tol=None,
            rel_tol=None,
            source=f"long_time_100_period.csv:{observable}",
            gate="informational only; no declared numeric cross-device tolerance in baseline contract",
        )

    failures = [row for row in rows if row["comparison_status"] != "PASS"]
    status = "REPRODUCTION_WITHIN_DECLARED_TOLERANCES" if not failures else "REPRODUCTION_OUTSIDE_DECLARED_TOLERANCES"
    summary = {
        "schema_version": "qm-reproduction-comparison/1.0",
        "comparison_status": status,
        "valid_statuses": sorted(VALID_FINAL_STATUSES),
        "target_run": target_run,
        "baseline_id": baseline_manifest["baseline_id"],
        "failed_observables": [row["observable"] for row in failures],
        "compared_observable_count": len(rows),
        "informational_observables_without_declared_tolerances": [
            "final_delta_theta",
            "final_orbital_distance",
            "final_phase_aligned_distance",
        ],
        "scientific_claim_boundary": "Infrastructure/numerical reproduction only; no new scientific conclusion is inferred.",
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a TG-B1S-D reproduction against a frozen baseline.")
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--reproduction-run", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    baseline_path = Path(args.baseline_manifest)
    run_dir = Path(args.reproduction_run)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_manifest = read_json(baseline_path)
    missing = validate_required_outputs(run_dir, baseline_manifest["expected_result_files"])
    command_record = {
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "timestamp_utc": now_iso(),
    }
    write_json(out_dir / "analysis_command.json", command_record)

    shutil.copy2(baseline_path, out_dir / "baseline_manifest.json")
    environment = {
        "timestamp_utc": now_iso(),
        "python": sys.version,
        "platform": platform.platform(),
        "reproduction_run": str(run_dir),
    }
    write_json(out_dir / "environment_report.json", environment)

    if missing:
        summary = {
            "comparison_status": "REPRODUCTION_INCOMPLETE",
            "missing_outputs": missing,
            "scientific_claim_boundary": "No scientific conclusion inferred.",
        }
        write_json(out_dir / "comparison_summary.json", summary)
        write_json(out_dir / "completion_status.json", summary)
        write_json(out_dir / "artifact_hashes.json", {"files": artifact_hashes(out_dir)})
        raise SystemExit(f"Missing required reproduction outputs: {missing}")

    rows, summary = compare(baseline_manifest, run_dir)
    write_csv(out_dir / "observable_comparison.csv", rows)
    write_json(out_dir / "observable_comparison.json", {"rows": rows})

    reproduction_manifest = {
        "schema_version": "qm-reproduction/1.0",
        "reproduction_run": str(run_dir),
        "target_run": baseline_manifest["target_run"],
        "artifact_hashes": artifact_hashes(run_dir),
    }
    write_json(out_dir / "reproduction_manifest.json", reproduction_manifest)
    write_json(out_dir / "comparison_summary.json", summary)
    write_json(
        out_dir / "completion_status.json",
        {
            "status": summary["comparison_status"],
            "completed_utc": now_iso(),
            "valid_statuses": sorted(VALID_FINAL_STATUSES),
        },
    )
    write_json(out_dir / "artifact_hashes.json", {"files": artifact_hashes(out_dir)})
    if summary["comparison_status"] == "REPRODUCTION_OUTSIDE_DECLARED_TOLERANCES":
        raise SystemExit("Reproduction outside declared tolerances.")


if __name__ == "__main__":
    main()
