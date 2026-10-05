"""Phase R TG-B2 robustness completion driver.

This v2 capsule completes the evidence missing from the returned v1 run without
rerunning the two successful v1 sentinels. It imports the reviewed v1 evidence,
runs the epsilon rows first, then attempts a repaired larger-box sentinel
(`L=20,N=96`). Independent row failures are recorded and the archive is still
allowed to return for review.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REQUIRED_PROTECTED_FILES = [
    "jax_scout/phase_d_c3_wave.py",
    "jax_scout/gravity_TG_B1S_state_load_feedback_gpu.py",
    "jax_scout/gravity_TG_B2_two_node_awell.py",
    "jax_scout/gravity_TG_B2_definitive_force.py",
]

MATRIX = [
    {
        "row_id": "r2_eps_003_sep3",
        "phase": "R2_epsilon_scaling",
        "reason": "epsilon half-scale point",
        "args": ["--seps", "3.0", "--epsilon-G", "0.03", "--T", "180"],
        "epsilon_G": 0.03,
    },
    {
        "row_id": "r2_eps_012_sep3",
        "phase": "R2_epsilon_scaling",
        "reason": "epsilon double-scale point",
        "args": ["--seps", "3.0", "--epsilon-G", "0.12", "--T", "180"],
        "epsilon_G": 0.12,
    },
    {
        "row_id": "r1_box_L20_N96_sep3",
        "phase": "R1_convergence",
        "reason": "larger-box sentinel repaired from failed v1 L20,N64 row",
        "args": ["--seps", "3.0", "--L", "20", "--N", "96", "--T", "180"],
        "compare_to_baseline_sep": 3.0,
        "drift_tolerance": 0.20,
        "replaces_failed_v1_row": "r1_box_L20_sep3",
    },
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=float), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_protected_files(project_root: Path, registry_path: Path) -> list[dict[str, Any]]:
    registry = load_json(registry_path)
    files = registry.get("files", {})
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for rel in REQUIRED_PROTECTED_FILES:
        if rel not in files:
            failures.append(f"{rel}: missing from protected registry")
            continue
        path = project_root / rel
        if not path.exists():
            failures.append(f"{rel}: missing from capsule")
            continue
        expected = str(files[rel]["sha256"])
        actual = sha256_file(path)
        ok = actual == expected
        records.append(
            {
                "path": rel,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "ok": ok,
                "tier": files[rel].get("tier"),
            }
        )
        if not ok:
            failures.append(f"{rel}: hash mismatch")
    if failures:
        raise RuntimeError("Protected-file verification failed: " + "; ".join(failures))
    return records


def stream_subprocess(command: list[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="") as log:
        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return proc.wait()


def baseline_by_sep(baseline: dict[str, Any]) -> dict[float, dict[str, Any]]:
    return {float(row["sep"]): row for row in baseline["baseline_rows"]}


def row_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    rows = summary.get("rows", [])
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one sep row in robustness run, got {len(rows)}")
    row = dict(rows[0])
    well = float(row["F_R_well_avg"])
    hill = float(row["F_R_hill_avg"])
    scale = max(abs(well), abs(hill), 1e-30)
    row["well_hill_antisym_rel"] = abs(well + hill) / scale
    row["well_negative"] = well < 0.0
    return row


def linear_fit_through_origin(xs: list[float], ys: list[float]) -> dict[str, float]:
    denom = sum(x * x for x in xs)
    slope = sum(x * y for x, y in zip(xs, ys)) / denom
    predictions = [slope * x for x in xs]
    mean_y = sum(ys) / len(ys)
    sse = sum((y - p) ** 2 for y, p in zip(ys, predictions))
    sst = sum((y - mean_y) ** 2 for y in ys)
    r2 = 1.0 if sst == 0.0 and sse == 0.0 else 1.0 - sse / (sst + 1e-30)
    return {"slope": slope, "r2": r2, "sse": sse}


def all_required_completed(rows: list[dict[str, Any]]) -> bool:
    required = {item["row_id"] for item in MATRIX}
    complete = {row["row_id"] for row in rows if row.get("status") == "COMPLETE"}
    return required.issubset(complete)


def assess(combined_rows: list[dict[str, Any]], baseline: dict[str, Any]) -> dict[str, Any]:
    baseline_rows = baseline_by_sep(baseline)
    baseline_sep3 = baseline_rows[3.0]
    baseline_force = float(baseline_sep3["F_R_well_avg"])
    baseline_eps = float(baseline["config"]["epsilon_G"])

    completed = [row for row in combined_rows if row.get("status") == "COMPLETE" and "metrics" in row]
    failed = [row for row in combined_rows if row.get("status") != "COMPLETE"]
    sign_gate_failures: list[str] = []
    convergence_failures: list[str] = []
    convergence_rows: list[dict[str, Any]] = []
    eps_points = [(baseline_eps, baseline_force)]

    for rec in completed:
        metrics = rec["metrics"]
        if not (
            metrics.get("gates_pass")
            and metrics.get("well_negative")
            and metrics.get("sign_reverses")
            and metrics.get("off_null_ok")
        ):
            sign_gate_failures.append(rec["row_id"])

        if rec["phase"] == "R1_convergence":
            drift = rec.get("drift_vs_baseline_sep3")
            if drift in ("", None):
                drift = abs(float(metrics["F_R_well_avg"]) - baseline_force) / (abs(baseline_force) + 1e-30)
                rec["drift_vs_baseline_sep3"] = drift
            rec["drift_tolerance"] = float(rec.get("drift_tolerance", 0.20))
            rec["drift_pass"] = bool(drift <= rec["drift_tolerance"])
            convergence_rows.append(rec)
            if not rec["drift_pass"]:
                convergence_failures.append(rec["row_id"])
        elif rec["phase"] == "R2_epsilon_scaling":
            eps_points.append((float(rec["epsilon_G"]), float(metrics["F_R_well_avg"])))

    eps_points = sorted(eps_points)
    eps_complete = len(eps_points) == 3
    if eps_complete:
        fit = linear_fit_through_origin([point[0] for point in eps_points], [point[1] for point in eps_points])
        eps_signs_ok = all(y < 0.0 for _, y in eps_points)
        eps_r2_pass = fit["r2"] >= 0.99
    else:
        fit = {"slope": None, "r2": None, "sse": None}
        eps_signs_ok = False
        eps_r2_pass = False

    if failed or not all_required_completed(combined_rows) or not eps_complete:
        verdict = "TG_R_ROBUSTNESS_PARTIAL"
    elif sign_gate_failures:
        verdict = "TG_R_ROBUSTNESS_FAIL"
    elif convergence_failures or not eps_signs_ok or not eps_r2_pass:
        verdict = "TG_R_ROBUSTNESS_PARTIAL"
    else:
        verdict = "TG_R_ROBUSTNESS_PASS"

    return {
        "verdict": verdict,
        "baseline_sep3_F_R_well_avg": baseline_force,
        "completed_rows": [row["row_id"] for row in completed],
        "failed_rows": [row["row_id"] for row in failed],
        "sign_gate_failures": sign_gate_failures,
        "convergence_failures": convergence_failures,
        "epsilon_points": [{"epsilon_G": x, "F_R_well_avg": y} for x, y in eps_points],
        "epsilon_complete": eps_complete,
        "epsilon_fit_through_origin": fit,
        "epsilon_signs_ok": eps_signs_ok,
        "epsilon_r2_pass": eps_r2_pass,
        "convergence_rows": [
            {
                "row_id": rec["row_id"],
                "source": rec.get("source", "v2_capsule"),
                "F_R_well_avg": rec["metrics"]["F_R_well_avg"],
                "drift_vs_baseline_sep3": rec["drift_vs_baseline_sep3"],
                "drift_tolerance": rec["drift_tolerance"],
                "drift_pass": rec["drift_pass"],
            }
            for rec in convergence_rows
        ],
    }


def write_matrix_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "row_id",
        "phase",
        "reason",
        "source",
        "status",
        "return_code",
        "duration_seconds",
        "sep",
        "F_R_well_avg",
        "F_R_hill_avg",
        "F_R_off_avg",
        "F_R_well_std",
        "well_hill_antisym_rel",
        "well_negative",
        "sign_reverses",
        "off_null_ok",
        "charge_ok",
        "nodes_distinct",
        "gates_pass",
        "sep_min",
        "drift_vs_baseline_sep3",
        "drift_pass",
        "epsilon_G",
        "log_path",
        "summary_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rec in rows:
            metrics = rec.get("metrics", {})
            out = {key: rec.get(key, metrics.get(key, "")) for key in fields}
            writer.writerow(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--v1-evidence", required=True)
    parser.add_argument("--protected-registry", default="docs/gravity_maturity/TG_PROTECTED_REGISTRY.json")
    parser.add_argument("--entrypoint", default="jax_scout/gravity_TG_B2_definitive_force.py")
    args = parser.parse_args()

    project_root = Path.cwd()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logs_dir = out / "logs"
    rows_dir = out / "rows"

    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    baseline_path = project_root / args.baseline_summary
    v1_path = project_root / args.v1_evidence
    registry_path = project_root / args.protected_registry
    baseline = load_json(baseline_path)
    v1_evidence = load_json(v1_path)
    protected_records = verify_protected_files(project_root, registry_path)

    shutil.copy2(baseline_path, out / "baseline_reference.json")
    shutil.copy2(v1_path, out / "v1_partial_evidence_reference.json")
    write_json(out / "protected_hash_verification.json", {"files": protected_records})
    write_json(
        out / "robustness_matrix_plan.json",
        {
            "started_utc": started_utc,
            "v2_policy": "complete missing evidence; do not rerun successful v1 rows",
            "matrix": MATRIX,
            "baseline_summary": args.baseline_summary,
            "v1_evidence": args.v1_evidence,
            "protected_registry": args.protected_registry,
            "entrypoint": args.entrypoint,
            "boundary": "Phase R robustness only; mirror TG-B2 A-well branch; frozen modules imported unchanged.",
        },
    )

    prior_records = []
    for rec in v1_evidence.get("rows", []):
        item = dict(rec)
        item.setdefault("source", "v1_returned_capsule")
        item.setdefault("return_code", 0)
        item.setdefault("duration_seconds", "")
        item.setdefault("log_path", "")
        item.setdefault("summary_path", "")
        prior_records.append(item)

    v2_records: list[dict[str, Any]] = []
    for item in MATRIX:
        row_id = item["row_id"]
        row_out = rows_dir / row_id
        log_path = logs_dir / f"{row_id}.log"
        command = [sys.executable, "-u", args.entrypoint, *item["args"], "--out", str(row_out)]
        print(f"\n=== Phase R v2 row {row_id}: {' '.join(command)} ===", flush=True)
        started = time.time()
        write_json(row_out / f"{row_id}_DRIVER_STARTED.json", {"row_id": row_id, "command": command})
        return_code = stream_subprocess(command, project_root, log_path)
        duration = time.time() - started
        rec: dict[str, Any] = {
            **item,
            "source": "v2_capsule",
            "return_code": return_code,
            "duration_seconds": duration,
            "command": command,
            "log_path": str(log_path.relative_to(out)),
            "summary_path": str((row_out / "summary.json").relative_to(out)),
        }
        if return_code != 0:
            rec["status"] = "FAILED_SUBPROCESS"
            v2_records.append(rec)
            write_json(row_out / f"{row_id}_DRIVER_FAILED.json", rec)
            print(f"Row {row_id} failed with return code {return_code}; continuing to preserve independent evidence.", flush=True)
            continue
        summary = load_json(row_out / "summary.json")
        rec["status"] = "COMPLETE"
        rec["script_verdict"] = summary.get("verdict")
        rec["metrics"] = row_metrics(summary)
        v2_records.append(rec)
        write_json(row_out / f"{row_id}_DRIVER_COMPLETE.json", rec)

    combined_records = prior_records + v2_records
    assessment = assess(combined_records, baseline)
    verdict = assessment["verdict"]

    write_matrix_csv(out / "robustness_matrix.csv", combined_records)
    write_matrix_csv(out / "robustness_matrix_v2_only.csv", v2_records)
    summary_payload = {
        "verdict": verdict,
        "started_utc": started_utc,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline": baseline,
        "v1_evidence": v1_evidence,
        "assessment": assessment,
        "rows": combined_records,
        "v2_rows": v2_records,
        "notes": [
            "V2 preserves successful v1 N96 and dt/2 rows as frozen returned evidence.",
            "V2 runs epsilon rows before the repaired larger-box row so setup failure cannot block epsilon evidence.",
            "The repaired larger-box sentinel is L=20,N=96, replacing the failed v1 L=20,N=64 setup.",
            "Driver return code indicates infrastructure/archive success; science status is carried by the verdict.",
        ],
        "boundary": "Mirror-only robustness check; frozen TG-B1S and TG-B2 protected modules unchanged; not gravity, UFF, or IRER validation.",
    }
    write_json(out / "robustness_summary.json", summary_payload)
    write_json(
        out / "ROBUSTNESS_RUN_COMPLETE.json",
        {
            "verdict": verdict,
            "finished_utc": summary_payload["finished_utc"],
            "completed_rows": assessment["completed_rows"],
            "failed_rows": assessment["failed_rows"],
            "v2_rows_attempted": [row["row_id"] for row in v2_records],
        },
    )
    print(json.dumps({"verdict": verdict, "v2_rows_attempted": len(v2_records)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
