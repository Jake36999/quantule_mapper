"""Phase R TG-B2 robustness capsule driver.

Runs a compact Colab A100 matrix around the reviewed TG-B2 definitive force
baseline without modifying frozen scientific modules. Each row invokes
`jax_scout/gravity_TG_B2_definitive_force.py` as a subprocess, then this driver
aggregates sign/gate, convergence, and epsilon-scaling checks.
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
        "row_id": "r1_grid_N96_sep3",
        "phase": "R1_convergence",
        "reason": "grid refinement sentinel",
        "args": ["--seps", "3.0", "--N", "96", "--T", "180"],
        "compare_to_baseline_sep": 3.0,
        "drift_tolerance": 0.20,
    },
    {
        "row_id": "r1_dt_half_sep3",
        "phase": "R1_convergence",
        "reason": "timestep-halved sentinel",
        "args": ["--seps", "3.0", "--dt", "0.001", "--T", "180"],
        "compare_to_baseline_sep": 3.0,
        "drift_tolerance": 0.20,
    },
    {
        "row_id": "r1_box_L20_sep3",
        "phase": "R1_convergence",
        "reason": "larger-box sentinel",
        "args": ["--seps", "3.0", "--L", "20", "--T", "180"],
        "compare_to_baseline_sep": 3.0,
        "drift_tolerance": 0.20,
    },
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
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
    preds = [slope * x for x in xs]
    mean_y = sum(ys) / len(ys)
    sse = sum((y - p) ** 2 for y, p in zip(ys, preds))
    sst = sum((y - mean_y) ** 2 for y in ys)
    r2 = 1.0 if sst == 0.0 and sse == 0.0 else 1.0 - sse / (sst + 1e-30)
    return {"slope": slope, "r2": r2, "sse": sse}


def assess(rows: list[dict[str, Any]], baseline: dict[str, Any]) -> dict[str, Any]:
    baseline_rows = baseline_by_sep(baseline)
    baseline_sep3 = baseline_rows[3.0]
    baseline_force = float(baseline_sep3["F_R_well_avg"])
    baseline_eps = float(baseline["config"]["epsilon_G"])

    sign_gate_failures = []
    convergence_failures = []
    convergence_rows = []
    eps_points = [(baseline_eps, baseline_force)]

    for rec in rows:
        metrics = rec["metrics"]
        if not (
            metrics.get("gates_pass")
            and metrics.get("well_negative")
            and metrics.get("sign_reverses")
            and metrics.get("off_null_ok")
        ):
            sign_gate_failures.append(rec["row_id"])

        if rec["phase"] == "R1_convergence":
            drift = abs(float(metrics["F_R_well_avg"]) - baseline_force) / (abs(baseline_force) + 1e-30)
            rec["drift_vs_baseline_sep3"] = drift
            rec["drift_tolerance"] = float(rec["drift_tolerance"])
            rec["drift_pass"] = drift <= float(rec["drift_tolerance"])
            convergence_rows.append(rec)
            if not rec["drift_pass"]:
                convergence_failures.append(rec["row_id"])
        elif rec["phase"] == "R2_epsilon_scaling":
            eps_points.append((float(rec["epsilon_G"]), float(metrics["F_R_well_avg"])))

    eps_points = sorted(eps_points)
    fit = linear_fit_through_origin([p[0] for p in eps_points], [p[1] for p in eps_points])
    eps_signs_ok = all(y < 0.0 for _, y in eps_points)
    eps_r2_pass = fit["r2"] >= 0.99

    if sign_gate_failures:
        verdict = "TG_R_ROBUSTNESS_FAIL"
    elif convergence_failures or not eps_signs_ok or not eps_r2_pass:
        verdict = "TG_R_ROBUSTNESS_PARTIAL"
    else:
        verdict = "TG_R_ROBUSTNESS_PASS"

    return {
        "verdict": verdict,
        "baseline_sep3_F_R_well_avg": baseline_force,
        "sign_gate_failures": sign_gate_failures,
        "convergence_failures": convergence_failures,
        "epsilon_points": [{"epsilon_G": x, "F_R_well_avg": y} for x, y in eps_points],
        "epsilon_fit_through_origin": fit,
        "epsilon_signs_ok": eps_signs_ok,
        "epsilon_r2_pass": eps_r2_pass,
        "convergence_rows": [
            {
                "row_id": rec["row_id"],
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
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for rec in rows:
            metrics = rec.get("metrics", {})
            out = {key: rec.get(key, metrics.get(key, "")) for key in fields}
            writer.writerow(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline-summary", required=True)
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
    registry_path = project_root / args.protected_registry
    baseline = load_json(baseline_path)
    protected_records = verify_protected_files(project_root, registry_path)

    shutil.copy2(baseline_path, out / "baseline_reference.json")
    write_json(out / "protected_hash_verification.json", {"files": protected_records})
    write_json(
        out / "robustness_matrix_plan.json",
        {
            "started_utc": started_utc,
            "matrix": MATRIX,
            "baseline_summary": args.baseline_summary,
            "protected_registry": args.protected_registry,
            "entrypoint": args.entrypoint,
            "boundary": "Phase R robustness only; mirror TG-B2 A-well branch; frozen modules imported unchanged.",
        },
    )

    records: list[dict[str, Any]] = []
    for item in MATRIX:
        row_id = item["row_id"]
        row_out = rows_dir / row_id
        log_path = logs_dir / f"{row_id}.log"
        command = [sys.executable, "-u", args.entrypoint, *item["args"], "--out", str(row_out)]
        print(f"\n=== Phase R row {row_id}: {' '.join(command)} ===", flush=True)
        started = time.time()
        write_json(row_out / f"{row_id}_DRIVER_STARTED.json", {"row_id": row_id, "command": command})
        return_code = stream_subprocess(command, project_root, log_path)
        duration = time.time() - started
        rec: dict[str, Any] = {
            **item,
            "return_code": return_code,
            "duration_seconds": duration,
            "command": command,
            "log_path": str(log_path.relative_to(out)),
            "summary_path": str((row_out / "summary.json").relative_to(out)),
        }
        if return_code != 0:
            rec["status"] = "FAILED_SUBPROCESS"
            records.append(rec)
            write_json(row_out / f"{row_id}_DRIVER_FAILED.json", rec)
            break
        summary = load_json(row_out / "summary.json")
        rec["status"] = "COMPLETE"
        rec["script_verdict"] = summary.get("verdict")
        rec["metrics"] = row_metrics(summary)
        records.append(rec)
        write_json(row_out / f"{row_id}_DRIVER_COMPLETE.json", rec)

    if any(rec.get("return_code") != 0 for rec in records) or len(records) != len(MATRIX):
        verdict = "TG_R_ROBUSTNESS_INCOMPLETE"
        assessment = {
            "verdict": verdict,
            "completed_rows": [rec["row_id"] for rec in records if rec.get("status") == "COMPLETE"],
            "failed_rows": [rec["row_id"] for rec in records if rec.get("status") != "COMPLETE"],
        }
    else:
        assessment = assess(records, baseline)
        verdict = assessment["verdict"]

    write_matrix_csv(out / "robustness_matrix.csv", records)
    summary_payload = {
        "verdict": verdict,
        "started_utc": started_utc,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline": baseline,
        "assessment": assessment,
        "rows": records,
        "notes": [
            "R1 is a compact one-separation sentinel for N, dt and box robustness, compared to sep=3.0 baseline.",
            "R2 uses epsilon values 0.03, 0.06 baseline, and 0.12 with a through-origin linear fit.",
            "Auto-verdict is an infrastructure/science-check label; primary review remains required.",
        ],
        "boundary": "Mirror-only robustness check; frozen TG-B1S and TG-B2 protected modules unchanged; not gravity, UFF, or IRER validation.",
    }
    write_json(out / "robustness_summary.json", summary_payload)
    write_json(
        out / "ROBUSTNESS_RUN_COMPLETE.json",
        {
            "verdict": verdict,
            "finished_utc": summary_payload["finished_utc"],
            "completed_rows": [rec["row_id"] for rec in records if rec.get("status") == "COMPLETE"],
            "failed_rows": [rec["row_id"] for rec in records if rec.get("status") != "COMPLETE"],
        },
    )
    print(json.dumps({"verdict": verdict, "rows": len(records)}, indent=2), flush=True)
    return 0 if verdict != "TG_R_ROBUSTNESS_INCOMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
