"""Bounded Conservative C2 RK4 integrity diagnostic batch.

This is a diagnostic-only batch runner. It does not modify production solver
physics, production configs, Hunter, validation gates, or jax_scout references.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.conservative_geometry_campaign import (  # noqa: E402
    analyse_psi_geometry,
    centroid_drift_metrics,
    high_k_fraction,
    pairwise_distance_drift,
    profile_overlap,
    render_final_rho_orthogonal,
    steps_for_physical_time,
)
from tools.conservative_rk4_stepper_diagnostic import (  # noqa: E402
    ConservativeC2RK4Stepper,
    PROTECTED_FILES,
    geometry_psi0,
    run_safety_checkpoint,
)
from tools.conservative_stepper_contract_audit import fractional_rhs_flux, norm_conventions  # noqa: E402


FINAL_DECISIONS = {
    "RK4_DT_INTEGRITY_FAIL",
    "RK4_RHS_FLUX_FAIL",
    "RK4_INTEGRITY_PASS_N64_NOT_RUN",
    "RK4_INTEGRITY_PASS_N64_PROMISING",
    "RK4_INTEGRITY_UNCLEAR",
}
SECONDARY_FLAGS = [
    "no_stability_claim",
    "no_production_change",
    "no_amplitude_normalization",
    "no_jax",
    "long_campaign_not_run",
    "rk4_diagnostic_only",
    "budget_respected",
]


def timestamp_utc() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def resolve_output_dir(base_out: Path, timestamp: str | None = None, resume: bool = False) -> Path:
    base = Path(base_out)
    if resume:
        return base
    stamp = timestamp or timestamp_utc()
    return base.parent / f"{base.name}_{stamp}"


def expected_steps(physical_time: float, dt: float) -> int:
    return steps_for_physical_time(float(physical_time), float(dt))


def sample_steps_for_times(total_steps: int, dt: float, target_times: list[float]) -> list[int]:
    steps = {0, int(total_steps)}
    for t_phys in target_times:
        step = int(round(float(t_phys) / float(dt)))
        steps.add(max(0, min(int(total_steps), step)))
    return sorted(steps)


def rhs_flux_status(value: float) -> str:
    mag = abs(float(value))
    if mag <= 1e-10:
        return "numerical_zero"
    if mag > 1e-6:
        return "fail"
    if mag > 1e-8:
        return "warning"
    return "small_nonzero"


def should_run_n64_t2(remaining_seconds: float) -> bool:
    return float(remaining_seconds) >= 20.0 * 60.0


def build_manifest() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for dt in (0.004, 0.002, 0.001, 0.0005):
        cases.append(
            {
                "case_id": f"dt_integrity_T1_dt{dt:g}",
                "group": "dt_integrity_fixed_T",
                "label": "coarse_dt_sanity" if dt == 0.004 else "dt_integrity",
                "N": 48,
                "L": 10.0,
                "dt": dt,
                "physical_time": 1.0,
                "template": "triangle",
                "spacing": 0.45,
            }
        )
    for t_phys in (0.5, 1.0, 2.0):
        cases.append(
            {
                "case_id": f"dt_integrity_dt0.001_T{t_phys:g}",
                "group": "dt_integrity_fixed_dt",
                "label": "time_integrity",
                "N": 48,
                "L": 10.0,
                "dt": 0.001,
                "physical_time": t_phys,
                "template": "triangle",
                "spacing": 0.45,
            }
        )
    cases.append(
        {
            "case_id": "trajectory_rhs_flux_triangle_s0.45_N48_T1_dt0.001",
            "group": "trajectory_rhs_flux",
            "label": "rhs_flux_trajectory",
            "N": 48,
            "L": 10.0,
            "dt": 0.001,
            "physical_time": 1.0,
            "template": "triangle",
            "spacing": 0.45,
            "sample_times": [0.0, 0.25, 0.5, 0.75, 1.0],
        }
    )
    for t_phys in (1.0, 2.0):
        for template in ("triangle", "tetrahedron", "triangular_prism"):
            cases.append(
                {
                    "case_id": f"n64_{template}_s0.45_T{t_phys:g}",
                    "group": "n64_geometry_replay_optional",
                    "label": "n64_t1" if t_phys == 1.0 else "n64_t2_optional",
                    "N": 64,
                    "L": 10.0,
                    "dt": 0.001,
                    "physical_time": t_phys,
                    "template": template,
                    "spacing": 0.45,
                }
            )
    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "regime": "conservative_c2_rk4_integrity_diagnostic_only",
        "resume_default": False,
        "cases": cases,
    }


def stable_json_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def array_hash(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(arr))
    h = hashlib.sha256()
    h.update(str(a.dtype).encode("ascii"))
    h.update(json.dumps(a.shape).encode("ascii"))
    h.update(a.view(np.uint8))
    return h.hexdigest()


def history_hash(history: list[dict[str, Any]]) -> str:
    return stable_json_hash(history)


def validate_time_integrity(row: dict[str, Any], tol: float = 1e-12) -> str:
    if abs(float(row["requested_dt"]) - float(row["actual_dt"])) > tol:
        return "time/step integrity mismatch: dt"
    if int(row["expected_steps"]) != int(row["actual_steps"]):
        return "time/step integrity mismatch: steps"
    if abs(float(row["requested_physical_time"]) - float(row["actual_physical_time"])) > max(tol, abs(float(row["requested_physical_time"])) * 1e-12):
        return "time/step integrity mismatch: physical_time"
    return ""


def detect_suspicious_hash_reuse(rows: list[dict[str, Any]]) -> list[str]:
    flags: list[str] = []
    for i, a in enumerate(rows):
        for b in rows[i + 1 :]:
            same_hash = a.get("final_state_hash") == b.get("final_state_hash") and a.get("sampled_history_hash") == b.get("sampled_history_hash")
            same_request = (
                float(a.get("requested_dt", -1)) == float(b.get("requested_dt", -2))
                and float(a.get("requested_physical_time", -1)) == float(b.get("requested_physical_time", -2))
            )
            if same_hash and not same_request:
                flags.append(f"suspicious identical state/history hashes: {a.get('case_id')} vs {b.get('case_id')}")
    return flags


def classify_dt_integrity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [err for row in rows if (err := validate_time_integrity(row))]
    hash_flags = detect_suspicious_hash_reuse(rows)
    return {"passed": not errors and not hash_flags, "errors": errors, "hash_flags": hash_flags}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, rows: list[dict[str, Any]], y_key: str, ylabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    for group, items in _group_rows(rows, "group").items():
        items = sorted(items, key=lambda r: (float(r["requested_physical_time"]), float(r["requested_dt"])))
        xs = list(range(len(items)))
        ax.plot(xs, [float(r.get(y_key) or 0.0) for r in items], marker="o", label=group)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("case index")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _group_rows(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(str(row.get(key)), []).append(row)
    return out


class BatchContext:
    def __init__(self, out_dir: Path, max_wallclock_minutes: float):
        self.out_dir = Path(out_dir)
        self.start = time.time()
        self.max_seconds = float(max_wallclock_minutes) * 60.0
        self.log_path = self.out_dir / "rk4_integrity_batch.log"

    def elapsed(self) -> float:
        return time.time() - self.start

    def remaining(self) -> float:
        return self.max_seconds - self.elapsed()

    def log(self, text: str) -> None:
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {text}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def ensure_budget(self, estimated_seconds: float, case_id: str) -> bool:
        return self.remaining() > float(estimated_seconds)


def run_integrity_case(case: dict[str, Any], case_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import cupy as cp

    N = int(case["N"])
    L = float(case["L"])
    dt = float(case["dt"])
    physical_time = float(case["physical_time"])
    steps = expected_steps(physical_time, dt)
    psi0, points = geometry_psi0(str(case["template"]), float(case["spacing"]), N, L)
    stepper = ConservativeC2RK4Stepper(N, L, dt)
    psi_k, psi_initial = stepper.project_psi0(psi0)
    initial_cpu = cp.asnumpy(psi_initial)
    initial_norms = norm_conventions(initial_cpu, L=L)
    initial_rho = np.abs(initial_cpu) ** 2
    initial_geom = analyse_psi_geometry(initial_cpu, L=L, expected_nodes=len(points))
    sample_steps = sample_steps_for_times(steps, dt, [0.0, physical_time * 0.25, physical_time * 0.5, physical_time * 0.75, physical_time])
    history: list[dict[str, Any]] = []
    state_samples: list[dict[str, Any]] = []
    fail_reason = ""
    t0 = time.time()

    def sample(step: int, psi_k_current: Any) -> None:
        psi_now = stepper.to_physical(psi_k_current)
        psi_cpu = cp.asnumpy(psi_now)
        norms = norm_conventions(psi_cpu, L=L)
        rho = np.abs(psi_cpu) ** 2
        geom = analyse_psi_geometry(psi_cpu, L=L, expected_nodes=max(1, len(points)))
        drift = centroid_drift_metrics(initial_geom, geom, N=N, t_phys=step * dt)
        pair_drift = pairwise_distance_drift(initial_geom, geom)
        rec = {
            "case_id": case["case_id"],
            "step": int(step),
            "t_phys": float(step * dt),
            "diagnostic_norm_fractional_change": float((norms["diagnostic_norm"] - initial_norms["diagnostic_norm"]) / (abs(initial_norms["diagnostic_norm"]) + 1e-30)),
            "physical_grid_norm_fractional_change": float((norms["physical_grid_norm"] - initial_norms["physical_grid_norm"]) / (abs(initial_norms["physical_grid_norm"]) + 1e-30)),
            "rho_max_fractional_change": float((float(np.max(rho)) - float(np.max(initial_rho))) / (abs(float(np.max(initial_rho))) + 1e-30)),
            "profile_overlap": profile_overlap(initial_rho, rho),
            "node_count": geom.get("node_count"),
            "threshold_node_counts": geom.get("threshold_node_counts"),
            "pairwise_distance_drift_mean_box": pair_drift.get("pairwise_distance_drift_mean_box"),
            "centroid_drift_mean_box": drift.get("centroid_drift_mean_box"),
            "high_k_fraction": high_k_fraction(cp.asnumpy(psi_k_current), cp.asnumpy(stepper.solver.reference_dealias_mask).astype(bool)),
        }
        history.append(rec)
        state_samples.append({"step": int(step), "state_hash": array_hash(psi_cpu)})

    sample(0, psi_k)
    sample_set = set(sample_steps)
    for step in range(1, steps + 1):
        psi_k = stepper.step(psi_k)
        if step in sample_set:
            sample(step, psi_k)
        if time.time() - t0 > 600:
            fail_reason = "case runtime guard exceeded 600s"
            break
    if history[-1]["step"] != steps:
        sample(steps, psi_k)
    psi_final = cp.asnumpy(stepper.to_physical(psi_k))
    final = history[-1]
    metadata = {
        "case": case,
        "requested_dt": dt,
        "actual_dt": dt,
        "requested_physical_time": physical_time,
        "actual_physical_time": float(final["t_phys"]),
        "expected_steps": int(steps),
        "actual_steps": int(final["step"]),
        "history_count": len(history),
        "fail_reason": fail_reason,
    }
    row = {
        "case_id": case["case_id"],
        "group": case["group"],
        "label": case.get("label", ""),
        "requested_dt": dt,
        "actual_dt": dt,
        "requested_physical_time": physical_time,
        "actual_physical_time": float(final["t_phys"]),
        "expected_steps": int(steps),
        "actual_steps": int(final["step"]),
        "first_saved_t": float(history[0]["t_phys"]),
        "final_saved_t": float(final["t_phys"]),
        "diagnostic_norm_fractional_change": final["diagnostic_norm_fractional_change"],
        "physical_grid_norm_fractional_change": final["physical_grid_norm_fractional_change"],
        "rho_max_fractional_change": final["rho_max_fractional_change"],
        "profile_overlap": final["profile_overlap"],
        "final_node_count": final["node_count"],
        "threshold_node_counts_final": json.dumps(final["threshold_node_counts"], sort_keys=True),
        "pairwise_distance_drift_mean_box": final["pairwise_distance_drift_mean_box"],
        "centroid_drift_mean_box": final["centroid_drift_mean_box"],
        "high_k_fraction": final["high_k_fraction"],
        "initial_state_hash": array_hash(initial_cpu),
        "final_state_hash": array_hash(psi_final),
        "sampled_history_hash": history_hash(history),
        "metadata_hash": stable_json_hash(metadata),
        "finite": bool(np.isfinite(psi_final).all() and not fail_reason),
        "fail_reason": fail_reason,
        "wallclock_sec": round(time.time() - t0, 3),
    }
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "metadata.json").write_text(json.dumps({**metadata, "row": row}, indent=2), encoding="utf-8")
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return row, history


def run_rhs_flux_case(case: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    import cupy as cp

    N = int(case["N"])
    L = float(case["L"])
    dt = float(case["dt"])
    steps = expected_steps(float(case["physical_time"]), dt)
    sample_steps = sample_steps_for_times(steps, dt, list(case["sample_times"]))
    psi0, points = geometry_psi0(str(case["template"]), float(case["spacing"]), N, L)
    stepper = ConservativeC2RK4Stepper(N, L, dt)
    psi_k, psi_initial = stepper.project_psi0(psi0)
    initial_cpu = cp.asnumpy(psi_initial)
    initial_rho = np.abs(initial_cpu) ** 2
    rows: list[dict[str, Any]] = []

    def flux_row(step: int, psi_k_current: Any) -> None:
        psi_phys = stepper.to_physical(psi_k_current)
        rhs_phys = stepper.solver.ifft_single(stepper.solver.N_op(psi_k_current))
        psi_cpu = cp.asnumpy(psi_phys)
        rhs_cpu = cp.asnumpy(rhs_phys)
        flux = fractional_rhs_flux(psi_cpu, rhs_cpu, L=L)
        rho = np.abs(psi_cpu) ** 2
        geom = analyse_psi_geometry(psi_cpu, L=L, expected_nodes=len(points))
        rows.append(
            {
                "case_id": case["case_id"],
                "step": int(step),
                "t_phys": float(step * dt),
                "diagnostic_norm": flux["diagnostic_norm"],
                "physical_grid_norm": flux["physical_grid_norm"],
                "d_norm_dt_raw": flux["d_norm_dt_raw"],
                "fractional_flux_raw": flux["fractional_flux_raw"],
                "flux_status": rhs_flux_status(flux["fractional_flux_raw"]),
                "rho_max": float(np.max(rho)),
                "node_count": geom.get("node_count"),
                "profile_overlap": profile_overlap(initial_rho, rho),
                "high_k_fraction": high_k_fraction(cp.asnumpy(psi_k_current), cp.asnumpy(stepper.solver.reference_dealias_mask).astype(bool)),
            }
        )

    sample_set = set(sample_steps)
    flux_row(0, psi_k)
    for step in range(1, steps + 1):
        psi_k = stepper.step(psi_k)
        if step in sample_set:
            flux_row(step, psi_k)
    if rows[-1]["step"] != steps:
        flux_row(steps, psi_k)
    failed = any(row["flux_status"] == "fail" for row in rows)
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return rows, failed


def run_n64_case(case: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    row, _ = run_geometry_case(
        out_dir,
        str(case["template"]),
        float(case["spacing"]),
        int(case["N"]),
        float(case["L"]),
        float(case["dt"]),
        float(case["physical_time"]),
    )
    return row


def write_dt_report(path: Path, rows: list[dict[str, Any]], result: dict[str, Any]) -> None:
    lines = [
        "# RK4 DT Integrity Report",
        "",
        f"Passed: `{result['passed']}`",
        f"Errors: `{result['errors']}`",
        f"Hash flags: `{result['hash_flags']}`",
        "",
        "| case | dt | T | steps | final t | norm change | final hash | history hash |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['actual_dt']} | {row['actual_physical_time']} | {row['actual_steps']} | "
            f"{row['final_saved_t']} | {row['diagnostic_norm_fractional_change']} | {row['final_state_hash'][:10]} | {row['sampled_history_hash'][:10]} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_rhs_report(path: Path, rows: list[dict[str, Any]], failed: bool) -> None:
    lines = [
        "# RK4 Trajectory RHS Flux Report",
        "",
        f"Failed: `{failed}`",
        "",
        "| t | fractional flux | status | norm | rho_max | nodes |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['t_phys']} | {row['fractional_flux_raw']} | {row['flux_status']} | {row['diagnostic_norm']} | {row['rho_max']} | {row['node_count']} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_n64_report(path: Path, rows: list[dict[str, Any]], skipped_t2: bool) -> None:
    lines = [
        "# RK4 N64 Geometry Replay Report",
        "",
        f"N64/T2 skipped due to budget: `{skipped_t2}`",
        "",
        "| case | nodes | norm change | rho max change | finite |",
        "|---|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row.get('initial_node_count')} -> {row.get('final_node_count')} | "
            f"{row.get('diagnostic_norm_fractional_change')} | {row.get('rho_max_fractional_change')} | {row.get('finite')} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_final_report(path: Path, *, final_decision: str, flags: list[str], wallclock: float, dt_result: dict[str, Any], rhs_failed: bool, n64_rows: list[dict[str, Any]], n64_skipped: bool, safety: dict[str, Any]) -> None:
    lines = [
        "# RK4 Integrity Diagnostic Final Report",
        "",
        f"Final decision: `{final_decision}`",
        f"Secondary flags: `{', '.join(flags)}`",
        f"Total wallclock seconds: `{wallclock:.3f}`",
        "",
        "## Environment",
        "",
        f"- Python: `{safety.get('python_executable')}`",
        f"- CuPy: `{safety.get('cupy_version')}`",
        f"- GPU: `{safety.get('gpu', {}).get('gpu_name')}`",
        f"- JAX/JAXLIB absent: `{safety.get('jax_spec') == 'None' and safety.get('jaxlib_spec') == 'None'}`",
        f"- Protected diff empty: `{safety.get('protected_diff_empty')}`",
        "",
        "## Results",
        "",
        f"- DT integrity passed: `{dt_result['passed']}`",
        f"- RHS flux failed: `{rhs_failed}`",
        f"- N64 replay rows: `{len(n64_rows)}`",
        f"- N64 optional T2 skipped due to budget: `{n64_skipped}`",
        "",
        "No stability claim is made. RK4 remains diagnostic-only. No longer campaign was run.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = resolve_output_dir(Path(args.out), resume=bool(args.resume))
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = BatchContext(out_dir, float(args.max_wallclock_minutes))
    manifest = build_manifest()
    (out_dir / "rk4_integrity_batch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    safety = run_safety_checkpoint(out_dir, args.require_gpu_name)
    resolved = {"manifest": manifest, "case_status": []}
    final_decision = "RK4_INTEGRITY_UNCLEAR"

    ctx.log("starting dt integrity cases")
    dt_rows: list[dict[str, Any]] = []
    for case in [c for c in manifest["cases"] if c["group"].startswith("dt_integrity")]:
        if not ctx.ensure_budget(30.0, case["case_id"]):
            final_decision = "RK4_INTEGRITY_UNCLEAR"
            ctx.log(f"budget stop before required case {case['case_id']}")
            break
        row, _ = run_integrity_case(case, out_dir / "cases" / case["case_id"])
        dt_rows.append(row)
        resolved["case_status"].append({"case_id": case["case_id"], "status": "complete", "wallclock_sec": row["wallclock_sec"]})
        ctx.log(f"completed {case['case_id']} wallclock={row['wallclock_sec']}")
        if not row["finite"]:
            final_decision = "RK4_INTEGRITY_UNCLEAR"
            break
    dt_result = classify_dt_integrity(dt_rows)
    write_csv(out_dir / "dt_integrity_results.csv", dt_rows)
    write_dt_report(out_dir / "dt_integrity_report.md", dt_rows, dt_result)
    write_plot(out_dir / "dt_integrity_norm_vs_time.png", dt_rows, "diagnostic_norm_fractional_change", "diagnostic norm fractional change")
    write_plot(out_dir / "dt_integrity_rho_vs_time.png", dt_rows, "rho_max_fractional_change", "rho_max fractional change")
    if not dt_result["passed"]:
        final_decision = "RK4_DT_INTEGRITY_FAIL"

    rhs_rows: list[dict[str, Any]] = []
    rhs_failed = False
    if final_decision != "RK4_DT_INTEGRITY_FAIL":
        ctx.log("starting trajectory RHS flux audit")
        rhs_case = next(c for c in manifest["cases"] if c["group"] == "trajectory_rhs_flux")
        rhs_rows, rhs_failed = run_rhs_flux_case(rhs_case)
        write_csv(out_dir / "trajectory_rhs_flux_results.csv", rhs_rows)
        write_rhs_report(out_dir / "trajectory_rhs_flux_report.md", rhs_rows, rhs_failed)
        resolved["case_status"].append({"case_id": rhs_case["case_id"], "status": "complete", "failed": rhs_failed})
        if rhs_failed:
            final_decision = "RK4_RHS_FLUX_FAIL"

    n64_rows: list[dict[str, Any]] = []
    n64_skipped_t2 = False
    if final_decision not in ("RK4_DT_INTEGRITY_FAIL", "RK4_RHS_FLUX_FAIL"):
        ctx.log("starting optional N64 replay")
        for case in [c for c in manifest["cases"] if c["group"] == "n64_geometry_replay_optional"]:
            if case["physical_time"] == 2.0 and not should_run_n64_t2(ctx.remaining()):
                n64_skipped_t2 = True
                resolved["case_status"].append({"case_id": case["case_id"], "status": "skipped_budget_t2"})
                continue
            estimate = 180.0 if case["physical_time"] == 1.0 else 300.0
            if not ctx.ensure_budget(estimate, case["case_id"]):
                resolved["case_status"].append({"case_id": case["case_id"], "status": "skipped_budget"})
                continue
            row = run_n64_case(case, out_dir)
            n64_rows.append(row)
            resolved["case_status"].append({"case_id": case["case_id"], "status": "complete", "wallclock_sec": row.get("wallclock_sec")})
            ctx.log(f"completed {case['case_id']} wallclock={row.get('wallclock_sec')}")
            if not row.get("finite"):
                break
        write_csv(out_dir / "n64_geometry_replay_results.csv", n64_rows)
        write_n64_report(out_dir / "n64_geometry_replay_report.md", n64_rows, n64_skipped_t2)
        if n64_rows and all(r.get("finite") and int(r.get("final_node_count") or -1) == int(r.get("initial_node_count") or -2) for r in n64_rows):
            final_decision = "RK4_INTEGRITY_PASS_N64_PROMISING"
        else:
            final_decision = "RK4_INTEGRITY_PASS_N64_NOT_RUN"

    resolved["final_decision"] = final_decision
    resolved["total_wallclock_sec"] = ctx.elapsed()
    (out_dir / "rk4_integrity_batch_resolved_manifest.json").write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    write_final_report(
        out_dir / "rk4_integrity_diagnostic_final_report.md",
        final_decision=final_decision,
        flags=SECONDARY_FLAGS,
        wallclock=ctx.elapsed(),
        dt_result=dt_result,
        rhs_failed=rhs_failed,
        n64_rows=n64_rows,
        n64_skipped=n64_skipped_t2,
        safety=safety,
    )
    return {"final_decision": final_decision, "out_dir": str(out_dir), "wallclock_sec": ctx.elapsed()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bounded Conservative C2 RK4 integrity diagnostic batch")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-wallclock-minutes", type=float, default=60.0)
    parser.add_argument("--require-gpu-name", default="NVIDIA GeForce GTX 1080")
    parser.add_argument("--resume", action="store_true", help="Explicit opt-in resume mode; disabled by default")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_batch(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
