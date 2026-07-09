"""Diagnostic-only Conservative C2 full-RHS RK4 stepper.

This module is not a production solver replacement. It reuses the already
verified Conservative C2 CuPy diagnostic wrapper and compares direct full-RHS
RK4 against the current ETDRK4 diagnostic path.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.conservative_geometry_campaign import (  # noqa: E402
    CONSERVATIVE_PARAMS,
    DEFAULT_L,
    DEFAULT_OUT,
    DEFAULT_SEED,
    DEFAULT_WIDTH_BOX,
    ConservativeCupySolver,
    _find_native_c2_profile,
    _gpu_info,
    _load_native_profile,
    analyse_psi_geometry,
    build_geometry_ic,
    centroid_drift_metrics,
    high_k_fraction,
    late_trends,
    pairwise_distance_drift,
    points_for_case,
    profile_overlap,
    render_final_rho_orthogonal,
    steps_for_physical_time,
    write_centroid_plotly_html,
)
from tools.conservative_stepper_contract_audit import (  # noqa: E402
    classify_rhs_flux,
    fractional_rhs_flux,
    norm_conventions,
    run_rhs_flux_probe,
    rhs_flux_probe_specs,
)


DEFAULT_RK4_OUT = DEFAULT_OUT / "rk4_stepper_diagnostic"
RUNTIME_CASE_WARN_SEC = 600.0
PROTECTED_FILES = [
    "solver/core.py",
    "solver/run.py",
    "worker_cupy.py",
    "aste_hunter.py",
    "validation_pipeline.py",
    "config_utils.py",
    "tools/production_h7_revalidation.py",
    "jax_scout/physics.py",
    "jax_scout/phase_d_c2_transport.py",
    "jax_scout/phase_d_c2_soliton_scout.py",
    "jax_scout/phase_d_c2_2_loss_source.py",
]
FINAL_DECISIONS = {
    "RK4_STEPPER_REJECTED",
    "RK4_STEPPER_DIAGNOSTIC_PROMISING",
    "RK4_GEOMETRY_REPLAY_PROMISING",
    "GEOMETRY_DEPENDED_ON_LOSSY_ETDRK4",
    "CONSERVATIVE_STEPPER_STILL_BLOCKED",
}


def rk4_diagnostic_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "safety_checkpoint": out_dir / "safety_checkpoint.md",
        "protected_diff_before": out_dir / "protected_diff_before.txt",
        "one_step_csv": out_dir / "one_step_rk4_vs_etdrk4.csv",
        "one_step_report": out_dir / "one_step_rk4_vs_etdrk4.md",
        "one_step_norm_plot": out_dir / "one_step_rk4_vs_etdrk4_norm_defect_vs_dt.png",
        "one_step_rho_plot": out_dir / "one_step_rk4_vs_etdrk4_rho_max_defect_vs_dt.png",
        "multistep_csv": out_dir / "multistep_rk4_vs_etdrk4_results.csv",
        "multistep_history": out_dir / "multistep_rk4_vs_etdrk4_history.json",
        "multistep_norm_plot": out_dir / "multistep_norm_vs_time.png",
        "multistep_rho_plot": out_dir / "multistep_rho_max_vs_time.png",
        "multistep_overlap_plot": out_dir / "multistep_profile_overlap_vs_time.png",
        "multistep_report": out_dir / "multistep_rk4_vs_etdrk4_report.md",
        "geometry_csv": out_dir / "rk4_geometry_replay_results.csv",
        "geometry_history": out_dir / "rk4_geometry_replay_history.json",
        "geometry_report": out_dir / "rk4_geometry_replay_report.md",
        "final_report": out_dir / "rk4_stepper_diagnostic_final_report.md",
        "native_profile_report": out_dir / "native_c2_profile_rk4_check.md",
    }


def parse_float_list(text: str) -> list[float]:
    vals = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if not vals:
        raise ValueError("expected at least one numeric value")
    return vals


def run_cmd(args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)
    return proc.returncode, (proc.stdout + proc.stderr)


def protected_diff_text() -> str:
    _, text = run_cmd(["git", "diff", "--", *PROTECTED_FILES])
    return text


def run_safety_checkpoint(out_dir: Path, require_gpu_name: str | None) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = rk4_diagnostic_paths(out_dir)
    status_code, status_text = run_cmd(["git", "status", "--short"])
    diff_text = protected_diff_text()
    paths["protected_diff_before"].write_text(diff_text, encoding="utf-8")

    import cupy as cp

    gpu = _gpu_info(require_gpu_name)
    device_count = int(cp.cuda.runtime.getDeviceCount())
    jax_spec = importlib.util.find_spec("jax")
    jaxlib_spec = importlib.util.find_spec("jaxlib")
    checkpoint = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_executable": sys.executable,
        "git_status_exit_code": status_code,
        "git_status_short": status_text,
        "protected_diff_empty": diff_text.strip() == "",
        "jax_spec": str(jax_spec),
        "jaxlib_spec": str(jaxlib_spec),
        "cupy_version": cp.__version__,
        "device_count": device_count,
        "gpu": gpu,
    }
    lines = [
        "# RK4 Stepper Diagnostic Safety Checkpoint",
        "",
        f"Created UTC: `{checkpoint['created_utc']}`",
        f"Python: `{sys.executable}`",
        f"CuPy: `{cp.__version__}`",
        f"CUDA device count: `{device_count}`",
        f"GPU: `{gpu['gpu_name']}`",
        f"JAX spec: `{jax_spec}`",
        f"JAXLIB spec: `{jaxlib_spec}`",
        f"Protected diff empty: `{checkpoint['protected_diff_empty']}`",
        "",
        "## Git Status",
        "```",
        status_text.rstrip(),
        "```",
        "",
        "Protected diff is written to `protected_diff_before.txt`.",
    ]
    paths["safety_checkpoint"].write_text("\n".join(lines), encoding="utf-8")
    if diff_text.strip():
        raise RuntimeError("protected production/reference files have diffs; stopping")
    if device_count != 1:
        raise RuntimeError(f"expected exactly 1 CUDA device, got {device_count}")
    if jax_spec is not None or jaxlib_spec is not None:
        raise RuntimeError("jax/jaxlib are present in active runtime; stopping")
    return checkpoint


class ConservativeC2RK4Stepper:
    """Diagnostic full-RHS RK4 wrapper using the Conservative C2 CuPy RHS."""

    def __init__(self, N: int, L: float, dt: float, nonlinear_enabled: bool = True):
        self.N = int(N)
        self.L = float(L)
        self.dt = float(dt)
        self.solver = ConservativeCupySolver(
            self.N,
            self.L,
            self.dt,
            dict(CONSERVATIVE_PARAMS),
            nonlinear_enabled=nonlinear_enabled,
            mask_mode="current",
        )

    def project_psi0(self, psi0: np.ndarray):
        import cupy as cp

        psi = cp.asarray(psi0, dtype=cp.complex128)
        psi_k = self.solver.fft_single(psi) * self.solver.dealias_mask
        return psi_k.astype(cp.complex128, copy=False), self.solver.ifft_single(psi_k).astype(cp.complex128, copy=False)

    def to_physical(self, psi_k: Any):
        return self.solver.ifft_single(psi_k).astype(self.solver.batch_real.dtype, copy=False)

    def rhs(self, psi_k: Any):
        return self.solver.L_k * psi_k + self.solver.N_op(psi_k)

    def step(self, psi_k: Any):
        return rk4_full_rhs_step(psi_k, self.dt, self.rhs)


def rk4_full_rhs_step(psi_k: Any, dt: float, rhs_func):
    k1 = rhs_func(psi_k)
    k2 = rhs_func(psi_k + 0.5 * dt * k1)
    k3 = rhs_func(psi_k + 0.5 * dt * k2)
    k4 = rhs_func(psi_k + dt * k3)
    return psi_k + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def triangle_psi0(N: int, L: float, spacing: float) -> np.ndarray:
    pts = points_for_case("triangle", spacing, {}, seed=DEFAULT_SEED)
    return build_geometry_ic(N=N, L=L, points_box=pts, width_box=DEFAULT_WIDTH_BOX, amplitude=1.0, phases=[0.0, 0.0, 0.0])


def geometry_psi0(template: str, spacing: float, N: int, L: float) -> tuple[np.ndarray, np.ndarray]:
    pts = points_for_case(template, spacing, {}, seed=DEFAULT_SEED)
    psi0 = build_geometry_ic(N=N, L=L, points_box=pts, width_box=DEFAULT_WIDTH_BOX, amplitude=1.0, phases=[0.0] * len(pts))
    return psi0, pts


def rhs_flux_controls_are_zero(N: int, L: float, dt: float, spacing: float) -> dict[str, str]:
    out: dict[str, str] = {}
    for spec in rhs_flux_probe_specs(spacing):
        if spec["case_id"] not in ("uniform_constant_control", f"triangle_spacing_{spacing:g}"):
            continue
        row = run_rhs_flux_probe(spec, N=N, L=L, dt=dt, spacing=spacing)
        out[str(row["case_id"])] = str(row["rhs_flux_status"])
    return out


def _row_common(case_id: str, group: str, mode: str, N: int, L: float, dt: float, steps: int) -> dict[str, Any]:
    return {"case_id": case_id, "group": group, "mode": mode, "N": int(N), "L": float(L), "dt": float(dt), "steps": int(steps)}


def run_one_step_mode(mode: str, N: int, L: float, dt: float, spacing: float) -> dict[str, Any]:
    import cupy as cp

    psi0 = triangle_psi0(N, L, spacing)
    initial_geom = analyse_psi_geometry(psi0, L=L, expected_nodes=3)
    t0 = time.time()
    if mode == "etdrk4_current":
        solver = ConservativeCupySolver(N, L, dt, dict(CONSERVATIVE_PARAMS))
        psi = cp.asarray(psi0, dtype=cp.complex128)
        psi_k = solver.fft_single(psi) * solver.dealias_mask
        psi_initial = solver.ifft_single(psi_k)
        psi_next_k = solver.step(psi_k)
        psi_next = solver.ifft_single(psi_next_k)
        mask = cp.asnumpy(getattr(solver, "reference_dealias_mask", solver.dealias_mask)).astype(bool)
    elif mode == "rk4_full_rhs":
        stepper = ConservativeC2RK4Stepper(N, L, dt)
        psi_k, psi_initial = stepper.project_psi0(psi0)
        psi_next_k = stepper.step(psi_k)
        psi_next = stepper.to_physical(psi_next_k)
        mask = cp.asnumpy(stepper.solver.reference_dealias_mask).astype(bool)
    else:
        raise ValueError(f"unsupported one-step mode {mode!r}")
    initial_cpu = cp.asnumpy(psi_initial)
    final_cpu = cp.asnumpy(psi_next)
    initial_norms = norm_conventions(initial_cpu, L=L)
    final_norms = norm_conventions(final_cpu, L=L)
    initial_rho = np.abs(initial_cpu) ** 2
    final_rho = np.abs(final_cpu) ** 2
    final_geom = analyse_psi_geometry(final_cpu, L=L, expected_nodes=3)
    defect = (final_norms["diagnostic_norm"] - initial_norms["diagnostic_norm"]) / (abs(initial_norms["diagnostic_norm"]) + 1e-30)
    physical_defect = (final_norms["physical_grid_norm"] - initial_norms["physical_grid_norm"]) / (abs(initial_norms["physical_grid_norm"]) + 1e-30)
    row = {
        **_row_common(f"one_step_{mode}_dt_{dt:g}", "one_step", mode, N, L, dt, 1),
        "finite": bool(np.isfinite(final_cpu).all()),
        "fail_reason": "",
        "initial_diagnostic_norm": initial_norms["diagnostic_norm"],
        "diagnostic_norm": final_norms["diagnostic_norm"],
        "initial_physical_grid_norm": initial_norms["physical_grid_norm"],
        "physical_grid_norm": final_norms["physical_grid_norm"],
        "fractional_norm_defect": float(defect),
        "fractional_physical_norm_defect": float(physical_defect),
        "rho_max": float(np.max(final_rho)),
        "rho_max_fractional_change": float((float(np.max(final_rho)) - float(np.max(initial_rho))) / (abs(float(np.max(initial_rho))) + 1e-30)),
        "profile_overlap": profile_overlap(initial_rho, final_rho),
        "initial_node_count": initial_geom.get("node_count"),
        "final_node_count": final_geom.get("node_count"),
        "threshold_node_counts_final": json.dumps(final_geom.get("threshold_node_counts", {})),
        "high_k_fraction": high_k_fraction(cp.asnumpy(psi_next_k), mask),
        "wallclock_sec": round(time.time() - t0, 3),
    }
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return row


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


def plot_rows(path: Path, rows: list[dict[str, Any]], x_key: str, y_key: str, title: str, ylabel: str, loglog: bool = False) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(str(row.get("mode")), []).append(row)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=120)
    for mode, items in by_mode.items():
        items = sorted(items, key=lambda r: float(r[x_key]))
        xs = [float(r[x_key]) for r in items]
        ys = [abs(float(r[y_key])) if loglog else float(r[y_key]) for r in items]
        if loglog:
            ax.loglog(xs, ys, marker="o", label=mode)
        else:
            ax.plot(xs, ys, marker="o", label=mode)
    ax.set_title(title)
    ax.set_xlabel(x_key)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_history(path: Path, history: list[dict[str, Any]], y_key: str, ylabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_case: dict[str, list[dict[str, Any]]] = {}
    for item in history:
        by_case.setdefault(str(item["case_id"]), []).append(item)
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=120)
    for case_id, items in by_case.items():
        items = sorted(items, key=lambda r: float(r["t_phys"]))
        ax.plot([r["t_phys"] for r in items], [r.get(y_key, np.nan) for r in items], label=case_id, linewidth=1)
    ax.set_xlabel("physical time")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if by_case:
        ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run_one_step_comparison(out_dir: Path, N: int, L: float, spacing: float, dt_values: list[float]) -> tuple[list[dict[str, Any]], bool]:
    paths = rk4_diagnostic_paths(out_dir)
    rows: list[dict[str, Any]] = []
    for dt in dt_values:
        rows.append(run_one_step_mode("etdrk4_current", N, L, dt, spacing))
        rows.append(run_one_step_mode("rk4_full_rhs", N, L, dt, spacing))
    write_csv(paths["one_step_csv"], rows)
    plot_rows(paths["one_step_norm_plot"], rows, "dt", "fractional_norm_defect", "One-step norm defect", "|fractional norm defect|", loglog=True)
    plot_rows(paths["one_step_rho_plot"], rows, "dt", "rho_max_fractional_change", "One-step rho_max change", "rho_max fractional change")
    by_dt = {}
    for row in rows:
        by_dt.setdefault(float(row["dt"]), {})[row["mode"]] = abs(float(row["fractional_norm_defect"]))
    better_each = all(by_dt[dt]["rk4_full_rhs"] < by_dt[dt]["etdrk4_current"] for dt in by_dt)
    et_vals = [v["etdrk4_current"] for v in by_dt.values()]
    rk_vals = [v["rk4_full_rhs"] for v in by_dt.values()]
    ratio = float(np.median(et_vals) / (np.median(rk_vals) + 1e-30))
    proceed = bool(better_each and ratio >= 4.0)
    lines = [
        "# One-step RK4 vs ETDRK4",
        "",
        f"RK4 better at every dt: `{better_each}`",
        f"Median ETDRK4/RK4 defect ratio: `{ratio}`",
        f"Proceed to multistep: `{proceed}`",
        "",
        "| dt | ETDRK4 defect | RK4 defect |",
        "|---:|---:|---:|",
    ]
    for dt in sorted(by_dt):
        lines.append(f"| {dt} | {by_dt[dt]['etdrk4_current']:.12g} | {by_dt[dt]['rk4_full_rhs']:.12g} |")
    paths["one_step_report"].write_text("\n".join(lines), encoding="utf-8")
    return rows, proceed


def multistep_pass_threshold(t_phys: float, loss: float) -> bool:
    mag = abs(float(loss))
    if t_phys <= 0.5:
        return mag <= 0.001
    if t_phys <= 1.0:
        return mag <= 0.0025
    if t_phys <= 2.0:
        return mag <= 0.005
    if t_phys <= 4.0:
        return mag <= 0.02
    return False


def run_multistep_mode(mode: str, N: int, L: float, dt: float, spacing: float, physical_time: float, sample_count: int = 40) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import cupy as cp

    psi0 = triangle_psi0(N, L, spacing)
    steps = steps_for_physical_time(physical_time, dt)
    sample_every = max(1, steps // max(1, sample_count))
    if mode == "rk4_full_rhs":
        stepper = ConservativeC2RK4Stepper(N, L, dt)
        psi_k, psi_initial = stepper.project_psi0(psi0)
        solver = stepper.solver
        do_step = stepper.step
        to_phys = stepper.to_physical
    elif mode in ("etdrk4_current", "etdrk4_no_mask"):
        solver = ConservativeCupySolver(N, L, dt, dict(CONSERVATIVE_PARAMS), mask_mode="none" if mode == "etdrk4_no_mask" else "current")
        psi = cp.asarray(psi0, dtype=cp.complex128)
        psi_k = solver.fft_single(psi) * solver.dealias_mask
        psi_initial = solver.ifft_single(psi_k)
        do_step = solver.step
        to_phys = solver.ifft_single
    else:
        raise ValueError(f"unsupported mode {mode!r}")
    initial_cpu = cp.asnumpy(psi_initial)
    initial_norms = norm_conventions(initial_cpu, L=L)
    initial_rho = np.abs(initial_cpu) ** 2
    initial_geom = analyse_psi_geometry(initial_cpu, L=L, expected_nodes=3)
    history: list[dict[str, Any]] = []
    fail_reason = ""
    t0 = time.time()
    final_step = 0
    for step in range(steps):
        psi_k = do_step(psi_k)
        final_step = step + 1
        if step % sample_every == 0 or step == steps - 1:
            psi_now = to_phys(psi_k)
            psi_cpu = cp.asnumpy(psi_now)
            if not np.isfinite(psi_cpu).all():
                fail_reason = f"nonfinite at step {step + 1}"
                break
            norms = norm_conventions(psi_cpu, L=L)
            rho = np.abs(psi_cpu) ** 2
            geom = analyse_psi_geometry(psi_cpu, L=L, expected_nodes=3)
            drift = centroid_drift_metrics(initial_geom, geom, N=N, t_phys=(step + 1) * dt)
            pair_drift = pairwise_distance_drift(initial_geom, geom)
            history.append(
                {
                    "case_id": f"{mode}_dt_{dt:g}_T_{physical_time:g}",
                    "group": "multistep",
                    "mode": mode,
                    "step": int(step + 1),
                    "t_phys": float((step + 1) * dt),
                    "dt": float(dt),
                    "diagnostic_norm": norms["diagnostic_norm"],
                    "physical_grid_norm": norms["physical_grid_norm"],
                    "diagnostic_norm_fractional_change": float((norms["diagnostic_norm"] - initial_norms["diagnostic_norm"]) / (abs(initial_norms["diagnostic_norm"]) + 1e-30)),
                    "physical_grid_norm_fractional_change": float((norms["physical_grid_norm"] - initial_norms["physical_grid_norm"]) / (abs(initial_norms["physical_grid_norm"]) + 1e-30)),
                    "rho_max": float(np.max(rho)),
                    "rho_max_fractional_change": float((float(np.max(rho)) - float(np.max(initial_rho))) / (abs(float(np.max(initial_rho))) + 1e-30)),
                    "profile_overlap": profile_overlap(initial_rho, rho),
                    "node_count": geom.get("node_count"),
                    "threshold_node_counts": geom.get("threshold_node_counts"),
                    "node_width_mean_box": geom.get("node_width_mean_box"),
                    "node_width_cv": geom.get("node_width_cv"),
                    "high_k_fraction": high_k_fraction(cp.asnumpy(psi_k), cp.asnumpy(getattr(solver, "reference_dealias_mask", solver.dealias_mask)).astype(bool)),
                    **drift,
                    **pair_drift,
                }
            )
        if time.time() - t0 > RUNTIME_CASE_WARN_SEC:
            fail_reason = f"runtime guard exceeded {RUNTIME_CASE_WARN_SEC:g}s"
            break
    final = history[-1] if history else {}
    row = {
        **_row_common(f"{mode}_dt_{dt:g}_T_{physical_time:g}", "multistep", mode, N, L, dt, final_step),
        "requested_steps": int(steps),
        "finite": not fail_reason,
        "fail_reason": fail_reason,
        "diagnostic_norm_fractional_change": final.get("diagnostic_norm_fractional_change"),
        "physical_grid_norm_fractional_change": final.get("physical_grid_norm_fractional_change"),
        "rho_max_fractional_change": final.get("rho_max_fractional_change"),
        "profile_overlap": final.get("profile_overlap"),
        "final_node_count": final.get("node_count"),
        "threshold_node_counts_final": json.dumps(final.get("threshold_node_counts", {})),
        "node_width_mean_box": final.get("node_width_mean_box"),
        "node_width_cv": final.get("node_width_cv"),
        "centroid_drift_mean_box": final.get("centroid_drift_mean_box"),
        "pairwise_distance_drift_mean_box": final.get("pairwise_distance_drift_mean_box"),
        "high_k_fraction": final.get("high_k_fraction"),
        "wallclock_sec": round(time.time() - t0, 3),
    }
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return row, history


def run_multistep_comparison(out_dir: Path, N: int, L: float, spacing: float, dt_values: list[float], times: list[float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool, float | None]:
    paths = rk4_diagnostic_paths(out_dir)
    rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    passed = True
    best_dt: float | None = None
    matched_losses: dict[tuple[float, float], dict[str, float]] = {}
    for dt in dt_values:
        for t_phys in times:
            modes = ["rk4_full_rhs"]
            if t_phys <= 1.0:
                modes = ["etdrk4_current", "etdrk4_no_mask", "rk4_full_rhs"]
            elif t_phys > 4.0:
                continue
            for mode in modes:
                row, hist = run_multistep_mode(mode, N, L, dt, spacing, t_phys)
                rows.append(row)
                history.extend(hist)
                if not row["finite"]:
                    passed = False
                    break
                loss = abs(float(row.get("diagnostic_norm_fractional_change") or 0.0))
                if mode == "rk4_full_rhs" and not multistep_pass_threshold(t_phys, loss):
                    passed = False
                if t_phys <= 1.0:
                    matched_losses.setdefault((dt, t_phys), {})[mode] = loss
                    if mode == "rk4_full_rhs" and len(matched_losses[(dt, t_phys)]) >= 2:
                        et_loss = matched_losses[(dt, t_phys)].get("etdrk4_current")
                        if et_loss is not None and loss > et_loss:
                            passed = False
                if float(row.get("wallclock_sec") or 0.0) > RUNTIME_CASE_WARN_SEC:
                    passed = False
                    break
            if not passed:
                break
        if not passed:
            break
    rk4_t1 = [r for r in rows if r["mode"] == "rk4_full_rhs" and abs(float(r["steps"]) * float(r["dt"]) - 1.0) < 1e-9 and r["finite"]]
    passing_t1 = [r for r in rk4_t1 if multistep_pass_threshold(1.0, abs(float(r.get("diagnostic_norm_fractional_change") or 0.0)))]
    if passing_t1:
        best_dt = max(float(r["dt"]) for r in passing_t1)
    write_csv(paths["multistep_csv"], rows)
    paths["multistep_history"].write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_history(paths["multistep_norm_plot"], history, "diagnostic_norm_fractional_change", "diagnostic norm fractional change")
    plot_history(paths["multistep_rho_plot"], history, "rho_max_fractional_change", "rho_max fractional change")
    plot_history(paths["multistep_overlap_plot"], history, "profile_overlap", "profile overlap")
    trends = late_trends(history)
    lines = [
        "# Multi-step RK4 vs ETDRK4",
        "",
        f"Passed invariant gates: `{passed}`",
        f"Best RK4 dt for replay: `{best_dt}`",
        f"Late trends: `{trends}`",
        "",
        "| case | loss | rho_max change | nodes | wallclock |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row.get('diagnostic_norm_fractional_change')} | {row.get('rho_max_fractional_change')} | "
            f"{row.get('final_node_count')} | {row.get('wallclock_sec')} |"
        )
    paths["multistep_report"].write_text("\n".join(lines), encoding="utf-8")
    return rows, history, passed, best_dt


def geometry_replay_cases() -> list[tuple[str, float]]:
    return [("triangle", 0.45), ("triangle", 0.36), ("ablated_triangle", 0.45), ("tetrahedron", 0.45), ("triangular_prism", 0.45)]


def run_geometry_case(out_dir: Path, template: str, spacing: float, N: int, L: float, dt: float, physical_time: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import cupy as cp

    case_id = f"rk4_{template}_s{spacing:.2f}_N{N}_T{physical_time:g}_dt{dt:g}".replace(".", "p")
    case_dir = out_dir / "geometry_replay" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    psi0, points = geometry_psi0(template, spacing, N, L)
    stepper = ConservativeC2RK4Stepper(N, L, dt)
    psi_k, psi_initial = stepper.project_psi0(psi0)
    initial_cpu = cp.asnumpy(psi_initial)
    initial_norms = norm_conventions(initial_cpu, L=L)
    initial_rho = np.abs(initial_cpu) ** 2
    initial_geom = analyse_psi_geometry(initial_cpu, L=L, expected_nodes=len(points))
    steps = steps_for_physical_time(physical_time, dt)
    sample_every = max(1, steps // 40)
    history: list[dict[str, Any]] = []
    geometry_history: list[dict[str, Any]] = []
    fail_reason = ""
    t0 = time.time()
    for step in range(steps):
        psi_k = stepper.step(psi_k)
        if step % sample_every == 0 or step == steps - 1:
            psi_now = stepper.to_physical(psi_k)
            psi_cpu = cp.asnumpy(psi_now)
            if not np.isfinite(psi_cpu).all():
                fail_reason = f"nonfinite at step {step + 1}"
                break
            norms = norm_conventions(psi_cpu, L=L)
            rho = np.abs(psi_cpu) ** 2
            geom = analyse_psi_geometry(psi_cpu, L=L, expected_nodes=max(1, len(points)))
            drift = centroid_drift_metrics(initial_geom, geom, N=N, t_phys=(step + 1) * dt)
            pair_drift = pairwise_distance_drift(initial_geom, geom)
            rec = {
                "case_id": case_id,
                "step": int(step + 1),
                "t_phys": float((step + 1) * dt),
                "diagnostic_norm_fractional_change": float((norms["diagnostic_norm"] - initial_norms["diagnostic_norm"]) / (abs(initial_norms["diagnostic_norm"]) + 1e-30)),
                "physical_grid_norm_fractional_change": float((norms["physical_grid_norm"] - initial_norms["physical_grid_norm"]) / (abs(initial_norms["physical_grid_norm"]) + 1e-30)),
                "rho_max_fractional_change": float((float(np.max(rho)) - float(np.max(initial_rho))) / (abs(float(np.max(initial_rho))) + 1e-30)),
                "profile_overlap": profile_overlap(initial_rho, rho),
                "node_count": geom.get("node_count"),
                "threshold_node_counts": geom.get("threshold_node_counts"),
                "mass_cv": geom.get("node_mass_cv"),
                "peak_cv": geom.get("rho_peak_cv"),
                "z_spread_box": geom.get("z_spread_box"),
                "planarity_score": geom.get("planarity_score"),
                "node_width_mean_box": geom.get("node_width_mean_box"),
                "node_width_cv": geom.get("node_width_cv"),
                "high_k_fraction": high_k_fraction(cp.asnumpy(psi_k), cp.asnumpy(stepper.solver.reference_dealias_mask).astype(bool)),
                **drift,
                **pair_drift,
            }
            history.append(rec)
            geometry_history.append({"step": int(step + 1), "t_phys": rec["t_phys"], "geometry": geom})
        if time.time() - t0 > RUNTIME_CASE_WARN_SEC:
            fail_reason = f"runtime guard exceeded {RUNTIME_CASE_WARN_SEC:g}s"
            break
    psi_fin = stepper.to_physical(psi_k)
    psi_fin_cpu = cp.asnumpy(psi_fin)
    rho_fin = np.abs(psi_fin_cpu) ** 2
    final_geom = analyse_psi_geometry(psi_fin_cpu, L=L, expected_nodes=max(1, len(points)))
    render_final_rho_orthogonal(rho_fin, final_geom, case_dir / f"{case_id}_final_rho.png")
    cfg = {"case_id": case_id, "geometry": {"template": template, "spacing_box": spacing}, "simulation": {"n_grid": N, "dt": dt, "physical_time": physical_time}}
    write_centroid_plotly_html(case_dir / f"{case_id}_centroids.html", geometry_history, cfg)
    final = history[-1] if history else {}
    row = {
        "case_id": case_id,
        "template": template,
        "spacing_box": float(spacing),
        "N": int(N),
        "L": float(L),
        "dt": float(dt),
        "physical_time": float(physical_time),
        "finite": not fail_reason,
        "fail_reason": fail_reason,
        "initial_node_count": initial_geom.get("node_count"),
        "final_node_count": final_geom.get("node_count"),
        "threshold_node_counts_final": json.dumps(final_geom.get("threshold_node_counts", {})),
        "diagnostic_norm_fractional_change": final.get("diagnostic_norm_fractional_change"),
        "physical_grid_norm_fractional_change": final.get("physical_grid_norm_fractional_change"),
        "rho_max_fractional_change": final.get("rho_max_fractional_change"),
        "profile_overlap": final.get("profile_overlap"),
        "mass_cv": final.get("mass_cv"),
        "peak_cv": final.get("peak_cv"),
        "pairwise_distance_drift_mean_box": final.get("pairwise_distance_drift_mean_box"),
        "centroid_drift_mean_box": final.get("centroid_drift_mean_box"),
        "z_spread_box": final.get("z_spread_box"),
        "planarity_score": final.get("planarity_score"),
        "node_width_mean_box": final.get("node_width_mean_box"),
        "node_width_cv": final.get("node_width_cv"),
        "high_k_fraction": final.get("high_k_fraction"),
        "final_rho_image": str(case_dir / f"{case_id}_final_rho.png"),
        "plotly_html": str(case_dir / f"{case_id}_centroids.html"),
        "wallclock_sec": round(time.time() - t0, 3),
    }
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return row, history


def run_geometry_replay(out_dir: Path, N: int, L: float, best_dt: float, times: list[float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    paths = rk4_diagnostic_paths(out_dir)
    rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    classification = "RK4_GEOMETRY_REPLAY_PROMISING"
    for t_phys in (1.0, 2.0, 4.0):
        if t_phys not in times:
            continue
        time_rows: list[dict[str, Any]] = []
        for template, spacing in geometry_replay_cases():
            row, hist = run_geometry_case(out_dir, template, spacing, N, L, best_dt, t_phys)
            rows.append(row)
            time_rows.append(row)
            history.extend(hist)
            if not row["finite"]:
                classification = "CONSERVATIVE_STEPPER_STILL_BLOCKED"
                break
            loss = abs(float(row.get("diagnostic_norm_fractional_change") or 0.0))
            if not multistep_pass_threshold(t_phys, loss):
                classification = "RK4_INVARIANT_NOT_SUFFICIENT"
                break
        if classification != "RK4_GEOMETRY_REPLAY_PROMISING":
            break
        if any(int(r.get("final_node_count") or -1) != int(r.get("initial_node_count") or -2) for r in time_rows):
            classification = "GEOMETRY_DEPENDED_ON_LOSSY_ETDRK4"
            break
    write_csv(paths["geometry_csv"], rows)
    paths["geometry_history"].write_text(json.dumps(history, indent=2), encoding="utf-8")
    lines = [
        "# RK4 Geometry Replay",
        "",
        f"Replay classification: `{classification}`",
        "",
        "| case | nodes | norm change | rho max change | finite |",
        "|---|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row.get('initial_node_count')} -> {row.get('final_node_count')} | "
            f"{row.get('diagnostic_norm_fractional_change')} | {row.get('rho_max_fractional_change')} | {row.get('finite')} |"
        )
    paths["geometry_report"].write_text("\n".join(lines), encoding="utf-8")
    return rows, history, classification


def native_profile_check(out_dir: Path, L: float, dt: float) -> str:
    paths = rk4_diagnostic_paths(out_dir)
    path = _find_native_c2_profile()
    if path is None:
        paths["native_profile_report"].write_text("# Native C2 Profile RK4 Check\n\nNo native Conservative C2 psi-like artifact found.\n", encoding="utf-8")
        return "profile_mismatch_untested"
    try:
        psi0 = _load_native_profile(path)
        stepper = ConservativeC2RK4Stepper(int(psi0.shape[0]), L, dt)
        psi_k, psi_initial = stepper.project_psi0(psi0)
        initial = norm_conventions(__import__("cupy").asnumpy(psi_initial), L=L)
        psi_next = stepper.to_physical(stepper.step(psi_k))
        final = norm_conventions(__import__("cupy").asnumpy(psi_next), L=L)
        frac = (final["diagnostic_norm"] - initial["diagnostic_norm"]) / (abs(initial["diagnostic_norm"]) + 1e-30)
        paths["native_profile_report"].write_text(
            f"# Native C2 Profile RK4 Check\n\nArtifact: `{path}`\n\nOne-step fractional norm change: `{frac}`\n",
            encoding="utf-8",
        )
        return "native_profile_checked"
    except Exception as exc:
        paths["native_profile_report"].write_text(f"# Native C2 Profile RK4 Check\n\nArtifact check failed: `{exc}`\n", encoding="utf-8")
        return "profile_mismatch_untested"


def write_final_report(
    out_dir: Path,
    *,
    commands_run: list[str],
    safety: dict[str, Any],
    one_step_proceed: bool,
    multistep_passed: bool | None,
    geometry_classification: str | None,
    final_decision: str,
    secondary_flags: list[str],
) -> None:
    paths = rk4_diagnostic_paths(out_dir)
    lines = [
        "# RK4 Stepper Diagnostic Final Report",
        "",
        f"Final decision: `{final_decision}`",
        f"Secondary flags: `{', '.join(secondary_flags)}`",
        "",
        "## Files Added Or Modified",
        "",
        "- `tools/conservative_rk4_stepper_diagnostic.py`",
        "- `tests/test_conservative_rk4_stepper_diagnostic.py`",
        "- Diagnostic outputs under `quantule_viz/outputs/conservative_geometry_campaign/rk4_stepper_diagnostic/`",
        "",
        "## Environment",
        "",
        f"- Python path: `{safety.get('python_executable')}`",
        f"- CuPy version: `{safety.get('cupy_version')}`",
        f"- GPU: `{safety.get('gpu', {}).get('gpu_name')}`",
        f"- JAX/JAXLIB absent: `{safety.get('jax_spec') == 'None' and safety.get('jaxlib_spec') == 'None'}`",
        "",
        "## Commands Run",
        "",
        *[f"- `{cmd}`" for cmd in commands_run],
        "",
        "## Summary",
        "",
        "- Conservative C2 spectral contract reused from prior audit.",
        "- RK4 is diagnostic-only and not promoted as production.",
        f"- One-step RK4 gate passed: `{one_step_proceed}`",
        f"- Multi-step RK4 gate passed: `{multistep_passed}`",
        f"- Geometry replay classification: `{geometry_classification}`",
        "- No stability claim is made.",
        "- No longer conservative campaign was run.",
    ]
    paths["final_report"].write_text("\n".join(lines), encoding="utf-8")


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out)
    commands = ["smoke"]
    safety = run_safety_checkpoint(out_dir, args.require_gpu_name)
    dt_values = [float(args.dt)]
    one_step, proceed = run_one_step_comparison(out_dir, args.N, args.L, args.spacing, dt_values)
    row, hist = run_multistep_mode("rk4_full_rhs", args.N, args.L, args.dt, args.spacing, args.physical_time, sample_count=8)
    paths = rk4_diagnostic_paths(out_dir)
    write_csv(paths["multistep_csv"], [row])
    paths["multistep_history"].write_text(json.dumps(hist, indent=2), encoding="utf-8")
    profile_flag = native_profile_check(out_dir, args.L, args.dt)
    flags = ["no_stability_claim", "no_production_change", "no_amplitude_normalization", "long_campaign_not_run", "explicit_rk4_diagnostic_only", profile_flag]
    final_decision = "RK4_STEPPER_DIAGNOSTIC_PROMISING" if row["finite"] else "CONSERVATIVE_STEPPER_STILL_BLOCKED"
    write_final_report(out_dir, commands_run=commands, safety=safety, one_step_proceed=proceed, multistep_passed=row["finite"], geometry_classification=None, final_decision=final_decision, secondary_flags=flags)
    return {"final_decision": final_decision, "paths": {k: str(v) for k, v in paths.items()}, "one_step_rows": one_step, "smoke_row": row}


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    commands = ["audit"]
    safety = run_safety_checkpoint(out_dir, args.require_gpu_name)
    dt_values = parse_float_list(args.dt_list)
    times = parse_float_list(args.times)
    one_step_rows, one_step_proceed = run_one_step_comparison(out_dir, args.N, args.L, args.spacing, dt_values)
    profile_flag = native_profile_check(out_dir, args.L, min(dt_values))
    flags = ["no_stability_claim", "no_production_change", "no_amplitude_normalization", "long_campaign_not_run", "explicit_rk4_diagnostic_only", profile_flag]
    final_decision = "RK4_STEPPER_REJECTED"
    multistep_passed: bool | None = None
    geometry_classification: str | None = None
    if one_step_proceed:
        _, _, multistep_passed, best_dt = run_multistep_comparison(out_dir, args.N, args.L, args.spacing, dt_values, times)
        if multistep_passed and best_dt is not None:
            geometry_rows, _, geometry_classification = run_geometry_replay(out_dir, args.N, args.L, best_dt, times)
            if geometry_classification == "RK4_GEOMETRY_REPLAY_PROMISING":
                final_decision = "RK4_GEOMETRY_REPLAY_PROMISING"
            elif geometry_classification == "GEOMETRY_DEPENDED_ON_LOSSY_ETDRK4":
                final_decision = "GEOMETRY_DEPENDED_ON_LOSSY_ETDRK4"
            elif geometry_classification == "RK4_INVARIANT_NOT_SUFFICIENT":
                final_decision = "RK4_STEPPER_DIAGNOSTIC_PROMISING"
            else:
                final_decision = "CONSERVATIVE_STEPPER_STILL_BLOCKED"
        elif multistep_passed:
            final_decision = "RK4_STEPPER_DIAGNOSTIC_PROMISING"
        else:
            final_decision = "RK4_STEPPER_REJECTED"
    write_final_report(
        out_dir,
        commands_run=commands,
        safety=safety,
        one_step_proceed=one_step_proceed,
        multistep_passed=multistep_passed,
        geometry_classification=geometry_classification,
        final_decision=final_decision,
        secondary_flags=flags,
    )
    return {
        "final_decision": final_decision,
        "one_step_proceed": one_step_proceed,
        "multistep_passed": multistep_passed,
        "geometry_classification": geometry_classification,
        "paths": {k: str(v) for k, v in rk4_diagnostic_paths(out_dir).items()},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnostic Conservative C2 full-RHS RK4 stepper")
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--out", type=Path, required=True)
    smoke.add_argument("--N", type=int, default=32)
    smoke.add_argument("--L", type=float, default=DEFAULT_L)
    smoke.add_argument("--spacing", type=float, default=0.45)
    smoke.add_argument("--dt", type=float, default=0.001)
    smoke.add_argument("--physical-time", type=float, default=0.02)
    smoke.add_argument("--require-gpu-name", default="NVIDIA GeForce GTX 1080")
    smoke.set_defaults(func=run_smoke)

    audit = sub.add_parser("audit")
    audit.add_argument("--out", type=Path, default=DEFAULT_RK4_OUT)
    audit.add_argument("--N", type=int, default=48)
    audit.add_argument("--L", type=float, default=DEFAULT_L)
    audit.add_argument("--spacing", type=float, default=0.45)
    audit.add_argument("--dt-list", default="0.001,0.0005,0.00025,0.000125")
    audit.add_argument("--times", default="0.5,1.0,2.0,4.0")
    audit.add_argument("--require-gpu-name", default="NVIDIA GeForce GTX 1080")
    audit.set_defaults(func=run_audit)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = args.func(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
