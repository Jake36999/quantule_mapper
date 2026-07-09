"""Standalone Conservative C2 nonlinear stepper contract audit.

This diagnostic is intentionally additive. It does not modify production solver
physics, worker paths, Hunter objectives, validation gates, configs, or the
``jax_scout`` reference files. It uses CuPy only at runtime.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
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
    high_k_fraction,
    late_trends,
    points_for_case,
    profile_overlap,
    steps_for_physical_time,
)


DEFAULT_AUDIT_OUT = DEFAULT_OUT / "stepper_contract_audit"
FLUX_ZERO_THRESHOLD = 1e-10
FLUX_WARNING_THRESHOLD = 1e-8
FLUX_HARD_STOP_THRESHOLD = 1e-6
MATERIAL_NORM_CHANGE_THRESHOLD = 1e-6
RUNTIME_CASE_WARN_SEC = 600.0


def stepper_audit_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "contract_snapshot": out_dir / "contract_snapshot.json",
        "rhs_flux_report": out_dir / "rhs_flux_report.md",
        "csv": out_dir / "conservative_stepper_contract_audit_results.csv",
        "report": out_dir / "conservative_stepper_contract_audit_report.md",
        "history": out_dir / "conservative_stepper_contract_audit_history.json",
        "norm_plot": out_dir / "norm_loss_vs_dt.png",
        "one_step_plot": out_dir / "one_step_norm_defect_vs_dt.png",
        "rho_plot": out_dir / "rho_max_vs_time.png",
    }


def classify_rhs_flux(fractional_flux_raw: float) -> str:
    mag = abs(float(fractional_flux_raw))
    if mag <= FLUX_ZERO_THRESHOLD:
        return "numerical_zero"
    if mag > FLUX_HARD_STOP_THRESHOLD:
        return "hard_stop"
    if mag > FLUX_WARNING_THRESHOLD:
        return "warning"
    return "small_nonzero"


def norm_conventions(psi_phys: np.ndarray, L: float) -> dict[str, float]:
    arr = np.asarray(psi_phys, dtype=np.complex128)
    dx = float(L) / float(arr.shape[0])
    diagnostic_norm = float(np.sum(np.abs(arr) ** 2, dtype=np.float64))
    return {
        "diagnostic_norm": diagnostic_norm,
        "dx": dx,
        "physical_grid_norm": float((dx**3) * diagnostic_norm),
    }


def fractional_rhs_flux(psi_phys: np.ndarray, rhs_phys: np.ndarray, L: float) -> dict[str, float]:
    psi = np.asarray(psi_phys, dtype=np.complex128)
    rhs = np.asarray(rhs_phys, dtype=np.complex128)
    norms = norm_conventions(psi, L=L)
    d_norm_dt_raw = float(2.0 * np.real(np.sum(np.conj(psi) * rhs, dtype=np.complex128)))
    diagnostic_norm = norms["diagnostic_norm"]
    dx = norms["dx"]
    return {
        **norms,
        "d_norm_dt_raw": d_norm_dt_raw,
        "fractional_flux_raw": float(d_norm_dt_raw / (abs(diagnostic_norm) + 1e-30)),
        "d_physical_grid_norm_dt": float((dx**3) * d_norm_dt_raw),
        "fractional_physical_grid_flux": float(d_norm_dt_raw / (abs(diagnostic_norm) + 1e-30)),
    }


def rhs_flux_probe_specs(spacing: float) -> list[dict[str, Any]]:
    return [
        {"case_id": "uniform_constant_control", "kind": "uniform", "expected_nodes": 0},
        {"case_id": "single_gaussian_node", "kind": "single_gaussian", "expected_nodes": 1},
        {"case_id": f"ablated_triangle_spacing_{spacing:g}", "kind": "ablated_triangle", "expected_nodes": 2},
        {"case_id": f"triangle_spacing_{spacing:g}", "kind": "triangle", "expected_nodes": 3},
    ]


def classify_audit_result(
    *,
    contract_ok: bool,
    rhs_flux_hard_stop: bool,
    timestep_error: bool,
    material_norm_change: bool,
    profile_mismatch_evidence: bool,
    native_profile_tested: bool,
    total_norm_conserved: bool,
    rho_or_profile_changed: bool,
    secondary_flags: list[str],
) -> dict[str, Any]:
    flags = list(dict.fromkeys(secondary_flags))
    if not contract_ok:
        return {
            "primary_classification": "cupy_wrapper_contract_mismatch",
            "secondary_flags": flags,
            "final_decision": "WRAPPER_PARITY_REVIEW_REQUIRED",
        }
    if rhs_flux_hard_stop:
        return {
            "primary_classification": "nonlinear_rhs_algebra",
            "secondary_flags": flags,
            "final_decision": "RHS_CONTRACT_REVIEW_REQUIRED",
        }
    if profile_mismatch_evidence and native_profile_tested:
        return {
            "primary_classification": "synthetic_profile_mismatch",
            "secondary_flags": flags,
            "final_decision": "UNCLEAR_BLOCKED",
        }
    if profile_mismatch_evidence and not native_profile_tested and "profile_mismatch_untested" not in flags:
        flags.append("profile_mismatch_untested")
    if timestep_error:
        return {
            "primary_classification": "etdrk4_timestep_error",
            "secondary_flags": flags,
            "final_decision": "ETDRK4_ERROR_CONFIRMED",
        }
    if total_norm_conserved and rho_or_profile_changed and not material_norm_change:
        return {
            "primary_classification": "true_conservative_dispersion_without_norm_loss",
            "secondary_flags": flags,
            "final_decision": "STEPPER_READY_FOR_SMALL_GEOMETRY_REPLAY",
        }
    return {"primary_classification": "unclear", "secondary_flags": flags, "final_decision": "UNCLEAR_BLOCKED"}


def parse_dt_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("dt-list must contain at least one value")
    return values


def _source_contract_findings() -> dict[str, Any]:
    import tools.conservative_geometry_campaign as campaign

    ref_path = ROOT / "jax_scout" / "physics.py"
    ref_file = ref_path.read_text(encoding="utf-8")
    n_op_start = ref_file.find("def n_op(")
    step_start = ref_file.find("def step(")
    ops_start = ref_file.find("def _construct_ops(")
    if min(n_op_start, step_start, ops_start) < 0:
        ref_src = ref_step_src = ref_ops_src = ""
    else:
        ref_src = ref_file[n_op_start:step_start]
        ref_step_src = ref_file[step_start:ops_start]
        ref_ops_src = ref_file[ops_start:]
    wrapper_src = inspect.getsource(campaign.ConservativeCupySolver)

    findings = {
        "reference_n_op_expects": "spectral_psi_k" if "def n_op(psi_k" in ref_src and "ifftn(psi_k)" in ref_src else "ambiguous",
        "reference_n_op_returns": "spectral_rhs" if "fftn(n_real)" in ref_src and "return n_k" in ref_src else "ambiguous",
        "reference_step_boundary": "spectral_psi_k" if "def step(psi_k" in ref_step_src else "ambiguous",
        "wrapper_n_op_expects": "spectral_psi_k" if "def N_op(self, psi_k)" in wrapper_src else "ambiguous",
        "wrapper_n_op_returns": "spectral_rhs" if "return (1j * super().N_op(psi_k))" in wrapper_src else "ambiguous",
        "reference_conservative_linear_operator": "-1j * D * k^2" if "L_k = (-1j * D_diff * k_sq)" in ref_ops_src else "ambiguous",
        "reference_param_eta_inactive_in_conservative": "if kinetic_mode == \"conservative\"" in ref_ops_src
        and "NO gain/loss" in ref_ops_src,
        "reference_nonlinear_factor": "kfac=1j" if "kfac=1j" in ref_ops_src.replace(" ", "") and "n_real = n_real * ops.kfac" in ref_src else "ambiguous",
        "wrapper_nonlinear_factor": "1j" if "return (1j * super().N_op(psi_k))" in wrapper_src else "ambiguous",
        "quasi_conservative_contract": "quasi-conservative" in ref_ops_src.lower() or "not norm-conserv" in ref_ops_src.lower(),
    }
    findings["source_paths"] = {
        "reference": str(ref_path),
        "wrapper": str(ROOT / "tools" / "conservative_geometry_campaign.py"),
    }
    domain_match = (
        findings["reference_n_op_expects"] == findings["wrapper_n_op_expects"] == "spectral_psi_k"
        and findings["reference_n_op_returns"] == findings["wrapper_n_op_returns"] == "spectral_rhs"
    )
    factor_match = findings["reference_nonlinear_factor"] == "kfac=1j" and findings["wrapper_nonlinear_factor"] == "1j"
    findings["domain_match"] = bool(domain_match)
    findings["factor_match"] = bool(factor_match)
    findings["source_contract_ok"] = bool(domain_match and factor_match and findings["reference_conservative_linear_operator"] != "ambiguous")
    return findings


def build_contract_snapshot(N: int, L: float, dt: float, require_gpu_name: str | None) -> tuple[dict[str, Any], bool, list[str]]:
    import cupy as cp

    gpu = _gpu_info(require_gpu_name)
    source = _source_contract_findings()
    params = dict(CONSERVATIVE_PARAMS)
    solver = ConservativeCupySolver(N, L, dt, params)
    eta_params = dict(params)
    eta_params["param_eta"] = float(params.get("param_eta", 0.0)) + 123.0
    solver_eta = ConservativeCupySolver(N, L, dt, eta_params)
    expected = (-1j * float(params["param_D"]) * solver.k_sq).astype(cp.complex128)
    max_l_diff = float(cp.max(cp.abs(solver.L_k - expected)))
    max_eta_diff = float(cp.max(cp.abs(solver.L_k - solver_eta.L_k)))
    snapshot = {
        "python_executable": sys.executable,
        "cupy": gpu,
        "jax_spec": str(importlib.util.find_spec("jax")),
        "jaxlib_spec": str(importlib.util.find_spec("jaxlib")),
        "N": int(N),
        "L": float(L),
        "dt": float(dt),
        "dx": float(L) / float(N),
        "dtype": "complex128",
        "params": params,
        "mask_mode": "current",
        "source_contract": source,
        "linear_operator_max_abs_diff": max_l_diff,
        "param_eta_inactive_max_abs_diff": max_eta_diff,
        "linear_operator_ok": max_l_diff <= 1e-12,
        "param_eta_inactive_ok": max_eta_diff <= 1e-12,
        "rhs_boundary": {
            "reference_expects": source["reference_n_op_expects"],
            "reference_returns": source["reference_n_op_returns"],
            "wrapper_expects": source["wrapper_n_op_expects"],
            "wrapper_returns": source["wrapper_n_op_returns"],
        },
    }
    flags: list[str] = []
    if source["quasi_conservative_contract"]:
        flags.append("CONTRACT_INVARIANT_AMBIGUOUS")
    if not source["domain_match"]:
        flags.append("fft_domain_mismatch")
    if not source["factor_match"]:
        flags.append("nonlinear_factor_mismatch")
    if not snapshot["linear_operator_ok"]:
        flags.append("linear_operator_mismatch")
    if not snapshot["param_eta_inactive_ok"]:
        flags.append("param_eta_active_in_conservative_lk")
    contract_ok = not flags
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return snapshot, contract_ok, flags


def _psi_for_probe(kind: str, N: int, L: float, spacing: float) -> np.ndarray:
    if kind == "uniform":
        return np.ones((N, N, N), dtype=np.complex128)
    if kind == "single_gaussian":
        pts = np.array([[0.5, 0.5, 0.5]], dtype=float)
        return build_geometry_ic(N, L, pts, DEFAULT_WIDTH_BOX, 1.0)
    if kind == "ablated_triangle":
        pts = points_for_case("ablated_triangle", spacing, {}, seed=DEFAULT_SEED)
        return build_geometry_ic(N, L, pts, DEFAULT_WIDTH_BOX, 1.0, phases=[0.0, 0.0])
    if kind == "triangle":
        pts = points_for_case("triangle", spacing, {}, seed=DEFAULT_SEED)
        return build_geometry_ic(N, L, pts, DEFAULT_WIDTH_BOX, 1.0, phases=[0.0, 0.0, 0.0])
    raise ValueError(f"unsupported probe kind {kind!r}")


def _solver_for(N: int, L: float, dt: float, *, mask_mode: str = "current", nonlinear_enabled: bool = True) -> ConservativeCupySolver:
    return ConservativeCupySolver(N, L, dt, dict(CONSERVATIVE_PARAMS), nonlinear_enabled=nonlinear_enabled, mask_mode=mask_mode)


def _project_initial(solver: ConservativeCupySolver, psi_phys: Any):
    import cupy as cp

    psi = cp.asarray(psi_phys, dtype=cp.complex128)
    psi_k = solver.fft_single(psi) * solver.dealias_mask
    projected = solver.ifft_single(psi_k)
    return psi_k, projected


def run_rhs_flux_probe(spec: dict[str, Any], N: int, L: float, dt: float, spacing: float) -> dict[str, Any]:
    import cupy as cp

    solver = _solver_for(N, L, dt)
    psi0 = _psi_for_probe(str(spec["kind"]), N=N, L=L, spacing=spacing)
    psi_k, psi_phys = _project_initial(solver, psi0)
    rhs_k = solver.N_op(psi_k)
    rhs_phys = solver.ifft_single(rhs_k)
    psi_cpu = cp.asnumpy(psi_phys)
    rhs_cpu = cp.asnumpy(rhs_phys)
    flux = fractional_rhs_flux(psi_cpu, rhs_cpu, L=L)
    rho = np.abs(psi_cpu) ** 2
    geom = analyse_psi_geometry(psi_cpu, L=L, expected_nodes=max(1, int(spec.get("expected_nodes", 1))))
    status = classify_rhs_flux(flux["fractional_flux_raw"])
    row = {
        "case_id": spec["case_id"],
        "group": "rhs_flux",
        "N": int(N),
        "L": float(L),
        "dt": float(dt),
        "steps": 0,
        "mode": "instantaneous_conservative_nonlinear_rhs",
        "finite": bool(np.isfinite(psi_cpu).all() and np.isfinite(rhs_cpu).all()),
        "fail_reason": "",
        "diagnostic_norm": flux["diagnostic_norm"],
        "physical_grid_norm": flux["physical_grid_norm"],
        "d_norm_dt_raw": flux["d_norm_dt_raw"],
        "fractional_flux_raw": flux["fractional_flux_raw"],
        "rhs_flux_status": status,
        "rho_max": float(np.max(rho)),
        "rho_max_fractional_change": 0.0,
        "node_count": geom.get("node_count"),
        "profile_overlap": 1.0,
        "high_k_fraction": high_k_fraction(cp.asnumpy(psi_k), cp.asnumpy(solver.reference_dealias_mask).astype(bool)),
        "recombination_mask_removed_fraction": 0.0,
        "wallclock_sec": 0.0,
    }
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return row


def find_native_profile_note() -> tuple[np.ndarray | None, str]:
    path = _find_native_c2_profile()
    if path is None:
        return None, "not_run_no_artifact"
    try:
        return _load_native_profile(path), f"loaded:{path}"
    except Exception as exc:
        return None, f"not_run_load_failed:{path}:{exc}"


def run_native_rhs_flux_probe(psi0: np.ndarray, note: str, L: float, dt: float) -> dict[str, Any]:
    import cupy as cp

    N = int(psi0.shape[0])
    solver = _solver_for(N, L, dt)
    psi_k, psi_phys = _project_initial(solver, psi0)
    rhs_k = solver.N_op(psi_k)
    rhs_phys = solver.ifft_single(rhs_k)
    psi_cpu = cp.asnumpy(psi_phys)
    rhs_cpu = cp.asnumpy(rhs_phys)
    flux = fractional_rhs_flux(psi_cpu, rhs_cpu, L=L)
    status = classify_rhs_flux(flux["fractional_flux_raw"])
    geom = analyse_psi_geometry(psi_cpu, L=L, expected_nodes=8)
    row = {
        "case_id": "native_c2_profile",
        "group": "rhs_flux",
        "N": N,
        "L": float(L),
        "dt": float(dt),
        "steps": 0,
        "mode": "instantaneous_conservative_nonlinear_rhs",
        "finite": bool(np.isfinite(psi_cpu).all() and np.isfinite(rhs_cpu).all()),
        "fail_reason": note,
        "diagnostic_norm": flux["diagnostic_norm"],
        "physical_grid_norm": flux["physical_grid_norm"],
        "d_norm_dt_raw": flux["d_norm_dt_raw"],
        "fractional_flux_raw": flux["fractional_flux_raw"],
        "rhs_flux_status": status,
        "rho_max": float(np.max(np.abs(psi_cpu) ** 2)),
        "rho_max_fractional_change": 0.0,
        "node_count": geom.get("node_count"),
        "profile_overlap": 1.0,
        "high_k_fraction": high_k_fraction(cp.asnumpy(psi_k), cp.asnumpy(solver.reference_dealias_mask).astype(bool)),
        "recombination_mask_removed_fraction": 0.0,
        "wallclock_sec": 0.0,
    }
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return row


def write_rhs_flux_report(path: Path, rows: list[dict[str, Any]], hard_stop: bool, native_note: str) -> None:
    lines = [
        "# Conservative C2 RHS Flux Report",
        "",
        "CuPy-only standalone diagnostic. No stability claim is made.",
        "",
        f"Native C2 profile: `{native_note}`",
        f"Hard stop triggered: `{hard_stop}`",
        "",
        "Thresholds: `<=1e-10` numerical zero; `>1e-8` warning; `>1e-6` hard stop unless C2 contract says norm is not conserved.",
        "",
        "| case | status | diagnostic norm | physical-grid norm | fractional flux | d_norm_dt_raw | nodes |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row.get('rhs_flux_status')} | {row.get('diagnostic_norm'):.12g} | "
            f"{row.get('physical_grid_norm'):.12g} | {row.get('fractional_flux_raw'):.12g} | "
            f"{row.get('d_norm_dt_raw'):.12g} | {row.get('node_count')} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_one_step_case(N: int, L: float, dt: float, spacing: float) -> dict[str, Any]:
    import cupy as cp

    solver = _solver_for(N, L, dt)
    psi0 = _psi_for_probe("triangle", N=N, L=L, spacing=spacing)
    psi_k, psi_phys = _project_initial(solver, psi0)
    initial_cpu = cp.asnumpy(psi_phys)
    initial_norms = norm_conventions(initial_cpu, L=L)
    rho0_max = float(np.max(np.abs(initial_cpu) ** 2))
    t0 = time.time()
    N_n = solver.N_op(psi_k)
    a_k = solver.E2 * psi_k + solver.Q * N_n
    N_a = solver.N_op(a_k)
    b_k = solver.E2 * psi_k + solver.Q * N_a
    N_b = solver.N_op(b_k)
    c_k = solver.E2 * a_k + solver.Q * (2.0 * N_b - N_a)
    N_c = solver.N_op(c_k)
    psi_next_k = solver.step(psi_k)
    psi_next = solver.ifft_single(psi_next_k)
    final_cpu = cp.asnumpy(psi_next)
    final_norms = norm_conventions(final_cpu, L=L)
    defect = (final_norms["diagnostic_norm"] - initial_norms["diagnostic_norm"]) / (abs(initial_norms["diagnostic_norm"]) + 1e-30)
    row = {
        "case_id": f"one_step_etdrk4_dt_{dt:g}",
        "group": "one_step",
        "N": int(N),
        "L": float(L),
        "dt": float(dt),
        "steps": 1,
        "mode": "etdrk4_current",
        "finite": bool(np.isfinite(final_cpu).all()),
        "fail_reason": "",
        "diagnostic_norm": final_norms["diagnostic_norm"],
        "physical_grid_norm": final_norms["physical_grid_norm"],
        "initial_diagnostic_norm": initial_norms["diagnostic_norm"],
        "initial_physical_grid_norm": initial_norms["physical_grid_norm"],
        "fractional_norm_defect": float(defect),
        "rho_max": float(np.max(np.abs(final_cpu) ** 2)),
        "rho_max_fractional_change": float((float(np.max(np.abs(final_cpu) ** 2)) - rho0_max) / (abs(rho0_max) + 1e-30)),
        "stage_norm_psi": initial_norms["diagnostic_norm"],
        "stage_norm_a": norm_conventions(cp.asnumpy(solver.ifft_single(a_k)), L=L)["diagnostic_norm"],
        "stage_norm_b": norm_conventions(cp.asnumpy(solver.ifft_single(b_k)), L=L)["diagnostic_norm"],
        "stage_norm_c": norm_conventions(cp.asnumpy(solver.ifft_single(c_k)), L=L)["diagnostic_norm"],
        "nonlinear_norm_N_n": float(cp.linalg.norm(N_n).get()),
        "nonlinear_norm_N_a": float(cp.linalg.norm(N_a).get()),
        "nonlinear_norm_N_b": float(cp.linalg.norm(N_b).get()),
        "nonlinear_norm_N_c": float(cp.linalg.norm(N_c).get()),
        "high_k_fraction": high_k_fraction(cp.asnumpy(psi_next_k), cp.asnumpy(solver.reference_dealias_mask).astype(bool)),
        "recombination_mask_removed_fraction": float(getattr(solver, "last_recombination_mask_removed_fraction", 0.0)),
        "wallclock_sec": round(time.time() - t0, 3),
    }
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return row


def _rhs_full_k(solver: ConservativeCupySolver, psi_k: Any):
    return solver.L_k * psi_k + solver.N_op(psi_k)


def _rk4_step_k(solver: ConservativeCupySolver, psi_k: Any, rhs_func):
    dt = solver.dt
    k1 = rhs_func(solver, psi_k)
    k2 = rhs_func(solver, psi_k + 0.5 * dt * k1)
    k3 = rhs_func(solver, psi_k + 0.5 * dt * k2)
    k4 = rhs_func(solver, psi_k + dt * k3)
    return psi_k + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def run_multistep_case(
    N: int,
    L: float,
    dt: float,
    spacing: float,
    physical_time: float,
    mode: str,
    sample_count: int = 40,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import cupy as cp

    mask_mode = "none" if mode == "etdrk4_no_mask" else "current"
    solver = _solver_for(N, L, dt, mask_mode=mask_mode)
    psi0 = _psi_for_probe("triangle", N=N, L=L, spacing=spacing)
    psi_k, psi_phys = _project_initial(solver, psi0)
    initial_cpu = cp.asnumpy(psi_phys)
    initial_norms = norm_conventions(initial_cpu, L=L)
    initial_rho = np.abs(initial_cpu) ** 2
    steps = steps_for_physical_time(physical_time, dt)
    sample_every = max(1, steps // max(1, sample_count))
    history: list[dict[str, Any]] = []
    fail_reason = ""
    t0 = time.time()
    for step in range(steps):
        if mode in ("etdrk4_current", "etdrk4_no_mask"):
            psi_k = solver.step(psi_k)
        elif mode == "rk4_full_rhs":
            psi_k = _rk4_step_k(solver, psi_k, _rhs_full_k)
        elif mode == "rk4_nonlinear_only":
            psi_k = _rk4_step_k(solver, psi_k, lambda s, y: s.N_op(y))
        else:
            raise ValueError(f"unsupported mode {mode!r}")
        if step % sample_every == 0 or step == steps - 1:
            psi_phys_now = solver.ifft_single(psi_k)
            psi_cpu = cp.asnumpy(psi_phys_now)
            if not np.isfinite(psi_cpu).all():
                fail_reason = f"nonfinite at step {step}"
                break
            norms = norm_conventions(psi_cpu, L=L)
            rho = np.abs(psi_cpu) ** 2
            geom = analyse_psi_geometry(psi_cpu, L=L, expected_nodes=3)
            history.append(
                {
                    "case_id": f"{mode}_dt_{dt:g}_T_{physical_time:g}",
                    "group": "multistep",
                    "step": int(step + 1),
                    "t_phys": float((step + 1) * dt),
                    "mode": mode,
                    "dt": float(dt),
                    "diagnostic_norm": norms["diagnostic_norm"],
                    "physical_grid_norm": norms["physical_grid_norm"],
                    "diagnostic_norm_fractional_change": float(
                        (norms["diagnostic_norm"] - initial_norms["diagnostic_norm"]) / (abs(initial_norms["diagnostic_norm"]) + 1e-30)
                    ),
                    "rho_max": float(np.max(rho)),
                    "rho_max_fractional_change": float(
                        (float(np.max(rho)) - float(np.max(initial_rho))) / (abs(float(np.max(initial_rho))) + 1e-30)
                    ),
                    "node_count": geom.get("node_count"),
                    "profile_overlap": profile_overlap(initial_rho, rho),
                    "high_k_fraction": high_k_fraction(cp.asnumpy(psi_k), cp.asnumpy(solver.reference_dealias_mask).astype(bool)),
                    "recombination_mask_removed_fraction": float(getattr(solver, "last_recombination_mask_removed_fraction", 0.0)),
                }
            )
        if time.time() - t0 > RUNTIME_CASE_WARN_SEC:
            fail_reason = f"runtime guard exceeded {RUNTIME_CASE_WARN_SEC:g}s"
            break
    final = history[-1] if history else {}
    row = {
        "case_id": f"{mode}_dt_{dt:g}_T_{physical_time:g}",
        "group": "multistep",
        "N": int(N),
        "L": float(L),
        "dt": float(dt),
        "steps": int(steps),
        "mode": mode,
        "finite": not fail_reason,
        "fail_reason": fail_reason,
        "diagnostic_norm": final.get("diagnostic_norm"),
        "physical_grid_norm": final.get("physical_grid_norm"),
        "initial_diagnostic_norm": initial_norms["diagnostic_norm"],
        "initial_physical_grid_norm": initial_norms["physical_grid_norm"],
        "fractional_norm_defect": final.get("diagnostic_norm_fractional_change"),
        "rho_max": final.get("rho_max"),
        "rho_max_fractional_change": final.get("rho_max_fractional_change"),
        "node_count": final.get("node_count"),
        "profile_overlap": final.get("profile_overlap"),
        "high_k_fraction": final.get("high_k_fraction"),
        "recombination_mask_removed_fraction": final.get("recombination_mask_removed_fraction"),
        "wallclock_sec": round(time.time() - t0, 3),
    }
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return row, history


def loglog_slope_dt(rows: list[dict[str, Any]]) -> float | None:
    pts = []
    for row in rows:
        val = abs(float(row.get("fractional_norm_defect") or 0.0))
        dt = float(row.get("dt") or 0.0)
        if dt > 0 and val > 0:
            pts.append((dt, val))
    if len(pts) < 2:
        return None
    xs = np.log([p[0] for p in pts])
    ys = np.log([p[1] for p in pts])
    return float(np.polyfit(xs, ys, 1)[0])


def split_step_feasibility_note() -> tuple[str, str]:
    src = (ROOT / "solver" / "core.py").read_text(encoding="utf-8")
    has_cov_laplacian = "calculate_cov_laplacian_fused" in src or "lap_cov" in src
    has_lap_flat = "lap_flat" in src
    if has_cov_laplacian or has_lap_flat:
        return (
            "split_step_not_supported_by_rhs_structure",
            "The nonlinear RHS uses covariant/flat Laplacian terms and is not safely expressible as dpsi/dt = i*F(rho,geometry)*psi with real multiplicative F.",
        )
    return (
        "split_step_ambiguous",
        "Could not prove a pure real multiplicative phase structure; no split-stepper was implemented.",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_plots(paths: dict[str, Path], rows: list[dict[str, Any]], history: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    one_step = [r for r in rows if r.get("group") == "one_step"]
    if one_step:
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
        xs = [float(r["dt"]) for r in one_step]
        ys = [abs(float(r.get("fractional_norm_defect") or 0.0)) for r in one_step]
        ax.loglog(xs, ys, marker="o")
        ax.set_xlabel("dt")
        ax.set_ylabel("|one-step fractional norm defect|")
        ax.grid(True, alpha=0.25, which="both")
        fig.tight_layout()
        fig.savefig(paths["one_step_plot"])
        plt.close(fig)

    by_case: dict[str, list[dict[str, Any]]] = {}
    for item in history:
        by_case.setdefault(str(item["case_id"]), []).append(item)
    for path_key, metric, ylabel in (
        ("norm_plot", "diagnostic_norm_fractional_change", "diagnostic norm fractional change"),
        ("rho_plot", "rho_max_fractional_change", "rho_max fractional change"),
    ):
        fig, ax = plt.subplots(figsize=(9, 4.8), dpi=120)
        for case_id, items in by_case.items():
            sorted_items = sorted(items, key=lambda x: x["t_phys"])
            ax.plot([x["t_phys"] for x in sorted_items], [x.get(metric, np.nan) for x in sorted_items], label=case_id, linewidth=1)
        ax.set_xlabel("physical time")
        ax.set_ylabel(ylabel)
        if by_case:
            ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(paths[path_key])
        plt.close(fig)


def write_report(
    path: Path,
    *,
    snapshot: dict[str, Any],
    rows: list[dict[str, Any]],
    classification: dict[str, Any],
    native_note: str,
    split_note: tuple[str, str],
    stopped_after: str | None,
    one_step_slope: float | None,
) -> None:
    lines = [
        "# Conservative C2 Nonlinear Stepper Contract Audit",
        "",
        "Standalone CuPy-only diagnostic. No production solver, worker, Hunter, validation, config, or JAX scout reference files were modified.",
        "",
        "## Summary",
        "",
        f"- Primary classification: `{classification['primary_classification']}`",
        f"- Secondary flags: `{', '.join(classification['secondary_flags']) if classification['secondary_flags'] else 'none'}`",
        f"- Final decision: `{classification['final_decision']}`",
        f"- Stopped after: `{stopped_after or 'completed_available_audit'}`",
        f"- Native C2 profile: `{native_note}`",
        f"- One-step norm-defect dt slope: `{one_step_slope}`",
        "",
        "No stability claim is made. Geometry persistence is separate from invariant preservation. No longer geometry campaign was run. No amplitude normalization was applied. No JAX was used or installed.",
        "",
        "## Contract Snapshot",
        "",
        f"- RHS boundary: `{snapshot.get('rhs_boundary')}`",
        f"- Linear operator max abs diff: `{snapshot.get('linear_operator_max_abs_diff')}`",
        f"- param_eta inactive max abs diff: `{snapshot.get('param_eta_inactive_max_abs_diff')}`",
        f"- Source contract ok: `{snapshot.get('source_contract', {}).get('source_contract_ok')}`",
        "",
        "## Split-Step Feasibility",
        "",
        f"- `{split_note[0]}`: {split_note[1]}",
        "",
        "## Results",
        "",
        "| case | group | mode | dt | steps | norm defect/flux | rho max change | status | finite |",
        "|---|---|---|---:|---:|---:|---:|---|:---:|",
    ]
    for row in rows:
        norm_value = row.get("fractional_flux_raw", row.get("fractional_norm_defect", ""))
        lines.append(
            f"| {row.get('case_id')} | {row.get('group')} | {row.get('mode')} | {row.get('dt')} | {row.get('steps')} | "
            f"{norm_value} | {row.get('rho_max_fractional_change', '')} | {row.get('rhs_flux_status', row.get('fail_reason', ''))} | {row.get('finite')} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_stepper_contract_audit(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = stepper_audit_paths(out_dir)
    dt_values = parse_dt_list(args.dt_list)
    rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    secondary_flags: list[str] = ["geometry_persistence_short_horizon", "long_campaign_not_justified", "explicit_rk4_not_conservative_benchmark"]
    stopped_after: str | None = None

    snapshot, contract_ok, contract_flags = build_contract_snapshot(args.N, args.L, args.dt, args.require_gpu_name)
    secondary_flags.extend(contract_flags)
    paths["contract_snapshot"].write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    split_note = split_step_feasibility_note()
    secondary_flags.append(split_note[0])
    native_profile, native_note = find_native_profile_note()
    native_profile_tested = native_profile is not None
    if not native_profile_tested:
        secondary_flags.append("profile_mismatch_untested")

    if "CONTRACT_INVARIANT_AMBIGUOUS" in contract_flags:
        classification = {
            "primary_classification": "unclear",
            "secondary_flags": list(dict.fromkeys(secondary_flags)),
            "final_decision": "UNCLEAR_BLOCKED",
            "contract_stop": "CONTRACT_INVARIANT_AMBIGUOUS",
        }
        write_rhs_flux_report(paths["rhs_flux_report"], [], False, native_note)
        write_csv(paths["csv"], rows)
        paths["history"].write_text(json.dumps(history, indent=2), encoding="utf-8")
        write_report(paths["report"], snapshot=snapshot, rows=rows, classification=classification, native_note=native_note, split_note=split_note, stopped_after="contract_snapshot", one_step_slope=None)
        return {"classification": classification, "paths": {k: str(v) for k, v in paths.items()}}

    if not contract_ok:
        classification = classify_audit_result(
            contract_ok=False,
            rhs_flux_hard_stop=False,
            timestep_error=False,
            material_norm_change=False,
            profile_mismatch_evidence=False,
            native_profile_tested=native_profile_tested,
            total_norm_conserved=False,
            rho_or_profile_changed=False,
            secondary_flags=secondary_flags,
        )
        write_rhs_flux_report(paths["rhs_flux_report"], [], False, native_note)
        write_csv(paths["csv"], rows)
        paths["history"].write_text(json.dumps(history, indent=2), encoding="utf-8")
        write_report(paths["report"], snapshot=snapshot, rows=rows, classification=classification, native_note=native_note, split_note=split_note, stopped_after="contract_snapshot", one_step_slope=None)
        return {"classification": classification, "paths": {k: str(v) for k, v in paths.items()}}

    rhs_rows: list[dict[str, Any]] = []
    for spec in rhs_flux_probe_specs(args.spacing):
        row = run_rhs_flux_probe(spec, N=args.N, L=args.L, dt=args.dt, spacing=args.spacing)
        rhs_rows.append(row)
        rows.append(row)
    if native_profile is not None:
        row = run_native_rhs_flux_probe(native_profile, native_note, L=args.L, dt=args.dt)
        rhs_rows.append(row)
        rows.append(row)
    rhs_hard_stop = any(str(r.get("rhs_flux_status")) == "hard_stop" for r in rhs_rows)
    write_rhs_flux_report(paths["rhs_flux_report"], rhs_rows, rhs_hard_stop, native_note)
    if rhs_hard_stop:
        classification = classify_audit_result(
            contract_ok=True,
            rhs_flux_hard_stop=True,
            timestep_error=False,
            material_norm_change=True,
            profile_mismatch_evidence=False,
            native_profile_tested=native_profile_tested,
            total_norm_conserved=False,
            rho_or_profile_changed=False,
            secondary_flags=secondary_flags,
        )
        write_csv(paths["csv"], rows)
        paths["history"].write_text(json.dumps(history, indent=2), encoding="utf-8")
        write_report(paths["report"], snapshot=snapshot, rows=rows, classification=classification, native_note=native_note, split_note=split_note, stopped_after="rhs_flux_hard_stop", one_step_slope=None)
        return {"classification": classification, "paths": {k: str(v) for k, v in paths.items()}}

    one_step_rows = []
    for dt in dt_values:
        row = run_one_step_case(args.N, args.L, dt, args.spacing)
        rows.append(row)
        one_step_rows.append(row)
        if float(row.get("wallclock_sec") or 0.0) > RUNTIME_CASE_WARN_SEC or not row.get("finite", True):
            secondary_flags.append("runtime_guard_triggered")
            stopped_after = "one_step_runtime_guard"
            break
    one_step_slope = loglog_slope_dt(one_step_rows)

    if stopped_after is None:
        modes = ("etdrk4_current", "etdrk4_no_mask", "rk4_full_rhs", "rk4_nonlinear_only")
        for dt in dt_values:
            for mode in modes:
                physical_time = args.physical_time if mode in ("etdrk4_current", "etdrk4_no_mask") else args.short_time
                row, hist = run_multistep_case(args.N, args.L, dt, args.spacing, physical_time, mode, sample_count=args.sample_count)
                rows.append(row)
                history.extend(hist)
                if float(row.get("wallclock_sec") or 0.0) > RUNTIME_CASE_WARN_SEC or not row.get("finite", True):
                    secondary_flags.append("runtime_guard_triggered")
                    stopped_after = "multistep_runtime_guard"
                    break
            if stopped_after is not None:
                break

    etdrk_rows = [r for r in rows if r.get("group") == "multistep" and r.get("mode") == "etdrk4_current"]
    timestep_error = False
    if len(etdrk_rows) >= 2:
        sorted_rows = sorted(etdrk_rows, key=lambda r: float(r["dt"]), reverse=True)
        losses = [abs(float(r.get("fractional_norm_defect") or 0.0)) for r in sorted_rows]
        timestep_error = all(losses[i + 1] < losses[i] * 0.9 for i in range(len(losses) - 1))
    material_norm_change = any(abs(float(r.get("fractional_norm_defect") or 0.0)) > MATERIAL_NORM_CHANGE_THRESHOLD for r in rows if r.get("fractional_norm_defect") is not None)
    total_norm_conserved = bool(etdrk_rows) and not material_norm_change
    rho_or_profile_changed = any(abs(float(r.get("rho_max_fractional_change") or 0.0)) > 1e-3 for r in rows if r.get("rho_max_fractional_change") is not None)
    classification = classify_audit_result(
        contract_ok=True,
        rhs_flux_hard_stop=False,
        timestep_error=timestep_error,
        material_norm_change=material_norm_change,
        profile_mismatch_evidence=False,
        native_profile_tested=native_profile_tested,
        total_norm_conserved=total_norm_conserved,
        rho_or_profile_changed=rho_or_profile_changed,
        secondary_flags=secondary_flags,
    )
    write_csv(paths["csv"], rows)
    paths["history"].write_text(json.dumps(history, indent=2), encoding="utf-8")
    write_plots(paths, rows, history)
    write_report(paths["report"], snapshot=snapshot, rows=rows, classification=classification, native_note=native_note, split_note=split_note, stopped_after=stopped_after, one_step_slope=one_step_slope)
    return {"classification": classification, "paths": {k: str(v) for k, v in paths.items()}}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Conservative C2 nonlinear stepper contract audit")
    parser.add_argument("--out", type=Path, default=DEFAULT_AUDIT_OUT)
    parser.add_argument("--N", type=int, default=48)
    parser.add_argument("--L", type=float, default=DEFAULT_L)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--spacing", type=float, default=0.45)
    parser.add_argument("--physical-time", type=float, default=4.0)
    parser.add_argument("--short-time", type=float, default=0.5)
    parser.add_argument("--dt-list", default="0.001,0.0005,0.00025,0.000125")
    parser.add_argument("--sample-count", type=int, default=40)
    parser.add_argument("--require-gpu-name", default="NVIDIA GeForce GTX 1080")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_stepper_contract_audit(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
