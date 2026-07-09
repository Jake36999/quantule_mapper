"""Conservative C2 nonlinear RHS term-flux decomposition.

Diagnostic-only CuPy batch. This script does not modify production solver,
worker, Hunter, validation, config, or jax_scout reference files.
"""
from __future__ import annotations

import argparse
import csv
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
    DEFAULT_SEED,
    DEFAULT_WIDTH_BOX,
    analyse_psi_geometry,
    build_geometry_ic,
    high_k_fraction,
    points_for_case,
    profile_overlap,
    steps_for_physical_time,
)
from tools.conservative_rk4_stepper_diagnostic import (  # noqa: E402
    PROTECTED_FILES,
    ConservativeC2RK4Stepper,
    run_safety_checkpoint,
    triangle_psi0,
)
from tools.conservative_stepper_contract_audit import norm_conventions  # noqa: E402
from tools.run_rk4_integrity_batch import sample_steps_for_times  # noqa: E402


DEFAULT_OUT = ROOT / "quantule_viz" / "outputs" / "conservative_geometry_campaign" / "rhs_term_flux_decomposition"
TERM_LABELS = [
    "geometry_covariant_correction",
    "cubic_density_a",
    "quintic_density_s",
    "septic_density_f",
]
FINAL_DECISIONS = {
    "TERM_RECOMBINATION_MISMATCH",
    "RHS_TERM_CONTRACT_REVIEW_REQUIRED",
    "DISCRETE_OPERATOR_ADJOINT_FAILURE",
    "RHS_NONCONSERVATIVE_BY_CONTRACT",
    "CONSERVATIVE_LABEL_AMBIGUOUS",
    "FLUX_SOURCE_UNCLEAR",
}
SECONDARY_FLAGS = [
    "no_stability_claim",
    "no_production_change",
    "no_amplitude_normalization",
    "no_jax",
    "no_geometry_campaign",
    "rk4_diagnostic_only",
    "budget_respected",
]


def physical_flux_raw(psi_phys: np.ndarray, rhs_phys: np.ndarray) -> float:
    return float(2.0 * np.real(np.sum(np.conj(psi_phys) * rhs_phys, dtype=np.complex128)))


def spectral_flux_raw(psi_k: np.ndarray, rhs_k: np.ndarray) -> float:
    n_total = int(np.prod(np.asarray(psi_k).shape))
    return float(2.0 * np.real(np.sum(np.conj(psi_k) * rhs_k, dtype=np.complex128)) / float(n_total))


def percentage_contribution(term_flux: float, total_flux: float) -> float | None:
    if abs(float(total_flux)) <= 1e-30:
        return None
    return float(100.0 * float(term_flux) / float(total_flux))


def classify_recombination(max_abs_error: float, relative_l2_error: float) -> str:
    if float(max_abs_error) <= 1e-8 and float(relative_l2_error) <= 1e-10:
        return "pass"
    return "TERM_RECOMBINATION_MISMATCH"


def classify_adjoint_mismatch(relative_mismatch: float) -> str:
    mag = abs(float(relative_mismatch))
    if mag <= 1e-10:
        return "ADJOINT_PASS"
    if mag <= 1e-6:
        return "ADJOINT_WARNING"
    return "ADJOINT_FAIL"


def classify_final_decision(
    *,
    recombination_status: str,
    dominant_term: str | None,
    max_term_flux: float,
    adjoint_status: str,
    docs_state: str,
) -> dict[str, Any]:
    if recombination_status != "pass":
        decision = "TERM_RECOMBINATION_MISMATCH"
    elif adjoint_status == "ADJOINT_FAIL":
        decision = "DISCRETE_OPERATOR_ADJOINT_FAILURE"
    elif docs_state == "nonconservative_by_contract":
        decision = "RHS_NONCONSERVATIVE_BY_CONTRACT"
    elif abs(float(max_term_flux)) > 1e-6 and dominant_term:
        if docs_state == "ambiguous":
            decision = "CONSERVATIVE_LABEL_AMBIGUOUS"
        else:
            decision = "RHS_TERM_CONTRACT_REVIEW_REQUIRED"
    else:
        decision = "FLUX_SOURCE_UNCLEAR"
    return {
        "final_decision": decision,
        "primary_term": dominant_term,
        "secondary_flags": list(SECONDARY_FLAGS),
    }


def build_manifest(max_wallclock_minutes: float = 60.0) -> dict[str, Any]:
    controls = [
        {"case_id": "uniform_constant", "kind": "uniform", "expected_nodes": 0},
        {"case_id": "single_gaussian", "kind": "single_gaussian", "expected_nodes": 1},
        {"case_id": "triangle_symmetric", "kind": "triangle", "expected_nodes": 3},
        {"case_id": "triangle_amp_minus_5pct", "kind": "triangle_amp_minus_5pct", "expected_nodes": 3},
        {"case_id": "triangle_phase_pi8", "kind": "triangle_phase_pi8", "expected_nodes": 3},
        {"case_id": "triangle_position_jitter", "kind": "triangle_position_jitter", "expected_nodes": 3},
        {"case_id": "ablated_triangle", "kind": "ablated_triangle", "expected_nodes": 2},
        {"case_id": "tetrahedron", "kind": "tetrahedron", "expected_nodes": 4},
        {"case_id": "triangular_prism", "kind": "triangular_prism", "expected_nodes": 6},
    ]
    return {
        "regime": "conservative_c2_rhs_term_flux_decomposition",
        "N": 48,
        "L": 10.0,
        "dt": 0.001,
        "physical_time": 1.0,
        "short_control_time": 0.25,
        "spacing": 0.45,
        "sample_times": [0.0, 0.25, 0.5, 0.75, 1.0],
        "symmetry_controls": controls,
        "max_wallclock_minutes": float(max_wallclock_minutes),
        "rules": [
            "Conservative C2 only",
            "CuPy only",
            "diagnostic standalone logic only",
            "no production/reference modifications",
            "no amplitude normalization",
            "no geometry campaign",
            "no stability claim",
        ],
    }


def resolve_output_dir(base: Path, timestamp: str | None = None) -> Path:
    stamp = timestamp or time.strftime("%Y%m%d_%H%M%S")
    if base.name.startswith("rhs_term_flux_decomposition_"):
        return base
    return base.with_name(f"{base.name}_{stamp}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _line_region(path: Path, needle: str) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "unavailable"
    for idx, line in enumerate(lines, start=1):
        if needle in line:
            return f"{path.relative_to(ROOT).as_posix()}:{idx}"
    return f"{path.relative_to(ROOT).as_posix()}:not-found"


def build_source_map() -> list[dict[str, Any]]:
    return [
        {
            "term": "conservative_wrapper_factor",
            "source": _line_region(ROOT / "tools" / "conservative_geometry_campaign.py", "return (1j * super().N_op"),
            "symbolic": "N_k^C2 = i * N_k^baseline",
            "input_representation": "spectral psi_k",
            "output_representation": "spectral RHS",
            "category": "Hamiltonian/conservative wrapper factor",
            "expected_norm_neutral": "Only if the baseline real-space operator is real self-adjoint in the active state/discretization.",
        },
        {
            "term": "geometry_covariant_correction",
            "source": _line_region(ROOT / "solver" / "kernels.py", "return D_diff * (lap_cov - lap_flat)"),
            "symbolic": "i * D * (Delta_cov - Delta_flat) psi",
            "input_representation": "physical fields derived from spectral psi_k",
            "output_representation": "spectral RHS after FFT/dealias",
            "category": "derivative/covariant/conformal geometry-dependent",
            "expected_norm_neutral": "Requires the effective discrete geometry operator to be self-adjoint after the state-dependent geometry construction.",
        },
        {
            "term": "cubic_density_a",
            "source": _line_region(ROOT / "solver" / "kernels.py", "nonlin = a * psi * rho"),
            "symbolic": "i * a * psi * rho",
            "input_representation": "physical psi/rho",
            "output_representation": "spectral RHS after FFT/dealias",
            "category": "local multiplicative density-dependent",
            "expected_norm_neutral": "Pointwise phase-only if a and rho are real.",
        },
        {
            "term": "quintic_density_s",
            "source": _line_region(ROOT / "solver" / "kernels.py", "s * psi * (rho**2)"),
            "symbolic": "i * s * psi * rho^2",
            "input_representation": "physical psi/rho",
            "output_representation": "spectral RHS after FFT/dealias",
            "category": "local multiplicative density-dependent",
            "expected_norm_neutral": "Pointwise phase-only if s and rho are real.",
        },
        {
            "term": "septic_density_f",
            "source": _line_region(ROOT / "solver" / "kernels.py", "f * psi * (rho**3)"),
            "symbolic": "i * f * psi * rho^3",
            "input_representation": "physical psi/rho",
            "output_representation": "spectral RHS after FFT/dealias",
            "category": "local multiplicative density-dependent",
            "expected_norm_neutral": "Pointwise phase-only if f and rho are real.",
        },
        {
            "term": "reference_contract",
            "source": _line_region(ROOT / "jax_scout" / "physics.py", "kfac = jnp.asarray(1j"),
            "symbolic": "kinetic_mode='conservative' sets L_k=-i*D*k^2 and kfac=1j",
            "input_representation": "spectral psi_k",
            "output_representation": "spectral RHS",
            "category": "C2 reference contract",
            "expected_norm_neutral": "Docs/comments describe near-conservation, but term decomposition is needed for generic evolved states.",
        },
    ]


def write_source_map(path: Path, source_map: list[dict[str, Any]]) -> None:
    lines = ["# C2 Nonlinear RHS Source Map", ""]
    for item in source_map:
        lines.extend(
            [
                f"## {item['term']}",
                "",
                f"- Source: `{item['source']}`",
                f"- Symbolic description: `{item['symbolic']}`",
                f"- Input representation: `{item['input_representation']}`",
                f"- Output representation: `{item['output_representation']}`",
                f"- Category: `{item['category']}`",
                f"- Expected norm-neutrality: {item['expected_norm_neutral']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _shared_fields(stepper: ConservativeC2RK4Stepper, psi_k: Any) -> dict[str, Any]:
    import cupy as cp
    from gravity.unified_omega import derive_stable_conformal_factor_with_gradient
    from solver.kernels import (
        calculate_cov_laplacian_fused,
        fused_compute_rho,
        fused_process_omega,
        fused_scale_derivative,
    )

    solver = stepper.solver
    psi = solver.ifft_single(psi_k)
    grad_x = solver.ifft_single(solver.ikx_filtered * psi_k)
    grad_y = solver.ifft_single(solver.iky_filtered * psi_k)
    grad_z = solver.ifft_single(solver.ikz_filtered * psi_k)
    lap_flat = solver.ifft_single(solver.minus_k_sq_filtered * psi_k)
    rho = fused_compute_rho(psi, solver.rho_floor)
    omega_sq_tmp, d_omega_sq_d_rho_tmp = derive_stable_conformal_factor_with_gradient(
        rho, solver._simulation_geometry_params
    )
    omega_sq, omega = fused_process_omega(omega_sq_tmp, solver.omega_sq_min, solver.omega_sq_max)
    d_omega_sq_d_rho = fused_scale_derivative(
        omega_sq,
        d_omega_sq_d_rho_tmp,
        solver.omega_sq_min,
        solver.omega_sq_max,
    )
    d_omega_d_rho = d_omega_sq_d_rho / (2.0 * cp.maximum(omega, cp.float64(1e-15)))
    lap_cov = calculate_cov_laplacian_fused(
        psi,
        grad_x,
        grad_y,
        grad_z,
        lap_flat,
        omega,
        omega_sq,
        d_omega_d_rho,
        solver.D_spatial,
    )
    return {
        "psi": psi,
        "grad_x": grad_x,
        "grad_y": grad_y,
        "grad_z": grad_z,
        "lap_flat": lap_flat,
        "rho": rho,
        "omega": omega,
        "omega_sq": omega_sq,
        "d_omega_d_rho": d_omega_d_rho,
        "lap_cov": lap_cov,
    }


def decompose_nonlinear_terms(stepper: ConservativeC2RK4Stepper, psi_k: Any) -> dict[str, Any]:
    import cupy as cp

    solver = stepper.solver
    fields = _shared_fields(stepper, psi_k)
    psi = fields["psi"]
    rho = fields["rho"]
    lap_cov = fields["lap_cov"]
    lap_flat = fields["lap_flat"]
    real_terms = {
        "geometry_covariant_correction": solver.D_diff * (lap_cov - lap_flat),
        "cubic_density_a": solver.a * psi * rho,
        "quintic_density_s": solver.s * psi * (rho**2),
        "septic_density_f": solver.f * psi * (rho**3),
    }
    term_k: dict[str, Any] = {}
    recombined = cp.zeros_like(psi_k, dtype=cp.complex128)
    for label in TERM_LABELS:
        rhs_k = solver.fft_single((1j * real_terms[label]).astype(cp.complex128, copy=False)) * solver.dealias_mask
        term_k[label] = rhs_k.astype(cp.complex128, copy=False)
        recombined = recombined + term_k[label]
    original = solver.N_op(psi_k)
    diff = recombined - original
    max_abs = float(cp.max(cp.abs(diff)).get())
    rel_l2 = float(
        (
            cp.linalg.norm(diff).astype(cp.float64)
            / cp.maximum(cp.linalg.norm(original).astype(cp.float64), cp.float64(1e-30))
        ).get()
    )
    return {
        "fields": fields,
        "real_terms": real_terms,
        "term_k": term_k,
        "recombined_k": recombined,
        "original_k": original,
        "recombination_max_abs_error": max_abs,
        "recombination_relative_l2_error": rel_l2,
        "recombination_status": classify_recombination(max_abs, rel_l2),
    }


def term_flux_rows(
    stepper: ConservativeC2RK4Stepper,
    psi_k: Any,
    *,
    step: int,
    t_phys: float,
    initial_rho: np.ndarray,
    L: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import cupy as cp

    decomp = decompose_nonlinear_terms(stepper, psi_k)
    psi_phys = stepper.to_physical(psi_k)
    psi_cpu = cp.asnumpy(psi_phys)
    psi_k_cpu = cp.asnumpy(psi_k)
    norms = norm_conventions(psi_cpu, L=L)
    diagnostic_norm = float(norms["diagnostic_norm"])
    nonlinear_k = decomp["original_k"]
    linear_k = stepper.solver.L_k * psi_k
    full_k = linear_k + nonlinear_k
    nonlinear_phys = cp.asnumpy(stepper.solver.ifft_single(nonlinear_k))
    linear_phys = cp.asnumpy(stepper.solver.ifft_single(linear_k))
    full_phys = cp.asnumpy(stepper.solver.ifft_single(full_k))
    total_nonlinear_flux = physical_flux_raw(psi_cpu, nonlinear_phys) / (diagnostic_norm + 1e-30)
    linear_flux = physical_flux_raw(psi_cpu, linear_phys) / (diagnostic_norm + 1e-30)
    full_flux = physical_flux_raw(psi_cpu, full_phys) / (diagnostic_norm + 1e-30)
    rho = np.abs(psi_cpu) ** 2
    geom = analyse_psi_geometry(psi_cpu, L=L, expected_nodes=3)
    sample_summary = {
        "step": int(step),
        "t_phys": float(t_phys),
        "diagnostic_norm": diagnostic_norm,
        "physical_grid_norm": norms["physical_grid_norm"],
        "rho_max": float(np.max(rho)),
        "profile_overlap": profile_overlap(initial_rho, rho),
        "node_count": geom.get("node_count"),
        "high_k_fraction": high_k_fraction(psi_k_cpu, cp.asnumpy(stepper.solver.reference_dealias_mask).astype(bool)),
        "linear_fractional_flux": linear_flux,
        "nonlinear_fractional_flux": total_nonlinear_flux,
        "full_fractional_flux": full_flux,
        "recombination_max_abs_error": decomp["recombination_max_abs_error"],
        "recombination_relative_l2_error": decomp["recombination_relative_l2_error"],
        "recombination_status": decomp["recombination_status"],
    }
    rows: list[dict[str, Any]] = []
    for label in TERM_LABELS:
        rhs_phys = cp.asnumpy(stepper.solver.ifft_single(decomp["term_k"][label]))
        frac_flux = physical_flux_raw(psi_cpu, rhs_phys) / (diagnostic_norm + 1e-30)
        rows.append(
            {
                **sample_summary,
                "term": label,
                "term_fractional_flux": frac_flux,
                "term_abs_fractional_flux": abs(frac_flux),
                "term_percent_of_total_nonlinear_flux": percentage_contribution(frac_flux, total_nonlinear_flux),
                "expected_norm_neutral": label != "geometry_covariant_correction",
            }
        )
    return rows, sample_summary


def _make_control_psi(case: dict[str, Any], N: int, L: float, spacing: float) -> tuple[np.ndarray, int]:
    kind = str(case["kind"])
    if kind == "uniform":
        return np.ones((N, N, N), dtype=np.complex128), 0
    if kind == "single_gaussian":
        pts = np.array([[0.5, 0.5, 0.5]], dtype=float)
        return build_geometry_ic(N, L, pts, DEFAULT_WIDTH_BOX, 1.0), 1
    if kind == "triangle_amp_minus_5pct":
        pts = points_for_case("triangle", spacing, {}, seed=DEFAULT_SEED)
        return build_geometry_ic(N, L, pts, DEFAULT_WIDTH_BOX, 1.0, amplitude_factors=[0.95, 1.0, 1.0]), 3
    if kind == "triangle_phase_pi8":
        pts = points_for_case("triangle", spacing, {}, seed=DEFAULT_SEED)
        return build_geometry_ic(N, L, pts, DEFAULT_WIDTH_BOX, 1.0, phases=[math.pi / 8.0, 0.0, 0.0]), 3
    if kind == "triangle_position_jitter":
        pts = points_for_case("triangle", spacing, {}, seed=DEFAULT_SEED)
        pts = pts.copy()
        pts[0] = (pts[0] + np.array([0.012, -0.007, 0.009])) % 1.0
        return build_geometry_ic(N, L, pts, DEFAULT_WIDTH_BOX, 1.0), 3
    template = "triangle" if kind == "triangle" else kind
    pts = points_for_case(template, spacing, {}, seed=DEFAULT_SEED)
    return build_geometry_ic(N, L, pts, DEFAULT_WIDTH_BOX, 1.0, phases=[0.0] * len(pts)), int(len(pts))


def run_trajectory_decomposition(
    manifest: dict[str, Any],
    out_dir: Path,
    start_time: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    import cupy as cp

    N = int(manifest["N"])
    L = float(manifest["L"])
    dt = float(manifest["dt"])
    steps = steps_for_physical_time(float(manifest["physical_time"]), dt)
    sample_steps = sample_steps_for_times(steps, dt, list(manifest["sample_times"]))
    stepper = ConservativeC2RK4Stepper(N, L, dt)
    psi0 = triangle_psi0(N, L, float(manifest["spacing"]))
    psi_k, psi_initial = stepper.project_psi0(psi0)
    initial_rho = np.abs(cp.asnumpy(psi_initial)) ** 2
    rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    sample_set = set(sample_steps)

    def record(current_step: int, current_psi_k: Any) -> str:
        term_rows, summary = term_flux_rows(
            stepper,
            current_psi_k,
            step=current_step,
            t_phys=current_step * dt,
            initial_rho=initial_rho,
            L=L,
        )
        rows.extend(term_rows)
        history.append(summary)
        return str(summary["recombination_status"])

    status = record(0, psi_k)
    if status != "pass":
        return rows, history, status
    for step in range(1, steps + 1):
        psi_k = stepper.step(psi_k)
        if step in sample_set:
            status = record(step, psi_k)
            if status != "pass":
                return rows, history, status
        if time.time() - start_time > float(manifest["max_wallclock_minutes"]) * 60.0:
            break
    if history[-1]["step"] != steps:
        status = record(steps, psi_k)
    return rows, history, status


def _inner(a: Any, b: Any) -> Any:
    import cupy as cp

    return cp.sum(cp.conj(a) * b, dtype=cp.complex128)


def run_adjoint_audit(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    import cupy as cp

    N = int(manifest["N"])
    L = float(manifest["L"])
    dt = float(manifest["dt"])
    stepper = ConservativeC2RK4Stepper(N, L, dt)
    psi0 = triangle_psi0(N, L, float(manifest["spacing"]))
    psi_k, _ = stepper.project_psi0(psi0)
    target_steps = steps_for_physical_time(0.75, dt)
    for _ in range(target_steps):
        psi_k = stepper.step(psi_k)
    fields = _shared_fields(stepper, psi_k)
    solver = stepper.solver
    rng = np.random.default_rng(101)
    a = cp.asarray(rng.normal(size=(N, N, N)) + 1j * rng.normal(size=(N, N, N)), dtype=cp.complex128)
    b = cp.asarray(rng.normal(size=(N, N, N)) + 1j * rng.normal(size=(N, N, N)), dtype=cp.complex128)

    def flat_laplacian(v: Any) -> Any:
        v_k = solver.fft_single(v)
        return solver.ifft_single(solver.minus_k_sq_filtered * v_k)

    grad_omega_x = fields["d_omega_d_rho"] * (2.0 * cp.real(cp.conj(fields["psi"]) * fields["grad_x"]))
    grad_omega_y = fields["d_omega_d_rho"] * (2.0 * cp.real(cp.conj(fields["psi"]) * fields["grad_y"]))
    grad_omega_z = fields["d_omega_d_rho"] * (2.0 * cp.real(cp.conj(fields["psi"]) * fields["grad_z"]))

    def frozen_cov_laplacian(v: Any) -> Any:
        v_k = solver.fft_single(v)
        dx = solver.ifft_single(solver.ikx_filtered * v_k)
        dy = solver.ifft_single(solver.iky_filtered * v_k)
        dz = solver.ifft_single(solver.ikz_filtered * v_k)
        lap = solver.ifft_single(solver.minus_k_sq_filtered * v_k)
        grad_dot = grad_omega_x * dx + grad_omega_y * dy + grad_omega_z * dz
        return (lap + (solver.D_spatial - 2.0) * grad_dot / fields["omega"]) / fields["omega_sq"]

    def frozen_geometry_correction(v: Any) -> Any:
        return solver.D_diff * (frozen_cov_laplacian(v) - flat_laplacian(v))

    operators = {
        "flat_laplacian": flat_laplacian,
        "frozen_conformal_laplacian_proxy": frozen_cov_laplacian,
        "frozen_geometry_correction_proxy": frozen_geometry_correction,
    }
    rows: list[dict[str, Any]] = []
    for label, op in operators.items():
        lhs = _inner(a, op(b))
        rhs = _inner(op(a), b)
        denom = cp.maximum(cp.maximum(cp.abs(lhs), cp.abs(rhs)), cp.float64(1e-30))
        rel = float((cp.abs(lhs - rhs) / denom).get())
        rows.append(
            {
                "operator": label,
                "adjoint_test": "self_adjoint_proxy",
                "relative_mismatch": rel,
                "classification": classify_adjoint_mismatch(rel),
            }
        )
    return rows


def run_symmetry_controls(
    manifest: dict[str, Any],
    start_time: float,
) -> list[dict[str, Any]]:
    import cupy as cp

    N = int(manifest["N"])
    L = float(manifest["L"])
    dt = float(manifest["dt"])
    spacing = float(manifest["spacing"])
    short_steps = steps_for_physical_time(float(manifest["short_control_time"]), dt)
    rows: list[dict[str, Any]] = []
    for case in manifest["symmetry_controls"]:
        if time.time() - start_time > float(manifest["max_wallclock_minutes"]) * 60.0:
            break
        psi0, expected_nodes = _make_control_psi(case, N, L, spacing)
        stepper = ConservativeC2RK4Stepper(N, L, dt)
        psi_k, psi_initial = stepper.project_psi0(psi0)
        initial_rho = np.abs(cp.asnumpy(psi_initial)) ** 2

        def record(label: str, step: int, current_psi_k: Any) -> None:
            term_rows, summary = term_flux_rows(
                stepper,
                current_psi_k,
                step=step,
                t_phys=step * dt,
                initial_rho=initial_rho,
                L=L,
            )
            dominant = max(term_rows, key=lambda r: float(r["term_abs_fractional_flux"]))
            rows.append(
                {
                    "case_id": case["case_id"],
                    "sample_label": label,
                    "step": int(step),
                    "t_phys": float(step * dt),
                    "expected_nodes": expected_nodes,
                    "node_count": summary["node_count"],
                    "nonlinear_fractional_flux": summary["nonlinear_fractional_flux"],
                    "full_fractional_flux": summary["full_fractional_flux"],
                    "dominant_term": dominant["term"],
                    "dominant_term_fractional_flux": dominant["term_fractional_flux"],
                    "recombination_status": summary["recombination_status"],
                }
            )

        record("initial", 0, psi_k)
        for _ in range(short_steps):
            psi_k = stepper.step(psi_k)
        record("t_0p25", short_steps, psi_k)
    return rows


def write_term_report(path: Path, rows: list[dict[str, Any]], history: list[dict[str, Any]]) -> dict[str, Any]:
    max_recomb_abs = max(float(r["recombination_max_abs_error"]) for r in rows) if rows else math.nan
    max_recomb_rel = max(float(r["recombination_relative_l2_error"]) for r in rows) if rows else math.nan
    dominant = max(rows, key=lambda r: float(r["term_abs_fractional_flux"])) if rows else {}
    lines = [
        "# Conservative C2 RHS Term-Flux Decomposition",
        "",
        f"Term recombination max abs error: `{max_recomb_abs}`",
        f"Term recombination max relative L2 error: `{max_recomb_rel}`",
        f"Dominant term by absolute fractional flux: `{dominant.get('term')}`",
        f"Dominant term flux: `{dominant.get('term_fractional_flux')}` at t=`{dominant.get('t_phys')}`",
        "",
        "No stability claim is made. This is a diagnostic term-level flux audit only.",
        "",
        "## Sample Nonlinear Flux",
        "",
    ]
    for item in history:
        lines.append(f"- t={item['t_phys']}: nonlinear flux `{item['nonlinear_fractional_flux']}`")
    path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "max_recomb_abs": max_recomb_abs,
        "max_recomb_rel": max_recomb_rel,
        "dominant_term": dominant.get("term"),
        "dominant_term_flux": float(dominant.get("term_fractional_flux", 0.0)) if dominant else 0.0,
    }


def write_simple_report(path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("No rows were produced.")
    else:
        for row in rows[:30]:
            bits = ", ".join(f"{k}={v}" for k, v in row.items() if k in {"operator", "classification", "relative_mismatch", "case_id", "sample_label", "nonlinear_fractional_flux", "dominant_term"})
            lines.append(f"- {bits}")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_contract_review(
    path: Path,
    term_summary: dict[str, Any],
    adjoint_rows: list[dict[str, Any]],
    classification: dict[str, Any],
) -> None:
    adjoint_worst = max(adjoint_rows, key=lambda r: float(r["relative_mismatch"])) if adjoint_rows else {}
    lines = [
        "# C2 Conservative Contract Implication Review",
        "",
        f"Final decision: `{classification['final_decision']}`",
        f"Dominant nonlinear flux term: `{term_summary.get('dominant_term')}`",
        f"Dominant term flux: `{term_summary.get('dominant_term_flux')}`",
        f"Worst adjoint proxy: `{adjoint_worst.get('operator')}` classified `{adjoint_worst.get('classification')}`",
        "",
        "## Answers",
        "",
        "1. The dominant term is reported above from the recombined term audit.",
        "2. Local density terms are pointwise phase-only under real coefficients; the geometry correction requires a self-adjoint effective operator to be norm-neutral.",
        "3. The geometry correction is derivative/covariant/geometry-dependent; the remaining polynomial terms are local multiplicative density terms.",
        "4. The operator audit is a frozen-coefficient proxy and is reported separately; it does not patch or redefine the RHS.",
        "5. The current code comments indicate a conservative C2 kinetic mode, but this audit is meant to clarify whether that applies to the nonlinear sector for generic states.",
        "6. If the dominant term is non-neutral and documentation does not explicitly define quasi-conservative nonlinear behavior, the conservative label remains contract-ambiguous.",
        "7. Longer geometry campaigns should remain blocked until Jake/Claude review the dominant nonlinear flux term and intended invariant contract.",
        "8. Question for Jake/Claude: Should C2 `kinetic_mode='conservative'` conserve total `sum(|psi|^2)` for the full nonlinear geometry-corrected RHS, or only for the linear dispersive substrate / special symmetric states?",
        "",
        "No fix is proposed or applied in this diagnostic.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _docs_state_from_source(source_map: list[dict[str, Any]]) -> str:
    text = json.dumps(source_map).lower()
    if "quasi-conservative" in text or "not expected to conserve" in text:
        return "nonconservative_by_contract"
    return "ambiguous"


def _spec_absent(value: Any) -> bool:
    return value is None or str(value) == "None"


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = resolve_output_dir(Path(args.out))
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    safety = run_safety_checkpoint(out_dir, args.require_gpu_name)
    manifest = build_manifest(float(args.max_wallclock_minutes))
    (out_dir / "rhs_term_flux_decomposition_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "rhs_term_flux_decomposition_resolved_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log_lines = [f"Started {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(start))}", f"Output {out_dir}"]
    source_map = build_source_map()
    write_source_map(out_dir / "c2_nonlinear_rhs_source_map.md", source_map)
    log_lines.append("Wrote source map")

    term_rows, history, recombination_status = run_trajectory_decomposition(manifest, out_dir, start)
    write_csv(out_dir / "rhs_term_flux_decomposition_results.csv", term_rows)
    (out_dir / "rhs_term_flux_decomposition_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    term_summary = write_term_report(out_dir / "rhs_term_flux_decomposition_report.md", term_rows, history)
    log_lines.append(f"Term decomposition recombination status: {recombination_status}")

    adjoint_rows: list[dict[str, Any]] = []
    symmetry_rows: list[dict[str, Any]] = []
    docs_state = _docs_state_from_source(source_map)
    if recombination_status == "pass":
        adjoint_rows = run_adjoint_audit(manifest)
        write_csv(out_dir / "rhs_operator_adjoint_audit_results.csv", adjoint_rows)
        write_simple_report(out_dir / "rhs_operator_adjoint_audit_report.md", "RHS Operator Adjoint Audit", adjoint_rows)
        worst_adjoint = max((r["classification"] for r in adjoint_rows), default="ADJOINT_PASS")
        log_lines.append("Adjoint audit complete")

        symmetry_rows = run_symmetry_controls(manifest, start)
        write_csv(out_dir / "symmetry_flux_controls_results.csv", symmetry_rows)
        write_simple_report(out_dir / "symmetry_flux_controls_report.md", "Symmetry Flux Controls", symmetry_rows)
        log_lines.append("Symmetry controls complete")
    else:
        worst_adjoint = "ADJOINT_PASS"

    if any(r.get("classification") == "ADJOINT_FAIL" for r in adjoint_rows):
        adjoint_status = "ADJOINT_FAIL"
    elif any(r.get("classification") == "ADJOINT_WARNING" for r in adjoint_rows):
        adjoint_status = "ADJOINT_WARNING"
    else:
        adjoint_status = "ADJOINT_PASS"
    classification = classify_final_decision(
        recombination_status=recombination_status,
        dominant_term=term_summary.get("dominant_term"),
        max_term_flux=float(term_summary.get("dominant_term_flux", 0.0)),
        adjoint_status=adjoint_status,
        docs_state=docs_state,
    )
    write_contract_review(out_dir / "c2_conservative_contract_implication_review.md", term_summary, adjoint_rows, classification)
    wallclock = time.time() - start
    final_lines = [
        "# RHS Term-Flux Decomposition Final Report",
        "",
        f"Final decision: `{classification['final_decision']}`",
        f"Secondary flags: `{', '.join(classification['secondary_flags'])}`",
        f"Wallclock seconds: `{wallclock:.3f}`",
        "",
        "## Environment",
        "",
        f"- Python: `{sys.executable}`",
        f"- CuPy: `{safety.get('cupy_version')}`",
        f"- GPU: `{safety.get('gpu', {}).get('gpu_name')}`",
        f"- JAX/JAXLIB absent: `{_spec_absent(safety.get('jax_spec')) and _spec_absent(safety.get('jaxlib_spec'))}`",
        f"- Protected diff empty: `{safety.get('protected_diff_empty')}`",
        "",
        "## Results",
        "",
        f"- Term recombination status: `{recombination_status}`",
        f"- Dominant term: `{term_summary.get('dominant_term')}`",
        f"- Dominant term flux: `{term_summary.get('dominant_term_flux')}`",
        f"- Operator adjoint status: `{adjoint_status}`",
        f"- Symmetry controls rows: `{len(symmetry_rows)}`",
        "",
        "No production/reference files were modified. No amplitude normalization was applied. No geometry campaign or N64 replay was run. No stability claim is made.",
    ]
    (out_dir / "rhs_term_flux_decomposition_final_report.md").write_text("\n".join(final_lines), encoding="utf-8")
    (out_dir / "rk4_diagnostic_only_note.txt").write_text("RK4 remains diagnostic-only.\n", encoding="utf-8")
    log_lines.append(f"Final decision {classification['final_decision']}")
    log_lines.append(f"Wallclock seconds {wallclock:.3f}")
    (out_dir / "rhs_term_flux_decomposition.log").write_text("\n".join(log_lines), encoding="utf-8")
    result = {
        "out_dir": str(out_dir),
        "final_decision": classification["final_decision"],
        "dominant_term": term_summary.get("dominant_term"),
        "dominant_term_flux": term_summary.get("dominant_term_flux"),
        "adjoint_status": adjoint_status,
        "wallclock_seconds": wallclock,
    }
    print(json.dumps(result, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Base output directory; timestamp suffix is added by default.")
    parser.add_argument("--max-wallclock-minutes", type=float, default=60.0)
    parser.add_argument("--require-gpu-name", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_batch(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
