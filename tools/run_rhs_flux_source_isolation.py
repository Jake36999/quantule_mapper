"""Conservative C2 trajectory RHS-flux source isolation.

Diagnostic-only CuPy batch. No production solver/reference files are modified.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.conservative_geometry_campaign import analyse_psi_geometry, high_k_fraction, profile_overlap, steps_for_physical_time  # noqa: E402
from tools.conservative_rk4_stepper_diagnostic import ConservativeC2RK4Stepper, run_safety_checkpoint, triangle_psi0  # noqa: E402
from tools.conservative_stepper_contract_audit import norm_conventions  # noqa: E402
from tools.run_rk4_integrity_batch import sample_steps_for_times  # noqa: E402


DEFAULT_OUT = ROOT / "quantule_viz" / "outputs" / "conservative_geometry_campaign" / "rhs_flux_source_isolation"
FINAL_DECISIONS = {
    "RHS_NONLINEAR_ALGEBRAIC_FLUX_CONFIRMED",
    "RHS_FLUX_MEASUREMENT_CONVENTION_FAIL",
    "RHS_FLUX_RECONSTRUCTION_FAIL",
    "RHS_LINEAR_TERM_SOURCE",
    "RHS_FLUX_SOURCE_UNCLEAR",
}


def physical_flux_raw(psi_phys: np.ndarray, rhs_phys: np.ndarray) -> float:
    return float(2.0 * np.real(np.sum(np.conj(psi_phys) * rhs_phys, dtype=np.complex128)))


def spectral_flux_raw(psi_k: np.ndarray, rhs_k: np.ndarray) -> float:
    n_total = int(np.prod(np.asarray(psi_k).shape))
    return float(2.0 * np.real(np.sum(np.conj(psi_k) * rhs_k, dtype=np.complex128)) / float(n_total))


def fractional(raw_flux: float, diagnostic_norm: float) -> float:
    return float(raw_flux / (abs(float(diagnostic_norm)) + 1e-30))


def flux_status(value: float) -> str:
    mag = abs(float(value))
    if mag <= 1e-10:
        return "numerical_zero"
    if mag > 1e-6:
        return "fail"
    if mag > 1e-8:
        return "warning"
    return "small_nonzero"


def build_manifest() -> dict[str, Any]:
    return {
        "regime": "conservative_c2_rhs_flux_source_isolation",
        "N": 48,
        "L": 10.0,
        "dt": 0.001,
        "physical_time": 1.0,
        "spacing": 0.45,
        "sample_times": [0.0, 0.25, 0.5, 0.75, 1.0],
        "rules": [
            "CuPy only",
            "diagnostic only",
            "no production/reference modifications",
            "no amplitude normalization",
            "no stability claim",
        ],
    }


def classify_source(
    *,
    max_physical_spectral_diff: float,
    max_roundtrip_diff: float,
    max_linear_flux: float,
    max_full_flux: float,
    max_nonlinear_flux: float,
) -> dict[str, Any]:
    if abs(max_physical_spectral_diff) > 1e-10:
        decision = "RHS_FLUX_MEASUREMENT_CONVENTION_FAIL"
    elif abs(max_roundtrip_diff) > 1e-10:
        decision = "RHS_FLUX_RECONSTRUCTION_FAIL"
    elif abs(max_linear_flux) > 1e-8 and abs(max_linear_flux) >= abs(max_nonlinear_flux) * 0.5:
        decision = "RHS_LINEAR_TERM_SOURCE"
    elif abs(max_full_flux) > 1e-6 and abs(max_nonlinear_flux) > 1e-6:
        decision = "RHS_NONLINEAR_ALGEBRAIC_FLUX_CONFIRMED"
    else:
        decision = "RHS_FLUX_SOURCE_UNCLEAR"
    return {"final_decision": decision}


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


def component_fluxes(stepper: ConservativeC2RK4Stepper, psi_k: Any, initial_rho: np.ndarray, L: float) -> dict[str, Any]:
    import cupy as cp

    psi_phys = stepper.to_physical(psi_k)
    nonlinear_k = stepper.solver.N_op(psi_k)
    linear_k = stepper.solver.L_k * psi_k
    full_k = linear_k + nonlinear_k
    nonlinear_phys = stepper.solver.ifft_single(nonlinear_k)
    linear_phys = stepper.solver.ifft_single(linear_k)
    full_phys = stepper.solver.ifft_single(full_k)

    psi_cpu = cp.asnumpy(psi_phys)
    psi_k_cpu = cp.asnumpy(psi_k)
    nonlinear_k_cpu = cp.asnumpy(nonlinear_k)
    linear_k_cpu = cp.asnumpy(linear_k)
    full_k_cpu = cp.asnumpy(full_k)
    nonlinear_phys_cpu = cp.asnumpy(nonlinear_phys)
    linear_phys_cpu = cp.asnumpy(linear_phys)
    full_phys_cpu = cp.asnumpy(full_phys)
    norms = norm_conventions(psi_cpu, L=L)
    diagnostic_norm = norms["diagnostic_norm"]

    nonlinear_raw_phys = physical_flux_raw(psi_cpu, nonlinear_phys_cpu)
    linear_raw_phys = physical_flux_raw(psi_cpu, linear_phys_cpu)
    full_raw_phys = physical_flux_raw(psi_cpu, full_phys_cpu)
    nonlinear_raw_spec = spectral_flux_raw(psi_k_cpu, nonlinear_k_cpu)
    linear_raw_spec = spectral_flux_raw(psi_k_cpu, linear_k_cpu)
    full_raw_spec = spectral_flux_raw(psi_k_cpu, full_k_cpu)

    # Reconstruction check: physical state -> FFT -> RHS/flux.
    psi_k_roundtrip = stepper.solver.fft_single(psi_phys)
    roundtrip_nonlinear = stepper.solver.N_op(psi_k_roundtrip)
    roundtrip_flux = spectral_flux_raw(cp.asnumpy(psi_k_roundtrip), cp.asnumpy(roundtrip_nonlinear))
    roundtrip_frac = fractional(roundtrip_flux, diagnostic_norm)
    nonlinear_frac = fractional(nonlinear_raw_phys, diagnostic_norm)

    rho = np.abs(psi_cpu) ** 2
    geom = analyse_psi_geometry(psi_cpu, L=L, expected_nodes=3)
    return {
        "diagnostic_norm": diagnostic_norm,
        "physical_grid_norm": norms["physical_grid_norm"],
        "rho_max": float(np.max(rho)),
        "node_count": geom.get("node_count"),
        "profile_overlap": profile_overlap(initial_rho, rho),
        "high_k_fraction": high_k_fraction(psi_k_cpu, cp.asnumpy(stepper.solver.reference_dealias_mask).astype(bool)),
        "nonlinear_fractional_flux_physical": nonlinear_frac,
        "linear_fractional_flux_physical": fractional(linear_raw_phys, diagnostic_norm),
        "full_fractional_flux_physical": fractional(full_raw_phys, diagnostic_norm),
        "nonlinear_fractional_flux_spectral": fractional(nonlinear_raw_spec, diagnostic_norm),
        "linear_fractional_flux_spectral": fractional(linear_raw_spec, diagnostic_norm),
        "full_fractional_flux_spectral": fractional(full_raw_spec, diagnostic_norm),
        "nonlinear_physical_spectral_abs_diff": abs(fractional(nonlinear_raw_phys - nonlinear_raw_spec, diagnostic_norm)),
        "linear_physical_spectral_abs_diff": abs(fractional(linear_raw_phys - linear_raw_spec, diagnostic_norm)),
        "full_physical_spectral_abs_diff": abs(fractional(full_raw_phys - full_raw_spec, diagnostic_norm)),
        "roundtrip_nonlinear_fractional_flux": roundtrip_frac,
        "roundtrip_abs_diff": abs(roundtrip_frac - nonlinear_frac),
        "nonlinear_status": flux_status(nonlinear_frac),
        "full_status": flux_status(fractional(full_raw_phys, diagnostic_norm)),
    }


def run_isolation(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    safety = run_safety_checkpoint(out_dir, args.require_gpu_name)
    manifest = build_manifest()
    manifest["max_wallclock_minutes"] = float(args.max_wallclock_minutes)
    (out_dir / "rhs_flux_source_isolation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    t0 = time.time()

    N = int(manifest["N"])
    L = float(manifest["L"])
    dt = float(manifest["dt"])
    physical_time = float(manifest["physical_time"])
    steps = steps_for_physical_time(physical_time, dt)
    sample_steps = sample_steps_for_times(steps, dt, list(manifest["sample_times"]))
    stepper = ConservativeC2RK4Stepper(N, L, dt)
    psi0 = triangle_psi0(N, L, float(manifest["spacing"]))
    psi_k, psi_initial = stepper.project_psi0(psi0)
    import cupy as cp

    initial_rho = np.abs(cp.asnumpy(psi_initial)) ** 2
    rows: list[dict[str, Any]] = []

    def record(step: int, psi_k_current: Any) -> None:
        values = component_fluxes(stepper, psi_k_current, initial_rho, L=L)
        rows.append({"step": int(step), "t_phys": float(step * dt), **values})

    sample_set = set(sample_steps)
    record(0, psi_k)
    for step in range(1, steps + 1):
        psi_k = stepper.step(psi_k)
        if step in sample_set:
            record(step, psi_k)
        if time.time() - t0 > float(args.max_wallclock_minutes) * 60.0:
            break
    if rows[-1]["step"] != steps:
        record(steps, psi_k)

    max_physical_spectral_diff = max(float(r["full_physical_spectral_abs_diff"]) for r in rows)
    max_roundtrip_diff = max(float(r["roundtrip_abs_diff"]) for r in rows)
    max_linear_flux = max(abs(float(r["linear_fractional_flux_physical"])) for r in rows)
    max_full_flux = max(abs(float(r["full_fractional_flux_physical"])) for r in rows)
    max_nonlinear_flux = max(abs(float(r["nonlinear_fractional_flux_physical"])) for r in rows)
    classification = classify_source(
        max_physical_spectral_diff=max_physical_spectral_diff,
        max_roundtrip_diff=max_roundtrip_diff,
        max_linear_flux=max_linear_flux,
        max_full_flux=max_full_flux,
        max_nonlinear_flux=max_nonlinear_flux,
    )
    write_csv(out_dir / "rhs_flux_source_isolation_results.csv", rows)
    report_lines = [
        "# Conservative C2 RHS Flux Source Isolation",
        "",
        f"Final decision: `{classification['final_decision']}`",
        f"Wallclock seconds: `{time.time() - t0:.3f}`",
        "",
        "## Maxima",
        "",
        f"- max physical/spectral full flux diff: `{max_physical_spectral_diff}`",
        f"- max reconstruction roundtrip diff: `{max_roundtrip_diff}`",
        f"- max linear flux: `{max_linear_flux}`",
        f"- max nonlinear flux: `{max_nonlinear_flux}`",
        f"- max full flux: `{max_full_flux}`",
        "",
        "## Samples",
        "",
        "| t | nonlinear flux | linear flux | full flux | spectral diff | roundtrip diff | status |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        report_lines.append(
            f"| {row['t_phys']} | {row['nonlinear_fractional_flux_physical']} | {row['linear_fractional_flux_physical']} | "
            f"{row['full_fractional_flux_physical']} | {row['full_physical_spectral_abs_diff']} | {row['roundtrip_abs_diff']} | {row['full_status']} |"
        )
    report_lines.extend(
        [
            "",
            "No stability claim is made. This is diagnostic-only and did not run geometry campaigns or N64 replay.",
        ]
    )
    (out_dir / "rhs_flux_source_isolation_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    final = {
        "final_decision": classification["final_decision"],
        "out_dir": str(out_dir),
        "wallclock_sec": time.time() - t0,
        "safety": safety,
    }
    (out_dir / "rhs_flux_source_isolation_summary.json").write_text(json.dumps(final, indent=2, default=str), encoding="utf-8")
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return final


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Conservative C2 RHS flux source isolation")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-wallclock-minutes", type=float, default=60.0)
    parser.add_argument("--require-gpu-name", default="NVIDIA GeForce GTX 1080")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_isolation(args)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
