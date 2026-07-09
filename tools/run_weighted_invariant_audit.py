"""Conservative C2 weighted invariant audit.

Diagnostic-only CuPy batch. It tests candidate Omega-weighted measures against
the RK4 trajectory and frozen-operator adjoint probes without changing physics.
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

from tools.conservative_geometry_campaign import steps_for_physical_time  # noqa: E402
from tools.conservative_rk4_stepper_diagnostic import (  # noqa: E402
    ConservativeC2RK4Stepper,
    run_safety_checkpoint,
    triangle_psi0,
)
from tools.run_rhs_term_flux_decomposition import (  # noqa: E402
    PROTECTED_FILES,
    _shared_fields,
    classify_adjoint_mismatch,
    decompose_nonlinear_terms,
    physical_flux_raw,
    write_csv,
)
from tools.run_rk4_integrity_batch import sample_steps_for_times  # noqa: E402


DEFAULT_OUT = ROOT / "quantule_viz" / "outputs" / "conservative_geometry_campaign" / "weighted_invariant_audit"
RECOMMENDATIONS = {
    "ORDINARY_NORM_NOT_CONSERVED_WEIGHTED_CANDIDATE_FOUND",
    "NO_WEIGHTED_INVARIANT_FOUND",
    "CONTRACT_REVIEW_REQUIRED",
}


def build_weight_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {"label": "ordinary_norm", "kind": "ordinary", "scale": "none", "exploratory": False},
        {"label": "dx_scaled_norm", "kind": "ordinary", "scale": "dx3", "exploratory": False},
        {"label": "omega_exact_helper_weight", "kind": "omega_power", "power": 1.0, "scale": "none", "exploratory": True},
        {"label": "omega_sq_exact_helper_weight", "kind": "omega_sq", "scale": "none", "exploratory": True},
        {"label": "inverse_omega_sq_exact_helper_weight", "kind": "inverse_omega_sq", "scale": "none", "exploratory": True},
        {"label": "sqrt_g_weighted_norm", "kind": "omega_power", "power": 3.0, "scale": "none", "exploratory": True},
        {"label": "inverse_sqrt_g_weighted_norm", "kind": "omega_power", "power": -3.0, "scale": "none", "exploratory": True},
    ]
    for power in [-6.0, -4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0, 6.0]:
        label_power = int(power) if float(power).is_integer() else power
        specs.append(
            {
                "label": f"omega_power_p{label_power}_exploratory",
                "kind": "omega_power",
                "power": power,
                "scale": "none",
                "exploratory": True,
            }
        )
    return specs


def build_manifest(max_wallclock_minutes: float = 60.0) -> dict[str, Any]:
    return {
        "regime": "conservative_c2_weighted_invariant_audit",
        "N": 48,
        "L": 10.0,
        "dt": 0.001,
        "physical_time": 1.0,
        "spacing": 0.45,
        "sample_times": [0.0, 0.25, 0.5, 0.75, 1.0],
        "max_wallclock_minutes": float(max_wallclock_minutes),
        "weights": build_weight_specs(),
        "rules": [
            "Conservative C2 only",
            "CuPy only",
            "diagnostic only",
            "state-dependent Omega weights are exploratory unless confirmed by contract",
            "no amplitude normalization",
            "no geometry campaign",
            "no stability claim",
        ],
    }


def resolve_output_dir(base: Path, timestamp: str | None = None) -> Path:
    stamp = timestamp or time.strftime("%Y%m%d_%H%M%S")
    if base.name.startswith("weighted_invariant_audit_"):
        return base
    return base.with_name(f"{base.name}_{stamp}")


def compute_weight_from_spec(spec: dict[str, Any], omega: Any, omega_sq: Any) -> Any:
    kind = spec["kind"]
    if kind == "ordinary":
        return omega * 0.0 + 1.0
    if kind == "omega_power":
        return omega ** float(spec["power"])
    if kind == "omega_sq":
        return omega_sq
    if kind == "inverse_omega_sq":
        return 1.0 / omega_sq
    raise ValueError(f"unsupported weight kind {kind!r}")


def weighted_invariant(psi_phys: Any, weight: Any, scale: float = 1.0) -> float:
    val = np.sum(np.asarray(weight) * (np.abs(np.asarray(psi_phys)) ** 2), dtype=np.float64)
    return float(scale * val)


def weighted_fractional_flux(psi_phys: Any, rhs_phys: Any, weight: Any) -> float:
    invariant = weighted_invariant(psi_phys, weight, scale=1.0)
    raw = physical_flux_raw(np.asarray(psi_phys) * np.sqrt(np.asarray(weight)), np.asarray(rhs_phys) * np.sqrt(np.asarray(weight)))
    return float(raw / (abs(invariant) + 1e-30))


def classify_recommendation(
    *,
    ordinary_abs_drift: float,
    best_weight_abs_drift: float,
    ordinary_adjoint_mismatch: float,
    best_weight_adjoint_mismatch: float,
    best_label: str | None,
) -> dict[str, Any]:
    if not best_label:
        return {"recommendation": "CONTRACT_REVIEW_REQUIRED", "best_label": None}
    drift_improves = float(best_weight_abs_drift) <= 0.25 * max(float(ordinary_abs_drift), 1e-30)
    adjoint_improves = float(best_weight_adjoint_mismatch) <= 0.25 * max(float(ordinary_adjoint_mismatch), 1e-30)
    if drift_improves and adjoint_improves:
        rec = "ORDINARY_NORM_NOT_CONSERVED_WEIGHTED_CANDIDATE_FOUND"
    elif float(ordinary_abs_drift) > 1e-6:
        rec = "NO_WEIGHTED_INVARIANT_FOUND"
    else:
        rec = "CONTRACT_REVIEW_REQUIRED"
    return {"recommendation": rec, "best_label": best_label}


def _cp_float(value: Any) -> float:
    try:
        return float(value.get())
    except AttributeError:
        return float(value)


def _weighted_invariant_cp(psi: Any, weight: Any, scale: float) -> float:
    import cupy as cp

    return _cp_float(scale * cp.sum(weight * (cp.abs(psi) ** 2), dtype=cp.float64))


def _weighted_flux_cp(psi: Any, rhs: Any, weight: Any) -> float:
    import cupy as cp

    invariant = cp.sum(weight * (cp.abs(psi) ** 2), dtype=cp.float64)
    raw = 2.0 * cp.real(cp.sum(weight * cp.conj(psi) * rhs, dtype=cp.complex128))
    return _cp_float(raw / cp.maximum(cp.abs(invariant), cp.float64(1e-30)))


def _collect_trajectory(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    import cupy as cp

    N = int(manifest["N"])
    L = float(manifest["L"])
    dt = float(manifest["dt"])
    steps = steps_for_physical_time(float(manifest["physical_time"]), dt)
    sample_steps = sample_steps_for_times(steps, dt, list(manifest["sample_times"]))
    sample_set = set(sample_steps)
    stepper = ConservativeC2RK4Stepper(N, L, dt)
    psi0 = triangle_psi0(N, L, float(manifest["spacing"]))
    psi_k, _ = stepper.project_psi0(psi0)
    states: list[dict[str, Any]] = []

    def record(step: int, current_psi_k: Any) -> None:
        fields = _shared_fields(stepper, current_psi_k)
        decomp = decompose_nonlinear_terms(stepper, current_psi_k)
        nonlinear_k = decomp["original_k"]
        linear_k = stepper.solver.L_k * current_psi_k
        full_k = linear_k + nonlinear_k
        states.append(
            {
                "step": int(step),
                "t_phys": float(step * dt),
                "stepper": stepper,
                "psi_k": current_psi_k.copy(),
                "psi_phys": fields["psi"].copy(),
                "omega": fields["omega"].copy(),
                "omega_sq": fields["omega_sq"].copy(),
                "full_rhs_phys": stepper.solver.ifft_single(full_k).copy(),
                "nonlinear_rhs_phys": stepper.solver.ifft_single(nonlinear_k).copy(),
                "geometry_rhs_phys": stepper.solver.ifft_single(decomp["term_k"]["geometry_covariant_correction"]).copy(),
                "decomp": decomp,
            }
        )
        cp.cuda.Stream.null.synchronize()

    record(0, psi_k)
    for step in range(1, steps + 1):
        psi_k = stepper.step(psi_k)
        if step in sample_set:
            record(step, psi_k)
    if states[-1]["step"] != steps:
        record(steps, psi_k)
    return states


def run_invariant_audit(manifest: dict[str, Any], states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import cupy as cp

    L = float(manifest["L"])
    N = int(manifest["N"])
    dx = L / N
    rows: list[dict[str, Any]] = []
    initial_values: dict[str, float] = {}
    for state in states:
        for spec in manifest["weights"]:
            weight = compute_weight_from_spec(spec, state["omega"], state["omega_sq"])
            scale = dx**3 if spec.get("scale") == "dx3" else 1.0
            invariant = _weighted_invariant_cp(state["psi_phys"], weight, scale)
            initial_values.setdefault(spec["label"], invariant)
            initial = initial_values[spec["label"]]
            row = {
                "label": spec["label"],
                "kind": spec["kind"],
                "power": spec.get("power"),
                "scale": spec.get("scale"),
                "exploratory": spec.get("exploratory"),
                "step": state["step"],
                "t_phys": state["t_phys"],
                "invariant": invariant,
                "fractional_drift_from_t0": (invariant - initial) / (abs(initial) + 1e-30),
                "frozen_weight_full_fractional_flux": _weighted_flux_cp(state["psi_phys"], state["full_rhs_phys"], weight),
                "frozen_weight_nonlinear_fractional_flux": _weighted_flux_cp(state["psi_phys"], state["nonlinear_rhs_phys"], weight),
                "frozen_weight_geometry_fractional_flux": _weighted_flux_cp(state["psi_phys"], state["geometry_rhs_phys"], weight),
                "ordinary_norm_reference": _cp_float(cp.sum(cp.abs(state["psi_phys"]) ** 2, dtype=cp.float64)),
            }
            rows.append(row)
    return rows


def _weighted_inner(a: Any, b: Any, weight: Any) -> Any:
    import cupy as cp

    return cp.sum(weight * cp.conj(a) * b, dtype=cp.complex128)


def run_weighted_adjoint_audit(manifest: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    import cupy as cp

    stepper = state["stepper"]
    solver = stepper.solver
    fields = state["decomp"]["fields"]
    N = int(manifest["N"])
    rng = np.random.default_rng(2112)
    a = cp.asarray(rng.normal(size=(N, N, N)) + 1j * rng.normal(size=(N, N, N)), dtype=cp.complex128)
    b = cp.asarray(rng.normal(size=(N, N, N)) + 1j * rng.normal(size=(N, N, N)), dtype=cp.complex128)

    grad_omega_x = fields["d_omega_d_rho"] * (2.0 * cp.real(cp.conj(fields["psi"]) * fields["grad_x"]))
    grad_omega_y = fields["d_omega_d_rho"] * (2.0 * cp.real(cp.conj(fields["psi"]) * fields["grad_y"]))
    grad_omega_z = fields["d_omega_d_rho"] * (2.0 * cp.real(cp.conj(fields["psi"]) * fields["grad_z"]))

    def flat_laplacian(v: Any) -> Any:
        v_k = solver.fft_single(v)
        return solver.ifft_single(solver.minus_k_sq_filtered * v_k)

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
    for spec in manifest["weights"]:
        weight = compute_weight_from_spec(spec, state["omega"], state["omega_sq"])
        for op_label, op in operators.items():
            lhs = _weighted_inner(a, op(b), weight)
            rhs = _weighted_inner(op(a), b, weight)
            denom = cp.maximum(cp.maximum(cp.abs(lhs), cp.abs(rhs)), cp.float64(1e-30))
            rel = _cp_float(cp.abs(lhs - rhs) / denom)
            rows.append(
                {
                    "label": spec["label"],
                    "operator": op_label,
                    "t_phys": state["t_phys"],
                    "relative_mismatch": rel,
                    "classification": classify_adjoint_mismatch(rel),
                    "exploratory": spec.get("exploratory"),
                }
            )
    return rows


def _summaries(invariant_rows: list[dict[str, Any]], adjoint_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = sorted({str(r["label"]) for r in invariant_rows})
    summaries: list[dict[str, Any]] = []
    for label in labels:
        inv_for_label = [r for r in invariant_rows if r["label"] == label]
        adj_for_label = [
            r for r in adjoint_rows if r["label"] == label and r["operator"] == "frozen_geometry_correction_proxy"
        ]
        summaries.append(
            {
                "label": label,
                "max_abs_fractional_drift": max(abs(float(r["fractional_drift_from_t0"])) for r in inv_for_label),
                "final_fractional_drift": float(inv_for_label[-1]["fractional_drift_from_t0"]),
                "max_abs_frozen_weight_full_flux": max(abs(float(r["frozen_weight_full_fractional_flux"])) for r in inv_for_label),
                "geometry_correction_adjoint_mismatch": min(
                    (float(r["relative_mismatch"]) for r in adj_for_label),
                    default=float("inf"),
                ),
            }
        )
    return summaries


def _write_report(path: Path, recommendation: dict[str, Any], summaries: list[dict[str, Any]]) -> None:
    ranked = sorted(summaries, key=lambda r: (float(r["max_abs_fractional_drift"]), float(r["geometry_correction_adjoint_mismatch"])))
    lines = [
        "# Conservative C2 Weighted Invariant Audit",
        "",
        f"Recommendation: `{recommendation['recommendation']}`",
        f"Best candidate: `{recommendation.get('best_label')}`",
        "",
        "State-dependent Omega weights are exploratory frozen-weight probes unless the C2 contract identifies them as exact measures.",
        "No stability claim is made, no amplitude normalization was applied, and no geometry campaign was run.",
        "",
        "## Top Candidates By Drift Then Adjointness",
        "",
    ]
    for item in ranked[:8]:
        lines.append(
            f"- `{item['label']}`: max drift `{item['max_abs_fractional_drift']}`, "
            f"final drift `{item['final_fractional_drift']}`, "
            f"full flux max `{item['max_abs_frozen_weight_full_flux']}`, "
            f"geometry-adjoint mismatch `{item['geometry_correction_adjoint_mismatch']}`"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = resolve_output_dir(Path(args.out))
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    safety = run_safety_checkpoint(out_dir, args.require_gpu_name)
    manifest = build_manifest(float(args.max_wallclock_minutes))
    (out_dir / "weighted_invariant_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    states = _collect_trajectory(manifest)
    invariant_rows = run_invariant_audit(manifest, states)
    adjoint_state = min(states, key=lambda item: abs(float(item["t_phys"]) - 0.75))
    adjoint_rows = run_weighted_adjoint_audit(manifest, adjoint_state)
    summaries = _summaries(invariant_rows, adjoint_rows)
    ordinary = next(item for item in summaries if item["label"] == "ordinary_norm")
    non_ordinary = [item for item in summaries if item["label"] not in {"ordinary_norm", "dx_scaled_norm"}]
    best = min(non_ordinary, key=lambda r: (float(r["max_abs_fractional_drift"]), float(r["geometry_correction_adjoint_mismatch"])), default=None)
    recommendation = classify_recommendation(
        ordinary_abs_drift=float(ordinary["max_abs_fractional_drift"]),
        best_weight_abs_drift=float(best["max_abs_fractional_drift"]) if best else float("inf"),
        ordinary_adjoint_mismatch=float(ordinary["geometry_correction_adjoint_mismatch"]),
        best_weight_adjoint_mismatch=float(best["geometry_correction_adjoint_mismatch"]) if best else float("inf"),
        best_label=str(best["label"]) if best else None,
    )
    write_csv(out_dir / "weighted_invariant_audit_results.csv", invariant_rows + [{"record_type": "summary", **row} for row in summaries])
    write_csv(out_dir / "weighted_operator_adjoint_results.csv", adjoint_rows)
    _write_report(out_dir / "weighted_invariant_audit_report.md", recommendation, summaries)
    final = {
        "out_dir": str(out_dir),
        "recommendation": recommendation["recommendation"],
        "best_label": recommendation.get("best_label"),
        "ordinary_max_abs_drift": ordinary["max_abs_fractional_drift"],
        "best_max_abs_drift": best["max_abs_fractional_drift"] if best else None,
        "wallclock_seconds": time.time() - start,
        "protected_diff_empty": safety.get("protected_diff_empty"),
    }
    (out_dir / "weighted_invariant_audit_summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2))
    return final


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--max-wallclock-minutes", type=float, default=60.0)
    parser.add_argument("--require-gpu-name", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_batch(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
