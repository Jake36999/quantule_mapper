"""Standalone CuPy triangle IC diagnostic for Phase C geometry screens.

This tool is deliberately outside the production worker path. It reuses
``solver.core.ETDRK4Solver`` and mirrors the production stepping loop around an
externally constructed synthetic ``psi0``. It does not modify solver physics,
Hunter logic, validation gates, production defaults, or existing configs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_utils import generate_canonical_hash  # noqa: E402
from tools.triangle_layout_diagnostic import build_triangle_ic, node_geometry_from_psi  # noqa: E402


FEB_PARAMS = {
    "param_D": 2.7329,
    "param_eta": 0.0704,
    "param_rho_vac": 1.1866,
    "param_omega0": 0.0,
    "param_a_coupling": 2.3098,
    "param_s": 0.0129,
    "param_f": -0.4861,
    "param_a": 0.4802,
}
DEFAULT_OUT = ROOT / "quantule_viz" / "outputs" / "triangle_cupy_screen"
DEFAULT_L = 10.0
DEFAULT_DT = 0.005
DEFAULT_SEED = 20260619
DEFAULT_WIDTH_BOX = 1 / 12


def build_case_config(case_id: str, N: int, steps: int, spacing: float, dt: float, L: float) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "case_id": case_id,
        "phase_c_regime": "dissipative_geometry_diagnostic_only",
        "solver_component": "solver.core.ETDRK4Solver",
        "loop_contract": "diagnostic_external_psi0_mirrors_solver.run.run_simulation_step_order",
        "params": dict(FEB_PARAMS),
        "simulation": {"n_grid": int(N), "t_steps": int(steps), "dt": float(dt), "l_domain": float(L)},
        "triangle_ic": {
            "spacing_box": float(spacing),
            "variant": "aligned_phase",
            "phases": [0.0, 0.0, 0.0],
            "width_box": DEFAULT_WIDTH_BOX,
            "amplitude": 1.0,
            "noise_level": 0.0,
            "seed": DEFAULT_SEED,
            "generator": "tools.triangle_layout_diagnostic.build_triangle_ic",
        },
        "limitations": [
            "Standalone diagnostic path only.",
            "Phase C dissipative geometry prior only.",
            "No claim for conservative or moving substrate behavior.",
        ],
    }
    cfg["config_hash"] = generate_canonical_hash(cfg)
    return cfg


def case_paths(out_dir: Path, case_id: str) -> dict[str, Path]:
    case_dir = Path(out_dir) / case_id
    return {
        "case_dir": case_dir,
        "artifact": case_dir / f"{case_id}_smoke.npz",
        "metadata": case_dir / f"{case_id}_metadata.json",
        "final_rho_image": case_dir / f"{case_id}_final_rho.png",
    }


def _gpu_info(require_name: str | None = None) -> dict[str, Any]:
    import cupy as cp

    count = int(cp.cuda.runtime.getDeviceCount())
    if count < 1:
        raise RuntimeError("CuPy sees no CUDA devices")
    props = cp.cuda.runtime.getDeviceProperties(0)
    raw_name = props["name"]
    name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
    if require_name and name != require_name:
        raise RuntimeError(f"Unexpected GPU name {name!r}; expected {require_name!r}")
    return {
        "python_executable": sys.executable,
        "cupy_version": cp.__version__,
        "device_count": count,
        "gpu_name": name,
        "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
        "cuda_driver": int(cp.cuda.runtime.driverGetVersion()),
    }


def _run_external_psi0_loop(psi0_np: np.ndarray, cfg: dict[str, Any], collapse_threshold: float = 1e6) -> dict[str, Any]:
    import cupy as cp
    from solver.core import ETDRK4Solver

    sim = cfg["simulation"]
    N = int(sim["n_grid"])
    L = float(sim["l_domain"])
    dt = float(sim["dt"])
    steps = int(sim["t_steps"])
    solver = ETDRK4Solver(N, L, dt, dict(cfg["params"]))
    psi = cp.asarray(psi0_np, dtype=cp.complex128)
    psi_k = solver.fft_single(psi) * solver.dealias_mask
    ic_e_raw = float(cp.sum(cp.abs(solver.ifft_single(psi_k)) ** 2, dtype=cp.float64))
    dV = (L / N) ** 3
    history: list[dict[str, float | int]] = []
    final_step = -1
    fail_reason = ""
    t0 = time.time()

    for step in range(steps):
        final_step = step
        psi_real_step = solver.ifft_single(psi_k)
        rho_real = cp.maximum(cp.abs(psi_real_step) ** 2, cp.float64(1e-7))
        rho_k = solver.fft_single(rho_real) * solver.dealias_mask
        solver.update_field_of_affect(rho_k, dt)

        psi_k = solver.step(psi_k)

        if step % 10 == 0 or step == steps - 1:
            solver.update_dynamic_filters()

        if step % 50 == 0:
            mean_psi = psi_k[0, 0, 0] / (N**3)
            mean_phase = cp.angle(mean_psi)
            psi_k *= cp.exp(-1j * mean_phase)

        if step % 10 == 0 or step == steps - 1:
            psi_real = solver.ifft_single(psi_k)
            if not bool(cp.isfinite(psi_real).all()):
                fail_reason = f"nonfinite at step {step}"
                break
            max_amp = float(cp.max(cp.abs(psi_real)))
            if max_amp > collapse_threshold:
                fail_reason = f"amplitude {max_amp:.3e} exceeded collapse threshold"
                break
            rho = cp.maximum(cp.abs(psi_real) ** 2, cp.float64(1e-7))
            history.append(
                {
                    "step": int(step),
                    "energy": float(cp.sum(rho, dtype=cp.float64)) * dV,
                    "raw_energy": float(cp.sum(cp.abs(psi_real) ** 2, dtype=cp.float64)),
                    "rho_max": float(cp.max(rho)),
                    "rho_mean": float(cp.mean(rho)),
                    "max_amp": max_amp,
                }
            )

    psi_fin = solver.ifft_single(psi_k)
    rho_fin = cp.abs(psi_fin) ** 2
    finite = bool(cp.isfinite(psi_fin).all())
    cp.cuda.Device().synchronize()
    result = {
        "psi0": cp.asnumpy(psi),
        "psi_fin": cp.asnumpy(psi_fin),
        "rho_fin": cp.asnumpy(rho_fin),
        "history": history,
        "summary": {
            "finite": finite and not fail_reason,
            "fail_reason": fail_reason,
            "final_step": int(final_step),
            "requested_steps": steps,
            "ic_raw_energy": ic_e_raw,
            "final_raw_energy": float(cp.sum(cp.abs(psi_fin) ** 2, dtype=cp.float64)),
            "rho_max": float(cp.max(rho_fin)),
            "rho_mean": float(cp.mean(rho_fin)),
            "rho_min": float(cp.min(rho_fin)),
            "max_abs_psi": float(cp.max(cp.abs(psi_fin))),
            "wallclock_sec": round(time.time() - t0, 3),
        },
    }
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return result


def run_case(
    out_dir: Path,
    N: int,
    steps: int,
    spacing: float,
    dt: float,
    L: float,
    collapse_threshold: float,
    require_gpu_name: str | None,
    mode: str = "smoke",
) -> dict[str, Any]:
    case_id = f"triangle_s{spacing:.2f}_N{N}_T{steps}"
    paths = case_paths(out_dir, case_id)
    paths["case_dir"].mkdir(parents=True, exist_ok=True)
    cfg = build_case_config(case_id, N, steps, spacing, dt, L)
    gpu = _gpu_info(require_gpu_name)
    psi0 = build_triangle_ic(
        N=N,
        L=L,
        side_length_box=spacing,
        width_box=DEFAULT_WIDTH_BOX,
        amplitude=1.0,
        phases=[0.0, 0.0, 0.0],
        noise_level=0.0,
        seed=DEFAULT_SEED,
    )
    result = _run_external_psi0_loop(psi0, cfg, collapse_threshold=collapse_threshold)
    initial_geom = node_geometry_from_psi(psi0, L=L, expected_nodes=3)
    final_geom = node_geometry_from_psi(result["psi_fin"], L=L, expected_nodes=8)
    late = late_trends(result["history"])
    spacing_drift = None
    if np.isfinite(initial_geom["nn_spacing_mean_box"]) and np.isfinite(final_geom["nn_spacing_mean_box"]):
        spacing_drift = float(final_geom["nn_spacing_mean_box"] - initial_geom["nn_spacing_mean_box"])
    render_final_rho(result["rho_fin"], final_geom, paths["final_rho_image"])
    row = {
        "case_id": case_id,
        "spacing_box": float(spacing),
        "N": int(N),
        "steps": int(steps),
        "final_finite": bool(result["summary"]["finite"]),
        "fail_reason": result["summary"]["fail_reason"],
        "final_rho_max": result["summary"]["rho_max"],
        "final_rho_mean": result["summary"]["rho_mean"],
        "final_raw_energy": result["summary"]["final_raw_energy"],
        "detected_final_node_count": final_geom["node_count"],
        "pairwise_node_distances_box": json.dumps(final_geom["pairwise_distances_box"]),
        "final_node_mass_cv": final_geom["node_mass_cv"],
        "final_rho_peak_cv": final_geom["rho_peak_cv"],
        "initial_nn_spacing_mean_box": initial_geom["nn_spacing_mean_box"],
        "final_nn_spacing_mean_box": final_geom["nn_spacing_mean_box"],
        "spacing_drift_mean_box": spacing_drift,
        "late_raw_energy_fractional_trend": late["raw_energy_fractional_trend"],
        "late_rho_max_fractional_trend": late["rho_max_fractional_trend"],
        "wallclock_sec": result["summary"]["wallclock_sec"],
        "artifact": str(paths["artifact"]),
        "metadata": str(paths["metadata"]),
        "final_rho_image": str(paths["final_rho_image"]),
        "caveat": "Phase C dissipative diagnostic only; no conservative/moving substrate claim.",
    }
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": mode,
        "gpu": gpu,
        "config": cfg,
        "summary": result["summary"],
        "initial_geometry": initial_geom,
        "final_geometry": final_geom,
        "screen_row": row,
        "artifact": str(paths["artifact"]),
        "production_files_modified": False,
        "caveat": "Diagnostic only; not conservative/moving substrate evidence.",
    }
    np.savez_compressed(
        paths["artifact"],
        psi0=result["psi0"].astype(np.complex128),
        psi_fin=result["psi_fin"].astype(np.complex128),
        rho_fin=result["rho_fin"].astype(np.float64),
        history_json=json.dumps(result["history"]),
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def late_trends(history: list[dict[str, Any]]) -> dict[str, float | None]:
    if len(history) < 4:
        return {"raw_energy_fractional_trend": None, "rho_max_fractional_trend": None}

    def trend(key: str) -> float | None:
        half = len(history) // 2
        vals = np.asarray([float(item[key]) for item in history[half:] if key in item], dtype=float)
        if vals.size < 3 or not np.all(np.isfinite(vals)):
            return None
        xs = np.arange(vals.size, dtype=float)
        slope, intercept = np.polyfit(xs, vals, 1)
        start = float(intercept)
        end = float(slope * (vals.size - 1) + intercept)
        return float((end - start) / (abs(start) + 1e-30))

    return {"raw_energy_fractional_trend": trend("raw_energy"), "rho_max_fractional_trend": trend("rho_max")}


def render_final_rho(rho_fin: np.ndarray, geometry: dict[str, Any], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    projection = np.max(np.asarray(rho_fin, dtype=float), axis=2)
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=120)
    im = ax.imshow(projection, origin="lower", cmap="magma")
    centroids = geometry.get("centroids_vox", [])
    if centroids:
        xs = [float(c[1]) for c in centroids]
        ys = [float(c[0]) for c in centroids]
        ax.scatter(xs, ys, s=50, facecolors="none", edgecolors="cyan", linewidths=1.1)
        for idx, (x, y) in enumerate(zip(xs, ys), start=1):
            ax.text(x + 0.5, y + 0.5, str(idx), color="cyan", fontsize=7)
    ax.set_title(f"final rho | nodes={geometry.get('node_count')}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    return run_case(
        out_dir=args.out,
        N=args.N,
        steps=args.steps,
        spacing=args.spacing,
        dt=args.dt,
        L=args.L,
        collapse_threshold=args.collapse_threshold,
        require_gpu_name=args.require_gpu_name,
        mode="smoke",
    )


def _parse_spacings(raw: str) -> list[float]:
    spacings = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not spacings:
        raise ValueError("at least one spacing is required")
    return spacings


def run_screen(args: argparse.Namespace) -> dict[str, Any]:
    spacings = _parse_spacings(args.spacings)
    rows = []
    for spacing in spacings:
        rows.append(
            run_case(
                out_dir=args.out,
                N=args.N,
                steps=args.steps,
                spacing=spacing,
                dt=args.dt,
                L=args.L,
                collapse_threshold=args.collapse_threshold,
                require_gpu_name=args.require_gpu_name,
                mode="screen",
            )
        )
    result_rows = [case["screen_row"] for case in rows]
    csv_path = args.out / f"triangle_spacing_screen_N{args.N}_T{args.steps}_results.csv"
    report_path = args.out / f"triangle_spacing_screen_N{args.N}_T{args.steps}_report.md"
    write_rows_csv(csv_path, result_rows)
    write_screen_report(report_path, result_rows, args.N, args.steps, spacings)
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "screen",
        "N": args.N,
        "steps": args.steps,
        "spacings": spacings,
        "cases": rows,
        "results_csv": str(csv_path),
        "report_md": str(report_path),
        "caveat": "Full diagnostic spacing screen only; not a production path and not conservative/moving substrate evidence.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"screen_manifest_N{args.N}_T{args.steps}.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def write_screen_report(path: Path, rows: list[dict[str, Any]], N: int, steps: int, spacings: list[float]) -> None:
    ranked = sorted(rows, key=promotion_score, reverse=True)
    lines = [
        f"# Triangle Spacing Screen N{N} T{steps}",
        "",
        "Standalone diagnostic CuPy screen using `solver.core.ETDRK4Solver` with externally constructed aligned-phase triangle ICs.",
        "",
        "This is Phase C dissipative geometry testing only. These results do not apply to the conservative or moving substrate.",
        "",
        f"- Spacings: `{spacings}`",
        f"- Fixed params hash family: FEB Phase C dissipative parameters",
        "",
        "## Results",
        "",
        "| spacing | finite | final nodes | rho max | raw energy | spacing drift | mass CV | peak CV | late raw-energy trend |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['spacing_box']:.2f} | {row['final_finite']} | {row['detected_final_node_count']} | "
            f"{float(row['final_rho_max']):.6g} | {float(row['final_raw_energy']):.6g} | "
            f"{fmt(row['spacing_drift_mean_box'])} | {fmt(row['final_node_mass_cv'])} | "
            f"{fmt(row['final_rho_peak_cv'])} | {fmt(row['late_raw_energy_fractional_trend'])} |"
        )
    lines.extend(["", "## Recommendation", ""])
    promotable = [
        row for row in ranked
        if bool(row["final_finite"]) and int(row["detected_final_node_count"]) == 3
    ]
    if promotable:
        lines.append("Promote at most the following spacing(s) to N=96/T=4000, still as diagnostic-only priors:")
        for row in promotable[:3]:
            lines.append(
                f"- `{row['spacing_box']:.2f}`: final nodes `{row['detected_final_node_count']}`, "
                f"spacing drift `{fmt(row['spacing_drift_mean_box'])}`, mass CV `{fmt(row['final_node_mass_cv'])}`, "
                f"late raw-energy trend `{fmt(row['late_raw_energy_fractional_trend'])}`."
            )
    else:
        lines.append("No spacing preserved a clean finite 3-node final state by this detector at this intermediate horizon.")
    lines.extend(["", "## Caveats", ""])
    lines.append("- Node counts are diagnostic detector outputs from final rho, not validation-gate certifications.")
    lines.append("- Screen artifacts are standalone diagnostics and do not alter production defaults or configs.")
    lines.append("- No long GIFs were generated; each case has a compact final rho PNG.")
    path.write_text("\n".join(lines), encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(val):
        return ""
    return f"{val:.6g}"


def promotion_score(row: dict[str, Any]) -> float:
    score = 0.0
    if bool(row["final_finite"]):
        score += 20.0
    if int(row["detected_final_node_count"]) == 3:
        score += 40.0
    drift = row.get("spacing_drift_mean_box")
    if drift is not None and np.isfinite(float(drift)):
        score -= min(25.0, abs(float(drift)) * 100.0)
    for key, weight in (("final_node_mass_cv", 20.0), ("final_rho_peak_cv", 15.0)):
        value = row.get(key)
        if value is not None and np.isfinite(float(value)):
            score -= min(weight, float(value) * weight)
    trend = row.get("late_raw_energy_fractional_trend")
    if trend is not None and np.isfinite(float(trend)):
        score -= min(15.0, abs(float(trend)) * 50.0)
    return score


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    smoke = sub.add_parser("smoke", help="Run one tiny CuPy GPU triangle IC smoke test")
    smoke.add_argument("--out", type=Path, default=DEFAULT_OUT)
    smoke.add_argument("--N", type=int, default=32)
    smoke.add_argument("--steps", type=int, default=50)
    smoke.add_argument("--spacing", type=float, default=0.32)
    smoke.add_argument("--dt", type=float, default=DEFAULT_DT)
    smoke.add_argument("--L", type=float, default=DEFAULT_L)
    smoke.add_argument("--collapse-threshold", type=float, default=1e6)
    smoke.add_argument("--require-gpu-name", default="NVIDIA GeForce GTX 1080")
    smoke.set_defaults(func=run_smoke)
    screen = sub.add_parser("screen", help="Run an explicit multi-spacing CuPy diagnostic screen")
    screen.add_argument("--out", type=Path, default=DEFAULT_OUT)
    screen.add_argument("--N", type=int, default=96)
    screen.add_argument("--steps", type=int, default=4000)
    screen.add_argument("--spacings", default="0.28,0.32,0.36,0.40,0.45,0.49,0.53")
    screen.add_argument("--dt", type=float, default=DEFAULT_DT)
    screen.add_argument("--L", type=float, default=DEFAULT_L)
    screen.add_argument("--collapse-threshold", type=float, default=1e6)
    screen.add_argument("--require-gpu-name", default="NVIDIA GeForce GTX 1080")
    screen.set_defaults(func=run_screen)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
