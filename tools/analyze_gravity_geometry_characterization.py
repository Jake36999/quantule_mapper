"""Read-only geometry characterization for IRER Gravity Ladder Rung A+D.

This diagnostic tool reads saved Gravity A+D artifacts and characterizes the
effective Omega^2(rho) soft-clip law. It does not run probes or modify solver
physics, production defaults, Hunter, validation, or configs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROTECTED_FILES = [
    "solver/core.py",
    "solver/run.py",
    "worker_cupy.py",
    "aste_hunter.py",
    "validation_pipeline.py",
    "config_utils.py",
    "tools/production_h7_revalidation.py",
    "gravity/unified_omega.py",
    "jax_scout/gravity_ladder_A_D.py",
    "jax_scout/physics.py",
]


def production_geometry_params() -> dict[str, float]:
    """Production Rung A+D geometry params from jax_scout/gravity_ladder_A_D.py.

    Kept local to avoid importing JAX-side scout modules in this read-only
    analyzer.
    """

    return {
        "rho_vac": 1.1866,
        "a_coupling": 2.3098,
        "softclip_beta": 3.0,
        "omega_sq_min": 1e-9,
        "omega_sq_max": 1e6,
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_command(args: list[str], cwd: Path) -> tuple[int, str]:
    try:
        proc = subprocess.run(args, cwd=str(cwd), text=True, capture_output=True, check=False)
        return proc.returncode, (proc.stdout + proc.stderr).strip()
    except Exception as exc:  # pragma: no cover - defensive metadata path
        return 1, repr(exc)


def soft_clip_log(
    raw_omega_sq: np.ndarray,
    omega_min: float,
    omega_max: float,
    beta: float,
) -> np.ndarray:
    raw = np.asarray(raw_omega_sq, dtype=np.float64)
    raw = np.maximum(raw, 1e-300)
    log_v = np.log(raw)
    log_lo = math.log(float(omega_min))
    log_hi = math.log(float(omega_max))
    center = 0.5 * (log_lo + log_hi)
    half = 0.5 * (log_hi - log_lo)
    z = float(beta) * ((log_v - center) / (half + 1e-12))
    return np.exp(center + half * np.tanh(z))


def omega_sq_from_rho(rho: np.ndarray, params: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    rho_safe = np.maximum(np.asarray(rho, dtype=np.float64), 1e-12)
    nominal = (float(params["rho_vac"]) / rho_safe) ** float(params["a_coupling"])
    impl = soft_clip_log(
        nominal,
        float(params["omega_sq_min"]),
        float(params["omega_sq_max"]),
        float(params["softclip_beta"]),
    )
    return nominal, impl


def log_slope(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_safe = np.maximum(np.asarray(x, dtype=np.float64), 1e-300)
    y_safe = np.maximum(np.asarray(y, dtype=np.float64), 1e-300)
    if len(x_safe) < 3:
        return np.zeros_like(x_safe)
    return np.gradient(np.log(y_safe), np.log(x_safe))


def classify_curve_region(omega_sq: float, slope: float, params: dict[str, float]) -> str:
    if omega_sq >= 0.9 * float(params["omega_sq_max"]):
        return "cap_saturated"
    if omega_sq <= 1.1 * float(params["omega_sq_min"]):
        return "floor_saturated"
    if abs(slope) < 0.1:
        return "flat_softclip"
    return "graded"


def evaluate_curve_arrays(rho: np.ndarray, params: dict[str, float]) -> dict[str, np.ndarray]:
    nominal, impl = omega_sq_from_rho(rho, params)
    slope = log_slope(rho, impl)
    ratio = impl / np.maximum(nominal, 1e-300)
    return {
        "rho": np.asarray(rho, dtype=np.float64),
        "omega_sq_nominal": nominal,
        "omega_sq_impl": impl,
        "impl_over_nominal": ratio,
        "log_slope": slope,
    }


def evaluate_geometry_curve(rho: np.ndarray, params: dict[str, float]) -> list[dict[str, Any]]:
    arrays = evaluate_curve_arrays(rho, params)
    rows: list[dict[str, Any]] = []
    for i in range(len(arrays["rho"])):
        rows.append(
            {
                "rho": float(arrays["rho"][i]),
                "omega_sq_nominal": float(arrays["omega_sq_nominal"][i]),
                "omega_sq_impl": float(arrays["omega_sq_impl"][i]),
                "impl_over_nominal": float(arrays["impl_over_nominal"][i]),
                "log_slope": float(arrays["log_slope"][i]),
                "region": classify_curve_region(
                    float(arrays["omega_sq_impl"][i]),
                    float(arrays["log_slope"][i]),
                    params,
                ),
            }
        )
    return rows


def score_scan_candidate(
    rho: np.ndarray,
    curve: dict[str, np.ndarray],
    params: dict[str, float],
) -> dict[str, float]:
    impl = np.asarray(curve["omega_sq_impl"], dtype=np.float64)
    slope = np.asarray(curve["log_slope"], dtype=np.float64)
    cap = impl >= 0.9 * float(params["omega_sq_max"])
    floor = impl <= 1.1 * float(params["omega_sq_min"])
    flat = np.abs(slope) < 0.1
    graded = (~cap) & (~floor) & (np.abs(slope) >= 0.1) & (np.abs(slope) <= 10.0)
    finite = bool(np.isfinite(impl).all() and np.isfinite(slope).all())
    log_impl = np.log10(np.maximum(impl, 1e-300))
    max_adjacent_log_jump = float(np.max(np.abs(np.diff(log_impl)))) if len(log_impl) > 1 else 0.0
    dynamic_range_log10 = float(np.max(log_impl) - np.min(log_impl))
    smooth_penalty = max(0.0, max_adjacent_log_jump - 0.2)
    score = (
        float(np.mean(graded))
        - float(np.mean(cap))
        - float(np.mean(floor))
        - 0.5 * float(np.mean(flat))
        - smooth_penalty
        + 0.05 * min(dynamic_range_log10, 6.0)
    )
    return {
        "finite": float(finite),
        "score": float(score if finite else -999.0),
        "graded_fraction": float(np.mean(graded)),
        "cap_fraction": float(np.mean(cap)),
        "floor_fraction": float(np.mean(floor)),
        "flat_fraction": float(np.mean(flat)),
        "dynamic_range_log10": dynamic_range_log10,
        "median_abs_log_slope": float(np.median(np.abs(slope))),
        "max_adjacent_log_jump": max_adjacent_log_jump,
        "rho_min": float(np.min(rho)),
        "rho_max": float(np.max(rho)),
    }


def radial_profile(
    field: np.ndarray,
    center: tuple[int, int, int],
    L: float,
    bins: int = 48,
    max_radius: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(field, dtype=np.float64)
    n = arr.shape[0]
    dx = float(L) / float(n)
    if max_radius is None:
        max_radius = float(L) / 2.0
    axes = []
    for axis, c in zip(arr.shape, center):
        coords = np.arange(axis, dtype=np.float64) - float(c)
        coords = (coords + axis / 2.0) % axis - axis / 2.0
        axes.append(coords * dx)
    x, y, z = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    r = np.sqrt(x * x + y * y + z * z)
    edges = np.linspace(0.0, float(max_radius), bins + 1)
    radii = 0.5 * (edges[:-1] + edges[1:])
    values = np.full(bins, np.nan, dtype=np.float64)
    for i in range(bins):
        mask = (r >= edges[i]) & (r < edges[i + 1])
        if np.any(mask):
            values[i] = float(np.mean(arr[mask]))
    return radii, values


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fieldnames:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_summary(run_dir: Path) -> dict[str, Any]:
    with (run_dir / "summary.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_field(path: Path) -> np.ndarray:
    arr = np.load(path)
    return np.asarray(arr)


def field_rho_stats(psi: np.ndarray) -> dict[str, float]:
    rho = np.abs(psi) ** 2
    positive = rho[rho > 0]
    quantiles = np.quantile(positive, [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]) if positive.size else np.zeros(7)
    return {
        "rho_min": float(np.min(rho)),
        "rho_positive_min": float(np.min(positive)) if positive.size else 0.0,
        "rho_max": float(np.max(rho)),
        "rho_mean": float(np.mean(rho)),
        "rho_q001": float(quantiles[0]),
        "rho_q01": float(quantiles[1]),
        "rho_q10": float(quantiles[2]),
        "rho_q50": float(quantiles[3]),
        "rho_q90": float(quantiles[4]),
        "rho_q99": float(quantiles[5]),
        "rho_q999": float(quantiles[6]),
    }


def collect_metadata(repo: Path, out_dir: Path, run_dirs: list[Path]) -> dict[str, Any]:
    _, commit = run_command(["git", "rev-parse", "HEAD"], repo)
    _, status = run_command(["git", "status", "--short"], repo)
    protected_args = ["git", "diff", "--"] + PROTECTED_FILES
    _, protected_diff = run_command(protected_args, repo)
    (out_dir / "protected_diff.txt").write_text(protected_diff + ("\n" if protected_diff else ""), encoding="utf-8")

    cupy_info: dict[str, Any] = {"available": False}
    try:
        import cupy as cp  # type: ignore

        count = int(cp.cuda.runtime.getDeviceCount())
        name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode("utf-8") if count else ""
        cupy_info = {"available": True, "version": cp.__version__, "device_count": count, "gpu_name": name}
    except Exception as exc:  # pragma: no cover - depends on runtime
        cupy_info = {"available": False, "error": repr(exc)}

    checksums = {}
    for run_dir in run_dirs:
        for name in ["summary.json", "load_relaxed.npy", "load_probe_on.npy", "load_probe_off.npy"]:
            path = run_dir / name
            if path.exists():
                checksums[str(path.relative_to(repo))] = sha256_file(path)

    metadata = {
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "python": sys.executable,
        "platform": platform.platform(),
        "git_commit": commit.strip(),
        "git_status_short": status,
        "protected_diff_empty": protected_diff.strip() == "",
        "cupy": cupy_info,
        "production_geometry_params": production_geometry_params(),
        "source_run_dirs": [str(p) for p in run_dirs],
        "source_checksums": checksums,
        "attestation": {
            "no_production_solver_change": True,
            "no_hunter_validation_config_change": True,
            "no_probe_or_path_bending_run": True,
            "read_existing_outputs_only": True,
            "no_gravity_claim": True,
        },
    }
    (out_dir / "geometry_characterization_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata


def build_curve_outputs(params: dict[str, float], out_dir: Path, rho_min: float, rho_max: float) -> dict[str, Any]:
    rho_grid = np.logspace(math.log10(max(rho_min, 1e-12)), math.log10(max(rho_max, rho_min * 10)), 512)
    curve_rows = evaluate_geometry_curve(rho_grid, params)
    write_csv(out_dir / "geometry_law_curve.csv", curve_rows)

    arrays = evaluate_curve_arrays(rho_grid, params)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(arrays["rho"], arrays["omega_sq_nominal"], label="nominal (rho_vac/rho)^a", linewidth=1.5)
    ax.loglog(arrays["rho"], arrays["omega_sq_impl"], label="implemented soft-clipped Omega^2", linewidth=2)
    ax.axhline(params["omega_sq_max"], color="0.5", linestyle="--", linewidth=1, label="cap")
    ax.set_xlabel("rho")
    ax.set_ylabel("Omega^2")
    ax.set_title("Production geometry law over observed/load rho range")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "geometry_law_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(arrays["rho"], arrays["log_slope"], linewidth=2)
    ax.axhline(-params["a_coupling"], color="0.5", linestyle="--", label="nominal slope")
    ax.axhline(0.0, color="0.7", linewidth=1)
    ax.set_xlabel("rho")
    ax.set_ylabel("d log(Omega^2) / d log(rho)")
    ax.set_title("Local log-slope of implemented geometry law")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "geometry_law_log_slope.png", dpi=160)
    plt.close(fig)

    counts: dict[str, int] = {}
    for row in curve_rows:
        counts[row["region"]] = counts.get(row["region"], 0) + 1
    return {"rho_min": float(rho_grid[0]), "rho_max": float(rho_grid[-1]), "region_counts": counts}


def run_softclip_scan(base_params: dict[str, float], out_dir: Path) -> list[dict[str, Any]]:
    rho = np.logspace(-4, math.log10(2.0), 512)
    betas = [0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    mins = [1e-12, 1e-9, 1e-6, 1e-3, 1e-2, 1e-1]
    maxs = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
    rows: list[dict[str, Any]] = []
    for beta in betas:
        for omega_min in mins:
            for omega_max in maxs:
                if omega_min >= omega_max:
                    continue
                params = dict(base_params)
                params.update({"softclip_beta": beta, "omega_sq_min": omega_min, "omega_sq_max": omega_max})
                curve = evaluate_curve_arrays(rho, params)
                score = score_scan_candidate(rho, curve, params)
                rows.append(
                    {
                        "softclip_beta": beta,
                        "omega_sq_min": omega_min,
                        "omega_sq_max": omega_max,
                        **score,
                        "diagnostic_only": True,
                    }
                )
    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    write_csv(out_dir / "softclip_parameter_scan.csv", rows)

    top = rows[:40]
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"b={r['softclip_beta']}, [{r['omega_sq_min']:.0e},{r['omega_sq_max']:.0e}]" for r in top[:15]]
    scores = [float(r["score"]) for r in top[:15]]
    ax.barh(np.arange(len(scores))[::-1], scores[::-1])
    ax.set_yticks(np.arange(len(scores))[::-1], labels[::-1], fontsize=7)
    ax.set_xlabel("diagnostic graded-curve score")
    ax.set_title("Top diagnostic soft-clip scan candidates")
    fig.tight_layout()
    fig.savefig(out_dir / "softclip_scan_top_candidates.png", dpi=170)
    plt.close(fig)
    return rows


def analyze_radial_profiles(repo: Path, run_dirs: list[Path], params: dict[str, float], out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        run_name = run_dir.name
        relaxed = load_field(run_dir / "load_relaxed.npy")
        rho_relaxed = np.abs(relaxed) ** 2
        center = tuple(int(i) for i in np.unravel_index(np.argmax(rho_relaxed), rho_relaxed.shape))
        for field_name in ["load_relaxed.npy", "load_probe_on.npy", "load_probe_off.npy"]:
            path = run_dir / field_name
            if not path.exists():
                continue
            psi = load_field(path)
            rho = np.abs(psi) ** 2
            _, omega = omega_sq_from_rho(rho, params)
            radii, rho_prof = radial_profile(rho, center=center, L=10.0, bins=48)
            _, omega_prof = radial_profile(omega, center=center, L=10.0, bins=48)
            far_omega = float(np.nanmean(omega_prof[-5:]))
            for r, rv, ov in zip(radii, rho_prof, omega_prof):
                rows.append(
                    {
                        "run": run_name,
                        "field": field_name,
                        "radius": float(r),
                        "rho_mean": float(rv),
                        "omega_sq_mean": float(ov),
                        "omega_sq_minus_far": float(ov - far_omega),
                        "center_i": center[0],
                        "center_j": center[1],
                        "center_k": center[2],
                    }
                )
    write_csv(out_dir / "radial_profiles.csv", rows)

    fig, ax = plt.subplots(figsize=(7, 5))
    for run_dir in run_dirs:
        for field_name, style in [("load_relaxed.npy", "-"), ("load_probe_on.npy", "--"), ("load_probe_off.npy", ":")]:
            subset = [r for r in rows if r["run"] == run_dir.name and r["field"] == field_name]
            if not subset:
                continue
            ax.plot(
                [r["radius"] for r in subset],
                [r["omega_sq_mean"] for r in subset],
                style,
                label=f"{run_dir.name}/{field_name.replace('.npy', '')}",
                linewidth=1.6,
            )
    ax.set_xlabel("radius")
    ax.set_ylabel("radial mean Omega^2")
    ax.set_yscale("log")
    ax.set_title("Saved Rung A+D radial Omega^2 profiles")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "radial_omega_profiles.png", dpi=170)
    plt.close(fig)
    return rows


def summarize_runs(run_dirs: list[Path], out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    observed_min = 1e-4
    observed_max = 2.0
    for run_dir in run_dirs:
        summary = load_summary(run_dir)
        for name in ["load_relaxed.npy", "load_probe_on.npy", "load_probe_off.npy"]:
            path = run_dir / name
            if not path.exists():
                continue
            stats = field_rho_stats(load_field(path))
            observed_min = min(observed_min, max(stats["rho_q001"], 1e-12))
            observed_max = max(observed_max, stats["rho_max"])
            rows.append({"run": run_dir.name, "field": name, **stats})
        load = summary.get("load", {})
        rows.append(
            {
                "run": run_dir.name,
                "field": "summary_load_metrics",
                "rho_min": summary.get("geometry_law", {}).get("rho_range", [None, None])[0],
                "rho_max": summary.get("geometry_law", {}).get("rho_range", [None, None])[1],
                "rho_mean": "",
                "rho_q001": "",
                "rho_q01": "",
                "rho_q10": "",
                "rho_q50": "",
                "rho_q90": "",
                "rho_q99": "",
                "rho_q999": "",
                "mass": load.get("mass"),
                "amp": load.get("amp"),
                "n_nodes": load.get("n_nodes"),
                "omega_center": load.get("omega_center"),
                "omega_far": load.get("omega_far"),
                "omega_well": load.get("omega_well"),
                "tensor_shear": load.get("tensor_shear"),
                "verdict": summary.get("verdict"),
            }
        )
    write_csv(out_dir / "source_run_summary.csv", rows)
    return rows, {"rho_min": float(observed_min), "rho_max": float(observed_max)}


def write_report(
    out_dir: Path,
    metadata: dict[str, Any],
    run_rows: list[dict[str, Any]],
    curve_summary: dict[str, Any],
    scan_rows: list[dict[str, Any]],
    radial_rows: list[dict[str, Any]],
) -> None:
    top = scan_rows[:5]
    bg0_summary = next((r for r in run_rows if r["run"] == "GRAVITY_AD_bg0" and r["field"] == "summary_load_metrics"), {})
    bgvac_summary = next((r for r in run_rows if r["run"] == "GRAVITY_AD_bgvac" and r["field"] == "summary_load_metrics"), {})

    production = metadata["production_geometry_params"]
    protected = "empty" if metadata["protected_diff_empty"] else "non-empty"
    lines = [
        "# IRER Gravity Ladder Geometry Characterization / De-Saturation Audit",
        "",
        "Scope: read-only characterization of existing Rung A+D telemetry. No probe/path-bending run was performed, "
        "and no production solver, Hunter, validation, Phase C path, or default config was modified.",
        "",
        "## Inputs",
        "",
        f"- Source runs: `{', '.join(Path(p).name for p in metadata['source_run_dirs'])}`",
        f"- Git commit: `{metadata['git_commit']}`",
        f"- Python: `{metadata['python']}`",
        f"- CuPy/GPU metadata: `{metadata['cupy']}`",
        f"- Protected-file diff: `{protected}`",
        "",
        "## Production Geometry Law",
        "",
        f"- rho_vac: `{production['rho_vac']}`",
        f"- a_coupling: `{production['a_coupling']}`",
        f"- softclip beta: `{production['softclip_beta']}`",
        f"- Omega^2 window: `[{production['omega_sq_min']}, {production['omega_sq_max']}]`",
        f"- Curve evaluation range: rho `{curve_summary['rho_min']:.3e}` to `{curve_summary['rho_max']:.3e}`",
        f"- Region counts across sampled curve: `{curve_summary['region_counts']}`",
        "",
        "The implemented Omega^2(rho) curve is dominated by the log-space soft clip over the saved load density range. "
        "The bg0 radial profile retains the previously observed pattern: a low, nearly flat dense-core response followed "
        "by a jump toward the high-Omega^2 ambient/cap region near the load boundary. This is a geometry-law "
        "characterization result only, not a gravity interpretation.",
        "",
        "## Saved Rung A+D Cross-Check",
        "",
        f"- bg0 verdict: `{bg0_summary.get('verdict')}`; n_nodes `{bg0_summary.get('n_nodes')}`, "
        f"omega_center `{bg0_summary.get('omega_center')}`, omega_far `{bg0_summary.get('omega_far')}`, "
        f"tensor_shear `{bg0_summary.get('tensor_shear')}`.",
        f"- bgvac verdict: `{bgvac_summary.get('verdict')}`; n_nodes `{bgvac_summary.get('n_nodes')}`, "
        f"omega_center `{bgvac_summary.get('omega_center')}`, omega_far `{bgvac_summary.get('omega_far')}`.",
        "",
        "## Diagnostic Soft-Clip Scan",
        "",
        "The scan below is diagnostic-only. It does not change defaults and should be read as a map of possible "
        "non-saturated parameter regimes for future geometry-contract review.",
        "",
        "| rank | beta | omega_min | omega_max | score | graded_fraction | cap_fraction | flat_fraction | dyn_range_log10 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(top, start=1):
        lines.append(
            f"| {i} | {row['softclip_beta']} | {row['omega_sq_min']:.0e} | {row['omega_sq_max']:.0e} | "
            f"{row['score']:.3f} | {row['graded_fraction']:.3f} | {row['cap_fraction']:.3f} | "
            f"{row['flat_fraction']:.3f} | {row['dynamic_range_log10']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## C2 Prime Cross-Reference",
            "",
            "`docs/PHASE_D_C2PRIME_CANONICAL_GEOMETRY_RFC.md` is the relevant principled alternative to the current "
            "soft-clip-dominated geometry contract. This audit does not select or alter that contract; it provides the "
            "geometry-law data needed for that decision.",
            "",
            "## Outputs",
            "",
            "- `geometry_characterization_metadata.json`",
            "- `source_run_summary.csv`",
            "- `geometry_law_curve.csv` and `geometry_law_curve.png`",
            "- `geometry_law_log_slope.png`",
            "- `softclip_parameter_scan.csv` and `softclip_scan_top_candidates.png`",
            "- `radial_profiles.csv` and `radial_omega_profiles.png`",
            "",
            "## Reproducible Command",
            "",
            "```powershell",
            "F:\\quantule_mapper\\.venv\\Scripts\\python.exe tools\\analyze_gravity_geometry_characterization.py "
            "--out F:\\quantule_mapper\\quantule_viz\\outputs\\gravity_geometry_characterization",
            "```",
            "",
            "## Attestation",
            "",
            "- No production solver change.",
            "- No Hunter, validation, default config, or Phase C path change.",
            "- No probe/path-bending run.",
            "- No gravity claim.",
            "- Existing saved outputs were used for telemetry.",
        ]
    )
    report = "\n".join(lines) + "\n"
    (out_dir / "geometry_characterization_report.md").write_text(report, encoding="utf-8")

    handoff = out_dir / "IRER_GRAVITY_GEOMETRY_CHARACTERIZATION_AUDIT.md"
    handoff.write_text(report, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=Path("quantule_viz/outputs/gravity_geometry_characterization"))
    parser.add_argument("--timestamped", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bg0", type=Path, default=Path("sweep_runs/GRAVITY_AD_bg0"))
    parser.add_argument("--bgvac", type=Path, default=Path("sweep_runs/GRAVITY_AD_bgvac"))
    args = parser.parse_args(argv)

    repo = args.repo.resolve()
    run_dirs = [(repo / args.bg0).resolve(), (repo / args.bgvac).resolve()]
    for run_dir in run_dirs:
        if not (run_dir / "summary.json").exists():
            raise FileNotFoundError(f"Missing saved Rung A+D summary: {run_dir / 'summary.json'}")

    root_out = (repo / args.out).resolve() if not args.out.is_absolute() else args.out.resolve()
    if args.timestamped:
        out_dir = root_out / f"geometry_characterization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        out_dir = root_out
    out_dir.mkdir(parents=True, exist_ok=True)

    params = production_geometry_params()
    metadata = collect_metadata(repo, out_dir, run_dirs)
    run_rows, observed = summarize_runs(run_dirs, out_dir)
    curve_summary = build_curve_outputs(params, out_dir, min(observed["rho_min"], 1e-4), max(observed["rho_max"], 2.0))
    scan_rows = run_softclip_scan(params, out_dir)
    radial_rows = analyze_radial_profiles(repo, run_dirs, params, out_dir)
    write_report(out_dir, metadata, run_rows, curve_summary, scan_rows, radial_rows)

    summary = {
        "output_dir": str(out_dir),
        "observed_rho_range": observed,
        "curve_summary": curve_summary,
        "top_softclip_candidates": scan_rows[:10],
        "protected_diff_empty": metadata["protected_diff_empty"],
        "decision": "GEOMETRY_CHARACTERIZED_RUNG_B_REMAINS_BLOCKED",
        "no_gravity_claim": True,
    }
    (out_dir / "geometry_characterization_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
