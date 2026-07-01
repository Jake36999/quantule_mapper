"""Long-time core characterization saved-result renderer."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from ..io import IncompleteRunError, guard_outputs, load_json, resolve_run_dir
from ..plots import save_figure

PATTERNS = ["CORE_BASIN_*/core_characterize"]
OUTPUTS = ["persistence.png", "profiles.png", "perturbation.png"]


def _load_run(run_dir: Path) -> dict[str, object]:
    result_path = run_dir / "core_characterize.json"
    if not result_path.exists():
        raise IncompleteRunError(f"{run_dir} is incomplete: missing core_characterize.json")
    return load_json(result_path)


def _plot_persistence(result: dict[str, object], outdir: Path) -> Path:
    targets = result["targets"]  # type: ignore[index]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for target_id, payload in targets.items():
        series = payload["series"]
        times = [row["t_step"] for row in series]
        label = f"idx{target_id} eta={payload['eta']:+.3f} [{payload['outcome']}]"
        axes[0].plot(times, [row["er"] for row in series], "-o", ms=2, label=label)
        axes[1].plot(times, [row.get("core_rho", float("nan")) for row in series], "-o", ms=2, label=f"idx{target_id}")
        axes[2].plot(times, [row.get("v_t", float("nan")) for row in series], "-o", ms=2, label=f"idx{target_id}")
    axes[0].set_title("energy ratio er(t)")
    axes[1].set_title("core density(t)")
    axes[2].set_title("tangential v_t(t)")
    for axis in axes:
        axis.set_xlabel("t step")
        axis.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Long-time core characterization (stored outcomes only)", fontsize=13)
    return save_figure(fig, outdir / "persistence.png")


def _plot_profiles(result: dict[str, object], outdir: Path) -> Path:
    profiles = result.get("primary_profiles") or {}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for label, profile in profiles.items():
        axes[0].plot(profile["r"], profile["rho"], "-o", ms=3, label=f"{label} (t={profile['t_step']})")
        axes[1].plot(profile["r"], profile["v_r"], "-o", ms=3, label=label)
        axes[2].plot(profile["r"], profile["v_t"], "-o", ms=3, label=label)
    axes[0].set_title("rho(r)")
    axes[1].set_title("v_r(r)")
    axes[1].axhline(0.0, color="k", lw=0.6)
    axes[2].set_title("v_t(r)")
    for axis in axes:
        axis.set_xlabel("r (voxels)")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    fig.suptitle("Primary-core radial profiles", fontsize=13)
    return save_figure(fig, outdir / "profiles.png")


def _plot_perturbation(result: dict[str, object], outdir: Path) -> Path:
    perturbation = result.get("primary_perturbation")
    if not perturbation:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.text(0.5, 0.5, "No primary_perturbation block is stored for this run.", ha="center", va="center")
        ax.set_axis_off()
        ax.set_title("Perturbation response")
        return save_figure(fig, outdir / "perturbation.png")
    series = perturbation["series"]
    times = [row["t_step"] for row in series]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(times, [row["base_core"] for row in series], "-o", ms=3, label="unperturbed", color="tab:blue")
    axes[0].plot(times, [row["kick_core"] for row in series], "-o", ms=3, label="kicked", color="tab:red")
    axes[0].set_title("core density: kicked vs unperturbed")
    axes[1].plot(times, [row["base_vt"] for row in series], "-o", ms=3, label="unperturbed", color="tab:blue")
    axes[1].plot(times, [row["kick_vt"] for row in series], "-o", ms=3, label="kicked", color="tab:red")
    axes[1].set_title("tangential v_t: kicked vs unperturbed")
    for axis in axes:
        axis.set_xlabel("t step after kick")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    fig.suptitle("Primary-core perturbation response", fontsize=13)
    return save_figure(fig, outdir / "perturbation.png")


def render(
    run_dir: str | None,
    *,
    outdir: str | None,
    overwrite: bool,
    latest: bool,
    candidate: str | None,
) -> list[Path]:
    del candidate
    resolved = resolve_run_dir(run_dir, latest=latest, patterns=PATTERNS, exclude_parts=("REFINE", "CALIB"))
    output_dir = Path(outdir).resolve() if outdir else resolved
    result = _load_run(resolved)
    guard_outputs(output_dir, OUTPUTS, overwrite)
    return [
        _plot_persistence(result, output_dir),
        _plot_profiles(result, output_dir),
        _plot_perturbation(result, output_dir),
    ]
