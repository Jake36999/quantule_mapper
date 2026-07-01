"""feb56dc7 saved-result renderer."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ..io import IncompleteRunError, choose_npz_key, guard_outputs, load_json, load_npz, resolve_run_dir
from ..plots import find_component_centroids, render_density_slices, save_figure, vector_slice
from . import LONG_TIME_STABLE_4_NODE_ATTRACTOR

PATTERNS = ["SUBSTRATE_HUNT_*/feb56dc7_bound_state"]
OUTPUTS = [
    "relaxation_timeline.png",
    "node_tracks.png",
    "node_geometry.png",
    "pairwise_distances.png",
    "density_slices.png",
    "vector_maps.png",
    "radial_profiles.png",
    "perturbation_summary.png",
]


def _load_run(run_dir: Path) -> tuple[dict[str, object], dict[str, object] | None, np.ndarray]:
    result_path = run_dir / "feb_bound_state.json"
    frames_path = run_dir / "frames.npz"
    if not result_path.exists():
        raise IncompleteRunError(f"{run_dir} is incomplete: missing feb_bound_state.json")
    if not frames_path.exists():
        raise IncompleteRunError(f"{run_dir} is incomplete: missing frames.npz")
    result = load_json(result_path)
    bond_path = run_dir / "feb_bond_test.json"
    bond = load_json(bond_path) if bond_path.exists() and bond_path.stat().st_size > 0 else None
    bundle = load_npz(frames_path)
    key = choose_npz_key(bundle, preferred=("psi",))
    frames = np.asarray(bundle[key])
    if frames.ndim != 4:
        raise IncompleteRunError(f"{frames_path} is incomplete: expected 4D psi frames, got {frames.shape}")
    return result, bond, frames


def _pairwise_matrix(centroids: list[list[float]]) -> np.ndarray:
    points = np.asarray(centroids, dtype=float)
    if len(points) == 0:
        return np.zeros((0, 0), dtype=float)
    delta = points[:, None, :] - points[None, :, :]
    return np.sqrt((delta**2).sum(axis=-1))


def _plot_relaxation(result: dict[str, object], outdir: Path) -> Path:
    dynamics = list(result["dynamics"])  # type: ignore[index]
    times = [row["t_step"] for row in dynamics]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    panels = axes.ravel()
    panels[0].plot(times, [row["er"] for row in dynamics], "-o", ms=2)
    panels[0].axhspan(0.5, 2.0, color="#1a9850", alpha=0.06)
    panels[0].set_title("energy ratio er(t)")
    panels[1].plot(times, [row["n_nodes"] for row in dynamics], "-o", ms=2, color="tab:purple")
    panels[1].set_title("node count(t)")
    panels[2].errorbar(
        times,
        [row["dist_mean"] for row in dynamics],
        yerr=[row["dist_std"] for row in dynamics],
        fmt="-o",
        ms=2,
        color="tab:brown",
    )
    panels[2].set_title("pairwise distance mean +/- std")
    panels[3].plot(times, [row["core_rho"] for row in dynamics], "-o", ms=2, color="tab:green")
    panels[3].set_title("dominant core density(t)")
    panels[4].plot(times, [row["v_t"] for row in dynamics], "-o", ms=2, color="tab:blue", label="v_t")
    panels[4].plot(times, [row["v_r"] for row in dynamics], "-o", ms=2, color="tab:red", label="v_r")
    panels[4].axhline(0.0, color="k", lw=0.6)
    panels[4].legend(fontsize=8)
    panels[4].set_title("circulation decay")
    metadata = [
        f"label: {LONG_TIME_STABLE_4_NODE_ATTRACTOR}",
        f"saved_result_id: {result.get('result_id', 'unknown')}",
        f"bond_verdict: {result.get('bond_verdict', 'unknown')}",
        f"final nodes: {result['final_geometry']['n_nodes']}",  # type: ignore[index]
        f"rotating_at_end: {result.get('rotating_at_end', 'unknown')}",
    ]
    panels[5].text(0.03, 0.97, "\n".join(metadata), va="top", family="monospace", fontsize=10.5)
    panels[5].set_axis_off()
    panels[5].set_title("stored metadata")
    for axis in panels[:5]:
        axis.set_xlabel("t step")
        axis.grid(alpha=0.3)
    fig.suptitle("feb56dc7 relaxation timeline", fontsize=13)
    return save_figure(fig, outdir / "relaxation_timeline.png")


def _plot_node_tracks(frames: np.ndarray, outdir: Path) -> Path:
    projections = [(0, 1, "x-y"), (0, 2, "x-z"), (1, 2, "y-z")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(frames)))
    for axis, (dim_x, dim_y, label) in zip(axes, projections):
        for idx, frame in enumerate(frames):
            for centroid in find_component_centroids(frame):
                axis.scatter(centroid[dim_x], centroid[dim_y], s=22, color=colors[idx], alpha=0.85)
        axis.set_title(f"node tracks {label}")
        axis.set_aspect("equal")
        axis.grid(alpha=0.2)
    fig.suptitle("Node tracks from saved frame bundle", fontsize=13)
    return save_figure(fig, outdir / "node_tracks.png")


def _plot_node_geometry(result: dict[str, object], outdir: Path) -> Path:
    centroids = result["final_geometry"]["centroids"]  # type: ignore[index]
    points = np.asarray(centroids, dtype=float)
    projections = [(0, 1, "x-y"), (0, 2, "x-z"), (1, 2, "y-z")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for axis, (dim_x, dim_y, label) in zip(axes, projections):
        axis.scatter(points[:, dim_x], points[:, dim_y], s=70, c=np.arange(len(points)), cmap="tab10")
        for i, j in combinations(range(len(points)), 2):
            axis.plot(
                [points[i, dim_x], points[j, dim_x]],
                [points[i, dim_y], points[j, dim_y]],
                color="0.75",
                lw=0.8,
            )
        axis.set_title(f"final centroid geometry {label}")
        axis.set_aspect("equal")
        axis.grid(alpha=0.2)
    fig.suptitle(LONG_TIME_STABLE_4_NODE_ATTRACTOR, fontsize=13)
    return save_figure(fig, outdir / "node_geometry.png")


def _plot_pairwise_distances(result: dict[str, object], outdir: Path) -> Path:
    dynamics = list(result["dynamics"])  # type: ignore[index]
    times = [row["t_step"] for row in dynamics]
    pairs = result["final_geometry"]["pair_dists"]  # type: ignore[index]
    labels = [f"{pair['pair'][0]}-{pair['pair'][1]}" for pair in pairs]
    distances = [pair["dist"] for pair in pairs]
    matrix = _pairwise_matrix(result["final_geometry"]["centroids"])  # type: ignore[index]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].errorbar(
        times,
        [row["dist_mean"] for row in dynamics],
        yerr=[row["dist_std"] for row in dynamics],
        fmt="-o",
        ms=2,
        color="tab:brown",
    )
    axes[0].set_title("pairwise distance mean +/- std")
    axes[0].set_xlabel("t step")
    axes[0].grid(alpha=0.3)

    axes[1].bar(labels, distances, color="tab:olive")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set_title("final pair distances")
    axes[1].grid(axis="y", alpha=0.25)

    image = axes[2].imshow(matrix, cmap="viridis")
    axes[2].set_title("final centroid distance matrix")
    axes[2].set_xticks(range(len(matrix)))
    axes[2].set_yticks(range(len(matrix)))
    fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
    fig.suptitle("Pairwise distance summary", fontsize=13)
    return save_figure(fig, outdir / "pairwise_distances.png")


def _plot_density_slices(frames: np.ndarray, outdir: Path) -> Path:
    return render_density_slices(frames, outdir, filename="density_slices.png")


def _plot_vector_maps(frames: np.ndarray, outdir: Path) -> Path:
    picks = [0, len(frames) - 1] if len(frames) > 1 else [0]
    fig, axes = plt.subplots(len(picks), 2, figsize=(12, 5 * len(picks)))
    if len(picks) == 1:
        axes = np.asarray([axes])
    for row_axes, frame_idx in zip(axes, picks):
        rho_slice, u, v, vort = vector_slice(np.asarray(frames[frame_idx], dtype=np.complex128))
        vmax = float(np.percentile(np.abs(vort), 99)) if np.any(vort) else 1.0
        row_axes[0].imshow(rho_slice.T, origin="lower", cmap="gray_r")
        step = max(1, rho_slice.shape[0] // 18)
        xx, yy = np.meshgrid(
            np.arange(0, rho_slice.shape[0], step),
            np.arange(0, rho_slice.shape[1], step),
            indexing="ij",
        )
        row_axes[0].quiver(xx, yy, u[::step, ::step], v[::step, ::step], color="#1f77b4", width=0.004)
        row_axes[0].set_title(f"frame {frame_idx}: saved-current preview")
        row_axes[0].set_xticks([])
        row_axes[0].set_yticks([])

        row_axes[1].imshow(vort.T, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        row_axes[1].set_title(f"frame {frame_idx}: pseudo-vorticity")
        row_axes[1].set_xticks([])
        row_axes[1].set_yticks([])
    fig.suptitle("Vector-map previews from saved frames", fontsize=13)
    return save_figure(fig, outdir / "vector_maps.png")


def _plot_radial_profiles(result: dict[str, object], outdir: Path) -> Path:
    profiles = result["node_profiles"]  # type: ignore[index]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for name, profile in profiles.items():
        axes[0].plot(profile["r"], profile["rho"], "-o", ms=3, label=name)
        axes[1].plot(profile["r"], profile["v_r"], "-o", ms=3, label=name)
        axes[2].plot(profile["r"], profile["v_t"], "-o", ms=3, label=name)
    axes[0].set_title("rho(r) per node")
    axes[1].set_title("v_r(r)")
    axes[1].axhline(0.0, color="k", lw=0.6)
    axes[2].set_title("v_t(r)")
    for axis in axes:
        axis.set_xlabel("r (voxels)")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    fig.suptitle("Per-node radial profiles", fontsize=13)
    return save_figure(fig, outdir / "radial_profiles.png")


def _plot_perturbation(result: dict[str, object], bond: dict[str, object] | None, outdir: Path) -> Path:
    perturbations = result["perturbation_tests"]  # type: ignore[index]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for name, payload in perturbations.items():
        series = payload["series"]
        times = [row["t_step"] for row in series]
        axes[0].plot(times, [row["n_nodes"] for row in series], "-o", ms=3, label=f"{name} [{payload['klass']}]")
        axes[1].plot(times, [row["dist_mean"] for row in series], "-o", ms=3, label=name)
    base_nodes = result["final_geometry"]["n_nodes"]  # type: ignore[index]
    base_dist = float(np.mean([item["dist"] for item in result["final_geometry"]["pair_dists"]]))  # type: ignore[index]
    axes[0].axhline(base_nodes, color="k", ls="--", lw=0.8, label="baseline")
    axes[1].axhline(base_dist, color="k", ls="--", lw=0.8, label="baseline")
    axes[0].set_title("n_nodes after perturbation")
    axes[1].set_title("pairwise distance after perturbation")
    axes[0].set_xlabel("t step after perturbation")
    axes[1].set_xlabel("t step after perturbation")
    axes[0].grid(alpha=0.3)
    axes[1].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)

    notes = [f"label: {LONG_TIME_STABLE_4_NODE_ATTRACTOR}"]
    if bond:
        for test_name, payload in bond.get("tests", {}).items():
            notes.append(f"{test_name}: {payload.get('klass', 'unknown')}")
    else:
        notes.append("feb_bond_test.json not present")
    axes[2].text(0.03, 0.97, "\n".join(notes), va="top", family="monospace", fontsize=10.5)
    axes[2].set_axis_off()
    axes[2].set_title("stored perturbation verdicts")
    fig.suptitle("Perturbation summary", fontsize=13)
    return save_figure(fig, outdir / "perturbation_summary.png")


def render(
    run_dir: str | None,
    *,
    outdir: str | None,
    overwrite: bool,
    latest: bool,
) -> list[Path]:
    resolved = resolve_run_dir(run_dir, latest=latest, patterns=PATTERNS)
    output_dir = Path(outdir).resolve() if outdir else resolved
    result, bond, frames = _load_run(resolved)
    guard_outputs(output_dir, OUTPUTS, overwrite)
    return [
        _plot_relaxation(result, output_dir),
        _plot_node_tracks(frames, output_dir),
        _plot_node_geometry(result, output_dir),
        _plot_pairwise_distances(result, output_dir),
        _plot_density_slices(frames, output_dir),
        _plot_vector_maps(frames, output_dir),
        _plot_radial_profiles(result, output_dir),
        _plot_perturbation(result, bond, output_dir),
    ]
