"""Shared plotting and field helpers for saved Quantule Mapper artifacts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRUE_CLASS = "TRUE_SATURATED_BOUND_STATE"
NEAR_CLASS = "NEAR_SATURATED_BOUND_STATE"

CLASS_COLORS = {
    TRUE_CLASS: "#1a9850",
    NEAR_CLASS: "#66bd63",
    "TRANSIENT_GROWER_REJECT": "#fdae61",
    "LATE_BLOWUP_REJECT": "#d73027",
    "FRAGMENTATION_REJECT": "#984ea3",
    "DELOCALIZED_HALO_REJECT": "#999999",
    "SPIN_DOWN_REJECT": "#4575b4",
    "WINDOW_ARTIFACT_REJECT": "#8c510a",
    "SUSTAIN": "#1a9850",
    "SPIN_DOWN": "#fdae61",
    "COLLAPSE": "#4575b4",
    "BLOWUP": "#d73027",
    "FRAGMENT": "#984ea3",
    "VIABLE_NO_NODES": "#999999",
}

NEIGHBOR_OFFSETS = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
]


def save_figure(fig: plt.Figure, out: str | Path, *, dpi: int = 130) -> Path:
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(target, dpi=dpi)
    plt.close(fig)
    return target


def class_counts(rows: list[dict[str, str]]) -> Counter[str]:
    return Counter(row.get("klass", "UNKNOWN") for row in rows)


def ordered_class_labels(counts: Counter[str]) -> list[str]:
    return sorted(counts, key=lambda key: (-counts[key], key))


def blank_figure(message: str, out: str | Path, *, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11)
    ax.set_axis_off()
    ax.set_title(title)
    return save_figure(fig, out)


def as_density(field: np.ndarray) -> np.ndarray:
    return np.abs(field) ** 2 if np.iscomplexobj(field) else np.asarray(field, dtype=float)


def density_slice(field: np.ndarray) -> tuple[np.ndarray, int, int]:
    rho = as_density(field)
    axis_sums = [rho.sum(axis=tuple(dim for dim in range(3) if dim != axis)) for axis in range(3)]
    axis = int(np.argmax([float(s.max()) for s in axis_sums]))
    center = int(np.unravel_index(int(np.argmax(rho)), rho.shape)[axis])
    if axis == 0:
        return rho[center, :, :], axis, center
    if axis == 1:
        return rho[:, center, :], axis, center
    return rho[:, :, center], axis, center


def phase_current(psi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    field = np.asarray(psi, dtype=np.complex128)
    grads = np.gradient(field)
    current = tuple(np.imag(np.conj(field) * grad) for grad in grads)
    return current  # type: ignore[return-value]


def vector_slice(psi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rho_slice, axis, center = density_slice(psi)
    jx, jy, jz = phase_current(psi)
    if axis == 0:
        u = jy[center, :, :]
        v = jz[center, :, :]
    elif axis == 1:
        u = jx[:, center, :]
        v = jz[:, center, :]
    else:
        u = jx[:, :, center]
        v = jy[:, :, center]
    vorticity = np.gradient(v, axis=0) - np.gradient(u, axis=1)
    return rho_slice, u, v, vorticity


def find_component_centroids(
    field: np.ndarray,
    *,
    quantile: float = 0.997,
    min_size: int = 4,
) -> list[np.ndarray]:
    rho = as_density(field)
    threshold = float(np.quantile(rho, quantile))
    coords = np.argwhere(rho > threshold)
    if len(coords) == 0:
        return []

    active = {tuple(int(v) for v in coord) for coord in coords}
    visited: set[tuple[int, int, int]] = set()
    centroids: list[np.ndarray] = []

    for seed in active:
        if seed in visited:
            continue
        stack = [seed]
        component: list[tuple[int, int, int]] = []
        while stack:
            point = stack.pop()
            if point in visited:
                continue
            visited.add(point)
            component.append(point)
            px, py, pz = point
            for dx, dy, dz in NEIGHBOR_OFFSETS:
                nxt = (px + dx, py + dy, pz + dz)
                if nxt in active and nxt not in visited:
                    stack.append(nxt)
        if len(component) >= min_size:
            centroids.append(np.mean(np.asarray(component, dtype=float), axis=0))
    return centroids


def render_density_slices(frames: np.ndarray, outdir: str | Path, *, filename: str = "frame_density_slices.png") -> Path:
    total = int(len(frames))
    picks = sorted({0, max(0, total // 2), total - 1})
    fig, axes = plt.subplots(1, len(picks), figsize=(5 * len(picks), 4.4))
    if len(picks) == 1:
        axes = [axes]
    for ax, frame_idx in zip(axes, picks):
        slc, axis, center = density_slice(frames[frame_idx])
        image = ax.imshow(slc.T, origin="lower", cmap="magma")
        ax.set_title(f"frame {frame_idx} axis {axis} slice {center}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Density slices from saved frame bundle", fontsize=12)
    return save_figure(fig, Path(outdir) / filename)


def render_vector_preview(frames: np.ndarray, outdir: str | Path, *, filename: str = "frame_vector_preview.png") -> Path:
    rho_slice, u, v, vort = vector_slice(np.asarray(frames[-1], dtype=np.complex128))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    finite_vort = np.asarray(vort, dtype=float)
    finite_vort = finite_vort[np.isfinite(finite_vort)]
    vmax = float(np.percentile(np.abs(finite_vort), 99)) if finite_vort.size else 1.0

    axes[0].imshow(rho_slice.T, origin="lower", cmap="gray_r")
    step = max(1, rho_slice.shape[0] // 18)
    xx, yy = np.meshgrid(
        np.arange(0, rho_slice.shape[0], step),
        np.arange(0, rho_slice.shape[1], step),
        indexing="ij",
    )
    quiver_u = np.asarray(u[::step, ::step], dtype=float)
    quiver_v = np.asarray(v[::step, ::step], dtype=float)
    mask = np.isfinite(quiver_u) & np.isfinite(quiver_v)
    if np.any(mask):
        axes[0].quiver(xx[mask], yy[mask], quiver_u[mask], quiver_v[mask], color="#1f77b4", width=0.004)
    else:
        axes[0].text(0.5, 0.5, "no coherent finite vector field", transform=axes[0].transAxes, ha="center", va="center", fontsize=10)
    axes[0].set_title("Final-slice current vectors")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    vort_image = np.nan_to_num(np.asarray(vort, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    axes[1].imshow(vort_image.T, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[1].set_title("Final-slice pseudo-vorticity")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    fig.suptitle("Vector preview from saved frame bundle", fontsize=12)
    return save_figure(fig, Path(outdir) / filename)


def render_frame_pack(npz_path: str | Path, outdir: str | Path) -> list[Path]:
    bundle = np.load(npz_path, allow_pickle=True)
    key = "psi" if "psi" in bundle.files else ("frames" if "frames" in bundle.files else bundle.files[0])
    frames = np.asarray(bundle[key])
    if frames.ndim == 3:
        frames = frames[None, ...]
    if frames.ndim != 4:
        raise ValueError(f"Expected a 3D field or time x 3D frame array, got shape {frames.shape}")
    return [
        render_density_slices(frames, outdir),
        render_vector_preview(frames, outdir),
    ]
