"""Read-only field visual-analysis suite for HDF5/NPZ simulation artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import h5py
except Exception:  # pragma: no cover - optional in some local environments
    h5py = None

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover - GIFs become unavailable, not fatal
    imageio = None

try:
    from skimage import measure
except Exception:  # pragma: no cover - topology degrades gracefully
    measure = None


@dataclass(frozen=True)
class FieldArtifact:
    source_path: Path
    rho_frames: np.ndarray
    psi_frames: np.ndarray | None
    times: np.ndarray
    source_kind: str
    source_datasets: dict[str, str]
    telemetry: dict[str, np.ndarray]


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_source(source: str | Path) -> Path:
    path = Path(source)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if path.is_dir():
        candidates = []
        for suffix in ("*.h5", "*.hdf5", "*.npz"):
            candidates.extend(path.glob(suffix))
        if not candidates:
            raise FileNotFoundError(f"No .h5/.hdf5/.npz artifact found in {path}")
        path = sorted(candidates, key=lambda item: item.stat().st_mtime)[-1]
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _as_frame_stack(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr)
    if data.ndim == 3:
        data = data[None, ...]
    if data.ndim != 4:
        raise ValueError(f"Expected 3D or time x 3D field data, got shape {data.shape}")
    return data


def _rho_from_field(data: np.ndarray) -> np.ndarray:
    return (np.abs(data) ** 2 if np.iscomplexobj(data) else np.asarray(data, dtype=float)).astype(np.float32)


def _read_hdf5(source_path: Path) -> FieldArtifact:
    if h5py is None:
        raise RuntimeError("h5py is required to read HDF5 artifacts.")
    with h5py.File(source_path, "r") as handle:
        sources: dict[str, str] = {}
        psi_frames: np.ndarray | None = None
        rho_frames: np.ndarray | None = None

        for key in ("psi_history", "psi_frames", "psi", "frames", "field_history"):
            if key in handle:
                arr = _as_frame_stack(np.asarray(handle[key][()]))
                if np.iscomplexobj(arr):
                    psi_frames = arr
                    sources["psi"] = f"/{key}"
                rho_frames = _rho_from_field(arr)
                sources["rho"] = f"/{key}"
                break
        if rho_frames is None:
            for key in ("rho_history", "rho_frames", "rho"):
                if key in handle:
                    rho_frames = _as_frame_stack(np.asarray(handle[key][()]))
                    sources["rho"] = f"/{key}"
                    break
        if rho_frames is None and "psi_final" in handle:
            psi_frames = _as_frame_stack(np.asarray(handle["psi_final"][()]))
            rho_frames = _rho_from_field(psi_frames)
            sources["psi"] = "/psi_final"
            sources["rho"] = "/psi_final"
        if rho_frames is None and "rho_final" in handle:
            rho_frames = _as_frame_stack(np.asarray(handle["rho_final"][()]))
            sources["rho"] = "/rho_final"
        if rho_frames is None:
            raise ValueError(f"No supported rho/psi field dataset found in {source_path}")

        times = np.arange(len(rho_frames), dtype=float)
        for key in ("times", "time", "t", "steps", "step"):
            if key in handle:
                raw = np.asarray(handle[key][()]).reshape(-1)
                if len(raw) == len(rho_frames):
                    times = raw.astype(float)
                    sources["time"] = f"/{key}"
                    break

        telemetry: dict[str, np.ndarray] = {}
        if "telemetry" in handle:
            group = handle["telemetry"]
            for key in ("step", "time", "energy", "energy_total", "C_invariant", "max_amplitude"):
                if key in group:
                    telemetry[key] = np.asarray(group[key][()]).reshape(-1)
        else:
            for key in ("energy", "energy_total", "C_invariant", "max_amplitude"):
                if key in handle:
                    telemetry[key] = np.asarray(handle[key][()]).reshape(-1)

    return FieldArtifact(
        source_path=source_path,
        rho_frames=rho_frames,
        psi_frames=psi_frames,
        times=times,
        source_kind="hdf5",
        source_datasets=sources,
        telemetry=telemetry,
    )


def _read_npz(source_path: Path) -> FieldArtifact:
    with np.load(source_path, allow_pickle=True) as bundle:
        sources: dict[str, str] = {}
        psi_frames: np.ndarray | None = None
        rho_frames: np.ndarray | None = None
        for key in ("psi", "psi_history", "frames", "field_history"):
            if key in bundle.files:
                arr = _as_frame_stack(np.asarray(bundle[key]))
                if np.iscomplexobj(arr):
                    psi_frames = arr
                    rho_frames = _rho_from_field(arr)
                    sources["psi"] = key
                    sources["rho"] = key
                    break
        if rho_frames is None:
            for key in ("rho", "rho_history", "density"):
                if key in bundle.files:
                    rho_frames = _as_frame_stack(np.asarray(bundle[key]))
                    sources["rho"] = key
                    break
        if rho_frames is None:
            candidates = [key for key in bundle.files if np.asarray(bundle[key]).ndim in {3, 4}]
            if not candidates:
                raise ValueError(f"No supported field array found in {source_path}")
            arr = _as_frame_stack(np.asarray(bundle[candidates[0]]))
            if np.iscomplexobj(arr):
                psi_frames = arr
            rho_frames = _rho_from_field(arr)
            sources["rho"] = candidates[0]
            if psi_frames is not None:
                sources["psi"] = candidates[0]

        times = np.arange(len(rho_frames), dtype=float)
        for key in ("times", "time", "t", "steps"):
            if key in bundle.files:
                raw = np.asarray(bundle[key]).reshape(-1)
                if len(raw) == len(rho_frames):
                    times = raw.astype(float)
                    sources["time"] = key
                    break
        telemetry = {
            key: np.asarray(bundle[key]).reshape(-1)
            for key in ("energy", "energy_total", "C_invariant", "max_amplitude")
            if key in bundle.files
        }
    return FieldArtifact(
        source_path=source_path,
        rho_frames=rho_frames,
        psi_frames=psi_frames,
        times=times,
        source_kind="npz",
        source_datasets=sources,
        telemetry=telemetry,
    )


def load_artifact(source: str | Path) -> FieldArtifact:
    source_path = _resolve_source(source)
    if source_path.suffix.lower() in {".h5", ".hdf5"}:
        return _read_hdf5(source_path)
    if source_path.suffix.lower() == ".npz":
        return _read_npz(source_path)
    raise ValueError(f"Unsupported artifact suffix: {source_path.suffix}")


def _peak_indices(rho: np.ndarray) -> tuple[int, int, int]:
    return tuple(int(v) for v in np.unravel_index(int(np.nanargmax(rho)), rho.shape))


def _finite_percentile(data: np.ndarray, percentile: float, default: float = 1.0) -> float:
    finite = np.asarray(data)[np.isfinite(data)]
    if finite.size == 0:
        return default
    return float(np.percentile(finite, percentile))


def _save_density_slices(artifact: FieldArtifact, out: Path, *, percentile: float) -> Path:
    rho = np.asarray(artifact.rho_frames[-1], dtype=float)
    px, py, pz = _peak_indices(rho)
    vmax = max(_finite_percentile(rho, percentile, 1.0), 1e-12)
    panels = [
        (f"x={px}", rho[px, :, :]),
        (f"y={py}", rho[:, py, :]),
        (f"z={pz}", rho[:, :, pz]),
        ("max-z projection", np.nanmax(rho, axis=2)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), dpi=140, squeeze=False)
    image = None
    for ax, (label, data) in zip(axes[0], panels):
        image = ax.imshow(np.asarray(data).T, origin="lower", cmap="magma", vmin=0.0, vmax=vmax)
        ax.set_title(label, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    if image is not None:
        fig.colorbar(image, ax=list(axes[0]), fraction=0.018, pad=0.02)
    fig.suptitle("Density final-frame slices")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _phase_current(psi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    field = np.asarray(psi, dtype=np.complex128)
    grads = np.gradient(field)
    return tuple(np.imag(np.conj(field) * grad) for grad in grads)  # type: ignore[return-value]


def _save_vector_plot(artifact: FieldArtifact, out: Path) -> tuple[Path | None, str]:
    if artifact.psi_frames is None:
        note = out.with_suffix(".md")
        note.write_text(
            "# Vector Current Unavailable\n\n"
            "No complex `psi` field was found in the saved artifact, so current/vector analysis was not rendered.\n",
            encoding="utf-8",
        )
        return None, str(note)
    psi = artifact.psi_frames[-1]
    rho = artifact.rho_frames[-1]
    px, py, pz = _peak_indices(rho)
    jx, jy, jz = _phase_current(psi)
    rho_slice = rho[:, :, pz]
    u = jx[:, :, pz]
    v = jy[:, :, pz]
    vort = np.gradient(v, axis=0) - np.gradient(u, axis=1)
    vmax = max(_finite_percentile(np.abs(vort), 99.0, 1.0), 1e-12)
    step = max(1, int(rho_slice.shape[0] // 28))
    xx, yy = np.meshgrid(
        np.arange(0, rho_slice.shape[0], step),
        np.arange(0, rho_slice.shape[1], step),
        indexing="ij",
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=140)
    axes[0].imshow(rho_slice.T, origin="lower", cmap="gray_r")
    qu = np.asarray(u[::step, ::step], dtype=float)
    qv = np.asarray(v[::step, ::step], dtype=float)
    mask = np.isfinite(qu) & np.isfinite(qv)
    if np.any(mask):
        axes[0].quiver(xx[mask], yy[mask], qu[mask], qv[mask], color="#1f77b4", width=0.0035)
    axes[0].set_title(f"Current vectors on peak z={pz}")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    im = axes[1].imshow(np.nan_to_num(vort).T, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[1].set_title("Pseudo-vorticity")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out, "rendered"


def _topology_components(rho: np.ndarray, *, threshold_quantile: float) -> list[dict[str, Any]]:
    threshold = float(np.quantile(rho[np.isfinite(rho)], threshold_quantile)) if np.isfinite(rho).any() else 0.0
    mask = np.asarray(rho >= threshold)
    if measure is not None:
        labels = measure.label(mask, connectivity=1)
        rows = []
        for label in range(1, int(labels.max()) + 1):
            coords = np.argwhere(labels == label)
            if coords.size == 0:
                continue
            vals = rho[labels == label]
            centroid = coords.mean(axis=0)
            rows.append(
                {
                    "component": label,
                    "voxel_count": int(coords.shape[0]),
                    "rho_sum": float(np.sum(vals)),
                    "rho_max": float(np.max(vals)),
                    "centroid_x": float(centroid[0]),
                    "centroid_y": float(centroid[1]),
                    "centroid_z": float(centroid[2]),
                    "threshold": threshold,
                }
            )
        return sorted(rows, key=lambda item: (-int(item["voxel_count"]), int(item["component"])))
    coords = np.argwhere(mask)
    if coords.size == 0:
        return []
    centroid = coords.mean(axis=0)
    vals = rho[mask]
    return [
        {
            "component": 1,
            "voxel_count": int(coords.shape[0]),
            "rho_sum": float(np.sum(vals)),
            "rho_max": float(np.max(vals)),
            "centroid_x": float(centroid[0]),
            "centroid_y": float(centroid[1]),
            "centroid_z": float(centroid[2]),
            "threshold": threshold,
        }
    ]


def _save_topology(artifact: FieldArtifact, out_png: Path, out_csv: Path, *, threshold_quantile: float) -> tuple[Path, Path]:
    rho = np.asarray(artifact.rho_frames[-1], dtype=float)
    components = _topology_components(rho, threshold_quantile=threshold_quantile)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["component", "voxel_count", "rho_sum", "rho_max", "centroid_x", "centroid_y", "centroid_z", "threshold"]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(components)

    threshold = components[0]["threshold"] if components else _finite_percentile(rho, threshold_quantile * 100.0, 0.0)
    coords = np.argwhere(rho >= float(threshold))
    if len(coords) > 6000:
        stride = max(1, len(coords) // 6000)
        coords = coords[::stride]
    fig = plt.figure(figsize=(7, 6), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    if len(coords):
        vals = rho[tuple(coords.T)]
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=vals, cmap="viridis", s=2, alpha=0.65)
    for comp in components[:8]:
        ax.scatter(comp["centroid_x"], comp["centroid_y"], comp["centroid_z"], c="red", s=28, marker="x")
    ax.set_title(f"Topology active set q={threshold_quantile:.3f}; components={len(components)}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return out_png, out_csv


def _energy_series(artifact: FieldArtifact) -> tuple[np.ndarray, np.ndarray, str]:
    for key in ("energy", "energy_total", "C_invariant"):
        if key in artifact.telemetry:
            y = np.asarray(artifact.telemetry[key], dtype=float).reshape(-1)
            x = np.arange(len(y), dtype=float)
            if "step" in artifact.telemetry and len(artifact.telemetry["step"]) == len(y):
                x = np.asarray(artifact.telemetry["step"], dtype=float)
            if "time" in artifact.telemetry and len(artifact.telemetry["time"]) == len(y):
                x = np.asarray(artifact.telemetry["time"], dtype=float)
            return x, y, key
    rho_mass = np.sum(artifact.rho_frames, axis=(1, 2, 3))
    return artifact.times, np.asarray(rho_mass, dtype=float), "rho_integral_proxy"


def _save_energy_plot(artifact: FieldArtifact, out: Path) -> tuple[Path, str]:
    x, y, label = _energy_series(artifact)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    ax.plot(x, y, lw=1.5)
    ax.set_title(f"Energy/Invariant series: {label}")
    ax.set_xlabel("time/step")
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.25)
    if label.endswith("_proxy"):
        ax.text(
            0.02,
            0.02,
            "Proxy only: no stored energy telemetry found.",
            transform=ax.transAxes,
            fontsize=8,
            color="#8a5a00",
        )
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out, label


def _write_density_gif(artifact: FieldArtifact, out: Path, *, fps: int, percentile: float, max_frames: int) -> tuple[Path | None, str]:
    if imageio is None:
        note = out.with_suffix(".md")
        note.write_text("# GIF Unavailable\n\n`imageio` is not installed in this runtime.\n", encoding="utf-8")
        return None, str(note)
    count = len(artifact.rho_frames)
    if count <= 0:
        return None, "no frames"
    if count > max_frames:
        picks = np.linspace(0, count - 1, max_frames).round().astype(int)
    else:
        picks = np.arange(count)
    vmax = max(_finite_percentile(artifact.rho_frames, percentile, 1.0), 1e-12)
    images = []
    for idx in picks:
        frame = np.nanmax(artifact.rho_frames[int(idx)], axis=2)
        fig, ax = plt.subplots(figsize=(5.2, 4.8), dpi=120)
        im = ax.imshow(frame.T, origin="lower", cmap="magma", vmin=0.0, vmax=vmax)
        ax.set_title(f"density max-z frame {int(idx)}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.canvas.draw()
        images.append(np.asarray(fig.canvas.buffer_rgba()).copy())
        plt.close(fig)
    imageio.mimsave(out, images, duration=1.0 / max(1, fps), loop=0)
    return out, f"rendered {len(images)} frames"


def _write_manifest(artifact: FieldArtifact, out: Path, outputs: list[dict[str, Any]], settings: dict[str, Any]) -> Path:
    payload = {
        "schema_version": "quantule-field-visual-suite/1.0",
        "source_path": str(artifact.source_path),
        "source_sha256": _sha256_file(artifact.source_path),
        "source_kind": artifact.source_kind,
        "source_datasets": artifact.source_datasets,
        "frame_count": int(len(artifact.rho_frames)),
        "grid_shape": [int(v) for v in artifact.rho_frames.shape[-3:]],
        "has_complex_field": artifact.psi_frames is not None,
        "settings": settings,
        "outputs": outputs,
        "limitations": [
            "Read-only visualization pass over saved artifacts.",
            "Vector plots require a saved complex psi field.",
            "Energy plot uses stored energy telemetry when present; otherwise it is labelled as a rho integral proxy.",
            "Visuals are diagnostic aids and are not standalone scientific validation.",
        ],
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out


def render(
    source: str | Path,
    *,
    outdir: str | Path,
    overwrite: bool = False,
    fps: int = 8,
    rho_percentile: float = 99.7,
    topology_quantile: float = 0.995,
    max_gif_frames: int = 160,
) -> list[str]:
    artifact = load_artifact(source)
    out = Path(outdir)
    if not out.is_absolute():
        out = (Path.cwd() / out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    expected = [
        out / "density_slices.png",
        out / "density_evolution.gif",
        out / "vector_current.png",
        out / "topology_active_set.png",
        out / "topology_components.csv",
        out / "energy_timeseries.png",
        out / "visual_suite_manifest.json",
    ]
    if not overwrite:
        existing = [path for path in expected if path.exists()]
        if existing:
            raise FileExistsError("Refusing to overwrite existing visual-suite outputs without --overwrite:\n" + "\n".join(str(path) for path in existing))

    outputs: list[dict[str, Any]] = []

    density = _save_density_slices(artifact, out / "density_slices.png", percentile=rho_percentile)
    outputs.append({"kind": "density", "path": str(density), "status": "rendered"})

    gif_path, gif_status = _write_density_gif(
        artifact,
        out / "density_evolution.gif",
        fps=fps,
        percentile=rho_percentile,
        max_frames=max_gif_frames,
    )
    outputs.append({"kind": "gif", "path": str(gif_path) if gif_path else None, "status": gif_status})

    vector_path, vector_status = _save_vector_plot(artifact, out / "vector_current.png")
    outputs.append({"kind": "vector", "path": str(vector_path) if vector_path else None, "status": vector_status})

    topo_png, topo_csv = _save_topology(
        artifact,
        out / "topology_active_set.png",
        out / "topology_components.csv",
        threshold_quantile=topology_quantile,
    )
    outputs.append({"kind": "topology", "path": str(topo_png), "status": "rendered"})
    outputs.append({"kind": "topology_table", "path": str(topo_csv), "status": "rendered"})

    energy_path, energy_status = _save_energy_plot(artifact, out / "energy_timeseries.png")
    outputs.append({"kind": "energy", "path": str(energy_path), "status": energy_status})

    settings = {
        "fps": fps,
        "rho_percentile": rho_percentile,
        "topology_quantile": topology_quantile,
        "max_gif_frames": max_gif_frames,
    }
    manifest = _write_manifest(artifact, out / "visual_suite_manifest.json", outputs, settings)
    outputs.append({"kind": "manifest", "path": str(manifest), "status": "rendered"})
    return [str(item["path"]) for item in outputs if item.get("path")]
