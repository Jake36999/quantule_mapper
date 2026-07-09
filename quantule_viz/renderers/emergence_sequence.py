"""Read-only emergence-sequence renderer for saved Quantule Mapper frame bundles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class FrameBundle:
    source_path: Path
    psi: np.ndarray
    times: np.ndarray
    rho: np.ndarray
    psi_key: str
    time_key: str | None


def _resolve_source(source: str | Path) -> Path:
    path = Path(source)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if path.is_dir():
        path = path / "frames.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _decode_json(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _find_summary(source_path: Path) -> Path | None:
    for name in ("summary.json", "diagnostic_summary.json", "feb_bound_state.json", "hifi_series.json"):
        candidate = source_path.parent / name
        if candidate.exists():
            return candidate
    return None


def _params_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    for key in ("replay_params", "saved_params", "params"):
        value = summary.get(key)
        if isinstance(value, dict):
            return value
    candidate = summary.get("candidate")
    if isinstance(candidate, dict):
        value = candidate.get("params")
        if isinstance(value, dict):
            return value
    return {}


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(payload: dict[str, Any]) -> str | None:
    if not payload:
        return None
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_frame_bundle(source: str | Path) -> FrameBundle:
    source_path = _resolve_source(source)
    bundle = np.load(source_path, allow_pickle=True)
    if "psi" in bundle.files:
        psi_key = "psi"
    else:
        candidates = [key for key in bundle.files if np.asarray(bundle[key]).ndim >= 3]
        if not candidates:
            raise ValueError(f"No 3D or time x 3D frame array found in {source_path}")
        psi_key = candidates[0]

    psi = np.asarray(bundle[psi_key])
    if psi.ndim == 3:
        psi = psi[None, ...]
    if psi.ndim != 4:
        raise ValueError(f"Expected psi as [time, x, y, z], got {psi.shape}")

    time_key = "times" if "times" in bundle.files else ("frames" if "frames" in bundle.files else None)
    times = np.asarray(bundle[time_key], dtype=float) if time_key else np.arange(len(psi), dtype=float)
    if len(times) != len(psi):
        times = np.arange(len(psi), dtype=float)
        time_key = None

    rho = np.abs(psi) ** 2
    return FrameBundle(source_path=source_path, psi=psi, times=times, rho=rho, psi_key=psi_key, time_key=time_key)


def selected_frame_indices(frame_count: int) -> list[int]:
    if frame_count <= 0:
        return []
    if frame_count <= 6:
        return list(range(frame_count))
    fractions = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
    return sorted({min(frame_count - 1, int(round((frame_count - 1) * frac))) for frac in fractions})


def derive_omega_sq(rho: np.ndarray, summary_path: str | Path | None) -> np.ndarray | None:
    summary = _decode_json(Path(summary_path) if summary_path else None)
    params = _params_from_summary(summary)
    try:
        rho_vac = float(params["param_rho_vac"])
        a_coupling = float(params["param_a_coupling"])
    except (KeyError, TypeError, ValueError):
        return None
    rho_safe = np.maximum(np.asarray(rho, dtype=np.float64), 1e-12)
    return (rho_vac / rho_safe) ** a_coupling


def _projection(volume: np.ndarray, *, stride: int = 1) -> np.ndarray:
    arr = np.asarray(volume)
    if stride > 1:
        arr = arr[::stride, ::stride, ::stride]
    return np.nanmax(arr, axis=2)


def _final_peak_slice(rho: np.ndarray) -> tuple[int, int]:
    final_rho = np.asarray(rho[-1])
    peak = np.unravel_index(int(np.nanargmax(final_rho)), final_rho.shape)
    axis = int(np.argmax(final_rho.shape))
    return axis, int(peak[axis])


def _slice(volume: np.ndarray, axis: int, index: int, *, stride: int = 1) -> np.ndarray:
    arr = np.asarray(volume)
    if axis == 0:
        out = arr[index, :, :]
    elif axis == 1:
        out = arr[:, index, :]
    else:
        out = arr[:, :, index]
    if stride > 1:
        out = out[::stride, ::stride]
    return out


def _frame_to_rgba(data: np.ndarray, *, title: str, cmap: str, vmin: float, vmax: float) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(5.2, 4.8), dpi=120)
    image = ax.imshow(np.asarray(data).T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba


def _save_still(data: np.ndarray, out: Path, *, title: str, cmap: str, vmin: float, vmax: float) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.8), dpi=140)
    image = ax.imshow(np.asarray(data).T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def _save_montage(
    panels: list[tuple[str, np.ndarray]],
    out: Path,
    *,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
) -> Path:
    cols = len(panels)
    fig, axes = plt.subplots(1, cols, figsize=(3.0 * cols, 3.2), dpi=140, squeeze=False)
    last = None
    for ax, (label, data) in zip(axes[0], panels):
        last = ax.imshow(np.asarray(data).T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(label, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    if last is not None:
        fig.colorbar(last, ax=list(axes[0]), fraction=0.018, pad=0.02)
    fig.suptitle(title, fontsize=11)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _write_gif(
    frames: list[np.ndarray],
    out: Path,
    *,
    times: np.ndarray,
    cmap: str,
    vmin: float,
    vmax: float,
    fps: int,
    label: str,
) -> Path:
    images = [
        _frame_to_rgba(frame, title=f"{label}  t={float(time):.1f}", cmap=cmap, vmin=vmin, vmax=vmax)
        for frame, time in zip(frames, times)
    ]
    imageio.mimsave(out, images, duration=1.0 / max(1, fps), loop=0)
    return out


def _metadata_markdown(
    *,
    bundle: FrameBundle,
    summary_path: Path | None,
    summary: dict[str, Any],
    outputs: list[Path],
    render_settings: dict[str, Any],
    omega_status: str,
) -> str:
    params = _params_from_summary(summary)
    config_hash = summary.get("config_hash") or _stable_hash(params) or "not found in source metadata"
    git_commit = summary.get("git_commit") or "not found in source metadata"
    source_hash = _sha256_file(bundle.source_path)
    time_start = float(bundle.times[0]) if len(bundle.times) else 0.0
    time_end = float(bundle.times[-1]) if len(bundle.times) else 0.0
    lines = [
        "# Quantule Emergence Visualisation Metadata",
        "",
        "This is a read-only visualisation pass over saved artifacts. It does not modify solver, physics, Hunter, validation gates, or configs. Visuals alone are not treated as scientific evidence.",
        "",
        "## Source",
        f"- source artifact: `{bundle.source_path}`",
        f"- source SHA256: `{source_hash}`",
        f"- summary metadata: `{summary_path}`" if summary_path else "- summary metadata: not found",
        f"- config hash / proxy: `{config_hash}`",
        f"- git commit: `{git_commit}`",
        f"- psi dataset key: `{bundle.psi_key}`",
        f"- time dataset key: `{bundle.time_key or 'implicit arange'}`",
        f"- frame shape: `{tuple(int(v) for v in bundle.psi.shape)}`",
        f"- timestep range rendered: `{time_start}` to `{time_end}`",
        "",
        "## Render Settings",
    ]
    for key, value in render_settings.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            f"- omega / geometry render: {omega_status}",
            "",
            "## Outputs",
        ]
    )
    lines.extend(f"- `{path}`" for path in outputs)
    lines.extend(
        [
            "",
            "## Limitations",
            "- Density is rendered as `rho = abs(psi)^2` from saved frames only.",
            "- Density GIF uses a max-intensity projection for visibility; selected stills use the same projection.",
            "- Phase render is a fixed final-peak slice and is masked only by the visual colormap, not by an analytic phase-quality gate.",
            "- Derived omega is a local proxy from saved params and rho, not a stored solver omega field unless explicitly stated otherwise.",
            "- No scientific claims should be made from these visuals alone.",
            "",
        ]
    )
    return "\n".join(lines)


def render(
    source: str | Path,
    *,
    outdir: str | Path,
    overwrite: bool = False,
    fps: int = 6,
    spatial_stride: int = 1,
    rho_percentile: float = 99.7,
) -> list[str]:
    bundle = load_frame_bundle(source)
    out = Path(outdir)
    if not out.is_absolute():
        out = (Path.cwd() / out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    expected = [
        out / "rho_emergence.gif",
        out / "rho_stills_montage.png",
        out / "phase_emergence.gif",
        out / "phase_stills_montage.png",
        out / "omega_proxy_emergence.gif",
        out / "omega_proxy_stills_montage.png",
        out / "metadata.md",
    ]
    if not overwrite:
        existing = [path for path in expected if path.exists()]
        if existing:
            raise FileExistsError("Refusing to overwrite existing outputs without --overwrite:\n" + "\n".join(str(path) for path in existing))

    finite_rho = bundle.rho[np.isfinite(bundle.rho)]
    vmax = float(np.percentile(finite_rho, rho_percentile)) if finite_rho.size else 1.0
    vmax = max(vmax, 1e-12)
    projections = [_projection(frame, stride=spatial_stride) for frame in bundle.rho]
    outputs: list[Path] = []

    rho_gif = _write_gif(
        projections,
        out / "rho_emergence.gif",
        times=bundle.times,
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
        fps=fps,
        label="rho=max_z abs(psi)^2",
    )
    outputs.append(rho_gif)

    picks = selected_frame_indices(len(bundle.rho))
    rho_panels: list[tuple[str, np.ndarray]] = []
    for index in picks:
        label = f"t={float(bundle.times[index]):.0f}"
        data = projections[index]
        rho_panels.append((label, data))
        outputs.append(_save_still(data, out / f"rho_still_{index:03d}.png", title=f"rho {label}", cmap="magma", vmin=0.0, vmax=vmax))
    outputs.append(_save_montage(rho_panels, out / "rho_stills_montage.png", title="rho emergence selected stills", cmap="magma", vmin=0.0, vmax=vmax))

    axis, slice_index = _final_peak_slice(bundle.rho)
    phase_frames = [_slice(np.angle(frame), axis, slice_index, stride=spatial_stride) for frame in bundle.psi]
    phase_gif = _write_gif(
        phase_frames,
        out / "phase_emergence.gif",
        times=bundle.times,
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
        fps=fps,
        label=f"phase slice axis={axis} index={slice_index}",
    )
    outputs.append(phase_gif)
    phase_panels = []
    for index in picks:
        label = f"t={float(bundle.times[index]):.0f}"
        data = phase_frames[index]
        phase_panels.append((label, data))
        outputs.append(_save_still(data, out / f"phase_still_{index:03d}.png", title=f"phase {label}", cmap="twilight", vmin=-np.pi, vmax=np.pi))
    outputs.append(_save_montage(phase_panels, out / "phase_stills_montage.png", title="phase selected stills", cmap="twilight", vmin=-np.pi, vmax=np.pi))

    summary_path = _find_summary(bundle.source_path)
    summary = _decode_json(summary_path)
    omega_status = "not rendered; no stored omega field or derivable params found"
    omega = derive_omega_sq(bundle.rho, summary_path)
    if omega is not None:
        finite_omega = omega[np.isfinite(omega)]
        omega_vmax = float(np.percentile(finite_omega, 99.0)) if finite_omega.size else 1.0
        omega_vmin = float(np.percentile(finite_omega, 1.0)) if finite_omega.size else 0.0
        omega_frames = [_projection(frame, stride=spatial_stride) for frame in omega]
        outputs.append(
            _write_gif(
                omega_frames,
                out / "omega_proxy_emergence.gif",
                times=bundle.times,
                cmap="viridis",
                vmin=omega_vmin,
                vmax=omega_vmax,
                fps=fps,
                label="derived omega_sq proxy",
            )
        )
        omega_panels = []
        for index in picks:
            label = f"t={float(bundle.times[index]):.0f}"
            data = omega_frames[index]
            omega_panels.append((label, data))
            outputs.append(_save_still(data, out / f"omega_proxy_still_{index:03d}.png", title=f"omega proxy {label}", cmap="viridis", vmin=omega_vmin, vmax=omega_vmax))
        outputs.append(
            _save_montage(
                omega_panels,
                out / "omega_proxy_stills_montage.png",
                title="derived omega_sq proxy selected stills",
                cmap="viridis",
                vmin=omega_vmin,
                vmax=omega_vmax,
            )
        )
        omega_status = "rendered as derived local proxy `(param_rho_vac / rho)^param_a_coupling`"

    render_settings = {
        "rho_formula": "abs(psi)^2",
        "rho_projection": "max over z axis",
        "rho_color_scale": f"0 to p{rho_percentile}={vmax:.6g}",
        "phase_render": f"angle(psi) on fixed final-peak slice axis={axis}, index={slice_index}",
        "selected_frame_indices": picks,
        "gif_fps": fps,
        "spatial_stride": spatial_stride,
        "time_downsampling": "none",
    }
    metadata = _metadata_markdown(
        bundle=bundle,
        summary_path=summary_path,
        summary=summary,
        outputs=outputs,
        render_settings=render_settings,
        omega_status=omega_status,
    )
    metadata_path = out / "metadata.md"
    metadata_path.write_text(metadata, encoding="utf-8")
    outputs.append(metadata_path)
    return [str(path) for path in outputs]
