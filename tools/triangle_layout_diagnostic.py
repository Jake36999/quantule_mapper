"""Diagnostic-only Phase C triangle layout screen.

This script is intentionally standalone. It reads existing Phase C probe artifacts,
builds synthetic three-node triangle initial conditions, and runs fixed-parameter
short diagnostics through the existing JAX scout physics entry points.

It does not modify solver physics, Hunter logic, validation gates, production defaults,
or existing configs. Results are geometry priors for Phase C dissipative tests only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.ndimage as ndi


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
PARAM_ORDER = [
    "param_D",
    "param_eta",
    "param_rho_vac",
    "param_omega0",
    "param_a_coupling",
    "param_s",
    "param_f",
    "param_a",
]
SEED = 20260619
SAT_SLOPE = 1.5e-4
NEAR_SLOPE = 3.5e-4
LATE_DRIFT_MAX = 0.15
BREATHING_FLOOR_RATIO_MIN = 0.85
BREATHING_PEAK_MARGIN = 0.05
NODE_SIGMA = 2.5
MIN_NODE_VOXELS = 3

L_DEFAULT = 10.0
DT_DEFAULT = 0.005
N_DEFAULT = 96
SPACINGS_BOX = [0.28, 0.32, 0.36, 0.40, 0.45, 0.49, 0.53]
OUT_DEFAULT = ROOT / "quantule_viz" / "outputs" / "triangle_layout_diagnostic"
PROBE_SPECS = [
    {
        "key": "FEB_BASIN_CONFIRM_20260625_154503/K3_s20260621_T12000",
        "artifact": ROOT
        / "sweep_runs"
        / "FEB_BASIN_CONFIRM_20260625_154503"
        / "K3_s20260621_T12000_probe.npz",
        "summary": ROOT
        / "sweep_runs"
        / "FEB_BASIN_CONFIRM_20260625_154503"
        / "feb_basin_confirm_summary.json",
    },
    {
        "key": "FEB_BASIN_CONFIRM_20260625_154503/K3_s20260620_T12000",
        "artifact": ROOT
        / "sweep_runs"
        / "FEB_BASIN_CONFIRM_20260625_154503"
        / "K3_s20260620_T12000_probe.npz",
        "summary": ROOT
        / "sweep_runs"
        / "FEB_BASIN_CONFIRM_20260625_154503"
        / "feb_basin_confirm_summary.json",
    },
    {
        "key": "FEB_PARAM_BASIN_20260626_004039/revalid_K5_T24000",
        "artifact": ROOT
        / "sweep_runs"
        / "FEB_PARAM_BASIN_20260626_004039"
        / "revalid_K5_T24000_probe.npz",
        "summary": ROOT
        / "sweep_runs"
        / "FEB_PARAM_BASIN_20260626_004039"
        / "feb_param_basin_summary.json",
    },
]


def canonical_params_hash(params: dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def pairwise_periodic_distances(points: np.ndarray, box_size_vox: int, normalize: bool = True) -> list[float]:
    pts = np.asarray(points, dtype=float)
    out: list[float] = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = pts[i] - pts[j]
            d = d - box_size_vox * np.round(d / box_size_vox)
            dist = float(np.linalg.norm(d))
            out.append(dist / box_size_vox if normalize else dist)
    return out


def classify_triangle_shape(distances_box: list[float] | np.ndarray) -> str:
    d = np.sort(np.asarray(distances_box, dtype=float))
    if d.size != 3 or not np.all(np.isfinite(d)) or d[0] <= 0:
        return "unclassified"
    triangle_slack = float((d[0] + d[1] - d[2]) / d[2])
    cv = float(d.std() / (d.mean() + 1e-30))
    stretch = float(d[-1] / d[0])
    if triangle_slack < 0.12:
        return "line-like"
    if cv <= 0.08 and stretch <= 1.18:
        return "equilateral-like"
    return "stretched"


def circular_mean_phase(phases: np.ndarray, weights: np.ndarray) -> float:
    if phases.size == 0 or weights.size == 0 or float(np.sum(weights)) <= 0:
        return float("nan")
    z = np.sum(weights * np.exp(1j * phases))
    if abs(z) <= 1e-30:
        return float("nan")
    return float(np.angle(z))


def _circ_centroid(coords: np.ndarray, weights: np.ndarray, N: int) -> float:
    theta = 2.0 * np.pi * coords / N
    c = np.average(np.cos(theta), weights=weights)
    s = np.average(np.sin(theta), weights=weights)
    angle = np.arctan2(s, c) % (2.0 * np.pi)
    return float(angle * N / (2.0 * np.pi))


def _component_nodes(psi: np.ndarray, rho_threshold_frac: float | None = None) -> list[dict[str, Any]]:
    rho = np.abs(psi) ** 2
    if rho_threshold_frac is None:
        thr = float(rho.mean() + NODE_SIGMA * rho.std())
    else:
        thr = float(rho.max() * rho_threshold_frac)
    mask = rho > thr
    lbl, nn = ndi.label(mask)
    nodes: list[dict[str, Any]] = []
    dx = L_DEFAULT / rho.shape[0]
    for label in range(1, nn + 1):
        sel = lbl == label
        size = int(sel.sum())
        if size < MIN_NODE_VOXELS:
            continue
        idx = np.array(np.nonzero(sel))
        weights = rho[sel]
        centroid = np.array([_circ_centroid(idx[axis], weights, rho.shape[axis]) for axis in range(3)])
        peak_index_local = int(np.argmax(weights))
        peak_vox = idx[:, peak_index_local].astype(float)
        phase_values = np.angle(psi[sel])
        phase_mask = weights >= max(float(rho.max()) * 0.08, float(weights.max()) * 0.25)
        phase = circular_mean_phase(phase_values[phase_mask], weights[phase_mask])
        nodes.append(
            {
                "centroid": centroid,
                "peak_vox": peak_vox,
                "M": float(np.sqrt(weights).sum()) * dx**3,
                "E": float(weights.sum()) * dx**3,
                "rho_peak": float(weights.max()),
                "phase": phase,
                "size": size,
            }
        )
    nodes.sort(key=lambda nd: nd["rho_peak"], reverse=True)
    return nodes


def node_geometry_from_psi(
    psi: np.ndarray,
    L: float = L_DEFAULT,
    expected_nodes: int = 3,
    rho_threshold_frac: float | None = None,
) -> dict[str, Any]:
    nodes = _component_nodes(psi, rho_threshold_frac=rho_threshold_frac)
    if len(nodes) > expected_nodes:
        nodes = nodes[:expected_nodes]
    N = int(psi.shape[0])
    centroids = np.array([nd["centroid"] for nd in nodes], dtype=float) if nodes else np.empty((0, 3))
    distances = pairwise_periodic_distances(centroids, N, normalize=True) if len(nodes) >= 2 else []
    masses = np.array([nd["M"] for nd in nodes], dtype=float)
    peaks = np.array([nd["rho_peak"] for nd in nodes], dtype=float)
    phases = np.array([nd["phase"] for nd in nodes], dtype=float)
    return {
        "node_count": int(len(nodes)),
        "centroids_vox": centroids.tolist(),
        "pairwise_distances_box": distances,
        "nn_spacing_min_box": float(np.min(distances)) if distances else float("nan"),
        "nn_spacing_mean_box": float(np.mean(distances)) if distances else float("nan"),
        "nn_spacing_max_box": float(np.max(distances)) if distances else float("nan"),
        "node_masses": masses.tolist(),
        "node_mass_cv": float(masses.std() / (masses.mean() + 1e-30)) if masses.size else float("nan"),
        "rho_peaks": peaks.tolist(),
        "rho_peak_cv": float(peaks.std() / (peaks.mean() + 1e-30)) if peaks.size else float("nan"),
        "centroid_phases": phases.tolist(),
        "phase_reliable": bool(phases.size == len(nodes) and np.all(np.isfinite(phases))),
        "shape_class": classify_triangle_shape(distances) if len(nodes) == 3 else "not-3-node",
    }


def equilateral_triangle_points(N: int, side_length_box: float) -> np.ndarray:
    side_vox = float(side_length_box) * N
    radius = side_vox / math.sqrt(3.0)
    center = np.array([N / 2.0, N / 2.0, N / 2.0], dtype=float)
    angles = np.array([math.pi / 2.0, math.pi / 2.0 + 2.0 * math.pi / 3.0, math.pi / 2.0 + 4.0 * math.pi / 3.0])
    pts = np.column_stack(
        [
            center[0] + radius * np.cos(angles),
            center[1] + radius * np.sin(angles),
            np.full(3, center[2]),
        ]
    )
    return pts % N


def _periodic_delta_axis(axis_values: np.ndarray, center_value: float, L: float) -> np.ndarray:
    delta = axis_values - center_value
    return (delta + L / 2.0) % L - L / 2.0


def build_triangle_ic(
    N: int,
    L: float,
    side_length_box: float,
    width_box: float,
    amplitude: float,
    phases: list[float] | np.ndarray,
    noise_level: float = 0.0,
    seed: int = SEED,
) -> np.ndarray:
    pts_vox = equilateral_triangle_points(N, side_length_box)
    coords = np.linspace(-L / 2.0, L / 2.0, N, endpoint=False)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
    width = float(width_box) * L
    phases_arr = np.asarray(phases, dtype=float)
    if phases_arr.size != 3:
        raise ValueError("phases must contain exactly three values")
    psi = np.zeros((N, N, N), dtype=np.complex128)
    for idx, center_vox in enumerate(pts_vox):
        center_phys = (center_vox / N) * L - L / 2.0
        dx = _periodic_delta_axis(X, center_phys[0], L)
        dy = _periodic_delta_axis(Y, center_phys[1], L)
        dz = _periodic_delta_axis(Z, center_phys[2], L)
        profile = amplitude * np.exp(-(dx * dx + dy * dy + dz * dz) / (2.0 * width * width))
        psi += profile * np.exp(1j * phases_arr[idx])
    if noise_level > 0:
        rng = np.random.default_rng(seed)
        psi += noise_level * (rng.standard_normal(psi.shape) + 1j * rng.standard_normal(psi.shape))
    return psi.astype(np.complex128)


def estimate_profile_from_probe(paths: list[Path], fallback_width_box: float = 1 / 12) -> dict[str, float]:
    # The historical Phase C generator documents fixed w=L/12 and amplitude=1.
    # We keep that as the primary contract and only report probe-derived peak scale as context.
    peaks = []
    for path in paths:
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as data:
            key = "psi0" if "psi0" in data.files else "psi_fin"
            rho = np.abs(data[key]) ** 2
            peaks.append(float(np.sqrt(np.max(rho))))
    return {
        "width_box": float(fallback_width_box),
        "amplitude": 1.0,
        "observed_probe_peak_amplitude_mean": float(np.mean(peaks)) if peaks else float("nan"),
        "profile_source": "Phase C multiseed_ic contract: fixed w=L/12, amplitude=1; observed peaks recorded only as context",
    }


def load_summary_row(summary_path: Path, key: str) -> dict[str, Any]:
    if not summary_path.exists():
        return {}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    short_key = key.split("/")[-1]
    for row in payload.get("rows", []):
        if row.get("key") == short_key:
            return row
    return {}


def extract_probe_geometry(out_dir: Path) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for spec in PROBE_SPECS:
        path = Path(spec["artifact"])
        row_meta = load_summary_row(Path(spec["summary"]), spec["key"])
        if not path.exists():
            rows.append({"source_key": spec["key"], "artifact_path": str(path), "caveat": "missing artifact"})
            continue
        with np.load(path, allow_pickle=False) as data:
            psi = np.asarray(data["psi_fin"], dtype=np.complex128)
            er = np.asarray(data["er"], dtype=float) if "er" in data.files else np.array([])
        geom = node_geometry_from_psi(psi, L=L_DEFAULT, expected_nodes=3)
        params = FEB_PARAMS
        rows.append(
            {
                "source_key": spec["key"],
                "artifact_path": str(path),
                "config_hash": canonical_params_hash(params),
                "seed": row_meta.get("seed", ""),
                "T_steps": row_meta.get("T", len(er) if er.size else ""),
                "klass": row_meta.get("klass", ""),
                "summary_n_fin": row_meta.get("n_fin", ""),
                "detected_final_node_count": geom["node_count"],
                "shape_class": geom["shape_class"],
                "pairwise_distances_box": json.dumps(geom["pairwise_distances_box"]),
                "nn_spacing_min_box": geom["nn_spacing_min_box"],
                "nn_spacing_mean_box": geom["nn_spacing_mean_box"],
                "nn_spacing_max_box": geom["nn_spacing_max_box"],
                "node_mass_cv": geom["node_mass_cv"],
                "rho_peak_cv": geom["rho_peak_cv"],
                "node_masses": json.dumps(geom["node_masses"]),
                "rho_peaks": json.dumps(geom["rho_peaks"]),
                "centroids_vox": json.dumps(geom["centroids_vox"]),
                "centroid_phases": json.dumps(geom["centroid_phases"]),
                "phase_reliable": geom["phase_reliable"],
                "late_drift": row_meta.get("late_drift", ""),
                "bounded_breathing": row_meta.get("bounded_breathing", ""),
                "caveat": "probe-derived geometry prior; Phase C dissipative artifact only",
            }
        )
    write_csv(out_dir / "triangle_geometry_index.csv", rows)
    write_geometry_report(out_dir / "triangle_geometry_report.md", rows)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def write_geometry_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Triangle Geometry Report",
        "",
        "Diagnostic-only extraction from existing Phase C dissipative probe artifacts. These rows are geometry priors for follow-up seed/layout tests and do not establish behavior in the conservative or moving substrate.",
        "",
        f"- Fixed parameter hash: `{canonical_params_hash(FEB_PARAMS)}`",
        f"- Node detector: `jax_scout.transfer_diag` threshold conventions plus rho-masked centroid phase extraction.",
        "",
        "## Extracted Artifacts",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"### {row.get('source_key')}",
                "",
                f"- Artifact: `{row.get('artifact_path')}`",
                f"- Verdict/source class: `{row.get('klass', '')}`",
                f"- Summary n_fin: `{row.get('summary_n_fin', '')}`; detected final nodes: `{row.get('detected_final_node_count', '')}`",
                f"- Shape: `{row.get('shape_class', '')}`",
                f"- Pairwise distances, box units: `{row.get('pairwise_distances_box', '')}`",
                f"- Mass CV: `{row.get('node_mass_cv', '')}`; rho-peak CV: `{row.get('rho_peak_cv', '')}`",
                f"- Phase reliable: `{row.get('phase_reliable', '')}`; centroid phases: `{row.get('centroid_phases', '')}`",
                f"- Caveat: {row.get('caveat', '')}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_ic_manifest(out_dir: Path, profile: dict[str, float], geometry_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    phase_source = next(
        (row for row in geometry_rows if str(row.get("phase_reliable", "")).lower() == "true"),
        None,
    )
    extracted_phases: list[float] | None = None
    if phase_source:
        try:
            extracted_phases = [float(v) for v in json.loads(str(phase_source["centroid_phases"]))]
        except Exception:
            extracted_phases = None
    manifest: list[dict[str, Any]] = []
    for spacing in SPACINGS_BOX:
        manifest.append(
            {
                "ic_id": f"triangle_s{spacing:.2f}_aligned",
                "spacing_box": spacing,
                "variant": "aligned_phase",
                "phases": [0.0, 0.0, 0.0],
                "N": N_DEFAULT,
                "L": L_DEFAULT,
                "width_box": profile["width_box"],
                "amplitude": profile["amplitude"],
                "noise_level": 0.0,
                "generator": "tools.triangle_layout_diagnostic.build_triangle_ic",
                "caveat": "synthetic diagnostic IC; not a production default",
            }
        )
        if extracted_phases and len(extracted_phases) == 3:
            manifest.append(
                {
                    "ic_id": f"triangle_s{spacing:.2f}_extracted_phase_control",
                    "spacing_box": spacing,
                    "variant": "extracted_phase_control",
                    "phases": extracted_phases,
                    "phase_source": phase_source["source_key"],
                    "N": N_DEFAULT,
                    "L": L_DEFAULT,
                    "width_box": profile["width_box"],
                    "amplitude": profile["amplitude"],
                    "noise_level": 0.0,
                    "generator": "tools.triangle_layout_diagnostic.build_triangle_ic",
                    "caveat": "phase-control diagnostic; centroid phase is rho-masked and may be noisy",
                }
            )
    payload = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Phase C dissipative synthetic 3-node triangle layout diagnostics",
        "fixed_params": FEB_PARAMS,
        "fixed_params_hash": canonical_params_hash(FEB_PARAMS),
        "profile": profile,
        "ics": manifest,
        "limitations": [
            "No solver physics, Hunter, gates, or production defaults are modified.",
            "These ICs are geometry priors only and do not prove conservative/moving substrate stability.",
        ],
    }
    (out_dir / "triangle_ic_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest


@dataclass
class ScreenResult:
    row: dict[str, Any]
    snaps: np.ndarray


def _jax_imports():
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax import lax

    from jax_scout import physics

    return jax, jnp, lax, physics


def capture_probe_snapshots(
    params: dict[str, float],
    psi0: np.ndarray,
    N: int,
    L: float,
    dt: float,
    T_steps: int,
    n_snap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    jax, jnp, lax, physics = _jax_imports()
    stride = max(1, T_steps // n_snap)
    effective_steps = stride * n_snap
    pvec = np.asarray([params[k] for k in PARAM_ORDER], dtype=np.float64)

    @jax.jit
    def _run(pvec_j, psi0_j):
        ops = physics._ops_from_vec(pvec_j, N, L, dt, jnp.float64, jnp.complex128)
        psi_k = jnp.fft.fftn(psi0_j) * ops.dealias_mask

        def inner(pk, _):
            pk = physics.step(pk, ops)
            psi = jnp.fft.ifftn(pk)
            return pk, (jnp.sum(jnp.abs(psi) ** 2), jnp.max(jnp.abs(psi)))

        def outer(pk, _):
            pk, (energy, max_amp) = lax.scan(inner, pk, None, length=stride)
            return pk, (jnp.fft.ifftn(pk), energy, max_amp)

        psi_k, (snaps, energy_segments, amp_segments) = lax.scan(outer, psi_k, None, length=n_snap)
        final = jnp.fft.ifftn(psi_k)
        finite = jnp.all(jnp.isfinite(jnp.abs(final)))
        return snaps, energy_segments.reshape((-1,)), amp_segments.reshape((-1,)), finite

    snaps, energy, amps, finite = _run(jnp.asarray(pvec), jnp.asarray(psi0))
    snaps_np = np.concatenate([psi0[None, ...], np.asarray(snaps)], axis=0)
    return snaps_np, np.asarray(energy), np.asarray(amps), bool(np.asarray(finite))


def phase_c_classify(finite: bool, er: np.ndarray, n_mid: int, n_fin: int, core_fin: float) -> tuple[str, dict[str, Any]]:
    er = np.asarray(er, dtype=float)
    er_max = float(np.max(er))
    er_fin = float(er[-1])
    base: dict[str, Any] = {"er_fin": er_fin, "er_max": er_max, "n_mid": n_mid, "n_fin": n_fin, "core_fin": core_fin}
    half = len(er) // 2
    late = er[half:]
    xs = np.arange(len(late))
    if len(xs) > 2:
        coef = np.polyfit(xs, late, 1)
        slope = float(coef[0])
        fit_start = float(coef[1])
        fit_end = float(coef[0] * (len(late) - 1) + coef[1])
        late_drift = (fit_end - fit_start) / (abs(fit_start) + 1e-9)
    else:
        slope, late_drift = 0.0, 0.0
    er0 = float(er[0])
    er_min = float(np.min(er))
    floor_ratio = er_min / (abs(er0) + 1e-9)
    bounded_breathing = (
        er_max <= 3.0
        and 0.5 <= er_fin <= 2.5
        and floor_ratio >= BREATHING_FLOOR_RATIO_MIN
        and er_fin <= (1.0 - BREATHING_PEAK_MARGIN) * er_max
    )
    base.update(
        {
            "late_slope": slope,
            "late_drift": late_drift,
            "er0": er0,
            "er_min": er_min,
            "floor_ratio": floor_ratio,
            "bounded_breathing": bool(bounded_breathing),
        }
    )
    if not finite or not np.isfinite(er_max) or er_max > 3.0:
        return "LATE_BLOWUP_REJECT", base
    if er_fin < 0.3:
        return "SPIN_DOWN_REJECT", base
    if n_fin > 8 or n_mid > 8:
        return "FRAGMENTATION_REJECT", base
    if n_fin < 1 or core_fin < 0.15:
        return "DELOCALIZED_HALO_REJECT", base
    if slope > NEAR_SLOPE:
        return "TRANSIENT_GROWER_REJECT", base
    if abs(late_drift) > LATE_DRIFT_MAX:
        if bounded_breathing:
            return "TRUE_SATURATED_BOUND_STATE", base
        return ("TRANSIENT_GROWER_REJECT" if late_drift > 0 else "SPIN_DOWN_REJECT"), base
    if abs(slope) <= SAT_SLOPE and 0.5 <= er_fin <= 2.5:
        return "TRUE_SATURATED_BOUND_STATE", base
    if abs(slope) <= NEAR_SLOPE and 0.5 <= er_fin <= 2.5:
        return "NEAR_SATURATED_BOUND_STATE", base
    return ("SPIN_DOWN_REJECT" if slope < 0 else "TRANSIENT_GROWER_REJECT"), base


def classify_screen_result(finite: bool, er: np.ndarray, psi_mid: np.ndarray, psi_fin: np.ndarray, dx: float) -> tuple[str, dict[str, Any]]:
    if er.size == 0:
        return "ERROR", {}
    viable = bool(finite) and np.isfinite(er).all() and float(np.max(er)) <= 3.0 and float(er[-1]) >= 0.3
    n_mid = node_geometry_from_psi(psi_mid, L=dx * psi_mid.shape[0], expected_nodes=9)["node_count"] if viable else 0
    n_fin = node_geometry_from_psi(psi_fin, L=dx * psi_fin.shape[0], expected_nodes=9)["node_count"] if viable else 0
    core_fin = float(np.max(np.abs(psi_fin) ** 2)) if viable else 0.0
    return phase_c_classify(bool(finite), er, n_mid, n_fin, core_fin)


def run_spacing_screen(out_dir: Path, manifest: list[dict[str, Any]], T_steps: int, n_snap: int, render: bool = True) -> list[ScreenResult]:
    results: list[ScreenResult] = []
    render_dir = out_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    for ic in [m for m in manifest if m["variant"] == "aligned_phase"]:
        spacing = float(ic["spacing_box"])
        case_id = ic["ic_id"]
        case_dir = render_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        psi0 = build_triangle_ic(
            N=int(ic["N"]),
            L=float(ic["L"]),
            side_length_box=spacing,
            width_box=float(ic["width_box"]),
            amplitude=float(ic["amplitude"]),
            phases=[float(v) for v in ic["phases"]],
            noise_level=float(ic["noise_level"]),
            seed=SEED,
        )
        t0 = time.time()
        snaps, energy, amps, finite = capture_probe_snapshots(FEB_PARAMS, psi0, int(ic["N"]), float(ic["L"]), DT_DEFAULT, T_steps, n_snap)
        wallclock = time.time() - t0
        ic_e = float(np.sum(np.abs(psi0) ** 2)) + 1e-30
        er = energy / ic_e
        mid_idx = max(0, len(snaps) // 2)
        klass, metrics = classify_screen_result(finite, er, snaps[mid_idx], snaps[-1], float(ic["L"]) / int(ic["N"]))
        final_geom = node_geometry_from_psi(snaps[-1], L=float(ic["L"]), expected_nodes=5)
        initial_geom = node_geometry_from_psi(snaps[0], L=float(ic["L"]), expected_nodes=3)
        drift = float(final_geom["nn_spacing_mean_box"] - initial_geom["nn_spacing_mean_box"]) if np.isfinite(final_geom["nn_spacing_mean_box"]) else float("nan")
        row = {
            "case_id": case_id,
            "spacing_box_initial": spacing,
            "variant": ic["variant"],
            "T_steps": T_steps,
            "effective_steps": int((T_steps // n_snap) * n_snap),
            "n_snap": n_snap,
            "fixed_params_hash": canonical_params_hash(FEB_PARAMS),
            "finite": finite,
            "verdict": klass,
            "final_node_count": final_geom["node_count"],
            "initial_node_count": initial_geom["node_count"],
            "initial_nn_spacing_mean_box": initial_geom["nn_spacing_mean_box"],
            "final_nn_spacing_min_box": final_geom["nn_spacing_min_box"],
            "final_nn_spacing_mean_box": final_geom["nn_spacing_mean_box"],
            "spacing_drift_mean_box": drift,
            "final_shape_class": final_geom["shape_class"],
            "final_node_mass_cv": final_geom["node_mass_cv"],
            "final_rho_peak_cv": final_geom["rho_peak_cv"],
            "er_fin": float(er[-1]) if er.size else float("nan"),
            "er_max": float(np.max(er)) if er.size else float("nan"),
            "late_drift": metrics.get("late_drift", ""),
            "late_slope": metrics.get("late_slope", ""),
            "bounded_breathing": metrics.get("bounded_breathing", ""),
            "rho_gif": str(case_dir / "rho.gif") if render else "",
            "rho_montage": str(case_dir / "rho_montage.png") if render else "",
            "wallclock_sec": round(wallclock, 2),
            "caveat": "short Phase C dissipative screen; promote only as diagnostic prior, not a conservative/moving-substrate claim",
        }
        if render:
            render_case(snaps, case_dir, case_id, row)
        results.append(ScreenResult(row=row, snaps=snaps))
        np.savez_compressed(case_dir / "frames.npz", psi=snaps.astype(np.complex64), t=np.linspace(0, int(row["effective_steps"]), len(snaps), dtype=np.int32))
        (case_dir / "summary.json").write_text(json.dumps(row, indent=2, default=float), encoding="utf-8")
        print(f"{case_id}: {klass} n_fin={row['final_node_count']} sp_fin={row['final_nn_spacing_mean_box']:.3f} ({wallclock:.1f}s)", flush=True)
    write_csv(out_dir / "triangle_spacing_screen_results.csv", [r.row for r in results])
    return results


def _rho_projection(psi: np.ndarray) -> np.ndarray:
    rho = np.abs(psi) ** 2
    return np.max(rho, axis=2)


def _project_centroids(centroids_vox: list[list[float]]) -> tuple[list[float], list[float]]:
    xs = [float(c[1]) for c in centroids_vox]
    ys = [float(c[0]) for c in centroids_vox]
    return xs, ys


def render_case(snaps: np.ndarray, case_dir: Path, case_id: str, row: dict[str, Any]) -> None:
    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rho_max = max(float(np.max(np.abs(s) ** 2)) for s in snaps)
    frames = []
    tvals = np.linspace(0, int(row["effective_steps"]), len(snaps), dtype=int)
    final_geom = node_geometry_from_psi(snaps[-1], L=L_DEFAULT, expected_nodes=5)
    cx, cy = _project_centroids(final_geom["centroids_vox"])
    for idx, psi in enumerate(snaps):
        fig, ax = plt.subplots(figsize=(5.8, 5.2), dpi=96)
        im = ax.imshow(_rho_projection(psi), origin="lower", cmap="magma", vmin=0, vmax=rho_max)
        if idx == len(snaps) - 1 and cx:
            ax.scatter(cx, cy, s=70, facecolors="none", edgecolors="cyan", linewidths=1.5)
            for n, (x, y) in enumerate(zip(cx, cy), start=1):
                ax.text(x + 1, y + 1, str(n), color="cyan", fontsize=8)
        ax.set_title(f"{case_id} t={tvals[idx]}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        frames.append(frame)
    imageio.mimsave(case_dir / "rho.gif", frames, duration=0.12, loop=0)

    picks = np.linspace(0, len(snaps) - 1, min(6, len(snaps)), dtype=int)
    fig, axes = plt.subplots(1, len(picks), figsize=(3.1 * len(picks), 3.3), dpi=110)
    if len(picks) == 1:
        axes = [axes]
    for ax, idx in zip(axes, picks):
        im = ax.imshow(_rho_projection(snaps[idx]), origin="lower", cmap="magma", vmin=0, vmax=rho_max)
        if idx == len(snaps) - 1 and cx:
            ax.scatter(cx, cy, s=50, facecolors="none", edgecolors="cyan", linewidths=1.2)
        ax.set_title(f"t={tvals[idx]}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"rho montage {case_id}", fontsize=11)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    fig.tight_layout()
    fig.savefig(case_dir / "rho_montage.png")
    plt.close(fig)


def score_for_promotion(row: dict[str, Any]) -> float:
    score = 0.0
    if row.get("verdict") == "TRUE_SATURATED_BOUND_STATE":
        score += 40
    elif row.get("verdict") == "NEAR_SATURATED_BOUND_STATE":
        score += 28
    elif "REJECT" not in str(row.get("verdict")):
        score += 10
    if int(row.get("final_node_count", 0) or 0) == 3:
        score += 28
    score -= min(20, abs(float(row.get("spacing_drift_mean_box", 999) or 999)) * 100)
    score -= min(15, float(row.get("final_node_mass_cv", 1) or 1) * 20)
    sp = float(row.get("final_nn_spacing_mean_box", float("nan")) or float("nan"))
    if np.isfinite(sp) and 0.28 <= sp <= 0.53:
        score += 10
    return score


def write_final_report(out_dir: Path, results: list[ScreenResult]) -> None:
    rows = [r.row for r in results]
    ranked = sorted(rows, key=score_for_promotion, reverse=True)
    lines = [
        "# Triangle Spacing Screen Report",
        "",
        "Diagnostic-only Phase C dissipative triangle layout experiment. No solver physics, Hunter logic, validation gates, production defaults, or existing configs were modified.",
        "",
        "Important limitation: these outputs are geometry priors and diagnostic seed/layout candidates only. They do not prove stable 3-node configurations in the conservative or moving substrate.",
        "",
        "## Screen Summary",
        "",
        f"- Cases run: {len(rows)} aligned-phase spacings",
        f"- Fixed parameter hash: `{canonical_params_hash(FEB_PARAMS)}`",
        f"- Fixed params: `{json.dumps(FEB_PARAMS, sort_keys=True)}`",
        "",
        "## Ranked Promotion Candidates",
        "",
    ]
    for i, row in enumerate(ranked[:5], start=1):
        lines.extend(
            [
                f"{i}. `{row['case_id']}`",
                f"   - verdict: `{row['verdict']}`; final nodes: `{row['final_node_count']}`; final spacing mean: `{row['final_nn_spacing_mean_box']}`",
                f"   - spacing drift: `{row['spacing_drift_mean_box']}`; mass CV: `{row['final_node_mass_cv']}`; rho GIF: `{row['rho_gif']}`",
                f"   - caveat: {row['caveat']}",
                "",
            ]
        )
    lines.extend(["## Recommendation", ""])
    promotable = [r for r in ranked if int(r.get("final_node_count", 0) or 0) == 3 and "BLOWUP" not in str(r.get("verdict"))]
    if promotable:
        top = promotable[:3]
        lines.append(
            "Promote the following spacing(s) to T=12000 first, then T=24000 only if the longer run preserves a 3-node count, bounded energy ratio, and acceptable spacing drift:"
        )
        for row in top:
            lines.append(
                f"- `{row['case_id']}`: final spacing mean `{row['final_nn_spacing_mean_box']}`, drift `{row['spacing_drift_mean_box']}`, verdict `{row['verdict']}`."
            )
    else:
        lines.append(
            "No aligned-phase spacing cleanly preserved a 3-node final state in this short screen. Use the least-bad non-blowup case only as a control, or revisit extracted-phase controls before long T promotion."
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `triangle_geometry_index.csv`",
            "- `triangle_geometry_report.md`",
            "- `triangle_ic_manifest.json`",
            "- `triangle_spacing_screen_results.csv`",
            "- `renders/<case_id>/rho.gif`, `rho_montage.png`, `frames.npz`, `summary.json`",
        ]
    )
    (out_dir / "triangle_spacing_screen_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--T", type=int, default=4000)
    ap.add_argument("--n-snap", type=int, default=40)
    ap.add_argument("--geometry-only", action="store_true")
    ap.add_argument("--no-render", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    geometry_rows = extract_probe_geometry(args.out)
    profile = estimate_profile_from_probe([Path(spec["artifact"]) for spec in PROBE_SPECS])
    manifest = write_ic_manifest(args.out, profile, geometry_rows)
    if args.geometry_only:
        print(f"wrote geometry + IC manifest to {args.out}")
        return 0
    results = run_spacing_screen(args.out, manifest, args.T, args.n_snap, render=not args.no_render)
    write_final_report(args.out, results)
    print(f"wrote triangle diagnostic outputs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
