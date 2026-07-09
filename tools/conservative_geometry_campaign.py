"""Conservative-only geometry campaign diagnostics for Phase D/C2.

This is a standalone diagnostic tool. It does not modify solver physics,
production defaults, Hunter objectives, validation gates, or existing configs.

The conservative substrate contract is taken from ``jax_scout.physics``:
``kinetic_mode="conservative"`` uses ``L_k = -1j * D * k^2`` and multiplies the
nonlinear RHS by ``1j``. This module implements that contract in CuPy by wrapping
the existing ``solver.core.ETDRK4Solver`` buffer/FFT machinery in a diagnostic
subclass.

Findings from Phase C dissipative triangle runs are treated only as geometry
priors. Outputs are marked conservative-only and diagnostic-only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.ndimage as ndi


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_utils import generate_canonical_hash  # noqa: E402
from tools.triangle_layout_diagnostic import node_geometry_from_psi  # noqa: E402


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
C2_PARAM_A_FACTOR = 1.15
CONSERVATIVE_PARAMS = {**FEB_PARAMS, "param_a": FEB_PARAMS["param_a"] * C2_PARAM_A_FACTOR, "kinetic_mode": "conservative"}
DEFAULT_OUT = ROOT / "quantule_viz" / "outputs" / "conservative_geometry_campaign"
DEFAULT_L = 10.0
DEFAULT_DT = 0.001
DEFAULT_WIDTH_BOX = 0.08333333333333333
DEFAULT_SEED = 20260708


def pairwise_periodic_box_distances(points_box: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_box, dtype=float)
    out: list[float] = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = pts[i] - pts[j]
            d = d - np.round(d)
            out.append(float(np.linalg.norm(d)))
    return np.asarray(out, dtype=float)


def geometry_points_box(template: str, spacing_box: float, z_offset_box: float | None = None) -> np.ndarray:
    """Return centered node coordinates in box-fraction units, wrapped to [0, 1)."""
    s = float(spacing_box)
    name = template.strip().lower()

    tri = np.array(
        [
            [0.0, s / math.sqrt(3.0), 0.0],
            [-s / 2.0, -s / (2.0 * math.sqrt(3.0)), 0.0],
            [s / 2.0, -s / (2.0 * math.sqrt(3.0)), 0.0],
        ],
        dtype=float,
    )

    if name == "triangle":
        offsets = tri
    elif name == "ablated_triangle":
        offsets = tri[:2]
    elif name == "tetrahedron":
        scale = s / (2.0 * math.sqrt(2.0))
        offsets = scale * np.array(
            [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]],
            dtype=float,
        )
    elif name == "triangular_prism":
        h = s if z_offset_box is None else float(z_offset_box)
        offsets = np.vstack([tri + np.array([0.0, 0.0, -h / 2.0]), tri + np.array([0.0, 0.0, h / 2.0])])
    elif name == "stacked_triangles":
        h = s if z_offset_box is None else float(z_offset_box)
        angle = math.pi / 3.0
        rot = np.array(
            [[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]],
            dtype=float,
        )
        offsets = np.vstack([tri + np.array([0.0, 0.0, -h / 2.0]), (tri @ rot.T) + np.array([0.0, 0.0, h / 2.0])])
    elif name == "octahedron":
        a = s / math.sqrt(2.0)
        offsets = np.array(
            [[a, 0.0, 0.0], [-a, 0.0, 0.0], [0.0, a, 0.0], [0.0, -a, 0.0], [0.0, 0.0, a], [0.0, 0.0, -a]],
            dtype=float,
        )
    else:
        raise ValueError(f"unsupported geometry template {template!r}")

    return (offsets + np.array([0.5, 0.5, 0.5], dtype=float)) % 1.0


def _periodic_delta_axis(coords: np.ndarray, center: float, L: float) -> np.ndarray:
    delta = coords - center
    return (delta + L / 2.0) % L - L / 2.0


def build_geometry_ic(
    N: int,
    L: float,
    points_box: np.ndarray,
    width_box: float,
    amplitude: float,
    amplitude_factors: list[float] | None = None,
    phases: list[float] | None = None,
    noise_level: float = 0.0,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    points = np.asarray(points_box, dtype=float)
    n_nodes = int(points.shape[0])
    amp_factors = np.ones(n_nodes, dtype=float) if amplitude_factors is None else np.asarray(amplitude_factors, dtype=float)
    phase_arr = np.zeros(n_nodes, dtype=float) if phases is None else np.asarray(phases, dtype=float)
    if amp_factors.size != n_nodes:
        raise ValueError("amplitude_factors length must match node count")
    if phase_arr.size != n_nodes:
        raise ValueError("phases length must match node count")

    coords = np.linspace(-L / 2.0, L / 2.0, int(N), endpoint=False)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
    width = float(width_box) * L
    psi = np.zeros((int(N), int(N), int(N)), dtype=np.complex128)
    for idx, point in enumerate(points):
        center = point * L - L / 2.0
        dx = _periodic_delta_axis(X, center[0], L)
        dy = _periodic_delta_axis(Y, center[1], L)
        dz = _periodic_delta_axis(Z, center[2], L)
        profile = float(amplitude) * float(amp_factors[idx]) * np.exp(-(dx * dx + dy * dy + dz * dz) / (2.0 * width * width))
        psi += profile * np.exp(1j * float(phase_arr[idx]))
    if noise_level > 0:
        rng = np.random.default_rng(seed)
        psi += float(noise_level) * (rng.standard_normal(psi.shape) + 1j * rng.standard_normal(psi.shape))
    return psi.astype(np.complex128)


def _apply_perturbations(points: np.ndarray, perturbations: dict[str, Any], seed: int) -> np.ndarray:
    pts = np.asarray(points, dtype=float).copy()
    if perturbations.get("position_jitter_box", 0.0):
        rng = np.random.default_rng(seed)
        pts += rng.normal(0.0, float(perturbations["position_jitter_box"]), size=pts.shape)
    if perturbations.get("z_offset_box", 0.0) and len(pts):
        idx = int(perturbations.get("z_offset_node", len(pts) - 1))
        pts[idx % len(pts), 2] += float(perturbations["z_offset_box"])
    if perturbations.get("remove_node") is not None and len(pts):
        idx = int(perturbations["remove_node"]) % len(pts)
        pts = np.delete(pts, idx, axis=0)
    return pts % 1.0


def points_for_case(template: str, spacing_box: float, perturbations: dict[str, Any], seed: int) -> np.ndarray:
    pts = geometry_points_box(template, spacing_box, perturbations.get("z_stack_box"))
    return _apply_perturbations(pts, perturbations, seed)


def build_case_config(
    case_id: str,
    template: str,
    N: int,
    steps: int,
    spacing_box: float,
    dt: float,
    L: float,
    perturbations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pert = dict(perturbations or {})
    cfg: dict[str, Any] = {
        "case_id": case_id,
        "phase_d_regime": "conservative_c2_geometry_diagnostic_only",
        "solver_component": "tools.conservative_geometry_campaign.ConservativeCupySolver",
        "source_conservative_path": {
            "implementation_reference": "jax_scout/physics.py",
            "mode": "kinetic_mode='conservative'",
            "linear_operator": "L_k = -1j * param_D * k^2",
            "nonlinear_factor": "kfac = 1j",
            "entry_scripts": [
                "jax_scout/phase_d_c2_transport.py",
                "jax_scout/phase_d_c2_soliton_scout.py",
                "jax_scout/phase_d_c2_2_loss_source.py",
            ],
        },
        "params": dict(CONSERVATIVE_PARAMS),
        "simulation": {"n_grid": int(N), "t_steps": int(steps), "dt": float(dt), "l_domain": float(L)},
        "geometry": {
            "template": template,
            "spacing_box": float(spacing_box),
            "width_box": DEFAULT_WIDTH_BOX,
            "amplitude": 1.0,
            "seed": DEFAULT_SEED,
            "perturbations": pert,
            "generator": "tools.conservative_geometry_campaign.build_geometry_ic",
        },
        "limitations": [
            "Standalone diagnostic path only.",
            "Conservative C2 moving-substrate geometry test only.",
            "Dissipative Phase C triangle results are historical geometry priors only.",
            "No claim that dissipative stability transfers to the conservative substrate.",
            "No production solver defaults, Hunter objectives, validation gates, or configs are changed.",
        ],
    }
    cfg["config_hash"] = generate_canonical_hash(cfg)
    return cfg


class ConservativeCupySolver:
    """Diagnostic CuPy implementation of the Phase D/C2 conservative substrate."""

    def __init__(
        self,
        N_grid: int,
        L_domain: float,
        dt: float,
        params: dict[str, Any],
        nonlinear_enabled: bool = True,
        mask_mode: str = "current",
    ):
        import cupy as cp
        from solver.core import ETDRK4Solver
        from solver.kernels import combine_kt_etdrk4, compute_kt_stage_base, compute_kt_stage_c

        class _Wrapped(ETDRK4Solver):
            def __init__(self, n_grid: int, l_domain: float, dt_inner: float, params_inner: dict[str, Any]):
                super().__init__(n_grid, l_domain, dt_inner, params_inner)
                self.kinetic_mode = "conservative"
                self.nonlinear_enabled = bool(nonlinear_enabled)
                self.mask_mode = str(mask_mode)
                self.last_recombination_mask_removed_fraction = 0.0
                self.L_k = (-1j * self.D_diff * self.k_sq).astype(cp.complex128)
                self.reference_dealias_mask = self.dealias_mask.copy()
                if self.mask_mode == "none":
                    self.dealias_mask = cp.ones_like(self.dealias_mask, dtype=cp.float64)
                self._rebuild_etdrk4_coefficients()

            def _rebuild_etdrk4_coefficients(self) -> None:
                M = 64
                theta = cp.exp(1j * cp.pi * (cp.arange(1, M + 1, dtype=cp.float64) - 0.5) / M).astype(cp.complex128)
                w = (self.L_k * self.dt).astype(cp.complex128, copy=False)
                Q_acc = cp.zeros_like(w, dtype=cp.complex128)
                f1_acc = cp.zeros_like(w, dtype=cp.complex128)
                f2_acc = cp.zeros_like(w, dtype=cp.complex128)
                f3_acc = cp.zeros_like(w, dtype=cp.complex128)
                for i in range(M):
                    we = w + theta[i]
                    ew = cp.exp(we)
                    we3 = we**3
                    Q_acc += (cp.exp(we / 2.0) - 1.0) / we
                    f1_acc += (-4.0 - we + ew * (4.0 - 3.0 * we + we**2)) / we3
                    f2_acc += (2.0 + we + ew * (we - 2.0)) / we3
                    f3_acc += (-4.0 - 3.0 * we - we**2 + ew * (4.0 - we)) / we3
                self.Q = self.dt * cp.real(Q_acc / M)
                self.f1 = self.dt * cp.real(f1_acc / M)
                self.f2 = self.dt * cp.real(f2_acc / M)
                self.f3 = self.dt * cp.real(f3_acc / M)
                self.E = cp.exp(w)
                self.E2 = cp.exp(w / 2.0)

            def N_op(self, psi_k):  # noqa: N802 - inherited public API
                if not self.nonlinear_enabled:
                    return cp.zeros_like(psi_k, dtype=self.batch_k.dtype)
                return (1j * super().N_op(psi_k)).astype(self.batch_k.dtype, copy=False)

            def step(self, psi_k):
                N_n = self.N_op(psi_k)
                a_k = compute_kt_stage_base(self.E2, psi_k, self.Q, N_n)
                N_a = self.N_op(a_k)
                self.last_N_a = N_a
                b_k = compute_kt_stage_base(self.E2, psi_k, self.Q, N_a)
                N_b = self.N_op(b_k)
                self.last_N_b = N_b
                c_k = compute_kt_stage_c(self.E2, a_k, self.Q, N_b, N_a)
                N_c = self.N_op(c_k)
                self.last_N_c = N_c
                psi_next_k = combine_kt_etdrk4(psi_k, N_n, N_a, N_b, N_c, self.E, self.f1, self.f2, self.f3)
                if self.mask_mode == "current":
                    before = cp.sum(cp.abs(psi_next_k) ** 2, dtype=cp.float64)
                    psi_masked = psi_next_k * self.dealias_mask
                    after = cp.sum(cp.abs(psi_masked) ** 2, dtype=cp.float64)
                    self.last_recombination_mask_removed_fraction = float((before - after) / cp.maximum(before, cp.float64(1e-30)))
                    psi_next_k = psi_masked
                else:
                    self.last_recombination_mask_removed_fraction = 0.0
                return psi_next_k.astype(self.batch_k.dtype, copy=False)

        self.impl = _Wrapped(N_grid, L_domain, dt, params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.impl, name)


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


def analyse_psi_geometry(psi: np.ndarray, L: float, expected_nodes: int) -> dict[str, Any]:
    geom = node_geometry_from_psi(np.asarray(psi, dtype=np.complex128), L=L, expected_nodes=max(int(expected_nodes), 1))
    profile_metrics = node_profile_metrics(psi, L=L)
    geom["node_widths_box"] = profile_metrics["node_widths_box"]
    geom["node_width_mean_box"] = profile_metrics["node_width_mean_box"]
    geom["node_width_cv"] = profile_metrics["node_width_cv"]
    geom["threshold_node_counts"] = threshold_sensitivity_counts(psi)
    centroids = np.asarray(geom.get("centroids_vox", []), dtype=float)
    N = int(psi.shape[0])
    if centroids.size and len(centroids) >= 2:
        base = centroids[0]
        rel = centroids - base
        rel = rel - N * np.round(rel / N)
        rel_box = rel / N
        z_vals = rel_box[:, 2]
        geom["z_spread_box"] = float(np.max(z_vals) - np.min(z_vals))
        centered = rel_box - rel_box.mean(axis=0)
        _, svals, _ = np.linalg.svd(centered, full_matrices=False)
        geom["planarity_score"] = float(svals[-1] / (svals[0] + 1e-30)) if svals.size else float("nan")
    else:
        geom["z_spread_box"] = float("nan")
        geom["planarity_score"] = float("nan")
    return geom


def profile_overlap(rho_initial: np.ndarray, rho_current: np.ndarray) -> float:
    a = np.asarray(rho_initial, dtype=float).ravel()
    b = np.asarray(rho_current, dtype=float).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-30:
        return float("nan")
    return float(np.clip(np.dot(a, b) / denom, 0.0, 1.0))


def threshold_sensitivity_counts(psi: np.ndarray, fractions: tuple[float, ...] = (0.2, 0.35, 0.5)) -> dict[str, int]:
    rho = np.abs(np.asarray(psi, dtype=np.complex128)) ** 2
    rho_max = float(np.max(rho)) if rho.size else 0.0
    counts: dict[str, int] = {}
    for frac in fractions:
        if rho_max <= 0.0:
            counts[f"thr_{frac:g}"] = 0
            continue
        mask = rho >= (rho_max * float(frac))
        _, count = ndi.label(mask)
        counts[f"thr_{frac:g}"] = int(count)
    return counts


def node_profile_metrics(psi: np.ndarray, L: float, threshold_frac: float = 0.25) -> dict[str, Any]:
    rho = np.abs(np.asarray(psi, dtype=np.complex128)) ** 2
    if rho.size == 0 or float(np.max(rho)) <= 0.0:
        return {"node_widths_box": [], "node_width_mean_box": float("nan"), "node_width_cv": float("nan")}
    N = int(rho.shape[0])
    mask = rho >= (float(np.max(rho)) * float(threshold_frac))
    labels, count = ndi.label(mask)
    widths: list[float] = []
    for label in range(1, count + 1):
        sel = labels == label
        if int(sel.sum()) < 2:
            continue
        idx = np.asarray(np.nonzero(sel), dtype=float).T
        weights = rho[sel].astype(float)
        if float(weights.sum()) <= 0.0:
            continue
        centroid = np.array([_circular_centroid(idx[:, axis], weights, N) for axis in range(3)], dtype=float)
        delta = idx - centroid
        delta = delta - N * np.round(delta / N)
        dist2_box = np.sum((delta / N) ** 2, axis=1)
        widths.append(float(np.sqrt(np.average(dist2_box, weights=weights))))
    arr = np.asarray(widths, dtype=float)
    return {
        "node_widths_box": widths,
        "node_width_mean_box": float(np.mean(arr)) if arr.size else float("nan"),
        "node_width_cv": float(np.std(arr) / (np.mean(arr) + 1e-30)) if arr.size else float("nan"),
    }


def _circular_centroid(coords: np.ndarray, weights: np.ndarray, N: int) -> float:
    theta = 2.0 * np.pi * coords / N
    c = float(np.average(np.cos(theta), weights=weights))
    s = float(np.average(np.sin(theta), weights=weights))
    return float((np.arctan2(s, c) % (2.0 * np.pi)) * N / (2.0 * np.pi))


def centroid_drift_metrics(initial_geom: dict[str, Any], current_geom: dict[str, Any], N: int, t_phys: float) -> dict[str, float | None]:
    initial = np.asarray(initial_geom.get("centroids_vox", []), dtype=float)
    current = np.asarray(current_geom.get("centroids_vox", []), dtype=float)
    if initial.ndim != 2 or current.ndim != 2 or len(initial) == 0 or len(initial) != len(current):
        return {"centroid_drift_mean_box": None, "centroid_drift_max_box": None, "centroid_speed_mean_box": None}
    dist = np.empty((len(initial), len(current)), dtype=float)
    for i, a in enumerate(initial):
        for j, b in enumerate(current):
            d = b - a
            d = d - N * np.round(d / N)
            dist[i, j] = float(np.linalg.norm(d) / N)
    if len(initial) <= 8:
        import itertools

        best = min((sum(dist[i, perm[i]] for i in range(len(initial))), perm) for perm in itertools.permutations(range(len(initial))))[1]
        vals = np.asarray([dist[i, best[i]] for i in range(len(initial))], dtype=float)
    else:
        vals = np.min(dist, axis=1)
    mean = float(np.mean(vals))
    return {
        "centroid_drift_mean_box": mean,
        "centroid_drift_max_box": float(np.max(vals)),
        "centroid_speed_mean_box": mean / float(t_phys) if t_phys > 0 else None,
    }


def pairwise_distance_drift(initial_geom: dict[str, Any], current_geom: dict[str, Any]) -> dict[str, float | None]:
    a = np.sort(np.asarray(initial_geom.get("pairwise_distances_box", []), dtype=float))
    b = np.sort(np.asarray(current_geom.get("pairwise_distances_box", []), dtype=float))
    if a.size == 0 or a.size != b.size or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return {"pairwise_distance_drift_mean_box": None, "pairwise_distance_drift_max_box": None}
    diff = np.abs(b - a)
    return {
        "pairwise_distance_drift_mean_box": float(np.mean(diff)),
        "pairwise_distance_drift_max_box": float(np.max(diff)),
    }


def numerical_loss_warning(norm_initial: float, norm_final: float, warn_fraction: float = 0.02) -> str:
    if not np.isfinite(norm_initial) or abs(norm_initial) <= 1e-30 or not np.isfinite(norm_final):
        return "norm unavailable"
    frac = (float(norm_final) - float(norm_initial)) / abs(float(norm_initial))
    if abs(frac) >= float(warn_fraction):
        return f"total norm changed by {frac:.3%}; investigate numerical loss/dispersion/dealiasing"
    return ""


def late_trends(history: list[dict[str, Any]]) -> dict[str, float | None]:
    if len(history) < 4:
        return {"raw_energy_fractional_trend": None, "rho_max_fractional_trend": None, "node_count_trend": None}

    def trend(key: str) -> float | None:
        half = len(history) // 2
        vals = np.asarray([float(item[key]) for item in history[half:] if key in item and item[key] is not None], dtype=float)
        if vals.size < 3 or not np.all(np.isfinite(vals)):
            return None
        xs = np.arange(vals.size, dtype=float)
        slope, intercept = np.polyfit(xs, vals, 1)
        start = float(intercept)
        end = float(slope * (vals.size - 1) + intercept)
        return float((end - start) / (abs(start) + 1e-30))

    return {
        "raw_energy_fractional_trend": trend("raw_energy"),
        "rho_max_fractional_trend": trend("rho_max"),
        "node_count_trend": trend("node_count"),
    }


def _max_abs_history_value(history: list[dict[str, Any]], key: str) -> float | None:
    vals = []
    for item in history:
        value = item.get(key)
        if value is None:
            continue
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(val):
            vals.append(abs(val))
    return float(max(vals)) if vals else None


def run_external_psi0_loop(
    psi0_np: np.ndarray,
    cfg: dict[str, Any],
    sample_every: int,
    collapse_threshold: float,
) -> dict[str, Any]:
    import cupy as cp

    sim = cfg["simulation"]
    N = int(sim["n_grid"])
    L = float(sim["l_domain"])
    dt = float(sim["dt"])
    steps = int(sim["t_steps"])
    expected_nodes = int(np.asarray(cfg["geometry"]["points_box"]).shape[0])
    solver = ConservativeCupySolver(N, L, dt, dict(cfg["params"]))
    psi = cp.asarray(psi0_np, dtype=cp.complex128)
    psi_k = solver.fft_single(psi) * solver.dealias_mask
    ic_raw_energy = float(cp.sum(cp.abs(solver.ifft_single(psi_k)) ** 2, dtype=cp.float64))
    initial_rho_profile = np.abs(np.asarray(psi0_np, dtype=np.complex128)) ** 2
    initial_geom = analyse_psi_geometry(psi0_np, L=L, expected_nodes=expected_nodes)
    history: list[dict[str, Any]] = []
    geometry_history: list[dict[str, Any]] = []
    final_step = -1
    fail_reason = ""
    t0 = time.time()

    for step in range(steps):
        final_step = step
        psi_k = solver.step(psi_k)
        if step % int(sample_every) == 0 or step == steps - 1:
            psi_real = solver.ifft_single(psi_k)
            if not bool(cp.isfinite(psi_real).all()):
                fail_reason = f"nonfinite at step {step}"
                break
            max_amp = float(cp.max(cp.abs(psi_real)))
            if max_amp > collapse_threshold:
                fail_reason = f"amplitude {max_amp:.3e} exceeded collapse threshold"
                break
            rho = cp.abs(psi_real) ** 2
            psi_cpu = cp.asnumpy(psi_real)
            rho_cpu = np.abs(psi_cpu) ** 2
            geom = analyse_psi_geometry(psi_cpu, L=L, expected_nodes=expected_nodes)
            t_phys = float((step + 1) * dt)
            drift = centroid_drift_metrics(initial_geom, geom, N=N, t_phys=t_phys)
            pair_drift = pairwise_distance_drift(initial_geom, geom)
            total_norm = float(cp.sum(rho, dtype=cp.float64))
            record = {
                "step": int(step),
                "t_phys": t_phys,
                "total_norm": total_norm,
                "total_norm_fractional_change_from_t0": float((total_norm - ic_raw_energy) / (abs(ic_raw_energy) + 1e-30)),
                "raw_energy": total_norm,
                "raw_energy_fractional_change_from_t0": float((total_norm - ic_raw_energy) / (abs(ic_raw_energy) + 1e-30)),
                "rho_max": float(cp.max(rho)),
                "rho_mean": float(cp.mean(rho)),
                "max_amp": max_amp,
                "node_count": geom.get("node_count"),
                "mass_cv": geom.get("node_mass_cv"),
                "peak_cv": geom.get("rho_peak_cv"),
                "node_masses": geom.get("node_masses"),
                "node_widths_box": geom.get("node_widths_box"),
                "node_width_mean_box": geom.get("node_width_mean_box"),
                "node_width_cv": geom.get("node_width_cv"),
                "profile_overlap_initial_rho": profile_overlap(initial_rho_profile, rho_cpu),
                "threshold_node_counts": geom.get("threshold_node_counts"),
                "z_spread_box": geom.get("z_spread_box"),
                "planarity_score": geom.get("planarity_score"),
                **drift,
                **pair_drift,
            }
            history.append(record)
            geometry_history.append({"step": int(step), "t_phys": record["t_phys"], "geometry": geom})

    psi_fin = solver.ifft_single(psi_k)
    rho_fin = cp.abs(psi_fin) ** 2
    finite = bool(cp.isfinite(psi_fin).all())
    cp.cuda.Device().synchronize()
    psi_fin_np = cp.asnumpy(psi_fin)
    rho_fin_np = cp.asnumpy(rho_fin)
    result = {
        "psi0": cp.asnumpy(psi),
        "psi_fin": psi_fin_np,
        "rho_fin": rho_fin_np,
        "history": history,
        "geometry_history": geometry_history,
        "summary": {
            "finite": finite and not fail_reason,
            "fail_reason": fail_reason,
            "final_step": int(final_step),
            "requested_steps": steps,
            "ic_raw_energy": ic_raw_energy,
            "final_raw_energy": float(cp.sum(rho_fin, dtype=cp.float64)),
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


def case_paths(out_dir: Path, case_id: str) -> dict[str, Path]:
    case_dir = Path(out_dir) / case_id
    return {
        "case_dir": case_dir,
        "artifact": case_dir / f"{case_id}_conservative.npz",
        "metadata": case_dir / f"{case_id}_metadata.json",
        "final_rho_image": case_dir / f"{case_id}_final_rho.png",
        "plotly_html": case_dir / f"{case_id}_centroids.html",
    }


def _case_id(template: str, spacing: float, N: int, steps: int, suffix: str = "") -> str:
    safe = template.lower().replace(" ", "_")
    tail = f"_{suffix}" if suffix else ""
    return f"c2_{safe}_s{spacing:.2f}_N{N}_T{steps}{tail}"


def _phase_and_amp_controls(n_nodes: int, perturbations: dict[str, Any]) -> tuple[list[float], list[float]]:
    phases = list(perturbations.get("phase_offsets", [0.0] * n_nodes))
    amps = list(perturbations.get("amplitude_factors", [1.0] * n_nodes))
    if len(phases) < n_nodes:
        phases.extend([0.0] * (n_nodes - len(phases)))
    if len(amps) < n_nodes:
        amps.extend([1.0] * (n_nodes - len(amps)))
    return phases[:n_nodes], amps[:n_nodes]


def run_case(
    out_dir: Path,
    template: str,
    spacing_box: float,
    N: int,
    steps: int,
    dt: float,
    L: float,
    perturbations: dict[str, Any] | None = None,
    sample_every: int = 100,
    collapse_threshold: float = 1e6,
    require_gpu_name: str | None = "NVIDIA GeForce GTX 1080",
    force: bool = False,
    suffix: str = "",
) -> dict[str, Any]:
    pert = dict(perturbations or {})
    case_id = _case_id(template, spacing_box, N, steps, suffix)
    paths = case_paths(out_dir, case_id)
    if paths["metadata"].exists() and not force:
        return json.loads(paths["metadata"].read_text(encoding="utf-8"))

    paths["case_dir"].mkdir(parents=True, exist_ok=True)
    cfg = build_case_config(case_id, template, N, steps, spacing_box, dt, L, pert)
    points = points_for_case(template, spacing_box, pert, seed=int(pert.get("seed", DEFAULT_SEED)))
    cfg["geometry"]["points_box"] = points.tolist()
    cfg["geometry"]["node_count_initial_design"] = int(points.shape[0])
    cfg["config_hash"] = generate_canonical_hash(cfg)
    phases, amps = _phase_and_amp_controls(len(points), pert)
    gpu = _gpu_info(require_gpu_name)
    psi0 = build_geometry_ic(
        N=N,
        L=L,
        points_box=points,
        width_box=float(pert.get("width_box", DEFAULT_WIDTH_BOX)),
        amplitude=float(pert.get("amplitude", 1.0)),
        amplitude_factors=amps,
        phases=phases,
        noise_level=float(pert.get("noise_level", pert.get("low_noise", 0.0))),
        seed=int(pert.get("seed", DEFAULT_SEED)),
    )
    result = run_external_psi0_loop(psi0, cfg, sample_every=sample_every, collapse_threshold=collapse_threshold)
    initial_geom = analyse_psi_geometry(psi0, L=L, expected_nodes=len(points))
    final_geom = analyse_psi_geometry(result["psi_fin"], L=L, expected_nodes=max(len(points), 1))
    trends = late_trends(result["history"])
    final_history = result["history"][-1] if result["history"] else {}
    max_norm_change = _max_abs_history_value(result["history"], "total_norm_fractional_change_from_t0")
    render_final_rho_orthogonal(result["rho_fin"], final_geom, paths["final_rho_image"])
    write_centroid_plotly_html(paths["plotly_html"], result["geometry_history"], cfg)

    row = {
        "case_id": case_id,
        "template": template,
        "spacing_box": float(spacing_box),
        "N": int(N),
        "steps": int(steps),
        "dt": float(dt),
        "final_finite": bool(result["summary"]["finite"]),
        "fail_reason": result["summary"]["fail_reason"],
        "initial_node_count": initial_geom.get("node_count"),
        "final_node_count": final_geom.get("node_count"),
        "pairwise_node_distances_box": json.dumps(final_geom.get("pairwise_distances_box", [])),
        "spacing_drift_mean_box": _spacing_drift(initial_geom, final_geom),
        "mass_cv": final_geom.get("node_mass_cv"),
        "peak_cv": final_geom.get("rho_peak_cv"),
        "z_spread_box": final_geom.get("z_spread_box"),
        "planarity_score": final_geom.get("planarity_score"),
        "rho_max": result["summary"]["rho_max"],
        "rho_mean": result["summary"]["rho_mean"],
        "raw_energy": result["summary"]["final_raw_energy"],
        "initial_total_norm": result["summary"]["ic_raw_energy"],
        "final_total_norm": result["summary"]["final_raw_energy"],
        "total_norm_fractional_change": float(
            (result["summary"]["final_raw_energy"] - result["summary"]["ic_raw_energy"])
            / (abs(result["summary"]["ic_raw_energy"]) + 1e-30)
        ),
        "max_abs_total_norm_fractional_change": max_norm_change,
        "node_width_mean_box": final_geom.get("node_width_mean_box"),
        "node_width_cv": final_geom.get("node_width_cv"),
        "node_widths_box": json.dumps(final_geom.get("node_widths_box", [])),
        "profile_overlap_initial_rho_final": final_history.get("profile_overlap_initial_rho"),
        "centroid_drift_mean_box": final_history.get("centroid_drift_mean_box"),
        "centroid_drift_max_box": final_history.get("centroid_drift_max_box"),
        "centroid_speed_mean_box": final_history.get("centroid_speed_mean_box"),
        "pairwise_distance_drift_mean_box": final_history.get("pairwise_distance_drift_mean_box"),
        "pairwise_distance_drift_max_box": final_history.get("pairwise_distance_drift_max_box"),
        "threshold_node_counts_final": json.dumps(final_geom.get("threshold_node_counts", {})),
        "late_raw_energy_fractional_trend": trends["raw_energy_fractional_trend"],
        "late_rho_max_fractional_trend": trends["rho_max_fractional_trend"],
        "late_node_count_trend": trends["node_count_trend"],
        "wallclock_sec": result["summary"]["wallclock_sec"],
        "artifact": str(paths["artifact"]),
        "metadata": str(paths["metadata"]),
        "final_rho_image": str(paths["final_rho_image"]),
        "plotly_html": str(paths["plotly_html"]),
        "config_hash": cfg["config_hash"],
        "caveat": "Conservative C2 diagnostic only; dissipative results are geometry priors only.",
    }
    row["numerical_loss_warning"] = numerical_loss_warning(row["initial_total_norm"], row["final_total_norm"])
    row["arrangement_outcome"] = arrangement_outcome(row)
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "conservative_case",
        "gpu": gpu,
        "config": cfg,
        "summary": result["summary"],
        "initial_geometry": initial_geom,
        "final_geometry": final_geom,
        "history": result["history"],
        "geometry_history": result["geometry_history"],
        "screen_row": row,
        "artifact": str(paths["artifact"]),
        "production_files_modified": False,
        "caveat": "Conservative C2 diagnostic only; no conservative stability claim from visuals alone.",
    }
    np.savez_compressed(
        paths["artifact"],
        psi0=result["psi0"].astype(np.complex128),
        psi_fin=result["psi_fin"].astype(np.complex128),
        rho_fin=result["rho_fin"].astype(np.float64),
        history_json=json.dumps(result["history"]),
        geometry_history_json=json.dumps(result["geometry_history"]),
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def _spacing_drift(initial_geom: dict[str, Any], final_geom: dict[str, Any]) -> float | None:
    a = initial_geom.get("nn_spacing_mean_box")
    b = final_geom.get("nn_spacing_mean_box")
    try:
        af = float(a)
        bf = float(b)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(af) and np.isfinite(bf)):
        return None
    return float(bf - af)


def arrangement_outcome(row: dict[str, Any]) -> str:
    if not bool(row.get("final_finite")):
        reason = str(row.get("fail_reason") or "")
        return "nonfinite" if "nonfinite" in reason.lower() else "failed"
    try:
        initial = int(row.get("initial_node_count", 0))
        final = int(row.get("final_node_count", 0))
    except (TypeError, ValueError):
        return "unclassified"
    if final <= 0:
        return "dispersed"
    if final < initial:
        return "merged"
    if final > initial:
        return "fragmented"
    z_spread = float(row.get("z_spread_box") or 0.0)
    planarity = float(row.get("planarity_score") or 0.0)
    is_planar = z_spread <= 0.06 and planarity <= 0.10
    if initial <= 3:
        return "stayed_planar" if is_planar else "became_volumetric"
    return "became_planar" if is_planar else "stayed_volumetric"


def render_final_rho_orthogonal(rho_fin: np.ndarray, geometry: dict[str, Any], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rho = np.asarray(rho_fin, dtype=float)
    projections = [np.max(rho, axis=2), np.max(rho, axis=1), np.max(rho, axis=0)]
    titles = ["XY max", "XZ max", "YZ max"]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), dpi=120)
    vmax = float(np.max(rho)) if rho.size else 1.0
    for ax, img, title in zip(axes, projections, titles):
        im = ax.imshow(img, origin="lower", cmap="magma", vmin=0.0, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"conservative final rho | nodes={geometry.get('node_count')}", fontsize=10)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_centroid_plotly_html(path: Path, geometry_history: list[dict[str, Any]], cfg: dict[str, Any]) -> None:
    frames = []
    for item in geometry_history:
        geom = item.get("geometry", {})
        frames.append(
            {
                "step": item.get("step"),
                "t_phys": item.get("t_phys"),
                "centroids_vox": geom.get("centroids_vox", []),
                "node_count": geom.get("node_count"),
                "pairwise_distances_box": geom.get("pairwise_distances_box", []),
                "mass_cv": geom.get("node_mass_cv"),
                "peak_cv": geom.get("rho_peak_cv"),
                "z_spread_box": geom.get("z_spread_box"),
                "planarity_score": geom.get("planarity_score"),
            }
        )
    payload = json.dumps({"config": cfg, "frames": frames})
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{cfg['case_id']} centroid diagnostics</title>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <style>body{{font-family:Arial,sans-serif;margin:18px}} #plot{{width:100%;height:680px}} pre{{white-space:pre-wrap}}</style>
</head>
<body>
  <h1>{cfg['case_id']}</h1>
  <p>Conservative C2 diagnostic only. Dissipative triangle results are geometry priors only.</p>
  <div id="plot"></div>
  <pre id="metrics"></pre>
  <script>
  const payload = {payload};
  const traces = [];
  const maxNodes = Math.max(...payload.frames.map(f => (f.centroids_vox || []).length), 0);
  for (let i = 0; i < maxNodes; i++) {{
    const xs=[], ys=[], zs=[], text=[];
    for (const f of payload.frames) {{
      if ((f.centroids_vox || [])[i]) {{
        xs.push(f.centroids_vox[i][0]); ys.push(f.centroids_vox[i][1]); zs.push(f.centroids_vox[i][2]);
        text.push(`step=${{f.step}} nodes=${{f.node_count}}`);
      }}
    }}
    traces.push({{type:'scatter3d', mode:'lines+markers', x:xs, y:ys, z:zs, text:text, name:`node ${{i+1}}`}});
  }}
  Plotly.newPlot('plot', traces, {{scene:{{xaxis:{{title:'x vox'}},yaxis:{{title:'y vox'}},zaxis:{{title:'z vox'}}}}, margin:{{l:0,r:0,b:0,t:20}}}});
  document.getElementById('metrics').textContent = JSON.stringify(payload.frames.map(f => ({{
    step:f.step, t_phys:f.t_phys, node_count:f.node_count, distances:f.pairwise_distances_box,
    mass_cv:f.mass_cv, peak_cv:f.peak_cv, z_spread_box:f.z_spread_box, planarity_score:f.planarity_score
  }})), null, 2);
  </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


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


def write_campaign_report(path: Path, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    lines = [
        f"# Conservative Geometry Campaign {manifest.get('campaign_id', '')}",
        "",
        "Standalone conservative-only Phase D/C2 diagnostic campaign using a CuPy wrapper of the C2 substrate contract.",
        "",
        "These results do not inherit or prove the dissipative Phase C triangle behaviour.",
        "",
        "## Results",
        "",
        "| case | template | outcome | finite | nodes | rho max | raw energy | drift | mass CV | peak CV | z spread | late rho max trend |",
        "|---|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['template']} | {row.get('arrangement_outcome', '')} | "
            f"{row['final_finite']} | {row['final_node_count']} | "
            f"{fmt(row['rho_max'])} | {fmt(row['raw_energy'])} | {fmt(row['spacing_drift_mean_box'])} | "
            f"{fmt(row['mass_cv'])} | {fmt(row['peak_cv'])} | {fmt(row['z_spread_box'])} | "
            f"{fmt(row['late_rho_max_fractional_trend'])} |"
        )
    lines.extend(
        [
            "",
            "## Invariant/Profile Audit",
            "",
            "| case | norm change | max norm change | width mean | profile overlap | centroid drift | pairwise drift | warning |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {fmt(row.get('total_norm_fractional_change'))} | "
            f"{fmt(row.get('max_abs_total_norm_fractional_change'))} | {fmt(row.get('node_width_mean_box'))} | "
            f"{fmt(row.get('profile_overlap_initial_rho_final'))} | {fmt(row.get('centroid_drift_mean_box'))} | "
            f"{fmt(row.get('pairwise_distance_drift_mean_box'))} | {row.get('numerical_loss_warning', '')} |"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Node counts are diagnostic detector outputs, not validation-gate certifications.",
            "- The runner is conservative-only and standalone; it does not alter production defaults or configs.",
            "- Plotly HTML uses a CDN script when opened in a browser; JSON data are embedded in the file.",
            "- Unstable, merging, dispersing, and non-finite outcomes are useful campaign observations, not failed automation.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def campaign_output_paths(out_dir: Path, campaign_id: str) -> dict[str, Path]:
    return {
        "named_csv": out_dir / f"{campaign_id}_results.csv",
        "named_report": out_dir / f"{campaign_id}_report.md",
        "named_summary": out_dir / f"{campaign_id}_summary.json",
        "stable_csv": out_dir / "campaign_results.csv",
        "stable_report": out_dir / "campaign_report.md",
        "resolved_manifest": out_dir / "campaign_manifest_resolved.json",
        "dashboard": out_dir / "campaign_dashboard.html",
    }


def invariant_audit_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "csv": out_dir / "conservative_invariant_audit_results.csv",
        "report": out_dir / "conservative_invariant_audit_report.md",
    }


def norm_loss_isolation_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "csv": out_dir / "conservative_norm_loss_isolation_results.csv",
        "report": out_dir / "conservative_norm_loss_isolation_report.md",
        "norm_plot": out_dir / "conservative_norm_loss_norm_vs_time.png",
        "rho_plot": out_dir / "conservative_norm_loss_rho_max_vs_time.png",
    }


def steps_for_physical_time(physical_time: float, dt: float) -> int:
    if dt <= 0:
        raise ValueError("dt must be positive")
    return int(round(float(physical_time) / float(dt)))


def high_k_fraction(psi_k: np.ndarray, dealias_mask: np.ndarray) -> float:
    spec = np.abs(np.asarray(psi_k)) ** 2
    total = float(np.sum(spec))
    if total <= 1e-30:
        return 0.0
    mask = np.asarray(dealias_mask).astype(bool)
    return float(np.sum(spec[~mask]) / total)


def write_invariant_audit_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    audit_fields = [
        "case_id",
        "template",
        "spacing_box",
        "final_finite",
        "fail_reason",
        "initial_node_count",
        "final_node_count",
        "arrangement_outcome",
        "initial_total_norm",
        "final_total_norm",
        "total_norm_fractional_change",
        "max_abs_total_norm_fractional_change",
        "raw_energy",
        "late_raw_energy_fractional_trend",
        "rho_max",
        "rho_mean",
        "late_rho_max_fractional_trend",
        "node_width_mean_box",
        "node_width_cv",
        "node_widths_box",
        "mass_cv",
        "peak_cv",
        "profile_overlap_initial_rho_final",
        "centroid_drift_mean_box",
        "centroid_drift_max_box",
        "centroid_speed_mean_box",
        "pairwise_distance_drift_mean_box",
        "pairwise_distance_drift_max_box",
        "threshold_node_counts_final",
        "z_spread_box",
        "planarity_score",
        "numerical_loss_warning",
        "metadata",
        "artifact",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=audit_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def audit_recommendation(rows: list[dict[str, Any]]) -> str:
    norm_changes = [abs(float(r.get("total_norm_fractional_change") or 0.0)) for r in rows]
    rho_trends = [float(r.get("late_rho_max_fractional_trend") or 0.0) for r in rows]
    overlaps = [float(r.get("profile_overlap_initial_rho_final") or 0.0) for r in rows]
    warnings = [r for r in rows if r.get("numerical_loss_warning")]
    if warnings or (norm_changes and max(norm_changes) >= 0.02):
        return "numerical loss / dealiasing-dispersion warning"
    if rho_trends and np.nanmedian(rho_trends) < -0.10:
        return "dispersion or profile broadening likely"
    if overlaps and np.nanmedian(overlaps) < 0.70:
        return "profile broadening / shape drift likely"
    return "unclear; no large norm loss but longer confirmation is needed"


def write_invariant_audit_report(path: Path, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    recommendation = audit_recommendation(rows)
    lines = [
        "# Conservative Invariant Audit",
        "",
        "Conservative C2 diagnostic-only audit. Dissipative results remain historical geometry priors only.",
        "",
        "## Loop Contract",
        "",
        "- Conservative CuPy wrapper linear operator: `L_k = -1j * param_D * k^2`.",
        "- Nonlinear RHS is multiplied by `1j`, matching `jax_scout.physics` C2 `kfac = 1j` contract.",
        "- `param_eta` remains present in provenance but is inactive in the conservative linear operator.",
        "- The conservative diagnostic loop does not call dynamic filters or phase centering.",
        "- The ETDRK4 step still applies the existing spectral dealias mask after nonlinear transforms and after recombination; this can remove high-k content and reduce total norm.",
        "- No post-step amplitude normalization is applied.",
        "",
        "## Results",
        "",
        "| case | outcome | norm change | width mean | profile overlap | centroid drift | pairwise drift | late rho max trend | warning |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row.get('arrangement_outcome', '')} | "
            f"{fmt(row.get('total_norm_fractional_change'))} | {fmt(row.get('node_width_mean_box'))} | "
            f"{fmt(row.get('profile_overlap_initial_rho_final'))} | {fmt(row.get('centroid_drift_mean_box'))} | "
            f"{fmt(row.get('pairwise_distance_drift_mean_box'))} | {fmt(row.get('late_rho_max_fractional_trend'))} | "
            f"{row.get('numerical_loss_warning', '')} |"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            recommendation,
            "",
            "## Caveats",
            "",
            "- `raw_energy` and `total_norm` are both reported as `sum(abs(psi)^2)` in this diagnostic audit.",
            "- Node widths are estimated from thresholded rho components and are detector-sensitive.",
            "- Profile overlap is against the initial full rho field, not an exact conservative soliton eigenprofile.",
            "- Stability is not claimed from this medium campaign.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _find_native_c2_profile() -> Path | None:
    for path in (ROOT / "sweep_runs").glob("**/*.npz"):
        text = str(path).lower()
        if "c2" not in text and "soliton" not in text and "phase_d" not in text:
            continue
        try:
            with np.load(path, allow_pickle=False) as data:
                if any(key in data.files for key in ("psi", "psi0", "psi_fin", "settled", "psi_settled")):
                    return path
        except Exception:
            continue
    return None


def _load_native_profile(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        for key in ("psi", "psi_fin", "psi0", "settled", "psi_settled"):
            if key in data.files:
                return np.asarray(data[key], dtype=np.complex128)
    raise ValueError(f"no psi-like array in {path}")


def run_norm_loss_case(
    case_id: str,
    group: str,
    out_dir: Path,
    N: int,
    dt: float,
    physical_time: float,
    nonlinear_enabled: bool,
    mask_mode: str,
    psi0: np.ndarray,
    sample_count: int = 40,
    collapse_threshold: float = 1e6,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import cupy as cp

    steps = steps_for_physical_time(physical_time, dt)
    sample_every = max(1, steps // max(1, sample_count))
    params = dict(CONSERVATIVE_PARAMS)
    solver = ConservativeCupySolver(N, DEFAULT_L, dt, params, nonlinear_enabled=nonlinear_enabled, mask_mode=mask_mode)
    psi = cp.asarray(psi0, dtype=cp.complex128)
    input_norm = float(cp.sum(cp.abs(psi) ** 2, dtype=cp.float64))
    psi_k = solver.fft_single(psi) * solver.dealias_mask
    projected = solver.ifft_single(psi_k)
    initial_norm = float(cp.sum(cp.abs(projected) ** 2, dtype=cp.float64))
    initial_rho = cp.asnumpy(cp.abs(projected) ** 2)
    initial_geom = analyse_psi_geometry(cp.asnumpy(projected), L=DEFAULT_L, expected_nodes=3)
    mask_bool = cp.asnumpy(getattr(solver, "reference_dealias_mask", solver.dealias_mask)).astype(bool)
    history: list[dict[str, Any]] = []
    fail_reason = ""
    t0 = time.time()
    recomb_removed_samples: list[float] = []

    for step in range(steps):
        psi_k = solver.step(psi_k)
        recomb_removed_samples.append(float(getattr(solver, "last_recombination_mask_removed_fraction", 0.0)))
        if step % sample_every == 0 or step == steps - 1:
            psi_real = solver.ifft_single(psi_k)
            if not bool(cp.isfinite(psi_real).all()):
                fail_reason = f"nonfinite at step {step}"
                break
            max_amp = float(cp.max(cp.abs(psi_real)))
            if max_amp > collapse_threshold:
                fail_reason = f"amplitude {max_amp:.3e} exceeded collapse threshold"
                break
            rho = cp.abs(psi_real) ** 2
            psi_cpu = cp.asnumpy(psi_real)
            rho_cpu = np.abs(psi_cpu) ** 2
            geom = analyse_psi_geometry(psi_cpu, L=DEFAULT_L, expected_nodes=8)
            drift = centroid_drift_metrics(initial_geom, geom, N=N, t_phys=(step + 1) * dt)
            pair_drift = pairwise_distance_drift(initial_geom, geom)
            spec = cp.abs(psi_k) ** 2
            spec_total = float(cp.sum(spec, dtype=cp.float64))
            high_frac = 0.0 if spec_total <= 1e-30 else float(cp.sum(spec[~cp.asarray(mask_bool)], dtype=cp.float64) / spec_total)
            total_norm = float(cp.sum(rho, dtype=cp.float64))
            history.append(
                {
                    "case_id": case_id,
                    "group": group,
                    "step": int(step),
                    "t_phys": float((step + 1) * dt),
                    "total_norm": total_norm,
                    "total_norm_fractional_change_from_projected_t0": float((total_norm - initial_norm) / (abs(initial_norm) + 1e-30)),
                    "total_norm_fractional_change_from_input": float((total_norm - input_norm) / (abs(input_norm) + 1e-30)),
                    "rho_max": float(cp.max(rho)),
                    "rho_max_fractional_change_from_t0": float((float(cp.max(rho)) - float(np.max(initial_rho))) / (abs(float(np.max(initial_rho))) + 1e-30)),
                    "rho_mean": float(cp.mean(rho)),
                    "node_count": geom.get("node_count"),
                    "threshold_node_counts": geom.get("threshold_node_counts"),
                    "node_width_mean_box": geom.get("node_width_mean_box"),
                    "profile_overlap_initial_rho": profile_overlap(initial_rho, rho_cpu),
                    "high_k_fraction": high_frac,
                    "recombination_mask_removed_fraction_last": float(recomb_removed_samples[-1]) if recomb_removed_samples else 0.0,
                    **drift,
                    **pair_drift,
                }
            )

    final = history[-1] if history else {}
    trends = late_trends(history)
    row = {
        "case_id": case_id,
        "group": group,
        "N": int(N),
        "steps": int(steps),
        "dt": float(dt),
        "physical_time": float(physical_time),
        "nonlinear_enabled": bool(nonlinear_enabled),
        "mask_mode": mask_mode,
        "finite": not fail_reason,
        "fail_reason": fail_reason,
        "input_norm": input_norm,
        "projected_initial_norm": initial_norm,
        "projection_norm_fractional_change": float((initial_norm - input_norm) / (abs(input_norm) + 1e-30)),
        "final_norm": final.get("total_norm"),
        "final_norm_fractional_change_from_projected_t0": final.get("total_norm_fractional_change_from_projected_t0"),
        "final_norm_fractional_change_from_input": final.get("total_norm_fractional_change_from_input"),
        "rho_max_fractional_change": final.get("rho_max_fractional_change_from_t0"),
        "rho_max_final": final.get("rho_max"),
        "node_width_mean_box_final": final.get("node_width_mean_box"),
        "profile_overlap_initial_rho_final": final.get("profile_overlap_initial_rho"),
        "centroid_drift_mean_box": final.get("centroid_drift_mean_box"),
        "pairwise_distance_drift_mean_box": final.get("pairwise_distance_drift_mean_box"),
        "high_k_fraction_final": final.get("high_k_fraction"),
        "recombination_mask_removed_fraction_mean": float(np.mean(recomb_removed_samples)) if recomb_removed_samples else 0.0,
        "recombination_mask_removed_fraction_max": float(np.max(recomb_removed_samples)) if recomb_removed_samples else 0.0,
        "final_node_count": final.get("node_count"),
        "threshold_node_counts_final": json.dumps(final.get("threshold_node_counts", {})),
        "late_norm_trend": trends.get("raw_energy_fractional_trend"),
        "late_rho_max_trend": trends.get("rho_max_fractional_trend"),
        "wallclock_sec": round(time.time() - t0, 3),
    }
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return row, history


def write_norm_loss_plots(out_dir: Path, histories: list[dict[str, Any]], paths: dict[str, Path]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_case: dict[str, list[dict[str, Any]]] = {}
    for item in histories:
        by_case.setdefault(str(item["case_id"]), []).append(item)
    for key, out_path, ylabel in (
        ("total_norm_fractional_change_from_projected_t0", paths["norm_plot"], "norm fractional change from projected t0"),
        ("rho_max_fractional_change_from_t0", paths["rho_plot"], "rho_max fractional change from t0"),
    ):
        fig, ax = plt.subplots(figsize=(9, 4.8), dpi=120)
        for case_id, rows in by_case.items():
            rows_sorted = sorted(rows, key=lambda r: r["t_phys"])
            ax.plot([r["t_phys"] for r in rows_sorted], [r.get(key, np.nan) for r in rows_sorted], label=case_id, linewidth=1)
        ax.set_xlabel("physical time")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)


def infer_norm_loss_mechanism(rows: list[dict[str, Any]]) -> str:
    lookup = {r["case_id"]: r for r in rows}
    current = [r for r in rows if r["group"] == "mask_isolation" and r["mask_mode"] == "current"]
    none = [r for r in rows if r["group"] == "mask_isolation" and r["mask_mode"] == "none"]
    if current and none:
        cur_loss = abs(float(current[0].get("final_norm_fractional_change_from_projected_t0") or 0.0))
        none_loss = abs(float(none[0].get("final_norm_fractional_change_from_projected_t0") or 0.0))
        if cur_loss > 0.05 and none_loss < cur_loss * 0.5:
            return "dealiasing/mask-driven numerical loss"
    dt_rows = sorted([r for r in rows if r["group"] == "timestep"], key=lambda r: float(r["dt"]), reverse=True)
    if len(dt_rows) >= 2:
        losses = [abs(float(r.get("final_norm_fractional_change_from_projected_t0") or 0.0)) for r in dt_rows]
        if all(losses[i + 1] < losses[i] * 0.85 for i in range(len(losses) - 1)):
            return "timestep-sensitive numerical loss"
    res_rows = sorted([r for r in rows if r["group"] == "resolution"], key=lambda r: int(r["N"]))
    if len(res_rows) >= 2:
        losses = [abs(float(r.get("final_norm_fractional_change_from_projected_t0") or 0.0)) for r in res_rows]
        if losses[-1] > losses[0] * 1.2:
            return "resolution/high-k sensitivity"
    return "unclear"


def write_norm_loss_isolation_report(path: Path, rows: list[dict[str, Any]], native_profile_note: str) -> None:
    recommendation = infer_norm_loss_mechanism(rows)
    lines = [
        "# Conservative C2 Norm-Loss Isolation",
        "",
        "Standalone CuPy-only diagnostic. No production solver, Hunter, validation, config, or JAX scout reference files were modified.",
        "",
        "## Summary",
        "",
        f"Recommendation: **{recommendation}**.",
        "",
        f"Native C2 profile test: {native_profile_note}",
        "",
        "## Results",
        "",
        "| case | group | N | dt | nonlinear | mask | norm loss | rho max change | high-k final | recomb mask mean | nodes |",
        "|---|---|---:|---:|:---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['group']} | {row['N']} | {row['dt']} | {row['nonlinear_enabled']} | "
            f"{row['mask_mode']} | {fmt(row.get('final_norm_fractional_change_from_projected_t0'))} | "
            f"{fmt(row.get('rho_max_fractional_change'))} | {fmt(row.get('high_k_fraction_final'))} | "
            f"{fmt(row.get('recombination_mask_removed_fraction_mean'))} | {row.get('final_node_count')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- Linear-only cases test the conservative spectral unitary operator with nonlinear RHS disabled.",
            "- `current` keeps both nonlinear-transform and recombination masks.",
            "- `nonlinear_only` and `no_recombination` keep nonlinear-transform masking but disable the recombination mask.",
            "- `none` is diagnostic-only: the dealias mask is replaced with ones, so nonlinear and recombination masks are disabled.",
            "- Recombination-mask removal is measured directly; nonlinear-transform mask removal is inferred by mode contrasts.",
            "- No stability claim is made from these isolation tests.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_norm_loss_isolation(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = norm_loss_isolation_paths(out_dir)
    histories: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    physical_time = float(args.physical_time)

    def triangle(N: int) -> np.ndarray:
        pts = points_for_case("triangle", 0.45, {}, seed=DEFAULT_SEED)
        return build_geometry_ic(N, DEFAULT_L, pts, DEFAULT_WIDTH_BOX, 1.0, phases=[0.0, 0.0, 0.0])

    matrix: list[dict[str, Any]] = []
    for mode in ("current", "none"):
        matrix.append({"group": "linear_only", "case_id": f"linear_{mode}_N48_dt0.001", "N": 48, "dt": 0.001, "nonlinear": False, "mask": mode})
    for mode in ("current", "nonlinear_only", "no_recombination", "none"):
        matrix.append({"group": "mask_isolation", "case_id": f"mask_{mode}_N48_dt0.001", "N": 48, "dt": 0.001, "nonlinear": True, "mask": mode})
    for dt in (0.001, 0.0005, 0.00025):
        matrix.append({"group": "timestep", "case_id": f"dt_{dt:g}_N48", "N": 48, "dt": dt, "nonlinear": True, "mask": "current"})
    for N in (48, 64):
        matrix.append({"group": "resolution", "case_id": f"resolution_N{N}_dt0.001", "N": N, "dt": 0.001, "nonlinear": True, "mask": "current"})

    for spec in matrix:
        row, hist = run_norm_loss_case(
            case_id=spec["case_id"],
            group=spec["group"],
            out_dir=out_dir,
            N=int(spec["N"]),
            dt=float(spec["dt"]),
            physical_time=physical_time,
            nonlinear_enabled=bool(spec["nonlinear"]),
            mask_mode=str(spec["mask"]),
            psi0=triangle(int(spec["N"])),
            sample_count=int(args.sample_count),
        )
        rows.append(row)
        histories.extend(hist)

    native = _find_native_c2_profile()
    native_note = "not run; no existing native C2 soliton NPZ artifact with psi-like field was found"
    if native is not None:
        psi_native = _load_native_profile(native)
        if psi_native.shape[0] in (48, 64, 96):
            row, hist = run_norm_loss_case(
                case_id=f"profile_native_{native.stem}",
                group="profile_mismatch",
                out_dir=out_dir,
                N=int(psi_native.shape[0]),
                dt=0.001,
                physical_time=physical_time,
                nonlinear_enabled=True,
                mask_mode="current",
                psi0=psi_native,
                sample_count=int(args.sample_count),
            )
            row["native_profile_path"] = str(native)
            rows.append(row)
            histories.extend(hist)
            native_note = f"run from existing artifact {native}"
        else:
            native_note = f"not run; native artifact shape {psi_native.shape} is outside supported diagnostic sizes"

    write_rows_csv(paths["csv"], rows)
    write_norm_loss_plots(out_dir, histories, paths)
    write_norm_loss_isolation_report(paths["report"], rows, native_note)
    history_path = out_dir / "conservative_norm_loss_isolation_history.json"
    history_path.write_text(json.dumps(histories, indent=2), encoding="utf-8")
    summary = {
        "rows": rows,
        "history_json": str(history_path),
        "csv": str(paths["csv"]),
        "report": str(paths["report"]),
        "norm_plot": str(paths["norm_plot"]),
        "rho_plot": str(paths["rho_plot"]),
        "recommendation": infer_norm_loss_mechanism(rows),
        "native_profile_note": native_note,
    }
    print(json.dumps(summary, indent=2, default=str))
    return summary


def write_campaign_dashboard_html(path: Path, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    payload = json.dumps({"manifest": manifest, "rows": rows})
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{manifest.get('campaign_id', 'conservative campaign')}</title>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <style>body{{font-family:Arial,sans-serif;margin:18px}} #nodes,#balance{{height:420px}} table{{border-collapse:collapse}} td,th{{border:1px solid #ccc;padding:4px 6px}}</style>
</head>
<body>
  <h1>{manifest.get('campaign_id', 'conservative campaign')}</h1>
  <p>Conservative C2 diagnostic only. Dissipative runs are historical geometry priors only.</p>
  <div id="nodes"></div>
  <div id="balance"></div>
  <table id="summary"></table>
  <script>
  const payload = {payload};
  const rows = payload.rows;
  Plotly.newPlot('nodes', [{{
    type:'bar',
    x: rows.map(r => r.case_id),
    y: rows.map(r => r.final_node_count),
    text: rows.map(r => r.arrangement_outcome),
    name:'final nodes'
  }}], {{title:'Final Node Count', xaxis:{{tickangle:-25}}, yaxis:{{rangemode:'tozero'}}}});
  Plotly.newPlot('balance', [
    {{type:'scatter', mode:'markers+lines', x: rows.map(r => r.case_id), y: rows.map(r => r.mass_cv), name:'mass CV'}},
    {{type:'scatter', mode:'markers+lines', x: rows.map(r => r.case_id), y: rows.map(r => r.peak_cv), name:'peak CV'}}
  ], {{title:'Final Balance Metrics', xaxis:{{tickangle:-25}}, yaxis:{{rangemode:'tozero'}}}});
  const table = document.getElementById('summary');
  table.innerHTML = '<tr><th>case</th><th>template</th><th>outcome</th><th>finite</th><th>nodes</th><th>rho max</th><th>metadata</th></tr>' +
    rows.map(r => `<tr><td>${{r.case_id}}</td><td>${{r.template}}</td><td>${{r.arrangement_outcome}}</td><td>${{r.final_finite}}</td><td>${{r.final_node_count}}</td><td>${{Number(r.rho_max).toPrecision(5)}}</td><td>${{r.metadata}}</td></tr>`).join('');
  </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


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


def write_recommended_manifest(path: Path, N: int = 48, steps: int = 4000) -> dict[str, Any]:
    manifest = {
        "campaign_id": f"conservative_geometry_medium_N{N}_T{steps}",
        "regime": "conservative_c2_only",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime": {"N": int(N), "steps": int(steps), "dt": DEFAULT_DT, "L": DEFAULT_L, "sample_every": max(100, steps // 24)},
        "provenance": {
            "conservative_path_found": True,
            "reference_files": [
                "jax_scout/physics.py",
                "jax_scout/phase_d_c2_transport.py",
                "jax_scout/phase_d_c2_soliton_scout.py",
                "docs/PHASE_D_C2_TRANSPORT_BRANCH_PLAN.md",
                "docs/PHASE_D_C2_NATIVE_SOLITON_RESULTS.md",
                "docs/PHASE_D_C2_2_LOSS_SOURCE_RESULTS.md",
            ],
            "dissipative_reference_only": True,
        },
        "cases": [
            {"template": "triangle", "spacing_box": 0.36, "label": "triangle_transfer_036", "perturbations": {}},
            {"template": "triangle", "spacing_box": 0.45, "label": "triangle_transfer_045", "perturbations": {}},
            {"template": "ablated_triangle", "spacing_box": 0.45, "label": "ablated_triangle_control", "perturbations": {}},
            {"template": "tetrahedron", "spacing_box": 0.45, "label": "tetrahedron_matched_045", "perturbations": {}},
            {"template": "triangular_prism", "spacing_box": 0.45, "label": "triangular_prism_6node_045", "perturbations": {}},
        ],
        "limitations": [
            "Medium-length conservative diagnostic campaign only.",
            "No overnight run is implied by this manifest.",
            "Dissipative triangle outcomes are not treated as conservative evidence.",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run_campaign_manifest(
    manifest_path: Path,
    out_dir: Path,
    force: bool = False,
    require_gpu_name: str | None = "NVIDIA GeForce GTX 1080",
    max_cases: int | None = None,
) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("regime") != "conservative_c2_only":
        raise ValueError("manifest regime must be conservative_c2_only")
    runtime = manifest["runtime"]
    rows = []
    cases = manifest.get("cases", [])
    if max_cases is not None:
        cases = cases[: int(max_cases)]
    resolved_manifest = {
        **manifest,
        "resolved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "resolved_output_dir": str(out_dir),
        "resolved_cases": cases,
        "physical_time": float(runtime["steps"]) * float(runtime["dt"]),
        "caveat": "Conservative C2 diagnostic campaign only; dissipative results are historical geometry priors only.",
    }
    for case in cases:
        meta = run_case(
            out_dir=out_dir,
            template=case["template"],
            spacing_box=float(case["spacing_box"]),
            N=int(runtime["N"]),
            steps=int(runtime["steps"]),
            dt=float(runtime["dt"]),
            L=float(runtime["L"]),
            perturbations=case.get("perturbations", {}),
            sample_every=int(runtime.get("sample_every", max(100, int(runtime["steps"]) // 24))),
            collapse_threshold=float(runtime.get("collapse_threshold", 1e6)),
            require_gpu_name=require_gpu_name,
            force=force,
            suffix=str(case.get("label", "")),
        )
        rows.append(meta["screen_row"])
    paths = campaign_output_paths(out_dir, manifest["campaign_id"])
    audit_paths = invariant_audit_paths(out_dir)
    write_rows_csv(paths["named_csv"], rows)
    write_rows_csv(paths["stable_csv"], rows)
    write_invariant_audit_csv(audit_paths["csv"], rows)
    write_campaign_report(paths["named_report"], rows, resolved_manifest)
    write_campaign_report(paths["stable_report"], rows, resolved_manifest)
    write_invariant_audit_report(audit_paths["report"], rows, resolved_manifest)
    write_campaign_dashboard_html(paths["dashboard"], rows, resolved_manifest)
    paths["resolved_manifest"].write_text(json.dumps(resolved_manifest, indent=2), encoding="utf-8")
    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest": resolved_manifest,
        "rows": rows,
        "results_csv": str(paths["stable_csv"]),
        "report_md": str(paths["stable_report"]),
        "campaign_manifest_resolved": str(paths["resolved_manifest"]),
        "invariant_audit_csv": str(audit_paths["csv"]),
        "invariant_audit_report_md": str(audit_paths["report"]),
        "dashboard_html": str(paths["dashboard"]),
        "named_results_csv": str(paths["named_csv"]),
        "named_report_md": str(paths["named_report"]),
        "caveat": "Conservative C2 diagnostic campaign only.",
    }
    paths["named_summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    return run_case(
        out_dir=args.out,
        template=args.template,
        spacing_box=args.spacing,
        N=args.N,
        steps=args.steps,
        dt=args.dt,
        L=args.L,
        perturbations={},
        sample_every=args.sample_every,
        collapse_threshold=args.collapse_threshold,
        require_gpu_name=args.require_gpu_name,
        force=args.force,
        suffix="smoke",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    smoke = sub.add_parser("smoke", help="Run one tiny conservative CuPy GPU smoke test")
    smoke.add_argument("--out", type=Path, default=DEFAULT_OUT)
    smoke.add_argument("--N", type=int, default=32)
    smoke.add_argument("--steps", type=int, default=50)
    smoke.add_argument("--template", default="triangle")
    smoke.add_argument("--spacing", type=float, default=0.45)
    smoke.add_argument("--dt", type=float, default=DEFAULT_DT)
    smoke.add_argument("--L", type=float, default=DEFAULT_L)
    smoke.add_argument("--sample-every", type=int, default=10)
    smoke.add_argument("--collapse-threshold", type=float, default=1e6)
    smoke.add_argument("--require-gpu-name", default="NVIDIA GeForce GTX 1080")
    smoke.add_argument("--force", action="store_true")
    smoke.set_defaults(func=run_smoke)

    init = sub.add_parser("init-manifest", help="Write the first recommended conservative medium campaign manifest")
    init.add_argument("--out", type=Path, default=DEFAULT_OUT / "first_conservative_campaign_manifest.json")
    init.add_argument("--N", type=int, default=48)
    init.add_argument("--steps", type=int, default=4000)
    init.set_defaults(func=lambda args: write_recommended_manifest(args.out, N=args.N, steps=args.steps))

    campaign = sub.add_parser("campaign", help="Run/resume a conservative-only campaign manifest")
    campaign.add_argument("--manifest", type=Path, required=True)
    campaign.add_argument("--out", type=Path, default=DEFAULT_OUT)
    campaign.add_argument("--force", action="store_true")
    campaign.add_argument("--require-gpu-name", default="NVIDIA GeForce GTX 1080")
    campaign.add_argument("--max-cases", type=int, default=None)
    campaign.set_defaults(
        func=lambda args: run_campaign_manifest(
            args.manifest,
            args.out,
            force=args.force,
            require_gpu_name=args.require_gpu_name,
            max_cases=args.max_cases,
        )
    )

    isolation = sub.add_parser("norm-loss-isolation", help="Run conservative C2 norm-loss isolation diagnostics")
    isolation.add_argument("--out", type=Path, default=DEFAULT_OUT / "norm_loss_isolation")
    isolation.add_argument("--physical-time", type=float, default=4.0)
    isolation.add_argument("--sample-count", type=int, default=40)
    isolation.set_defaults(func=run_norm_loss_isolation)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = args.func(args)
    if result is not None:
        print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
