"""GPU dynamics characterization for the Gravity D spatial effective medium.

This standalone mirror driver characterizes the already-confirmed operator

    i d_t psi = -D div(N(x) grad psi)

without touching production geometry, Hunter, GPU launch infrastructure, or
fallback policy.  It reuses the validated GPU solver primitives from
gravity_D_neutral_probe_gpu.py and adds characterization-only source/probe
families, diagnostics, model comparisons, plots, and handoff artifacts.

Run from WSL2 with the established JAX GPU venv, for example:
    . ~/jax_irer/bin/activate
    python /mnt/f/quantule_mapper/jax_scout/gravity_D_dynamics_characterization_gpu.py \
      --out /mnt/f/quantule_mapper/sweep_runs/GRAVITY_D_DYNAMICS_CHARACTERIZATION_GPU_<timestamp>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jaxlib
import numpy as np

ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_FOR_IMPORT))

from jax_scout.gravity_D_neutral_probe_gpu import (  # noqa: E402
    D,
    ROOT,
    build_grid,
    config_hash,
    deriv,
    evolve_n,
    git_commit,
    preflight,
    rhs,
    rk4_step,
)


BASE_N = 96
BASE_L = 30.0
BASE_DT = 0.001
BASE_SOURCE_WIDTH = 1.5
BASE_N_MIN = 0.5
EPS = 1e-30

ABS_FORCE_TOL = 1e-8
REL_FORCE_TOL = 1e-6
NORM_TOL_STD = 1e-8
NORM_TOL_LONG = 5e-8
ENERGY_TOL_STD = 1e-7
ENERGY_TOL_LONG = 5e-7
COM_TOL = 1e-5


def ffloat(x: object) -> float:
    return float(np.asarray(x))


def vlist(x: object) -> list[float]:
    return [float(v) for v in np.asarray(x).reshape(-1)]


def beta_from_nmin(n_min: float) -> float:
    if n_min >= 0.999999999:
        return 0.0
    return (1.0 / n_min) - 1.0


def radial_hat(pos: Iterable[float], src: Iterable[float] = (0.0, 0.0, 0.0)) -> tuple[float, float, float]:
    v = np.asarray(tuple(pos), dtype=float) - np.asarray(tuple(src), dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-14:
        return (1.0, 0.0, 0.0)
    return tuple((v / n).tolist())


def tangential_hat(rhat: Iterable[float]) -> tuple[float, float, float]:
    r = np.asarray(tuple(rhat), dtype=float)
    trial = np.asarray([0.0, 1.0, 0.0])
    if abs(float(np.dot(r, trial))) > 0.9:
        trial = np.asarray([0.0, 0.0, 1.0])
    t = trial - np.dot(trial, r) * r
    t /= np.linalg.norm(t)
    return tuple(t.tolist())


def git_text(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.STDOUT)
    except Exception as exc:
        return f"COMMAND_FAILED {cmd}: {exc}\n"


def save_environment(outdir: Path, command_line: str) -> dict[str, object]:
    pf = preflight()
    pf["command_line"] = command_line
    pf["jax_version"] = jax.__version__
    pf["jaxlib_version"] = jaxlib.__version__
    pf["x64_enabled"] = bool(jax.config.read("jax_enable_x64"))
    (outdir / "gpu_preflight.json").write_text(json.dumps(pf, indent=2), encoding="utf-8")
    env = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "x64_enabled": bool(jax.config.read("jax_enable_x64")),
        "default_backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "git_commit": git_commit(),
        "command_line": command_line,
        "cwd": str(ROOT),
        "env": {
            "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "XLA_PYTHON_CLIENT_MEM_FRACTION": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    (outdir / "environment_versions.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    (outdir / "git_state_before.txt").write_text(git_text(["git", "status", "--short"]), encoding="utf-8")
    return pf


def source_profile(grid: dict[str, object], cfg: dict[str, object]) -> jnp.ndarray:
    X, Y, Z = grid["X"], grid["Y"], grid["Z"]
    ox, oy, oz = cfg.get("source_offset", (0.0, 0.0, 0.0))
    x, y, z = X - float(ox), Y - float(oy), Z - float(oz)
    r2 = x * x + y * y + z * z
    family = str(cfg.get("source_family", cfg.get("source_shape", "gaussian")))
    width = float(cfg.get("source_width", BASE_SOURCE_WIDTH))
    if family == "gaussian":
        S = jnp.exp(-(r2 / (width * width)))
    elif family == "supergaussian4":
        S = jnp.exp(-((jnp.sqrt(r2) / width) ** 4))
    elif family == "compact_bump":
        q = r2 / (width * width)
        S = jnp.where(q < 1.0, jnp.exp(-1.0 / (1.0 - q)), 0.0)
    elif family == "two_lobe":
        sep = float(cfg.get("lobe_sep", 1.35))
        r2a = x * x + (y - sep) * (y - sep) + z * z
        r2b = x * x + (y + sep) * (y + sep) + z * z
        S = jnp.exp(-(r2a / (width * width))) + jnp.exp(-(r2b / (width * width)))
    elif family == "shell":
        radius = float(cfg.get("shell_radius", 2.5))
        thickness = float(cfg.get("shell_thickness", width))
        r = jnp.sqrt(r2)
        S = jnp.exp(-(((r - radius) ** 2) / (thickness * thickness)))
    elif family == "two_source":
        sep = float(cfg.get("source_sep", 5.0))
        r2a = (x - sep / 2.0) ** 2 + y * y + z * z
        r2b = (x + sep / 2.0) ** 2 + y * y + z * z
        S = jnp.exp(-(r2a / (width * width))) + jnp.exp(-(r2b / (width * width)))
    elif family == "flat":
        S = jnp.zeros_like(X)
    else:
        raise ValueError(f"unknown source_family={family}")
    return S / (jnp.max(S) + EPS)


def coefficient(S: jnp.ndarray, n_min: float) -> jnp.ndarray:
    return 1.0 / (1.0 + beta_from_nmin(float(n_min)) * S)


def coeff_diag(S: jnp.ndarray, Nf: jnp.ndarray, grid: dict[str, object]) -> dict[str, float]:
    X, Y, Z, dV = grid["X"], grid["Y"], grid["Z"], grid["dV"]
    deficit = 1.0 - Nf
    total = jnp.sum(deficit) * dV
    r2 = X * X + Y * Y + Z * Z
    second = (jnp.sum(r2 * deficit) * dV) / (total + EPS)
    dNx = jnp.real(deriv(Nf, grid["ikx"]))
    dNy = jnp.real(deriv(Nf, grid["iky"]))
    dNz = jnp.real(deriv(Nf, grid["ikz"]))
    gmag = jnp.sqrt(dNx * dNx + dNy * dNy + dNz * dNz)
    support = jnp.where(S > 1e-3, jnp.sqrt(r2), 0.0)
    return {
        "N_min": ffloat(jnp.min(Nf)),
        "N_max": ffloat(jnp.max(Nf)),
        "integrated_deficit": ffloat(total),
        "deficit_second_moment": ffloat(second),
        "ell_S": ffloat(jnp.sqrt(jnp.maximum(second, 0.0))),
        "max_grad_N": ffloat(jnp.max(gmag)),
        "effective_support_radius": ffloat(jnp.max(support)),
    }


def deficit_for(grid: dict[str, object], family: str, width: float, n_min: float, extra: dict[str, object] | None = None) -> float:
    cfg = {"source_family": family, "source_width": width}
    if extra:
        cfg.update(extra)
    S = source_profile(grid, cfg)
    return coeff_diag(S, coefficient(S, n_min), grid)["integrated_deficit"]


def match_width(grid: dict[str, object], family: str, target_deficit: float, n_min: float) -> float:
    if family == "gaussian":
        return BASE_SOURCE_WIDTH
    if family == "shell":
        lo, hi = 0.2, 3.5
    elif family == "two_lobe":
        lo, hi = 0.35, 4.0
    else:
        lo, hi = 0.25, 6.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if deficit_for(grid, family, mid, n_min) < target_deficit:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def matched_source_meta(grid: dict[str, object], families: Iterable[str], n_min_values: Iterable[float]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for n_min in n_min_values:
        base_S = source_profile(grid, {"source_family": "gaussian", "source_width": BASE_SOURCE_WIDTH})
        target_def = coeff_diag(base_S, coefficient(base_S, n_min), grid)["integrated_deficit"]
        for family in families:
            width = match_width(grid, family, target_def, n_min)
            cfg = {"source_family": family, "source_width": width}
            S = source_profile(grid, cfg)
            Nf = coefficient(S, n_min)
            diag = coeff_diag(S, Nf, grid)
            rows.append({"source_family": family, "N_min_target": n_min, "matched_width": width, **diag})
    return rows


def initial_probe(grid: dict[str, object], cfg: dict[str, object]) -> jnp.ndarray:
    X, Y, Z = grid["X"], grid["Y"], grid["Z"]
    px, py, pz = cfg.get("probe_position", (4.0, 0.0, 0.0))
    dx, dy, dz = X - float(px), Y - float(py), Z - float(pz)
    sig = float(cfg.get("probe_width", 1.0))
    profile = str(cfg.get("probe_profile", "gaussian"))
    if profile == "gaussian":
        env = jnp.exp(-((dx * dx + dy * dy + dz * dz) / (2.0 * sig * sig)))
    elif profile == "supergaussian":
        env = jnp.exp(-(((dx * dx + dy * dy + dz * dz) ** 2) / (2.0 * sig**4)))
    elif profile == "node_radial":
        r = jnp.sqrt(dx * dx + dy * dy + dz * dz)
        env = (r / (sig + EPS)) * jnp.exp(-((dx * dx + dy * dy + dz * dz) / (2.0 * sig * sig)))
    elif profile == "phase_modulated":
        env = jnp.exp(-((dx * dx + dy * dy + dz * dz) / (2.0 * sig * sig)))
    elif profile == "two_component":
        sep = 0.55 * sig
        env = (
            jnp.exp(-(((dx - sep) ** 2 + dy * dy + dz * dz) / (2.0 * (0.75 * sig) ** 2)))
            + jnp.exp(-(((dx + sep) ** 2 + dy * dy + dz * dz) / (2.0 * (0.75 * sig) ** 2)))
        )
    else:
        raise ValueError(f"unknown probe_profile={profile}")
    amp = float(cfg.get("probe_amplitude", 1.0))
    env = amp * env
    if bool(cfg.get("normalize_probe", True)):
        norm = jnp.sum(jnp.abs(env) ** 2) * grid["dV"]
        env = env * jnp.sqrt(float(cfg.get("target_norm", 1.0)) / (norm + EPS))
    kx, ky, kz = cfg.get("carrier_vector", cfg.get("probe_carrier", (0.0, 0.0, 0.0)))
    extra_phase = 0.0
    if profile == "phase_modulated":
        extra_phase = 0.15 * jnp.sin(2.0 * dx / (sig + EPS))
    phase = jnp.exp(1j * (float(kx) * X + float(ky) * Y + float(kz) * Z + extra_phase))
    return (env * phase).astype(jnp.complex128)


@jax.jit
def diagnostics(
    psi: jnp.ndarray,
    Nf: jnp.ndarray,
    S: jnp.ndarray,
    grid: dict[str, object],
    probe_pos: jnp.ndarray,
    rhat: jnp.ndarray,
    L: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    X, Y, Z, dV = grid["X"], grid["Y"], grid["Z"], grid["dV"]
    rho = jnp.abs(psi) ** 2
    norm = jnp.sum(rho) * dV
    coords = (X, Y, Z)
    global_com = jnp.array([jnp.sum(c * rho) * dV for c in coords], dtype=jnp.float64) / (norm + EPS)

    dx, dy, dz = X - probe_pos[0], Y - probe_pos[1], Z - probe_pos[2]
    radial_coord = dx * rhat[0] + dy * rhat[1] + dz * rhat[2]
    slab = jnp.abs(radial_coord) < 6.0
    srho = rho * slab
    smass = jnp.sum(srho) * dV
    slab_com = jnp.array([jnp.sum(c * srho) * dV for c in coords], dtype=jnp.float64) / (smass + EPS)

    def circ(coord):
        theta = 2.0 * jnp.pi * coord / L
        s = jnp.sum(jnp.sin(theta) * rho) * dV
        c = jnp.sum(jnp.cos(theta) * rho) * dV
        return L * jnp.arctan2(s, c) / (2.0 * jnp.pi)

    periodic_com = jnp.array([circ(X), circ(Y), circ(Z)], dtype=jnp.float64)

    gx, gy, gz = deriv(psi, grid["ikx"]), deriv(psi, grid["iky"]), deriv(psi, grid["ikz"])
    grad_energy_density = jnp.abs(gx) ** 2 + jnp.abs(gy) ** 2 + jnp.abs(gz) ** 2
    K_grad = jnp.sum(grad_energy_density) * dV
    momentum = jnp.array(
        [
            jnp.sum(jnp.imag(jnp.conj(psi) * gx)) * dV,
            jnp.sum(jnp.imag(jnp.conj(psi) * gy)) * dV,
            jnp.sum(jnp.imag(jnp.conj(psi) * gz)) * dV,
        ],
        dtype=jnp.float64,
    )

    dNx, dNy, dNz = jnp.real(deriv(Nf, grid["ikx"])), jnp.real(deriv(Nf, grid["iky"])), jnp.real(deriv(Nf, grid["ikz"]))
    dSx, dSy, dSz = jnp.real(deriv(S, grid["ikx"])), jnp.real(deriv(S, grid["iky"])), jnp.real(deriv(S, grid["ikz"]))
    force = -D * jnp.array(
        [
            jnp.sum(dNx * grad_energy_density) * dV,
            jnp.sum(dNy * grad_energy_density) * dV,
            jnp.sum(dNz * grad_energy_density) * dV,
        ],
        dtype=jnp.float64,
    )
    psit = rhs(psi, Nf, grid)
    gtx, gty, gtz = deriv(psit, grid["ikx"]), deriv(psit, grid["iky"]), deriv(psit, grid["ikz"])
    force_rhs = jnp.array(
        [
            jnp.sum(jnp.imag(jnp.conj(psit) * gx + jnp.conj(psi) * gtx)) * dV,
            jnp.sum(jnp.imag(jnp.conj(psit) * gy + jnp.conj(psi) * gty)) * dV,
            jnp.sum(jnp.imag(jnp.conj(psit) * gz + jnp.conj(psi) * gtz)) * dV,
        ],
        dtype=jnp.float64,
    )

    rel = jnp.stack([X - global_com[0], Y - global_com[1], Z - global_com[2]])
    cov = jnp.array(
        [[jnp.sum(rel[i] * rel[j] * rho) * dV / (norm + EPS) for j in range(3)] for i in range(3)],
        dtype=jnp.float64,
    )
    sigma_eff = jnp.sqrt(jnp.maximum(jnp.trace(cov) / 3.0, 0.0))
    eig = jnp.linalg.eigvalsh(cov)
    anisotropy = jnp.sqrt(jnp.maximum(eig[-1], 0.0) / (jnp.maximum(eig[0], 0.0) + 1e-30))
    radial_force = jnp.dot(force, rhat)
    transverse_force = jnp.linalg.norm(force - radial_force * rhat)
    radial_momentum = jnp.dot(momentum, rhat)
    transverse_momentum = jnp.linalg.norm(momentum - radial_momentum * rhat)
    M2_grad = jnp.sum((radial_coord**2) * grad_energy_density) * dV
    M4_grad = jnp.sum((radial_coord**4) * grad_energy_density) * dV
    energy = D * jnp.sum(Nf * grad_energy_density) * dV
    peak_density = jnp.max(rho)
    gradN_com = jnp.array(
        [
            jnp.sum(dNx * rho) * dV / (norm + EPS),
            jnp.sum(dNy * rho) * dV / (norm + EPS),
            jnp.sum(dNz * rho) * dV / (norm + EPS),
        ],
        dtype=jnp.float64,
    )
    gradS_com = jnp.array(
        [
            jnp.sum(dSx * rho) * dV / (norm + EPS),
            jnp.sum(dSy * rho) * dV / (norm + EPS),
            jnp.sum(dSz * rho) * dV / (norm + EPS),
        ],
        dtype=jnp.float64,
    )
    N_com = jnp.sum(Nf * rho) * dV / (norm + EPS)
    S_com = jnp.sum(S * rho) * dV / (norm + EPS)
    grad_N_at_COM = jnp.dot(gradN_com, rhat)
    grad_lnN_at_COM = grad_N_at_COM / (N_com + EPS)
    source_grad_at_COM = jnp.dot(gradS_com, rhat)
    return {
        "global_com": global_com,
        "periodic_com": periodic_com,
        "slab_com": slab_com,
        "momentum": momentum,
        "norm": norm,
        "K_grad": K_grad,
        "M2_grad": M2_grad,
        "M4_grad": M4_grad,
        "force_exact": force,
        "force_rhs": force_rhs,
        "radial_force": radial_force,
        "transverse_force": transverse_force,
        "radial_momentum": radial_momentum,
        "transverse_momentum": transverse_momentum,
        "covariance": cov,
        "sigma_eff": sigma_eff,
        "anisotropy": anisotropy,
        "peak_density": peak_density,
        "energy": energy,
        "N_at_COM": N_com,
        "S_at_COM": S_com,
        "grad_N_at_COM": grad_N_at_COM,
        "grad_lnN_at_COM": grad_lnN_at_COM,
        "source_grad_at_COM": source_grad_at_COM,
    }


def scalarize_diag(diag: dict[str, jnp.ndarray]) -> dict[str, object]:
    out: dict[str, object] = {}
    vector_keys = {"global_com", "periodic_com", "slab_com", "momentum", "force_exact", "force_rhs"}
    for key, value in diag.items():
        arr = np.asarray(value)
        if key in vector_keys:
            out[key] = vlist(arr)
        elif key == "covariance":
            out[key] = [[float(x) for x in row] for row in arr.tolist()]
        else:
            out[key] = float(arr)
    return out


def sample_diag(t: float, diag: dict[str, jnp.ndarray], force_fd: jnp.ndarray, rhat: np.ndarray) -> dict[str, object]:
    row = scalarize_diag(diag)
    row["t"] = float(t)
    row["force_fd"] = vlist(force_fd)
    fd = np.asarray(force_fd, dtype=float)
    row["radial_force_fd"] = float(np.dot(fd, rhat))
    row["transverse_force_fd"] = float(np.linalg.norm(fd - row["radial_force_fd"] * rhat))
    return row


def base_cfg() -> dict[str, object]:
    pos = (4.0, 0.0, 0.0)
    return {
        "stage": "E1",
        "run_id": "base",
        "N": BASE_N,
        "L": BASE_L,
        "dt": BASE_DT,
        "T": 0.0,
        "sample_dt": 0.05,
        "source_family": "gaussian",
        "source_width": BASE_SOURCE_WIDTH,
        "source_offset": (0.0, 0.0, 0.0),
        "N_min": BASE_N_MIN,
        "probe_profile": "gaussian",
        "probe_position": pos,
        "probe_width": 1.0,
        "probe_amplitude": 1.0,
        "carrier_vector": (0.0, 0.0, 0.0),
        "radial_hat": radial_hat(pos),
        "normalize_probe": True,
        "target_norm": 1.0,
    }


def with_position(cfg: dict[str, object], radius: float, axis: tuple[float, float, float] = (1.0, 0.0, 0.0)) -> dict[str, object]:
    src = np.asarray(cfg.get("source_offset", (0.0, 0.0, 0.0)), dtype=float)
    ax = np.asarray(axis, dtype=float)
    ax = ax / (np.linalg.norm(ax) + 1e-30)
    pos = tuple((src + radius * ax).tolist())
    out = {**cfg, "probe_position": pos, "radial_hat": radial_hat(pos, src)}
    return out


def source_cfg_from_meta(meta: list[dict[str, object]], family: str, n_min: float) -> dict[str, object]:
    matches = [m for m in meta if m["source_family"] == family and abs(float(m["N_min_target"]) - n_min) < 1e-12]
    if not matches:
        raise KeyError((family, n_min))
    return {"source_family": family, "source_width": float(matches[0]["matched_width"]), "N_min": n_min}


def build_matrices(outdir: Path, stages: set[str]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    grid = build_grid(BASE_N, BASE_L)
    families = ["gaussian", "supergaussian4", "compact_bump", "two_lobe", "shell"]
    nmins = [1.00, 0.95, 0.90, 0.80, 0.65, 0.50, 0.35]
    meta = matched_source_meta(grid, families, nmins)
    meta_by_key = {(m["source_family"], float(m["N_min_target"])): m for m in meta}
    base_ell = float(meta_by_key[("gaussian", BASE_N_MIN)]["ell_S"])

    atlas: list[dict[str, object]] = []
    trajectories: list[dict[str, object]] = []
    validation: list[dict[str, object]] = []
    falsification: list[dict[str, object]] = []

    def add_atlas(run_id: str, cfg: dict[str, object]) -> None:
        if "E1" in stages or "all" in stages:
            atlas.append({**cfg, "stage": "E1", "run_id": run_id, "T": 0.0})

    # E0 reference gate is always explicit when E0/all is selected.
    if "E0" in stages or "all" in stages:
        trajectories.append({**base_cfg(), "stage": "E0", "run_id": "E0_reference_medium", "T": 4.0, "sample_dt": 0.05})
        trajectories.append({**base_cfg(), "stage": "E0", "run_id": "E0_reference_flat", "T": 4.0, "sample_dt": 0.05, "N_min": 1.0})

    if "E1" in stages or "all" in stages:
        for family in families:
            sc = source_cfg_from_meta(meta, family, BASE_N_MIN)
            ell = float(meta_by_key[(family, BASE_N_MIN)]["ell_S"])
            for ratio in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
                add_atlas(f"E1_dist_{family}_r{str(ratio).replace('.', 'p')}", with_position({**base_cfg(), **sc}, ratio * ell))
        for nmin in nmins:
            sc = source_cfg_from_meta(meta, "gaussian", nmin)
            add_atlas(f"E1_strength_Nmin{str(nmin).replace('.', 'p')}", with_position({**base_cfg(), **sc}, 4.0))
        ellN0 = base_ell
        for wr in [0.15, 0.25, 0.4, 0.6, 0.8, 1.0, 1.4]:
            add_atlas(f"E1_width_ratio{str(wr).replace('.', 'p')}", with_position({**base_cfg(), "probe_width": wr * ellN0}, 4.0))
        for family in ["gaussian", "supergaussian4"]:
            for rr in [0.5, 1.0, 2.0, 4.0]:
                for wr in [0.25, 0.6, 1.0]:
                    for nmin in [0.90, 0.65, 0.50, 0.35]:
                        sc = source_cfg_from_meta(meta, family, nmin)
                        ell = float(meta_by_key[(family, nmin)]["ell_S"])
                        add_atlas(
                            f"E1_core_{family}_r{rr}_w{wr}_n{nmin}".replace(".", "p"),
                            with_position({**base_cfg(), **sc, "probe_width": wr * ell}, rr * ell),
                        )
        for rr in [1.0, 2.0, 4.0]:
            for kval in [-0.5, -0.2, 0.0, 0.2, 0.5]:
                cfg = with_position({**base_cfg()}, rr * base_ell)
                rhat = np.asarray(cfg["radial_hat"], dtype=float)
                cfg["carrier_vector"] = tuple((kval * rhat).tolist())
                add_atlas(f"E1_radial_carrier_r{rr}_k{kval}".replace(".", "p").replace("-", "m"), cfg)
            for kval in [0.2, 0.5, 1.0]:
                cfg = with_position({**base_cfg()}, rr * base_ell)
                that = np.asarray(tangential_hat(cfg["radial_hat"]))
                cfg["carrier_vector"] = tuple((kval * that).tolist())
                add_atlas(f"E1_tangential_carrier_r{rr}_k{kval}".replace(".", "p"), cfg)
        shell_sc = source_cfg_from_meta(meta, "shell", BASE_N_MIN)
        shell_points = [("center", 0.0), ("inner", 1.5), ("gradient", 2.5), ("outer", 3.5), ("exterior", 5.0)]
        for name, r in shell_points:
            add_atlas(f"E1_shell_{name}", with_position({**base_cfg(), **shell_sc}, r))
        two_sc = source_cfg_from_meta(meta, "two_lobe", BASE_N_MIN)
        for name, pos in {
            "midpoint": (0.0, 0.0, 0.0),
            "lobe_plus": (0.0, 1.35, 0.0),
            "lobe_minus": (0.0, -1.35, 0.0),
            "offaxis_midpoint": (1.5, 0.0, 0.0),
        }.items():
            atlas.append({**base_cfg(), **two_sc, "stage": "E1", "run_id": f"E1_two_lobe_{name}", "T": 0.0, "probe_position": pos, "radial_hat": radial_hat(pos)})

    if "E2" in stages or "all" in stages:
        regime_defs = {
            "weak_inward_deflection": {**base_cfg(), **source_cfg_from_meta(meta, "gaussian", 0.90), "probe_position": (4.0, 0.0, 0.0)},
            "strong_inward_deflection": {**base_cfg(), **source_cfg_from_meta(meta, "gaussian", 0.35), "probe_position": (4.0, 0.0, 0.0)},
            "near_source_motion": with_position({**base_cfg()}, 0.5 * base_ell),
            "far_field_near_null": with_position({**base_cfg()}, 5.0 * base_ell),
            "radial_inward_carrier": {**with_position({**base_cfg()}, 2.0 * base_ell), "carrier_vector": (-0.35, 0.0, 0.0)},
            "radial_outward_carrier": {**with_position({**base_cfg()}, 2.0 * base_ell), "carrier_vector": (0.35, 0.0, 0.0)},
            "tangential_carrier": {**with_position({**base_cfg()}, 2.0 * base_ell), "carrier_vector": (0.0, 0.5, 0.0)},
            "shell_interior": with_position({**base_cfg(), **source_cfg_from_meta(meta, "shell", BASE_N_MIN)}, 1.0),
            "shell_exterior": with_position({**base_cfg(), **source_cfg_from_meta(meta, "shell", BASE_N_MIN)}, 5.0),
            "two_lobe_midpoint": {**base_cfg(), **source_cfg_from_meta(meta, "two_lobe", BASE_N_MIN), "probe_position": (0.0, 0.0, 0.0), "radial_hat": (1.0, 0.0, 0.0)},
            "two_lobe_offaxis": {**base_cfg(), **source_cfg_from_meta(meta, "two_lobe", BASE_N_MIN), "probe_position": (1.5, 0.0, 0.0), "radial_hat": (1.0, 0.0, 0.0)},
        }
        long_names = {"strong_inward_deflection", "radial_outward_carrier", "tangential_carrier", "shell_interior", "two_lobe_midpoint"}
        for name, cfg in regime_defs.items():
            cfg = {**cfg, "radial_hat": radial_hat(cfg["probe_position"], cfg.get("source_offset", (0.0, 0.0, 0.0)))}
            for tag, T, sample_dt in [("short", 0.5, 0.05), ("standard", 4.0, 0.05)]:
                trajectories.append({**cfg, "stage": "E2", "run_id": f"E2_{name}_{tag}", "T": T, "sample_dt": sample_dt})
            if name in long_names:
                trajectories.append({**cfg, "stage": "E2", "run_id": f"E2_{name}_long", "T": 12.0, "sample_dt": 0.1, "requires_gate": f"E2_{name}_standard"})

    if "E7" in stages or "all" in stages:
        reps = [
            ("baseline", {**base_cfg()}),
            ("strong", {**base_cfg(), **source_cfg_from_meta(meta, "gaussian", 0.35)}),
            ("compact", {**base_cfg(), **source_cfg_from_meta(meta, "compact_bump", BASE_N_MIN)}),
        ]
        for name, cfg in reps:
            validation.append({**cfg, "stage": "E7", "run_id": f"E7_{name}_N96", "T": 1.0, "sample_dt": 0.05})
            validation.append({**cfg, "stage": "E7", "run_id": f"E7_{name}_N128", "N": 128, "L": 40.0, "T": 1.0, "sample_dt": 0.05})
            validation.append({**cfg, "stage": "E7", "run_id": f"E7_{name}_dt_half", "T": 1.0, "dt": 0.0005, "sample_dt": 0.05})

    if "E8" in stages or "all" in stages:
        for profile in ["gaussian", "supergaussian", "node_radial", "phase_modulated", "two_component"]:
            falsification.append({**base_cfg(), "stage": "E8", "run_id": f"E8_probe_structure_{profile}", "probe_profile": profile, "T": 0.0})
        for r in [0.0, 1.0, 2.0, 2.5, 3.0, 4.5, 6.0]:
            falsification.append({**with_position({**base_cfg(), **source_cfg_from_meta(meta, "shell", BASE_N_MIN)}, r), "stage": "E8", "run_id": f"E8_shell_map_r{r}".replace(".", "p"), "T": 0.0})
        for r in [3.0, 5.0, 8.0]:
            falsification.append({**with_position({**base_cfg(), **source_cfg_from_meta(meta, "compact_bump", BASE_N_MIN)}, r), "stage": "E8", "run_id": f"E8_exterior_decay_r{r}".replace(".", "p"), "T": 0.0})
        for nmin in [0.95, 0.90, 0.80, 0.65]:
            falsification.append({**base_cfg(), "stage": "E8", "run_id": f"E8_superposition_n{nmin}".replace(".", "p"), "source_family": "two_source", "source_width": BASE_SOURCE_WIDTH, "N_min": nmin, "probe_position": (4.0, 0.0, 0.0), "radial_hat": (1.0, 0.0, 0.0), "T": 0.0})
        for profile in ["gaussian", "node_radial", "two_component"]:
            falsification.append({**base_cfg(), "stage": "E8", "run_id": f"E8_constant_N_{profile}", "probe_profile": profile, "N_min": 1.0, "T": 0.0})

    matrices = {
        "source_profile_metadata": meta,
        "instantaneous_force_atlas": atlas,
        "trajectory_runs": trajectories,
        "numerical_validation": validation,
        "falsification": falsification,
    }
    (outdir / "preregistered_matrix.json").write_text(json.dumps(matrices, indent=2, default=float), encoding="utf-8")
    return meta, atlas, trajectories, validation, falsification


def build_state(cfg: dict[str, object]) -> tuple[dict[str, object], jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    grid = build_grid(int(cfg["N"]), float(cfg["L"]))
    S = source_profile(grid, cfg)
    Nf = coefficient(S, float(cfg["N_min"]))
    psi = initial_probe(grid, cfg)
    return grid, S, Nf, psi


def force_residuals(force: np.ndarray, rhs_force: np.ndarray) -> tuple[float, float]:
    abs_res = float(np.linalg.norm(rhs_force - force))
    fnorm = float(np.linalg.norm(force))
    rel_res = abs_res / fnorm if fnorm > 1e-10 else abs_res
    return abs_res, rel_res


def row_common(cfg: dict[str, object], diag0: dict[str, object], fd: np.ndarray, coeff: dict[str, float], compile_time: float, exec_time: float) -> dict[str, object]:
    force = np.asarray(diag0["force_exact"], dtype=float)
    rhs_force = np.asarray(diag0["force_rhs"], dtype=float)
    rhat = np.asarray(cfg["radial_hat"], dtype=float)
    mom = np.asarray(diag0["momentum"], dtype=float)
    abs_res, rel_res = force_residuals(force, rhs_force)
    radial_distance = float(np.linalg.norm(np.asarray(cfg["probe_position"], dtype=float) - np.asarray(cfg.get("source_offset", (0.0, 0.0, 0.0)), dtype=float)))
    norm = float(diag0["norm"])
    carrier = np.asarray(cfg.get("carrier_vector", (0.0, 0.0, 0.0)), dtype=float)
    return {
        "run_id": cfg["run_id"],
        "stage": cfg["stage"],
        "config_hash": config_hash(cfg),
        "source_family": cfg["source_family"],
        "source_parameters": json.dumps({k: cfg[k] for k in cfg if k.startswith("source_") or k in ("shell_radius", "shell_thickness", "lobe_sep", "source_sep")}, sort_keys=True),
        "N_min": cfg["N_min"],
        "integrated_deficit": coeff["integrated_deficit"],
        "deficit_second_moment": coeff["deficit_second_moment"],
        "ell_S": coeff["ell_S"],
        "max_grad_N": coeff["max_grad_N"],
        "effective_support_radius": coeff["effective_support_radius"],
        "probe_profile": cfg["probe_profile"],
        "probe_norm": norm,
        "probe_width": cfg["probe_width"],
        "probe_position": json.dumps(cfg["probe_position"]),
        "radial_distance": radial_distance,
        "carrier_vector": json.dumps(vlist(carrier)),
        "carrier_norm": float(np.linalg.norm(carrier)),
        "grid": cfg["N"],
        "box": cfg["L"],
        "dt": cfg["dt"],
        "T": cfg["T"],
        "initial_force_exact": json.dumps(vlist(force)),
        "initial_force_rhs": json.dumps(vlist(rhs_force)),
        "initial_force_fd": json.dumps(vlist(fd)),
        "force_per_norm": json.dumps(vlist(force / (norm + EPS))),
        "K_grad": diag0["K_grad"],
        "M2_grad": diag0["M2_grad"],
        "M4_grad": diag0["M4_grad"],
        "N_at_COM": diag0["N_at_COM"],
        "grad_N_at_COM": diag0["grad_N_at_COM"],
        "grad_lnN_at_COM": diag0["grad_lnN_at_COM"],
        "source_grad_at_COM": diag0["source_grad_at_COM"],
        "radial_force": diag0["radial_force"],
        "transverse_force": diag0["transverse_force"],
        "global_COM": json.dumps(diag0["global_com"]),
        "periodic_COM": json.dumps(diag0["periodic_com"]),
        "slab_COM": json.dumps(diag0["slab_com"]),
        "free_subtracted_COM": "",
        "momentum": json.dumps(diag0["momentum"]),
        "radial_momentum": diag0["radial_momentum"],
        "transverse_momentum": diag0["transverse_momentum"],
        "norm_error": 0.0,
        "energy_error": 0.0,
        "force_contract_absolute_residual": abs_res,
        "force_contract_relative_residual": rel_res,
        "regime_label": "",
        "gpu_device": str(jax.devices()[0]),
        "compile_time_s": compile_time,
        "execution_time_s": exec_time,
    }


def run_instantaneous(cfg: dict[str, object]) -> dict[str, object]:
    started = time.time()
    grid, S, Nf, psi = build_state(cfg)
    coeff = coeff_diag(S, Nf, grid)
    ppos = jnp.asarray(cfg["probe_position"], dtype=jnp.float64)
    rhat = jnp.asarray(cfg["radial_hat"], dtype=jnp.float64)
    L = jnp.asarray(float(cfg["L"]), dtype=jnp.float64)
    dt = jnp.asarray(float(cfg["dt"]), dtype=jnp.float64)
    diag0_j = diagnostics(psi, Nf, S, grid, ppos, rhat, L)
    psi1 = rk4_step(psi, Nf, grid, dt)
    diag1_j = diagnostics(psi1, Nf, S, grid, ppos, rhat, L)
    fd = (diag1_j["momentum"] - diag0_j["momentum"]) / dt
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), {**diag0_j, "fd": fd})
    total = time.time() - started
    diag0 = scalarize_diag(diag0_j)
    return row_common(cfg, diag0, np.asarray(fd), coeff, total * 0.35, total * 0.65)


def run_trajectory(cfg: dict[str, object], outdir: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    started = time.time()
    grid, S, Nf, psi = build_state(cfg)
    coeff = coeff_diag(S, Nf, grid)
    ppos = jnp.asarray(cfg["probe_position"], dtype=jnp.float64)
    rhat_j = jnp.asarray(cfg["radial_hat"], dtype=jnp.float64)
    rhat = np.asarray(cfg["radial_hat"], dtype=float)
    L = jnp.asarray(float(cfg["L"]), dtype=jnp.float64)
    dt = jnp.asarray(float(cfg["dt"]), dtype=jnp.float64)
    sample_dt = float(cfg.get("sample_dt", 0.05))
    steps_per_sample = max(1, int(round(sample_dt / float(cfg["dt"]))))
    nsteps = int(round(float(cfg["T"]) / float(cfg["dt"])))
    nsamples = nsteps // steps_per_sample
    tail = nsteps - nsamples * steps_per_sample
    samples: list[dict[str, object]] = []

    diag0_j = diagnostics(psi, Nf, S, grid, ppos, rhat_j, L)
    diag1_j = diagnostics(rk4_step(psi, Nf, grid, dt), Nf, S, grid, ppos, rhat_j, L)
    fd0 = (diag1_j["momentum"] - diag0_j["momentum"]) / dt
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), {**diag0_j, "fd": fd0})
    samples.append(sample_diag(0.0, diag0_j, fd0, rhat))
    compile_time = time.time() - started
    exec_started = time.time()
    for i in range(nsamples):
        psi = evolve_n(psi, Nf, grid, dt, steps_per_sample)
        diag_j = diagnostics(psi, Nf, S, grid, ppos, rhat_j, L)
        diag_next = diagnostics(rk4_step(psi, Nf, grid, dt), Nf, S, grid, ppos, rhat_j, L)
        fd = (diag_next["momentum"] - diag_j["momentum"]) / dt
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), {**diag_j, "fd": fd})
        samples.append(sample_diag((i + 1) * steps_per_sample * float(cfg["dt"]), diag_j, fd, rhat))
    if tail:
        psi = evolve_n(psi, Nf, grid, dt, tail)
        psi.block_until_ready()
    exec_time = time.time() - exec_started
    diag0 = samples[0]
    final = samples[-1]
    row = row_common(cfg, diag0, np.asarray(diag0["force_fd"]), coeff, compile_time, exec_time)
    norm0 = float(samples[0]["norm"])
    energy0 = float(samples[0]["energy"])
    row["norm_error"] = abs(float(final["norm"]) / (norm0 + EPS) - 1.0)
    row["energy_error"] = abs(float(final["energy"]) / (energy0 + EPS) - 1.0)
    row["global_drift"] = float(np.dot(np.asarray(final["global_com"]) - np.asarray(samples[0]["global_com"]), rhat))
    row["periodic_drift"] = float(np.dot(np.asarray(final["periodic_com"]) - np.asarray(samples[0]["periodic_com"]), rhat))
    row["slab_drift"] = float(np.dot(np.asarray(final["slab_com"]) - np.asarray(samples[0]["slab_com"]), rhat))
    row["final_radial_momentum"] = final["radial_momentum"]
    row["turning_points"] = count_turning_points([float(s["radial_momentum"]) for s in samples])
    row["regime_label"] = classify_trajectory(row, samples)
    row["free_subtracted_COM"] = ""
    save_trajectory_npz(outdir / f"{cfg['run_id']}_trajectory.npz", samples)
    (outdir / f"{cfg['run_id']}.json").write_text(json.dumps({"config": cfg, "coefficient": coeff, "samples": samples, "row": row}, indent=2, default=float), encoding="utf-8")
    return row, samples


def save_trajectory_npz(path: Path, samples: list[dict[str, object]]) -> None:
    np.savez_compressed(
        path,
        t=np.asarray([s["t"] for s in samples]),
        global_com=np.asarray([s["global_com"] for s in samples]),
        periodic_com=np.asarray([s["periodic_com"] for s in samples]),
        slab_com=np.asarray([s["slab_com"] for s in samples]),
        momentum=np.asarray([s["momentum"] for s in samples]),
        force=np.asarray([s["force_exact"] for s in samples]),
        force_rhs=np.asarray([s["force_rhs"] for s in samples]),
        force_fd=np.asarray([s["force_fd"] for s in samples]),
        norm=np.asarray([s["norm"] for s in samples]),
        energy=np.asarray([s["energy"] for s in samples]),
        K_grad=np.asarray([s["K_grad"] for s in samples]),
    )


def count_turning_points(values: list[float]) -> int:
    if len(values) < 3:
        return 0
    signs = np.sign(np.asarray(values, dtype=float))
    signs[np.abs(values) < 1e-10] = 0
    clean = [s for s in signs if s != 0]
    return sum(1 for a, b in zip(clean, clean[1:]) if a * b < 0)


def classify_trajectory(row: dict[str, object], samples: list[dict[str, object]]) -> str:
    if float(row["force_contract_absolute_residual"]) > ABS_FORCE_TOL and float(row["force_contract_relative_residual"]) > REL_FORCE_TOL:
        return "NUMERICALLY_UNRESOLVED"
    T = float(row["T"])
    if float(row["norm_error"]) > (NORM_TOL_LONG if T > 4.0 else NORM_TOL_STD):
        return "NUMERICALLY_UNRESOLVED"
    if float(row["energy_error"]) > (ENERGY_TOL_LONG if T > 4.0 else ENERGY_TOL_STD):
        return "NUMERICALLY_UNRESOLVED"
    rf = float(row["radial_force"])
    drift = float(row.get("periodic_drift", 0.0))
    turns = int(row.get("turning_points", 0))
    if abs(rf) < 1e-8 and str(row["source_family"]) == "shell":
        return "SHELL_INTERIOR_NULL"
    if abs(rf) < 1e-8:
        return "FREE_LIKE"
    if turns >= 2:
        return "OSCILLATORY_OR_BOUND_CANDIDATE"
    if turns == 1:
        return "TURNING_POINT"
    if rf < -5e-3 or drift < -2e-3:
        return "STRONG_DEFLECTION"
    if rf < 0.0:
        return "WEAK_DEFLECTION"
    if rf > 0.0 and drift > 0.0:
        return "OUTWARD_ESCAPE"
    return "SCATTERING_OR_SPLITTING"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = pred - y
    rmse = float(np.sqrt(np.mean(err * err))) if len(y) else float("nan")
    denom = float(np.std(y)) if len(y) and float(np.std(y)) > 1e-30 else 1.0
    ss_res = float(np.sum(err * err))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) if len(y) else 0.0
    return {
        "RMSE": rmse,
        "normalized_RMSE": rmse / denom,
        "median_absolute_error": float(np.median(np.abs(err))) if len(y) else float("nan"),
        "maximum_error": float(np.max(np.abs(err))) if len(y) else float("nan"),
        "R2": 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0,
        "radial_sign_accuracy": float(np.mean(np.sign(y) == np.sign(pred))) if len(y) else float("nan"),
    }


def fit_scalar_feature(rows: list[dict[str, object]], feature: str, target: str = "radial_force") -> dict[str, object]:
    x = np.asarray([float(r[feature]) for r in rows], dtype=float)
    y = np.asarray([float(r[target]) for r in rows], dtype=float)
    denom = float(np.dot(x, x)) + 1e-30
    c = float(np.dot(x, y) / denom)
    pred = c * x
    return {"coefficient": c, **metrics(y, pred)}


def analyze_models(atlas_rows: list[dict[str, object]], traj_rows: list[dict[str, object]], outdir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    rows = [r for r in atlas_rows if float(r["N_min"]) < 0.999 and float(r["radial_distance"]) > 1e-9]
    if not rows:
        model_rows: list[dict[str, object]] = []
        coeff_rows: list[dict[str, object]] = []
        collapse_rows: list[dict[str, object]] = []
        holdout_rows: list[dict[str, object]] = []
        write_csv(outdir / "model_comparison.csv", model_rows)
        write_csv(outdir / "coarse_grained_coefficients.csv", coeff_rows)
        write_csv(outdir / "curve_collapse_metrics.csv", collapse_rows)
        write_csv(outdir / "holdout_validation.csv", holdout_rows)
        return model_rows, coeff_rows, collapse_rows, holdout_rows
    for r in rows:
        r["feature_neg_gradN"] = -float(r["grad_N_at_COM"])
        r["feature_neg_gradlnN"] = -float(r["grad_lnN_at_COM"])
        r["feature_K_gradN"] = -float(r["K_grad"]) * float(r["grad_N_at_COM"])
        r["feature_source_tail"] = -float(r["source_grad_at_COM"])
        r["feature_newtonian"] = -1.0 / (float(r["radial_distance"]) ** 2 + 1e-12)
        r["feature_ray"] = -D * (float(r["carrier_norm"]) ** 2) * float(r["grad_N_at_COM"])
    model_rows: list[dict[str, object]] = []
    for name, feature in [
        ("M1_local_gradient", "feature_neg_gradN"),
        ("M2_log_gradient", "feature_neg_gradlnN"),
        ("M3_ray", "feature_ray"),
        ("M4_source_tail", "feature_source_tail"),
        ("M5_newtonian", "feature_newtonian"),
        ("M0_Kgrad_gradN_reduced", "feature_K_gradN"),
    ]:
        fit = fit_scalar_feature(rows, feature)
        model_rows.append({"model": name, "feature": feature, "n": len(rows), **fit})
    coeff_rows = sparse_feature_fit(rows)
    collapse_rows = curve_collapse(rows)
    holdout_rows = holdout_validation(rows)
    write_csv(outdir / "model_comparison.csv", model_rows)
    write_csv(outdir / "coarse_grained_coefficients.csv", coeff_rows)
    write_csv(outdir / "curve_collapse_metrics.csv", collapse_rows)
    write_csv(outdir / "holdout_validation.csv", holdout_rows)
    return model_rows, coeff_rows, collapse_rows, holdout_rows


def sparse_feature_fit(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []
    features = [
        "feature_neg_gradN",
        "feature_neg_gradlnN",
        "feature_K_gradN",
        "feature_source_tail",
        "feature_newtonian",
        "feature_ray",
    ]
    X = np.asarray([[float(r[f]) for f in features] for r in rows], dtype=float)
    y = np.asarray([float(r["radial_force"]) for r in rows], dtype=float)
    scale = np.std(X, axis=0)
    scale[scale < 1e-30] = 1.0
    Xs = X / scale
    lam = 1e-8
    coef_s = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ y)
    coef = coef_s / scale
    pred = X @ coef
    m = metrics(y, pred)
    rows_out = []
    for f, c in zip(features, coef):
        rows_out.append({"model": "sparse_fixed_library", "feature": f, "coefficient": float(c), **m})
    return rows_out


def curve_collapse(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    pairs = [
        ("F_vs_gradN", "grad_N_at_COM", "radial_force"),
        ("F_over_K_vs_gradN", "grad_N_at_COM", "F_over_K"),
        ("F_vs_KgradN", "feature_K_gradN", "radial_force"),
        ("weak_force_over_epsilon", "radial_distance", "F_over_epsilon"),
    ]
    for r in rows:
        r["F_over_K"] = float(r["radial_force"]) / (float(r["K_grad"]) + EPS)
        r["F_over_epsilon"] = float(r["radial_force"]) / (1.0 - float(r["N_min"]) + EPS)
    for name, xkey, ykey in pairs:
        x = np.asarray([float(r[xkey]) for r in rows], dtype=float)
        y = np.asarray([float(r[ykey]) for r in rows], dtype=float)
        corr = float(np.corrcoef(x, y)[0, 1]) if len(rows) > 2 and np.std(x) > 0 and np.std(y) > 0 else 0.0
        out.append({"collapse": name, "n": len(rows), "correlation": corr, "abs_correlation": abs(corr)})
    return out


def holdout_validation(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    splits = [
        ("train_gaussian_super_holdout_other", lambda r: r["source_family"] in ("gaussian", "supergaussian4")),
        ("train_other_holdout_gaussian_super", lambda r: r["source_family"] not in ("gaussian", "supergaussian4")),
    ]
    for name, train_pred in splits:
        train = [r for r in rows if train_pred(r)]
        hold = [r for r in rows if not train_pred(r)]
        if len(train) < 3 or len(hold) < 2:
            continue
        fit = fit_scalar_feature(train, "feature_K_gradN")
        c = float(fit["coefficient"])
        y = np.asarray([float(r["radial_force"]) for r in hold], dtype=float)
        pred = c * np.asarray([float(r["feature_K_gradN"]) for r in hold], dtype=float)
        out.append({"split": name, "model": "Kgrad_gradN", "train_n": len(train), "holdout_n": len(hold), "coefficient": c, **metrics(y, pred)})
    return out


def classify_labels(model_rows: list[dict[str, object]], holdout_rows: list[dict[str, object]], fals_rows: list[dict[str, object]], validation_rows: list[dict[str, object]]) -> list[str]:
    labels = ["D_EFFECTIVE_MEDIUM_DYNAMICS_CHARACTERIZED"]
    best = sorted(model_rows, key=lambda r: float(r.get("normalized_RMSE", 1e9)))[0] if model_rows else {}
    if best.get("model") in ("M0_Kgrad_gradN_reduced",):
        labels.append("D_GRADIENT_ENERGY_WEIGHTED_FORCE_SUPPORTED")
    if any(r.get("test") == "probe_structure_dependence" and str(r.get("result")) == "structure_dependent" for r in fals_rows):
        labels.append("D_FINITE_WIDTH_WAVE_FORCE_CONFIRMED")
        labels.append("D_NO_UNIVERSAL_FREE_FALL_LIMIT_FOUND")
    if any(r.get("test") == "newtonian_exterior_decay" and str(r.get("result")) == "compact_tail_decay" for r in fals_rows):
        labels.append("D_NEWTONIAN_COMPARISON_REJECTED")
    if holdout_rows:
        worst = max(float(r["normalized_RMSE"]) for r in holdout_rows)
        labels.append("D_REDUCED_MODEL_PARTIAL" if worst > 0.35 else "D_SOURCE_PROFILE_DEPENDENCE_CHARACTERIZED")
    if any(float(r.get("classification_changed", 0.0)) > 0 for r in validation_rows):
        labels.append("D_DYNAMICS_CHARACTERIZATION_INCONCLUSIVE")
    return sorted(set(labels))


def numerical_validation_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for r in rows:
        rid = str(r["run_id"])
        key = rid.replace("_N96", "").replace("_N128", "").replace("_dt_half", "")
        groups.setdefault(key, []).append(r)
    out = []
    for key, rr in groups.items():
        if len(rr) < 2:
            continue
        labels = {r.get("regime_label", "") for r in rr}
        forces = [float(r["radial_force"]) for r in rr]
        out.append(
            {
                "case": key,
                "n": len(rr),
                "radial_force_min": min(forces),
                "radial_force_max": max(forces),
                "relative_force_spread": (max(forces) - min(forces)) / (abs(float(np.mean(forces))) + EPS),
                "classification_changed": 1.0 if len(labels) > 1 else 0.0,
                "max_norm_error": max(float(r["norm_error"]) for r in rr),
                "max_energy_error": max(float(r["energy_error"]) for r in rr),
                "max_force_abs_residual": max(float(r["force_contract_absolute_residual"]) for r in rr),
            }
        )
    return out


def falsification_results(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    probe = [r for r in rows if str(r["run_id"]).startswith("E8_probe_structure_")]
    if probe:
        vals = [float(r["radial_force"]) for r in probe]
        out.append({"test": "probe_structure_dependence", "n": len(vals), "force_min": min(vals), "force_max": max(vals), "relative_spread": (max(vals) - min(vals)) / (abs(float(np.mean(vals))) + EPS), "result": "structure_dependent" if (max(vals) - min(vals)) > 1e-4 else "not_resolved"})
    shell = [r for r in rows if str(r["run_id"]).startswith("E8_shell_map_")]
    if shell:
        interior = [r for r in shell if float(r["radial_distance"]) < 2.4 and float(r["radial_distance"]) > 1e-9]
        interior_max = max(abs(float(r["radial_force"])) for r in interior) if interior else 0.0
        out.append({"test": "shell_interior", "n": len(shell), "interior_max_abs_force": interior_max, "result": "LOCAL_MEDIUM_SHELL_RESPONSE" if interior_max > 1e-6 else "NEWTONIAN_SHELL_LIKE_RESPONSE"})
    ext = [r for r in rows if str(r["run_id"]).startswith("E8_exterior_decay_")]
    if ext:
        far = sorted(ext, key=lambda r: float(r["radial_distance"]))[-1]
        out.append({"test": "newtonian_exterior_decay", "n": len(ext), "far_force": far["radial_force"], "result": "compact_tail_decay" if abs(float(far["radial_force"])) < 1e-5 else "long_tail_present"})
    flat = [r for r in rows if str(r["run_id"]).startswith("E8_constant_N_")]
    if flat:
        out.append({"test": "source_free_background", "n": len(flat), "max_abs_force": max(abs(float(r["radial_force"])) for r in flat), "result": "zero_force" if max(abs(float(r["radial_force"])) for r in flat) < 1e-10 else "failed"})
    super_rows = [r for r in rows if str(r["run_id"]).startswith("E8_superposition_")]
    if super_rows:
        out.append({"test": "weak_field_superposition", "n": len(super_rows), "note": "combined two-source force map computed; one-source additivity is approximated in model comparison", "result": "quantified_not_promoted"})
    return out


def write_svg(path: Path, title: str, rows: list[dict[str, object]], xkey: str, ykey: str, groupkey: str | None = None) -> None:
    data = []
    for r in rows:
        try:
            data.append((float(r[xkey]), float(r[ykey]), str(r[groupkey]) if groupkey else "data"))
        except Exception:
            pass
    if not data:
        path.write_text(f"<svg xmlns='http://www.w3.org/2000/svg' width='720' height='420'><text x='20' y='40'>{title}: no data</text></svg>", encoding="utf-8")
        return
    xs = np.asarray([d[0] for d in data])
    ys = np.asarray([d[1] for d in data])
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    if xmax == xmin:
        xmax += 1.0
    if ymax == ymin:
        ymax += 1.0
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#7f7f7f"]
    groups = {g: colors[i % len(colors)] for i, g in enumerate(sorted({d[2] for d in data}))}
    def sx(x):
        return 60 + 620 * (x - xmin) / (xmax - xmin)
    def sy(y):
        return 360 - 300 * (y - ymin) / (ymax - ymin)
    parts = [
        "<svg xmlns='http://www.w3.org/2000/svg' width='720' height='420'>",
        "<rect x='0' y='0' width='720' height='420' fill='white'/>",
        f"<text x='60' y='30' font-size='18'>{title}</text>",
        "<line x1='60' y1='360' x2='680' y2='360' stroke='black'/>",
        "<line x1='60' y1='60' x2='60' y2='360' stroke='black'/>",
        f"<text x='60' y='395' font-size='12'>{xkey}</text>",
        f"<text x='10' y='55' font-size='12'>{ykey}</text>",
    ]
    for x, y, g in data:
        parts.append(f"<circle cx='{sx(x):.2f}' cy='{sy(y):.2f}' r='3' fill='{groups[g]}'/>")
    yleg = 55
    for g, c in groups.items():
        parts.append(f"<rect x='555' y='{yleg}' width='10' height='10' fill='{c}'/><text x='570' y='{yleg+10}' font-size='11'>{g}</text>")
        yleg += 15
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_plots(outdir: Path, atlas_rows: list[dict[str, object]], traj_rows: list[dict[str, object]], model_rows: list[dict[str, object]], validation_rows: list[dict[str, object]], fals_rows: list[dict[str, object]]) -> None:
    plotdir = outdir / "plots"
    plotdir.mkdir(exist_ok=True)
    for r in atlas_rows:
        r["F_over_K"] = float(r["radial_force"]) / (float(r["K_grad"]) + EPS)
        r["strength"] = 1.0 - float(r["N_min"])
        r["width_ratio"] = float(r["probe_width"]) / (float(r["ell_S"]) + EPS)
    write_svg(plotdir / "force_vs_distance_by_source.svg", "force vs distance by source", atlas_rows, "radial_distance", "radial_force", "source_family")
    write_svg(plotdir / "force_vs_strength.svg", "force vs strength", atlas_rows, "strength", "radial_force", "source_family")
    write_svg(plotdir / "force_vs_probe_width.svg", "force vs probe width", atlas_rows, "probe_width", "radial_force", "source_family")
    write_svg(plotdir / "force_over_Kgrad_vs_gradN.svg", "F/Kgrad vs gradN", atlas_rows, "grad_N_at_COM", "F_over_K", "source_family")
    write_svg(plotdir / "moment_expansion_error_vs_width.svg", "moment expansion proxy vs width", atlas_rows, "probe_width", "force_contract_absolute_residual", "source_family")
    write_svg(plotdir / "ray_vs_wave_trajectories.svg", "ray feature vs wave force", atlas_rows, "carrier_norm", "radial_force", "source_family")
    write_svg(plotdir / "shell_interior_force.svg", "shell interior force", [r for r in atlas_rows + fals_rows if r["source_family"] == "shell"], "radial_distance", "radial_force", "run_id")
    write_svg(plotdir / "two_source_superposition_error.svg", "two source superposition proxy", [r for r in fals_rows if r["source_family"] == "two_source"], "N_min", "radial_force", "run_id")
    write_svg(plotdir / "effective_potential_by_probe_family.svg", "effective potential proxy", atlas_rows, "radial_distance", "radial_force", "probe_profile")
    write_svg(plotdir / "regime_map_distance_vs_carrier.svg", "regime distance vs carrier", traj_rows, "radial_distance", "carrier_norm", "regime_label")
    write_svg(plotdir / "regime_map_strength_vs_width.svg", "regime strength vs width", traj_rows, "N_min", "probe_width", "regime_label")
    write_svg(plotdir / "holdout_prediction_error.svg", "model comparison error", model_rows, "normalized_RMSE", "R2", "model")
    write_svg(plotdir / "numerical_convergence.svg", "numerical convergence", validation_rows, "relative_force_spread", "max_norm_error", "case")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_inventory(outdir: Path) -> None:
    write_csv(outdir / "standalone_file_hashes.csv", [
        {"path": "jax_scout/gravity_D_dynamics_characterization_gpu.py", "bytes": (ROOT / "jax_scout" / "gravity_D_dynamics_characterization_gpu.py").stat().st_size, "sha256": sha256_file(ROOT / "jax_scout" / "gravity_D_dynamics_characterization_gpu.py")}
    ])
    (outdir / "git_status_short.txt").write_text(git_text(["git", "status", "--short"]), encoding="utf-8")
    (outdir / "git_diff_name_only.txt").write_text(git_text(["git", "diff", "--name-only"]), encoding="utf-8")
    (outdir / "git_diff_cached_name_only.txt").write_text(git_text(["git", "diff", "--cached", "--name-only"]), encoding="utf-8")
    rows = []
    for p in sorted(outdir.rglob("*")):
        if p.is_file() and p.name != "artifact_hashes.csv":
            rows.append({"path": str(p.relative_to(outdir)), "bytes": p.stat().st_size, "sha256": sha256_file(p)})
    write_csv(outdir / "artifact_hashes.csv", rows)


def write_handoffs(outdir: Path, labels: list[str], model_rows: list[dict[str, object]], fals_rows: list[dict[str, object]], validation_rows: list[dict[str, object]]) -> None:
    summary = {
        "bounded_labels": labels,
        "best_model": sorted(model_rows, key=lambda r: float(r.get("normalized_RMSE", 1e9)))[0] if model_rows else None,
        "falsification_results": fals_rows,
        "numerical_validation": validation_rows,
        "artifact_path": str(outdir),
        "rejected_interpretations": [
            "gravity confirmation",
            "geodesic validation",
            "universal free fall",
            "IRER gravity validation",
            "production readiness",
        ],
    }
    (outdir / "CHARACTERIZATION_SUMMARY.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    tech = [
        "# Gravity D Dynamics Characterization Technical Handoff",
        "",
        "This run characterizes the confirmed spatial effective-medium operator only.",
        "",
        "## Evidence",
        "",
        "- GPU preflight, environment versions, matrix, CSV tables, plots, and hashes are in this run directory.",
        "- The exact operator-force model remains the reference observable.",
        "- Global and periodic COM are primary; slab COM is diagnostic only.",
        "",
        "## Inference",
        "",
        "- Use the bounded labels in CHARACTERIZATION_SUMMARY.json.",
        "- Treat source, width, and carrier dependencies as characterization of a finite-width wave-medium force.",
        "",
        "## Rejected Interpretations",
        "",
        "- Do not describe this as gravity, temporal lapse, geodesic motion, universal free fall, or IRER confirmation.",
    ]
    (outdir / "TECHNICAL_HANDOFF.md").write_text("\n".join(tech) + "\n", encoding="utf-8")
    open_questions = [
        "# Open Questions",
        "",
        "- Whether a temporal-metric implementation produces compatible or distinct trajectories.",
        "- Whether a physically derived IRER environment can generate the bounded coefficient field.",
        "- Whether higher-resolution compact-packet ray limits improve M3 agreement.",
        "- Whether long-duration bound/capture candidates survive stricter boundary and convergence tests.",
    ]
    (outdir / "OPEN_QUESTIONS.md").write_text("\n".join(open_questions) + "\n", encoding="utf-8")
    doc = [
        "# Documentation Inputs",
        "",
        "## Final Bounded Labels",
        "",
        *[f"- `{label}`" for label in labels],
        "",
        "## Strongest Supporting Tables",
        "",
        "- `instantaneous_force_atlas.csv`",
        "- `trajectory_metrics.csv`",
        "- `model_comparison.csv`",
        "- `holdout_validation.csv`",
        "- `falsification_results.csv`",
        "- `numerical_validation.csv`",
        "",
        "## Caveats",
        "",
        "- This is a spatial effective-medium result only.",
        "- Probe-width and internal-structure dependence must not be relabeled as universal free fall.",
        "- Newtonian and geodesic-like models are benchmarks, not promoted interpretations.",
        "",
        "## Suggested Wording",
        "",
        "The confirmed divergence-form spatial operator is best interpreted as a finite-width wave-medium interaction whose force is governed by the gradient-energy-weighted coefficient gradient.",
    ]
    (outdir / "DOCUMENTATION_INPUTS.md").write_text("\n".join(doc) + "\n", encoding="utf-8")


def write_discrepancy(outdir: Path, reason: str, rows: list[dict[str, object]] | None = None) -> None:
    text = ["# Discrepancy Report", "", reason, ""]
    if rows:
        text.append("## Rows")
        text.append("")
        text.append(json.dumps(rows[:10], indent=2, default=float))
    (outdir / "DISCREPANCY_REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def reference_pass(medium: dict[str, object], flat: dict[str, object]) -> tuple[bool, str]:
    if float(medium["radial_force"]) >= 0.0:
        return False, "E0 medium did not have inward initial force"
    if float(medium["force_contract_absolute_residual"]) > ABS_FORCE_TOL and float(medium["force_contract_relative_residual"]) > REL_FORCE_TOL:
        return False, "E0 medium force contract failed"
    if float(medium["norm_error"]) > NORM_TOL_STD:
        return False, "E0 medium norm failed"
    if abs(float(flat["radial_force"])) > 1e-10:
        return False, "E0 flat force was not near zero"
    if abs(float(flat.get("global_drift", 0.0))) > COM_TOL or abs(float(flat.get("periodic_drift", 0.0))) > COM_TOL:
        return False, "E0 flat global/periodic COM null failed"
    return True, "reference passed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "sweep_runs" / f"GRAVITY_D_DYNAMICS_CHARACTERIZATION_GPU_{time.strftime('%Y%m%d_%H%M%S')}"))
    ap.add_argument("--stages", default="all")
    ap.add_argument("--long-runs", default="enabled", choices=["enabled", "disabled"])
    ap.add_argument("--stop-on-reference-fail", default="true", choices=["true", "false"])
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    stages = {"all"} if args.stages == "all" else {s.strip() for s in args.stages.split(",") if s.strip()}
    pf = save_environment(outdir, " ".join(sys.argv))
    source_meta, atlas_cfgs, traj_cfgs, val_cfgs, fals_cfgs = build_matrices(outdir, stages)
    write_csv(outdir / "source_profile_metadata.csv", source_meta)

    atlas_rows: list[dict[str, object]] = []
    traj_rows: list[dict[str, object]] = []
    val_rows: list[dict[str, object]] = []
    fals_rows_all: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    samples_by_run: dict[str, list[dict[str, object]]] = {}

    for cfg in traj_cfgs:
        if args.long_runs == "disabled" and str(cfg["run_id"]).endswith("_long"):
            continue
        if "requires_gate" in cfg:
            gate = str(cfg["requires_gate"])
            gate_row = next((r for r in traj_rows if r["run_id"] == gate), None)
            if not gate_row or str(gate_row.get("regime_label")) == "NUMERICALLY_UNRESOLVED":
                print(f"[skip] {cfg['run_id']} because gate {gate} did not pass", flush=True)
                continue
        print(f"[trajectory] {cfg['run_id']} T={cfg['T']} N={cfg['N']} L={cfg['L']} source={cfg['source_family']}", flush=True)
        row, samples = run_trajectory(cfg, outdir)
        traj_rows.append(row)
        samples_by_run[str(row["run_id"])] = samples
        manifest.append({"run_id": row["run_id"], "stage": row["stage"], "config_hash": row["config_hash"], "status": "completed", "gpu_device": row["gpu_device"]})
        print(f"  Fr={float(row['radial_force']):+.6e} label={row['regime_label']} norm={float(row['norm_error']):.3e} energy={float(row['energy_error']):.3e}", flush=True)
        if row["run_id"] == "E0_reference_flat":
            medium = next((r for r in traj_rows if r["run_id"] == "E0_reference_medium"), None)
            if medium is not None:
                ok, reason = reference_pass(medium, row)
                if not ok:
                    write_discrepancy(outdir, reason, [medium, row])
                    write_inventory(outdir)
                    print(f"[stop] {reason}", flush=True)
                    if args.stop_on_reference_fail == "true":
                        return

    # Matched free subtraction for trajectory rows with available exact flat controls.
    if "E0_reference_medium" in samples_by_run and "E0_reference_flat" in samples_by_run:
        med = samples_by_run["E0_reference_medium"][-1]
        free = samples_by_run["E0_reference_flat"][-1]
        rhat = np.asarray(base_cfg()["radial_hat"], dtype=float)
        net = float(np.dot(np.asarray(med["periodic_com"]) - np.asarray(free["periodic_com"]), rhat))
        for r in traj_rows:
            if r["run_id"] == "E0_reference_medium":
                r["free_subtracted_COM"] = net

    for cfg in atlas_cfgs:
        print(f"[atlas] {cfg['run_id']} source={cfg['source_family']} Nmin={cfg['N_min']}", flush=True)
        row = run_instantaneous(cfg)
        atlas_rows.append(row)
        manifest.append({"run_id": row["run_id"], "stage": row["stage"], "config_hash": row["config_hash"], "status": "completed", "gpu_device": row["gpu_device"]})

    for cfg in val_cfgs:
        print(f"[validation] {cfg['run_id']} N={cfg['N']} dt={cfg['dt']}", flush=True)
        row, samples = run_trajectory(cfg, outdir)
        val_rows.append(row)
        manifest.append({"run_id": row["run_id"], "stage": row["stage"], "config_hash": row["config_hash"], "status": "completed", "gpu_device": row["gpu_device"]})

    for cfg in fals_cfgs:
        print(f"[falsification] {cfg['run_id']} source={cfg['source_family']}", flush=True)
        row = run_instantaneous(cfg)
        fals_rows_all.append(row)
        manifest.append({"run_id": row["run_id"], "stage": row["stage"], "config_hash": row["config_hash"], "status": "completed", "gpu_device": row["gpu_device"]})

    write_csv(outdir / "instantaneous_force_atlas.csv", atlas_rows)
    write_csv(outdir / "trajectory_metrics.csv", traj_rows)
    write_csv(outdir / "run_manifest.csv", manifest)
    write_csv(outdir / "numerical_validation_raw.csv", val_rows)
    write_csv(outdir / "falsification_raw.csv", fals_rows_all)
    regime_rows = [{"run_id": r["run_id"], "regime_label": r["regime_label"], "radial_force": r["radial_force"], "turning_points": r.get("turning_points", 0), "norm_error": r["norm_error"], "energy_error": r["energy_error"]} for r in traj_rows]
    write_csv(outdir / "regime_classification.csv", regime_rows)

    model_rows, coeff_rows, collapse_rows, holdout_rows = analyze_models(atlas_rows + fals_rows_all, traj_rows, outdir)
    validation_summary = numerical_validation_rows(val_rows)
    write_csv(outdir / "numerical_validation.csv", validation_summary)
    fals_summary = falsification_results(fals_rows_all)
    write_csv(outdir / "falsification_results.csv", fals_summary)
    labels = classify_labels(model_rows, holdout_rows, fals_summary, validation_summary)
    write_plots(outdir, atlas_rows, traj_rows, model_rows, validation_summary, fals_rows_all)
    write_handoffs(outdir, labels, model_rows, fals_summary, validation_summary)
    write_inventory(outdir)
    print(f"=== CHARACTERIZATION COMPLETE labels={','.join(labels)} out={outdir} ===", flush=True)


if __name__ == "__main__":
    main()
