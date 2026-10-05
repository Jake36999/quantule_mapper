"""G1 temporal clock calibration audit.

This driver starts the emergent-gravity maturity programme at the temporal
clock calibration gate.  It keeps the source supplied and objective, and tests
whether localized KG clock systems have a calibrated frequency contract.

Two independent numerical paths are used:

* CPU SciPy sparse generalized eigenproblem:
      A u = omega^2 M u
  with A = -div(N_t grad) + N_t (m^2 + V_trap), M = diag(1/N_t).
* JAX GPU time-domain evolution of the same local first-order KG system:
      phi_t = N_t Pi
      Pi_t  = div(N_t grad phi) - N_t (m^2 + V_trap) phi

The local clock domain is a bounded calibration instrument, not a full
temporal-spatial metric unification and not a gravity verdict.
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
from functools import partial
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jaxlib
import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from jax import lax

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SIGMA = 1.5


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def git_state() -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    except Exception as exc:
        return f"git status failed: {exc}\n"


def command_line() -> str:
    return " ".join([sys.executable, *sys.argv])


def config_hash(cfg: dict[str, Any]) -> str:
    data = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def serialize(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize(row.get(key, "")) for key in keys})


def preflight() -> dict[str, Any]:
    devices = jax.devices()
    print("backend:", jax.default_backend(), flush=True)
    print("devices:", devices, flush=True)
    assert jax.default_backend() == "gpu", f"GPU backend required; got {jax.default_backend()} with {devices}"
    assert any(device.platform == "gpu" for device in devices), devices
    return {
        "backend": jax.default_backend(),
        "devices": [str(device) for device in devices],
        "selected_device": str(devices[0]),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "x64_enabled": bool(jax.config.read("jax_enable_x64")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "command_line": command_line(),
        "env": {
            "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "XLA_PYTHON_CLIENT_MEM_FRACTION": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }


def environment_record(preflight_record: dict[str, Any]) -> dict[str, Any]:
    return {
        **preflight_record,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "operator": "A u = omega^2 M u; phi_t=N_t Pi; Pi_t=div(N_t grad phi)-N_t(m^2+V)phi",
        "scope": "local clock calibration; supplied objective temporal field only",
    }


def local_axes(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    dx = float(cfg["L"]) / int(cfg["N"])
    nloc = int(cfg["local_points"])
    if nloc % 2 == 0:
        raise ValueError("local_points must be odd")
    half = nloc // 2
    center = np.asarray(cfg["clock_position"], dtype=float)
    offsets = (np.arange(nloc, dtype=float) - half) * dx
    return center[0] + offsets, center[1] + offsets, center[2] + offsets, dx


def objective_lapse(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    xs, ys, zs, dx = local_axes(cfg)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    if cfg["field_case"] == "flat":
        Nt = np.ones_like(X)
    else:
        r2 = X * X + Y * Y + Z * Z
        source = np.exp(-(r2 / (SOURCE_SIGMA * SOURCE_SIGMA)))
        nmin = float(cfg["N_min"])
        beta = 1.0 / nmin - 1.0
        Nt = 1.0 / (1.0 + beta * source)
    return X, Y, Z, Nt.astype(np.float64), dx


def trap_potential(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    center = np.asarray(cfg["clock_position"], dtype=float)
    r2 = (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2
    # A weak, probe-independent calibration trap included in flat and shifted controls.
    return float(cfg["trap_k"]) * r2


def idx(i: int, j: int, k: int, n: int) -> int:
    return (i * n + j) * n + k


def build_sparse_operator(cfg: dict[str, Any]) -> tuple[sp.csr_matrix, sp.csr_matrix, dict[str, float], np.ndarray, np.ndarray]:
    X, Y, Z, Nt, dx = objective_lapse(cfg)
    V = trap_potential(X, Y, Z, cfg)
    n = Nt.shape[0]
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    diag = np.zeros(n ** 3, dtype=np.float64)
    inv_dx2 = 1.0 / (dx * dx)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                p = idx(i, j, k, n)
                for di, dj, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                    ii, jj, kk = i + di, j + dj, k + dk
                    if 0 <= ii < n and 0 <= jj < n and 0 <= kk < n:
                        q = idx(ii, jj, kk, n)
                        face = 0.5 * (Nt[i, j, k] + Nt[ii, jj, kk]) * inv_dx2
                        diag[p] += face
                        rows.append(p)
                        cols.append(q)
                        data.append(-face)
                diag[p] += Nt[i, j, k] * (float(cfg["mass"]) ** 2 + V[i, j, k])
    rows.extend(range(n ** 3))
    cols.extend(range(n ** 3))
    data.extend(diag.tolist())
    A = sp.csr_matrix((data, (rows, cols)), shape=(n ** 3, n ** 3))
    M = sp.diags((1.0 / Nt).ravel(), format="csr")
    meta = {
        "Nt_min": float(np.min(Nt)),
        "Nt_max": float(np.max(Nt)),
        "Nt_center": float(Nt[n // 2, n // 2, n // 2]),
        "dx": dx,
        "local_points": n,
        "local_extent": dx * (n - 1),
        "operator_symmetry_residual": float(spla.norm(A - A.T) / max(spla.norm(A), 1e-30)),
    }
    return A, M, meta, Nt, V


def solve_eigen(cfg: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    started = time.time()
    A, M, meta, Nt, V = build_sparse_operator(cfg)
    values, vectors = spla.eigsh(A, k=1, M=M, which="SM", tol=float(cfg["eig_tol"]), maxiter=int(cfg["eig_maxiter"]))
    omega2 = float(values[0])
    mode = np.asarray(vectors[:, 0], dtype=np.float64).reshape(Nt.shape)
    mass_norm = float(np.sum((mode.ravel() ** 2) / Nt.ravel()))
    mode = mode / math.sqrt(max(mass_norm, 1e-30))
    X, Y, Z, _, _ = objective_lapse(cfg)
    mode_weight = (mode * mode) / Nt
    mode_mass = float(np.sum(mode_weight))
    mode_com = [
        float(np.sum(X * mode_weight) / max(mode_mass, 1e-30)),
        float(np.sum(Y * mode_weight) / max(mode_mass, 1e-30)),
        float(np.sum(Z * mode_weight) / max(mode_mass, 1e-30)),
    ]
    intended = np.asarray(cfg["clock_position"], dtype=float)
    localization_error = float(np.linalg.norm(np.asarray(mode_com) - intended))
    row = {
        "run_id": cfg["run_id"],
        "config_hash": config_hash(cfg),
        "clock_family": cfg["clock_family"],
        "field_case": cfg["field_case"],
        "grid": cfg["N"],
        "box": cfg["L"],
        "dt": cfg["dt"],
        "mass": cfg["mass"],
        "clock_width": cfg["clock_width"],
        "amplitude": cfg["amplitude"],
        "N_min": cfg["N_min"],
        "clock_position": cfg["clock_position"],
        "omega_eigen": math.sqrt(max(omega2, 0.0)),
        "omega2_eigen": omega2,
        "fractional_rate_eigen": math.sqrt(max(omega2, 0.0)) / float(cfg["mass"]),
        "local_N_center": meta["Nt_center"],
        "mode_com": mode_com,
        "mode_localization_error": localization_error,
        "Nt_min": meta["Nt_min"],
        "Nt_max": meta["Nt_max"],
        "operator_symmetry_residual": meta["operator_symmetry_residual"],
        "eig_elapsed_s": time.time() - started,
        "local_points": meta["local_points"],
        "local_extent": meta["local_extent"],
    }
    return row, mode, Nt, V


def jnp_div_weighted(phi: jnp.ndarray, coeff: jnp.ndarray, dx: jnp.ndarray) -> jnp.ndarray:
    pad = jnp.pad(phi, ((1, 1), (1, 1), (1, 1)), mode="constant")
    cpad = jnp.pad(coeff, ((1, 1), (1, 1), (1, 1)), mode="edge")
    center = pad[1:-1, 1:-1, 1:-1]
    out = jnp.zeros_like(center)
    for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)):
        sl = [slice(1, -1), slice(1, -1), slice(1, -1)]
        sl[axis] = slice(2, None) if shift > 0 else slice(0, -2)
        neigh = pad[tuple(sl)]
        cn = cpad[tuple(sl)]
        face = 0.5 * (coeff + cn)
        out = out + face * (neigh - center) / (dx * dx)
    return out


@jax.jit
def kg_rhs_local(phi: jnp.ndarray, Pi: jnp.ndarray, Nt: jnp.ndarray, V: jnp.ndarray, dx: jnp.ndarray, mass: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    phi_t = Nt * Pi
    Pi_t = jnp_div_weighted(phi, Nt, dx) - Nt * (mass * mass + V) * phi
    return phi_t, Pi_t


@jax.jit
def kg_step_local(phi: jnp.ndarray, Pi: jnp.ndarray, Nt: jnp.ndarray, V: jnp.ndarray, dx: jnp.ndarray, mass: jnp.ndarray, dt: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    k1p, k1q = kg_rhs_local(phi, Pi, Nt, V, dx, mass)
    k2p, k2q = kg_rhs_local(phi + 0.5 * dt * k1p, Pi + 0.5 * dt * k1q, Nt, V, dx, mass)
    k3p, k3q = kg_rhs_local(phi + 0.5 * dt * k2p, Pi + 0.5 * dt * k2q, Nt, V, dx, mass)
    k4p, k4q = kg_rhs_local(phi + dt * k3p, Pi + dt * k3q, Nt, V, dx, mass)
    return (
        phi + (dt / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p),
        Pi + (dt / 6.0) * (k1q + 2.0 * k2q + 2.0 * k3q + k4q),
    )


@partial(jax.jit, static_argnames=("nsteps",))
def kg_evolve_local(phi: jnp.ndarray, Pi: jnp.ndarray, Nt: jnp.ndarray, V: jnp.ndarray, dx: jnp.ndarray, mass: jnp.ndarray, dt: jnp.ndarray, nsteps: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    def body(_, state):
        return kg_step_local(state[0], state[1], Nt, V, dx, mass, dt)

    return lax.fori_loop(0, nsteps, body, (phi, Pi))


@jax.jit
def local_energy(phi: jnp.ndarray, Pi: jnp.ndarray, Nt: jnp.ndarray, V: jnp.ndarray, dx: jnp.ndarray, mass: jnp.ndarray) -> jnp.ndarray:
    grad = 0.0
    for axis in range(3):
        forward = jnp.roll(phi, -1, axis=axis)
        diff = (forward - phi) / dx
        mask = jnp.ones_like(phi)
        index = [slice(None), slice(None), slice(None)]
        index[axis] = -1
        mask = mask.at[tuple(index)].set(0.0)
        grad = grad + mask * diff * diff
    return 0.5 * jnp.sum(Nt * (Pi * Pi + grad + (mass * mass + V) * phi * phi)) * dx**3


def initial_phi(cfg: dict[str, Any], mode: np.ndarray, Nt: np.ndarray) -> np.ndarray:
    amp = float(cfg["amplitude"])
    if cfg["clock_family"] == "trapped_eigenmode":
        return amp * mode
    X, Y, Z, _, _ = objective_lapse(cfg)
    center = np.asarray(cfg["clock_position"], dtype=float)
    r2 = (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2
    phi = np.exp(-(r2 / (2.0 * float(cfg["clock_width"]) ** 2)))
    mnorm = np.sum((phi.ravel() ** 2) / Nt.ravel())
    return amp * phi / math.sqrt(max(mnorm, 1e-30))


def estimate_frequency(samples: list[dict[str, float]]) -> dict[str, Any]:
    t = np.asarray([row["t"] for row in samples], dtype=np.float64)
    q = np.asarray([row["q"] for row in samples], dtype=np.float64)
    changes = np.where(np.diff(np.signbit(q)))[0]
    crossings = []
    for idx0 in changes:
        t0, t1 = t[idx0], t[idx0 + 1]
        q0, q1 = q[idx0], q[idx0 + 1]
        if q1 == q0:
            crossings.append(t0)
        else:
            crossings.append(t0 - q0 * (t1 - t0) / (q1 - q0))
    if len(crossings) < 3:
        return {"omega_time": float("nan"), "zero_crossings": len(crossings)}
    omega = math.pi / float(np.mean(np.diff(np.asarray(crossings))))
    return {"omega_time": omega, "zero_crossings": len(crossings)}


def run_time_domain(cfg: dict[str, Any], mode: np.ndarray, Nt: np.ndarray, V: np.ndarray, outdir: Path) -> dict[str, Any]:
    started = time.time()
    phi0 = initial_phi(cfg, mode, Nt)
    Pi0 = np.zeros_like(phi0)
    weight = mode / Nt
    denom = float(np.sum(weight * mode))
    phi = jnp.asarray(phi0, dtype=jnp.float64)
    Pi = jnp.asarray(Pi0, dtype=jnp.float64)
    Nt_j = jnp.asarray(Nt, dtype=jnp.float64)
    V_j = jnp.asarray(V, dtype=jnp.float64)
    dx_j = jnp.asarray(float(cfg["L"]) / int(cfg["N"]), dtype=jnp.float64)
    mass_j = jnp.asarray(float(cfg["mass"]), dtype=jnp.float64)
    dt_j = jnp.asarray(float(cfg["dt"]), dtype=jnp.float64)
    every = max(1, int(round(float(cfg["sample_dt"]) / float(cfg["dt"]))))
    nsteps = int(round(float(cfg["T"]) / float(cfg["dt"])))
    chunks = nsteps // every
    tail = nsteps - chunks * every
    samples = []
    compile_start = time.time()
    e0 = local_energy(phi, Pi, Nt_j, V_j, dx_j, mass_j)
    e0.block_until_ready()
    compile_time = time.time() - compile_start
    mode_weight = jnp.asarray(weight, dtype=jnp.float64)
    denom_j = jnp.asarray(denom, dtype=jnp.float64)

    def sample(t: float, phi_now: jnp.ndarray, Pi_now: jnp.ndarray) -> dict[str, float]:
        q = jnp.sum(mode_weight * phi_now) / denom_j
        p = jnp.sum(mode_weight * Pi_now) / denom_j
        energy = local_energy(phi_now, Pi_now, Nt_j, V_j, dx_j, mass_j)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), {"q": q, "p": p, "energy": energy})
        return {"t": t, "q": float(np.asarray(q)), "p": float(np.asarray(p)), "energy": float(np.asarray(energy))}

    samples.append(sample(0.0, phi, Pi))
    exec_start = time.time()
    for i in range(chunks):
        phi, Pi = kg_evolve_local(phi, Pi, Nt_j, V_j, dx_j, mass_j, dt_j, every)
        samples.append(sample(float((i + 1) * every * float(cfg["dt"])), phi, Pi))
    if tail:
        phi, Pi = kg_evolve_local(phi, Pi, Nt_j, V_j, dx_j, mass_j, dt_j, tail)
        phi.block_until_ready()
    execution_time = time.time() - exec_start
    freq = estimate_frequency(samples)
    energy0 = samples[0]["energy"]
    energyT = samples[-1]["energy"]
    row = {
        "run_id": cfg["run_id"],
        "config_hash": config_hash(cfg),
        "clock_family": cfg["clock_family"],
        "field_case": cfg["field_case"],
        "grid": cfg["N"],
        "box": cfg["L"],
        "dt": cfg["dt"],
        "T": cfg["T"],
        "mass": cfg["mass"],
        "clock_width": cfg["clock_width"],
        "amplitude": cfg["amplitude"],
        "N_min": cfg["N_min"],
        "clock_position": cfg["clock_position"],
        "omega_time": freq["omega_time"],
        "fractional_rate_time": freq["omega_time"] / float(cfg["mass"]) if not math.isnan(freq["omega_time"]) else float("nan"),
        "zero_crossings": freq["zero_crossings"],
        "energy_error": abs(energyT - energy0),
        "energy_rel_error": abs(energyT - energy0) / max(abs(energy0), 1e-30),
        "compile_time_s": compile_time,
        "execution_time_s": execution_time,
        "wall_time_s": time.time() - started,
        "gpu_device": str(jax.devices()[0]),
    }
    write_json(outdir / "trajectories" / f"{cfg['run_id']}.json", {"config": cfg, "samples": samples, "summary": row})
    return row


def base_cfg(run_id: str, grid: int, L: float, dt: float, field_case: str, pos: tuple[float, float, float], nmin: float, mass: float, width: float, amp: float, family: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "N": grid,
        "L": L,
        "dt": dt,
        "T": 2.0,
        "sample_dt": 0.01,
        "field_case": field_case,
        "clock_position": pos,
        "N_min": nmin,
        "mass": mass,
        "clock_width": width,
        "amplitude": amp,
        "clock_family": family,
        "local_points": 19,
        "trap_k": 0.05,
        "eig_tol": 1e-9,
        "eig_maxiter": 5000,
    }


def build_matrix(kind: str) -> list[dict[str, Any]]:
    if kind != "bounded":
        raise ValueError("only --matrix bounded is implemented")
    rows: list[dict[str, Any]] = []
    near = (1.2, 0.0, 0.0)
    far = (4.0, 0.0, 0.0)
    for case, pos in (("flat", near), ("near", near), ("far", far)):
        rows.append(base_cfg(f"core_N64_{case}_m12_w15_a1", 64, 30.0, 0.001, case, pos, 0.5, 12.0, 1.5, 1.0, "trapped_eigenmode"))
    for nmin in (0.8, 0.65, 0.5):
        rows.append(base_cfg(f"strength_N64_near_Nmin{nmin:g}", 64, 30.0, 0.001, "near", near, nmin, 12.0, 1.5, 1.0, "trapped_eigenmode"))
    for mass in (8.0, 12.0, 20.0):
        rows.append(base_cfg(f"mass_N64_near_m{mass:g}", 64, 30.0, 0.001, "near", near, 0.5, mass, 1.5, 1.0, "trapped_eigenmode"))
    for width in (1.0, 1.5, 2.0):
        rows.append(base_cfg(f"width_N64_near_w{width:g}", 64, 30.0, 0.001, "near", near, 0.5, 12.0, width, 1.0, "trapped_eigenmode"))
    for amp in (1.0, 0.5, 0.1):
        rows.append(base_cfg(f"amp_N64_near_a{amp:g}", 64, 30.0, 0.001, "near", near, 0.5, 12.0, 1.5, amp, "trapped_eigenmode"))
    for grid, L, dt in ((96, 30.0, 0.0005), (128, 40.0, 0.0005)):
        for case, pos in (("flat", near), ("near", near), ("far", far)):
            rows.append(base_cfg(f"refine_N{grid}_{case}_m12_w15_a1", grid, L, dt, case, pos, 0.5, 12.0, 1.5, 1.0, "trapped_eigenmode"))
    for case, pos in (("flat", near), ("near", near), ("far", far)):
        rows.append(base_cfg(f"packet_N64_{case}_m20_w1_a1", 64, 30.0, 0.001, case, pos, 0.5, 20.0, 1.0, 1.0, "high_mass_compact_packet"))
    # Stable unique order.
    seen = set()
    unique = []
    for row in rows:
        if row["run_id"] not in seen:
            unique.append(row)
            seen.add(row["run_id"])
    return unique


def compare_rows(eigen_rows: list[dict[str, Any]], time_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    eig = {row["run_id"]: row for row in eigen_rows}
    comparisons = []
    for row in time_rows:
        erow = eig[row["run_id"]]
        delta = row["omega_time"] - erow["omega_eigen"]
        rel = abs(delta) / max(abs(erow["omega_eigen"]), 1e-30)
        comparisons.append({
            "run_id": row["run_id"],
            "clock_family": row["clock_family"],
            "field_case": row["field_case"],
            "grid": row["grid"],
            "mass": row["mass"],
            "clock_width": row["clock_width"],
            "amplitude": row["amplitude"],
            "omega_eigen": erow["omega_eigen"],
            "omega_time": row["omega_time"],
            "frequency_abs_error": abs(delta),
            "frequency_rel_error": rel,
            "contract_pass": rel <= 5e-4,
            "energy_rel_error": row["energy_rel_error"],
            "operator_symmetry_residual": erow["operator_symmetry_residual"],
        })
    refinement = [row for row in comparisons if str(row["run_id"]).startswith("refine_") or str(row["run_id"]).startswith("core_")]
    falsifiers = []
    flat = next((row for row in comparisons if row["run_id"] == "core_N64_flat_m12_w15_a1"), None)
    near = next((row for row in comparisons if row["run_id"] == "core_N64_near_m12_w15_a1"), None)
    far = next((row for row in comparisons if row["run_id"] == "core_N64_far_m12_w15_a1"), None)
    if flat and near and far:
        falsifiers.append({"test_id": "eigen_time_contract_core", "passed": all(row["contract_pass"] for row in (flat, near, far)), "measured": max(row["frequency_rel_error"] for row in (flat, near, far))})
        falsifiers.append({"test_id": "near_far_shift_detected", "passed": near["omega_time"] < far["omega_time"], "measured": {"near": near["omega_time"], "far": far["omega_time"]}})
        falsifiers.append({"test_id": "near_flat_shift_detected", "passed": near["omega_time"] < flat["omega_time"], "measured": {"near": near["omega_time"], "flat": flat["omega_time"]}})
    amp_rows = [row for row in comparisons if str(row["run_id"]).startswith("amp_N64")]
    if amp_rows:
        spread = max(row["omega_time"] for row in amp_rows) - min(row["omega_time"] for row in amp_rows)
        falsifiers.append({"test_id": "measured_amplitude_invariance", "passed": spread <= 1e-8, "measured": spread})
    packet_rows = [row for row in comparisons if str(row["run_id"]).startswith("packet_N64")]
    eig_rows = [row for row in comparisons if row["run_id"] in ("core_N64_flat_m12_w15_a1", "core_N64_near_m12_w15_a1", "core_N64_far_m12_w15_a1")]
    if packet_rows and eig_rows:
        packet_shift = next(row for row in packet_rows if row["field_case"] == "far")["omega_time"] - next(row for row in packet_rows if row["field_case"] == "near")["omega_time"]
        eigen_shift = next(row for row in eig_rows if row["field_case"] == "far")["omega_time"] - next(row for row in eig_rows if row["field_case"] == "near")["omega_time"]
        falsifiers.append({"test_id": "two_clock_family_shift_same_direction", "passed": packet_shift > 0 and eigen_shift > 0, "measured": {"packet_far_minus_near": packet_shift, "eigen_far_minus_near": eigen_shift}})
    return comparisons, refinement, falsifiers


def labels_from(falsifiers: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> list[str]:
    labels = []
    if falsifiers and all(row["passed"] for row in falsifiers if row["test_id"] in ("eigen_time_contract_core", "near_far_shift_detected", "measured_amplitude_invariance")):
        labels.append("TEMPORAL_KG_EIGENFREQUENCY_CONTRACT_CLOSED")
    family = next((row for row in falsifiers if row["test_id"] == "two_clock_family_shift_same_direction"), None)
    if family and family["passed"]:
        labels.append("TEMPORAL_CLOCK_MECHANISM_AGREEMENT_SUPPORTED")
    # Local-rate support is intentionally stringent and not expected to be awarded unless measured/eigen rates collapse toward local N_t.
    if labels:
        labels.append("TEMPORAL_CLOCK_CALIBRATION_PARTIAL")
    else:
        labels.append("TEMPORAL_CLOCK_CALIBRATION_FAILED")
    return labels


def hash_artifacts(outdir: Path) -> list[dict[str, str]]:
    rows = []
    for path in sorted(outdir.rglob("*")):
        if path.is_file() and path.name != "artifact_hashes.csv":
            rows.append({"path": str(path.relative_to(outdir)).replace("\\", "/"), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return rows


def write_handoffs(outdir: Path, labels: list[str], falsifiers: list[dict[str, Any]]) -> None:
    lines = [
        "# G1 Temporal Clock Calibration",
        "",
        "This run tests the eigenfrequency contract for supplied objective temporal fields.",
        "",
        "## Labels",
        "",
        *[f"- `{label}`" for label in labels],
        "",
        "## Falsification Checks",
        "",
        "| test | passed | measured |",
        "| --- | --- | --- |",
    ]
    for row in falsifiers:
        lines.append(f"| `{row['test_id']}` | {row['passed']} | `{json.dumps(row['measured'], sort_keys=True)}` |")
    lines.extend([
        "",
        "## Boundary",
        "",
        "This is G1 clock calibration only. It does not implement a source-to-geometry law, combined metric dynamics, or a gravity verdict.",
    ])
    (outdir / "TECHNICAL_HANDOFF.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (outdir / "OPEN_QUESTIONS.md").write_text(
        "# Open Questions\n\n"
        "- Does the calibrated local rate approach `m N_t(R)` under a stronger localized/high-mass limit?\n"
        "- Which clock family provides the cleanest external rerun benchmark?\n"
        "- Should G2 use `a_s` directly or test mappings from the previously validated spatial coefficient `A_s`?\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--stages", default="all")
    parser.add_argument("--matrix", default="bounded")
    parser.add_argument("--stop-on-gpu-fail", default="true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = args.out or (ROOT / "sweep_runs" / f"GRAVITY_MATURITY_G1_CLOCK_{timestamp}")
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "trajectories").mkdir(exist_ok=True)
    (outdir / "failed_runs").mkdir(exist_ok=True)
    (outdir / "plots").mkdir(exist_ok=True)
    try:
        preflight_record = preflight()
    except Exception as exc:
        write_json(outdir / "DISCREPANCY_REPORT.json", {"error": str(exc), "command_line": command_line()})
        if str(args.stop_on_gpu_fail).lower() == "true":
            raise
        preflight_record = {"backend": jax.default_backend(), "devices": [str(d) for d in jax.devices()]}
    write_json(outdir / "gpu_preflight.json", preflight_record)
    write_json(outdir / "environment_versions.json", environment_record(preflight_record))
    (outdir / "git_state_before.txt").write_text(git_state(), encoding="utf-8")
    matrix = build_matrix(args.matrix)
    write_json(outdir / "preregistered_matrix.json", matrix)

    eigen_rows = []
    time_rows = []
    for cfg in matrix:
        print(f"running {cfg['run_id']}", flush=True)
        erow, mode, Nt, V = solve_eigen(cfg)
        trow = run_time_domain(cfg, mode, Nt, V, outdir)
        eigen_rows.append(erow)
        time_rows.append(trow)

    comparisons, refinement, falsifiers = compare_rows(eigen_rows, time_rows)
    labels = labels_from(falsifiers, comparisons)
    write_csv(outdir / "eigenfrequency_metrics.csv", eigen_rows)
    write_csv(outdir / "time_domain_clock_metrics.csv", time_rows)
    write_csv(outdir / "clock_family_comparison.csv", comparisons)
    write_csv(outdir / "metrics.csv", comparisons)
    write_csv(outdir / "numerical_refinement.csv", refinement)
    write_csv(outdir / "falsification_results.csv", falsifiers)
    write_csv(
        outdir / "run_manifest.csv",
        [
            {
                "run_id": cfg["run_id"],
                "config_hash": config_hash(cfg),
                "clock_family": cfg["clock_family"],
                "field_case": cfg["field_case"],
                "grid": cfg["N"],
                "box": cfg["L"],
                "dt": cfg["dt"],
                "T": cfg["T"],
                "mass": cfg["mass"],
                "clock_width": cfg["clock_width"],
                "amplitude": cfg["amplitude"],
            }
            for cfg in matrix
        ],
    )
    write_json(outdir / "metrics.json", {"labels": labels, "falsification_results": falsifiers})
    write_handoffs(outdir, labels, falsifiers)
    (outdir / "git_status_short.txt").write_text(git_state(), encoding="utf-8")
    (outdir / "git_diff_name_only.txt").write_text(subprocess.check_output(["git", "diff", "--name-only"], cwd=ROOT, text=True), encoding="utf-8")
    (outdir / "git_diff_cached_name_only.txt").write_text(subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True), encoding="utf-8")
    write_csv(outdir / "artifact_hashes.csv", hash_artifacts(outdir))
    print(f"labels: {', '.join(labels)}", flush=True)
    print(f"outdir: {outdir}", flush=True)


if __name__ == "__main__":
    main()
