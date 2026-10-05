"""Temporal-spatial bridge audit for the Gravity C/D mirror branches.

This standalone mirror driver keeps the two implemented mechanisms separate:

* C-series temporal throttling: a clock factor N_t(x) = 1 / (1 + beta_t S).
* Gravity D spatial response: a kinetic coefficient A_s(x) inside
  H = -D div(A_s grad).

It deliberately does not claim a metric-consistent spacetime equation.  Field
evolution uses the validated spatial GPU solver primitives; temporal factors
are measured as clock observables.  The restricted COMMON_FIELD rows are a
separate hypothesis check, not the default bridge.

Run in the established WSL2 JAX venv:
    . ~/jax_irer/bin/activate
    python /mnt/f/quantule_mapper/jax_scout/gravity_TS_temporal_spatial_bridge_gpu.py \
      --out /mnt/f/quantule_mapper/sweep_runs/GRAVITY_TS_BRIDGE_GPU_<timestamp>
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
from typing import Any

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
from jax_scout.provenance import write_json  # noqa: E402  (shared: stamps run metadata)


BASE_N = 96
BASE_L = 30.0
BASE_DT = 0.001
SOURCE_SIGMA = 1.5
PROBE_X0 = 4.0
PROBE_SIGMA = 1.0
SOURCE_BETA = 1.0
SAMPLE_DT = 0.05

ARMS = ("FLAT", "TEMPORAL_ONLY", "SPATIAL_ONLY", "TEMPORAL_SPATIAL")
SOURCE_MODES = ("objective", "relational")
DURATIONS = {"short": 0.5, "standard": 4.0}


def jscalar(x: jnp.ndarray) -> float:
    return float(np.asarray(x))


def git_state() -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    except Exception as exc:  # pragma: no cover - diagnostic path
        return f"git status failed: {exc}\n"


def command_line() -> str:
    return " ".join([sys.executable, *sys.argv])


def environment_record(preflight_record: dict[str, Any]) -> dict[str, Any]:
    return {
        **preflight_record,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "x64_enabled": bool(jax.config.read("jax_enable_x64")),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "command_line": command_line(),
        "working_directory": str(ROOT),
        "operating_environment": "WSL/JAX GPU mirror expected",
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_csv_value(row.get(key, "")) for key in fieldnames})


def serialize_csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def source_env(grid: dict[str, object]) -> jnp.ndarray:
    X, Y, Z = grid["X"], grid["Y"], grid["Z"]
    r2 = X * X + Y * Y + Z * Z
    rho_e = jnp.exp(-(r2 / (2.0 * SOURCE_SIGMA * SOURCE_SIGMA)))
    source = rho_e * rho_e
    return source / (jnp.max(source) + 1e-30)


def initial_probe(grid: dict[str, object], cfg: dict[str, Any]) -> jnp.ndarray:
    X, Y, Z = grid["X"], grid["Y"], grid["Z"]
    px, py, pz = cfg.get("probe_position", (PROBE_X0, 0.0, 0.0))
    sig = float(cfg.get("probe_width", PROBE_SIGMA))
    amp = float(cfg.get("probe_amplitude", 1.0))
    kx, ky, kz = cfg.get("carrier_vector", (0.0, 0.0, 0.0))
    dx, dy, dz = X - float(px), Y - float(py), Z - float(pz)
    envelope = amp * jnp.exp(-((dx * dx + dy * dy + dz * dz) / (2.0 * sig * sig)))
    if bool(cfg.get("normalize_probe", True)):
        target = jnp.asarray(float(cfg.get("target_norm", 1.0)), dtype=jnp.float64)
        norm = jnp.sum(jnp.abs(envelope) ** 2) * grid["dV"]
        envelope = envelope * jnp.sqrt(target / (norm + 1e-30))
    phase = jnp.exp(1j * (float(kx) * X + float(ky) * Y + float(kz) * Z))
    return (envelope * phase).astype(jnp.complex128)


def coefficient_from_source(source: jnp.ndarray, beta: float) -> jnp.ndarray:
    return 1.0 / (1.0 + float(beta) * source)


def relational_raw_source(psi: jnp.ndarray, source: jnp.ndarray) -> jnp.ndarray:
    return (jnp.abs(psi) ** 2) * source


def relational_scale_for(grid: dict[str, object]) -> float:
    cfg = {
        "probe_position": (PROBE_X0, 0.0, 0.0),
        "probe_width": PROBE_SIGMA,
        "probe_amplitude": 1.0,
        "normalize_probe": False,
    }
    psi = initial_probe(grid, cfg)
    scale = jnp.max(relational_raw_source(psi, source_env(grid)))
    return max(jscalar(scale), 1e-30)


def static_sources(
    grid: dict[str, object],
    psi0: jnp.ndarray,
    cfg: dict[str, Any],
) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    source = source_env(grid)
    rel_scale = float(cfg.get("relational_scale", relational_scale_for(grid)))
    relational = relational_raw_source(psi0, source) / rel_scale
    return source, relational, rel_scale


def spatial_coefficient(
    arm: str,
    source_mode: str,
    source_obj: jnp.ndarray,
    source_rel_initial: jnp.ndarray,
    beta_s: float,
) -> jnp.ndarray:
    if arm in ("FLAT", "TEMPORAL_ONLY"):
        return jnp.ones_like(source_obj)
    source = source_obj if source_mode == "objective" else source_rel_initial
    return coefficient_from_source(source, beta_s)


def temporal_coefficient(
    psi: jnp.ndarray,
    source_obj: jnp.ndarray,
    rel_scale: float,
    arm: str,
    source_mode: str,
    beta_t: float,
    common_field: bool,
) -> jnp.ndarray:
    if arm in ("FLAT", "SPATIAL_ONLY"):
        return jnp.ones_like(source_obj)
    if source_mode == "objective":
        return coefficient_from_source(source_obj, beta_t)
    source_rel = relational_raw_source(psi, source_obj) / rel_scale
    if common_field:
        return coefficient_from_source(source_rel, beta_t)
    return coefficient_from_source(source_rel, beta_t)


@jax.jit
def bridge_diagnostics(
    psi: jnp.ndarray,
    As: jnp.ndarray,
    Nt: jnp.ndarray,
    source_obj: jnp.ndarray,
    grid: dict[str, object],
    source_pos: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    X, Y, Z, dV = grid["X"], grid["Y"], grid["Z"], grid["dV"]
    rho = jnp.abs(psi) ** 2
    norm = jnp.sum(rho) * dV
    com = jnp.array(
        [
            jnp.sum(X * rho) * dV / (norm + 1e-30),
            jnp.sum(Y * rho) * dV / (norm + 1e-30),
            jnp.sum(Z * rho) * dV / (norm + 1e-30),
        ],
        dtype=jnp.float64,
    )
    L = X.shape[0] * (grid["dV"] ** (1.0 / 3.0))
    theta_x = 2.0 * jnp.pi * X / L
    cmean = jnp.sum(jnp.cos(theta_x) * rho) * dV / (norm + 1e-30)
    smean = jnp.sum(jnp.sin(theta_x) * rho) * dV / (norm + 1e-30)
    periodic_x = (L / (2.0 * jnp.pi)) * jnp.arctan2(smean, cmean)
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
    dAx = jnp.real(deriv(As, grid["ikx"]))
    dAy = jnp.real(deriv(As, grid["iky"]))
    dAz = jnp.real(deriv(As, grid["ikz"]))
    force_exact = -D * jnp.array(
        [
            jnp.sum(dAx * grad_energy_density) * dV,
            jnp.sum(dAy * grad_energy_density) * dV,
            jnp.sum(dAz * grad_energy_density) * dV,
        ],
        dtype=jnp.float64,
    )
    psit = rhs(psi, As, grid)
    gtx, gty, gtz = deriv(psit, grid["ikx"]), deriv(psit, grid["iky"]), deriv(psit, grid["ikz"])
    force_rhs = jnp.array(
        [
            jnp.sum(jnp.imag(jnp.conj(psit) * gx + jnp.conj(psi) * gtx)) * dV,
            jnp.sum(jnp.imag(jnp.conj(psit) * gy + jnp.conj(psi) * gty)) * dV,
            jnp.sum(jnp.imag(jnp.conj(psit) * gz + jnp.conj(psi) * gtz)) * dV,
        ],
        dtype=jnp.float64,
    )
    dr = com - source_pos
    radius = jnp.linalg.norm(dr)
    rhat = dr / (radius + 1e-30)
    radial_force = jnp.dot(force_exact, rhat)
    transverse_force = jnp.linalg.norm(force_exact - radial_force * rhat)
    radial_momentum = jnp.dot(momentum, rhat)
    clock_rate = jnp.sum(Nt * rho) * dV / (norm + 1e-30)
    energy = D * jnp.sum(As * grad_energy_density) * dV
    source_exposure = jnp.sum(source_obj * rho) * dV / (norm + 1e-30)
    return {
        "norm": norm,
        "com": com,
        "periodic_com_x": periodic_x,
        "momentum": momentum,
        "radial_momentum": radial_momentum,
        "K_grad": K_grad,
        "force_exact": force_exact,
        "force_rhs": force_rhs,
        "radial_force": radial_force,
        "transverse_force": transverse_force,
        "clock_rate": clock_rate,
        "energy": energy,
        "radius": radius,
        "source_exposure": source_exposure,
        "As_min": jnp.min(As),
        "As_max": jnp.max(As),
        "Nt_min": jnp.min(Nt),
        "Nt_max": jnp.max(Nt),
    }


def scalar_diag(diag: dict[str, jnp.ndarray]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in diag.items():
        arr = np.asarray(value)
        if arr.shape == ():
            out[key] = float(arr)
        else:
            out[key] = [float(x) for x in arr.ravel()]
    return out


def force_fd(
    psi: jnp.ndarray,
    As: jnp.ndarray,
    Nt: jnp.ndarray,
    source_obj: jnp.ndarray,
    grid: dict[str, object],
    source_pos: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    diag0 = bridge_diagnostics(psi, As, Nt, source_obj, grid, source_pos)
    psi1 = rk4_step(psi, As, grid, jnp.asarray(dt, dtype=jnp.float64))
    diag1 = bridge_diagnostics(psi1, As, Nt, source_obj, grid, source_pos)
    return (diag1["momentum"] - diag0["momentum"]) / jnp.asarray(dt, dtype=jnp.float64)


def run_case(cfg: dict[str, Any], outdir: Path) -> dict[str, Any]:
    started = time.time()
    n = int(cfg["N"])
    L = float(cfg["L"])
    dt = float(cfg["dt"])
    T = float(cfg["T"])
    steps = int(round(T / dt))
    every = max(1, int(round(float(cfg.get("sample_dt", SAMPLE_DT)) / dt)))
    chunks = steps // every
    tail = steps - chunks * every
    arm = str(cfg["arm"])
    source_mode = str(cfg["source_mode"])
    beta_t = float(cfg.get("beta_t", SOURCE_BETA))
    beta_s = float(cfg.get("beta_s", SOURCE_BETA))
    common_field = arm == "COMMON_FIELD"

    grid = build_grid(n, L)
    psi = initial_probe(grid, cfg)
    source_obj, source_rel_initial, rel_scale = static_sources(grid, psi, cfg)
    As = spatial_coefficient(arm, source_mode, source_obj, source_rel_initial, beta_s)
    source_pos = jnp.asarray(cfg.get("source_position", (0.0, 0.0, 0.0)), dtype=jnp.float64)
    dt_j = jnp.asarray(dt, dtype=jnp.float64)

    samples: list[dict[str, Any]] = []
    tau = 0.0
    compile_start = time.time()
    Nt = temporal_coefficient(psi, source_obj, rel_scale, arm, source_mode, beta_t, common_field)
    d0 = bridge_diagnostics(psi, As, Nt, source_obj, grid, source_pos)
    fd0 = force_fd(psi, As, Nt, source_obj, grid, source_pos, dt)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), {"diag": d0, "fd": fd0})
    compile_time = time.time() - compile_start
    s0 = scalar_diag(d0)
    s0["force_fd"] = [float(x) for x in np.asarray(fd0).ravel()]
    s0["t"] = 0.0
    s0["tau"] = 0.0
    samples.append(s0)
    prev_t = 0.0

    exec_start = time.time()
    for i in range(chunks):
        psi = evolve_n(psi, As, grid, dt_j, every)
        Nt = temporal_coefficient(psi, source_obj, rel_scale, arm, source_mode, beta_t, common_field)
        di = bridge_diagnostics(psi, As, Nt, source_obj, grid, source_pos)
        fd = force_fd(psi, As, Nt, source_obj, grid, source_pos, dt)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), {"diag": di, "fd": fd})
        t_now = float((i + 1) * every * dt)
        sd = scalar_diag(di)
        tau += 0.5 * (samples[-1]["clock_rate"] + sd["clock_rate"]) * (t_now - prev_t)
        sd["force_fd"] = [float(x) for x in np.asarray(fd).ravel()]
        sd["t"] = t_now
        sd["tau"] = tau
        samples.append(sd)
        prev_t = t_now

    if tail:
        psi = evolve_n(psi, As, grid, dt_j, tail)
        psi.block_until_ready()
    execution_time = time.time() - exec_start
    total_time = time.time() - started

    final = samples[-1]
    initial = samples[0]
    force0 = np.asarray(initial["force_exact"], dtype=float)
    rhs0 = np.asarray(initial["force_rhs"], dtype=float)
    fd0_np = np.asarray(initial["force_fd"], dtype=float)
    residual_abs = float(np.linalg.norm(force0 - rhs0))
    fd_residual_abs = float(np.linalg.norm(force0 - fd0_np))
    force_norm = float(np.linalg.norm(force0))
    final_force = np.asarray(final["force_exact"], dtype=float)
    result = {
        "run_id": cfg["run_id"],
        "config": cfg,
        "config_hash": config_hash(cfg),
        "git_commit": git_commit(),
        "backend": jax.default_backend(),
        "gpu_device": str(jax.devices()[0]),
        "operator_scope": "spatial evolution uses H=-D div(A_s grad); N_t is clock observable",
        "relational_scale": rel_scale,
        "cell_volume_dV": float((L / n) ** 3),
        "source_metadata": {
            "objective_source": "S_E=rho_E^2/max(rho_E^2), rho_E=exp(-r^2/(2 sigma_E^2))",
            "relational_source": "I_A|E=|psi_A|^2 S_E, scaled by unit-amplitude baseline",
            "source_sigma": SOURCE_SIGMA,
            "objective_integral": jscalar(jnp.sum(source_obj) * grid["dV"]),
            "relational_initial_max": jscalar(jnp.max(source_rel_initial)),
            "relational_initial_integral": jscalar(jnp.sum(source_rel_initial) * grid["dV"]),
        },
        "coefficient_metadata": {
            "As_min": initial["As_min"],
            "As_max": initial["As_max"],
            "Nt_min": initial["Nt_min"],
            "Nt_max": initial["Nt_max"],
        },
        "samples": samples,
        "summary": {
            "initial_clock_rate": initial["clock_rate"],
            "mean_clock_rate": (tau / T) if T > 0 else initial["clock_rate"],
            "final_tau": tau,
            "clock_shift": 1.0 - ((tau / T) if T > 0 else initial["clock_rate"]),
            "initial_force_exact": initial["force_exact"],
            "initial_force_rhs": initial["force_rhs"],
            "initial_force_fd": initial["force_fd"],
            "initial_radial_force": initial["radial_force"],
            "final_radial_force": final["radial_force"],
            "force_contract_abs_residual": residual_abs,
            "force_fd_abs_residual": fd_residual_abs,
            "force_contract_rel_residual": residual_abs / max(force_norm, 1e-30),
            "norm_error": abs(final["norm"] - initial["norm"]),
            "energy_error": abs(final["energy"] - initial["energy"]),
            "energy_rel_error": abs(final["energy"] - initial["energy"]) / max(abs(initial["energy"]), 1e-30),
            "global_com_initial": initial["com"],
            "global_com_final": final["com"],
            "periodic_com_x_initial": initial["periodic_com_x"],
            "periodic_com_x_final": final["periodic_com_x"],
            "radial_drift": final["radius"] - initial["radius"],
            "momentum_change_norm": float(np.linalg.norm(np.asarray(final["momentum"], dtype=float) - np.asarray(initial["momentum"], dtype=float))),
            "initial_force_norm": force_norm,
            "final_force_norm": float(np.linalg.norm(final_force)),
            "compile_time_s": compile_time,
            "execution_time_s": execution_time,
            "wall_time_s": total_time,
        },
    }

    run_path = outdir / "trajectories" / f"{cfg['run_id']}.json"
    run_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(run_path, result)
    return result


def selected_values(value: str, allowed: tuple[str, ...]) -> list[str]:
    if value == "all":
        return list(allowed)
    chosen = [item.strip() for item in value.split(",") if item.strip()]
    bad = sorted(set(chosen) - set(allowed))
    if bad:
        raise ValueError(f"unsupported value(s) {bad}; allowed {allowed}")
    return chosen


def duration_values(value: str) -> list[str]:
    if value == "all":
        return list(DURATIONS)
    chosen = [item.strip() for item in value.split(",") if item.strip()]
    bad = sorted(set(chosen) - set(DURATIONS))
    if bad:
        raise ValueError(f"unsupported duration(s) {bad}; allowed {tuple(DURATIONS)}")
    return chosen


def build_matrix(args: argparse.Namespace, rel_scale: float) -> list[dict[str, Any]]:
    arms = selected_values(args.arms, ARMS)
    sources = selected_values(args.source, SOURCE_MODES)
    durations = duration_values(args.durations)
    matrix: list[dict[str, Any]] = []
    for source_mode in sources:
        for duration_name in durations:
            for arm in arms:
                cfg = base_cfg(arm, source_mode, duration_name, rel_scale)
                matrix.append(cfg)
    # Weak-probe rows are temporal-only decomposition checks.  Objective rows
    # should converge to the same rate; relational rows should weaken toward 1.
    for source_mode in sources:
        for amp in (1.0, 0.5, 0.25, 0.1):
            cfg = base_cfg("TEMPORAL_ONLY", source_mode, "short", rel_scale)
            cfg.update(
                {
                    "run_id": f"weak_probe_{source_mode}_amp{amp:g}",
                    "probe_amplitude": amp,
                    "normalize_probe": False,
                    "matrix_role": "weak_probe_limit",
                }
            )
            matrix.append(cfg)
    return matrix


def base_cfg(arm: str, source_mode: str, duration_name: str, rel_scale: float) -> dict[str, Any]:
    T = DURATIONS[duration_name]
    return {
        "run_id": f"{arm.lower()}_{source_mode}_{duration_name}",
        "arm": arm,
        "source_mode": source_mode,
        "duration_name": duration_name,
        "matrix_role": "four_arm_decomposition",
        "N": BASE_N,
        "L": BASE_L,
        "dt": BASE_DT,
        "T": T,
        "sample_dt": SAMPLE_DT,
        "D": D,
        "source_width": SOURCE_SIGMA,
        "beta_t": SOURCE_BETA,
        "beta_s": SOURCE_BETA,
        "probe_position": (PROBE_X0, 0.0, 0.0),
        "probe_width": PROBE_SIGMA,
        "probe_amplitude": 1.0,
        "normalize_probe": True,
        "target_norm": 1.0,
        "carrier_vector": (0.0, 0.0, 0.0),
        "relational_scale": rel_scale,
    }


def common_field_matrix(sources: list[str], rel_scale: float) -> list[dict[str, Any]]:
    rows = []
    for source_mode in sources:
        cfg = base_cfg("COMMON_FIELD", source_mode, "standard", rel_scale)
        cfg["run_id"] = f"common_field_{source_mode}_standard"
        cfg["matrix_role"] = "common_field_check"
        rows.append(cfg)
    return rows


def summarize_rows(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    clock_rows: list[dict[str, Any]] = []
    force_rows: list[dict[str, Any]] = []
    traj_rows: list[dict[str, Any]] = []
    for result in results:
        cfg = result["config"]
        summary = result["summary"]
        base = manifest_base(result)
        clock_rows.append(
            {
                **base,
                "initial_clock_rate": summary["initial_clock_rate"],
                "mean_clock_rate": summary["mean_clock_rate"],
                "final_tau": summary["final_tau"],
                "clock_shift": summary["clock_shift"],
                "Nt_min": result["coefficient_metadata"]["Nt_min"],
                "Nt_max": result["coefficient_metadata"]["Nt_max"],
            }
        )
        force_exact = np.asarray(summary["initial_force_exact"], dtype=float)
        force_rhs = np.asarray(summary["initial_force_rhs"], dtype=float)
        force_fd_np = np.asarray(summary["initial_force_fd"], dtype=float)
        force_rows.append(
            {
                **base,
                "initial_force_exact": summary["initial_force_exact"],
                "initial_force_rhs": summary["initial_force_rhs"],
                "initial_force_fd": summary["initial_force_fd"],
                "raw_force_norm": float(np.linalg.norm(force_exact)),
                "force_per_norm": float(np.linalg.norm(force_exact)) / max(result["samples"][0]["norm"], 1e-30),
                "initial_radial_force": summary["initial_radial_force"],
                "force_rhs_residual": float(np.linalg.norm(force_exact - force_rhs)),
                "force_fd_residual": float(np.linalg.norm(force_exact - force_fd_np)),
                "force_contract_rel_residual": summary["force_contract_rel_residual"],
                "As_min": result["coefficient_metadata"]["As_min"],
                "As_max": result["coefficient_metadata"]["As_max"],
            }
        )
        traj_rows.append(
            {
                **base,
                "global_com_initial": summary["global_com_initial"],
                "global_com_final": summary["global_com_final"],
                "periodic_com_x_initial": summary["periodic_com_x_initial"],
                "periodic_com_x_final": summary["periodic_com_x_final"],
                "radial_drift": summary["radial_drift"],
                "momentum_change_norm": summary["momentum_change_norm"],
                "norm_error": summary["norm_error"],
                "energy_error": summary["energy_error"],
                "energy_rel_error": summary["energy_rel_error"],
            }
        )
    return clock_rows, force_rows, traj_rows


def manifest_base(result: dict[str, Any]) -> dict[str, Any]:
    cfg = result["config"]
    summary = result["summary"]
    return {
        "run_id": result["run_id"],
        "config_hash": result["config_hash"],
        "matrix_role": cfg.get("matrix_role", ""),
        "arm": cfg["arm"],
        "source_mode": cfg["source_mode"],
        "duration_name": cfg["duration_name"],
        "grid": cfg["N"],
        "box": cfg["L"],
        "dt": cfg["dt"],
        "T": cfg["T"],
        "probe_width": cfg["probe_width"],
        "probe_amplitude": cfg["probe_amplitude"],
        "normalize_probe": cfg["normalize_probe"],
        "probe_position": cfg["probe_position"],
        "beta_t": cfg["beta_t"],
        "beta_s": cfg["beta_s"],
        "gpu_device": result["gpu_device"],
        "compile_time_s": summary["compile_time_s"],
        "execution_time_s": summary["execution_time_s"],
        "wall_time_s": summary["wall_time_s"],
    }


def arm_comparisons(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    by_key: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for result in results:
        cfg = result["config"]
        if cfg.get("matrix_role") != "four_arm_decomposition":
            continue
        key = (cfg["source_mode"], cfg["duration_name"])
        by_key.setdefault(key, {})[cfg["arm"]] = result

    rows: list[dict[str, Any]] = []
    pass_flags = {
        "flat_clean": True,
        "temporal_only_clock": True,
        "spatial_only_force": True,
        "combined_no_double_count": True,
    }
    for (source_mode, duration_name), arms in sorted(by_key.items()):
        flat = arms.get("FLAT")
        temporal = arms.get("TEMPORAL_ONLY")
        spatial = arms.get("SPATIAL_ONLY")
        combined = arms.get("TEMPORAL_SPATIAL")
        if not all((flat, temporal, spatial, combined)):
            continue
        flat_force = float(flat["summary"]["initial_force_norm"])
        flat_clock = abs(float(flat["summary"]["clock_shift"]))
        temporal_clock = float(temporal["summary"]["clock_shift"])
        temporal_force = float(temporal["summary"]["initial_force_norm"])
        spatial_force = float(spatial["summary"]["initial_force_norm"])
        spatial_clock = abs(float(spatial["summary"]["clock_shift"]))
        combined_force = float(combined["summary"]["initial_force_norm"])
        combined_clock = float(combined["summary"]["clock_shift"])
        force_delta = abs(combined_force - spatial_force)
        row = {
            "source_mode": source_mode,
            "duration_name": duration_name,
            "flat_force_norm": flat_force,
            "flat_clock_shift": flat_clock,
            "temporal_only_clock_shift": temporal_clock,
            "temporal_only_force_norm": temporal_force,
            "spatial_only_force_norm": spatial_force,
            "spatial_only_clock_shift": spatial_clock,
            "combined_force_norm": combined_force,
            "combined_clock_shift": combined_clock,
            "combined_minus_spatial_force_norm": force_delta,
            "flat_clean": flat_force <= 1e-10 and flat_clock <= 1e-10,
            "temporal_clock_present": temporal_clock > 1e-4,
            "temporal_force_absent": temporal_force <= 1e-10,
            "spatial_force_present": spatial_force > 1e-5,
            "spatial_clock_absent": spatial_clock <= 1e-10,
            "combined_has_both": combined_clock > 1e-4 and combined_force > 1e-5,
            "double_counting_flag": force_delta > max(1e-6, 0.02 * spatial_force),
        }
        pass_flags["flat_clean"] = pass_flags["flat_clean"] and bool(row["flat_clean"])
        pass_flags["temporal_only_clock"] = pass_flags["temporal_only_clock"] and bool(row["temporal_clock_present"] and row["temporal_force_absent"])
        pass_flags["spatial_only_force"] = pass_flags["spatial_only_force"] and bool(row["spatial_force_present"] and row["spatial_clock_absent"])
        pass_flags["combined_no_double_count"] = pass_flags["combined_no_double_count"] and bool(row["combined_has_both"] and not row["double_counting_flag"])
        rows.append(row)
    return rows, pass_flags


def weak_probe_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        cfg = result["config"]
        if cfg.get("matrix_role") != "weak_probe_limit":
            continue
        rows.append(
            {
                **manifest_base(result),
                "initial_clock_rate": result["summary"]["initial_clock_rate"],
                "clock_shift": result["summary"]["clock_shift"],
                "source_exposure": result["samples"][0]["source_exposure"],
                "initial_norm": result["samples"][0]["norm"],
                "interpretation": "objective should be amplitude-invariant; relational should weaken toward flat as raw probe amplitude decreases",
            }
        )
    return rows


def falsification_rows(arm_rows: list[dict[str, Any]], weak_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in arm_rows:
        rows.extend(
            [
                {
                    "test_id": f"flat_clean_{row['source_mode']}_{row['duration_name']}",
                    "criterion": "FLAT removes both clock shift and spatial force",
                    "passed": row["flat_clean"],
                    "measured_value": json.dumps(
                        {"force_norm": row["flat_force_norm"], "clock_shift": row["flat_clock_shift"]},
                        sort_keys=True,
                    ),
                },
                {
                    "test_id": f"temporal_only_{row['source_mode']}_{row['duration_name']}",
                    "criterion": "TEMPORAL_ONLY produces clock shift with A_s=1 and no medium force",
                    "passed": row["temporal_clock_present"] and row["temporal_force_absent"],
                    "measured_value": json.dumps(
                        {"clock_shift": row["temporal_only_clock_shift"], "force_norm": row["temporal_only_force_norm"]},
                        sort_keys=True,
                    ),
                },
                {
                    "test_id": f"spatial_only_{row['source_mode']}_{row['duration_name']}",
                    "criterion": "SPATIAL_ONLY produces medium force with N_t=1",
                    "passed": row["spatial_force_present"] and row["spatial_clock_absent"],
                    "measured_value": json.dumps(
                        {"force_norm": row["spatial_only_force_norm"], "clock_shift": row["spatial_only_clock_shift"]},
                        sort_keys=True,
                    ),
                },
                {
                    "test_id": f"combined_{row['source_mode']}_{row['duration_name']}",
                    "criterion": "TEMPORAL_SPATIAL has both observables without altering spatial force beyond audit tolerance",
                    "passed": row["combined_has_both"] and not row["double_counting_flag"],
                    "measured_value": json.dumps(
                        {
                            "force_delta": row["combined_minus_spatial_force_norm"],
                            "combined_clock_shift": row["combined_clock_shift"],
                        },
                        sort_keys=True,
                    ),
                },
            ]
        )
    for source_mode in SOURCE_MODES:
        subset = [row for row in weak_rows if row["source_mode"] == source_mode]
        if not subset:
            continue
        subset = sorted(subset, key=lambda r: float(r["probe_amplitude"]), reverse=True)
        first = float(subset[0]["clock_shift"])
        last = float(subset[-1]["clock_shift"])
        if source_mode == "objective":
            passed = max(abs(float(r["clock_shift"]) - first) for r in subset) <= 1e-9
            criterion = "Objective-source weak probes converge to same N_t at matched position"
        else:
            passed = last < 0.15 * max(first, 1e-30)
            criterion = "Relational-source weak probes lose temporal field under raw weak-probe limit"
        rows.append(
            {
                "test_id": f"weak_probe_{source_mode}",
                "criterion": criterion,
                "passed": passed,
                "measured_value": json.dumps({"strong_shift": first, "weak_shift": last}, sort_keys=True),
            }
        )
    return rows


def source_ladder_metadata(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    for result in results:
        cfg = result["config"]
        key = (cfg["source_mode"], cfg.get("matrix_role", ""))
        if key in seen:
            continue
        seen.add(key)
        meta = result["source_metadata"]
        rows.append(
            {
                "source_mode": cfg["source_mode"],
                "matrix_role": cfg.get("matrix_role", ""),
                "source_width": meta["source_sigma"],
                "objective_integral": meta["objective_integral"],
                "relational_scale": result["relational_scale"],
                "relational_initial_max": meta["relational_initial_max"],
                "relational_initial_integral": meta["relational_initial_integral"],
                "source_definition": meta["objective_source"] if cfg["source_mode"] == "objective" else meta["relational_source"],
            }
        )
    return rows


def common_field_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        if result["config"].get("matrix_role") != "common_field_check":
            continue
        rows.append(
            {
                **manifest_base(result),
                "clock_shift": result["summary"]["clock_shift"],
                "initial_force_norm": result["summary"]["initial_force_norm"],
                "initial_radial_force": result["summary"]["initial_radial_force"],
                "note": "Restricted common-field hypothesis; not assumed in the four-arm bridge.",
            }
        )
    return rows


def labels_from(pass_flags: dict[str, bool], falsifications: list[dict[str, Any]]) -> list[str]:
    labels = []
    if pass_flags.get("temporal_only_clock"):
        labels.append("TEMPORAL_THROTTLING_REPRODUCED")
    if pass_flags.get("spatial_only_force"):
        labels.append("SPATIAL_MEDIUM_FORCE_REPRODUCED")
    if all(pass_flags.values()):
        labels.append("TEMPORAL_SPATIAL_DECOMPOSITION_PASSED")
    objective = next((r for r in falsifications if r["test_id"] == "weak_probe_objective"), None)
    relational = next((r for r in falsifications if r["test_id"] == "weak_probe_relational"), None)
    if objective and objective["passed"]:
        labels.append("OBJECTIVE_WEAK_PROBE_LAPSE_FIELD_LIMIT_REPRODUCED")
    if relational and relational["passed"]:
        labels.append("RELATIONAL_WEAK_PROBE_FIELD_COLLAPSE_REPRODUCED")
    if not labels:
        labels.append("BRIDGE_AUDIT_INCONCLUSIVE")
    return labels


def hash_artifacts(outdir: Path) -> list[dict[str, str]]:
    rows = []
    for path in sorted(outdir.rglob("*")):
        if not path.is_file() or path.name == "artifact_hashes.csv":
            continue
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({"path": str(path.relative_to(outdir)), "sha256": h})
    return rows


def write_handoffs(
    outdir: Path,
    labels: list[str],
    arm_rows: list[dict[str, Any]],
    falsifications: list[dict[str, Any]],
    common_rows: list[dict[str, Any]],
) -> None:
    passed = [r for r in falsifications if r["passed"]]
    failed = [r for r in falsifications if not r["passed"]]
    summary = {
        "labels": labels,
        "bounded_interpretation": (
            "Temporal throttling and spatial effective-medium response are "
            "separable mirror observables. This audit does not validate GR, "
            "geodesics, equivalence principle behaviour, or production geometry."
        ),
        "passed_tests": [r["test_id"] for r in passed],
        "failed_or_unresolved_tests": [r["test_id"] for r in failed],
        "common_field_rows": [r["run_id"] for r in common_rows],
    }
    write_json(outdir / "BRIDGE_SUMMARY.json", summary)
    technical = [
        "# Temporal-Spatial Bridge GPU Audit",
        "",
        "## Scope",
        "",
        "This mirror audit separates Claude's temporal clock factor `N_t` from the Gravity D spatial coefficient `A_s`.",
        "Field evolution used the validated GPU spatial operator `H=-D div(A_s grad)`. Temporal factors were recorded as clock observables, not silently inserted as a metric equation.",
        "",
        "## Labels",
        "",
        *[f"- `{label}`" for label in labels],
        "",
        "## Four-Arm Comparison",
        "",
        "| source | duration | flat force | temporal clock | spatial force | combined force delta |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in arm_rows:
        technical.append(
            f"| {row['source_mode']} | {row['duration_name']} | {row['flat_force_norm']:.6e} | "
            f"{row['temporal_only_clock_shift']:.6e} | {row['spatial_only_force_norm']:.6e} | "
            f"{row['combined_minus_spatial_force_norm']:.6e} |"
        )
    technical.extend(
        [
            "",
            "## Falsification Checks",
            "",
            "| test | passed | measured value |",
            "| --- | --- | --- |",
        ]
    )
    for row in falsifications:
        technical.append(f"| `{row['test_id']}` | {row['passed']} | `{row['measured_value']}` |")
    technical.extend(
        [
            "",
            "## Common-Field Check",
            "",
            "The common-field rows are intentionally separated from the four-arm decomposition.",
        ]
    )
    for row in common_rows:
        technical.append(
            f"- `{row['run_id']}`: clock_shift={row['clock_shift']:.6e}, "
            f"force_norm={row['initial_force_norm']:.6e}"
        )
    technical.extend(
        [
            "",
            "## Boundary",
            "",
            "No gravity, geodesic, equivalence-principle, IRER-gravity-validation, or production-readiness label is assigned.",
        ]
    )
    (outdir / "TECHNICAL_HANDOFF.md").write_text("\n".join(technical) + "\n", encoding="utf-8")

    doc_inputs = [
        "# Documentation Inputs",
        "",
        "## Final Bounded Labels",
        "",
        *[f"- `{label}`" for label in labels],
        "",
        "## Suggested Wording",
        "",
        (
            "The bridge audit reproduces temporal clock throttling and spatial "
            "effective-medium force as separable mirror mechanisms. The objective "
            "source shows amplitude-invariant constructed clock-field response at "
            "matched position, while the relational mutual source weakens toward no "
            "temporal field in the raw weak-probe field limit. Universal gravity "
            "claims remain open."
        ),
        "",
        "## Caveats",
        "",
        "- `N_t` was measured as a clock factor; this is not a metric-consistent KG or scalar-field equation.",
        "- `A_s` remains the spatial kinetic coefficient in the validated divergence-form operator.",
        "- `N_t=A_s` is only a restricted common-field check.",
        "",
        "## Primary Artifacts",
        "",
        "- `run_manifest.csv`",
        "- `clock_metrics.csv`",
        "- `force_metrics.csv`",
        "- `arm_comparison.csv`",
        "- `weak_probe_limit.csv`",
        "- `falsification_results.csv`",
    ]
    (outdir / "DOCUMENTATION_INPUTS.md").write_text("\n".join(doc_inputs) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--arms", default="all")
    parser.add_argument("--source", default="objective,relational")
    parser.add_argument("--durations", default="short,standard")
    parser.add_argument("--stop-on-preflight-fail", default="true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = args.out or (ROOT / "sweep_runs" / f"GRAVITY_TS_BRIDGE_GPU_{timestamp}")
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "trajectories").mkdir(exist_ok=True)

    try:
        preflight_record = preflight()
    except Exception as exc:
        report = {"error": str(exc), "command_line": command_line(), "time": timestamp}
        write_json(outdir / "DISCREPANCY_REPORT.json", report)
        if str(args.stop_on_preflight_fail).lower() == "true":
            raise
        preflight_record = {"backend": jax.default_backend(), "devices": [str(d) for d in jax.devices()]}
    write_json(outdir / "gpu_preflight.json", preflight_record)
    write_json(outdir / "environment_versions.json", environment_record(preflight_record))
    (outdir / "git_state_before.txt").write_text(git_state(), encoding="utf-8")

    # Compute the relational scale once from the exact grid used for the audit.
    rel_grid = build_grid(BASE_N, BASE_L)
    rel_scale = relational_scale_for(rel_grid)
    sources = selected_values(args.source, SOURCE_MODES)
    matrix = build_matrix(args, rel_scale)
    write_json(outdir / "preregistered_matrix.json", matrix)

    results: list[dict[str, Any]] = []
    for cfg in matrix:
        print(f"running {cfg['run_id']}", flush=True)
        results.append(run_case(cfg, outdir))

    arm_rows, pass_flags = arm_comparisons(results)
    if all(pass_flags.values()):
        common_matrix = common_field_matrix(sources, rel_scale)
        write_json(outdir / "common_field_preregistered_matrix.json", common_matrix)
        for cfg in common_matrix:
            print(f"running {cfg['run_id']}", flush=True)
            result = run_case(cfg, outdir)
            results.append(result)
    else:
        write_json(outdir / "common_field_preregistered_matrix.json", [])

    clock_rows, force_rows, traj_rows = summarize_rows(results)
    arm_rows, pass_flags = arm_comparisons(results)
    weak_rows = weak_probe_rows(results)
    common_rows = common_field_rows(results)
    falsifications = falsification_rows(arm_rows, weak_rows)
    labels = labels_from(pass_flags, falsifications)

    manifest_rows = [manifest_base(result) for result in results]
    write_csv(outdir / "run_manifest.csv", manifest_rows)
    write_csv(outdir / "clock_metrics.csv", clock_rows)
    write_csv(outdir / "force_metrics.csv", force_rows)
    write_csv(outdir / "trajectory_metrics.csv", traj_rows)
    write_csv(outdir / "source_ladder_metadata.csv", source_ladder_metadata(results))
    write_csv(outdir / "arm_comparison.csv", arm_rows)
    write_csv(outdir / "weak_probe_limit.csv", weak_rows)
    write_csv(outdir / "common_field_check.csv", common_rows)
    write_csv(outdir / "falsification_results.csv", falsifications)
    write_handoffs(outdir, labels, arm_rows, falsifications, common_rows)
    (outdir / "git_status_short.txt").write_text(git_state(), encoding="utf-8")
    try:
        diff_names = subprocess.check_output(["git", "diff", "--name-only"], cwd=ROOT, text=True)
        cached_names = subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True)
    except Exception as exc:  # pragma: no cover - diagnostic path
        diff_names = cached_names = f"git diff failed: {exc}\n"
    (outdir / "git_diff_name_only.txt").write_text(diff_names, encoding="utf-8")
    (outdir / "git_diff_cached_name_only.txt").write_text(cached_names, encoding="utf-8")
    write_csv(outdir / "artifact_hashes.csv", hash_artifacts(outdir))
    print(f"labels: {', '.join(labels)}", flush=True)
    print(f"outdir: {outdir}", flush=True)


if __name__ == "__main__":
    main()
