"""Bounded GPU robustness battery for the spatial effective-medium D result.

This driver reuses the validated GPU mirror primitives in
gravity_D_neutral_probe_gpu.py.  It does not alter the accepted operator:

    i d_t psi = i D div(N grad psi),  H = -D div(N grad)

Run in the established WSL2 JAX venv:
    . ~/jax_irer/bin/activate
    python /mnt/f/quantule_mapper/jax_scout/gravity_D_robustness_gpu.py \
      --out /mnt/f/quantule_mapper/sweep_runs/GRAVITY_D_ROBUSTNESS_GPU_<timestamp>
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

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_FOR_IMPORT))

from jax_scout.gravity_D_neutral_probe_gpu import (
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
SOURCE_SIGMA = 1.5
REFERENCE_DRIFT = -1.38462041e-02


def jscalar(x: jnp.ndarray) -> float:
    return float(np.asarray(x))


def source_profile(grid: dict[str, object], cfg: dict[str, object]) -> jnp.ndarray:
    X, Y, Z = grid["X"], grid["Y"], grid["Z"]
    ox, oy, oz = cfg.get("source_offset", (0.0, 0.0, 0.0))
    dx, dy, dz = X - float(ox), Y - float(oy), Z - float(oz)
    r2 = dx * dx + dy * dy + dz * dz
    width = float(cfg.get("source_width", SOURCE_SIGMA))
    family = str(cfg.get("source_family", "rho2"))
    if family == "rho2":
        rho = jnp.exp(-(r2 / (2.0 * width * width)))
        S = rho * rho
    elif family == "rho":
        S = jnp.exp(-(r2 / (2.0 * width * width)))
    elif family == "supergaussian4":
        r = jnp.sqrt(r2)
        S = jnp.exp(-((r / width) ** 4))
    else:
        raise ValueError(f"unknown source family: {family}")
    return S / (jnp.max(S) + 1e-30)


def coefficient_from_source(S: jnp.ndarray, beta: float) -> jnp.ndarray:
    return 1.0 / (1.0 + beta * S)


def coefficient_diagnostics(S: jnp.ndarray, Nf: jnp.ndarray, grid: dict[str, object]) -> dict[str, float]:
    X, Y, Z, dV = grid["X"], grid["Y"], grid["Z"], grid["dV"]
    deficit = 1.0 - Nf
    mass = jnp.sum(deficit) * dV
    r2 = X * X + Y * Y + Z * Z
    second = (jnp.sum(r2 * deficit) * dV) / (mass + 1e-30)
    return {
        "N_min": jscalar(jnp.min(Nf)),
        "N_max": jscalar(jnp.max(Nf)),
        "integrated_deficit": jscalar(mass),
        "deficit_second_moment": jscalar(second),
        "S_max": jscalar(jnp.max(S)),
    }


def deficit_for_family(grid: dict[str, object], family: str, width: float, beta: float) -> float:
    S = source_profile(grid, {"source_family": family, "source_width": width})
    Nf = coefficient_from_source(S, beta)
    return coefficient_diagnostics(S, Nf, grid)["integrated_deficit"]


def match_width_for_deficit(grid: dict[str, object], family: str, target_deficit: float, beta: float) -> float:
    if family == "rho2":
        return SOURCE_SIGMA
    lo, hi = 0.2, 5.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if deficit_for_family(grid, family, mid, beta) < target_deficit:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def initial_probe(grid: dict[str, object], cfg: dict[str, object]) -> jnp.ndarray:
    X, Y, Z = grid["X"], grid["Y"], grid["Z"]
    px, py, pz = cfg.get("probe_position", (4.0, 0.0, 0.0))
    sig = float(cfg.get("probe_width", 1.0))
    amp = float(cfg.get("probe_amplitude", 1.0))
    kx, ky, kz = cfg.get("probe_carrier", (0.0, 0.0, 0.0))
    dx, dy, dz = X - float(px), Y - float(py), Z - float(pz)
    psi = amp * jnp.exp(-((dx * dx + dy * dy + dz * dz) / (2.0 * sig * sig)))
    if bool(cfg.get("normalize_probe", True)):
        target = jnp.asarray(float(cfg.get("target_norm", 1.0)), dtype=jnp.float64)
        norm = jnp.sum(jnp.abs(psi) ** 2) * grid["dV"]
        psi = psi * jnp.sqrt(target / (norm + 1e-30))
    phase = jnp.exp(1j * (float(kx) * X + float(ky) * Y + float(kz) * Z))
    return (psi * phase).astype(jnp.complex128)


@jax.jit
def vector_diagnostics(
    psi: jnp.ndarray,
    Nf: jnp.ndarray,
    grid: dict[str, object],
    probe_pos: jnp.ndarray,
    radial_hat: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    X, Y, Z, dV = grid["X"], grid["Y"], grid["Z"], grid["dV"]
    rho = jnp.abs(psi) ** 2
    dx, dy, dz = X - probe_pos[0], Y - probe_pos[1], Z - probe_pos[2]
    radial_coord = dx * radial_hat[0] + dy * radial_hat[1] + dz * radial_hat[2]
    # Match Claude's accepted COM diagnostic: a slab around the probe along
    # the tested axis. For x-axis runs this is exactly abs(X - x_p) < 6.
    window = jnp.abs(radial_coord) < 6.0
    wrho = rho * window
    wmass = jnp.sum(wrho) * dV
    com = jnp.array(
        [
            (jnp.sum(X * wrho) * dV) / (wmass + 1e-30),
            (jnp.sum(Y * wrho) * dV) / (wmass + 1e-30),
            (jnp.sum(Z * wrho) * dV) / (wmass + 1e-30),
        ],
        dtype=jnp.float64,
    )
    gx, gy, gz = deriv(psi, grid["ikx"]), deriv(psi, grid["iky"]), deriv(psi, grid["ikz"])
    grad_energy = jnp.abs(gx) ** 2 + jnp.abs(gy) ** 2 + jnp.abs(gz) ** 2
    momentum = jnp.array(
        [
            jnp.sum(jnp.imag(jnp.conj(psi) * gx)) * dV,
            jnp.sum(jnp.imag(jnp.conj(psi) * gy)) * dV,
            jnp.sum(jnp.imag(jnp.conj(psi) * gz)) * dV,
        ],
        dtype=jnp.float64,
    )
    dNx, dNy, dNz = jnp.real(deriv(Nf, grid["ikx"])), jnp.real(deriv(Nf, grid["iky"])), jnp.real(deriv(Nf, grid["ikz"]))
    force_analytic = -D * jnp.array(
        [
            jnp.sum(dNx * grad_energy) * dV,
            jnp.sum(dNy * grad_energy) * dV,
            jnp.sum(dNz * grad_energy) * dV,
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
    grad_norm = jnp.sum(grad_energy) * dV
    norm = jnp.sum(rho) * dV
    radial_force = jnp.dot(force_analytic, radial_hat)
    transverse = jnp.linalg.norm(force_analytic - radial_force * radial_hat)
    return {
        "com": com,
        "momentum": momentum,
        "norm": norm,
        "force_analytic": force_analytic,
        "force_rhs": force_rhs,
        "radial_force": radial_force,
        "transverse_force": transverse,
        "gradient_energy": grad_norm,
    }


def vector_to_list(x: jnp.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(x)]


def radial_hat_from_position(pos: tuple[float, float, float], source_offset: tuple[float, float, float]) -> tuple[float, float, float]:
    v = np.asarray(pos, dtype=float) - np.asarray(source_offset, dtype=float)
    n = float(np.linalg.norm(v))
    if n == 0:
        raise ValueError("probe position equals source offset")
    return tuple((v / n).tolist())


def run_case(cfg: dict[str, object], outdir: Path) -> dict[str, object]:
    started = time.time()
    grid = build_grid(int(cfg["N"]), float(cfg["L"]))
    S = source_profile(grid, cfg)
    Nf = coefficient_from_source(S, float(cfg["beta"]))
    coeff = coefficient_diagnostics(S, Nf, grid)
    psi = initial_probe(grid, cfg)
    dt = jnp.asarray(float(cfg["dt"]), dtype=jnp.float64)
    nsteps = int(round(float(cfg["T"]) / float(cfg["dt"])))
    nsnap = int(cfg.get("nsnap", 6))
    every = max(1, nsteps // nsnap) if nsteps else 1
    chunks = nsteps // every if nsteps else 0
    tail = nsteps - chunks * every
    probe_pos = jnp.asarray(cfg["probe_position"], dtype=jnp.float64)
    radial_hat = jnp.asarray(cfg["radial_hat"], dtype=jnp.float64)

    samples = []
    d0 = vector_diagnostics(psi, Nf, grid, probe_pos, radial_hat)
    psi1 = rk4_step(psi, Nf, grid, dt)
    d1 = vector_diagnostics(psi1, Nf, grid, probe_pos, radial_hat)
    force_fd0 = (d1["momentum"] - d0["momentum"]) / dt
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), {**d0, "force_fd": force_fd0})
    samples.append(sample_row(0.0, d0, force_fd0))

    for i in range(chunks):
        psi = evolve_n(psi, Nf, grid, dt, every)
        di = vector_diagnostics(psi, Nf, grid, probe_pos, radial_hat)
        psi_next = rk4_step(psi, Nf, grid, dt)
        di_next = vector_diagnostics(psi_next, Nf, grid, probe_pos, radial_hat)
        force_fd = (di_next["momentum"] - di["momentum"]) / dt
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), {**di, "force_fd": force_fd})
        samples.append(sample_row((i + 1) * every * float(cfg["dt"]), di, force_fd))
    if tail:
        psi = evolve_n(psi, Nf, grid, dt, tail)
        psi.block_until_ready()

    final = vector_diagnostics(psi, Nf, grid, probe_pos, radial_hat)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), final)
    final_row = sample_row(float(cfg["T"]), final, samples[-1]["force_fd"])
    com0 = np.asarray(samples[0]["com"])
    comT = np.asarray(final_row["com"])
    radial_drift = float(np.dot(comT - com0, np.asarray(cfg["radial_hat"], dtype=float)))
    momentum_change = np.asarray(final_row["momentum"]) - np.asarray(samples[0]["momentum"])
    force0 = np.asarray(samples[0]["force_analytic"])
    rhs0 = np.asarray(samples[0]["force_rhs"])
    fd0 = np.asarray(samples[0]["force_fd"])
    force_norm = float(np.linalg.norm(force0))
    rhs_abs = float(np.linalg.norm(rhs0 - force0))
    fd_abs = float(np.linalg.norm(fd0 - force0))
    rhs_res = rhs_abs / force_norm if force_norm > 1e-20 else rhs_abs
    fd_res = fd_abs / force_norm if force_norm > 1e-20 else fd_abs
    result = {
        "config": cfg,
        "config_hash": config_hash(cfg),
        "git_commit": git_commit(),
        "gpu_backend": jax.default_backend(),
        "gpu_device": str(jax.devices()[0]),
        "coefficient": coeff,
        "operator": "H=-D div(N grad)",
        "D": D,
        "samples": samples,
        "final": final_row,
        "metrics": {
            "initial_force_analytic": vector_to_list(jnp.asarray(force0)),
            "initial_force_rhs": vector_to_list(jnp.asarray(rhs0)),
            "initial_force_fd": vector_to_list(jnp.asarray(fd0)),
            "radial_force": float(samples[0]["radial_force"]),
            "transverse_force": float(samples[0]["transverse_force"]),
            "force_contract_residual": rhs_res,
            "force_contract_residual_abs": rhs_abs,
            "force_fd_residual": fd_res,
            "force_fd_residual_abs": fd_abs,
            "final_radial_drift": radial_drift,
            "momentum_change": momentum_change.tolist(),
            "norm_ratio": float(final_row["norm"] / (samples[0]["norm"] + 1e-30)),
            "gradient_energy": float(samples[0]["gradient_energy"]),
            "wall_time_s": time.time() - started,
        },
    }
    (outdir / f"{cfg['run_id']}.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    np.savez_compressed(
        outdir / f"{cfg['run_id']}_trajectory.npz",
        t=np.asarray([s["t"] for s in samples], dtype=float),
        com=np.asarray([s["com"] for s in samples], dtype=float),
        momentum=np.asarray([s["momentum"] for s in samples], dtype=float),
        force_analytic=np.asarray([s["force_analytic"] for s in samples], dtype=float),
        force_rhs=np.asarray([s["force_rhs"] for s in samples], dtype=float),
        force_fd=np.asarray([s["force_fd"] for s in samples], dtype=float),
        norm=np.asarray([s["norm"] for s in samples], dtype=float),
        radial_force=np.asarray([s["radial_force"] for s in samples], dtype=float),
    )
    return result


def sample_row(t: float, diag: dict[str, jnp.ndarray], force_fd: jnp.ndarray) -> dict[str, object]:
    return {
        "t": float(t),
        "com": vector_to_list(diag["com"]),
        "momentum": vector_to_list(diag["momentum"]),
        "norm": jscalar(diag["norm"]),
        "force_analytic": vector_to_list(diag["force_analytic"]),
        "force_rhs": vector_to_list(diag["force_rhs"]),
        "force_fd": vector_to_list(force_fd),
        "radial_force": jscalar(diag["radial_force"]),
        "transverse_force": jscalar(diag["transverse_force"]),
        "gradient_energy": jscalar(diag["gradient_energy"]),
    }


def base_cfg() -> dict[str, object]:
    pos = (4.0, 0.0, 0.0)
    src = (0.0, 0.0, 0.0)
    return {
        "N": BASE_N,
        "L": BASE_L,
        "dt": BASE_DT,
        "T": 1.0,
        "nsnap": 6,
        "source_family": "rho2",
        "source_width": SOURCE_SIGMA,
        "source_offset": src,
        "beta": 1.0,
        "probe_position": pos,
        "probe_radius": 4.0,
        "radial_hat": radial_hat_from_position(pos, src),
        "probe_width": 1.0,
        "probe_amplitude": 1.0,
        "probe_carrier": (0.0, 0.0, 0.0),
        "normalize_probe": True,
        "target_norm": 1.0,
    }


def build_matrix(outdir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grid = build_grid(BASE_N, BASE_L)
    baseline_S = source_profile(grid, base_cfg())
    baseline_N = coefficient_from_source(baseline_S, 1.0)
    baseline_coeff = coefficient_diagnostics(baseline_S, baseline_N, grid)
    target_def = baseline_coeff["integrated_deficit"]
    beta = 1.0
    source_rows = []
    widths = {
        fam: match_width_for_deficit(grid, fam, target_def, beta)
        for fam in ("rho2", "rho", "supergaussian4")
    }
    for fam, width in widths.items():
        S = source_profile(grid, {"source_family": fam, "source_width": width})
        Nf = coefficient_from_source(S, beta)
        cd = coefficient_diagnostics(S, Nf, grid)
        source_rows.append({"source_family": fam, "matched_width": width, **cd})

    matrix: list[dict[str, object]] = []

    r0 = {**base_cfg(), "run_id": "R0_reference_rho2_N96_T4", "stage": "R0", "T": 4.0}
    matrix.append(r0)
    r0_null = {**base_cfg(), "run_id": "R0_flat_null_N96_T4", "stage": "R0", "T": 4.0, "beta": 0.0}
    matrix.append(r0_null)

    for fam, width in widths.items():
        matrix.append({**base_cfg(), "run_id": f"R1_source_{fam}", "stage": "R1", "source_family": fam, "source_width": width})
    matrix.append({**base_cfg(), "run_id": "R1_flat_null", "stage": "R1", "beta": 0.0})

    for nmin in (1.0, 0.90, 0.80, 0.65, 0.50):
        beta_n = 0.0 if nmin == 1.0 else 1.0 / nmin - 1.0
        matrix.append({**base_cfg(), "run_id": f"R2_Nmin_{nmin:.2f}".replace(".", "p"), "stage": "R2", "beta": beta_n})

    r = 4.0
    orientations = [
        ("x", (r, 0.0, 0.0)),
        ("y", (0.0, r, 0.0)),
        ("z", (0.0, 0.0, r)),
        ("diag", (r / math.sqrt(3.0), r / math.sqrt(3.0), r / math.sqrt(3.0))),
    ]
    for name, pos in orientations:
        matrix.append({**base_cfg(), "run_id": f"R3_orientation_{name}", "stage": "R3", "probe_position": pos, "radial_hat": radial_hat_from_position(pos, (0.0, 0.0, 0.0))})
    off = (0.37, -0.23, 0.41)
    pos = (off[0] + 4.0, off[1], off[2])
    matrix.append({**base_cfg(), "run_id": "R3_translated_pair", "stage": "R3", "source_offset": off, "probe_position": pos, "radial_hat": radial_hat_from_position(pos, off)})

    for amp in (0.5, 1.0, 2.0):
        matrix.append({**base_cfg(), "run_id": f"R4_amplitude_{amp:g}".replace(".", "p"), "stage": "R4", "probe_amplitude": amp, "normalize_probe": False})

    for sigma in (0.5, 0.75, 1.0, 1.25, 1.6):
        matrix.append({**base_cfg(), "run_id": f"R5_width_{sigma:g}".replace(".", "p"), "stage": "R5", "probe_width": sigma})

    for kx in (-0.2, 0.0, 0.2):
        kid = f"{kx:+.1f}".replace("+", "p").replace("-", "m").replace(".", "p")
        matrix.append({**base_cfg(), "run_id": f"R6_carrier_{kid}", "stage": "R6", "probe_carrier": (kx, 0.0, 0.0)})
        matrix.append({**base_cfg(), "run_id": f"R6_free_{kid}", "stage": "R6", "probe_carrier": (kx, 0.0, 0.0), "beta": 0.0})

    (outdir / "preregistered_matrix.json").write_text(json.dumps(matrix, indent=2, default=float), encoding="utf-8")
    (outdir / "source_matching.json").write_text(json.dumps(source_rows, indent=2, default=float), encoding="utf-8")
    return matrix, source_rows


def flatten_row(result: dict[str, object], net_drift: float | None = None) -> dict[str, object]:
    cfg = result["config"]
    m = result["metrics"]
    c = result["coefficient"]
    force = m["initial_force_analytic"]
    rhs = m["initial_force_rhs"]
    fd = m["initial_force_fd"]
    return {
        "run_id": cfg["run_id"],
        "stage": cfg["stage"],
        "source_family": cfg["source_family"],
        "source_width": cfg["source_width"],
        "beta": cfg["beta"],
        "N_min": c["N_min"],
        "integrated_deficit": c["integrated_deficit"],
        "probe_position": json.dumps(cfg["probe_position"]),
        "probe_radius": cfg["probe_radius"],
        "probe_width": cfg["probe_width"],
        "probe_amplitude": cfg["probe_amplitude"],
        "probe_carrier": json.dumps(cfg["probe_carrier"]),
        "grid": cfg["N"],
        "box": cfg["L"],
        "dt": cfg["dt"],
        "T": cfg["T"],
        "initial_force_analytic": json.dumps(force),
        "initial_force_rhs": json.dumps(rhs),
        "initial_force_fd": json.dumps(fd),
        "radial_force": m["radial_force"],
        "transverse_force": m["transverse_force"],
        "force_contract_residual": m["force_contract_residual"],
        "force_fd_residual": m["force_fd_residual"],
        "final_net_radial_drift": net_drift if net_drift is not None else m["final_radial_drift"],
        "raw_final_radial_drift": m["final_radial_drift"],
        "norm_ratio": m["norm_ratio"],
        "gradient_energy": m["gradient_energy"],
        "gpu_device": result["gpu_device"],
        "config_hash": result["config_hash"],
    }


def compute_net_drifts(results: list[dict[str, object]]) -> dict[str, float]:
    by_id = {r["config"]["run_id"]: r for r in results}
    flat = by_id.get("R0_flat_null_N96_T4")
    flat_short = by_id.get("R1_flat_null")
    nets: dict[str, float] = {}
    for r in results:
        rid = r["config"]["run_id"]
        raw = float(r["metrics"]["final_radial_drift"])
        if rid.startswith("R6_carrier_"):
            free_id = rid.replace("R6_carrier_", "R6_free_")
            nets[rid] = raw - float(by_id[free_id]["metrics"]["final_radial_drift"])
        elif r["config"]["stage"] == "R0" and flat is not None and rid != "R0_flat_null_N96_T4":
            nets[rid] = raw - float(flat["metrics"]["final_radial_drift"])
        elif flat_short is not None and rid != "R1_flat_null":
            # Free zero-carrier drift is numerically tiny, but subtract the short null for consistency.
            nets[rid] = raw - float(flat_short["metrics"]["final_radial_drift"])
        else:
            nets[rid] = raw
    return nets


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(outdir: Path, rows: list[dict[str, object]]) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (outdir / "plot_warning.txt").write_text(f"matplotlib unavailable: {exc}\n", encoding="utf-8")
        return []

    plot_paths: list[str] = []

    def savefig(name: str) -> None:
        path = outdir / name
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        plot_paths.append(str(path))

    r1 = [r for r in rows if r["stage"] == "R1" and r["beta"] != 0.0]
    plt.figure()
    plt.bar([r["source_family"] for r in r1], [r["radial_force"] for r in r1])
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("initial radial force")
    savefig("radial_force_by_source.png")

    r2 = [r for r in rows if r["stage"] == "R2"]
    plt.figure()
    plt.plot([r["N_min"] for r in r2], [r["radial_force"] for r in r2], marker="o")
    plt.gca().invert_xaxis()
    plt.xlabel("N_min")
    plt.ylabel("initial radial force")
    savefig("radial_force_by_Nmin.png")

    r5 = [r for r in rows if r["stage"] == "R5"]
    plt.figure()
    plt.plot([r["probe_width"] for r in r5], [r["radial_force"] for r in r5], marker="o", label="radial force")
    plt.plot([r["probe_width"] for r in r5], [r["gradient_energy"] for r in r5], marker="s", label="gradient energy")
    plt.xlabel("probe sigma")
    plt.legend()
    savefig("width_force_gradient_energy.png")

    r3 = [r for r in rows if r["stage"] == "R3" and "orientation" in r["run_id"]]
    plt.figure()
    plt.bar([r["run_id"].replace("R3_orientation_", "") for r in r3], [r["radial_force"] for r in r3], label="radial")
    plt.plot([r["run_id"].replace("R3_orientation_", "") for r in r3], [r["transverse_force"] for r in r3], "o", label="transverse")
    plt.legend()
    plt.ylabel("force")
    savefig("orientation_radial_transverse_force.png")
    return plot_paths


def verdict(rows: list[dict[str, object]], reference_ok: bool) -> str:
    if not reference_ok:
        return "D_SPATIAL_EFFECTIVE_MEDIUM_ROBUSTNESS_FAILED"
    signal_rows = [r for r in rows if r["beta"] != 0.0 and not r["run_id"].startswith("R6_free_")]
    direction_ok = all(float(r["radial_force"]) < 0.0 for r in signal_rows if r["stage"] != "R3" or "translated" not in r["run_id"])
    force_ok = all(float(r["force_contract_residual"]) < 2e-4 for r in rows)
    norm_ok = all(abs(float(r["norm_ratio"]) - 1.0) < 1e-8 for r in rows)
    nulls = [r for r in rows if float(r["beta"]) == 0.0]
    null_ok = all(abs(float(r["radial_force"])) < 1e-10 for r in nulls)
    amp = [r for r in rows if r["stage"] == "R4"]
    amp_vals = [float(r["radial_force"]) / (float(r["probe_amplitude"]) ** 2) for r in amp]
    amp_ok = max(amp_vals) - min(amp_vals) < 1e-10 if amp_vals else True
    if direction_ok and force_ok and norm_ok and null_ok and amp_ok:
        return "D_SPATIAL_EFFECTIVE_MEDIUM_ROBUSTNESS_FIRST_PASS"
    if direction_ok and force_ok and norm_ok:
        return "D_SPATIAL_EFFECTIVE_MEDIUM_ROBUSTNESS_PARTIAL"
    return "D_SPATIAL_EFFECTIVE_MEDIUM_ROBUSTNESS_FAILED"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "sweep_runs" / f"GRAVITY_D_ROBUSTNESS_GPU_{time.strftime('%Y%m%d_%H%M%S')}"))
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    pf = preflight()
    (outdir / "gpu_preflight.json").write_text(json.dumps(pf, indent=2), encoding="utf-8")
    matrix, source_rows = build_matrix(outdir)

    results = []
    for cfg in matrix:
        print(f"[run] {cfg['run_id']} stage={cfg['stage']} beta={cfg['beta']} source={cfg['source_family']} width={cfg['probe_width']} carrier={cfg['probe_carrier']}", flush=True)
        res = run_case(cfg, outdir)
        results.append(res)
        print(
            f"      Fr={res['metrics']['radial_force']:+.6e} drift={res['metrics']['final_radial_drift']:+.6e} "
            f"norm={res['metrics']['norm_ratio']:.12f} rhs={res['metrics']['force_contract_residual']:.3e}",
            flush=True,
        )
        if cfg["run_id"] == "R0_reference_rho2_N96_T4":
            drift = float(res["metrics"]["final_radial_drift"])
            if abs(drift - REFERENCE_DRIFT) >= 5e-5:
                raise RuntimeError(
                    f"R0 reference mismatch: drift={drift:+.9e}, expected {REFERENCE_DRIFT:+.9e}"
                )

    nets = compute_net_drifts(results)
    rows = [flatten_row(r, nets.get(r["config"]["run_id"])) for r in results]
    write_csv(outdir / "robustness_results.csv", rows)
    write_csv(outdir / "source_matching.csv", source_rows)
    (outdir / "robustness_results.json").write_text(json.dumps({"preflight": pf, "source_matching": source_rows, "results": results}, indent=2, default=float), encoding="utf-8")
    plots = make_plots(outdir, rows)
    reference = next(r for r in rows if r["run_id"] == "R0_reference_rho2_N96_T4")
    reference_ok = abs(float(reference["raw_final_radial_drift"]) - REFERENCE_DRIFT) < 5e-5
    summary = {
        "verdict": verdict(rows, reference_ok),
        "reference_ok": reference_ok,
        "reference_raw_drift": reference["raw_final_radial_drift"],
        "reference_net_drift": reference["final_net_radial_drift"],
        "reference_expected": REFERENCE_DRIFT,
        "plots": plots,
    }
    (outdir / "robustness_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"=== {summary['verdict']} | out={outdir} ===", flush=True)


if __name__ == "__main__":
    main()
