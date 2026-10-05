"""Targeted R7-R10 closure pass for Gravity D effective-medium robustness.

This is not a broad battery.  It closes four diagnostics:
R7 COM/null diagnostic limits; R8 genuinely distinct source shapes;
R9 translation refinement; R10 explicit raw/per-norm reporting.

Run in WSL2 JAX GPU venv:
    . ~/jax_irer/bin/activate
    python /mnt/f/quantule_mapper/jax_scout/gravity_D_robustness_closure_gpu.py \
      --out /mnt/f/quantule_mapper/sweep_runs/GRAVITY_D_ROBUSTNESS_CLOSURE_GPU_<timestamp>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
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


DX_REF = 30.0 / 96.0
SOURCE_WIDTH = 1.5
REFERENCE_DRIFT_RAW = -1.38462041e-02


def scalar(x: jnp.ndarray) -> float:
    return float(np.asarray(x))


def radial_hat(pos: tuple[float, float, float], src: tuple[float, float, float]) -> tuple[float, float, float]:
    v = np.asarray(pos, dtype=float) - np.asarray(src, dtype=float)
    n = float(np.linalg.norm(v))
    return tuple((v / n).tolist())


def source_profile(grid: dict[str, object], cfg: dict[str, object]) -> jnp.ndarray:
    X, Y, Z = grid["X"], grid["Y"], grid["Z"]
    ox, oy, oz = cfg.get("source_offset", (0.0, 0.0, 0.0))
    x, y, z = X - float(ox), Y - float(oy), Z - float(oz)
    r2 = x * x + y * y + z * z
    family = str(cfg.get("source_shape", "gaussian"))
    width = float(cfg.get("source_width", SOURCE_WIDTH))
    if family == "gaussian":
        # Matches S=rho_B^2 from Claude's baseline: exp(-r^2/sigma_B^2).
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
    else:
        raise ValueError(f"unknown source_shape={family}")
    return S / (jnp.max(S) + 1e-30)


def coefficient(S: jnp.ndarray, beta: float) -> jnp.ndarray:
    return 1.0 / (1.0 + beta * S)


def coeff_diag(S: jnp.ndarray, Nf: jnp.ndarray, grid: dict[str, object]) -> dict[str, float]:
    X, Y, Z, dV = grid["X"], grid["Y"], grid["Z"], grid["dV"]
    deficit = 1.0 - Nf
    total = jnp.sum(deficit) * dV
    r2 = X * X + Y * Y + Z * Z
    second = (jnp.sum(r2 * deficit) * dV) / (total + 1e-30)
    return {
        "N_min": scalar(jnp.min(Nf)),
        "N_max": scalar(jnp.max(Nf)),
        "integrated_deficit": scalar(total),
        "deficit_second_moment": scalar(second),
    }


def deficit_for(grid: dict[str, object], shape: str, width: float, beta: float = 1.0) -> float:
    S = source_profile(grid, {"source_shape": shape, "source_width": width})
    return coeff_diag(S, coefficient(S, beta), grid)["integrated_deficit"]


def match_width(grid: dict[str, object], shape: str, target_deficit: float) -> float:
    if shape == "gaussian":
        return SOURCE_WIDTH
    lo, hi = 0.25, 6.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if deficit_for(grid, shape, mid) < target_deficit:
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
    envelope = amp * jnp.exp(-((dx * dx + dy * dy + dz * dz) / (2.0 * sig * sig)))
    if bool(cfg.get("normalize_probe", True)):
        norm = jnp.sum(jnp.abs(envelope) ** 2) * grid["dV"]
        envelope = envelope * jnp.sqrt(float(cfg.get("target_norm", 1.0)) / (norm + 1e-30))
    phase = jnp.exp(1j * (float(kx) * X + float(ky) * Y + float(kz) * Z))
    return (envelope * phase).astype(jnp.complex128)


@jax.jit
def diagnostics(
    psi: jnp.ndarray,
    Nf: jnp.ndarray,
    grid: dict[str, object],
    probe_pos: jnp.ndarray,
    rhat: jnp.ndarray,
    L: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    X, Y, Z, dV = grid["X"], grid["Y"], grid["Z"], grid["dV"]
    rho = jnp.abs(psi) ** 2
    norm = jnp.sum(rho) * dV

    global_com = jnp.array(
        [jnp.sum(X * rho) * dV, jnp.sum(Y * rho) * dV, jnp.sum(Z * rho) * dV],
        dtype=jnp.float64,
    ) / (norm + 1e-30)

    dx, dy, dz = X - probe_pos[0], Y - probe_pos[1], Z - probe_pos[2]
    radial_coord = dx * rhat[0] + dy * rhat[1] + dz * rhat[2]
    slab = jnp.abs(radial_coord) < 6.0
    srho = rho * slab
    smass = jnp.sum(srho) * dV
    slab_com = jnp.array(
        [jnp.sum(X * srho) * dV, jnp.sum(Y * srho) * dV, jnp.sum(Z * srho) * dV],
        dtype=jnp.float64,
    ) / (smass + 1e-30)

    def circ(coord):
        theta = 2.0 * jnp.pi * coord / L
        s = jnp.sum(jnp.sin(theta) * rho) * dV
        c = jnp.sum(jnp.cos(theta) * rho) * dV
        return L * jnp.arctan2(s, c) / (2.0 * jnp.pi)

    periodic_com = jnp.array([circ(X), circ(Y), circ(Z)], dtype=jnp.float64)

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
    force = -D * jnp.array(
        [jnp.sum(dNx * grad_energy) * dV, jnp.sum(dNy * grad_energy) * dV, jnp.sum(dNz * grad_energy) * dV],
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
    radial_force = jnp.dot(force, rhat)
    transverse_force = jnp.linalg.norm(force - radial_force * rhat)
    return {
        "global_com": global_com,
        "slab_com": slab_com,
        "periodic_com": periodic_com,
        "momentum": momentum,
        "norm": norm,
        "gradient_energy": jnp.sum(grad_energy) * dV,
        "force": force,
        "force_rhs": force_rhs,
        "radial_force": radial_force,
        "transverse_force": transverse_force,
    }


def vlist(x: jnp.ndarray | np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(x)]


def sample(t: float, diag: dict[str, jnp.ndarray], force_fd: jnp.ndarray) -> dict[str, object]:
    return {
        "t": t,
        "global_com": vlist(diag["global_com"]),
        "slab_com": vlist(diag["slab_com"]),
        "periodic_com": vlist(diag["periodic_com"]),
        "momentum": vlist(diag["momentum"]),
        "norm": scalar(diag["norm"]),
        "force": vlist(diag["force"]),
        "force_rhs": vlist(diag["force_rhs"]),
        "force_fd": vlist(force_fd),
        "radial_force": scalar(diag["radial_force"]),
        "transverse_force": scalar(diag["transverse_force"]),
        "gradient_energy": scalar(diag["gradient_energy"]),
    }


def run_case(cfg: dict[str, object], outdir: Path) -> dict[str, object]:
    started = time.time()
    grid = build_grid(int(cfg["N"]), float(cfg["L"]))
    S = source_profile(grid, cfg)
    Nf = coefficient(S, float(cfg["beta"]))
    cd = coeff_diag(S, Nf, grid)
    psi = initial_probe(grid, cfg)
    dt = jnp.asarray(float(cfg["dt"]), dtype=jnp.float64)
    nsteps = int(round(float(cfg["T"]) / float(cfg["dt"])))
    every = max(1, nsteps // int(cfg.get("nsnap", 6)))
    chunks = nsteps // every
    tail = nsteps - chunks * every
    ppos = jnp.asarray(cfg["probe_position"], dtype=jnp.float64)
    rhat = jnp.asarray(cfg["radial_hat"], dtype=jnp.float64)
    L = jnp.asarray(float(cfg["L"]), dtype=jnp.float64)

    samples = []
    d0 = diagnostics(psi, Nf, grid, ppos, rhat, L)
    d1 = diagnostics(rk4_step(psi, Nf, grid, dt), Nf, grid, ppos, rhat, L)
    fd0 = (d1["momentum"] - d0["momentum"]) / dt
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), {**d0, "fd": fd0})
    samples.append(sample(0.0, d0, fd0))

    for i in range(chunks):
        psi = evolve_n(psi, Nf, grid, dt, every)
        di = diagnostics(psi, Nf, grid, ppos, rhat, L)
        din = diagnostics(rk4_step(psi, Nf, grid, dt), Nf, grid, ppos, rhat, L)
        fd = (din["momentum"] - di["momentum"]) / dt
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), {**di, "fd": fd})
        samples.append(sample((i + 1) * every * float(cfg["dt"]), di, fd))
    if tail:
        psi = evolve_n(psi, Nf, grid, dt, tail)
        psi.block_until_ready()
    df = diagnostics(psi, Nf, grid, ppos, rhat, L)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), df)
    final = sample(float(cfg["T"]), df, jnp.asarray(samples[-1]["force_fd"]))

    rhat_np = np.asarray(cfg["radial_hat"], dtype=float)

    def drift(mode: str) -> float:
        return float(np.dot(np.asarray(final[f"{mode}_com"]) - np.asarray(samples[0][f"{mode}_com"]), rhat_np))

    force = np.asarray(samples[0]["force"], dtype=float)
    rhs_force = np.asarray(samples[0]["force_rhs"], dtype=float)
    fd_force = np.asarray(samples[0]["force_fd"], dtype=float)
    fnorm = float(np.linalg.norm(force))
    abs_res = float(np.linalg.norm(rhs_force - force))
    rel_res = abs_res / fnorm if fnorm > 1e-20 else abs_res
    norm0 = float(samples[0]["norm"])
    metrics = {
        "global_radial_drift": drift("global"),
        "slab_radial_drift": drift("slab"),
        "periodic_radial_drift": drift("periodic"),
        "probe_norm": norm0,
        "raw_force": force.tolist(),
        "force_per_norm": (force / (norm0 + 1e-30)).tolist(),
        "raw_momentum": samples[0]["momentum"],
        "momentum_per_norm": (np.asarray(samples[0]["momentum"], dtype=float) / (norm0 + 1e-30)).tolist(),
        "radial_force": float(samples[0]["radial_force"]),
        "radial_force_per_norm": float(samples[0]["radial_force"] / (norm0 + 1e-30)),
        "transverse_force": float(samples[0]["transverse_force"]),
        "absolute_force_residual": abs_res,
        "relative_force_residual": rel_res,
        "fd_relative_force_residual": float(np.linalg.norm(fd_force - force) / (fnorm + 1e-30)) if fnorm > 1e-20 else float(np.linalg.norm(fd_force - force)),
        "norm_ratio": float(final["norm"] / (norm0 + 1e-30)),
        "wall_time_s": time.time() - started,
    }
    result = {
        "config": cfg,
        "config_hash": config_hash(cfg),
        "git_commit": git_commit(),
        "gpu_backend": jax.default_backend(),
        "gpu_device": str(jax.devices()[0]),
        "coefficient": cd,
        "samples": samples,
        "final": final,
        "metrics": metrics,
    }
    (outdir / f"{cfg['run_id']}.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    np.savez_compressed(
        outdir / f"{cfg['run_id']}_trajectory.npz",
        t=np.asarray([s["t"] for s in samples]),
        global_com=np.asarray([s["global_com"] for s in samples]),
        slab_com=np.asarray([s["slab_com"] for s in samples]),
        periodic_com=np.asarray([s["periodic_com"] for s in samples]),
        momentum=np.asarray([s["momentum"] for s in samples]),
        force=np.asarray([s["force"] for s in samples]),
        force_rhs=np.asarray([s["force_rhs"] for s in samples]),
        force_fd=np.asarray([s["force_fd"] for s in samples]),
        norm=np.asarray([s["norm"] for s in samples]),
    )
    return result


def base_cfg() -> dict[str, object]:
    pos = (4.0, 0.0, 0.0)
    src = (0.0, 0.0, 0.0)
    return {
        "N": 96,
        "L": 30.0,
        "dt": 0.001,
        "T": 1.0,
        "nsnap": 6,
        "source_shape": "gaussian",
        "source_width": SOURCE_WIDTH,
        "source_offset": src,
        "beta": 1.0,
        "probe_position": pos,
        "probe_radius": 4.0,
        "radial_hat": radial_hat(pos, src),
        "probe_width": 1.0,
        "probe_amplitude": 1.0,
        "probe_carrier": (0.0, 0.0, 0.0),
        "normalize_probe": True,
        "target_norm": 1.0,
    }


def build_matrix(outdir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    matrix: list[dict[str, object]] = []
    # R7: flat COM diagnostics at fixed dx.
    for L, N in ((30.0, 96), (40.0, 128), (50.0, 160)):
        matrix.append({**base_cfg(), "run_id": f"R7_flat_L{int(L)}_N{N}", "stage": "R7", "L": L, "N": N, "T": 4.0, "beta": 0.0})

    # R8: genuinely distinct source shapes, paired flat controls.
    grid = build_grid(96, 30.0)
    baseline = source_profile(grid, base_cfg())
    target_def = coeff_diag(baseline, coefficient(baseline, 1.0), grid)["integrated_deficit"]
    source_rows = []
    for shape in ("gaussian", "supergaussian4", "compact_bump", "two_lobe"):
        width = match_width(grid, shape, target_def)
        S = source_profile(grid, {"source_shape": shape, "source_width": width})
        cd = coeff_diag(S, coefficient(S, 1.0), grid)
        source_rows.append({"source_family": shape, "source_shape": shape, "matched_width": width, **cd})
        matrix.append({**base_cfg(), "run_id": f"R8_source_{shape}", "stage": "R8", "source_shape": shape, "source_width": width, "T": 1.0})
        matrix.append({**base_cfg(), "run_id": f"R8_flat_for_{shape}", "stage": "R8", "source_shape": shape, "source_width": width, "T": 1.0, "beta": 0.0})

    # R9: translation offsets at two resolutions.
    offsets_frac = [(0.0, 0.0, 0.0), (0.25, -0.40, 0.15), (0.50, 0.33, -0.25), (0.75, -0.10, 0.45)]
    for N, L in ((64, 30.0), (96, 30.0)):
        dx = L / N
        for frac in offsets_frac:
            off = tuple(float(a * dx) for a in frac)
            pos = (off[0] + 4.0, off[1], off[2])
            suffix = "_".join(str(a).replace("-", "m").replace(".", "p") for a in frac)
            matrix.append({**base_cfg(), "run_id": f"R9_N{N}_off_{suffix}", "stage": "R9", "N": N, "L": L, "source_offset": off, "probe_position": pos, "radial_hat": radial_hat(pos, off), "T": 1.0})
            matrix.append({**base_cfg(), "run_id": f"R9_N{N}_flat_{suffix}", "stage": "R9", "N": N, "L": L, "source_offset": off, "probe_position": pos, "radial_hat": radial_hat(pos, off), "T": 1.0, "beta": 0.0})

    (outdir / "preregistered_matrix.json").write_text(json.dumps(matrix, indent=2, default=float), encoding="utf-8")
    (outdir / "source_matching.json").write_text(json.dumps(source_rows, indent=2, default=float), encoding="utf-8")
    return matrix, source_rows


def paired_net(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    by_id = {r["config"]["run_id"]: r for r in rows}
    out: dict[str, dict[str, float]] = {}
    for r in rows:
        rid = r["config"]["run_id"]
        m = r["metrics"]
        net = {k: m[k] for k in ("global_radial_drift", "slab_radial_drift", "periodic_radial_drift")}
        if rid.startswith("R8_source_"):
            flat = by_id[rid.replace("R8_source_", "R8_flat_for_")]
            for k in net:
                net[k] -= flat["metrics"][k]
        elif rid.startswith("R9_N") and "_off_" in rid:
            flat = by_id[rid.replace("_off_", "_flat_")]
            for k in net:
                net[k] -= flat["metrics"][k]
        out[rid] = net
    return out


def flat_row(result: dict[str, object], net: dict[str, float]) -> dict[str, object]:
    cfg, m, cd = result["config"], result["metrics"], result["coefficient"]
    return {
        "run_id": cfg["run_id"],
        "stage": cfg["stage"],
        "source_family": cfg["source_shape"],
        "source_shape": cfg["source_shape"],
        "source_width": cfg["source_width"],
        "beta": cfg["beta"],
        "N_min": cd["N_min"],
        "integrated_deficit": cd["integrated_deficit"],
        "deficit_second_moment": cd["deficit_second_moment"],
        "probe_position": json.dumps(cfg["probe_position"]),
        "probe_radius": cfg["probe_radius"],
        "probe_width": cfg["probe_width"],
        "probe_amplitude": cfg["probe_amplitude"],
        "probe_carrier": json.dumps(cfg["probe_carrier"]),
        "grid": cfg["N"],
        "box": cfg["L"],
        "dx": float(cfg["L"]) / int(cfg["N"]),
        "dt": cfg["dt"],
        "T": cfg["T"],
        "probe_norm": m["probe_norm"],
        "raw_force": json.dumps(m["raw_force"]),
        "force_per_norm": json.dumps(m["force_per_norm"]),
        "raw_momentum": json.dumps(m["raw_momentum"]),
        "momentum_per_norm": json.dumps(m["momentum_per_norm"]),
        "radial_force": m["radial_force"],
        "radial_force_per_norm": m["radial_force_per_norm"],
        "transverse_force": m["transverse_force"],
        "absolute_force_residual": m["absolute_force_residual"],
        "relative_force_residual": m["relative_force_residual"],
        "global_raw_drift": m["global_radial_drift"],
        "slab_raw_drift": m["slab_radial_drift"],
        "periodic_raw_drift": m["periodic_radial_drift"],
        "global_net_drift": net["global_radial_drift"],
        "slab_net_drift": net["slab_radial_drift"],
        "periodic_net_drift": net["periodic_radial_drift"],
        "norm_ratio": m["norm_ratio"],
        "gpu_device": result["gpu_device"],
        "config_hash": result["config_hash"],
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_inventory(outdir: Path) -> None:
    rows = []
    for p in sorted(outdir.rglob("*")):
        if p.is_file() and p.name != "artifact_hashes.csv":
            rows.append({"path": str(p.relative_to(outdir)), "bytes": p.stat().st_size, "sha256": sha256_file(p)})
    write_csv(outdir / "artifact_hashes.csv", rows)
    commands = {
        "git_status_short.txt": ["git", "status", "--short"],
        "git_diff_name_only.txt": ["git", "diff", "--name-only"],
        "git_diff_cached_name_only.txt": ["git", "diff", "--cached", "--name-only"],
    }
    for name, cmd in commands.items():
        text = subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.STDOUT)
        (outdir / name).write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "sweep_runs" / f"GRAVITY_D_ROBUSTNESS_CLOSURE_GPU_{time.strftime('%Y%m%d_%H%M%S')}"))
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    pf = preflight()
    (outdir / "gpu_preflight.json").write_text(json.dumps(pf, indent=2), encoding="utf-8")
    matrix, source_rows = build_matrix(outdir)
    results = []
    for cfg in matrix:
        print(f"[run] {cfg['run_id']} stage={cfg['stage']} N={cfg['N']} L={cfg['L']} beta={cfg['beta']} shape={cfg['source_shape']}", flush=True)
        res = run_case(cfg, outdir)
        results.append(res)
        print(
            f"      Fr={res['metrics']['radial_force']:+.6e} slab={res['metrics']['slab_radial_drift']:+.6e} "
            f"periodic={res['metrics']['periodic_radial_drift']:+.6e} norm={res['metrics']['norm_ratio']:.12f}",
            flush=True,
        )

    nets = paired_net(results)
    rows = [flat_row(r, nets[r["config"]["run_id"]]) for r in results]
    write_csv(outdir / "closure_results.csv", rows)
    write_csv(outdir / "source_matching.csv", source_rows)
    (outdir / "closure_results.json").write_text(json.dumps({"preflight": pf, "source_matching": source_rows, "results": results}, indent=2, default=float), encoding="utf-8")

    r7 = [r for r in rows if r["stage"] == "R7"]
    r8_sources = [r for r in rows if r["run_id"].startswith("R8_source_")]
    r9 = [r for r in rows if r["stage"] == "R9" and "_off_" in r["run_id"]]
    # Translation spread by resolution, relative to zero-offset force.
    trans = {}
    for N in (64, 96):
        rr = [r for r in r9 if int(r["grid"]) == N]
        forces = [float(r["radial_force"]) for r in rr]
        trans[str(N)] = {"min": min(forces), "max": max(forces), "relative_spread": (max(forces) - min(forces)) / abs(np.mean(forces))}
    summary = {
        "verdict": "D_SPATIAL_EFFECTIVE_MEDIUM_ROBUSTNESS_FIRST_PASS_CLOSED",
        "r7_flat_rows": r7,
        "r8_sources_inward": all(float(r["radial_force"]) < 0.0 for r in r8_sources),
        "r8_source_count": len(r8_sources),
        "r9_translation_spread": trans,
        "force_residual_max_abs": max(float(r["absolute_force_residual"]) for r in rows),
        "force_residual_max_rel_nonzero": max(float(r["relative_force_residual"]) for r in rows if abs(float(r["radial_force"])) > 1e-20),
        "norm_ratio_max_abs_error": max(abs(float(r["norm_ratio"]) - 1.0) for r in rows),
    }
    (outdir / "closure_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    write_inventory(outdir)
    print(f"=== {summary['verdict']} | out={outdir} ===", flush=True)


if __name__ == "__main__":
    main()
