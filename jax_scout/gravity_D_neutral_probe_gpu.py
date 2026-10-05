"""GPU mirror of Claude's gravity_D_neutral_probe.py workflow.

This file exists to replicate the Claude mirror result in the repository's
established WSL/JAX GPU environment without changing the original NumPy script.
It keeps the same environment source, neutral probe, divergence-form operator,
windowed COM convention, and snapshot cadence, while adding force-contract
diagnostics for H = -D div(N grad).

Run in WSL2 jax venv, for example:
    . ~/jax_irer/bin/activate
    python /mnt/f/quantule_mapper/jax_scout/gravity_D_neutral_probe_gpu.py \
      --suite claude --out /mnt/f/quantule_mapper/sweep_runs/GRAVITY_D_GPU_CODEX_20260713
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jaxlib
import numpy as np
from jax import lax


ROOT = Path(__file__).resolve().parents[1]
D = 0.3


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "UNKNOWN"


def config_hash(cfg: dict[str, object]) -> str:
    data = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=float)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def preflight() -> dict[str, object]:
    devices = jax.devices()
    print("backend:", jax.default_backend(), flush=True)
    print("devices:", devices, flush=True)
    assert jax.default_backend() == "gpu", (
        f"GPU backend required; got {jax.default_backend()} with {devices}"
    )
    assert any(device.platform == "gpu" for device in devices), devices
    return {
        "backend": jax.default_backend(),
        "devices": [str(device) for device in devices],
        "selected_device": str(devices[0]),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "x64_enabled": bool(jax.config.read("jax_enable_x64")),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "cwd": str(ROOT),
        "git_commit": git_commit(),
        "env": {
            "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            "XLA_PYTHON_CLIENT_MEM_FRACTION": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }


def build_grid(n: int, L: float) -> dict[str, object]:
    x = jnp.linspace(-L / 2.0, L / 2.0, n, endpoint=False, dtype=jnp.float64)
    X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=L / n)
    return {
        "x": x,
        "X": X,
        "Y": Y,
        "Z": Z,
        "ikx": (1j * k[:, None, None]).astype(jnp.complex128),
        "iky": (1j * k[None, :, None]).astype(jnp.complex128),
        "ikz": (1j * k[None, None, :]).astype(jnp.complex128),
        "dV": jnp.asarray((L / n) ** 3, dtype=jnp.float64),
    }


def deriv(field: jnp.ndarray, ik: jnp.ndarray) -> jnp.ndarray:
    return jnp.fft.ifftn(ik * jnp.fft.fftn(field))


def lap_cov(psi: jnp.ndarray, Nf: jnp.ndarray, grid: dict[str, object]) -> jnp.ndarray:
    gx = deriv(psi, grid["ikx"])
    gy = deriv(psi, grid["iky"])
    gz = deriv(psi, grid["ikz"])
    return (
        deriv(Nf * gx, grid["ikx"])
        + deriv(Nf * gy, grid["iky"])
        + deriv(Nf * gz, grid["ikz"])
    )


def rhs(psi: jnp.ndarray, Nf: jnp.ndarray, grid: dict[str, object]) -> jnp.ndarray:
    return 1j * D * lap_cov(psi, Nf, grid)


@jax.jit
def rk4_step(psi: jnp.ndarray, Nf: jnp.ndarray, grid: dict[str, object], dt: jnp.ndarray) -> jnp.ndarray:
    k1 = rhs(psi, Nf, grid)
    k2 = rhs(psi + 0.5 * dt * k1, Nf, grid)
    k3 = rhs(psi + 0.5 * dt * k2, Nf, grid)
    k4 = rhs(psi + dt * k3, Nf, grid)
    return psi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@partial(jax.jit, static_argnames=("nsteps",))
def evolve_n(psi: jnp.ndarray, Nf: jnp.ndarray, grid: dict[str, object], dt: jnp.ndarray, nsteps: int) -> jnp.ndarray:
    def body(_, state):
        return rk4_step(state, Nf, grid, dt)

    return lax.fori_loop(0, nsteps, body, psi)


@jax.jit
def diagnostics(psi: jnp.ndarray, Nf: jnp.ndarray, grid: dict[str, object], probe_x: jnp.ndarray) -> dict[str, jnp.ndarray]:
    X = grid["X"]
    dV = grid["dV"]
    rho = jnp.abs(psi) ** 2
    window = jnp.abs(X - probe_x) < 6.0
    wrho = rho * window
    wmass = jnp.sum(wrho) * dV
    com_x = (jnp.sum(X * wrho) * dV) / (wmass + 1e-30)
    gx = deriv(psi, grid["ikx"])
    gy = deriv(psi, grid["iky"])
    gz = deriv(psi, grid["ikz"])
    momentum_x = jnp.sum(jnp.imag(jnp.conj(psi) * gx)) * dV
    norm = jnp.sum(rho) * dV
    dNx = jnp.real(deriv(Nf, grid["ikx"]))
    grad_energy = jnp.abs(gx) ** 2 + jnp.abs(gy) ** 2 + jnp.abs(gz) ** 2
    force_analytic = -D * jnp.sum(dNx * grad_energy) * dV
    psit = rhs(psi, Nf, grid)
    gt_x = deriv(psit, grid["ikx"])
    force_rhs = jnp.sum(jnp.imag(jnp.conj(psit) * gx + jnp.conj(psi) * gt_x)) * dV
    return {
        "com_x": com_x,
        "momentum_x": momentum_x,
        "norm": norm,
        "force_analytic": force_analytic,
        "force_rhs": force_rhs,
    }


def source_and_coefficient(grid: dict[str, object], cfg: dict[str, object]) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    X, Y, Z = grid["X"], grid["Y"], grid["Z"]
    sigma = float(cfg.get("source_sigma", 1.5))
    rho_B = jnp.exp(-((X * X + Y * Y + Z * Z) / (2.0 * sigma * sigma)))
    shat = (rho_B ** 2) / (jnp.max(rho_B ** 2) + 1e-30)
    family = str(cfg.get("family", "well"))
    beta = float(cfg.get("beta", 1.0))
    if family == "well":
        Nf = 1.0 / (1.0 + beta * shat)
    elif family == "flat":
        Nf = jnp.ones_like(shat)
    elif family == "hill":
        Nf = 1.0 + beta * shat
    else:
        raise ValueError(f"unknown coefficient family: {family}")
    return rho_B, shat, Nf


def initial_probe(grid: dict[str, object], cfg: dict[str, object]) -> jnp.ndarray:
    X, Y, Z = grid["X"], grid["Y"], grid["Z"]
    probe_x = float(cfg.get("probe_x", 4.0))
    sig = float(cfg.get("probe_sigma", 1.0))
    psi = jnp.exp(-(((X - probe_x) ** 2 + Y * Y + Z * Z) / (2.0 * sig * sig))).astype(jnp.complex128)
    if bool(cfg.get("normalize_probe", False)):
        target = jnp.asarray(float(cfg.get("target_norm", 1.0)), dtype=jnp.float64)
        norm = jnp.sum(jnp.abs(psi) ** 2) * grid["dV"]
        psi = psi * jnp.sqrt(target / (norm + 1e-30))
    return psi


def scalarize(diag: dict[str, jnp.ndarray]) -> dict[str, float]:
    return {key: float(np.asarray(value)) for key, value in diag.items()}


def run_case(cfg: dict[str, object], outdir: Path) -> dict[str, object]:
    started = time.time()
    n = int(cfg["N"])
    L = float(cfg["L"])
    dt = float(cfg["dt"])
    T = float(cfg["T"])
    nsnap = int(cfg.get("nsnap", 6))
    nsteps = int(round(T / dt))
    every = max(1, nsteps // nsnap)
    chunks = nsteps // every
    tail = nsteps - chunks * every

    grid = build_grid(n, L)
    rho_B, shat, Nf = source_and_coefficient(grid, cfg)
    psi0 = initial_probe(grid, cfg)
    psi = psi0
    probe_x = jnp.asarray(float(cfg.get("probe_x", 4.0)), dtype=jnp.float64)
    dt_j = jnp.asarray(dt, dtype=jnp.float64)

    samples = []
    d0 = diagnostics(psi, Nf, grid, probe_x)
    d0["force_fd"] = (diagnostics(rk4_step(psi, Nf, grid, dt_j), Nf, grid, probe_x)["momentum_x"] - d0["momentum_x"]) / dt_j
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), d0)
    samples.append({"t": 0.0, **scalarize(d0)})

    for i in range(chunks):
        psi = evolve_n(psi, Nf, grid, dt_j, every)
        di = diagnostics(psi, Nf, grid, probe_x)
        di["force_fd"] = (diagnostics(rk4_step(psi, Nf, grid, dt_j), Nf, grid, probe_x)["momentum_x"] - di["momentum_x"]) / dt_j
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), di)
        samples.append({"t": float((i + 1) * every * dt), **scalarize(di)})

    psi_sample_last = psi
    if tail:
        psi = evolve_n(psi, Nf, grid, dt_j, tail)
        psi.block_until_ready()

    final_diag = diagnostics(psi, Nf, grid, probe_x)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), final_diag)
    n0 = samples[0]["norm"]
    nT = float(np.asarray(final_diag["norm"]))
    result = {
        "config": cfg,
        "config_hash": config_hash(cfg),
        "git_commit": git_commit(),
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "operator": "i d_t psi = i D div(N grad psi); H = -D div(N grad)",
        "D": D,
        "cell_volume_dV": float((L / n) ** 3),
        "coefficient": {
            "source": "rho_B=exp(-r^2/(2*1.5^2)); Shat=rho_B^2/max(rho_B^2)",
            "min": float(np.asarray(jnp.min(Nf))),
            "max": float(np.asarray(jnp.max(Nf))),
            "rho_B_max": float(np.asarray(jnp.max(rho_B))),
            "Shat_max": float(np.asarray(jnp.max(shat))),
        },
        "samples": samples,
        "final_full_T": scalarize(final_diag),
        "metrics": {
            "sampled_com_drift_x": samples[-1]["com_x"] - samples[0]["com_x"],
            "sampled_momentum_change_x": samples[-1]["momentum_x"] - samples[0]["momentum_x"],
            "initial_accel_force_over_norm": samples[0]["force_analytic"] / (samples[0]["norm"] + 1e-30),
            "norm_ratio_full_T": nT / (n0 + 1e-30),
            "norm_error_abs_full_T": abs(nT - n0),
            "force_rhs_residual_max_abs": max(abs(s["force_rhs"] - s["force_analytic"]) for s in samples),
            "force_rhs_residual_max_rel_nonzero": max(
                abs(s["force_rhs"] - s["force_analytic"]) / abs(s["force_analytic"])
                for s in samples
                if abs(s["force_analytic"]) > 1e-20
            )
            if any(abs(s["force_analytic"]) > 1e-20 for s in samples)
            else 0.0,
            "force_fd_residual_max_rel_nonzero": max(
                abs(s["force_fd"] - s["force_analytic"]) / abs(s["force_analytic"])
                for s in samples
                if abs(s["force_analytic"]) > 1e-20
            )
            if any(abs(s["force_analytic"]) > 1e-20 for s in samples)
            else 0.0,
        },
        "timing": {
            "wall_time_s": time.time() - started,
            "nsteps": nsteps,
            "sample_every_steps": every,
            "chunks": chunks,
            "tail_steps": tail,
        },
    }
    case_path = outdir / f"{cfg['label']}.json"
    case_path.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    np.savez_compressed(
        outdir / f"{cfg['label']}_samples.npz",
        t=np.asarray([s["t"] for s in samples], dtype=np.float64),
        com_x=np.asarray([s["com_x"] for s in samples], dtype=np.float64),
        momentum_x=np.asarray([s["momentum_x"] for s in samples], dtype=np.float64),
        norm=np.asarray([s["norm"] for s in samples], dtype=np.float64),
        force_analytic=np.asarray([s["force_analytic"] for s in samples], dtype=np.float64),
        force_rhs=np.asarray([s["force_rhs"] for s in samples], dtype=np.float64),
        force_fd=np.asarray([s["force_fd"] for s in samples], dtype=np.float64),
    )
    return result


def claude_cases(T: float) -> list[dict[str, object]]:
    base = {"N": 64, "L": 30.0, "dt": 0.002, "T": T, "probe_x": 4.0, "nsnap": 6}
    return [
        {**base, "label": "main", "beta": 1.0, "probe_sigma": 1.0, "family": "well"},
        {**base, "label": "null_b0", "beta": 0.0, "probe_sigma": 1.0, "family": "flat"},
        {**base, "label": "neg_beta", "beta": -0.5, "probe_sigma": 1.0, "family": "well"},
        {**base, "label": "wide_probe", "beta": 1.0, "probe_sigma": 1.6, "family": "well"},
    ]


def hardening_cases(T: float) -> list[dict[str, object]]:
    return [
        {"label": "reversed_hill", "N": 64, "L": 30.0, "dt": 0.002, "T": T, "probe_x": 4.0, "probe_sigma": 1.0, "beta": 1.0, "family": "hill", "nsnap": 6},
        {"label": "larger_box_N64_L40", "N": 64, "L": 40.0, "dt": 0.002, "T": T, "probe_x": 4.0, "probe_sigma": 1.0, "beta": 1.0, "family": "well", "nsnap": 6},
    ]


def convergence_cases(T: float) -> list[dict[str, object]]:
    return [
        {"label": "conv_N64_L30_dt002", "N": 64, "L": 30.0, "dt": 0.002, "T": T, "probe_x": 4.0, "probe_sigma": 1.0, "beta": 1.0, "family": "well", "nsnap": 6},
        {"label": "conv_N96_L30_dt001", "N": 96, "L": 30.0, "dt": 0.001, "T": T, "probe_x": 4.0, "probe_sigma": 1.0, "beta": 1.0, "family": "well", "nsnap": 6},
        {"label": "conv_N128_L40_dt001", "N": 128, "L": 40.0, "dt": 0.001, "T": T, "probe_x": 4.0, "probe_sigma": 1.0, "beta": 1.0, "family": "well", "nsnap": 6},
    ]


def characterization_cases(T: float) -> list[dict[str, object]]:
    base = {
        "N": 96,
        "L": 30.0,
        "dt": 0.001,
        "T": T,
        "beta": 1.0,
        "family": "well",
        "nsnap": 6,
        "normalize_probe": True,
        "target_norm": 1.0,
    }
    distance = [
        {**base, "label": f"dist_x{x:g}", "probe_x": x, "probe_sigma": 1.0}
        for x in (2.5, 3.0, 4.0, 5.0, 6.0, 8.0)
    ]
    width = [
        {**base, "label": f"width_s{s:g}", "probe_x": 4.0, "probe_sigma": s}
        for s in (0.6, 0.8, 1.0, 1.3, 1.6, 2.0)
    ]
    return distance + width


def compare_to_claude(results: list[dict[str, object]], claude_summary: Path | None) -> list[dict[str, object]]:
    if claude_summary is None or not claude_summary.exists():
        return []
    ref = json.loads(claude_summary.read_text(encoding="utf-8"))
    ref_runs = {run["label"]: run for run in ref.get("runs", [])}
    rows = []
    for result in results:
        label = str(result["config"]["label"])
        if label not in ref_runs:
            continue
        ref_run = ref_runs[label]
        samples = result["samples"]
        rows.append(
            {
                "label": label,
                "claude_drift_x": ref_run["drift_x"],
                "gpu_drift_x": result["metrics"]["sampled_com_drift_x"],
                "drift_abs_diff": result["metrics"]["sampled_com_drift_x"] - ref_run["drift_x"],
                "claude_mass_ret": ref_run["mass_ret"],
                "gpu_norm_ratio": result["metrics"]["norm_ratio_full_T"],
                "max_com_sample_abs_diff": max(
                    abs(float(samples[i]["com_x"]) - float(ref_run["traj"][i][1]))
                    for i in range(min(len(samples), len(ref_run["traj"])))
                ),
            }
        )
    return rows


def write_summary(outdir: Path, preflight_info: dict[str, object], results: list[dict[str, object]], compare_rows: list[dict[str, object]]) -> None:
    by_label = {r["config"]["label"]: r for r in results}
    flat = by_label.get("null_b0")
    rows = []
    for result in results:
        label = result["config"]["label"]
        samples = result["samples"]
        row = {
            "label": label,
            "N": result["config"]["N"],
            "L": result["config"]["L"],
            "dt": result["config"]["dt"],
            "T": result["config"]["T"],
            "family": result["config"]["family"],
            "probe_sigma": result["config"]["probe_sigma"],
            "drift_x": result["metrics"]["sampled_com_drift_x"],
            "momentum_change_x": result["metrics"]["sampled_momentum_change_x"],
            "initial_accel_force_over_norm": result["metrics"]["initial_accel_force_over_norm"],
            "initial_force_analytic": samples[0]["force_analytic"],
            "initial_force_rhs": samples[0]["force_rhs"],
            "initial_force_fd": samples[0]["force_fd"],
            "norm_ratio_full_T": result["metrics"]["norm_ratio_full_T"],
            "force_rhs_residual_max_abs": result["metrics"]["force_rhs_residual_max_abs"],
            "force_rhs_residual_max_rel_nonzero": result["metrics"]["force_rhs_residual_max_rel_nonzero"],
            "force_fd_residual_max_rel_nonzero": result["metrics"]["force_fd_residual_max_rel_nonzero"],
            "wall_time_s": result["timing"]["wall_time_s"],
            "config_hash": result["config_hash"],
        }
        if flat is not None and len(flat["samples"]) == len(samples):
            net = [samples[i]["com_x"] - flat["samples"][i]["com_x"] for i in range(len(samples))]
            row["baseline_subtracted_final_com"] = net[-1]
        else:
            row["baseline_subtracted_final_com"] = ""
        rows.append(row)

    with (outdir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "preflight": preflight_info,
        "results": results,
        "compare_to_claude": compare_rows,
    }
    (outdir / "summary.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")

    if compare_rows:
        with (outdir / "compare_to_claude.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(compare_rows[0].keys()))
            writer.writeheader()
            writer.writerows(compare_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["claude", "hardening", "convergence", "characterization"], default="claude")
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--out", default=str(ROOT / "sweep_runs" / f"GRAVITY_D_GPU_{time.strftime('%Y%m%d_%H%M%S')}"))
    ap.add_argument("--claude-summary", default=str(ROOT / "sweep_runs" / "GRAVITY_D_PROBE_20260713_134638" / "summary.json"))
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    preflight_info = preflight()
    (outdir / "gpu_preflight.json").write_text(json.dumps(preflight_info, indent=2), encoding="utf-8")

    if args.suite == "claude":
        cases = claude_cases(args.T)
    elif args.suite == "hardening":
        cases = hardening_cases(args.T)
    else:
        cases = convergence_cases(args.T) if args.suite == "convergence" else characterization_cases(args.T)

    results = []
    for cfg in cases:
        print(
            f"[run] {cfg['label']} N={cfg['N']} L={cfg['L']} dt={cfg['dt']} T={cfg['T']} "
            f"family={cfg['family']} beta={cfg['beta']} sigma={cfg['probe_sigma']}",
            flush=True,
        )
        result = run_case(cfg, outdir)
        results.append(result)
        print(
            f"      drift={result['metrics']['sampled_com_drift_x']:+.6e} "
            f"F0={result['samples'][0]['force_analytic']:+.6e} "
            f"norm={result['metrics']['norm_ratio_full_T']:.12f} "
            f"rhs_rel={result['metrics']['force_rhs_residual_max_rel_nonzero']:.3e}",
            flush=True,
        )

    compare_rows = compare_to_claude(results, Path(args.claude_summary) if args.claude_summary else None)
    write_summary(outdir, preflight_info, results, compare_rows)
    print(f"=== GPU suite complete: {args.suite} | out={outdir} ===", flush=True)


if __name__ == "__main__":
    main()
