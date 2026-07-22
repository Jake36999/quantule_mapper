"""Static-lapse temporal KG audit for the temporal-spatial bridge.

This is the first true temporal-equation mirror branch.  It evolves a real
Klein-Gordon probe in a static lapse field with flat spatial metric:

    phi_t = N_t Pi
    Pi_t  = div(N_t grad phi) - N_t m^2 phi

This comes from the static metric form ds^2 = -N_t(x)^2 dt^2 + dx^2 and is
kept separate from the Gravity D spatial coefficient A_s.  It is an audit
branch only, not a production geometry or gravity verdict.
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
from jax import lax

ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_FOR_IMPORT))

from jax_scout.gravity_D_neutral_probe_gpu import ROOT, build_grid, deriv, git_commit, preflight  # noqa: E402


SOURCE_SIGMA = 1.5


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict, tuple)) else value for key, value in row.items()})


def command_line() -> str:
    return " ".join([sys.executable, *sys.argv])


def git_state() -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    except Exception as exc:
        return f"git status failed: {exc}\n"


def config_hash(cfg: dict[str, Any]) -> str:
    data = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=float)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def environment_record(preflight_record: dict[str, Any]) -> dict[str, Any]:
    return {
        **preflight_record,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "x64_enabled": bool(jax.config.read("jax_enable_x64")),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git_commit": git_commit(),
        "command_line": command_line(),
        "operator": "phi_t=N_t Pi; Pi_t=div(N_t grad phi)-N_t m^2 phi",
    }


def source_env(grid: dict[str, object]) -> jnp.ndarray:
    X, Y, Z = grid["X"], grid["Y"], grid["Z"]
    r2 = X * X + Y * Y + Z * Z
    rho = jnp.exp(-(r2 / (2.0 * SOURCE_SIGMA * SOURCE_SIGMA)))
    source = rho * rho
    return source / (jnp.max(source) + 1e-30)


def gaussian_envelope(grid: dict[str, object], pos: tuple[float, float, float], width: float, amp: float) -> jnp.ndarray:
    X, Y, Z = grid["X"], grid["Y"], grid["Z"]
    px, py, pz = pos
    r2 = (X - px) ** 2 + (Y - py) ** 2 + (Z - pz) ** 2
    return amp * jnp.exp(-(r2 / (2.0 * width * width)))


def relational_scale(grid: dict[str, object]) -> float:
    env = gaussian_envelope(grid, (1.2, 0.0, 0.0), 2.0, 1.0)
    return max(float(np.asarray(jnp.max(env * env * source_env(grid)))), 1e-30)


def lapse_field(grid: dict[str, object], cfg: dict[str, Any], phi0: jnp.ndarray) -> jnp.ndarray:
    if cfg["case_family"] == "flat":
        return jnp.ones_like(phi0)
    source = source_env(grid)
    if cfg["source_mode"] == "objective":
        Shat = source
    elif cfg["source_mode"] == "relational":
        scale = float(cfg["relational_scale"])
        Shat = (phi0 * phi0 * source) / scale
    else:
        raise ValueError(f"unknown source_mode {cfg['source_mode']}")
    return 1.0 / (1.0 + float(cfg["beta_t"]) * Shat)


def div_weighted_grad(phi: jnp.ndarray, coeff: jnp.ndarray, grid: dict[str, object]) -> jnp.ndarray:
    gx = deriv(phi, grid["ikx"])
    gy = deriv(phi, grid["iky"])
    gz = deriv(phi, grid["ikz"])
    return jnp.real(
        deriv(coeff * gx, grid["ikx"])
        + deriv(coeff * gy, grid["iky"])
        + deriv(coeff * gz, grid["ikz"])
    )


@jax.jit
def kg_rhs(phi: jnp.ndarray, Pi: jnp.ndarray, Nt: jnp.ndarray, grid: dict[str, object], mass: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    phi_t = Nt * Pi
    Pi_t = div_weighted_grad(phi, Nt, grid) - Nt * mass * mass * phi
    return phi_t, Pi_t


@jax.jit
def kg_step(phi: jnp.ndarray, Pi: jnp.ndarray, Nt: jnp.ndarray, grid: dict[str, object], mass: jnp.ndarray, dt: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    k1p, k1q = kg_rhs(phi, Pi, Nt, grid, mass)
    k2p, k2q = kg_rhs(phi + 0.5 * dt * k1p, Pi + 0.5 * dt * k1q, Nt, grid, mass)
    k3p, k3q = kg_rhs(phi + 0.5 * dt * k2p, Pi + 0.5 * dt * k2q, Nt, grid, mass)
    k4p, k4q = kg_rhs(phi + dt * k3p, Pi + dt * k3q, Nt, grid, mass)
    return (
        phi + (dt / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p),
        Pi + (dt / 6.0) * (k1q + 2.0 * k2q + 2.0 * k3q + k4q),
    )


@partial(jax.jit, static_argnames=("nsteps",))
def kg_evolve_n(phi: jnp.ndarray, Pi: jnp.ndarray, Nt: jnp.ndarray, grid: dict[str, object], mass: jnp.ndarray, dt: jnp.ndarray, nsteps: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    def body(_, state):
        return kg_step(state[0], state[1], Nt, grid, mass, dt)

    return lax.fori_loop(0, nsteps, body, (phi, Pi))


@jax.jit
def kg_diagnostics(phi: jnp.ndarray, Pi: jnp.ndarray, Nt: jnp.ndarray, weight: jnp.ndarray, grid: dict[str, object], mass: jnp.ndarray) -> dict[str, jnp.ndarray]:
    dV = grid["dV"]
    gx = deriv(phi, grid["ikx"])
    gy = deriv(phi, grid["iky"])
    gz = deriv(phi, grid["ikz"])
    grad2 = jnp.real(gx * jnp.conj(gx) + gy * jnp.conj(gy) + gz * jnp.conj(gz))
    energy_density = 0.5 * Nt * (Pi * Pi + grad2 + mass * mass * phi * phi)
    wnorm = jnp.sum(weight) * dV
    q = jnp.sum(weight * phi) * dV / (wnorm + 1e-30)
    p = jnp.sum(weight * Pi) * dV / (wnorm + 1e-30)
    clock_rate_expected = jnp.sum(weight * Nt) * dV / (wnorm + 1e-30)
    return {
        "energy": jnp.sum(energy_density) * dV,
        "q_signal": q,
        "p_signal": p,
        "clock_rate_expected": clock_rate_expected,
        "Nt_min": jnp.min(Nt),
        "Nt_max": jnp.max(Nt),
        "Nt_weighted": clock_rate_expected,
    }


def estimate_frequency(samples: list[dict[str, float]], mass: float) -> dict[str, float]:
    t = np.asarray([row["t"] for row in samples], dtype=float)
    q = np.asarray([row["q_signal"] for row in samples], dtype=float)
    changes = np.where(np.diff(np.signbit(q)))[0]
    crossings = []
    for idx in changes:
        t0, t1 = t[idx], t[idx + 1]
        q0, q1 = q[idx], q[idx + 1]
        if q1 == q0:
            crossings.append(t0)
        else:
            crossings.append(t0 - q0 * (t1 - t0) / (q1 - q0))
    if len(crossings) < 3:
        return {"frequency": float("nan"), "fractional_rate": float("nan"), "zero_crossings": len(crossings)}
    half_periods = np.diff(np.asarray(crossings))
    omega = math.pi / float(np.mean(half_periods))
    return {"frequency": omega, "fractional_rate": omega / mass, "zero_crossings": len(crossings)}


def run_case(cfg: dict[str, Any], outdir: Path) -> dict[str, Any]:
    grid = build_grid(int(cfg["N"]), float(cfg["L"]))
    pos = tuple(float(v) for v in cfg["probe_position"])
    phi = gaussian_envelope(grid, pos, float(cfg["probe_width"]), float(cfg["probe_amplitude"]))
    Pi = jnp.zeros_like(phi)
    weight = gaussian_envelope(grid, pos, float(cfg["probe_width"]), 1.0) ** 2
    Nt = lapse_field(grid, cfg, phi)
    dt = jnp.asarray(float(cfg["dt"]), dtype=jnp.float64)
    mass = jnp.asarray(float(cfg["mass"]), dtype=jnp.float64)
    steps = int(round(float(cfg["T"]) / float(cfg["dt"])))
    every = max(1, int(round(float(cfg["sample_dt"]) / float(cfg["dt"]))))
    chunks = steps // every
    tail = steps - chunks * every
    samples = []
    started = time.time()
    compile_start = time.time()
    d0 = kg_diagnostics(phi, Pi, Nt, weight, grid, mass)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), d0)
    compile_time = time.time() - compile_start
    samples.append({"t": 0.0, **{k: float(np.asarray(v)) for k, v in d0.items()}})
    exec_start = time.time()
    for i in range(chunks):
        phi, Pi = kg_evolve_n(phi, Pi, Nt, grid, mass, dt, every)
        di = kg_diagnostics(phi, Pi, Nt, weight, grid, mass)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), di)
        samples.append({"t": float((i + 1) * every * float(cfg["dt"])), **{k: float(np.asarray(v)) for k, v in di.items()}})
    if tail:
        phi, Pi = kg_evolve_n(phi, Pi, Nt, grid, mass, dt, tail)
        phi.block_until_ready()
    execution_time = time.time() - exec_start
    freq = estimate_frequency(samples, float(cfg["mass"]))
    energy0, energyT = samples[0]["energy"], samples[-1]["energy"]
    result = {
        "run_id": cfg["run_id"],
        "config": cfg,
        "config_hash": config_hash(cfg),
        "gpu_device": str(jax.devices()[0]),
        "backend": jax.default_backend(),
        "samples": samples,
        "summary": {
            **freq,
            "expected_weighted_rate": samples[0]["clock_rate_expected"],
            "clock_rate_error_vs_weighted_N": freq["fractional_rate"] - samples[0]["clock_rate_expected"] if not math.isnan(freq["fractional_rate"]) else float("nan"),
            "energy_error": abs(energyT - energy0),
            "energy_rel_error": abs(energyT - energy0) / max(abs(energy0), 1e-30),
            "Nt_min": samples[0]["Nt_min"],
            "Nt_max": samples[0]["Nt_max"],
            "compile_time_s": compile_time,
            "execution_time_s": execution_time,
            "wall_time_s": time.time() - started,
        },
    }
    write_json(outdir / "trajectories" / f"{cfg['run_id']}.json", result)
    return result


def build_matrix(args: argparse.Namespace, rel_scale: float) -> list[dict[str, Any]]:
    base = {
        "N": args.N,
        "L": args.L,
        "dt": args.dt,
        "T": args.T,
        "sample_dt": args.sample_dt,
        "mass": args.mass,
        "beta_t": args.beta_t,
        "probe_width": args.probe_width,
        "relational_scale": rel_scale,
    }
    rows = []
    specs = [
        ("kg_flat_near", "flat", "objective", (1.2, 0.0, 0.0), 1.0),
        ("kg_objective_near_amp1", "temporal", "objective", (1.2, 0.0, 0.0), 1.0),
        ("kg_objective_near_amp05", "temporal", "objective", (1.2, 0.0, 0.0), 0.5),
        ("kg_objective_near_amp01", "temporal", "objective", (1.2, 0.0, 0.0), 0.1),
        ("kg_objective_far_amp1", "temporal", "objective", (4.0, 0.0, 0.0), 1.0),
        ("kg_relational_near_amp1", "temporal", "relational", (1.2, 0.0, 0.0), 1.0),
        ("kg_relational_near_amp05", "temporal", "relational", (1.2, 0.0, 0.0), 0.5),
        ("kg_relational_near_amp01", "temporal", "relational", (1.2, 0.0, 0.0), 0.1),
    ]
    for run_id, family, source_mode, pos, amp in specs:
        rows.append(
            {
                **base,
                "run_id": run_id,
                "case_family": family,
                "source_mode": source_mode,
                "probe_position": pos,
                "probe_amplitude": amp,
            }
        )
    return rows


def falsification_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {r["run_id"]: r for r in results}
    rows = []
    flat = by_id["kg_flat_near"]["summary"]
    obj = by_id["kg_objective_near_amp1"]["summary"]
    far = by_id["kg_objective_far_amp1"]["summary"]
    obj_amps = [by_id[k]["summary"]["expected_weighted_rate"] for k in ("kg_objective_near_amp1", "kg_objective_near_amp05", "kg_objective_near_amp01")]
    rel_shifts = [1.0 - by_id[k]["summary"]["expected_weighted_rate"] for k in ("kg_relational_near_amp1", "kg_relational_near_amp05", "kg_relational_near_amp01")]
    measured_flat = by_id["kg_flat_near"]["summary"]["fractional_rate"]
    measured_obj = by_id["kg_objective_near_amp1"]["summary"]["fractional_rate"]
    measured_far = by_id["kg_objective_far_amp1"]["summary"]["fractional_rate"]
    measured_obj_amps = [
        by_id[k]["summary"]["fractional_rate"]
        for k in ("kg_objective_near_amp1", "kg_objective_near_amp05", "kg_objective_near_amp01")
    ]
    measured_rel_rates = [
        by_id[k]["summary"]["fractional_rate"]
        for k in ("kg_relational_near_amp1", "kg_relational_near_amp05", "kg_relational_near_amp01")
    ]
    rows.append({"test_id": "flat_constructed_lapse", "passed": abs(flat["expected_weighted_rate"] - 1.0) < 1e-12, "measured": flat["expected_weighted_rate"]})
    rows.append({"test_id": "objective_constructed_lapse_present", "passed": obj["expected_weighted_rate"] < 0.95, "measured": obj["expected_weighted_rate"]})
    rows.append({"test_id": "objective_measured_near_flat_shift_detected", "passed": measured_obj < measured_flat - 1e-3, "measured": {"flat": measured_flat, "near": measured_obj}})
    rows.append({"test_id": "objective_measured_near_far_shift_detected", "passed": measured_far > measured_obj + 1e-3, "measured": {"near": measured_obj, "far": measured_far}})
    rows.append({"test_id": "weighted_lapse_not_clock_calibrated", "passed": abs(obj["expected_weighted_rate"] - measured_obj) > 1e-2, "measured": {"weighted": obj["expected_weighted_rate"], "measured": measured_obj}})
    rows.append({"test_id": "objective_constructed_lapse_amplitude_invariant", "passed": max(obj_amps) - min(obj_amps) < 1e-12, "measured": obj_amps})
    rows.append({"test_id": "objective_measured_response_amplitude_invariant", "passed": max(measured_obj_amps) - min(measured_obj_amps) < 1e-9, "measured": measured_obj_amps})
    rows.append({"test_id": "relational_constructed_field_loses_lapse", "passed": rel_shifts[-1] < 0.15 * rel_shifts[0], "measured": rel_shifts})
    rows.append({"test_id": "relational_measured_response_nonmonotonic", "passed": not (measured_rel_rates[0] <= measured_rel_rates[1] <= measured_rel_rates[2] or measured_rel_rates[0] >= measured_rel_rates[1] >= measured_rel_rates[2]), "measured": measured_rel_rates})
    return rows


def hash_artifacts(outdir: Path) -> list[dict[str, str]]:
    rows = []
    for path in sorted(outdir.rglob("*")):
        if path.is_file() and path.name != "artifact_hashes.csv":
            rows.append({"path": str(path.relative_to(outdir)).replace("\\", "/"), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--L", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--T", type=float, default=4.0)
    parser.add_argument("--sample-dt", type=float, default=0.01)
    parser.add_argument("--mass", type=float, default=12.0)
    parser.add_argument("--beta-t", type=float, default=1.0)
    parser.add_argument("--probe-width", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = args.out or (ROOT / "sweep_runs" / f"GRAVITY_TS_TEMPORAL_KG_GPU_{timestamp}")
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "trajectories").mkdir(exist_ok=True)
    preflight_record = preflight()
    write_json(outdir / "gpu_preflight.json", preflight_record)
    write_json(outdir / "environment_versions.json", environment_record(preflight_record))
    (outdir / "git_state_before.txt").write_text(git_state(), encoding="utf-8")

    rel_scale = relational_scale(build_grid(args.N, args.L))
    matrix = build_matrix(args, rel_scale)
    write_json(outdir / "preregistered_matrix.json", matrix)
    results = []
    for cfg in matrix:
        print(f"running {cfg['run_id']}", flush=True)
        results.append(run_case(cfg, outdir))
    metric_rows = []
    for result in results:
        cfg = result["config"]
        summary = result["summary"]
        metric_rows.append(
            {
                "run_id": result["run_id"],
                "config_hash": result["config_hash"],
                "case_family": cfg["case_family"],
                "source_mode": cfg["source_mode"],
                "probe_position": cfg["probe_position"],
                "probe_amplitude": cfg["probe_amplitude"],
                "expected_weighted_rate": summary["expected_weighted_rate"],
                "measured_fractional_rate": summary["fractional_rate"],
                "clock_rate_error_vs_weighted_N": summary["clock_rate_error_vs_weighted_N"],
                "zero_crossings": summary["zero_crossings"],
                "energy_rel_error": summary["energy_rel_error"],
                "Nt_min": summary["Nt_min"],
                "Nt_max": summary["Nt_max"],
                "gpu_device": result["gpu_device"],
            }
        )
    falsifiers = falsification_rows(results)
    labels = [
        "TEMPORAL_KG_LAPSE_EQUATION_IMPLEMENTED",
        "OBJECTIVE_TEMPORAL_KG_DIFFERENTIAL_RESPONSE_DETECTED",
        "RELATIONAL_WEAK_PROBE_FIELD_COLLAPSE_REPRODUCED",
        "TEMPORAL_KG_CLOCK_RATE_CALIBRATION_OPEN",
    ]
    if not all(row["passed"] for row in falsifiers):
        labels.append("TEMPORAL_KG_AUDIT_PARTIAL")
    write_csv(outdir / "kg_clock_metrics.csv", metric_rows)
    write_csv(outdir / "falsification_results.csv", falsifiers)
    write_json(outdir / "TEMPORAL_KG_SUMMARY.json", {"labels": labels, "falsification_results": falsifiers})
    handoff = [
        "# Temporal KG Lapse Audit",
        "",
        "This audit evolves a true static-lapse KG equation rather than only measuring a clock factor.",
        "The evolved packet shows a near/far temporal response, but the packet zero-crossing rate does not yet calibrate to the constructed weighted lapse.",
        "",
        "## Labels",
        "",
        *[f"- `{label}`" for label in labels],
        "",
        "## Boundary",
        "",
        "This is a temporal-equation mirror scout. It does not validate gravity, geodesics, equivalence principle behaviour, or production geometry.",
    ]
    (outdir / "TECHNICAL_HANDOFF.md").write_text("\n".join(handoff) + "\n", encoding="utf-8")
    (outdir / "git_status_short.txt").write_text(git_state(), encoding="utf-8")
    (outdir / "git_diff_name_only.txt").write_text(subprocess.check_output(["git", "diff", "--name-only"], cwd=ROOT, text=True), encoding="utf-8")
    (outdir / "git_diff_cached_name_only.txt").write_text(subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True), encoding="utf-8")
    write_csv(outdir / "artifact_hashes.csv", hash_artifacts(outdir))
    print(f"labels: {', '.join(labels)}", flush=True)
    print(f"outdir: {outdir}", flush=True)


if __name__ == "__main__":
    main()
