"""TG-B1S state-load feedback audit.

This standalone GPU script tests exactly the STATE_LOAD_FEEDBACK branch:

    S_state -> T_steady -> G_steady -> delta phi

It reuses the validated C3 KG/Q-ball machinery, uses only the TG-S-supported
state-load source, and does not enable R_relax, L_lock, or P_threshold.  It does
not modify or import the older TG-B1 feedback driver.
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

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxlib
from jax import lax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jax_scout.phase_d_c3_wave import (  # noqa: E402
    build_kg,
    invariants,
    qball_petviashvili,
)
from jax_scout.provenance import write_json  # noqa: E402  (shared: stamps run metadata)

S0_AUTHORITATIVE = 135.6862187684289


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
            writer.writerow(row)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_hashes(outdir: Path) -> list[dict[str, str]]:
    rows = []
    for path in sorted(outdir.rglob("*")):
        if path.is_file() and path.name != "artifact_hashes.csv":
            rows.append({"path": str(path.relative_to(outdir)), "sha256": sha256_file(path)})
    return rows


def config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def preflight(outdir: Path) -> dict[str, Any]:
    try:
        devices = jax.devices()
        print("backend:", jax.default_backend(), flush=True)
        print("devices:", devices, flush=True)
        assert jax.default_backend() == "gpu", f"GPU backend required; got {jax.default_backend()} with {devices}"
        assert any(device.platform == "gpu" for device in devices), devices
        record = {
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
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
                "XLA_PYTHON_CLIENT_MEM_FRACTION": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
            },
        }
        write_json(outdir / "gpu_preflight.json", record)
        return record
    except Exception as exc:
        (outdir / "DISCREPANCY_REPORT.md").write_text(
            "# TG-B1S Discrepancy Report\n\nGPU preflight failed. No field evolution was run.\n\n"
            f"```text\n{exc}\n```\n",
            encoding="utf-8",
        )
        write_json(outdir / "gpu_preflight.json", {"status": "FAILED", "error": str(exc)})
        raise


def make_grid(op: dict[str, Any]) -> dict[str, jnp.ndarray]:
    n = int(op["N"])
    L = float(op["L"])
    dx = L / n
    x = jnp.linspace(-0.5 * L, 0.5 * L, n, endpoint=False, dtype=jnp.float64)
    X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
    R = jnp.sqrt(X * X + Y * Y + Z * Z)
    return {
        "X": X,
        "Y": Y,
        "Z": Z,
        "R": R,
        "dx": jnp.asarray(dx, dtype=jnp.float64),
        "ikx": op["ikx"],
        "iky": op["iky"],
        "ikz": op["ikz"],
    }


def solve_qball(cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    op = build_kg(cfg["N"], cfg["L"], cfg["c"], cfg["m"], cfg["dt"])
    mu = cfg["m"] ** 2 - cfg["w"] ** 2
    for sig in (1.5, 1.2, 1.8, 2.0):
        phi, prof = qball_petviashvili(op, cfg["a"], cfg["s"], cfg["f"], mu, sig=sig)
        if phi is not None and prof["residual"] < 1e-6 and prof["occ"] < 0.5:
            return phi.astype(np.complex128), prof
    raise RuntimeError("validated Q-ball solve failed")


def initial_state(kind: str, phi: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    psi = phi.astype(np.complex128)
    if kind == "global_phase":
        psi = np.exp(1j * 1.173) * psi
    elif kind == "translated":
        shift = max(1, int(round((1.25 / cfg["L"]) * cfg["N"])))
        psi = np.roll(psi, shift, axis=0)
    pi = (-1j * cfg["w"] * psi).astype(np.complex128)
    return psi, pi


def ref_source_norms(phi: np.ndarray, cfg: dict[str, Any], op: dict[str, Any]) -> dict[str, float]:
    psi, pi = initial_state("base", phi, cfg)
    pk = np.fft.fftn(psi)
    gx = np.fft.ifftn(np.asarray(op["ikx"]) * pk)
    gy = np.fft.ifftn(np.asarray(op["iky"]) * pk)
    gz = np.fft.ifftn(np.asarray(op["ikz"]) * pk)
    rho = np.abs(psi) ** 2
    grad2 = np.abs(gx) ** 2 + np.abs(gy) ** 2 + np.abs(gz) ** 2
    Gpot = cfg["a"] * rho**2 / 2.0 + cfg["s"] * rho**3 / 3.0 + cfg["f"] * rho**4 / 4.0
    energy = np.abs(pi) ** 2 + cfg["c"] ** 2 * grad2 + cfg["m"] ** 2 * rho - Gpot
    charge = np.abs(np.imag(np.conj(psi) * pi))
    raw = 0.5 * np.maximum(energy, 0.0) / (np.max(np.abs(energy)) + 1e-12)
    raw += 0.5 * charge / (np.max(charge) + 1e-12)
    dV = (cfg["L"] / cfg["N"]) ** 3
    return {
        "reference_energy_max": float(np.max(np.abs(energy))),
        "reference_charge_max": float(np.max(charge)),
        "reference_spatial_integral": float(np.sum(raw) * dV),
        "source_global_norm_S0": S0_AUTHORITATIVE,
    }


def shell_flux_proxy_np(psi: np.ndarray, pi: np.ndarray, cfg: dict[str, Any], op: dict[str, Any]) -> float:
    n = int(cfg["N"])
    L = float(cfg["L"])
    x = np.linspace(-0.5 * L, 0.5 * L, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    R = np.sqrt(X * X + Y * Y + Z * Z)
    pk = np.fft.fftn(psi)
    gx = np.fft.ifftn(np.asarray(op["ikx"]) * pk)
    gy = np.fft.ifftn(np.asarray(op["iky"]) * pk)
    gz = np.fft.ifftn(np.asarray(op["ikz"]) * pk)
    flux_density = -np.real(np.conj(pi) * (gx * X + gy * Y + gz * Z) / (R + 1e-12))
    shell = np.abs(R - 0.35 * L) < (0.05 * L)
    return float(np.mean(np.abs(np.where(shell, flux_density, 0.0))))


@jax.jit
def deriv_x(f: jnp.ndarray, g: dict[str, jnp.ndarray]) -> jnp.ndarray:
    return jnp.fft.ifftn(g["ikx"] * jnp.fft.fftn(f))


@jax.jit
def deriv_y(f: jnp.ndarray, g: dict[str, jnp.ndarray]) -> jnp.ndarray:
    return jnp.fft.ifftn(g["iky"] * jnp.fft.fftn(f))


@jax.jit
def deriv_z(f: jnp.ndarray, g: dict[str, jnp.ndarray]) -> jnp.ndarray:
    return jnp.fft.ifftn(g["ikz"] * jnp.fft.fftn(f))


def lap_real(f: jnp.ndarray, g: dict[str, jnp.ndarray]) -> jnp.ndarray:
    return jnp.real(deriv_x(deriv_x(f, g), g) + deriv_y(deriv_y(f, g), g) + deriv_z(deriv_z(f, g), g))


def div_A_grad(phi: jnp.ndarray, A: jnp.ndarray, g: dict[str, jnp.ndarray]) -> jnp.ndarray:
    return deriv_x(A * deriv_x(phi, g), g) + deriv_y(A * deriv_y(phi, g), g) + deriv_z(A * deriv_z(phi, g), g)


def absorb_profile(g: dict[str, jnp.ndarray], L: jnp.ndarray, width: jnp.ndarray, strength: jnp.ndarray) -> jnp.ndarray:
    edge = 0.5 * L - width
    s = jnp.maximum(g["R"] - edge, 0.0) / jnp.maximum(width, 1e-12)
    return strength * s * s


def state_load(phi: jnp.ndarray, pi: jnp.ndarray, cfg: jnp.ndarray, refs: jnp.ndarray, g: dict[str, jnp.ndarray]) -> jnp.ndarray:
    # cfg = [dt,c,m,a,s,f,alpha_T,omega_T,omega_G,gamma_T,gamma_G,kappa,epsilon,cT,cG,L,abs_width,abs_strength,core_radius]
    c, m, a, s, f = cfg[1], cfg[2], cfg[3], cfg[4], cfg[5]
    e_ref, q_ref, S0 = refs
    rho = jnp.abs(phi) ** 2
    gx, gy, gz = deriv_x(phi, g), deriv_y(phi, g), deriv_z(phi, g)
    grad2 = jnp.abs(gx) ** 2 + jnp.abs(gy) ** 2 + jnp.abs(gz) ** 2
    Gpot = a * rho**2 / 2.0 + s * rho**3 / 3.0 + f * rho**4 / 4.0
    energy = jnp.abs(pi) ** 2 + c * c * grad2 + m * m * rho - Gpot
    charge = jnp.abs(jnp.imag(jnp.conj(phi) * pi))
    raw = 0.5 * jnp.maximum(energy, 0.0) / (e_ref + 1e-12)
    raw = raw + 0.5 * charge / (q_ref + 1e-12)
    return raw / S0


@jax.jit
def rhs(state: tuple[jnp.ndarray, ...], cfg: jnp.ndarray, refs: jnp.ndarray, g: dict[str, jnp.ndarray], flags: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
    phi, pi, T, VT, G, VG = state
    source_enabled, temporal_enabled, geometric_enabled, feedback_enabled = flags
    dt, c, m, a, s, f, alpha_T, omega_T, omega_G, gamma_T, gamma_G, kappa, eps_G, cT, cG, L, abs_width, abs_strength, _core = cfg
    rho = jnp.abs(phi) ** 2
    source = state_load(phi, pi, cfg, refs, g) * source_enabled * temporal_enabled
    absorb = absorb_profile(g, L, abs_width, abs_strength)
    A = jnp.exp(-eps_G * G * geometric_enabled * feedback_enabled)
    kg_force = c * c * div_A_grad(phi, A, g) - m * m * phi + (a * rho + s * rho**2 + f * rho**3) * phi
    T_t = VT * temporal_enabled
    VT_t = (
        cT * cT * lap_real(T, g)
        - omega_T * omega_T * T
        - gamma_T * VT
        + alpha_T * source
        - kappa * G * geometric_enabled
        - absorb * VT
    ) * temporal_enabled
    G_t = VG * geometric_enabled
    VG_t = (
        cG * cG * lap_real(G, g)
        - omega_G * omega_G * G
        - gamma_G * VG
        - kappa * T
        - absorb * VG
    ) * geometric_enabled
    return pi, kg_force, T_t, VT_t, G_t, VG_t


@jax.jit
def rk4_step(state: tuple[jnp.ndarray, ...], cfg: jnp.ndarray, refs: jnp.ndarray, g: dict[str, jnp.ndarray], flags: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
    dt = cfg[0]
    k1 = rhs(state, cfg, refs, g, flags)
    s2 = tuple(y + 0.5 * dt * dy for y, dy in zip(state, k1))
    k2 = rhs(s2, cfg, refs, g, flags)
    s3 = tuple(y + 0.5 * dt * dy for y, dy in zip(state, k2))
    k3 = rhs(s3, cfg, refs, g, flags)
    s4 = tuple(y + dt * dy for y, dy in zip(state, k3))
    k4 = rhs(s4, cfg, refs, g, flags)
    return tuple(y + (dt / 6.0) * (a + 2 * b + 2 * c + d) for y, a, b, c, d in zip(state, k1, k2, k3, k4))


@partial(jax.jit, static_argnames=("nsteps",))
def evolve_n(state: tuple[jnp.ndarray, ...], cfg: jnp.ndarray, refs: jnp.ndarray, g: dict[str, jnp.ndarray], flags: jnp.ndarray, nsteps: int) -> tuple[jnp.ndarray, ...]:
    def body(_, s):
        return rk4_step(s, cfg, refs, g, flags)

    return lax.fori_loop(0, nsteps, body, state)


@jax.jit
def diagnostics(state: tuple[jnp.ndarray, ...], cfg: jnp.ndarray, refs: jnp.ndarray, g: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    phi, pi, T, VT, G, VG = state
    dt, c, m, a, s, f, alpha_T, omega_T, omega_G, gamma_T, gamma_G, kappa, eps_G, cT, cG, L, abs_width, abs_strength, core_radius = cfg
    dx = g["dx"]
    dV = dx**3
    rho = jnp.abs(phi) ** 2
    gx, gy, gz = deriv_x(phi, g), deriv_y(phi, g), deriv_z(phi, g)
    grad2 = jnp.abs(gx) ** 2 + jnp.abs(gy) ** 2 + jnp.abs(gz) ** 2
    Gpot = a * rho**2 / 2.0 + s * rho**3 / 3.0 + f * rho**4 / 4.0
    E_phi_density = jnp.abs(pi) ** 2 + c * c * grad2 + m * m * rho - Gpot
    E_phi = jnp.sum(E_phi_density) * dV
    charge = jnp.sum(jnp.imag(jnp.conj(phi) * pi)) * dV
    mass = jnp.sum(rho) * dV
    x0 = jnp.sum(g["X"] * rho) * dV / (mass + 1e-12)
    y0 = jnp.sum(g["Y"] * rho) * dV / (mass + 1e-12)
    z0 = jnp.sum(g["Z"] * rho) * dV / (mass + 1e-12)
    rr2 = (g["X"] - x0) ** 2 + (g["Y"] - y0) ** 2 + (g["Z"] - z0) ** 2
    core = jnp.exp(-0.5 * rr2 / (core_radius * core_radius))
    core_norm = jnp.sum(core) * dV
    core_energy = jnp.sum(core * E_phi_density) * dV / (core_norm + 1e-12)
    core_complex = jnp.sum(core * phi) * dV / (core_norm + 1e-12)
    width = jnp.sqrt(jnp.sum(rr2 * rho) * dV / (mass + 1e-12))
    Tgx, Tgy, Tgz = deriv_x(T, g), deriv_y(T, g), deriv_z(T, g)
    Ggx, Ggy, Ggz = deriv_x(G, g), deriv_y(G, g), deriv_z(G, g)
    E_T_density = 0.5 * (VT * VT + cT * cT * (jnp.abs(Tgx) ** 2 + jnp.abs(Tgy) ** 2 + jnp.abs(Tgz) ** 2) + omega_T * omega_T * T * T)
    E_G_density = 0.5 * (VG * VG + cG * cG * (jnp.abs(Ggx) ** 2 + jnp.abs(Ggy) ** 2 + jnp.abs(Ggz) ** 2) + omega_G * omega_G * G * G)
    E_cross = jnp.sum(kappa * T * G) * dV
    source = state_load(phi, pi, cfg, refs, g)
    absorb = absorb_profile(g, L, abs_width, abs_strength)
    source_work = jnp.sum(alpha_T * source * VT) * dV
    damping = jnp.sum(gamma_T * VT * VT + gamma_G * VG * VG) * dV
    boundary = jnp.sum(absorb * (VT * VT + VG * VG)) * dV
    flux_density = -jnp.real(jnp.conj(pi) * (gx * g["X"] + gy * g["Y"] + gz * g["Z"]) / (g["R"] + 1e-12))
    shell = jnp.abs(g["R"] - 0.35 * L) < (0.05 * L)
    shell_flux = jnp.mean(jnp.abs(jnp.where(shell, flux_density, 0.0)))
    ggrad_core = jnp.sum(core * jnp.sqrt(jnp.real(Ggx) ** 2 + jnp.real(Ggy) ** 2 + jnp.real(Ggz) ** 2)) * dV / (core_norm + 1e-12)
    return {
        "E_phi": E_phi,
        "E_T": jnp.sum(E_T_density) * dV,
        "E_G": jnp.sum(E_G_density) * dV,
        "E_cross": E_cross,
        "charge": charge,
        "mass": mass,
        "core_energy": core_energy,
        "core_amp": jnp.max(jnp.abs(phi)),
        "node_width": width,
        "node_x": x0,
        "node_y": y0,
        "node_z": z0,
        "node_phase_real": jnp.real(core_complex),
        "node_phase_imag": jnp.imag(core_complex),
        "T_node": jnp.sum(core * T) * dV / (core_norm + 1e-12),
        "G_node": jnp.sum(core * G) * dV / (core_norm + 1e-12),
        "T_peak": jnp.max(jnp.abs(T)),
        "G_peak": jnp.max(jnp.abs(G)),
        "G_grad_core": ggrad_core,
        "source_integral": jnp.sum(source) * dV,
        "source_peak": jnp.max(source),
        "source_work_rate": source_work,
        "damping_rate": damping,
        "boundary_rate": boundary,
        "shell_flux_proxy": shell_flux,
    }


def cfg_array(cfg: dict[str, Any]) -> jnp.ndarray:
    return jnp.asarray(
        [
            cfg["dt"],
            cfg["c"],
            cfg["m"],
            cfg["a"],
            cfg["s"],
            cfg["f"],
            cfg["alpha_T"],
            cfg["omega_T"],
            cfg["omega_G"],
            cfg["gamma_T"],
            cfg["gamma_G"],
            cfg["kappa_TG"],
            cfg["epsilon_G"],
            cfg["cT"],
            cfg["cG"],
            cfg["L"],
            cfg["absorb_width"],
            cfg["absorb_strength"],
            cfg["core_radius"],
        ],
        dtype=jnp.float64,
    )


def refs_array(refs: dict[str, float]) -> jnp.ndarray:
    return jnp.asarray(
        [refs["reference_energy_max"], refs["reference_charge_max"], refs["source_global_norm_S0"]],
        dtype=jnp.float64,
    )


def arm_flags(arm: str) -> list[float]:
    if arm == "S0_baseline":
        return [1.0, 0.0, 0.0, 0.0]
    if arm == "S1_temporal":
        return [1.0, 1.0, 0.0, 0.0]
    if arm in ("S2_feed_forward", "S3_feedback_off"):
        return [1.0, 1.0, 1.0, 0.0]
    if arm in ("S4_full_loop", "S8_global_phase", "S9_translated"):
        return [1.0, 1.0, 1.0, 1.0]
    if arm == "S5_source_off":
        return [0.0, 1.0, 1.0, 0.0]
    if arm == "S6_temporal_off":
        return [1.0, 0.0, 1.0, 0.0]
    if arm == "S7_geometric_off":
        return [1.0, 1.0, 0.0, 0.0]
    raise ValueError(arm)


def dominant_frequency(times: np.ndarray, series: np.ndarray) -> tuple[float, str]:
    if len(times) < 5 or np.max(np.abs(series - np.mean(series))) < 1e-14:
        return float("nan"), "TRANSIENT_RINGDOWN"
    dt = float(np.median(np.diff(times)))
    freqs = np.fft.rfftfreq(len(series), dt)
    amps = np.abs(np.fft.rfft(series - np.mean(series)))
    if len(amps) <= 1:
        return float("nan"), "TRANSIENT_RINGDOWN"
    idx = int(np.argmax(amps[1:]) + 1)
    f = float(freqs[idx])
    return f, "INSERTED_T_MODE_OR_NODE_BREATHING"


def summarize_run(run_id: str, arm: str, cfg: dict[str, Any], rows: list[dict[str, float]], final_state: tuple[jnp.ndarray, ...], initial_phi: np.ndarray) -> dict[str, Any]:
    times = np.asarray([r["t"] for r in rows])
    Tvals = np.asarray([r["T_node"] for r in rows])
    Gvals = np.asarray([r["G_node"] for r in rows])
    Tpeaks = np.asarray([r["T_peak"] for r in rows])
    Gpeaks = np.asarray([r["G_peak"] for r in rows])
    E_active = np.asarray([r["E_phi"] + r["E_T"] + r["E_G"] + r["E_cross"] for r in rows])
    source_work = np.asarray([r["source_work_rate"] for r in rows])
    damping = np.asarray([r["damping_rate"] for r in rows])
    boundary = np.asarray([r["boundary_rate"] for r in rows])
    ledger = (E_active[-1] - E_active[0]) - np.trapezoid(source_work - damping - boundary, times)
    phase = np.unwrap(np.angle(np.asarray([r["node_phase_real"] + 1j * r["node_phase_imag"] for r in rows])))
    if len(times) > 4:
        cut = len(times) // 2
        freq_slope = float(np.polyfit(times[cut:], phase[cut:], 1)[0])
    else:
        freq_slope = float("nan")
    final_phi = np.asarray(final_state[0])
    rho0 = np.abs(initial_phi) ** 2
    rhof = np.abs(final_phi) ** 2
    overlap = float(np.sum(rho0 * rhof) / (math.sqrt(np.sum(rho0 * rho0) * np.sum(rhof * rhof)) + 1e-30))
    T_final = float(np.mean(np.abs(Tvals[max(0, int(0.8 * len(Tvals))):])))
    G_final = float(np.mean(np.abs(Gvals[max(0, int(0.8 * len(Gvals))):])))
    T_onset = float("nan")
    if np.max(Tpeaks) > 0:
        idx = np.flatnonzero(Tpeaks >= 0.1 * np.max(Tpeaks))
        if len(idx):
            T_onset = float(times[int(idx[0])])
    G_onset = float("nan")
    if np.max(Gpeaks) > 0:
        idx = np.flatnonzero(Gpeaks >= 0.1 * np.max(Gpeaks))
        if len(idx):
            G_onset = float(times[int(idx[0])])
    dom_f, freq_class = dominant_frequency(times, Tvals)
    return {
        "run_id": run_id,
        "arm": arm,
        "T_equilibrium_amp": T_final,
        "T_peak": float(np.max(Tpeaks)),
        "T_approach_time": T_onset,
        "G_equilibrium_amp": G_final,
        "G_peak": float(np.max(Gpeaks)),
        "G_response_lag": G_onset - T_onset if np.isfinite(T_onset) and np.isfinite(G_onset) else float("nan"),
        "G_grad_core_final": float(rows[-1]["G_grad_core"]),
        "source_integral_mean": float(np.mean([r["source_integral"] for r in rows])),
        "source_peak_max": float(np.max([r["source_peak"] for r in rows])),
        "E_phi_initial": rows[0]["E_phi"],
        "E_phi_final": rows[-1]["E_phi"],
        "E_T_final": rows[-1]["E_T"],
        "E_G_final": rows[-1]["E_G"],
        "E_cross_final": rows[-1]["E_cross"],
        "dissipated_E": float(np.trapezoid(damping, times)),
        "boundary_E": float(np.trapezoid(boundary, times)),
        "source_work_E": float(np.trapezoid(source_work, times)),
        "ledger_residual": float(ledger),
        "ledger_residual_abs": float(abs(ledger)),
        "charge_initial": rows[0]["charge"],
        "charge_final": rows[-1]["charge"],
        "core_energy_initial": rows[0]["core_energy"],
        "core_energy_final": rows[-1]["core_energy"],
        "core_amp_initial": rows[0]["core_amp"],
        "core_amp_final": rows[-1]["core_amp"],
        "node_width_initial": rows[0]["node_width"],
        "node_width_final": rows[-1]["node_width"],
        "profile_overlap": overlap,
        "breathing_amp_core_energy": float(np.std([r["core_energy"] for r in rows])),
        "shell_flux_max": float(np.max([r["shell_flux_proxy"] for r in rows])),
        "node_frequency": -freq_slope,
        "dominant_T_frequency": dom_f,
        "frequency_class": freq_class,
        "node_x_final": rows[-1]["node_x"],
        "node_y_final": rows[-1]["node_y"],
        "node_z_final": rows[-1]["node_z"],
    }


def run_arm(run_id: str, arm: str, cfg: dict[str, Any], phi: np.ndarray, refs: dict[str, float], kind: str = "base", T_override: float | None = None) -> tuple[dict[str, Any], list[dict[str, float]]]:
    op = build_kg(cfg["N"], cfg["L"], cfg["c"], cfg["m"], cfg["dt"])
    g = make_grid(op)
    psi, pi = initial_state(kind, phi, cfg)
    state = (
        jnp.asarray(psi),
        jnp.asarray(pi),
        jnp.zeros_like(jnp.real(jnp.asarray(psi))),
        jnp.zeros_like(jnp.real(jnp.asarray(psi))),
        jnp.zeros_like(jnp.real(jnp.asarray(psi))),
        jnp.zeros_like(jnp.real(jnp.asarray(psi))),
    )
    cfgv = cfg_array(cfg)
    refv = refs_array(refs)
    flags = jnp.asarray(arm_flags(arm), dtype=jnp.float64)
    Tphys = float(T_override if T_override is not None else cfg["T"])
    steps = int(round(Tphys / cfg["dt"]))
    every = max(1, int(round(cfg["sample_dt"] / cfg["dt"])))
    chunks = steps // every
    rows: list[dict[str, float]] = []
    compile_start = time.time()
    d0 = diagnostics(state, cfgv, refv, g)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), d0)
    compile_s = time.time() - compile_start
    rows.append({"t": 0.0, **{k: float(np.asarray(v)) for k, v in d0.items()}})
    exec_start = time.time()
    for idx in range(chunks):
        state = evolve_n(state, cfgv, refv, g, flags, every)
        d = diagnostics(state, cfgv, refv, g)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), d)
        rows.append({"t": (idx + 1) * every * cfg["dt"], **{k: float(np.asarray(v)) for k, v in d.items()}})
    exec_s = time.time() - exec_start
    summary = summarize_run(run_id, arm, cfg, rows, state, psi)
    summary["compile_time_s"] = compile_s
    summary["execution_time_s"] = exec_s
    summary["config_hash"] = config_hash(cfg)
    return summary, rows


def baseline_contract(cfg: dict[str, Any], phi: np.ndarray, prof: dict[str, Any]) -> dict[str, Any]:
    op = build_kg(cfg["N"], cfg["L"], cfg["c"], cfg["m"], cfg["dt"])
    psi, pi = initial_state("base", phi, cfg)
    inv0 = invariants(jnp.fft.fftn(jnp.asarray(psi)), jnp.fft.fftn(jnp.asarray(pi)), op, cfg["a"], cfg["s"], cfg["f"])
    # Check using the original C3 Strang machinery, not the coupled RK4 path.
    from jax_scout.phase_d_c3_wave import kg_evolve

    pk, qk = jnp.fft.fftn(jnp.asarray(psi)), jnp.fft.fftn(jnp.asarray(pi))
    steps = int(round(4.0 / cfg["dt"]))
    pk, qk = kg_evolve(pk, qk, op, cfg["a"], cfg["s"], cfg["f"], steps)
    pk.block_until_ready()
    qk.block_until_ready()
    inv1 = invariants(pk, qk, op, cfg["a"], cfg["s"], cfg["f"])
    psi1 = np.asarray(jnp.fft.ifftn(pk))
    pi1 = np.asarray(jnp.fft.ifftn(qk))
    rho0, rho1 = np.abs(psi) ** 2, np.abs(psi1) ** 2
    overlap = float(np.sum(rho0 * rho1) / (math.sqrt(np.sum(rho0 * rho0) * np.sum(rho1 * rho1)) + 1e-30))
    dE = abs(inv1["E"] - inv0["E"]) / (abs(inv0["E"]) + 1e-30)
    dQ = abs(inv1["Q"] - inv0["Q"]) / (abs(inv0["Q"]) + 1e-30)
    max_flux = max(shell_flux_proxy_np(psi, pi, cfg, op), shell_flux_proxy_np(psi1, pi1, cfg, op))
    return {
        "status": "TG_SOURCE_NODE_BASELINE_CLOSED" if dE <= 1e-8 and dQ <= 1e-8 and overlap >= 0.995 and max_flux < 1e-8 else "TG_SOURCE_NODE_NOT_STATIONARY",
        "qball_residual": prof["residual"],
        "qball_amp": prof["amp"],
        "qball_occ": prof["occ"],
        "dE_rel": dE,
        "dQ_rel": dQ,
        "profile_overlap": overlap,
        "max_shell_flux_proxy": max_flux,
        "TG_S_reference_shell_flux_proxy": 3.7709214715448365e-09,
    }


def classify(summary_by_id: dict[str, dict[str, Any]], validation_rows: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    s1 = summary_by_id["S1_temporal"]
    s2 = summary_by_id["S2_feed_forward"]
    s3 = summary_by_id["S3_feedback_off"]
    s4 = summary_by_id["S4_full_loop"]
    s5 = summary_by_id["S5_source_off"]
    s6 = summary_by_id["S6_temporal_off"]
    s7 = summary_by_id["S7_geometric_off"]
    phase = summary_by_id["S8_global_phase"]
    trans = summary_by_id["S9_translated"]
    temporal_floor = max(abs(s5["T_peak"]), abs(s6["T_peak"]), 1e-12)
    geometric_floor = max(abs(s5["G_peak"]), abs(s6["G_peak"]), abs(s7["G_peak"]), 1e-12)
    ff_pass = (
        s1["T_peak"] > 100.0 * temporal_floor
        and s2["G_peak"] > 100.0 * geometric_floor
        and s7["T_peak"] > 100.0 * temporal_floor
        and s7["G_peak"] <= 10.0 * geometric_floor
        and abs(phase["T_peak"] - s4["T_peak"]) / max(abs(s4["T_peak"]), 1e-12) < 0.05
        and abs(trans["T_peak"] - s4["T_peak"]) / max(abs(s4["T_peak"]), 1e-12) < 0.15
    )
    back_delta = {
        "delta_core_energy_final": s4["core_energy_final"] - s3["core_energy_final"],
        "delta_charge_final": s4["charge_final"] - s3["charge_final"],
        "delta_core_amp_final": s4["core_amp_final"] - s3["core_amp_final"],
        "delta_width_final": s4["node_width_final"] - s3["node_width_final"],
        "delta_shell_flux_max": s4["shell_flux_max"] - s3["shell_flux_max"],
        "delta_node_frequency": s4["node_frequency"] - s3["node_frequency"],
    }
    back_effect = max(abs(v) for v in back_delta.values()) > 1e-8
    validation_pass = any(row.get("case") == "dt_half" and row.get("pass") for row in validation_rows)
    dt_half_backreaction = any(row.get("case") == "dt_half" and row.get("backreaction_detected") for row in validation_rows)
    label = "TG_STATE_LOAD_FEEDFORWARD_CHAIN_SUPPORTED" if ff_pass and validation_pass else "TG_STATE_LOAD_NUMERICALLY_UNRESOLVED"
    if not ff_pass:
        if s1["T_peak"] <= 100.0 * temporal_floor:
            label = "TG_STATE_LOAD_TEMPORAL_RESPONSE_NULL"
        elif s2["G_peak"] <= 100.0 * geometric_floor:
            label = "TG_STATE_LOAD_GEOMETRIC_RESPONSE_NULL"
    if label == "TG_STATE_LOAD_FEEDFORWARD_CHAIN_SUPPORTED" and back_effect and dt_half_backreaction:
        back_label = "TG_STATE_LOAD_GEOMETRIC_BACKREACTION_DETECTED"
    elif label == "TG_STATE_LOAD_FEEDFORWARD_CHAIN_SUPPORTED" and back_effect:
        back_label = "TG_STATE_LOAD_NUMERICALLY_UNRESOLVED"
    elif label == "TG_STATE_LOAD_FEEDFORWARD_CHAIN_SUPPORTED":
        back_label = "TG_STATE_LOAD_BACKREACTION_NULL"
    else:
        back_label = ""
    intervention_rows = [
        {"test": "source_off_removes_T_G", "pass": s5["T_peak"] <= 10.0 * temporal_floor and s5["G_peak"] <= 10.0 * geometric_floor, "T_peak": s5["T_peak"], "G_peak": s5["G_peak"]},
        {"test": "temporal_off_removes_T_G", "pass": s6["T_peak"] <= 10.0 * temporal_floor and s6["G_peak"] <= 10.0 * geometric_floor, "T_peak": s6["T_peak"], "G_peak": s6["G_peak"]},
        {"test": "geometric_off_retains_T_removes_G", "pass": s7["T_peak"] > 100.0 * temporal_floor and s7["G_peak"] <= 10.0 * geometric_floor, "T_peak": s7["T_peak"], "G_peak": s7["G_peak"]},
        {"test": "global_phase_invariant", "pass": abs(phase["T_peak"] - s4["T_peak"]) / max(abs(s4["T_peak"]), 1e-12) < 0.05, "control_T_peak": phase["T_peak"], "reference_T_peak": s4["T_peak"]},
        {"test": "translated_response_nonzero", "pass": trans["T_peak"] > 100.0 * temporal_floor and trans["G_peak"] > 100.0 * geometric_floor, "T_peak": trans["T_peak"], "G_peak": trans["G_peak"]},
    ]
    falsification = [
        {"test": "feed_forward_chain", "pass": ff_pass, "label": label},
        {"test": "backreaction_detected", "pass": back_effect, "label": back_label, **back_delta},
        {"test": "dt_half_validation_present", "pass": validation_pass},
        {"test": "dt_half_backreaction_reproduced", "pass": dt_half_backreaction},
    ]
    return (back_label or label), intervention_rows, falsification


def write_docs(outdir: Path, status: str, labels: list[str], summary_by_id: dict[str, dict[str, Any]]) -> None:
    docdir = ROOT / "docs" / "gravity_maturity"
    docdir.mkdir(parents=True, exist_ok=True)
    s3 = summary_by_id.get("S3_feedback_off", {})
    s4 = summary_by_id.get("S4_full_loop", {})
    results = [
        "# TG-B1S State-Load Feedback Results",
        "",
        "Timestamp: 2026-07-14.",
        f"Run directory: `{outdir.as_posix()}`.",
        f"Status: `{status}`.",
        "",
        "## Summary",
        "",
        "This run tests exactly `STATE_LOAD_FEEDBACK` using the TG-S-supported `S_state` source. It does not enable `R_relax`, `L_lock`, or `P_threshold`.",
        "",
        "## Key Metrics",
        "",
        f"- Feed-forward T peak: `{summary_by_id.get('S1_temporal', {}).get('T_peak')}`.",
        f"- Feed-forward G peak: `{summary_by_id.get('S2_feed_forward', {}).get('G_peak')}`.",
        f"- Full-loop minus feedback-off final core energy: `{s4.get('core_energy_final', 0.0) - s3.get('core_energy_final', 0.0)}`.",
        f"- Full-loop minus feedback-off charge: `{s4.get('charge_final', 0.0) - s3.get('charge_final', 0.0)}`.",
        "",
        "## Bounded Labels",
        "",
        *(f"- `{label}`" for label in labels),
        "",
        "No gravity, photon, objective-time, geodesic, universal-free-fall, or IRER-validation claim is made.",
    ]
    (docdir / "TG_B1S_RESULTS.md").write_text("\n".join(results), encoding="utf-8")
    write_json(docdir / "TG_B1S_SUMMARY.json", {"status": status, "labels": labels, "run_directory": str(outdir)})
    (docdir / "TG_B1S_DOCUMENTATION_INPUTS.md").write_text(
        "\n".join(
            [
                "# TG-B1S Documentation Inputs",
                "",
                f"- Status: `{status}`.",
                f"- Labels: `{', '.join(labels)}`.",
                f"- Artifacts: `{outdir.as_posix()}`.",
                "- Source: `STATE_LOAD_FEEDBACK` only.",
                "- Preserve prior TG-B1/TG-S labels; do not update master catalogue until review.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--N", type=int, default=48)
    ap.add_argument("--L", type=float, default=10.0)
    ap.add_argument("--c", type=float, default=0.5477)
    ap.add_argument("--m", type=float, default=1.0)
    ap.add_argument("--a", type=float, default=0.8)
    ap.add_argument("--s", type=float, default=-0.5)
    ap.add_argument("--f", type=float, default=-0.1)
    ap.add_argument("--w", type=float, default=0.964)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--sample-dt", type=float, default=0.05)
    ap.add_argument("--alpha-T", type=float, default=0.35)
    ap.add_argument("--omega-T", type=float, default=1.25)
    ap.add_argument("--omega-G", type=float, default=0.85)
    ap.add_argument("--gamma-T", type=float, default=0.08)
    ap.add_argument("--gamma-G", type=float, default=0.06)
    ap.add_argument("--kappa-TG", type=float, default=0.55)
    ap.add_argument("--epsilon-G", type=float, default=0.06)
    ap.add_argument("--cT", type=float, default=0.7)
    ap.add_argument("--cG", type=float, default=0.55)
    ap.add_argument("--absorb-width", type=float, default=1.6)
    ap.add_argument("--absorb-strength", type=float, default=0.02)
    ap.add_argument("--core-radius", type=float, default=2.0)
    args = ap.parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.out) if args.out else ROOT / "sweep_runs" / f"TG_B1S_STATE_LOAD_GPU_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "git_state_before.txt").write_text(git_state(), encoding="utf-8")
    pf = preflight(outdir)
    cfg = vars(args).copy()
    cfg["config_hash"] = config_hash(cfg)
    write_json(outdir / "environment_versions.json", {**pf, "stage": "TG-B1S_STATE_LOAD_FEEDBACK"})
    phi, prof = solve_qball(cfg)
    op = build_kg(cfg["N"], cfg["L"], cfg["c"], cfg["m"], cfg["dt"])
    refs = ref_source_norms(phi, cfg, op)
    write_json(
        outdir / "source_spec.json",
        {
            "source_hypothesis": "STATE_LOAD_FEEDBACK",
            "enabled_source": "S_state",
            "disabled_sources": ["R_relax", "L_lock", "P_threshold"],
            "local_definition": "0.5*max(E_phi_density,0)/E_ref_max + 0.5*abs(charge_density)/Q_ref_max",
            "global_normalization": "S_hat = local_definition / S0",
            "S0_authoritative_TG_S": S0_AUTHORITATIVE,
            **refs,
            "casewise_normalization": False,
        },
    )
    base = baseline_contract(cfg, phi, prof)
    write_csv(outdir / "baseline_node_contract.csv", [base])
    if base["status"] != "TG_SOURCE_NODE_BASELINE_CLOSED":
        (outdir / "TECHNICAL_HANDOFF.md").write_text("# TG-B1S Technical Handoff\n\nStatus: `TG_SOURCE_NODE_NOT_STATIONARY`.\n", encoding="utf-8")
        write_csv(outdir / "artifact_hashes.csv", artifact_hashes(outdir))
        print(json.dumps({"status": "TG_SOURCE_NODE_NOT_STATIONARY", "outdir": str(outdir)}, indent=2))
        return

    arms = [
        ("S0_baseline", "S0_baseline", "base"),
        ("S1_temporal", "S1_temporal", "base"),
        ("S2_feed_forward", "S2_feed_forward", "base"),
        ("S3_feedback_off", "S3_feedback_off", "base"),
        ("S4_full_loop", "S4_full_loop", "base"),
        ("S5_source_off", "S5_source_off", "base"),
        ("S6_temporal_off", "S6_temporal_off", "base"),
        ("S7_geometric_off", "S7_geometric_off", "base"),
        ("S8_global_phase", "S8_global_phase", "global_phase"),
        ("S9_translated", "S9_translated", "translated"),
    ]
    summaries: list[dict[str, Any]] = []
    trajectories: dict[str, list[dict[str, float]]] = {}
    for run_id, arm, kind in arms:
        summary, rows = run_arm(run_id, arm, cfg, phi, refs, kind=kind)
        summaries.append(summary)
        trajectories[run_id] = rows
        write_csv(outdir / f"{run_id}_trajectory.csv", rows)

    # Validated nearby Q-ball state ladder, source/feed-forward only.
    scaling_rows: list[dict[str, Any]] = []
    for w in (0.956, 0.964, 0.968):
        ladder_cfg = dict(cfg)
        ladder_cfg["w"] = w
        phiw, _profw = solve_qball(ladder_cfg)
        summary, _rows = run_arm(f"S10_ladder_w{w:.3f}", "S2_feed_forward", ladder_cfg, phiw, refs, kind="base", T_override=1.0)
        scaling_rows.append({"w": w, **{k: summary[k] for k in ("source_integral_mean", "T_peak", "G_peak", "E_phi_initial", "charge_initial")}})
    write_csv(outdir / "state_load_scaling.csv", scaling_rows)

    # Bounded validation cases. These are short representative checks, not a long-term stability claim.
    validation_rows: list[dict[str, Any]] = []
    base_short = {rid: run_arm(f"val_base_{rid}", rid, cfg, phi, refs, kind="base", T_override=1.0)[0] for rid in ("S2_feed_forward", "S3_feedback_off", "S4_full_loop")}
    for name, overrides in [
        ("dt_half", {"dt": cfg["dt"] / 2.0}),
        ("grid_refined", {"N": 56}),
        ("larger_box", {"N": 56, "L": 12.0}),
        ("absorb_width_narrow", {"absorb_width": 1.0}),
        ("absorb_width_wide", {"absorb_width": 2.4}),
    ]:
        vcfg = dict(cfg)
        vcfg.update(overrides)
        try:
            vphi, _ = solve_qball(vcfg)
            vals = {rid: run_arm(f"val_{name}_{rid}", rid, vcfg, vphi, refs, kind="base", T_override=1.0)[0] for rid in ("S2_feed_forward", "S3_feedback_off", "S4_full_loop")}
            ratio_T = vals["S2_feed_forward"]["T_peak"] / max(base_short["S2_feed_forward"]["T_peak"], 1e-12)
            ratio_G = vals["S2_feed_forward"]["G_peak"] / max(base_short["S2_feed_forward"]["G_peak"], 1e-12)
            v_back_delta = {
                "backreaction_delta_core_energy_final": vals["S4_full_loop"]["core_energy_final"] - vals["S3_feedback_off"]["core_energy_final"],
                "backreaction_delta_charge_final": vals["S4_full_loop"]["charge_final"] - vals["S3_feedback_off"]["charge_final"],
                "backreaction_delta_core_amp_final": vals["S4_full_loop"]["core_amp_final"] - vals["S3_feedback_off"]["core_amp_final"],
                "backreaction_delta_width_final": vals["S4_full_loop"]["node_width_final"] - vals["S3_feedback_off"]["node_width_final"],
                "backreaction_delta_shell_flux_max": vals["S4_full_loop"]["shell_flux_max"] - vals["S3_feedback_off"]["shell_flux_max"],
                "backreaction_delta_node_frequency": vals["S4_full_loop"]["node_frequency"] - vals["S3_feedback_off"]["node_frequency"],
            }
            v_back_detected = max(abs(v) for v in v_back_delta.values()) > 1e-8
            validation_rows.append(
                {
                    "case": name,
                    "status": "completed",
                    "grid": vcfg["N"],
                    "box": vcfg["L"],
                    "dt": vcfg["dt"],
                    "T_peak_ratio_to_base": ratio_T,
                    "G_peak_ratio_to_base": ratio_G,
                    **v_back_delta,
                    "backreaction_detected": v_back_detected,
                    "pass": 0.25 < ratio_T < 4.0 and 0.25 < ratio_G < 4.0,
                }
            )
        except Exception as exc:
            validation_rows.append({"case": name, "status": "failed", "reason": str(exc), "pass": False})
    write_csv(outdir / "numerical_validation.csv", validation_rows)

    by_id = {row["run_id"]: row for row in summaries}
    status, intervention_rows, falsification_rows = classify(by_id, validation_rows)
    labels = [status]
    if status == "TG_STATE_LOAD_GEOMETRIC_BACKREACTION_DETECTED":
        labels.insert(0, "TG_STATE_LOAD_FEEDFORWARD_CHAIN_SUPPORTED")
    write_csv(outdir / "intervention_results.csv", intervention_rows)
    write_csv(outdir / "falsification_results.csv", falsification_rows)
    write_csv(outdir / "temporal_response.csv", summaries)
    write_csv(outdir / "geometric_response.csv", summaries)
    write_csv(outdir / "energy_ledger.csv", summaries)
    write_csv(outdir / "shell_flux.csv", summaries)
    write_csv(outdir / "frequency_classification.csv", summaries)
    write_csv(outdir / "backreaction_effects.csv", [row for row in falsification_rows if row["test"] == "backreaction_detected"])
    handoff = [
        "# TG-B1S Technical Handoff",
        "",
        f"Status: `{status}`.",
        f"Run directory: `{outdir.as_posix()}`.",
        "",
        "## Labels",
        "",
        *(f"- `{label}`" for label in labels),
        "",
        "## Rule",
        "",
        "Do not treat this as gravity, photon emission, objective time dilation, geodesic dynamics, universal free fall, or IRER validation.",
    ]
    (outdir / "TECHNICAL_HANDOFF.md").write_text("\n".join(handoff), encoding="utf-8")
    (outdir / "OPEN_QUESTIONS.md").write_text(
        "# TG-B1S Open Questions\n\n- Should the accepted state-load feedback be rerun for longer to test bounded stability?\n- Should TG-B1R now test phase-tension relaxation separately?\n",
        encoding="utf-8",
    )
    write_docs(outdir, status, labels, by_id)
    write_csv(outdir / "artifact_hashes.csv", artifact_hashes(outdir))
    print(json.dumps({"status": status, "labels": labels, "outdir": str(outdir)}, indent=2))


if __name__ == "__main__":
    main()
