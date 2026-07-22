"""TG-S source semantics characterization for the temporal-geometric branch.

This standalone GPU audit classifies candidate resolution sources before any
T/G feedback rerun.  It does not evolve T, G, or geometric feedback.  Full KG
field evolutions use the established WSL/JAX CUDA environment and the validated
C3 Q-ball machinery from ``phase_d_c3_wave.py``.

Bounded labels are source-instrument labels only.  This script does not claim
gravity, photons, objective time dilation, geodesics, universal free fall, or
IRER validation.
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

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jax_scout.phase_d_c3_wave import (  # noqa: E402
    build_kg,
    contract_axis0,
    invariants,
    kg_evolve,
    qball_petviashvili,
)


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


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float), encoding="utf-8")


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
            "# TG-S Discrepancy Report\n\nGPU preflight failed. Full KG evolution was not run.\n\n"
            f"```text\n{exc}\n```\n",
            encoding="utf-8",
        )
        write_json(outdir / "gpu_preflight.json", {"status": "FAILED", "error": str(exc)})
        raise


def config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def make_xyz(N: int, L: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-L / 2.0, L / 2.0, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    R = np.sqrt(X * X + Y * Y + Z * Z)
    return X, Y, Z, R


def roll3(arr: np.ndarray, shift: int, axis: int = 0) -> np.ndarray:
    return np.roll(arr, int(shift), axis=axis)


def fft_grad(psi: np.ndarray, op: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pk = np.fft.fftn(psi)
    gx = np.fft.ifftn(np.asarray(op["ikx"]) * pk)
    gy = np.fft.ifftn(np.asarray(op["iky"]) * pk)
    gz = np.fft.ifftn(np.asarray(op["ikz"]) * pk)
    return gx, gy, gz


def local_fields(psi: np.ndarray, pi: np.ndarray, op: dict[str, Any], cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    gx, gy, gz = fft_grad(psi, op)
    rho = np.abs(psi) ** 2
    grad2 = np.abs(gx) ** 2 + np.abs(gy) ** 2 + np.abs(gz) ** 2
    a, s, f = cfg["a"], cfg["s"], cfg["f"]
    Gpot = a * rho**2 / 2.0 + s * rho**3 / 3.0 + f * rho**4 / 4.0
    energy = np.abs(pi) ** 2 + cfg["c"] ** 2 * grad2 + cfg["m"] ** 2 * rho - Gpot
    charge = np.imag(np.conj(psi) * pi)
    jx = np.imag(np.conj(psi) * gx)
    jy = np.imag(np.conj(psi) * gy)
    jz = np.imag(np.conj(psi) * gz)
    current_mag2 = jx * jx + jy * jy + jz * jz
    mismatch = current_mag2 / (rho + 1e-12)
    z = psi / np.sqrt(rho + 1e-12)
    return {
        "rho": rho,
        "energy_density": energy,
        "charge_density": charge,
        "current_x": jx,
        "current_y": jy,
        "current_z": jz,
        "current_mag": np.sqrt(current_mag2),
        "phase_gradient_cost": mismatch,
        "z_phase": z,
        "grad2": grad2,
    }


def gaussian_kernel_fft(N: int, L: float, ell: float) -> np.ndarray:
    X, Y, Z, _ = make_xyz(N, L)
    # Periodic convolution kernel centered at index 0.
    x = np.fft.ifftshift(X[:, 0, 0])
    y = np.fft.ifftshift(Y[0, :, 0])
    z = np.fft.ifftshift(Z[0, 0, :])
    XX, YY, ZZ = np.meshgrid(x, y, z, indexing="ij")
    ker = np.exp(-(XX * XX + YY * YY + ZZ * ZZ) / (2.0 * ell * ell))
    ker = ker / np.sum(ker)
    return np.fft.fftn(ker)


def convolve_periodic(field: np.ndarray, kernel_k: np.ndarray) -> np.ndarray:
    return np.fft.ifftn(np.fft.fftn(field) * kernel_k)


def coherence_field(fields: dict[str, np.ndarray], kernel_k: np.ndarray) -> np.ndarray:
    rho = fields["rho"]
    numerator = np.abs(convolve_periodic(rho * fields["z_phase"], kernel_k))
    denominator = np.real(convolve_periodic(rho, kernel_k)) + 1e-12
    return np.clip(numerator / denominator, 0.0, 1.0)


def positive_derivative(values: np.ndarray, times: np.ndarray, sign: float = 1.0, method: str = "forward") -> np.ndarray:
    deriv = np.zeros_like(values)
    if len(times) < 2:
        return deriv
    if method == "central" and len(times) > 2:
        dt = times[2:] - times[:-2]
        deriv[1:-1] = (values[2:] - values[:-2]) / dt[:, None, None, None]
        deriv[0] = (values[1] - values[0]) / (times[1] - times[0])
        deriv[-1] = (values[-1] - values[-2]) / (times[-1] - times[-2])
    elif method == "fourth" and len(times) > 4:
        h = float(np.median(np.diff(times)))
        deriv[2:-2] = (-values[4:] + 8 * values[3:-1] - 8 * values[1:-3] + values[:-4]) / (12 * h)
        deriv[0] = (values[1] - values[0]) / h
        deriv[1] = (values[2] - values[0]) / (2 * h)
        deriv[-2] = (values[-1] - values[-3]) / (2 * h)
        deriv[-1] = (values[-1] - values[-2]) / h
    else:
        dt = np.diff(times)
        deriv[1:] = (values[1:] - values[:-1]) / dt[:, None, None, None]
    return np.maximum(sign * deriv, 0.0)


def smooth_time(values: np.ndarray, passes: int = 1) -> np.ndarray:
    out = values.copy()
    for _ in range(passes):
        if len(out) < 3:
            return out
        tmp = out.copy()
        tmp[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        out = tmp
    return out


def normalize_field(field: np.ndarray) -> np.ndarray:
    mx = float(np.max(np.abs(field)))
    if mx <= 1e-30:
        return np.zeros_like(field)
    return field / mx


def source_arrays(fields_seq: list[dict[str, np.ndarray]], times: np.ndarray, kernel_k: np.ndarray, method: str = "forward") -> dict[str, np.ndarray]:
    rho = np.asarray([f["rho"] for f in fields_seq])
    energy = np.asarray([f["energy_density"] for f in fields_seq])
    charge = np.asarray([np.abs(f["charge_density"]) for f in fields_seq])
    mismatch = np.asarray([f["phase_gradient_cost"] for f in fields_seq])
    coherence = np.asarray([coherence_field(f, kernel_k) for f in fields_seq])
    activity = rho / (np.max(rho, axis=(1, 2, 3), keepdims=True) + 1e-12)
    state_raw = 0.5 * energy / (np.max(np.abs(energy), axis=(1, 2, 3), keepdims=True) + 1e-12)
    state_raw += 0.5 * charge / (np.max(np.abs(charge), axis=(1, 2, 3), keepdims=True) + 1e-12)
    lock_raw = positive_derivative(coherence, times, sign=1.0, method=method) * activity
    relax_raw = positive_derivative(mismatch, times, sign=-1.0, method=method)
    p_density = activity
    p_gate = 1.0 / (1.0 + np.exp(-(p_density - 0.5) / 0.08))
    threshold_raw = positive_derivative(p_density, times, sign=1.0, method=method) * p_gate
    return {
        "S_state": np.maximum(state_raw, 0.0),
        "L_lock": lock_raw,
        "R_relax": relax_raw,
        "P_threshold": threshold_raw,
        "coherence": coherence,
        "mismatch": mismatch,
        "rho": rho,
        "energy": energy,
        "charge": charge,
    }


def corr_flat(a: np.ndarray, b: np.ndarray) -> float:
    af = np.ravel(a)
    bf = np.ravel(b)
    if np.std(af) < 1e-30 or np.std(bf) < 1e-30:
        return float("nan")
    return float(np.corrcoef(af, bf)[0, 1])


def source_metrics_for_event(event_id: str, arrays: dict[str, np.ndarray], times: np.ndarray, xyzr: tuple[np.ndarray, ...], dV: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    X, Y, Z, R = xyzr
    rows: list[dict[str, Any]] = []
    dt_total = max(float(times[-1] - times[0]), 1e-12)
    rho = arrays["rho"]
    d_rho = np.zeros_like(rho)
    if len(times) > 1:
        d_rho[1:] = (rho[1:] - rho[:-1]) / np.diff(times)[:, None, None, None]
    event_summary: dict[str, Any] = {"event_id": event_id}
    for source_name in ("S_state", "L_lock", "R_relax", "P_threshold"):
        src = arrays[source_name]
        spatial_total = np.sum(src, axis=(1, 2, 3)) * dV
        integral = float(np.trapezoid(spatial_total, times))
        peak = float(np.max(src))
        peak_idx = int(np.argmax(spatial_total)) if len(spatial_total) else 0
        onset = float("nan")
        if peak > 0.0:
            active = np.flatnonzero(spatial_total >= 0.1 * np.max(spatial_total))
            if len(active):
                onset = float(times[int(active[0])])
        time_weights = spatial_total
        if np.sum(time_weights) > 1e-30:
            t_centroid = float(np.sum(times * time_weights) / np.sum(time_weights))
        else:
            t_centroid = float("nan")
        src_sum = np.sum(src, axis=0)
        mass = float(np.sum(src_sum) * dV)
        if mass > 1e-30:
            cx = float(np.sum(X * src_sum) * dV / mass)
            cy = float(np.sum(Y * src_sum) * dV / mass)
            cz = float(np.sum(Z * src_sum) * dV / mass)
            radius = float(np.sqrt(cx * cx + cy * cy + cz * cz))
            width = float(np.sqrt(max(np.sum(((R - radius) ** 2) * src_sum) * dV / mass, 0.0)))
        else:
            cx = cy = cz = radius = width = float("nan")
        support = float(np.mean(src > 0.05 * peak)) if peak > 0 else 0.0
        rows.append(
            {
                "event_id": event_id,
                "source_family": source_name,
                "source_integral": integral,
                "source_peak": peak,
                "source_onset": onset,
                "source_time_centroid": t_centroid,
                "source_centroid_x": cx,
                "source_centroid_y": cy,
                "source_centroid_z": cz,
                "source_centroid_radius": radius,
                "source_width": width,
                "support_fraction": support,
                "negative_fraction": 0.0,
                "correlation_with_density": corr_flat(src, rho),
                "correlation_with_d_density_dt": corr_flat(src, d_rho),
                "correlation_with_energy_density": corr_flat(src, arrays["energy"]),
                "correlation_with_current": corr_flat(src, np.sqrt(np.maximum(arrays["mismatch"] * rho, 0.0))),
                "peak_time_index": peak_idx,
            }
        )
        event_summary[f"{source_name}_integral"] = integral
        event_summary[f"{source_name}_peak"] = peak
    return rows, event_summary


def energy_regions(psi: np.ndarray, pi: np.ndarray, op: dict[str, Any], cfg: dict[str, Any], R: np.ndarray) -> dict[str, float]:
    fields = local_fields(psi, pi, op, cfg)
    dV = (cfg["L"] / cfg["N"]) ** 3
    energy = fields["energy_density"]
    masks = {
        "core": R < 2.0,
        "intermediate": (R >= 2.0) & (R < 4.0),
        "exterior": R >= 4.0,
    }
    return {name: float(np.sum(energy[mask]) * dV) for name, mask in masks.items()}


def flux_proxy(psi: np.ndarray, pi: np.ndarray, op: dict[str, Any], R: np.ndarray, radius: float) -> float:
    gx, gy, gz = fft_grad(psi, op)
    flux_x = -np.real(np.conj(pi) * gx)
    flux_y = -np.real(np.conj(pi) * gy)
    flux_z = -np.real(np.conj(pi) * gz)
    eps = 1e-12
    x = np.linspace(-0.5, 0.5, psi.shape[0], endpoint=False)
    Xn, Yn, Zn = np.meshgrid(x, x, x, indexing="ij")
    rn = np.sqrt(Xn * Xn + Yn * Yn + Zn * Zn) + eps
    radial = (flux_x * Xn + flux_y * Yn + flux_z * Zn) / rn
    shell = np.abs(R - radius) < (radius * 0.05 + 0.1)
    return float(np.mean(np.abs(radial[shell]))) if np.any(shell) else 0.0


def make_initial_state(event_id: str, phi: np.ndarray, w: float, op: dict[str, Any], cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    N, L = cfg["N"], cfg["L"]
    x = np.linspace(-L / 2.0, L / 2.0, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    if event_id == "stationary_node":
        psi = phi.astype(np.complex128)
        pi = (-1j * w * psi).astype(np.complex128)
    elif event_id == "amplitude_breathing":
        psi = (1.04 * phi).astype(np.complex128)
        pi = (-1j * w * psi).astype(np.complex128)
    elif event_id == "global_phase_rotation":
        phase = np.exp(1j * 1.173)
        psi = (phase * phi).astype(np.complex128)
        pi = (-1j * w * psi).astype(np.complex128)
    elif event_id == "translated_node":
        psi = roll3(phi, max(1, N // 8), axis=0).astype(np.complex128)
        pi = (-1j * w * psi).astype(np.complex128)
    elif event_id == "boosted_node":
        v = 0.08 * cfg["c"]
        gam = 1.0 / math.sqrt(1.0 - (v / cfg["c"]) ** 2)
        k = gam * w * v / cfg["c"] ** 2
        phi_c = contract_axis0(phi, gam, L)
        gx = np.asarray(jnp.fft.ifftn(op["ikx"] * jnp.fft.fftn(jnp.asarray(phi_c.astype(np.complex128)))))
        carrier = np.exp(1j * k * X)
        psi = (phi_c * carrier).astype(np.complex128)
        pi = ((-v * gx - 1j * gam * w * phi_c) * carrier).astype(np.complex128)
    elif event_id == "phase_perturb_relaxation":
        phase = 0.75 * np.sin(2.0 * np.pi * X / L) * np.exp(-(X * X + Y * Y + Z * Z) / 9.0)
        psi = (phi * np.exp(1j * phase)).astype(np.complex128)
        pi = (-1j * w * psi).astype(np.complex128)
    elif event_id == "density_matched_scrambled":
        phase = 1.1 * np.sin(6.0 * np.pi * X / L) + 0.7 * np.sin(4.0 * np.pi * Y / L + 0.3)
        psi = (np.abs(phi) * np.exp(1j * phase)).astype(np.complex128)
        pi = (-1j * w * psi).astype(np.complex128)
    elif event_id == "dephasing_unlocking":
        phase = 0.45 * np.sin(2.0 * np.pi * X / L) * np.exp(-(X * X + Y * Y + Z * Z) / 9.0)
        psi = (phi * np.exp(1j * phase)).astype(np.complex128)
        # Momentum pushes the phase perturbation larger rather than letting it relax.
        pi = (-1j * w * psi + 0.12j * np.sin(2.0 * np.pi * X / L) * psi).astype(np.complex128)
    elif event_id == "two_packet_phase_locking":
        sep = 4.0
        shift = int(round((sep / 2.0) / (L / N)))
        left = roll3(phi, -shift, axis=0)
        right = roll3(phi, shift, axis=0)
        psi = (left + np.exp(1j * 0.35 * np.pi) * right).astype(np.complex128)
        pi = (-1j * w * psi).astype(np.complex128)
    else:
        raise ValueError(event_id)
    return psi, pi


def evolve_samples(psi: np.ndarray, pi: np.ndarray, op: dict[str, Any], cfg: dict[str, Any], T: float, sample_dt: float) -> list[dict[str, Any]]:
    steps = int(round(T / cfg["dt"]))
    every = max(1, int(round(sample_dt / cfg["dt"])))
    chunks = steps // every
    psi_k = jnp.fft.fftn(jnp.asarray(psi))
    pi_k = jnp.fft.fftn(jnp.asarray(pi))
    samples: list[dict[str, Any]] = []

    def append_sample(t: float, pk: Any, qk: Any) -> None:
        cur = np.asarray(jnp.fft.ifftn(pk))
        vel = np.asarray(jnp.fft.ifftn(qk))
        samples.append({"t": t, "psi": cur, "pi": vel})

    append_sample(0.0, psi_k, pi_k)
    for idx in range(chunks):
        psi_k, pi_k = kg_evolve(psi_k, pi_k, op, cfg["a"], cfg["s"], cfg["f"], every)
        psi_k.block_until_ready()
        pi_k.block_until_ready()
        append_sample((idx + 1) * every * cfg["dt"], psi_k, pi_k)
    return samples


def save_selected_snapshots(outdir: Path, event_id: str, samples: list[dict[str, Any]], arrays: dict[str, np.ndarray], fields_seq: list[dict[str, np.ndarray]]) -> None:
    snapdir = outdir / "source_snapshots"
    snapdir.mkdir(exist_ok=True)
    picks = sorted(set([0, len(samples) // 2, len(samples) - 1]))
    for idx in picks:
        fields = fields_seq[idx]
        np.savez_compressed(
            snapdir / f"{event_id}_sample{idx:03d}.npz",
            t=np.asarray(samples[idx]["t"]),
            phi=samples[idx]["psi"],
            Pi=samples[idx]["pi"],
            rho=fields["rho"],
            energy_density=fields["energy_density"],
            charge_density=fields["charge_density"],
            current=fields["current_mag"],
            phase_coherence=arrays["coherence"][idx],
            phase_gradient_cost=arrays["mismatch"][idx],
            R_current=normalize_field(arrays["R_relax"])[idx],
            R_threshold=normalize_field(arrays["P_threshold"])[idx],
            R_lock=normalize_field(arrays["L_lock"])[idx],
            S_state=normalize_field(arrays["S_state"])[idx],
        )


def run_event(event_id: str, phi: np.ndarray, w: float, op: dict[str, Any], cfg: dict[str, Any], outdir: Path, kernel_k: np.ndarray, xyzr: tuple[np.ndarray, ...]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], tuple[np.ndarray, np.ndarray] | None]:
    duration = {
        "two_packet_phase_locking": 3.0,
        "boosted_node": 1.4,
    }.get(event_id, 2.0)
    psi, pi = make_initial_state(event_id, phi, w, op, cfg)
    samples = evolve_samples(psi, pi, op, cfg, duration, cfg["sample_dt"])
    fields_seq = [local_fields(s["psi"], s["pi"], op, cfg) for s in samples]
    times = np.asarray([s["t"] for s in samples])
    arrays = source_arrays(fields_seq, times, kernel_k, method="forward")
    dV = (cfg["L"] / cfg["N"]) ** 3
    source_rows, summary = source_metrics_for_event(event_id, arrays, times, xyzr, dV)
    inv0 = invariants(jnp.fft.fftn(jnp.asarray(samples[0]["psi"])), jnp.fft.fftn(jnp.asarray(samples[0]["pi"])), op, cfg["a"], cfg["s"], cfg["f"])
    inv1 = invariants(jnp.fft.fftn(jnp.asarray(samples[-1]["psi"])), jnp.fft.fftn(jnp.asarray(samples[-1]["pi"])), op, cfg["a"], cfg["s"], cfg["f"])
    _, _, _, R = xyzr
    shell_flux = max(flux_proxy(s["psi"], s["pi"], op, R, radius=0.35 * cfg["L"]) for s in samples)
    regions0 = energy_regions(samples[0]["psi"], samples[0]["pi"], op, cfg, R)
    regions1 = energy_regions(samples[-1]["psi"], samples[-1]["pi"], op, cfg, R)
    event_row = {
        "event_id": event_id,
        "T": duration,
        "samples": len(samples),
        "dE_rel": abs(inv1["E"] - inv0["E"]) / (abs(inv0["E"]) + 1e-30),
        "dQ_rel": abs(inv1["Q"] - inv0["Q"]) / (abs(inv0["Q"]) + 1e-30),
        "mass_rel_change": abs(inv1["mass"] - inv0["mass"]) / (abs(inv0["mass"]) + 1e-30),
        "amp_initial": inv0["amp"],
        "amp_final": inv1["amp"],
        "shell_flux_floor_proxy": shell_flux,
        "core_energy_initial": regions0["core"],
        "core_energy_final": regions1["core"],
        "core_energy_delta": regions1["core"] - regions0["core"],
        "intermediate_energy_delta": regions1["intermediate"] - regions0["intermediate"],
        "exterior_energy_delta": regions1["exterior"] - regions0["exterior"],
    }
    save_selected_snapshots(outdir, event_id, samples, arrays, fields_seq)
    mid = samples[len(samples) // 2]
    mid_state = (mid["psi"], mid["pi"]) if event_id == "phase_perturb_relaxation" else None
    return source_rows, summary, [event_row], fields_seq, mid_state


def run_time_reverse(mid_state: tuple[np.ndarray, np.ndarray], op: dict[str, Any], cfg: dict[str, Any], outdir: Path, kernel_k: np.ndarray, xyzr: tuple[np.ndarray, ...]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    psi_mid, pi_mid = mid_state
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for event_id, pi0 in [("time_forward_from_midpoint", pi_mid), ("time_reversed_from_midpoint", -pi_mid)]:
        samples = evolve_samples(psi_mid, pi0, op, cfg, 1.0, cfg["sample_dt"])
        fields_seq = [local_fields(s["psi"], s["pi"], op, cfg) for s in samples]
        times = np.asarray([s["t"] for s in samples])
        arrays = source_arrays(fields_seq, times, kernel_k, method="forward")
        dV = (cfg["L"] / cfg["N"]) ** 3
        source_rows, summary = source_metrics_for_event(event_id, arrays, times, xyzr, dV)
        rows.extend(source_rows)
        summaries.append(summary)
        save_selected_snapshots(outdir, event_id, samples, arrays, fields_seq)
    return rows, summaries


def derivative_audit(phi: np.ndarray, w: float, cfg: dict[str, Any], kernel_k: np.ndarray, xyzr: tuple[np.ndarray, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for factor in (1.0, 0.5, 0.25):
        local_cfg = dict(cfg)
        local_cfg["dt"] = cfg["dt"] * factor
        op = build_kg(local_cfg["N"], local_cfg["L"], local_cfg["c"], local_cfg["m"], local_cfg["dt"])
        psi, pi = make_initial_state("phase_perturb_relaxation", phi, w, op, local_cfg)
        samples = evolve_samples(psi, pi, op, local_cfg, 1.0, cfg["sample_dt"])
        fields_seq = [local_fields(s["psi"], s["pi"], op, local_cfg) for s in samples]
        times = np.asarray([s["t"] for s in samples])
        for method in ("forward", "central", "fourth", "smoothed_forward"):
            seq = fields_seq
            arrays = source_arrays(seq, times, kernel_k, method="forward" if method == "smoothed_forward" else method)
            if method == "smoothed_forward":
                arrays["L_lock"] = positive_derivative(smooth_time(arrays["coherence"], passes=2), times, sign=1.0, method="forward")
                arrays["R_relax"] = positive_derivative(smooth_time(arrays["mismatch"], passes=2), times, sign=-1.0, method="forward")
            source_rows, _ = source_metrics_for_event(f"derivative_dt_{factor}_{method}", arrays, times, xyzr, (cfg["L"] / cfg["N"]) ** 3)
            for row in source_rows:
                if row["source_family"] in ("L_lock", "R_relax"):
                    rows.append(
                        {
                            "dt_factor": factor,
                            "dt": local_cfg["dt"],
                            "derivative_method": method,
                            "source_family": row["source_family"],
                            "source_integral": row["source_integral"],
                            "source_peak": row["source_peak"],
                            "source_onset": row["source_onset"],
                        }
                    )
    return rows


def classify_sources(local_rows: list[dict[str, Any]], event_rows: list[dict[str, Any]], derivative_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str], str]:
    metric = {(r["event_id"], r["source_family"]): r for r in local_rows}

    def peak(event: str, source: str) -> float:
        return float(metric.get((event, source), {}).get("source_peak", 0.0))

    def integral(event: str, source: str) -> float:
        return float(metric.get((event, source), {}).get("source_integral", 0.0))

    stationary_lock = integral("stationary_node", "L_lock")
    stationary_relax = integral("stationary_node", "R_relax")
    locking_lock = integral("two_packet_phase_locking", "L_lock")
    dephase_lock = integral("dephasing_unlocking", "L_lock")
    breathing_lock = integral("amplitude_breathing", "L_lock")
    phase_relax = integral("phase_perturb_relaxation", "R_relax")
    scrambled_relax = integral("density_matched_scrambled", "R_relax")
    dephase_relax = integral("dephasing_unlocking", "R_relax")
    state_stationary = integral("stationary_node", "S_state")
    state_global = integral("global_phase_rotation", "S_state")
    state_trans = integral("translated_node", "S_state")
    threshold_breath = integral("amplitude_breathing", "P_threshold")
    threshold_stationary = integral("stationary_node", "P_threshold")

    derivative_ok = True
    for source in ("L_lock", "R_relax"):
        vals = [float(r["source_integral"]) for r in derivative_rows if r["source_family"] == source and r["derivative_method"] == "forward"]
        if len(vals) >= 3:
            ref = max(abs(vals[-1]), 1e-12)
            if abs(vals[0] - vals[-1]) / ref > 0.5:
                derivative_ok = False

    rows = []
    state_invariant = state_stationary > 1e-6 and abs(state_global - state_stationary) / max(state_stationary, 1e-12) < 0.05 and abs(state_trans - state_stationary) / max(state_stationary, 1e-12) < 0.1
    rows.append(
        {
            "source_family": "S_state",
            "classification": "NODE_STATE_LOAD" if state_invariant else "DENSITY_OR_BREATHING_ACTIVITY",
            "supported_label": "TG_NODE_STATE_LOAD_SOURCE_SUPPORTED" if state_invariant else "",
            "passes_semantics_gate": state_invariant,
            "reason": "stable nonzero stationary load; global phase and translation invariant" if state_invariant else "state source not invariant/stable enough",
            "stationary_integral": state_stationary,
        }
    )

    lock_supported = stationary_lock < 0.1 * max(locking_lock, 1e-12) and breathing_lock < 0.5 * max(locking_lock, 1e-12) and dephase_lock < locking_lock and derivative_ok
    rows.append(
        {
            "source_family": "L_lock",
            "classification": "PHASE_LOCKING_RATE" if lock_supported else "DENSITY_OR_BREATHING_ACTIVITY",
            "supported_label": "TG_PHASE_LOCKING_SOURCE_SUPPORTED" if lock_supported else "",
            "passes_semantics_gate": lock_supported,
            "reason": "active for two-packet locking and quiet in nulls" if lock_supported else "does not cleanly separate locking from null/breathing/unlocking controls",
            "stationary_integral": stationary_lock,
            "locking_integral": locking_lock,
            "breathing_integral": breathing_lock,
            "unlocking_integral": dephase_lock,
        }
    )

    relax_supported = phase_relax > 5.0 * max(stationary_relax, 1e-12) and scrambled_relax > 5.0 * max(stationary_relax, 1e-12) and dephase_relax < max(phase_relax, scrambled_relax) and derivative_ok
    rows.append(
        {
            "source_family": "R_relax",
            "classification": "PHASE_TENSION_RELAXATION" if relax_supported else "NUMERICAL_DERIVATIVE_NOISE",
            "supported_label": "TG_PHASE_TENSION_RELAXATION_SOURCE_SUPPORTED" if relax_supported else "",
            "passes_semantics_gate": relax_supported,
            "reason": "detects mismatch relaxation, including scrambled relaxation, and is quiet on the stationary node" if relax_supported else "relaxation source failed direction/refinement/null criteria",
            "stationary_integral": stationary_relax,
            "phase_perturb_integral": phase_relax,
            "scrambled_integral": scrambled_relax,
            "unlocking_integral": dephase_relax,
        }
    )

    threshold_density = threshold_breath > max(threshold_stationary, 1e-12)
    rows.append(
        {
            "source_family": "P_threshold",
            "classification": "DENSITY_OR_BREATHING_ACTIVITY" if threshold_density else "NUMERICAL_DERIVATIVE_NOISE",
            "supported_label": "",
            "passes_semantics_gate": False,
            "reason": "comparison-only threshold source; cannot promote feedback by itself",
            "stationary_integral": threshold_stationary,
            "breathing_integral": threshold_breath,
        }
    )

    labels = [str(row["supported_label"]) for row in rows if row["supported_label"]]
    status = labels[0] if labels else "TG_RESOLUTION_SOURCE_NOT_OPERATIONALIZED"
    return rows, labels, status


def write_docs(outdir: Path, status: str, labels: list[str], classification_rows: list[dict[str, Any]], baseline_pass: bool) -> None:
    docdir = ROOT / "docs" / "gravity_maturity"
    docdir.mkdir(parents=True, exist_ok=True)
    results = [
        "# TG-S Source Semantics Characterization Results",
        "",
        "Timestamp: 2026-07-14.",
        f"Run directory: `{outdir.as_posix()}`.",
        f"Status: `{status}`.",
        "",
        "## Summary",
        "",
        "This audit characterizes source instruments only. It does not rerun temporal/geometric feedback and does not relabel `TG_SOURCE_SEMANTICS_FAILED_ON_KG_NODE`.",
        f"Baseline node gate: `{baseline_pass}`.",
        "",
        "## Source Classifications",
        "",
        "| source | classification | passes | label | reason |",
        "| --- | --- | --- | --- | --- |",
        *[
            f"| {r['source_family']} | {r['classification']} | {r['passes_semantics_gate']} | {r['supported_label']} | {r['reason']} |"
            for r in classification_rows
        ],
        "",
        "## Feedback Rerun Rule",
        "",
        "A future TG-B1 rerun must choose exactly one supported source hypothesis and state it explicitly.",
    ]
    (docdir / "TG_S_RESULTS.md").write_text("\n".join(results), encoding="utf-8")
    write_json(
        docdir / "TG_S_SUMMARY.json",
        {
            "status": status,
            "labels": labels,
            "run_directory": str(outdir),
            "baseline_node_gate_passed": baseline_pass,
            "classifications": classification_rows,
            "feedback_rerun_allowed": bool(labels),
        },
    )
    (docdir / "TG_S_DOCUMENTATION_INPUTS.md").write_text(
        "\n".join(
            [
                "# TG-S Documentation Inputs",
                "",
                f"- Bounded status: `{status}`.",
                f"- Supported source labels: `{', '.join(labels) if labels else 'none'}`.",
                f"- Artifacts: `{outdir.as_posix()}`.",
                "- Preserve prior TG-B1 result: `TG_SOURCE_SEMANTICS_FAILED_ON_KG_NODE`.",
                "- Do not claim gravity, photon emission, objective time dilation, geodesics, universal free fall, or IRER validation.",
                "- Do not update the master catalogue until reviewed.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--stages", default="all")
    ap.add_argument("--matrix", default="bounded")
    ap.add_argument("--stop-on-gpu-fail", default="true")
    ap.add_argument("--N", type=int, default=48)
    ap.add_argument("--L", type=float, default=10.0)
    ap.add_argument("--c", type=float, default=0.5477)
    ap.add_argument("--m", type=float, default=1.0)
    ap.add_argument("--a", type=float, default=0.8)
    ap.add_argument("--s", type=float, default=-0.5)
    ap.add_argument("--f", type=float, default=-0.1)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--w", type=float, default=0.964)
    ap.add_argument("--sample-dt", type=float, default=0.05)
    ap.add_argument("--coherence-ell", type=float, default=0.45)
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.out) if args.out else ROOT / "sweep_runs" / f"TG_SOURCE_SEMANTICS_GPU_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "git_state_before.txt").write_text(git_state(), encoding="utf-8")
    pf = preflight(outdir)

    cfg = vars(args).copy()
    cfg["config_hash"] = config_hash(cfg)
    write_json(outdir / "environment_versions.json", {**pf, "stage": "TG-S_SOURCE_SEMANTICS_CHARACTERIZATION"})
    write_json(outdir / "source_family_definitions.json", {
        "S_state": "bounded normalized local KG energy/absolute charge density; state load, not a rate",
        "L_lock": "R_lock=[d_t C_l]_+ W_activity with C_l=|K_l*(rho*z)|/(K_l*rho+eps)",
        "R_relax": "R_relax=[-d_t K_mismatch]_+, K_mismatch=|j|^2/(rho+eps)",
        "P_threshold": "comparison-only threshold/PAS-like density activity source",
    })

    op = build_kg(args.N, args.L, args.c, args.m, args.dt)
    mu = args.m**2 - args.w**2
    phi = None
    prof = None
    for sig in (1.5, 1.2, 1.8, 2.0):
        candidate, details = qball_petviashvili(op, args.a, args.s, args.f, mu, sig=sig)
        if candidate is not None and details["residual"] < 1e-6 and details["occ"] < 0.5:
            phi, prof = candidate, details
            break
    if phi is None:
        write_csv(outdir / "baseline_node_contract.csv", [{"status": "TG_SOURCE_NODE_NOT_STATIONARY", "reason": "qball solve failed"}])
        write_docs(outdir, "TG_SOURCE_NODE_NOT_STATIONARY", [], [], False)
        write_csv(outdir / "artifact_hashes.csv", artifact_hashes(outdir))
        print(json.dumps({"status": "TG_SOURCE_NODE_NOT_STATIONARY", "outdir": str(outdir)}, indent=2))
        return

    X, Y, Z, R = make_xyz(args.N, args.L)
    xyzr = (X, Y, Z, R)
    kernel_k = gaussian_kernel_fft(args.N, args.L, args.coherence_ell)

    # Baseline gate on an exact rest Q-ball.
    psi0 = phi.astype(np.complex128)
    pi0 = (-1j * args.w * psi0).astype(np.complex128)
    base_samples = evolve_samples(psi0, pi0, op, cfg, 4.0, args.sample_dt)
    inv0 = invariants(jnp.fft.fftn(jnp.asarray(base_samples[0]["psi"])), jnp.fft.fftn(jnp.asarray(base_samples[0]["pi"])), op, args.a, args.s, args.f)
    inv1 = invariants(jnp.fft.fftn(jnp.asarray(base_samples[-1]["psi"])), jnp.fft.fftn(jnp.asarray(base_samples[-1]["pi"])), op, args.a, args.s, args.f)
    rho0 = np.abs(base_samples[0]["psi"]) ** 2
    rho1 = np.abs(base_samples[-1]["psi"]) ** 2
    overlap = float(np.sum(rho0 * rho1) / (math.sqrt(np.sum(rho0 * rho0) * np.sum(rho1 * rho1)) + 1e-30))
    regions0 = energy_regions(base_samples[0]["psi"], base_samples[0]["pi"], op, cfg, R)
    regions1 = energy_regions(base_samples[-1]["psi"], base_samples[-1]["pi"], op, cfg, R)
    max_flux = max(flux_proxy(s["psi"], s["pi"], op, R, radius=0.35 * args.L) for s in base_samples)
    dE = abs(inv1["E"] - inv0["E"]) / (abs(inv0["E"]) + 1e-30)
    dQ = abs(inv1["Q"] - inv0["Q"]) / (abs(inv0["Q"]) + 1e-30)
    max_region_delta = max(abs(regions1[k] - regions0[k]) / (abs(regions0[k]) + 1e-30) for k in regions0)
    baseline_pass = bool(dE <= 1e-8 and dQ <= 1e-8 and overlap >= 0.995 and max_flux < 1e-8 and max_region_delta < 1e-3)
    baseline_rows = [
        {
            "status": "TG_SOURCE_NODE_BASELINE_CLOSED" if baseline_pass else "TG_SOURCE_NODE_NOT_STATIONARY",
            "qball_residual": prof["residual"],
            "qball_amp": prof["amp"],
            "qball_occ": prof["occ"],
            "dE_rel": dE,
            "dQ_rel": dQ,
            "profile_overlap": overlap,
            "max_shell_flux_proxy": max_flux,
            "core_energy_initial": regions0["core"],
            "core_energy_final": regions1["core"],
            "intermediate_energy_initial": regions0["intermediate"],
            "intermediate_energy_final": regions1["intermediate"],
            "exterior_energy_initial": regions0["exterior"],
            "exterior_energy_final": regions1["exterior"],
            "old_TG_B1_core_drop_explanation": "old pilot used a Gaussian/chirped non-Q-ball node; this exact C3 Q-ball tests whether the source issue survives a valid stationary node",
        }
    ]
    write_csv(outdir / "baseline_node_contract.csv", baseline_rows)
    if not baseline_pass:
        write_docs(outdir, "TG_SOURCE_NODE_NOT_STATIONARY", [], [], False)
        (outdir / "TECHNICAL_HANDOFF.md").write_text("# TG-S Technical Handoff\n\nStatus: `TG_SOURCE_NODE_NOT_STATIONARY`.\n", encoding="utf-8")
        (outdir / "OPEN_QUESTIONS.md").write_text("# TG-S Open Questions\n\n- Why did the validated Q-ball baseline fail in this configuration?\n", encoding="utf-8")
        write_csv(outdir / "artifact_hashes.csv", artifact_hashes(outdir))
        print(json.dumps({"status": "TG_SOURCE_NODE_NOT_STATIONARY", "outdir": str(outdir)}, indent=2))
        return

    event_ids = [
        "stationary_node",
        "amplitude_breathing",
        "global_phase_rotation",
        "translated_node",
        "boosted_node",
        "phase_perturb_relaxation",
        "two_packet_phase_locking",
        "density_matched_scrambled",
        "dephasing_unlocking",
    ]
    all_source_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    mid_state: tuple[np.ndarray, np.ndarray] | None = None
    for event_id in event_ids:
        src_rows, summary, ev_rows, _, maybe_mid = run_event(event_id, phi, args.w, op, cfg, outdir, kernel_k, xyzr)
        all_source_rows.extend(src_rows)
        summaries.append(summary)
        event_rows.extend(ev_rows)
        if maybe_mid is not None:
            mid_state = maybe_mid

    reverse_rows: list[dict[str, Any]] = []
    if mid_state is not None:
        tr_rows, tr_summaries = run_time_reverse(mid_state, op, cfg, outdir, kernel_k, xyzr)
        all_source_rows.extend(tr_rows)
        summaries.extend(tr_summaries)
        reverse_rows = [row for row in tr_rows if row["source_family"] in ("L_lock", "R_relax")]
    write_csv(outdir / "event_suite.csv", event_rows)
    write_csv(outdir / "local_source_metrics.csv", all_source_rows)
    write_csv(outdir / "forward_reverse_comparison.csv", reverse_rows)

    # Density contamination and invariance tables are filtered views with explicit tests.
    density_rows = [
        row for row in all_source_rows
        if row["event_id"] in ("amplitude_breathing", "density_matched_scrambled", "stationary_node")
    ]
    invariance_rows = [
        row for row in all_source_rows
        if row["event_id"] in ("stationary_node", "global_phase_rotation", "translated_node", "boosted_node")
    ]
    write_csv(outdir / "density_contamination.csv", density_rows)
    write_csv(outdir / "invariance_results.csv", invariance_rows)

    deriv_rows = derivative_audit(phi, args.w, cfg, kernel_k, xyzr)
    write_csv(outdir / "derivative_convergence.csv", deriv_rows)
    class_rows, labels, status = classify_sources(all_source_rows, event_rows, deriv_rows)
    write_csv(outdir / "source_semantics_classification.csv", class_rows)

    falsification_rows = [
        {
            "test": "preserve_TG_B1_negative",
            "pass": True,
            "evidence": "TG_SOURCE_SEMANTICS_FAILED_ON_KG_NODE left unchanged; TG-S is a new source audit",
        },
        {
            "test": "baseline_node_gate",
            "pass": baseline_pass,
            "evidence": "baseline_node_contract.csv",
        },
        {
            "test": "at_least_one_non_threshold_source_supported",
            "pass": bool(labels),
            "labels": ",".join(labels),
        },
    ]
    if not labels:
        falsification_rows.append(
            {
                "test": "resolution_source_operationalized",
                "pass": False,
                "label": "TG_RESOLUTION_SOURCE_NOT_OPERATIONALIZED",
            }
        )
    write_csv(outdir / "falsification_results.csv", falsification_rows)

    handoff = [
        "# TG-S Technical Handoff",
        "",
        f"Status: `{status}`.",
        f"Run directory: `{outdir.as_posix()}`.",
        "",
        "## Supported Source Labels",
        "",
        *(f"- `{label}`" for label in labels),
        "",
        "## Gate Rule",
        "",
        "Do not rerun TG-B1 feedback unless a reviewer accepts one supported source family and chooses exactly one feedback hypothesis.",
    ]
    (outdir / "TECHNICAL_HANDOFF.md").write_text("\n".join(handoff), encoding="utf-8")
    (outdir / "OPEN_QUESTIONS.md").write_text(
        "\n".join(
            [
                "# TG-S Open Questions",
                "",
                "- Should a future TG-B1 rerun test `STATE_LOAD_FEEDBACK`, `PHASE_LOCKING_FEEDBACK`, or `PHASE_TENSION_RELAXATION_FEEDBACK` first?",
                "- Are the source-family thresholds strict enough for external review?",
                "- Should the two-packet locking event be repeated in the larger validated C3 two-Q-ball box before feedback use?",
            ]
        ),
        encoding="utf-8",
    )
    write_docs(outdir, status, labels, class_rows, baseline_pass)
    write_csv(outdir / "artifact_hashes.csv", artifact_hashes(outdir))
    print(json.dumps({"status": status, "labels": labels, "outdir": str(outdir)}, indent=2))


if __name__ == "__main__":
    main()
