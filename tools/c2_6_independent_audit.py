"""Independent C2.6 geometry-off audit.

Runs in the WSL JAX environment. Diagnostic-only; no production physics paths
are modified by this script.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_scout import core_saturation_search as css, physics  # noqa: E402
from jax_scout.phase_d_c1_transport import _evolve_chunk  # noqa: E402
from jax_scout.phase_d_c2_3_exact_soliton import petviashvili  # noqa: E402


def max_delta(a, b) -> float:
    return float(jnp.max(jnp.abs(jnp.asarray(a) - jnp.asarray(b))))


def circular_peak_index(psi: np.ndarray) -> int:
    rho = np.abs(psi) ** 2
    return int(np.unravel_index(int(rho.argmax()), rho.shape)[0])


def full_rhs(pk, ops):
    return ops.L_k * pk + physics.n_op(pk, ops)


def rk4_step(pk, ops, dt):
    k1 = full_rhs(pk, ops)
    k2 = full_rhs(pk + 0.5 * dt * k1, ops)
    k3 = full_rhs(pk + 0.5 * dt * k2, ops)
    k4 = full_rhs(pk + dt * k3, ops)
    return pk + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def norm_flux(pk, ops) -> float:
    psi = jnp.fft.ifftn(pk)
    rhs = jnp.fft.ifftn(full_rhs(pk, ops))
    norm = jnp.sum(jnp.abs(psi) ** 2)
    raw = 2.0 * jnp.real(jnp.sum(jnp.conj(psi) * rhs))
    return float(raw / jnp.maximum(jnp.abs(norm), 1e-300))


def polynomial_only_nk(pk, ops):
    psi = jnp.fft.ifftn(pk)
    rho = jnp.maximum(jnp.real(psi) ** 2 + jnp.imag(psi) ** 2, ops.rho_floor)
    n_real = (ops.a * psi * rho + ops.s * psi * rho**2 + ops.f * psi * rho**3) * ops.kfac
    return jnp.fft.fftn(n_real) * ops.dealias_mask


def packet(N, L, k, amp=0.01, sigma=1.5):
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    return (amp * np.exp(-(X**2 + Y**2 + Z**2) / (2 * sigma**2)) * np.exp(1j * k * X)).astype(np.complex128)


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    N = int(args.N)
    L = float(args.L)
    dt = float(args.dt)
    D = float(args.D)
    k1 = 2.0 * np.pi / L
    base = dict(css.FEB)
    base["param_a"] = float(css.FEB["param_a"]) * 1.15

    parity_base = physics.build_operators(N, L, dt, base)
    parity_zero = physics.build_operators(N, L, dt, {**base, "param_D_imag": 0.0})
    parity_disp = physics.build_operators(N, L, dt, {**base, "param_D_imag": 0.05})

    p = {
        **css.FEB,
        "param_a": 0.8,
        "param_s": -0.2,
        "param_f": 0.0,
        "param_D": D,
        "kinetic_mode": "conservative",
        "param_a_coupling": 0.0,
        "param_geom_off": True,
    }
    ops_flat = physics.build_operators(N, L, dt, p)
    ops_old = physics.build_operators(N, L, dt, {**p, "param_geom_off": False})

    rng = np.random.default_rng(1234)
    psi_rand = (rng.normal(size=(N, N, N)) + 1j * rng.normal(size=(N, N, N))).astype(np.complex128) * 1e-3
    pk_rand = physics.initial_psi_k(jnp.asarray(psi_rand), ops_flat)
    flat_n_diff = max_delta(physics.n_op(pk_rand, ops_flat), polynomial_only_nk(pk_rand, ops_flat))

    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    X, _, _ = np.meshgrid(x, x, x, indexing="ij")
    plane = (1e-6 * np.exp(1j * k1 * X)).astype(np.complex128)
    pk_plane_old = physics.initial_psi_k(jnp.asarray(plane), ops_old)
    old_ratio = complex(np.asarray(physics.step(pk_plane_old, ops_old))[1, 0, 0] / np.asarray(pk_plane_old)[1, 0, 0])
    old_phase = float(np.angle(old_ratio))
    expected_phase = float(-D * k1**2 * dt)
    old_d_eff_ratio = old_phase / expected_phase if expected_phase else None

    psi_packet = packet(N, L, k1)
    pk_packet = physics.initial_psi_k(jnp.asarray(psi_packet), ops_flat)
    packet_peak = [circular_peak_index(psi_packet)]
    for _ in range(5):
        pk_packet = _evolve_chunk(pk_packet, ops_flat, 100)
        packet_peak.append(circular_peak_index(np.asarray(jnp.fft.ifftn(pk_packet))))

    psi_c, prof = petviashvili(1.0, 0.08, ops_flat, N, 0.2)
    soliton = {"profile": prof}
    if psi_c is not None and prof.get("residual", 1.0) < 1e-6:
        boosted = (psi_c * np.exp(1j * k1 * X)).astype(np.complex128)
        pk_etd = physics.initial_psi_k(jnp.asarray(boosted), ops_flat)
        pk_rk4 = physics.initial_psi_k(jnp.asarray(boosted), ops_flat)
        etd_peaks = [circular_peak_index(boosted)]
        rk4_peaks = [circular_peak_index(boosted)]
        steps = int(round(0.5 / dt))
        for step in range(steps):
            pk_rk4 = rk4_step(pk_rk4, ops_flat, dt)
            if (step + 1) % 100 == 0:
                pk_etd = _evolve_chunk(pk_etd, ops_flat, 100)
                etd_peaks.append(circular_peak_index(np.asarray(jnp.fft.ifftn(pk_etd))))
                rk4_peaks.append(circular_peak_index(np.asarray(jnp.fft.ifftn(pk_rk4))))
        cur_etd = np.asarray(jnp.fft.ifftn(pk_etd))
        cur_rk4 = np.asarray(jnp.fft.ifftn(pk_rk4))
        m0 = float(np.sum(np.abs(boosted) ** 2))
        soliton.update(
            {
                "etdrk4_peak_indices": etd_peaks,
                "rk4_peak_indices": rk4_peaks,
                "etdrk4_mass_ret": float(np.sum(np.abs(cur_etd) ** 2) / m0),
                "rk4_mass_ret": float(np.sum(np.abs(cur_rk4) ** 2) / m0),
            }
        )

    flux_flat_rand = norm_flux(pk_rand, ops_flat)
    flux_flat_packet = norm_flux(physics.initial_psi_k(jnp.asarray(psi_packet), ops_flat), ops_flat)

    result = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "jax_devices": [str(d) for d in jax.devices()],
        "N": N,
        "L": L,
        "dt": dt,
        "D": D,
        "k1": k1,
        "expected_v": 2 * D * k1,
        "default_parity": {
            "max_delta_L_k": max_delta(parity_base.L_k, parity_zero.L_k),
            "max_delta_E": max_delta(parity_base.E, parity_zero.E),
            "max_delta_f1": max_delta(parity_base.f1, parity_zero.f1),
            "dispersion_channel_delta_L_k": max_delta(parity_base.L_k, parity_disp.L_k),
        },
        "geom_off": {
            "geom_fac": float(np.asarray(ops_flat.geom_fac)),
            "n_op_minus_polynomial_only_max_abs": flat_n_diff,
            "norm_flux_random": flux_flat_rand,
            "norm_flux_packet": flux_flat_packet,
        },
        "old_regression": {
            "geom_fac_without_param_geom_off": float(np.asarray(ops_old.geom_fac)),
            "one_step_phase": old_phase,
            "expected_true_flat_phase": expected_phase,
            "effective_D_ratio": old_d_eff_ratio,
            "expected_rough_ratio": 1.0 / 151.0,
        },
        "linear_packet": {
            "peak_indices": packet_peak,
            "expected_cells_over_t0p5": (2 * D * k1 * 0.5) / (L / N),
        },
        "soliton_transport": soliton,
    }
    (out / "c2_6_independent_audit_summary.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    lines = [
        "# C2.6 Independent Audit",
        "",
        f"Default parity max delta L_k: `{result['default_parity']['max_delta_L_k']}`",
        f"Default parity max delta E: `{result['default_parity']['max_delta_E']}`",
        f"Geometry-off geom_fac: `{result['geom_off']['geom_fac']}`",
        f"Geometry-off n_op minus polynomial-only max abs: `{flat_n_diff}`",
        f"Old effective D ratio: `{old_d_eff_ratio}`",
        f"Linear packet peak indices: `{packet_peak}`",
        f"Soliton ETDRK4 peak indices: `{soliton.get('etdrk4_peak_indices')}`",
        f"Soliton RK4 peak indices: `{soliton.get('rk4_peak_indices')}`",
        f"Geometry-off random flux: `{flux_flat_rand}`",
        f"Geometry-off packet flux: `{flux_flat_packet}`",
    ]
    (out / "c2_6_independent_audit_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--N", type=int, default=48)
    ap.add_argument("--L", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--D", type=float, default=1.0)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
