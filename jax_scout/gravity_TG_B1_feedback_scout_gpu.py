"""TG-B1 bounded 1D KG temporal-geometric feedback scout.

This is a small GPU-only pilot.  It evolves three fields on a 1D line:

    complex phi, complex Pi
    T, V_T
    G, V_G

No separate radiation field is added.  Outgoing relaxation is measured as KG
energy flux through shells/points away from the node.  The model class is the
explicitly budgeted dissipative response selected by TG-A.

The script refuses CPU fallback.  If GPU preflight fails, it writes a
DISCREPANCY_REPORT.md and exits nonzero.
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
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

try:
    import jax

    jax.config.update("jax_enable_x64", True)

    import jax.numpy as jnp
    import jaxlib
    import numpy as np
    from jax import lax

    JAX_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised in non-JAX host shells.
    JAX_IMPORT_ERROR = exc
    import numpy as np

    class _DummyConfig:
        def update(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def read(self, _key: str) -> bool:
            return False

    class _DummyTree:
        @staticmethod
        def tree_map(_fn: Any, tree: Any) -> Any:
            return tree

    class _DummyJax:
        config = _DummyConfig()
        tree_util = _DummyTree()

        @staticmethod
        def jit(fn: Any | None = None, **_kwargs: Any) -> Any:
            if fn is None:
                return lambda real_fn: real_fn
            return fn

        @staticmethod
        def devices() -> list[Any]:
            return []

        @staticmethod
        def default_backend() -> str:
            return "unavailable"

    class _DummyJaxlib:
        __version__ = "unavailable"

    class _DummyLax:
        @staticmethod
        def fori_loop(_lower: int, _upper: int, _body_fun: Any, init_val: Any) -> Any:
            return init_val

    jax = _DummyJax()
    jaxlib = _DummyJaxlib()
    lax = _DummyLax()
    jnp = None

ROOT = Path(__file__).resolve().parents[1]


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
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, tuple, dict)) else value for key, value in row.items()})


def config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def preflight(outdir: Path) -> dict[str, Any]:
    try:
        if JAX_IMPORT_ERROR is not None:
            raise RuntimeError(f"JAX import failed in this environment: {JAX_IMPORT_ERROR}")
        devices = jax.devices()
        print("backend:", jax.default_backend(), flush=True)
        print("devices:", devices, flush=True)
        assert jax.default_backend() == "gpu", f"GPU backend required; got {jax.default_backend()} with {devices}"
        assert any(device.platform == "gpu" for device in devices), devices
        record = {
            "backend": jax.default_backend(),
            "devices": [str(d) for d in devices],
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
        report = "\n".join(
            [
                "# TG-B1 Discrepancy Report",
                "",
                "GPU preflight failed. Full field evolution was not run.",
                "",
                "```text",
                str(exc),
                "```",
            ]
        )
        (outdir / "DISCREPANCY_REPORT.md").write_text(report, encoding="utf-8")
        write_json(outdir / "gpu_preflight.json", {"status": "FAILED", "error": str(exc)})
        raise


def grid(n: int, L: float) -> dict[str, jnp.ndarray]:
    dx = L / n
    x = jnp.linspace(-0.5 * L, 0.5 * L, n, endpoint=False, dtype=jnp.float64)
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=dx)
    return {"x": x, "dx": jnp.asarray(dx, dtype=jnp.float64), "ik": (1j * k).astype(jnp.complex128)}


def deriv(f: jnp.ndarray, g: dict[str, jnp.ndarray]) -> jnp.ndarray:
    return jnp.fft.ifft(g["ik"] * jnp.fft.fft(f))


def lap(f: jnp.ndarray, g: dict[str, jnp.ndarray]) -> jnp.ndarray:
    return jnp.real(deriv(deriv(f, g), g))


def absorb_profile(g: dict[str, jnp.ndarray], L: float, width: float, strength: float) -> jnp.ndarray:
    x = g["x"]
    edge = 0.5 * L - width
    s = jnp.maximum(jnp.abs(x) - edge, 0.0) / jnp.maximum(width, 1e-12)
    return strength * s * s


def initial_phi(g: dict[str, jnp.ndarray], cfg: dict[str, Any]) -> tuple[jnp.ndarray, jnp.ndarray]:
    x = g["x"]
    sigma = float(cfg["node_sigma"])
    amp = float(cfg["node_amplitude"])
    chirp = float(cfg["phase_chirp"])
    omega = float(cfg["node_omega"])
    envelope = amp * jnp.exp(-0.5 * (x / sigma) ** 2)
    phase = chirp * jnp.tanh(x / sigma)
    if cfg.get("phase_scrambled", False):
        phase = phase + 0.8 * jnp.sin(7.0 * 2.0 * jnp.pi * x / float(cfg["L"]))
    phi = (envelope * jnp.exp(1j * phase)).astype(jnp.complex128)
    Pi = (-1j * omega * phi).astype(jnp.complex128)
    return phi, Pi


@jax.jit
def local_metrics(phi: jnp.ndarray, prev_k: jnp.ndarray, prev_p: jnp.ndarray, cfg_arr: jnp.ndarray, g: dict[str, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # cfg_arr = [dt, P_c, delta_P]
    dt, pc, delta = cfg_arr
    dx = g["dx"]
    phix = deriv(phi, g)
    rho = jnp.abs(phi) ** 2
    current = jnp.imag(jnp.conj(phi) * phix)
    k_phase_density = (current * current) / (rho + 1e-12)
    p_density = rho / (jnp.max(rho) + 1e-12)
    r_coh = jnp.maximum((prev_k - k_phase_density) / dt, 0.0)
    gate = 1.0 / (1.0 + jnp.exp(-(p_density - pc) / delta))
    r_thr = gate * jnp.maximum((p_density - prev_p) / dt, 0.0)
    # Normalize source to keep B1 bounded across arms.
    r_coh = r_coh / (jnp.max(r_coh) + 1.0)
    r_thr = r_thr / (jnp.max(r_thr) + 1.0)
    return k_phase_density, p_density, r_coh, r_thr


@jax.jit
def rhs(state: tuple[jnp.ndarray, ...], cfg_arr: jnp.ndarray, g: dict[str, jnp.ndarray], arm_flags: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
    phi, Pi, T, VT, G, VG, prev_k, prev_p = state
    (
        dt,
        mass,
        alpha_T,
        omega_T,
        omega_G,
        gamma_T,
        gamma_G,
        kappa_TG,
        epsilon_G,
        cT,
        cG,
        absorb_strength,
    ) = cfg_arr[:12]
    # arm_flags = [source_enabled, temporal_enabled, geometric_enabled, feedback_enabled, source_kind]
    source_enabled, temporal_enabled, geometric_enabled, feedback_enabled, source_kind = arm_flags
    local_cfg = jnp.asarray((cfg_arr[0], cfg_arr[12], cfg_arr[13]), dtype=cfg_arr.dtype)
    k_density, p_density, r_coh, r_thr = local_metrics(phi, prev_k, prev_p, local_cfg, g)
    R = jnp.where(source_kind < 0.5, r_coh, r_thr)
    R = R * source_enabled * temporal_enabled
    absorb = absorb_profile(g, cfg_arr[14], cfg_arr[15], absorb_strength)
    A = jnp.exp(-epsilon_G * G * feedback_enabled * geometric_enabled)
    phix = deriv(phi, g)
    div_A_grad = deriv(A * phix, g)
    phi_t = Pi
    Pi_t = div_A_grad - mass * mass * phi - absorb * Pi
    T_t = VT
    VT_t = cT * cT * lap(T, g) - omega_T * omega_T * T - gamma_T * VT + alpha_T * R - kappa_TG * G * geometric_enabled - absorb * VT
    G_t = VG * geometric_enabled
    VG_t = (cG * cG * lap(G, g) - omega_G * omega_G * G - gamma_G * VG - kappa_TG * T - absorb * VG) * geometric_enabled
    return phi_t, Pi_t, T_t, VT_t, G_t, VG_t, jnp.zeros_like(prev_k), jnp.zeros_like(prev_p)


@jax.jit
def step(state: tuple[jnp.ndarray, ...], cfg_arr: jnp.ndarray, g: dict[str, jnp.ndarray], arm_flags: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
    dt = cfg_arr[0]
    k1 = rhs(state, cfg_arr, g, arm_flags)
    s2 = tuple(y + 0.5 * dt * dy for y, dy in zip(state, k1))
    k2 = rhs(s2, cfg_arr, g, arm_flags)
    s3 = tuple(y + 0.5 * dt * dy for y, dy in zip(state, k2))
    k3 = rhs(s3, cfg_arr, g, arm_flags)
    s4 = tuple(y + dt * dy for y, dy in zip(state, k3))
    k4 = rhs(s4, cfg_arr, g, arm_flags)
    out = tuple(y + (dt / 6.0) * (a + 2 * b + 2 * c + d) for y, a, b, c, d in zip(state, k1, k2, k3, k4))
    # Update previous source metrics after the RK step.
    local_cfg = jnp.asarray((cfg_arr[0], cfg_arr[12], cfg_arr[13]), dtype=cfg_arr.dtype)
    k_density, p_density, _, _ = local_metrics(out[0], state[6], state[7], local_cfg, g)
    return out[:6] + (k_density, p_density)


@partial(jax.jit, static_argnames=("nsteps",))
def evolve_n(state: tuple[jnp.ndarray, ...], cfg_arr: jnp.ndarray, g: dict[str, jnp.ndarray], arm_flags: jnp.ndarray, nsteps: int) -> tuple[jnp.ndarray, ...]:
    def body(_, s):
        return step(s, cfg_arr, g, arm_flags)

    return lax.fori_loop(0, nsteps, body, state)


@jax.jit
def diagnostics(state: tuple[jnp.ndarray, ...], cfg_arr: jnp.ndarray, g: dict[str, jnp.ndarray], shells: jnp.ndarray) -> dict[str, jnp.ndarray]:
    phi, Pi, T, VT, G, VG, prev_k, prev_p = state
    dx = g["dx"]
    mass = cfg_arr[1]
    phix = deriv(phi, g)
    rho = jnp.abs(phi) ** 2
    energy_phi_density = 0.5 * (jnp.abs(Pi) ** 2 + jnp.abs(phix) ** 2 + mass * mass * rho)
    energy_TG_density = 0.5 * (VT * VT + cfg_arr[3] ** 2 * T * T + VG * VG + cfg_arr[4] ** 2 * G * G) + cfg_arr[7] * T * G
    core = jnp.exp(-0.5 * (g["x"] / cfg_arr[16]) ** 2)
    core_norm = jnp.sum(core) * dx
    # KG energy flux proxy for complex field: -Re(conj(Pi) phi_x).
    flux_density = -jnp.real(jnp.conj(Pi) * phix)
    shell_flux = jnp.asarray([jnp.interp(r, g["x"], flux_density) for r in shells])
    return {
        "phi_energy": jnp.sum(energy_phi_density) * dx,
        "TG_energy": jnp.sum(energy_TG_density) * dx,
        "core_energy": jnp.sum(core * energy_phi_density) * dx / (core_norm + 1e-12),
        "core_amp": jnp.max(jnp.abs(phi)),
        "T_node": jnp.sum(core * T) * dx / (core_norm + 1e-12),
        "G_node": jnp.sum(core * G) * dx / (core_norm + 1e-12),
        "integrated_R_proxy": jnp.sum(prev_k) * dx,
        "flux_left": shell_flux[0],
        "flux_right": shell_flux[-1],
        "outgoing_flux_abs": jnp.sum(jnp.abs(shell_flux)),
        "K_phase": jnp.sum(prev_k) * dx,
        "P_proxy": jnp.sum(prev_p) * dx,
    }


def arm_flags(name: str, source_kind: str) -> list[float]:
    source_enabled = 0.0 if name == "source_off" else 1.0
    temporal_enabled = 0.0 if name == "temporal_off" else 1.0
    geometric_enabled = 0.0 if name in ("baseline", "temporal_only", "geometric_off") else 1.0
    feedback_enabled = 1.0 if name == "full_loop" else 0.0
    if name in ("baseline", "feed_forward", "feedback_off"):
        feedback_enabled = 0.0
    kind = 0.0 if source_kind == "coherence" else 1.0
    return [source_enabled, temporal_enabled, geometric_enabled, feedback_enabled, kind]


def run_arm(cfg: dict[str, Any], outdir: Path, name: str, source_kind: str, scrambled: bool = False) -> dict[str, Any]:
    g = grid(int(cfg["N"]), float(cfg["L"]))
    cfg_local = {**cfg, "phase_scrambled": scrambled}
    phi, Pi = initial_phi(g, cfg_local)
    k0, p0, _, _ = local_metrics(phi, jnp.zeros_like(jnp.real(phi)), jnp.zeros_like(jnp.real(phi)), jnp.asarray([cfg["dt"], cfg["P_c"], cfg["delta_P"]], dtype=jnp.float64), g)
    zeros = jnp.zeros_like(jnp.real(phi))
    state = (phi, Pi, zeros, zeros, zeros, zeros, k0, p0)
    cfg_arr = jnp.asarray(
        [
            cfg["dt"],
            cfg["mass"],
            cfg["alpha_T"],
            cfg["omega_T"],
            cfg["omega_G"],
            cfg["gamma_T"],
            cfg["gamma_G"],
            cfg["kappa_TG"],
            cfg["epsilon_G"],
            cfg["cT"],
            cfg["cG"],
            cfg["absorb_strength"],
            cfg["P_c"],
            cfg["delta_P"],
            cfg["L"],
            cfg["absorb_width"],
            cfg["node_sigma"],
        ],
        dtype=jnp.float64,
    )
    flags = jnp.asarray(arm_flags(name, source_kind), dtype=jnp.float64)
    shells = jnp.asarray([-0.35 * cfg["L"], 0.35 * cfg["L"]], dtype=jnp.float64)
    steps = int(round(cfg["T"] / cfg["dt"]))
    every = max(1, int(round(cfg["sample_dt"] / cfg["dt"])))
    chunks = steps // every
    samples: list[dict[str, float]] = []
    compile_start = time.time()
    d0 = diagnostics(state, cfg_arr, g, shells)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), d0)
    compile_time = time.time() - compile_start
    exec_start = time.time()
    samples.append({"t": 0.0, **{k: float(np.asarray(v)) for k, v in d0.items()}})
    for idx in range(chunks):
        state = evolve_n(state, cfg_arr, g, flags, every)
        d = diagnostics(state, cfg_arr, g, shells)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), d)
        samples.append({"t": (idx + 1) * every * cfg["dt"], **{k: float(np.asarray(v)) for k, v in d.items()}})
    exec_time = time.time() - exec_start
    write_csv(outdir / f"{name}_{source_kind}{'_scrambled' if scrambled else ''}_trajectory.csv", samples)
    t = np.asarray([s["t"] for s in samples])
    Tn = np.asarray([s["T_node"] for s in samples])
    Gn = np.asarray([s["G_node"] for s in samples])
    flux = np.asarray([s["outgoing_flux_abs"] for s in samples])
    phi_e = np.asarray([s["phi_energy"] for s in samples])
    summary = {
        "run_id": f"{name}_{source_kind}{'_scrambled' if scrambled else ''}",
        "arm": name,
        "source_kind": source_kind,
        "phase_scrambled": scrambled,
        "max_abs_T_node": float(np.max(np.abs(Tn))),
        "max_abs_G_node": float(np.max(np.abs(Gn))),
        "integrated_abs_flux": float(np.trapezoid(flux, t)),
        "final_core_energy": samples[-1]["core_energy"],
        "core_energy_delta": samples[-1]["core_energy"] - samples[0]["core_energy"],
        "phi_energy_rel_drift": float(abs(phi_e[-1] - phi_e[0]) / max(abs(phi_e[0]), 1e-12)),
        "compile_time_s": compile_time,
        "execution_time_s": exec_time,
    }
    return summary


def make_matrix(args: argparse.Namespace) -> list[dict[str, Any]]:
    arms = ["baseline", "temporal_only", "feed_forward", "full_loop", "feedback_off", "temporal_off", "geometric_off", "source_off"]
    matrix = []
    for arm in arms:
        matrix.append({"arm": arm, "source_kind": "coherence", "phase_scrambled": False})
    matrix.append({"arm": "feed_forward", "source_kind": "threshold", "phase_scrambled": False})
    matrix.append({"arm": "feed_forward", "source_kind": "coherence", "phase_scrambled": True})
    return matrix


def write_docs(outdir: Path, status: str, rows: list[dict[str, Any]], reason: str = "") -> None:
    docdir = ROOT / "docs" / "gravity_maturity"
    docdir.mkdir(parents=True, exist_ok=True)
    (docdir / "TG_B1_RESULTS.md").write_text(
        "\n".join(
            [
                "# TG-B1 Reduced Spatial KG Feedback Scout Results",
                "",
                "Timestamp: 2026-07-14.",
                f"Run directory: `{outdir.as_posix()}`.",
                f"Status: `{status}`.",
                "",
                reason,
                "",
                "## Rows",
                "",
                "| run | max T | max G | integrated abs flux | core energy delta |",
                "| --- | ---: | ---: | ---: | ---: |",
                *[
                    f"| {r.get('run_id')} | {r.get('max_abs_T_node', float('nan')):.6e} | {r.get('max_abs_G_node', float('nan')):.6e} | {r.get('integrated_abs_flux', float('nan')):.6e} | {r.get('core_energy_delta', float('nan')):.6e} |"
                    for r in rows
                ],
            ]
        ),
        encoding="utf-8",
    )
    write_json(docdir / "TG_B1_SUMMARY.json", {"status": status, "run_directory": str(outdir), "reason": reason, "rows": rows})
    (docdir / "TG_B1_DOCUMENTATION_INPUTS.md").write_text(
        f"# TG-B1 Documentation Inputs\n\n- Status: `{status}`.\n- Artifacts: `{outdir.as_posix()}`.\n- Reason: {reason}\n",
        encoding="utf-8",
    )


def artifact_hashes(outdir: Path) -> list[dict[str, str]]:
    rows = []
    for path in sorted(outdir.rglob("*")):
        if path.is_file() and path.name != "artifact_hashes.csv":
            rows.append({"path": str(path.relative_to(outdir)), "sha256": sha256_file(path)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--L", type=float, default=80.0)
    ap.add_argument("--T", type=float, default=8.0)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--sample-dt", type=float, default=0.05)
    ap.add_argument("--mass", type=float, default=1.0)
    ap.add_argument("--node-omega", type=float, default=0.94)
    ap.add_argument("--node-sigma", type=float, default=3.0)
    ap.add_argument("--node-amplitude", type=float, default=0.8)
    ap.add_argument("--phase-chirp", type=float, default=0.8)
    ap.add_argument("--alpha-T", type=float, default=0.35)
    ap.add_argument("--omega-T", type=float, default=1.25)
    ap.add_argument("--omega-G", type=float, default=0.85)
    ap.add_argument("--gamma-T", type=float, default=0.08)
    ap.add_argument("--gamma-G", type=float, default=0.06)
    ap.add_argument("--kappa-TG", type=float, default=0.55)
    ap.add_argument("--epsilon-G", type=float, default=0.06)
    ap.add_argument("--cT", type=float, default=0.7)
    ap.add_argument("--cG", type=float, default=0.55)
    ap.add_argument("--P-c", type=float, default=0.5)
    ap.add_argument("--delta-P", type=float, default=0.08)
    ap.add_argument("--absorb-width", type=float, default=10.0)
    ap.add_argument("--absorb-strength", type=float, default=0.35)
    args = ap.parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.out) if args.out else ROOT / "sweep_runs" / f"TG_B1_FEEDBACK_GPU_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "git_state_before.txt").write_text(git_state(), encoding="utf-8")
    cfg = vars(args).copy()
    cfg["config_hash"] = config_hash(cfg)
    write_json(
        outdir / "environment_versions.json",
        {
            "stage": "TG-B1",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "jax_import_error": str(JAX_IMPORT_ERROR) if JAX_IMPORT_ERROR is not None else None,
            "git_commit": git_commit(),
            "command_line": command_line(),
        },
    )
    write_json(outdir / "model_spec.json", cfg)
    write_json(outdir / "source_spec.json", {"primary": "R_coh", "secondary": "R_thr"})
    matrix = make_matrix(args)
    write_json(outdir / "preregistered_matrix.json", matrix)
    try:
        pf = preflight(outdir)
    except Exception as exc:
        reason = f"GPU preflight failed: {exc}"
        write_docs(outdir, "TG_B1_GPU_PREFLIGHT_FAILED", [], reason)
        write_csv(outdir / "artifact_hashes.csv", artifact_hashes(outdir))
        raise SystemExit(2)
    write_json(outdir / "environment_versions.json", {**pf, "operator": "1D complex KG + T/G response fields"})
    rows = []
    for entry in matrix:
        rows.append(run_arm(cfg, outdir, entry["arm"], entry["source_kind"], bool(entry["phase_scrambled"])))
    write_csv(outdir / "run_manifest.csv", rows)
    write_csv(outdir / "energy_ledger.csv", rows)
    write_csv(outdir / "frequency_metrics.csv", rows)
    write_csv(outdir / "causal_lag_metrics.csv", rows)
    write_csv(outdir / "attractor_classification.csv", rows)
    # Minimal falsification checks for the pilot.
    by = {r["run_id"]: r for r in rows}
    checks = [
        {
            "test": "source_off_removes_T",
            "pass": by.get("source_off_coherence", {}).get("max_abs_T_node", 1.0) < 1e-7,
        },
        {
            "test": "temporal_off_removes_T_and_G",
            "pass": by.get("temporal_off_coherence", {}).get("max_abs_T_node", 1.0) < 1e-7
            and by.get("temporal_off_coherence", {}).get("max_abs_G_node", 1.0) < 1e-7,
        },
        {
            "test": "feedback_changes_core_energy",
            "pass": abs(by.get("full_loop_coherence", {}).get("core_energy_delta", 0.0) - by.get("feedback_off_coherence", {}).get("core_energy_delta", 0.0)) > 1e-6,
        },
    ]
    write_csv(outdir / "falsification_results.csv", checks)
    status = "TG_B1_PILOT_COMPLETED" if all(c["pass"] for c in checks[:2]) else "TG_FEEDBACK_SEQUENCE_NOT_SUPPORTED"
    write_docs(outdir, status, rows)
    write_csv(outdir / "artifact_hashes.csv", artifact_hashes(outdir))
    print(json.dumps({"status": status, "outdir": str(outdir), "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
