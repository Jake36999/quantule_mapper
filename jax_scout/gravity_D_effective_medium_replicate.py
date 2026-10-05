"""Independent replication for the Audit-D spatial effective-medium result.

This is deliberately separate from gravity_D_neutral_probe.py.  It keeps the
same mirror setup (Gaussian environment source, neutral Gaussian probe, and
H = -D div(N grad)) but adds a force-contract diagnostic for the exact
implemented divergence-form operator.

The reversed-gradient control uses N = 1 + beta * Shat, a positive coefficient
hill.  This avoids the beta < 0 reciprocal denominator control, whose
denominator can cross zero if pushed too far.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
D = 0.3


@dataclass(frozen=True)
class CaseConfig:
    label: str
    n_grid: int = 64
    box_L: float = 30.0
    dt: float = 0.002
    duration_T: float = 3.0
    beta: float = 1.0
    coefficient_family: str = "well"
    source_sigma: float = 1.5
    probe_sigma: float = 1.0
    probe_x: float = 4.0
    probe_amplitude: float = 1.0
    nsnap: int = 12


def stable_json(data: object) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=float)


def hash_config(cfg: CaseConfig) -> str:
    return hashlib.sha256(stable_json(asdict(cfg)).encode("utf-8")).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "UNKNOWN"


def git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "UNKNOWN"


def build_grid(n: int, L: float) -> dict[str, np.ndarray | float]:
    x = np.linspace(-L / 2.0, L / 2.0, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=L / n)
    return {
        "x": x,
        "X": X,
        "Y": Y,
        "Z": Z,
        "ikx": 1j * k[:, None, None],
        "iky": 1j * k[None, :, None],
        "ikz": 1j * k[None, None, :],
        "dV": (L / n) ** 3,
    }


def integ(field: np.ndarray, dV: float) -> float:
    return float(np.sum(np.real(field)) * dV)


def deriv(field: np.ndarray, ik: np.ndarray) -> np.ndarray:
    return np.fft.ifftn(ik * np.fft.fftn(field))


def source_shat(grid: dict[str, np.ndarray | float], sigma: float) -> tuple[np.ndarray, np.ndarray]:
    X = grid["X"]
    Y = grid["Y"]
    Z = grid["Z"]
    rho = np.exp(-((X * X + Y * Y + Z * Z) / (2.0 * sigma * sigma)))
    S = rho * rho
    return rho, S / (float(np.max(S)) + 1e-30)


def coefficient(shat: np.ndarray, beta: float, family: str) -> np.ndarray:
    if family == "well":
        return 1.0 / (1.0 + beta * shat)
    if family == "flat":
        return np.ones_like(shat)
    if family == "hill":
        return 1.0 + beta * shat
    raise ValueError(f"unknown coefficient family: {family}")


def probe(grid: dict[str, np.ndarray | float], cfg: CaseConfig) -> np.ndarray:
    X = grid["X"]
    Y = grid["Y"]
    Z = grid["Z"]
    sig2 = 2.0 * cfg.probe_sigma * cfg.probe_sigma
    return (
        cfg.probe_amplitude
        * np.exp(-(((X - cfg.probe_x) ** 2 + Y * Y + Z * Z) / sig2))
    ).astype(np.complex128)


def lap_cov(psi: np.ndarray, Nf: np.ndarray, grid: dict[str, np.ndarray | float]) -> np.ndarray:
    gx = deriv(psi, grid["ikx"])
    gy = deriv(psi, grid["iky"])
    gz = deriv(psi, grid["ikz"])
    return (
        deriv(Nf * gx, grid["ikx"])
        + deriv(Nf * gy, grid["iky"])
        + deriv(Nf * gz, grid["ikz"])
    )


def rhs(psi: np.ndarray, Nf: np.ndarray, grid: dict[str, np.ndarray | float]) -> np.ndarray:
    return 1j * D * lap_cov(psi, Nf, grid)


def rk4_step(psi: np.ndarray, Nf: np.ndarray, grid: dict[str, np.ndarray | float], dt: float) -> np.ndarray:
    k1 = rhs(psi, Nf, grid)
    k2 = rhs(psi + 0.5 * dt * k1, Nf, grid)
    k3 = rhs(psi + 0.5 * dt * k2, Nf, grid)
    k4 = rhs(psi + dt * k3, Nf, grid)
    return psi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def gradients(psi: np.ndarray, grid: dict[str, np.ndarray | float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return deriv(psi, grid["ikx"]), deriv(psi, grid["iky"]), deriv(psi, grid["ikz"])


def momentum_x(psi: np.ndarray, grid: dict[str, np.ndarray | float]) -> float:
    gx = deriv(psi, grid["ikx"])
    return integ(np.imag(np.conjugate(psi) * gx), grid["dV"])


def norm(psi: np.ndarray, grid: dict[str, np.ndarray | float]) -> float:
    return integ(np.abs(psi) ** 2, grid["dV"])


def com_x(psi: np.ndarray, grid: dict[str, np.ndarray | float], probe_x: float, window_width: float = 6.0) -> float:
    X = grid["X"]
    rho = np.abs(psi) ** 2
    window = np.abs(X - probe_x) < window_width
    wrho = rho * window
    mass = integ(wrho, grid["dV"])
    return integ(X * wrho, grid["dV"]) / (mass + 1e-30)


def force_analytic(psi: np.ndarray, Nf: np.ndarray, grid: dict[str, np.ndarray | float]) -> float:
    gx, gy, gz = gradients(psi, grid)
    grad_energy = np.abs(gx) ** 2 + np.abs(gy) ** 2 + np.abs(gz) ** 2
    dNx = np.real(deriv(Nf, grid["ikx"]))
    return -D * integ(dNx * grad_energy, grid["dV"])


def force_rhs(psi: np.ndarray, Nf: np.ndarray, grid: dict[str, np.ndarray | float]) -> float:
    psit = rhs(psi, Nf, grid)
    gx = deriv(psi, grid["ikx"])
    gt_x = deriv(psit, grid["ikx"])
    density_t = np.imag(np.conjugate(psit) * gx + np.conjugate(psi) * gt_x)
    return integ(density_t, grid["dV"])


def force_fd(psi: np.ndarray, Nf: np.ndarray, grid: dict[str, np.ndarray | float], dt: float) -> float:
    p0 = momentum_x(psi, grid)
    p1 = momentum_x(rk4_step(psi, Nf, grid, dt), grid)
    return (p1 - p0) / dt


def diagnostics(
    psi: np.ndarray,
    Nf: np.ndarray,
    grid: dict[str, np.ndarray | float],
    cfg: CaseConfig,
    t: float,
) -> dict[str, float]:
    fa = force_analytic(psi, Nf, grid)
    fr = force_rhs(psi, Nf, grid)
    rel = abs(fr - fa) / (abs(fa) + 1e-30) if abs(fa) > 1e-20 else abs(fr - fa)
    return {
        "t": float(t),
        "com_x": float(com_x(psi, grid, cfg.probe_x)),
        "momentum_x": float(momentum_x(psi, grid)),
        "norm": float(norm(psi, grid)),
        "force_analytic": float(fa),
        "force_rhs": float(fr),
        "force_residual_abs": float(abs(fr - fa)),
        "force_residual_rel": float(rel),
    }


def run_case(cfg: CaseConfig) -> dict[str, object]:
    started = time.time()
    grid = build_grid(cfg.n_grid, cfg.box_L)
    rho_B, shat = source_shat(grid, cfg.source_sigma)
    Nf = coefficient(shat, cfg.beta, cfg.coefficient_family)
    psi0 = probe(grid, cfg)
    psi = psi0.copy()
    n0 = norm(psi0, grid)
    nsteps = int(round(cfg.duration_T / cfg.dt))
    every = max(1, nsteps // cfg.nsnap)
    samples = [diagnostics(psi, Nf, grid, cfg, 0.0)]
    short_fd = force_fd(psi0, Nf, grid, cfg.dt)

    for i in range(nsteps):
        psi = rk4_step(psi, Nf, grid, cfg.dt)
        if (i + 1) % every == 0 or i == nsteps - 1:
            samples.append(diagnostics(psi, Nf, grid, cfg, (i + 1) * cfg.dt))

    nT = norm(psi, grid)
    result = {
        "config": asdict(cfg),
        "config_hash": hash_config(cfg),
        "git_commit": git_commit(),
        "git_branch": git_branch(),
        "operator": "i d_t psi = i D div(N grad psi), H = -D div(N grad)",
        "D": D,
        "cell_volume_dV": float(grid["dV"]),
        "source_normalization": {
            "source": "rho_B=exp(-r^2/(2*sigma_B^2)); Shat=rho_B^2/max(rho_B^2)",
            "rho_B_max": float(np.max(rho_B)),
            "Shat_max": float(np.max(shat)),
            "coefficient_min": float(np.min(Nf)),
            "coefficient_max": float(np.max(Nf)),
            "coefficient_family_note": coefficient_note(cfg.coefficient_family),
        },
        "initial": {
            "norm": float(n0),
            "momentum_x": float(samples[0]["momentum_x"]),
            "force_analytic": float(samples[0]["force_analytic"]),
            "force_rhs": float(samples[0]["force_rhs"]),
            "force_fd_one_step": float(short_fd),
            "rhs_vs_analytic_rel": float(samples[0]["force_residual_rel"]),
            "fd_vs_analytic_rel": float(relative_or_abs(short_fd, samples[0]["force_analytic"])),
        },
        "final": {
            "norm": float(nT),
            "norm_ratio": float(nT / (n0 + 1e-30)),
            "norm_error_abs": float(abs(nT - n0)),
            "com_drift_x": float(samples[-1]["com_x"] - samples[0]["com_x"]),
            "momentum_change_x": float(samples[-1]["momentum_x"] - samples[0]["momentum_x"]),
        },
        "samples": samples,
        "wall_time_s": float(time.time() - started),
    }
    return result


def coefficient_note(family: str) -> str:
    if family == "well":
        return "reciprocal well: N=1/(1+beta*Shat), minimum at load, positive for beta>=0"
    if family == "flat":
        return "flat null: N=1"
    if family == "hill":
        return "reversed positive hill: N=1+beta*Shat, maximum at load, no denominator crossing"
    return ""


def relative_or_abs(measured: float, expected: float) -> float:
    if abs(expected) > 1e-20:
        return abs(measured - expected) / abs(expected)
    return abs(measured - expected)


def fit_initial_acceleration(times: list[float], net_com: list[float], nfit: int = 5) -> float:
    if len(times) < 3:
        return float("nan")
    m = min(nfit, len(times))
    coef = np.polyfit(np.asarray(times[:m]), np.asarray(net_com[:m]), deg=2)
    return float(2.0 * coef[0])


def summarize_cases(results: list[dict[str, object]]) -> list[dict[str, object]]:
    by_label = {r["config"]["label"]: r for r in results}
    flat = by_label.get("flat_null")
    rows = []
    for r in results:
        cfg = r["config"]
        samples = r["samples"]
        row = {
            "label": cfg["label"],
            "n_grid": cfg["n_grid"],
            "box_L": cfg["box_L"],
            "dt": cfg["dt"],
            "duration_T": cfg["duration_T"],
            "family": cfg["coefficient_family"],
            "probe_sigma": cfg["probe_sigma"],
            "com_drift_x": r["final"]["com_drift_x"],
            "momentum_change_x": r["final"]["momentum_change_x"],
            "initial_force_analytic": r["initial"]["force_analytic"],
            "initial_force_rhs": r["initial"]["force_rhs"],
            "initial_force_fd": r["initial"]["force_fd_one_step"],
            "rhs_vs_analytic_rel": r["initial"]["rhs_vs_analytic_rel"],
            "fd_vs_analytic_rel": r["initial"]["fd_vs_analytic_rel"],
            "norm_ratio": r["final"]["norm_ratio"],
            "config_hash": r["config_hash"],
            "wall_time_s": r["wall_time_s"],
        }
        if flat is not None and len(flat["samples"]) == len(samples):
            net = [
                float(samples[i]["com_x"] - flat["samples"][i]["com_x"])
                for i in range(len(samples))
            ]
            row["baseline_subtracted_final_com"] = net[-1]
            row["baseline_subtracted_initial_accel_fit"] = fit_initial_acceleration(
                [float(s["t"]) for s in samples], net
            )
        else:
            row["baseline_subtracted_final_com"] = None
            row["baseline_subtracted_initial_accel_fit"] = None
        rows.append(row)
    return rows


def default_cases(quick: bool, baseline_T: float | None) -> list[CaseConfig]:
    T = baseline_T if baseline_T is not None else (1.0 if quick else 3.0)
    nsnap = 8 if quick else 12
    return [
        CaseConfig("main_well", duration_T=T, nsnap=nsnap),
        CaseConfig("flat_null", duration_T=T, beta=0.0, coefficient_family="flat", nsnap=nsnap),
        CaseConfig("reversed_hill", duration_T=T, coefficient_family="hill", nsnap=nsnap),
        CaseConfig("wide_probe", duration_T=T, probe_sigma=1.6, nsnap=nsnap),
        CaseConfig("larger_box", box_L=40.0, duration_T=T, nsnap=nsnap),
    ]


def convergence_cases(convergence_T: float) -> list[CaseConfig]:
    cases: list[CaseConfig] = []
    for L in (30.0, 40.0):
        for n, dt in ((64, 0.002), (96, 0.001), (128, 0.0005)):
            cases.append(
                CaseConfig(
                    f"conv_N{n}_L{int(L)}",
                    n_grid=n,
                    box_L=L,
                    dt=dt,
                    duration_T=convergence_T,
                    nsnap=2,
                )
            )
    return cases


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary_rows: list[dict[str, object]], verdict: str) -> None:
    lines = [
        "# Spatial Effective-Medium Replication",
        "",
        f"Verdict: `{verdict}`",
        "",
        "Operator: `H = -D div(N grad)`, with momentum contract",
        "`d<P_x>/dt = -D int (d_x N) |grad psi|^2 dV` for the implemented periodic spectral operator.",
        "",
        "| label | drift_x | net_drift_x | F_analytic | F_rhs | F_fd | rhs_rel | norm_ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {label} | {com_drift_x:+.6e} | {net:+.6e} | {fa:+.6e} | {fr:+.6e} | {ff:+.6e} | {rel:.3e} | {nr:.9f} |".format(
                label=row["label"],
                com_drift_x=float(row["com_drift_x"]),
                net=float(row["baseline_subtracted_final_com"] or 0.0),
                fa=float(row["initial_force_analytic"]),
                fr=float(row["initial_force_rhs"]),
                ff=float(row["initial_force_fd"]),
                rel=float(row["rhs_vs_analytic_rel"]),
                nr=float(row["norm_ratio"]),
            )
        )
    lines.extend(
        [
            "",
            "The reversed control is a positive coefficient hill, `N=1+beta*Shat`, not a negative-beta reciprocal.",
            "This documents the sign reversal without introducing a denominator that can cross zero.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def decide_verdict(rows: list[dict[str, object]]) -> str:
    by_label = {r["label"]: r for r in rows}
    main = by_label.get("main_well")
    flat = by_label.get("flat_null")
    rev = by_label.get("reversed_hill")
    wide = by_label.get("wide_probe")
    large = by_label.get("larger_box")
    if not all([main, flat, rev, wide, large]):
        return "D_EFFECTIVE_MEDIUM_ATTRACTION_PARTIAL"
    main_net = float(main["baseline_subtracted_final_com"])
    rev_net = float(rev["baseline_subtracted_final_com"])
    wide_net = float(wide["baseline_subtracted_final_com"])
    large_net = float(large["com_drift_x"])
    flat_drift = abs(float(flat["com_drift_x"]))
    force_ok = abs(float(main["rhs_vs_analytic_rel"])) < 1e-6
    fd_ok = abs(float(main["fd_vs_analytic_rel"])) < 5e-3
    sign_ok = main_net < 0.0 and rev_net > 0.0 and wide_net < 0.0 and large_net < 0.0
    null_ok = flat_drift < max(2e-3, 0.25 * abs(main_net))
    norm_ok = all(abs(float(r["norm_ratio"]) - 1.0) < 1e-5 for r in rows)
    if sign_ok and null_ok and force_ok and fd_ok and norm_ok:
        return "D_EFFECTIVE_MEDIUM_ATTRACTION_REPLICATED"
    if sign_ok and null_ok and force_ok:
        return "D_EFFECTIVE_MEDIUM_ATTRACTION_PARTIAL_NUMERICS"
    return "D_EFFECTIVE_MEDIUM_ATTRACTION_UNCLEAR"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true", help="Shorten baseline trajectories for smoke testing.")
    ap.add_argument("--baseline-T", type=float, default=None, help="Override baseline/control trajectory duration.")
    ap.add_argument(
        "--convergence-T",
        type=float,
        default=0.0,
        help="Duration for refinement cases. The default 0 records force-contract rows only.",
    )
    ap.add_argument("--skip-convergence", action="store_true")
    args = ap.parse_args()

    out = Path(args.out) if args.out else ROOT / "sweep_runs" / f"GRAVITY_D_EFFECTIVE_MEDIUM_CODEX_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)

    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),
        "git_branch": git_branch(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "D": D,
    }

    case_list = default_cases(args.quick, args.baseline_T)
    if not args.skip_convergence:
        case_list.extend(convergence_cases(args.convergence_T))

    results = []
    for cfg in case_list:
        print(
            f"[run] {cfg.label}: N={cfg.n_grid} L={cfg.box_L} dt={cfg.dt} T={cfg.duration_T} "
            f"family={cfg.coefficient_family} sigma={cfg.probe_sigma}",
            flush=True,
        )
        result = run_case(cfg)
        results.append(result)
        final = result["final"]
        init = result["initial"]
        print(
            f"      drift={final['com_drift_x']:+.6e} Pchange={final['momentum_change_x']:+.6e} "
            f"F={init['force_analytic']:+.6e} rhs_rel={init['rhs_vs_analytic_rel']:.3e} "
            f"fd_rel={init['fd_vs_analytic_rel']:.3e} norm={final['norm_ratio']:.9f}",
            flush=True,
        )

    summary_rows = summarize_cases(results)
    verdict = decide_verdict(summary_rows)
    payload = {"metadata": metadata, "verdict": verdict, "summary": summary_rows, "results": results}
    (out / "results.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    write_csv(out / "summary.csv", summary_rows)
    write_report(out / "replication_report.md", summary_rows, verdict)
    print(f"=== {verdict} | out={out} ===", flush=True)


if __name__ == "__main__":
    main()
