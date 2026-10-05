"""TG-B0 reduced temporal-geometric feedback exchange test.

This reduced model is deliberately cheaper than the spatial KG scout.  It asks
whether the selected TG-A response class has bounded T/G exchange and whether
the dependency chain R_res -> T -> G survives intervention controls.

The reduced node variable a(t) is a phase-mismatch/coherence-cost proxy:
    a = K_phase / K_phase(0)

It relaxes toward an ordered floor and can optionally be weakly modified by G
in the closed-loop rows.  Resolution sources are the preregistered TG-A
definitions:
    R_coh = [-d_t a]_+
    R_thr = sigmoid((P-P_c)/delta_P) [d_t P]_+, P = 1-a
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

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


def source_value(a: float, adot_raw: float, cfg: dict[str, Any]) -> float:
    if cfg["source_family"] == "coherence":
        return max(-adot_raw, 0.0)
    if cfg["source_family"] == "threshold":
        p = 1.0 - a
        pdot = -adot_raw
        gate = 1.0 / (1.0 + math.exp(-(p - cfg["P_c"]) / cfg["delta_P"]))
        return gate * max(pdot, 0.0)
    if cfg["source_family"] == "off":
        return 0.0
    raise ValueError(cfg["source_family"])


def rhs(state: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
    a, T, VT, G, VG = state
    lam = cfg["lambda_a"]
    a_floor = cfg["a_floor"]
    fb = cfg["node_feedback_gain"] if cfg["full_loop"] else 0.0
    temporal_enabled = cfg["temporal_enabled"]
    geometric_enabled = cfg["geometric_enabled"]
    source_enabled = cfg["source_enabled"]
    # a is a phase-mismatch cost; positive G may raise or lower the cost depending
    # on feedback sign.  The default sign tests whether G can impede decay.
    adot_raw = -lam * (a - a_floor) + fb * G
    R = source_value(a, adot_raw, cfg) if source_enabled else 0.0
    if not temporal_enabled:
        R = 0.0
    alpha = cfg["alpha_T"]
    omega_t = cfg["omega_T"]
    omega_g = cfg["omega_G"]
    gamma_t = cfg["gamma_T"]
    gamma_g = cfg["gamma_G"]
    kappa = cfg["kappa_TG"] if geometric_enabled else 0.0
    dT = VT
    dVT = alpha * R - gamma_t * VT - omega_t * omega_t * T - kappa * G
    dG = VG if geometric_enabled else 0.0
    dVG = (-gamma_g * VG - omega_g * omega_g * G - kappa * T) if geometric_enabled else 0.0
    deriv = np.array([adot_raw, dT, dVT, dG, dVG], dtype=float)
    aux = {
        "R": R,
        "source_power": alpha * R * VT,
        "diss_T": gamma_t * VT * VT,
        "diss_G": gamma_g * VG * VG if geometric_enabled else 0.0,
        "energy_TG": energy_TG(state, cfg),
    }
    return deriv, aux


def energy_TG(state: np.ndarray, cfg: dict[str, Any]) -> float:
    _, T, VT, G, VG = state
    kappa = cfg["kappa_TG"] if cfg["geometric_enabled"] else 0.0
    return float(
        0.5 * VT * VT
        + 0.5 * cfg["omega_T"] ** 2 * T * T
        + 0.5 * VG * VG
        + 0.5 * cfg["omega_G"] ** 2 * G * G
        + kappa * T * G
    )


def rk4_step(state: np.ndarray, dt: float, cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
    k1, _ = rhs(state, cfg)
    k2, _ = rhs(state + 0.5 * dt * k1, cfg)
    k3, _ = rhs(state + 0.5 * dt * k2, cfg)
    k4, aux = rhs(state + dt * k3, cfg)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), aux


def dominant_frequency(t: np.ndarray, y: np.ndarray) -> float:
    if len(t) < 8 or np.max(np.abs(y - np.mean(y))) < 1e-12:
        return float("nan")
    dt = float(np.median(np.diff(t)))
    window = np.hanning(len(y))
    spec = np.fft.rfft((y - np.mean(y)) * window)
    freq = np.fft.rfftfreq(len(y), d=dt)
    if len(freq) < 2:
        return float("nan")
    idx = int(np.argmax(np.abs(spec[1:]) ** 2) + 1)
    return float(freq[idx])


def peak_time(t: np.ndarray, y: np.ndarray, threshold_frac: float = 0.2) -> float:
    y = np.asarray(y)
    if y.size == 0 or np.max(np.abs(y)) < 1e-12:
        return float("nan")
    th = threshold_frac * float(np.max(np.abs(y)))
    idx = np.where(np.abs(y) >= th)[0]
    return float(t[int(idx[0])]) if idx.size else float("nan")


def classify_frequency(freq: float, cfg: dict[str, Any]) -> str:
    if not np.isfinite(freq) or freq <= 0:
        return "UNRESOLVED"
    fT = cfg["omega_T"] / (2 * math.pi)
    fG = cfg["omega_G"] / (2 * math.pi)
    beat = abs(cfg["omega_T"] - cfg["omega_G"]) / (2 * math.pi)
    if abs(freq - fT) / max(fT, 1e-12) < 0.12:
        return "INSERTED_FIELD_MODE_T"
    if abs(freq - fG) / max(fG, 1e-12) < 0.12:
        return "INSERTED_FIELD_MODE_G"
    if beat > 0 and abs(freq - beat) / max(beat, 1e-12) < 0.15:
        return "BEAT_OR_COMBINATION_MODE"
    return "DRIVEN_OR_NONLINEAR_MODE"


def classify_attractor(t: np.ndarray, a: np.ndarray, T: np.ndarray, G: np.ndarray, cfg: dict[str, Any]) -> str:
    if not np.all(np.isfinite(a)) or np.max(np.abs([T, G])) > cfg["runaway_threshold"]:
        return "RUNAWAY"
    tail = slice(int(0.75 * len(t)), None)
    amp = max(float(np.ptp(T[tail])), float(np.ptp(G[tail])))
    drift = max(abs(float(T[-1] - T[int(0.75 * len(t))])), abs(float(G[-1] - G[int(0.75 * len(t))])))
    if amp < 1e-4:
        return "STABLE_FIXED_POINT"
    if drift < 0.25 * amp:
        return "BOUNDED_LIMIT_CYCLE"
    return "DAMPED_OSCILLATION"


def run_case(cfg: dict[str, Any], outdir: Path) -> dict[str, Any]:
    dt = float(cfg["dt"])
    steps = int(round(cfg["T"] / dt))
    every = max(1, int(round(cfg["sample_dt"] / dt)))
    state = np.array([cfg["a0"], 0.0, 0.0, 0.0, 0.0], dtype=float)
    rows: list[dict[str, float]] = []
    ledger_residual = 0.0
    prev_e = energy_TG(state, cfg)
    for step in range(steps + 1):
        if step % every == 0:
            _, aux = rhs(state, cfg)
            rows.append(
                {
                    "t": step * dt,
                    "a": state[0],
                    "T": state[1],
                    "VT": state[2],
                    "G": state[3],
                    "VG": state[4],
                    **aux,
                }
            )
        if step == steps:
            break
        deriv, aux0 = rhs(state, cfg)
        state_next, aux = rk4_step(state, dt, cfg)
        e_next = energy_TG(state_next, cfg)
        # Ledger residual for dE = source_power - dissipation over dt.
        expected_de = dt * (aux0["source_power"] - aux0["diss_T"] - aux0["diss_G"])
        ledger_residual += abs((e_next - prev_e) - expected_de)
        state = state_next
        prev_e = e_next
    t = np.asarray([r["t"] for r in rows])
    a = np.asarray([r["a"] for r in rows])
    T = np.asarray([r["T"] for r in rows])
    G = np.asarray([r["G"] for r in rows])
    R = np.asarray([r["R"] for r in rows])
    freq_T = dominant_frequency(t, T)
    freq_G = dominant_frequency(t, G)
    lag_rt = peak_time(t, T) - peak_time(t, R)
    lag_tg = peak_time(t, G) - peak_time(t, T)
    summary = {
        "run_id": cfg["run_id"],
        "config_hash": config_hash(cfg),
        "source_family": cfg["source_family"],
        "temporal_enabled": cfg["temporal_enabled"],
        "geometric_enabled": cfg["geometric_enabled"],
        "source_enabled": cfg["source_enabled"],
        "full_loop": cfg["full_loop"],
        "kappa_TG": cfg["kappa_TG"],
        "node_feedback_gain": cfg["node_feedback_gain"],
        "max_abs_T": float(np.max(np.abs(T))),
        "max_abs_G": float(np.max(np.abs(G))),
        "integrated_R": float(np.trapezoid(R, t)),
        "lag_R_to_T": float(lag_rt),
        "lag_T_to_G": float(lag_tg),
        "freq_T": freq_T,
        "freq_G": freq_G,
        "frequency_class_T": classify_frequency(freq_T, cfg),
        "frequency_class_G": classify_frequency(freq_G, cfg),
        "attractor_label": classify_attractor(t, a, T, G, cfg),
        "ledger_abs_residual": float(ledger_residual),
        "ledger_residual_per_time": float(ledger_residual / max(cfg["T"], 1e-12)),
        "bounded_pass": bool(np.max(np.abs(T)) < cfg["runaway_threshold"] and np.max(np.abs(G)) < cfg["runaway_threshold"]),
    }
    write_csv(outdir / f"{cfg['run_id']}_timeseries.csv", rows)
    return summary


def base_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "omega_T": args.omega_T,
        "omega_G": args.omega_G,
        "gamma_T": args.gamma_T,
        "gamma_G": args.gamma_G,
        "alpha_T": args.alpha_T,
        "lambda_a": args.lambda_a,
        "a_floor": args.a_floor,
        "a0": args.a0,
        "P_c": args.P_c,
        "delta_P": args.delta_P,
        "dt": args.dt,
        "T": args.T,
        "sample_dt": args.sample_dt,
        "runaway_threshold": args.runaway_threshold,
    }


def make_matrix(args: argparse.Namespace) -> list[dict[str, Any]]:
    base = base_config(args)
    rows: list[dict[str, Any]] = []
    for source_family in ("coherence", "threshold"):
        for kappa in (0.0, 0.25, 0.55, 0.85):
            cfg = {
                **base,
                "run_id": f"{source_family}_k{kappa:.2f}_open",
                "source_family": source_family,
                "kappa_TG": kappa,
                "node_feedback_gain": 0.0,
                "source_enabled": True,
                "temporal_enabled": True,
                "geometric_enabled": kappa > 0,
                "full_loop": False,
            }
            rows.append(cfg)
        cfg = {
            **base,
            "run_id": f"{source_family}_closed_feedback",
            "source_family": source_family,
            "kappa_TG": 0.55,
            "node_feedback_gain": args.node_feedback_gain,
            "source_enabled": True,
            "temporal_enabled": True,
            "geometric_enabled": True,
            "full_loop": True,
        }
        rows.append(cfg)
    # Intervention rows.
    for run_id, overrides in {
        "intervention_source_off": {"source_enabled": False},
        "intervention_temporal_off": {"temporal_enabled": False},
        "intervention_geometric_off": {"geometric_enabled": False, "kappa_TG": 0.0},
    }.items():
        cfg = {
            **base,
            "run_id": run_id,
            "source_family": "coherence",
            "kappa_TG": 0.55,
            "node_feedback_gain": args.node_feedback_gain,
            "source_enabled": True,
            "temporal_enabled": True,
            "geometric_enabled": True,
            "full_loop": False,
        }
        cfg.update(overrides)
        rows.append(cfg)
    return rows


def compare_interventions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {r["run_id"]: r for r in rows}
    ref = by_id.get("coherence_k0.55_open", {})
    src_off = by_id.get("intervention_source_off", {})
    temp_off = by_id.get("intervention_temporal_off", {})
    geom_off = by_id.get("intervention_geometric_off", {})
    return [
        {
            "test": "source_off_removes_T",
            "reference_max_T": ref.get("max_abs_T", float("nan")),
            "control_max_T": src_off.get("max_abs_T", float("nan")),
            "pass": bool(src_off.get("max_abs_T", 1.0) < 1e-8),
        },
        {
            "test": "temporal_off_removes_T_and_G",
            "control_max_T": temp_off.get("max_abs_T", float("nan")),
            "control_max_G": temp_off.get("max_abs_G", float("nan")),
            "pass": bool(temp_off.get("max_abs_T", 1.0) < 1e-8 and temp_off.get("max_abs_G", 1.0) < 1e-8),
        },
        {
            "test": "geometric_off_retains_T_removes_G",
            "control_max_T": geom_off.get("max_abs_T", float("nan")),
            "control_max_G": geom_off.get("max_abs_G", float("nan")),
            "pass": bool(geom_off.get("max_abs_T", 0.0) > 1e-4 and geom_off.get("max_abs_G", 1.0) < 1e-8),
        },
    ]


def stage_label(rows: list[dict[str, Any]], interventions: list[dict[str, Any]]) -> str:
    bounded = [r for r in rows if r["bounded_pass"]]
    if not bounded:
        return "TG_B0_NO_BOUNDED_FEEDBACK_REGION_FOUND"
    if not all(r["pass"] for r in interventions):
        return "TG_FEEDBACK_SEQUENCE_NOT_SUPPORTED"
    if any(r["attractor_label"] == "BOUNDED_LIMIT_CYCLE" for r in rows):
        return "TG_B0_LIMIT_CYCLE_SUPPORTED"
    return "TG_B0_BOUNDED_FEEDBACK_REGION_FOUND"


def write_docs(outdir: Path, summaries: list[dict[str, Any]], interventions: list[dict[str, Any]], label: str) -> None:
    docdir = ROOT / "docs" / "gravity_maturity"
    docdir.mkdir(parents=True, exist_ok=True)
    promoted = [r for r in summaries if r["run_id"] in ("coherence_k0.55_open", "coherence_closed_feedback", "threshold_k0.55_open")]
    (docdir / "TG_B0_RESULTS.md").write_text(
        "\n".join(
            [
                "# TG-B0 Reduced Feedback Exchange Results",
                "",
                "Timestamp: 2026-07-14.",
                f"Run directory: `{outdir.as_posix()}`.",
                f"Status: `{label}`.",
                "",
                "## Intervention Causality",
                "",
                "| test | pass |",
                "| --- | --- |",
                *[f"| {r['test']} | {r['pass']} |" for r in interventions],
                "",
                "## Representative Rows",
                "",
                "| run | source | attractor | max T | max G | lag R->T | lag T->G | freq T class | freq G class |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
                *[
                    f"| {r['run_id']} | {r['source_family']} | {r['attractor_label']} | {r['max_abs_T']:.6e} | {r['max_abs_G']:.6e} | {r['lag_R_to_T']:.6e} | {r['lag_T_to_G']:.6e} | {r['frequency_class_T']} | {r['frequency_class_G']} |"
                    for r in promoted
                ],
                "",
                "## Interpretation",
                "",
                "TG-B0 is a reduced exchange gate. It can support bounded response and intervention causality, but it cannot prove spatial radiation or node stabilization.",
            ]
        ),
        encoding="utf-8",
    )
    write_json(
        docdir / "TG_B0_SUMMARY.json",
        {
            "status": label,
            "run_directory": str(outdir),
            "interventions_pass": all(r["pass"] for r in interventions),
            "bounded_rows": sum(1 for r in summaries if r["bounded_pass"]),
            "total_rows": len(summaries),
        },
    )
    (docdir / "TG_B0_DOCUMENTATION_INPUTS.md").write_text(
        "\n".join(
            [
                "# TG-B0 Documentation Inputs",
                "",
                f"- Bounded label: `{label}`.",
                "- TG-B0 supports only reduced response-chain claims.",
                "- TG-B1 is permitted only for preregistered bounded rows if interventions pass.",
                f"- Artifacts: `{outdir.as_posix()}`.",
            ]
        ),
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
    ap.add_argument("--T", type=float, default=80.0)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--sample-dt", type=float, default=0.05)
    ap.add_argument("--omega-T", type=float, default=1.25)
    ap.add_argument("--omega-G", type=float, default=0.85)
    ap.add_argument("--gamma-T", type=float, default=0.08)
    ap.add_argument("--gamma-G", type=float, default=0.06)
    ap.add_argument("--alpha-T", type=float, default=0.4)
    ap.add_argument("--lambda-a", type=float, default=0.18)
    ap.add_argument("--a-floor", type=float, default=0.08)
    ap.add_argument("--a0", type=float, default=1.0)
    ap.add_argument("--P-c", type=float, default=0.5)
    ap.add_argument("--delta-P", type=float, default=0.08)
    ap.add_argument("--node-feedback-gain", type=float, default=0.06)
    ap.add_argument("--runaway-threshold", type=float, default=50.0)
    args = ap.parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.out) if args.out else ROOT / "sweep_runs" / f"TG_B0_EXCHANGE_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "git_state_before.txt").write_text(git_state(), encoding="utf-8")
    write_json(
        outdir / "environment_versions.json",
        {
            "stage": "TG-B0",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "git_commit": git_commit(),
            "command_line": command_line(),
        },
    )
    matrix = make_matrix(args)
    write_json(outdir / "preregistered_matrix.json", matrix)
    write_json(outdir / "model_spec.json", {"model_class": "TG-A explicitly budgeted dissipative response", "config": base_config(args)})
    summaries = [run_case(cfg, outdir) for cfg in matrix]
    interventions = compare_interventions(summaries)
    label = stage_label(summaries, interventions)
    write_csv(outdir / "run_manifest.csv", summaries)
    write_csv(outdir / "energy_ledger.csv", summaries)
    write_csv(outdir / "frequency_metrics.csv", summaries)
    write_csv(outdir / "causal_lag_metrics.csv", summaries)
    write_csv(outdir / "attractor_classification.csv", summaries)
    write_csv(outdir / "falsification_results.csv", interventions)
    write_json(outdir / "summary.json", {"status": label, "interventions": interventions, "rows": summaries})
    (outdir / "TECHNICAL_HANDOFF.md").write_text(f"# TG-B0 Technical Handoff\n\nStatus: `{label}`.\n", encoding="utf-8")
    (outdir / "OPEN_QUESTIONS.md").write_text(
        "# TG-B0 Open Questions\n\n- Does the reduced bounded response survive in the spatial KG pilot?\n- Does closing G->phi change the node state or only the response fields?\n",
        encoding="utf-8",
    )
    write_docs(outdir, summaries, interventions, label)
    write_csv(outdir / "artifact_hashes.csv", artifact_hashes(outdir))
    print(json.dumps({"status": label, "outdir": str(outdir), "interventions_pass": all(r["pass"] for r in interventions)}, indent=2))


if __name__ == "__main__":
    main()
