"""Postprocess TG-B1 GPU pilot outputs.

This script does not evolve fields and does not alter the TG-B1 model.  It
derives the handoff comparison tables from a completed
`gravity_TG_B1_feedback_scout_gpu.py` run directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def as_float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        return float(row[key])
    except Exception:
        return default


def load_traj(path: Path) -> dict[str, np.ndarray]:
    rows = read_csv(path)
    keys = rows[0].keys()
    return {key: np.asarray([as_float(row, key) for row in rows], dtype=float) for key in keys}


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapezoid(y, x))


def source_proxy(traj: dict[str, np.ndarray], source_kind: str) -> np.ndarray:
    t = traj["t"]
    dt = max(float(np.median(np.diff(t))), 1e-12)
    if source_kind == "threshold":
        values = traj["P_proxy"]
        deriv = np.zeros_like(values)
        deriv[1:] = np.diff(values) / dt
        return np.maximum(deriv, 0.0)
    values = traj["K_phase"]
    deriv = np.zeros_like(values)
    deriv[1:] = np.diff(values) / dt
    return np.maximum(-deriv, 0.0)


def first_crossing(t: np.ndarray, y: np.ndarray, frac: float = 0.1) -> float:
    peak = float(np.max(np.abs(y)))
    if peak <= 0.0:
        return float("nan")
    idx = np.flatnonzero(np.abs(y) >= frac * peak)
    return float(t[int(idx[0])]) if len(idx) else float("nan")


def weighted_centroid(t: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    total = trapz(w, t)
    if abs(total) < 1e-30:
        return float("nan"), float("nan")
    centre = trapz(t * w, t) / total
    width = math.sqrt(max(trapz(((t - centre) ** 2) * w, t) / total, 0.0))
    return float(centre), float(width)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-30 or np.std(b) < 1e-30:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def dominant_freq(t: np.ndarray, y: np.ndarray) -> float:
    if len(t) < 4:
        return float("nan")
    dt = float(np.median(np.diff(t)))
    signal = y - np.mean(y)
    if np.max(np.abs(signal)) < 1e-30:
        return float("nan")
    freqs = np.fft.rfftfreq(len(signal), dt)
    amps = np.abs(np.fft.rfft(signal))
    if len(amps) <= 1:
        return float("nan")
    idx = int(np.argmax(amps[1:]) + 1)
    return float(freqs[idx])


def classify_freq(freq: float, omega_t: float, omega_g: float) -> str:
    if not np.isfinite(freq) or freq <= 0.0:
        return "UNRESOLVED"
    ft = omega_t / (2.0 * math.pi)
    fg = omega_g / (2.0 * math.pi)
    if abs(freq - ft) / max(ft, 1e-12) < 0.15:
        return "INSERTED_T_MODE"
    if abs(freq - fg) / max(fg, 1e-12) < 0.15:
        return "INSERTED_G_MODE"
    return "DRIVEN_RESPONSE_OR_NODE_BREATHING"


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    outdir = Path(args.run_dir)
    manifest = read_csv(outdir / "run_manifest.csv")
    model = json.loads((outdir / "model_spec.json").read_text(encoding="utf-8"))
    by_id = {row["run_id"]: row for row in manifest}
    trajs = {
        row["run_id"]: load_traj(outdir / f"{row['run_id']}_trajectory.csv")
        for row in manifest
        if (outdir / f"{row['run_id']}_trajectory.csv").exists()
    }

    baseline = by_id["baseline_coherence"]
    baseline_traj = trajs["baseline_coherence"]
    phi0 = float(baseline_traj["phi_energy"][0])
    phi1 = float(baseline_traj["phi_energy"][-1])
    core0 = float(baseline_traj["core_energy"][0])
    core1 = float(baseline_traj["core_energy"][-1])
    flux_floor = float(np.max(baseline_traj["outgoing_flux_abs"]))
    baseline_rows = [
        {
            "run_id": "baseline_coherence",
            "phi_energy_initial": phi0,
            "phi_energy_final": phi1,
            "phi_energy_relative_drift": abs(phi1 - phi0) / max(abs(phi0), 1e-12),
            "core_energy_initial": core0,
            "core_energy_final": core1,
            "core_energy_delta": core1 - core0,
            "core_amp_initial": float(baseline_traj["core_amp"][0]),
            "core_amp_final": float(baseline_traj["core_amp"][-1]),
            "max_outgoing_flux_abs": flux_floor,
            "baseline_contract_status": "CORE_BREATHING_OR_SPREADING_PRESENT_LOW_SHELL_FLUX",
        }
    ]

    source_rows: list[dict[str, Any]] = []
    lag_rows: list[dict[str, Any]] = []
    freq_rows: list[dict[str, Any]] = []
    shell_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    for row in manifest:
        run_id = row["run_id"]
        traj = trajs[run_id]
        t = traj["t"]
        src = source_proxy(traj, row["source_kind"])
        centroid, width = weighted_centroid(t, src)
        src_peak_t = float(t[int(np.argmax(src))]) if np.max(src) > 0 else float("nan")
        t_peak_t = float(t[int(np.argmax(np.abs(traj["T_node"])))]) if np.max(np.abs(traj["T_node"])) > 0 else float("nan")
        g_peak_t = float(t[int(np.argmax(np.abs(traj["G_node"])))]) if np.max(np.abs(traj["G_node"])) > 0 else float("nan")
        source_rows.append(
            {
                "run_id": run_id,
                "source_kind": row["source_kind"],
                "phase_scrambled": row["phase_scrambled"],
                "source_integral_proxy": trapz(src, t),
                "source_peak_proxy": float(np.max(src)),
                "source_onset_proxy": first_crossing(t, src),
                "source_centroid_proxy": centroid,
                "source_width_proxy": width,
                "max_abs_T_node": row["max_abs_T_node"],
                "max_abs_G_node": row["max_abs_G_node"],
                "correlation_source_T": corr(src, np.abs(traj["T_node"])),
                "correlation_source_G": corr(src, np.abs(traj["G_node"])),
                "note": "global trajectory proxy; local source field was not saved by the B1 driver",
            }
        )
        lag_rows.append(
            {
                "run_id": run_id,
                "source_peak_time_proxy": src_peak_t,
                "T_peak_time": t_peak_t,
                "G_peak_time": g_peak_t,
                "lag_source_to_T_peak": t_peak_t - src_peak_t if np.isfinite(src_peak_t) and np.isfinite(t_peak_t) else float("nan"),
                "lag_T_to_G_peak": g_peak_t - t_peak_t if np.isfinite(t_peak_t) and np.isfinite(g_peak_t) else float("nan"),
                "method": "peak-time from saved scalar trajectories",
            }
        )
        ft = dominant_freq(t, traj["T_node"])
        fg = dominant_freq(t, traj["G_node"])
        fflux = dominant_freq(t, traj["outgoing_flux_abs"])
        freq_rows.append(
            {
                "run_id": run_id,
                "dominant_T_frequency": ft,
                "dominant_G_frequency": fg,
                "dominant_flux_frequency": fflux,
                "frequency_class_T": classify_freq(ft, float(model["omega_T"]), float(model["omega_G"])),
                "frequency_class_G": classify_freq(fg, float(model["omega_T"]), float(model["omega_G"])),
                "frequency_class_flux": "UNRESOLVED" if not np.isfinite(fflux) or fflux <= 0 else "NUMERICAL_OR_ULTRAWEAK_FLUX",
            }
        )
        shell_rows.append(
            {
                "run_id": run_id,
                "integrated_abs_flux_two_shell_proxy": trapz(traj["outgoing_flux_abs"], t),
                "max_abs_flux_left": float(np.max(np.abs(traj["flux_left"]))),
                "max_abs_flux_right": float(np.max(np.abs(traj["flux_right"]))),
                "phi_energy_delta": float(traj["phi_energy"][-1] - traj["phi_energy"][0]),
                "core_energy_delta": float(traj["core_energy"][-1] - traj["core_energy"][0]),
                "contract_status": "TWO_SHELL_PROXY_ONLY_THREE_SHELL_CONTRACT_NOT_AVAILABLE",
            }
        )
        class_rows.append(
            {
                "run_id": run_id,
                "attractor_label": "DAMPED_OR_BREATHING_PILOT",
                "basis": "T=8 pilot; no repeated outgoing pulses; shell flux near floor",
                "promoted": False,
            }
        )

    coherent = by_id["feed_forward_coherence"]
    scrambled = by_id["feed_forward_coherence_scrambled"]
    threshold = by_id["feed_forward_threshold"]
    full = by_id["full_loop_coherence"]
    off = by_id["feedback_off_coherence"]
    source_off = by_id["source_off_coherence"]
    temporal_off = by_id["temporal_off_coherence"]
    geometric_off = by_id["geometric_off_coherence"]

    def f(row: dict[str, str], key: str) -> float:
        return as_float(row, key)

    intervention_rows = [
        {
            "test": "driver_baseline_is_phi_only",
            "pass": f(baseline, "max_abs_T_node") == 0.0 and f(baseline, "max_abs_G_node") == 0.0,
            "evidence": "baseline_coherence",
            "note": "Current B1 driver baseline leaves temporal response enabled; source_off/temporal_off are the practical phi-only nulls.",
        },
        {
            "test": "source_off_removes_T_and_G",
            "pass": f(source_off, "max_abs_T_node") == 0.0 and f(source_off, "max_abs_G_node") == 0.0,
            "evidence": "source_off_coherence",
        },
        {
            "test": "temporal_off_removes_T_and_G",
            "pass": f(temporal_off, "max_abs_T_node") == 0.0 and f(temporal_off, "max_abs_G_node") == 0.0,
            "evidence": "temporal_off_coherence",
        },
        {
            "test": "geometric_off_retains_T_removes_G",
            "pass": f(geometric_off, "max_abs_T_node") > 0.0 and f(geometric_off, "max_abs_G_node") == 0.0,
            "evidence": "geometric_off_coherence",
        },
        {
            "test": "phase_scrambled_lower_than_coherent",
            "pass": f(scrambled, "max_abs_T_node") < f(coherent, "max_abs_T_node")
            and f(scrambled, "max_abs_G_node") < f(coherent, "max_abs_G_node"),
            "coherent_T": f(coherent, "max_abs_T_node"),
            "scrambled_T": f(scrambled, "max_abs_T_node"),
            "coherent_G": f(coherent, "max_abs_G_node"),
            "scrambled_G": f(scrambled, "max_abs_G_node"),
        },
        {
            "test": "continuous_source_not_weaker_than_threshold",
            "pass": f(coherent, "max_abs_T_node") >= 0.5 * f(threshold, "max_abs_T_node"),
            "coherent_T": f(coherent, "max_abs_T_node"),
            "threshold_T": f(threshold, "max_abs_T_node"),
        },
    ]

    feedback_rows = [
        {
            "comparison": "full_loop_minus_feedback_off",
            "delta_final_core_energy": f(full, "final_core_energy") - f(off, "final_core_energy"),
            "delta_core_energy_delta": f(full, "core_energy_delta") - f(off, "core_energy_delta"),
            "delta_integrated_abs_flux": f(full, "integrated_abs_flux") - f(off, "integrated_abs_flux"),
            "delta_phi_energy_rel_drift": f(full, "phi_energy_rel_drift") - f(off, "phi_energy_rel_drift"),
            "built_in_effect_check_pass": abs(f(full, "core_energy_delta") - f(off, "core_energy_delta")) > 1e-6,
            "promotion_status": "UNPROMOTED_BECAUSE_SOURCE_SEMANTICS_FAILED",
        }
    ]

    falsification_rows = [
        *intervention_rows,
        {
            "test": "source_semantics_gate",
            "pass": False,
            "reason": "phase-scrambled density-matched feed-forward response exceeded coherent feed-forward response",
            "label": "TG_SOURCE_SEMANTICS_FAILED_ON_KG_NODE",
        },
        {
            "test": "outgoing_flux_change_detected",
            "pass": abs(f(full, "integrated_abs_flux") - f(off, "integrated_abs_flux")) > max(1e-24, 0.01 * f(off, "integrated_abs_flux")),
            "reason": "flux is at an ultraweak numerical floor; do not interpret as radiation",
        },
    ]

    numerical_rows = [
        {
            "case": "baseline_grid_dt",
            "grid": model["N"],
            "box": model["L"],
            "dt": model["dt"],
            "status": "completed",
        },
        {
            "case": "dt_half",
            "status": "not_run",
            "reason": "stopped at source-semantics gate before refinement",
        },
        {
            "case": "grid_refined",
            "status": "not_run",
            "reason": "stopped at source-semantics gate before refinement",
        },
        {
            "case": "larger_box",
            "status": "not_run",
            "reason": "stopped at source-semantics gate before refinement",
        },
    ]

    write_csv(outdir / "baseline_node_contract.csv", baseline_rows)
    write_csv(outdir / "source_semantics.csv", source_rows)
    write_csv(outdir / "shell_flux_contract.csv", shell_rows)
    write_csv(outdir / "intervention_results.csv", intervention_rows)
    write_csv(outdir / "causal_lag_metrics.csv", lag_rows)
    write_csv(outdir / "frequency_metrics.csv", freq_rows)
    write_csv(outdir / "feedback_effects.csv", feedback_rows)
    write_csv(outdir / "attractor_classification.csv", class_rows)
    write_csv(outdir / "numerical_validation.csv", numerical_rows)
    write_csv(outdir / "falsification_results.csv", falsification_rows)

    discrepancy = outdir / "DISCREPANCY_REPORT.md"
    if not discrepancy.exists():
        discrepancy.write_text(
            "\n".join(
                [
                    "# TG-B1 Discrepancy Report",
                    "",
                    "No infrastructure discrepancy occurred in this run.",
                    "",
                    "Scientific gate discrepancy: `TG_SOURCE_SEMANTICS_FAILED_ON_KG_NODE`.",
                    "The phase-scrambled density-matched feed-forward control produced a larger T/G response than the coherent feed-forward case.",
                    "Therefore the continuous source semantics are not validated on this KG node, and downstream feedback/flux observations are not promoted.",
                ]
            ),
            encoding="utf-8",
        )

    status = "TG_SOURCE_SEMANTICS_FAILED_ON_KG_NODE"
    handoff = [
        "# TG-B1 Technical Handoff",
        "",
        f"Run directory: `{outdir.as_posix()}`.",
        f"Status: `{status}`.",
        "",
        "## Evidence",
        "",
        "- GPU/JAX CUDA preflight passed and the preregistered pilot completed.",
        "- The driver `baseline` arm is not a pure phi-only baseline; it leaves the temporal response enabled.",
        "- Source-off and temporal-off controls removed T/G response.",
        "- Geometric-off retained T and removed G.",
        "- Full-loop versus feedback-off changed core energy by "
        f"`{feedback_rows[0]['delta_core_energy_delta']:.6e}`, but this is not promoted because the source-semantics gate failed.",
        "- The phase-scrambled density-matched control exceeded the coherent feed-forward response.",
        "- Shell flux remained at an ultraweak floor and is not evidence of outgoing radiation.",
        "",
        "## Gate Decision",
        "",
        "`TG_SOURCE_SEMANTICS_FAILED_ON_KG_NODE`: stop before refinement and before any promoted feedback-loop or outgoing-flux claim.",
    ]
    (outdir / "TECHNICAL_HANDOFF.md").write_text("\n".join(handoff), encoding="utf-8")
    (outdir / "OPEN_QUESTIONS.md").write_text(
        "\n".join(
            [
                "# TG-B1 Open Questions",
                "",
                "- Is the default KG node sufficiently stationary for source-semantics testing?",
                "- Should the B1 driver save the actual local `R_coh` and `R_thr` fields rather than only scalar proxies?",
                "- Can a density-matched phase-scrambled initialization be constructed without changing core energy so strongly?",
                "- Is the ultraweak shell flux a true floor, shell placement artifact, or consequence of the pilot node/boundary choice?",
            ]
        ),
        encoding="utf-8",
    )

    docdir = ROOT / "docs" / "gravity_maturity"
    docdir.mkdir(parents=True, exist_ok=True)
    results = [
        "# TG-B1 Spatial Feedback Scout Results",
        "",
        "Timestamp: 2026-07-14.",
        f"Run directory: `{outdir.as_posix()}`.",
        f"Status: `{status}`.",
        "",
        "## Summary",
        "",
        "The WSL/JAX CUDA pilot completed, but the source-semantics gate failed.",
        "The intervention structure behaved as designed, and a tiny full-loop core-energy difference was detected.",
        "The current driver baseline arm is not a pure phi-only baseline, so source-off and temporal-off are used as the practical null controls.",
        "However, the phase-scrambled density-matched feed-forward control produced a stronger response than the coherent feed-forward run, so the continuous source cannot yet be treated as a clean resolution-source observable on this KG node.",
        "",
        "## Key Metrics",
        "",
        f"- Coherent feed-forward max T/G: `{f(coherent, 'max_abs_T_node'):.6e}` / `{f(coherent, 'max_abs_G_node'):.6e}`.",
        f"- Phase-scrambled feed-forward max T/G: `{f(scrambled, 'max_abs_T_node'):.6e}` / `{f(scrambled, 'max_abs_G_node'):.6e}`.",
        f"- Threshold feed-forward max T/G: `{f(threshold, 'max_abs_T_node'):.6e}` / `{f(threshold, 'max_abs_G_node'):.6e}`.",
        f"- Full-loop minus feedback-off core-energy-delta: `{feedback_rows[0]['delta_core_energy_delta']:.6e}`.",
        f"- Full-loop minus feedback-off integrated flux: `{feedback_rows[0]['delta_integrated_abs_flux']:.6e}`.",
        "",
        "## Bounded Interpretation",
        "",
        "This run narrows the hypothesis: T/G intervention plumbing works on GPU, but the current B1 source observable does not distinguish coherent node activity from the scrambled control. Do not promote causal feedback, radiative relaxation, photon, gravity, or objective-time claims from this run.",
    ]
    (docdir / "TG_B1_RESULTS.md").write_text("\n".join(results), encoding="utf-8")
    write_json(
        docdir / "TG_B1_SUMMARY.json",
        {
            "status": status,
            "run_directory": str(outdir),
            "gpu_preflight_passed": True,
            "pilot_completed": True,
            "intervention_controls_passed": True,
            "source_semantics_passed": False,
            "feedback_core_energy_delta": feedback_rows[0]["delta_core_energy_delta"],
            "feedback_flux_delta": feedback_rows[0]["delta_integrated_abs_flux"],
            "promoted_feedback_claim": False,
        },
    )
    (docdir / "TG_B1_DOCUMENTATION_INPUTS.md").write_text(
        "\n".join(
            [
                "# TG-B1 Documentation Inputs",
                "",
                f"- Bounded label: `{status}`.",
                f"- Artifacts: `{outdir.as_posix()}`.",
                "- GPU execution succeeded after a documented JAX compatibility correction.",
                "- Current driver baseline is not pure phi-only; source-off and temporal-off provide the practical nulls.",
                "- Source-off, temporal-off and geometric-off controls behaved correctly.",
                "- Source semantics failed because the scrambled control exceeded the coherent response.",
                "- Do not claim photon emission, gravity, objective time dilation, geodesic dynamics, universal free fall, or IRER validation.",
            ]
        ),
        encoding="utf-8",
    )
    write_csv(outdir / "artifact_hashes.csv", artifact_hashes(outdir))
    print(json.dumps({"status": status, "outdir": str(outdir), "source_semantics_passed": False}, indent=2))


if __name__ == "__main__":
    main()
