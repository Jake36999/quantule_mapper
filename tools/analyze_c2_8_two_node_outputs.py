"""Analyze Phase D C2.8 two-node output artifacts.

Reads an existing C2.8 output folder and produces Claude handover tables,
neutral reports, and PNG plots. It does not run simulations or alter solver
physics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


CASE_FILES = {
    "t0_hold": "rho_x_t0_hold.npz",
    "t1_headon_n2": "rho_x_t1_headon_n2.npz",
    "t1_headon_n4": "rho_x_t1_headon_n4.npz",
    "t2_inphase": "rho_x_t2_inphase.npz",
    "t2_antiphase": "rho_x_t2_antiphase.npz",
}


def load_summary(out_dir: Path) -> dict[str, Any]:
    return json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))


def two_peak(rho_x: np.ndarray, L: float = 20.0, min_index_sep: int = 6) -> tuple[list[float], float, list[int]]:
    n = int(rho_x.shape[0])
    idx = np.argsort(rho_x)[::-1]
    p1 = int(idx[0])
    p2 = None
    for candidate in idx[1:]:
        d = abs(int(candidate) - p1)
        d = min(d, n - d)
        if d > min_index_sep:
            p2 = int(candidate)
            break
    xs = np.linspace(-L / 2.0, L / 2.0, n, endpoint=False)
    if p2 is None:
        return [float(xs[p1])], math.nan, [p1]
    sep = min(abs(p1 - p2), n - abs(p1 - p2)) * (L / n)
    pairs = sorted([(float(xs[p1]), p1), (float(xs[p2]), p2)], key=lambda x: x[0])
    return [p[0] for p in pairs], float(sep), [p[1] for p in pairs]


def slope_for_window(t: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float | None:
    t_sel = np.asarray(t[mask], dtype=float)
    y_sel = np.asarray(y[mask], dtype=float)
    keep = np.isfinite(y_sel)
    t_sel = t_sel[keep]
    y_sel = y_sel[keep]
    if t_sel.size < 2:
        return None
    return float(np.polyfit(t_sel, y_sel, 1)[0])


def analyze_case(out_dir: Path, case_id: str, filename: str, summary: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    path = out_dir / filename
    data = np.load(path)
    rho_x = np.asarray(data["rho_x"], dtype=float)
    t = np.asarray(data["t"], dtype=float)
    if rho_x.ndim != 2:
        raise ValueError(f"{path} has unexpected rho_x shape {rho_x.shape}")
    peak_rows: list[dict[str, Any]] = []
    sep_rows: list[dict[str, Any]] = []
    masses = rho_x.sum(axis=1)
    m0 = float(masses[0]) if masses.size else 1.0
    seps: list[float] = []
    for i, tt in enumerate(t):
        positions, sep, indices = two_peak(rho_x[i])
        seps.append(sep)
        sep_rows.append(
            {
                "case_id": case_id,
                "t": float(tt),
                "sep": sep,
                "n_peaks": len(positions),
                "mass_from_rho_x": float(masses[i]),
                "mass_ret_from_first_saved": float(masses[i] / m0) if m0 else None,
            }
        )
        for j, pos in enumerate(positions):
            peak_rows.append(
                {
                    "case_id": case_id,
                    "t": float(tt),
                    "peak_rank_left_to_right": j,
                    "x": pos,
                    "index": int(indices[j]),
                    "rho_x_value": float(rho_x[i, indices[j]]),
                }
            )
    seps_arr = np.asarray(seps, dtype=float)
    finite = np.isfinite(seps_arr)
    min_sep = float(np.nanmin(seps_arr)) if finite.any() else None
    min_idx = int(np.nanargmin(seps_arr)) if finite.any() else 0
    pre_slope = slope_for_window(t, seps_arr, finite & (np.arange(len(t)) <= min_idx))
    post_slope = slope_for_window(t, seps_arr, finite & (np.arange(len(t)) >= min_idx))

    summary_key = {
        "t1_headon_n2": "T1_n2",
        "t1_headon_n4": "T1_n4",
        "t2_inphase": "T2_inphase",
        "t2_antiphase": "T2_antiphase",
    }.get(case_id)
    summary_block = summary.get(summary_key, {}) if summary_key else {}
    tail = summary_block.get("traj_tail", [])
    final_tail = tail[-1] if tail else {}
    if case_id.startswith("t1_headon"):
        neutral_class = "TWO_CORES_SURVIVE_CLOSE_ENCOUNTER" if final_tail.get("n_peaks") == 2 else "AMBIGUOUS"
    elif case_id == "t2_inphase":
        neutral_class = "STATIC_INPHASE_ATTRACTION"
    elif case_id == "t2_antiphase":
        neutral_class = "STATIC_ANTIPHASE_REPULSION"
    else:
        neutral_class = "HOLD_CONTROL"
    table = {
        "case_id": case_id,
        "source_npz": filename,
        "neutral_label": neutral_class,
        "min_sep_from_rho_x": min_sep,
        "final_sep_from_rho_x": float(seps_arr[-1]) if seps_arr.size and np.isfinite(seps_arr[-1]) else None,
        "final_n_peaks_from_rho_x": int(sep_rows[-1]["n_peaks"]) if sep_rows else None,
        "mass_ret_from_first_saved_final": float(masses[-1] / m0) if m0 else None,
        "pre_min_sep_slope": pre_slope,
        "post_min_sep_slope": post_slope,
        "summary_verdict_or_trend": summary_block.get("verdict", summary_block.get("trend")),
        "summary_final_mass": final_tail.get("mass"),
        "summary_final_P": final_tail.get("P"),
        "summary_final_amp": final_tail.get("amp"),
        "summary_final_n_peaks": final_tail.get("n_peaks"),
        "caveat": "rho_x lacks phase/identity information; pass-through vs rebound remains neutral unless identities are labelled.",
    }
    return peak_rows, sep_rows, table


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_heatmap(out_dir: Path, case_id: str, filename: str) -> None:
    data = np.load(out_dir / filename)
    rho_x = np.asarray(data["rho_x"], dtype=float)
    t = np.asarray(data["t"], dtype=float)
    x = np.linspace(-10.0, 10.0, rho_x.shape[1], endpoint=False)
    plt.figure(figsize=(9, 4.8))
    plt.imshow(
        rho_x,
        aspect="auto",
        origin="lower",
        extent=[float(x[0]), float(x[-1]), float(t[0]), float(t[-1])],
        cmap="magma",
    )
    plt.colorbar(label="rho_x")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title(f"{case_id}: rho_x spacetime")
    plt.tight_layout()
    plt.savefig(out_dir / f"{case_id}_rho_x_spacetime.png", dpi=160)
    plt.close()


def plot_lines(out_dir: Path, peak_rows: list[dict[str, Any]], sep_rows: list[dict[str, Any]]) -> None:
    cases = sorted({r["case_id"] for r in sep_rows})
    plt.figure(figsize=(8.8, 5.0))
    for case_id in cases:
        rows = [r for r in sep_rows if r["case_id"] == case_id]
        plt.plot([r["t"] for r in rows], [r["sep"] for r in rows], marker="o", label=case_id)
    plt.xlabel("t")
    plt.ylabel("separation")
    plt.title("C2.8 separation vs time")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "c2_8_separation_vs_time.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8.8, 5.0))
    for case_id in cases:
        rows = [r for r in sep_rows if r["case_id"] == case_id]
        plt.plot([r["t"] for r in rows], [r["mass_ret_from_first_saved"] for r in rows], marker="o", label=case_id)
    plt.xlabel("t")
    plt.ylabel("mass retention from first saved rho_x frame")
    plt.title("C2.8 mass retention proxy")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "c2_8_mass_retention_vs_time.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8.8, 5.0))
    for case_id in sorted({r["case_id"] for r in peak_rows}):
        for rank in sorted({r["peak_rank_left_to_right"] for r in peak_rows if r["case_id"] == case_id}):
            rows = [r for r in peak_rows if r["case_id"] == case_id and r["peak_rank_left_to_right"] == rank]
            plt.plot([r["t"] for r in rows], [r["x"] for r in rows], marker="o", label=f"{case_id} peak{rank}")
    plt.xlabel("t")
    plt.ylabel("x peak position")
    plt.title("C2.8 peak trajectories")
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "c2_8_peak_trajectories_vs_time.png", dpi=160)
    plt.close()


def plot_momentum_from_summary(out_dir: Path, summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mapping = {"T1_n2": "t1_headon_n2", "T1_n4": "t1_headon_n4", "T2_inphase": "t2_inphase", "T2_antiphase": "t2_antiphase"}
    for key, case_id in mapping.items():
        for item in summary.get(key, {}).get("traj_tail", []):
            rows.append(
                {
                    "case_id": case_id,
                    "t": item.get("t"),
                    "mass": item.get("mass"),
                    "P": item.get("P"),
                    "amp": item.get("amp"),
                    "n_peaks": item.get("n_peaks"),
                    "source": "summary_traj_tail_only",
                }
            )
    if rows:
        plt.figure(figsize=(8.8, 5.0))
        for case_id in sorted({r["case_id"] for r in rows}):
            cr = [r for r in rows if r["case_id"] == case_id]
            plt.plot([r["t"] for r in cr], [r["P"] for r in cr], marker="o", label=case_id)
        plt.xlabel("t")
        plt.ylabel("P from summary tail")
        plt.title("C2.8 momentum vs time (tail only)")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "c2_8_momentum_vs_time_tail_only.png", dpi=160)
        plt.close()
    return rows


def write_report(out_dir: Path, tables: list[dict[str, Any]], momentum_rows: list[dict[str, Any]]) -> None:
    by_case = {r["case_id"]: r for r in tables}
    lines = [
        "# C2.8 Two-Node Output Analysis",
        "",
        "Diagnostic-only analysis of saved C2.8 artifacts. No broad stability or matter claim is made.",
        "",
        "## Directly Supported Measurements",
        "",
    ]
    for case_id in ["t1_headon_n2", "t1_headon_n4", "t2_inphase", "t2_antiphase"]:
        row = by_case.get(case_id, {})
        lines.extend(
            [
                f"### {case_id}",
                "",
                f"- Neutral label: `{row.get('neutral_label')}`",
                f"- Min separation from rho_x: `{row.get('min_sep_from_rho_x')}`",
                f"- Final separation from rho_x: `{row.get('final_sep_from_rho_x')}`",
                f"- Final peak count from rho_x: `{row.get('final_n_peaks_from_rho_x')}`",
                f"- Summary final mass: `{row.get('summary_final_mass')}`",
                f"- Pre-min separation slope: `{row.get('pre_min_sep_slope')}`",
                f"- Post-min separation slope: `{row.get('post_min_sep_slope')}`",
                f"- Caveat: {row.get('caveat')}",
                "",
            ]
        )
    lines.extend(
        [
            "## Notes",
            "",
            "- Head-on cases are labelled `TWO_CORES_SURVIVE_CLOSE_ENCOUNTER`; this does not distinguish pass-through from rebound.",
            "- `rho_x_*.npz` files contain projected density only, not phase-labelled soliton identities.",
            "- Momentum plots use only the `traj_tail` entries preserved in `summary.json`.",
            "- Mass-retention time series from `rho_x` are normalized to the first saved projected-density frame, not t=0.",
            "",
            "## Generated Files",
            "",
            "- `c2_8_summary_tables.csv`",
            "- `c2_8_peak_trajectories.csv`",
            "- `c2_8_separation_vs_time.csv`",
            "- `c2_8_mass_momentum_vs_time.csv`",
            "- `*_rho_x_spacetime.png`",
            "- `c2_8_peak_trajectories_vs_time.png`",
            "- `c2_8_separation_vs_time.png`",
            "- `c2_8_mass_retention_vs_time.png`",
            "- `c2_8_momentum_vs_time_tail_only.png`" if momentum_rows else "- Momentum PNG skipped: no summary tail momentum rows found.",
        ]
    )
    (out_dir / "c2_8_analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def analyze(out_dir: Path) -> None:
    summary = load_summary(out_dir)
    all_peak_rows: list[dict[str, Any]] = []
    all_sep_rows: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    for case_id, filename in CASE_FILES.items():
        if not (out_dir / filename).exists():
            continue
        peak_rows, sep_rows, table = analyze_case(out_dir, case_id, filename, summary)
        all_peak_rows.extend(peak_rows)
        all_sep_rows.extend(sep_rows)
        table_rows.append(table)
        plot_heatmap(out_dir, case_id, filename)
    momentum_rows = plot_momentum_from_summary(out_dir, summary)
    plot_lines(out_dir, all_peak_rows, all_sep_rows)
    write_csv(out_dir / "c2_8_summary_tables.csv", table_rows)
    write_csv(out_dir / "c2_8_peak_trajectories.csv", all_peak_rows)
    write_csv(out_dir / "c2_8_separation_vs_time.csv", all_sep_rows)
    write_csv(out_dir / "c2_8_mass_momentum_vs_time.csv", momentum_rows)
    write_report(out_dir, table_rows, momentum_rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Existing C2.8 output folder")
    args = ap.parse_args()
    analyze(Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
