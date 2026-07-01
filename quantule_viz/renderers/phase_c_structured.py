"""Structured Phase C discovery visual-analysis renderer."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from ..io import choose_npz_key, guard_outputs, load_csv, load_json, load_npz, maybe_float, maybe_int, write_csv
from ..plots import (
    CLASS_COLORS,
    TRUE_CLASS,
    blank_figure,
    render_density_slices,
    render_vector_preview,
    save_figure,
)

DEFAULT_SUMMARY_CSV = Path("docs/phase_c_structured_discovery_B_summary.csv")
DEFAULT_SHORTLIST_JSON = Path("runtime_logs/phase_c_structured_discovery_B_shortlist_metrics.json")

POPULATION_OUTPUTS = [
    "phase_c_true_rate_heatmap.png",
    "phase_c_true_rate_by_seed.png",
    "phase_c_class_composition.png",
    "phase_c_true_nodecount_distribution.png",
    "phase_c_branch_landscape_summary.png",
]
SHORTLIST_OUTPUTS = [
    "phase_c_shortlist_table.csv",
    "phase_c_shortlist_overview.png",
    "phase_c_shortlist_diagnostic_summary.png",
]
CASE_OUTPUTS = [
    "timeline_panel.png",
    "density_slices.png",
    "vector_map.png",
    "case_summary.png",
]

CASE_ORDER = [
    "k6_high_mass_true",
    "k6_mid_mass_true",
    "k4_intermediate_true",
    "k2_intermediate_true",
    "k1_low_mass_true",
    "k1_high_mass_failure",
    "k6_near_threshold_near",
    "feb56dc7_control",
]

CASE_SPECS = {
    "k6_high_mass_true": {
        "source_slug": "best_k6_distributed_true_high_mass",
        "label": "K6 High-Mass TRUE",
        "role": "robust distributed branch",
    },
    "k6_mid_mass_true": {
        "source_slug": "best_k6_distributed_true_mid_mass",
        "label": "K6 Mid-Mass TRUE",
        "role": "robust distributed branch",
    },
    "k4_intermediate_true": {
        "source_slug": "best_k4_intermediate_distributed_true",
        "label": "K4 Intermediate TRUE",
        "role": "intermediate distributed branch",
    },
    "k2_intermediate_true": {
        "source_slug": "best_k2_intermediate_branch",
        "label": "K2 Intermediate TRUE",
        "role": "intermediate branch",
    },
    "k1_low_mass_true": {
        "source_slug": "best_k1_low_mass_true",
        "label": "K1 Low-Mass TRUE",
        "role": "fragile low-mass pocket",
    },
    "k1_high_mass_failure": {
        "source_slug": "best_k1_high_mass_failure_control",
        "label": "K1 High-Mass Failure",
        "role": "high-mass failure control",
    },
    "k6_near_threshold_near": {
        "source_slug": "best_k6_near_threshold_near_candidate",
        "label": "K6 Near-Threshold NEAR",
        "role": "near-threshold inconclusive branch",
    },
    "feb56dc7_control": {
        "source_slug": "feb56dc7_control",
        "label": "feb56dc7 Control",
        "role": "external control",
    },
}

CLASS_ORDER = [
    TRUE_CLASS,
    "NEAR_SATURATED_BOUND_STATE",
    "LATE_BLOWUP_REJECT",
    "SPIN_DOWN_REJECT",
    "TRANSIENT_GROWER_REJECT",
]
CLASS_SHORT = {
    TRUE_CLASS: "TRUE",
    "NEAR_SATURATED_BOUND_STATE": "NEAR",
    "LATE_BLOWUP_REJECT": "BLOWUP",
    "SPIN_DOWN_REJECT": "SPIN",
    "TRANSIENT_GROWER_REJECT": "GROW",
}

DOMINANT_CLASS_COLORS = {
    TRUE_CLASS: "#1a9850",
    "NEAR_SATURATED_BOUND_STATE": "#66bd63",
    "LATE_BLOWUP_REJECT": "#d73027",
    "SPIN_DOWN_REJECT": "#4575b4",
    "TRANSIENT_GROWER_REJECT": "#fdae61",
}


def _resolve_path(path: str | None, default: Path) -> Path:
    chosen = Path(path) if path else default
    if not chosen.is_absolute():
        chosen = (Path.cwd() / chosen).resolve()
    return chosen


def _run_name(value: str | None) -> str:
    if not value:
        return "external_control"
    text = str(value).rstrip("/\\")
    return Path(text).name


def _mass_label(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _safe_float(value: object, default: float = float("nan")) -> float:
    out = maybe_float(value, default=default)
    return out


def _safe_int(value: object, default: int = -1) -> int:
    return maybe_int(value, default=default)


def _structured_rows(path: Path) -> list[dict[str, Any]]:
    rows = load_csv(path)
    for row in rows:
        row["K_int"] = _safe_int(row.get("K"))
        row["mass_float"] = _safe_float(row.get("target_initial_mass"))
        row["dx_mass_float"] = _safe_float(row.get("dx_weighted_target_mass"))
        row["seed_int"] = _safe_int(row.get("ic_seed"))
        row["n_fin_int"] = _safe_int(row.get("n_fin"))
        row["late_slope_float"] = _safe_float(row.get("late_slope"))
        row["er_fin_float"] = _safe_float(row.get("er_fin"))
        row["compactness_float"] = _safe_float(row.get("compactness_max"))
        row["core_radius_float"] = _safe_float(row.get("core_radius_min"))
        row["high_k_float"] = _safe_float(row.get("high_k_fraction_max"))
    return rows


def _shortlist_rows(path: Path) -> list[dict[str, Any]]:
    rows = load_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected shortlist metrics JSON to contain a list, got {type(rows)!r}")
    return rows


def _summary_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        lookup[(_run_name(row.get("run_dir")), _safe_int(row.get("idx")))] = row
    return lookup


def _shortlist_table_rows(
    shortlist: list[dict[str, Any]],
    summary_lookup: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for case_name in CASE_ORDER:
        spec = CASE_SPECS[case_name]
        match = next((row for row in shortlist if row.get("slug") == spec["source_slug"]), None)
        if match is None:
            continue
        run_name = _run_name(match.get("run_dir"))
        idx = _safe_int(match.get("idx"))
        summary_row = summary_lookup.get((run_name, idx))
        output.append(
            {
                "candidate_label": spec["label"],
                "case_key": case_name,
                "source_run": run_name,
                "idx": "" if idx < 0 else idx,
                "K": _safe_int(match.get("K")),
                "raw_target_mass": match.get("target_initial_mass"),
                "dx_weighted_target_mass": (
                    summary_row.get("dx_weighted_target_mass")
                    if summary_row is not None
                    else None
                ),
                "ic_seed": match.get("ic_seed"),
                "class": match.get("class"),
                "diagnostic_label": match.get("diagnostic_label"),
                "final_node_count": match.get("node_count_last"),
                "late_slope": match.get("late_energy_slope"),
                "compactness": match.get("compactness_max"),
                "core_radius": match.get("core_radius_min"),
                "high_k_fraction": match.get("high_k_fraction_max"),
                "selection_reason": match.get("selection_reason"),
                "role": spec["role"],
            }
        )
    return output


def _grid_axes(rows: list[dict[str, Any]]) -> tuple[list[int], list[float], list[int]]:
    ks = sorted({_safe_int(row["K_int"]) for row in rows if _safe_int(row["K_int"]) >= 0})
    masses = sorted({float(row["mass_float"]) for row in rows if np.isfinite(row["mass_float"])})
    seeds = sorted({_safe_int(row["seed_int"]) for row in rows if _safe_int(row["seed_int"]) >= 0})
    return ks, masses, seeds


def _plot_true_rate_heatmap(rows: list[dict[str, Any]], outpath: Path) -> Path:
    ks, masses, _ = _grid_axes(rows)
    matrix = np.full((len(ks), len(masses)), np.nan, dtype=float)
    counts_text: list[list[str]] = [["" for _ in masses] for _ in ks]
    for i, kval in enumerate(ks):
        for j, mass in enumerate(masses):
            bucket = [row for row in rows if row["K_int"] == kval and abs(row["mass_float"] - mass) < 1e-9]
            total = len(bucket)
            true_count = sum(1 for row in bucket if row.get("klass") == TRUE_CLASS)
            matrix[i, j] = true_count / total if total else np.nan
            counts_text[i][j] = f"{true_count}/{total}" if total else "-"
    fig, ax = plt.subplots(figsize=(10, 4.8))
    image = ax.imshow(matrix, cmap="YlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(masses)))
    ax.set_xticklabels([_mass_label(mass) for mass in masses], rotation=30, ha="right")
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels([str(k) for k in ks])
    ax.set_xlabel("raw target mass")
    ax.set_ylabel("initial blob count K")
    ax.set_title("TRUE rate by K and raw target mass")
    for i in range(len(ks)):
        for j in range(len(masses)):
            text = counts_text[i][j]
            pct = matrix[i, j]
            label = f"{text}\n{pct * 100:.0f}%" if np.isfinite(pct) else text
            ax.text(j, i, label, ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="TRUE fraction")
    return save_figure(fig, outpath)


def _plot_true_rate_by_seed(rows: list[dict[str, Any]], outpath: Path) -> Path:
    ks, masses, seeds = _grid_axes(rows)
    fig, axes = plt.subplots(1, len(seeds), figsize=(4.8 * len(seeds), 4.6), sharey=True)
    if len(seeds) == 1:
        axes = [axes]
    for ax, seed in zip(axes, seeds):
        matrix = np.full((len(ks), len(masses)), np.nan, dtype=float)
        for i, kval in enumerate(ks):
            for j, mass in enumerate(masses):
                bucket = [
                    row
                    for row in rows
                    if row["seed_int"] == seed and row["K_int"] == kval and abs(row["mass_float"] - mass) < 1e-9
                ]
                total = len(bucket)
                true_count = sum(1 for row in bucket if row.get("klass") == TRUE_CLASS)
                matrix[i, j] = true_count / total if total else np.nan
                if total:
                    ax.text(j, i, f"{true_count}/{total}", ha="center", va="center", fontsize=8)
        image = ax.imshow(matrix, cmap="YlGn", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title(f"ic_seed={seed}")
        ax.set_xticks(range(len(masses)))
        ax.set_xticklabels([_mass_label(mass) for mass in masses], rotation=30, ha="right")
        ax.set_xlabel("raw target mass")
    axes[0].set_yticks(range(len(ks)))
    axes[0].set_yticklabels([str(k) for k in ks])
    axes[0].set_ylabel("initial blob count K")
    fig.suptitle("TRUE rate by K, mass, and IC seed", fontsize=13)
    fig.colorbar(image, ax=axes, fraction=0.024, pad=0.04, label="TRUE fraction")
    return save_figure(fig, outpath)


def _plot_class_composition(rows: list[dict[str, Any]], outpath: Path) -> Path:
    ks, masses, _ = _grid_axes(rows)
    fig, axes = plt.subplots(len(ks), 1, figsize=(11, 2.7 * len(ks)), sharex=True)
    if len(ks) == 1:
        axes = [axes]
    for ax, kval in zip(axes, ks):
        bottoms = np.zeros(len(masses), dtype=float)
        for klass in CLASS_ORDER:
            values = np.asarray(
                [
                    sum(
                        1
                        for row in rows
                        if row["K_int"] == kval and abs(row["mass_float"] - mass) < 1e-9 and row.get("klass") == klass
                    )
                    for mass in masses
                ],
                dtype=float,
            )
            ax.bar(
                range(len(masses)),
                values,
                bottom=bottoms,
                label=CLASS_SHORT.get(klass, klass),
                color=CLASS_COLORS.get(klass, "#777777"),
            )
            bottoms += values
        ax.set_ylabel(f"K={kval}")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, max(8.5, float(np.max(bottoms)) + 1))
    axes[0].legend(ncol=min(5, len(CLASS_ORDER)), fontsize=8, loc="upper right")
    axes[-1].set_xticks(range(len(masses)))
    axes[-1].set_xticklabels([_mass_label(mass) for mass in masses], rotation=30, ha="right")
    axes[-1].set_xlabel("raw target mass")
    fig.suptitle("Class composition by K and raw target mass", fontsize=13)
    return save_figure(fig, outpath)


def _plot_true_nodecount_distribution(rows: list[dict[str, Any]], outpath: Path) -> Path:
    ks, masses, _ = _grid_axes(rows)
    node_counts = sorted({_safe_int(row["n_fin_int"]) for row in rows if row.get("klass") == TRUE_CLASS and _safe_int(row["n_fin_int"]) >= 0})
    if not node_counts:
        return blank_figure("No TRUE rows found in the structured summary.", outpath, title="TRUE node-count distribution")
    fig, axes = plt.subplots(len(ks), 1, figsize=(11, 2.7 * len(ks)), sharex=True)
    if len(ks) == 1:
        axes = [axes]
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(node_counts))))
    for ax, kval in zip(axes, ks):
        bottoms = np.zeros(len(masses), dtype=float)
        for color, node_count in zip(colors, node_counts):
            values = np.asarray(
                [
                    sum(
                        1
                        for row in rows
                        if row["K_int"] == kval
                        and abs(row["mass_float"] - mass) < 1e-9
                        and row.get("klass") == TRUE_CLASS
                        and row["n_fin_int"] == node_count
                    )
                    for mass in masses
                ],
                dtype=float,
            )
            ax.bar(range(len(masses)), values, bottom=bottoms, color=color, label=f"n_fin={node_count}")
            bottoms += values
        ax.set_ylabel(f"K={kval}")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(ncol=min(6, len(node_counts)), fontsize=8, loc="upper right")
    axes[-1].set_xticks(range(len(masses)))
    axes[-1].set_xticklabels([_mass_label(mass) for mass in masses], rotation=30, ha="right")
    axes[-1].set_xlabel("raw target mass")
    fig.suptitle("TRUE final node-count distribution by K and raw target mass", fontsize=13)
    return save_figure(fig, outpath)


def _dominant_class(bucket: list[dict[str, Any]]) -> str:
    counts = Counter(str(row.get("klass")) for row in bucket)
    return max(counts, key=lambda key: (counts[key], key))


def _true_node_text(bucket: list[dict[str, Any]]) -> str:
    true_rows = [row for row in bucket if row.get("klass") == TRUE_CLASS and row["n_fin_int"] >= 0]
    if not true_rows:
        return "-"
    counts = Counter(int(row["n_fin_int"]) for row in true_rows)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    top = "/".join(str(node) for node, _ in ordered[:2])
    if len(ordered) > 2:
        top += "+"
    return top


def _plot_branch_landscape_summary(rows: list[dict[str, Any]], outpath: Path) -> Path:
    ks, masses, _ = _grid_axes(rows)
    fig, ax = plt.subplots(figsize=(11, 5.6))
    reject_handles: dict[str, Any] = {}
    scatter = None
    max_node = max(2, max((row["n_fin_int"] for row in rows if row.get("klass") == TRUE_CLASS and row["n_fin_int"] >= 0), default=6))
    for kval in ks:
        for mass in masses:
            bucket = [row for row in rows if row["K_int"] == kval and abs(row["mass_float"] - mass) < 1e-9]
            if not bucket:
                continue
            true_rows = [row for row in bucket if row.get("klass") == TRUE_CLASS and row["n_fin_int"] >= 0]
            near_rows = [row for row in bucket if row.get("klass") == "NEAR_SATURATED_BOUND_STATE"]
            x = mass
            y = kval
            if true_rows:
                median_nodes = float(np.median([row["n_fin_int"] for row in true_rows]))
                size = 120 + 40 * len(true_rows)
                scatter = ax.scatter(
                    [x],
                    [y],
                    s=size,
                    c=[median_nodes],
                    cmap="viridis",
                    vmin=1,
                    vmax=max_node,
                    edgecolors="black",
                    linewidths=0.8,
                    zorder=3,
                )
                ax.text(x, y, f"T{len(true_rows)}\n{_true_node_text(bucket)}", ha="center", va="center", fontsize=8, color="white", zorder=4)
            elif near_rows:
                size = 110 + 35 * len(near_rows)
                ax.scatter([x], [y], s=size, facecolors="none", edgecolors=CLASS_COLORS["NEAR_SATURATED_BOUND_STATE"], linewidths=2.0, zorder=2)
                ax.text(x, y, f"N{len(near_rows)}", ha="center", va="center", fontsize=8, color=CLASS_COLORS["NEAR_SATURATED_BOUND_STATE"], zorder=4)
            else:
                dominant = _dominant_class(bucket)
                short = CLASS_SHORT.get(dominant, dominant)
                marker = "x"
                color = DOMINANT_CLASS_COLORS.get(dominant, "#666666")
                handle = reject_handles.get(short)
                point = ax.scatter([x], [y], s=120, marker=marker, color=color, linewidths=2.0, zorder=1, label=short if handle is None else None)
                if handle is None:
                    reject_handles[short] = point
                ax.text(x, y - 0.16, short, ha="center", va="top", fontsize=7, color=color)
    ax.set_xticks(masses)
    ax.set_xticklabels([_mass_label(mass) for mass in masses], rotation=30, ha="right")
    ax.set_yticks(ks)
    ax.set_yticklabels([str(k) for k in ks])
    ax.set_xlabel("raw target mass")
    ax.set_ylabel("initial blob count K")
    ax.set_title("Branch landscape summary: TRUE branch density, TRUE node family, and reject-side gaps")
    ax.grid(alpha=0.22)
    if scatter is not None:
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="median TRUE final node count")
    legend_items = [
        plt.Line2D([0], [0], marker="o", color="black", markerfacecolor="white", markersize=9, linestyle="none", label="TRUE branch"),
        plt.Line2D([0], [0], marker="o", color=CLASS_COLORS["NEAR_SATURATED_BOUND_STATE"], markerfacecolor="none", markersize=9, linestyle="none", label="NEAR only"),
    ]
    legend_items.extend(reject_handles.values())
    ax.legend(handles=legend_items, fontsize=8, loc="upper left")
    return save_figure(fig, outpath)


def _render_shortlist_overview(rows: list[dict[str, Any]], outpath: Path) -> Path:
    fig, ax = plt.subplots(figsize=(13.5, 1.35 * len(rows) + 1.6))
    ax.set_axis_off()
    y = 0.98
    ax.text(0.01, y, "Phase C structured-discovery shortlist overview", fontsize=13, weight="bold", va="top")
    y -= 0.065
    for row in rows:
        summary_line = (
            f"{row['candidate_label']} | run={row['source_run']} idx={row['idx']} | "
            f"K={row['K']} mass={_mass_label(float(row['raw_target_mass'])) if row['raw_target_mass'] not in ('', None) else '-'} "
            f"seed={row['ic_seed']} | class={row['class']} | diag={row['diagnostic_label']} | n_fin={row['final_node_count']}"
        )
        metric_line = (
            f"compactness={_safe_float(row['compactness']):.4g} | "
            f"core_radius={_safe_float(row['core_radius']):.4g} | "
            f"high_k={_safe_float(row['high_k_fraction']):.4g} | role={row['role']}"
        )
        reason_line = f"selection reason: {row['selection_reason']}"
        ax.text(0.01, y, summary_line, fontsize=9.2, va="top", family="monospace")
        y -= 0.037
        ax.text(0.03, y, metric_line, fontsize=8.6, va="top", family="monospace")
        y -= 0.033
        ax.text(0.03, y, reason_line, fontsize=8.6, va="top")
        y -= 0.05
    return save_figure(fig, outpath, dpi=150)


def _render_shortlist_diagnostic_summary(rows: list[dict[str, Any]], outpath: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12.5, 5.3))
    yvals = np.arange(len(rows))
    class_colors = [CLASS_COLORS.get(str(row["class"]), "#666666") for row in rows]
    node_counts = [float(row["final_node_count"]) if str(row["final_node_count"]).strip() not in {"", "None"} else 0.0 for row in rows]
    ax.barh(yvals, np.maximum(node_counts, 0.15), color=class_colors, alpha=0.88)
    ax.set_yticks(yvals)
    ax.set_yticklabels([str(row["candidate_label"]) for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel("final node count (bar length)")
    ax.set_title("Shortlist diagnostic summary")
    for yval, row in zip(yvals, rows):
        text = f"class={row['class']} | diag={row['diagnostic_label']} | role={row['role']}"
        ax.text(max(node_counts[yval], 0.15) + 0.08, yval, text, va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    return save_figure(fig, outpath)


def _load_case_result(case_dir: Path) -> tuple[dict[str, Any], np.ndarray]:
    summary_path = case_dir / "diagnostic_summary.json"
    npz_path = case_dir / "frames.npz"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}")
    payload = load_json(summary_path)
    bundle = load_npz(npz_path)
    key = choose_npz_key(bundle, preferred=("psi", "frames"))
    frames = np.asarray(bundle[key])
    if frames.ndim == 3:
        frames = frames[None, ...]
    if frames.ndim != 4:
        raise ValueError(f"Expected frames array with ndim 4, got shape {frames.shape} for {npz_path}")
    return payload, frames


def _trace_arrays(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    trace = payload["trace"]
    times = np.asarray(payload["times"], dtype=float)
    initial_mass = float(payload["replay"]["initial_mass"])
    mass = np.asarray([_safe_float(row.get("mass")) for row in trace], dtype=float)
    return {
        "times": times,
        "energy_ratio": mass / max(initial_mass, 1e-30),
        "rho_peak": np.asarray([_safe_float(row.get("rho_peak")) for row in trace], dtype=float),
        "compactness": np.asarray([_safe_float(row.get("compactness")) for row in trace], dtype=float),
        "core_radius": np.asarray([_safe_float(row.get("core_radius")) for row in trace], dtype=float),
        "omega2_min": np.asarray([_safe_float(row.get("omega2_min")) for row in trace], dtype=float),
        "grad_log_omega": np.asarray([_safe_float(row.get("grad_log_omega_max")) for row in trace], dtype=float),
        "high_k_fraction": np.asarray([_safe_float(row.get("high_k_fraction")) for row in trace], dtype=float),
        "node_count": np.asarray([_safe_float(row.get("node_count")) for row in trace], dtype=float),
    }


def _render_case_timeline(payload: dict[str, Any], outpath: Path, title: str) -> Path:
    arrays = _trace_arrays(payload)
    summary = payload["summary"]
    fig, axes = plt.subplots(4, 2, figsize=(12.5, 11.5), sharex=True)
    panels = [
        ("energy_ratio", "Energy / Mass Proxy", "mass / initial_mass"),
        ("rho_peak", "Peak Density", "max |psi|^2"),
        ("compactness", "Compactness Proxy", "mass_inside_r / r"),
        ("core_radius", "Core Radius", "half-mass radius"),
        ("omega2_min", "Omega^2 Minimum", "min omega^2"),
        ("grad_log_omega", "Geometry-Gradient Proxy", "max |grad log omega|"),
        ("high_k_fraction", "High-k Fraction", "spectral tail"),
        ("node_count", "Node Count", "count"),
    ]
    flat_axes = list(axes.flat)
    for ax, (key, panel_title, ylabel) in zip(flat_axes, panels):
        ax.plot(arrays["times"], arrays[key], lw=1.6)
        ax.set_title(panel_title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    flat_axes[-1].set_xlabel("time")
    flat_axes[-2].set_xlabel("time")
    fig.suptitle(
        f"{title} | class={payload['candidate']['klass']} | diag={summary['diagnostic_label']} | "
        f"n_fin={summary['node_count_last']}",
        fontsize=11,
    )
    return save_figure(fig, outpath, dpi=150)


def _render_case_summary(payload: dict[str, Any], outpath: Path, *, label: str, role: str) -> Path:
    candidate = payload["candidate"]
    replay = payload["replay"]
    summary = payload["summary"]
    lines = [
        f"label: {label}",
        f"role: {role}",
        f"class: {candidate.get('klass')}",
        f"diagnostic: {summary.get('diagnostic_label')}",
        f"K: {candidate.get('K')}",
        f"ic_seed: {replay.get('ic_seed')}",
        f"ic_norm: {replay.get('ic_norm')}",
        f"raw target mass: {replay.get('target_initial_mass')}",
        f"final node count: {summary.get('node_count_last')}",
        f"late slope: {summary.get('late_energy_slope'):.6g}" if np.isfinite(_safe_float(summary.get("late_energy_slope"))) else "late slope: n/a",
        f"er_final_or_last: {summary.get('er_final_or_last'):.6g}" if np.isfinite(_safe_float(summary.get("er_final_or_last"))) else "er_final_or_last: n/a",
        f"compactness max: {summary.get('compactness_max'):.6g}" if np.isfinite(_safe_float(summary.get("compactness_max"))) else "compactness max: n/a",
        f"core radius min: {summary.get('core_radius_min'):.6g}" if np.isfinite(_safe_float(summary.get("core_radius_min"))) else "core radius min: n/a",
        f"high-k fraction max: {summary.get('high_k_fraction_max'):.6g}" if np.isfinite(_safe_float(summary.get("high_k_fraction_max"))) else "high-k fraction max: n/a",
        f"time to failure: {summary.get('time_to_failure')}" if summary.get("time_to_failure") is not None else "time to failure: none",
    ]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.set_axis_off()
    ax.text(0.03, 0.97, "\n".join(lines), va="top", family="monospace", fontsize=10)
    ax.set_title("Case summary")
    return save_figure(fig, outpath)


def _render_case_pack(case_dir: Path, *, label: str, role: str, overwrite: bool) -> list[Path]:
    guard_outputs(case_dir, CASE_OUTPUTS, overwrite)
    payload, frames = _load_case_result(case_dir)
    outputs = [
        _render_case_timeline(payload, case_dir / "timeline_panel.png", title=label),
        render_density_slices(frames, case_dir, filename="density_slices.png"),
        render_vector_preview(frames, case_dir, filename="vector_map.png"),
        _render_case_summary(payload, case_dir / "case_summary.png", label=label, role=role),
    ]
    return outputs


def render(
    summary_csv: str | None,
    *,
    outdir: str | None,
    overwrite: bool,
    shortlist: str | None = None,
    cases_root: str | None = None,
) -> list[Path]:
    summary_path = _resolve_path(summary_csv, DEFAULT_SUMMARY_CSV)
    shortlist_path = _resolve_path(shortlist, DEFAULT_SHORTLIST_JSON)
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing structured-discovery summary CSV: {summary_path}")
    if not shortlist_path.exists():
        raise FileNotFoundError(f"Missing shortlist metrics JSON: {shortlist_path}")
    if not outdir:
        raise ValueError("outdir is required")
    outdir_path = _resolve_path(outdir, Path(outdir))
    population_dir = outdir_path / "population"
    shortlist_dir = outdir_path / "shortlist"
    case_root = _resolve_path(cases_root, outdir_path / "cases")
    population_dir.mkdir(parents=True, exist_ok=True)
    shortlist_dir.mkdir(parents=True, exist_ok=True)
    case_root.mkdir(parents=True, exist_ok=True)

    guard_outputs(population_dir, POPULATION_OUTPUTS, overwrite)
    guard_outputs(shortlist_dir, SHORTLIST_OUTPUTS, overwrite)

    rows = _structured_rows(summary_path)
    shortlist_rows = _shortlist_rows(shortlist_path)
    summary_lookup = _summary_lookup(rows)
    shortlist_table = _shortlist_table_rows(shortlist_rows, summary_lookup)

    outputs = [
        _plot_true_rate_heatmap(rows, population_dir / "phase_c_true_rate_heatmap.png"),
        _plot_true_rate_by_seed(rows, population_dir / "phase_c_true_rate_by_seed.png"),
        _plot_class_composition(rows, population_dir / "phase_c_class_composition.png"),
        _plot_true_nodecount_distribution(rows, population_dir / "phase_c_true_nodecount_distribution.png"),
        _plot_branch_landscape_summary(rows, population_dir / "phase_c_branch_landscape_summary.png"),
    ]

    fieldnames = [
        "candidate_label",
        "case_key",
        "source_run",
        "idx",
        "K",
        "raw_target_mass",
        "dx_weighted_target_mass",
        "ic_seed",
        "class",
        "diagnostic_label",
        "final_node_count",
        "late_slope",
        "compactness",
        "core_radius",
        "high_k_fraction",
        "selection_reason",
        "role",
    ]
    shortlist_csv = shortlist_dir / "phase_c_shortlist_table.csv"
    write_csv(shortlist_csv, shortlist_table, fieldnames)
    outputs.append(shortlist_csv)
    outputs.extend(
        [
            _render_shortlist_overview(shortlist_table, shortlist_dir / "phase_c_shortlist_overview.png"),
            _render_shortlist_diagnostic_summary(shortlist_table, shortlist_dir / "phase_c_shortlist_diagnostic_summary.png"),
        ]
    )

    for case_name in CASE_ORDER:
        case_dir = case_root / case_name
        if not (case_dir / "diagnostic_summary.json").exists():
            continue
        spec = CASE_SPECS[case_name]
        outputs.extend(_render_case_pack(case_dir, label=spec["label"], role=spec["role"], overwrite=overwrite))
    return outputs
