"""Phase C saved-result renderer."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ..io import (
    IncompleteRunError,
    guard_outputs,
    load_csv,
    load_json,
    maybe_float,
    maybe_int,
    resolve_run_dir,
    write_csv,
    write_json,
)
from ..plots import CLASS_COLORS, NEAR_CLASS, TRUE_CLASS, blank_figure, class_counts, ordered_class_labels, save_figure
from . import KNOWN_INCOMPLETE_RUNS

PATTERNS = ["CORE_SAT_HUNT_*", "CORE_SAT_PILOT_*"]
OUTPUTS = [
    "summary_panel.png",
    "class_histogram.png",
    "class_counts_by_K.png",
    "node_counts_by_K.png",
    "saturation_slope_scatter.png",
    "best_candidates_table.csv",
    "analysis_summary.json",
]
K_COLUMNS = ("ic_blobs", "initial_blob_count", "initial_blobs", "k_init", "K", "k")


def _candidate_sort_key(row: dict[str, str]) -> tuple[object, ...]:
    klass = row.get("klass", "")
    rank = 0 if klass == TRUE_CLASS else 1 if klass == NEAR_CLASS else 2
    slope = abs(maybe_float(row.get("late_slope"), default=float("inf")))
    core = maybe_float(row.get("core_fin"), default=float("-inf"))
    er = maybe_float(row.get("er_fin"), default=float("-inf"))
    idx = maybe_int(row.get("idx"), default=10**9)
    return (rank, slope, -core, -er, idx)


def _detect_k_column(rows: list[dict[str, str]]) -> str | None:
    for key in K_COLUMNS:
        if any(str(row.get(key, "")).strip() for row in rows):
            return key
    return None


def _load_run(run_dir: Path) -> tuple[list[dict[str, str]], dict[str, object]]:
    name = run_dir.name
    if name in KNOWN_INCOMPLETE_RUNS:
        raise IncompleteRunError(KNOWN_INCOMPLETE_RUNS[name])
    csv_path = run_dir / "all_evals.csv"
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        raise IncompleteRunError(f"{run_dir} is incomplete: missing or empty all_evals.csv")
    rows = load_csv(csv_path)
    if not rows:
        raise IncompleteRunError(f"{run_dir} is incomplete: all_evals.csv has no rows")
    summary_path = run_dir / "summary.json"
    summary = load_json(summary_path) if summary_path.exists() and summary_path.stat().st_size > 0 else {}
    return rows, summary


def _plot_class_histogram(rows: list[dict[str, str]], outdir: Path) -> Path:
    counts = class_counts(rows)
    labels = ordered_class_labels(counts)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        range(len(labels)),
        [counts[label] for label in labels],
        color=[CLASS_COLORS.get(label, "#666666") for label in labels],
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("configs")
    ax.set_title("Phase C class counts")
    ax.grid(axis="y", alpha=0.25)
    return save_figure(fig, outdir / "class_histogram.png")


def _plot_class_counts_by_k(rows: list[dict[str, str]], outdir: Path, k_column: str | None) -> Path:
    if not k_column:
        return blank_figure(
            "K-varied initial-condition data is not recorded in this run.",
            outdir / "class_counts_by_K.png",
            title="Class counts by initial K",
        )
    by_k: dict[int, Counter[str]] = defaultdict(Counter)
    for row in rows:
        kval = maybe_int(row.get(k_column), default=-1)
        if kval >= 0:
            by_k[kval][row.get("klass", "UNKNOWN")] += 1
    if not by_k:
        return blank_figure(
            "No usable K column values were found in saved rows.",
            outdir / "class_counts_by_K.png",
            title="Class counts by initial K",
        )
    labels = sorted({klass for counts in by_k.values() for klass in counts})
    ks = sorted(by_k)
    bottoms = np.zeros(len(ks), dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for klass in labels:
        vals = np.asarray([by_k[k][klass] for k in ks], dtype=float)
        ax.bar([str(k) for k in ks], vals, bottom=bottoms, color=CLASS_COLORS.get(klass, "#666666"), label=klass)
        bottoms += vals
    ax.set_xlabel("initial blob count K")
    ax.set_ylabel("configs")
    ax.set_title("Stored class counts by initial K")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    return save_figure(fig, outdir / "class_counts_by_K.png")


def _plot_node_counts_by_k(rows: list[dict[str, str]], outdir: Path, k_column: str | None) -> Path:
    if not k_column:
        return blank_figure(
            "K-varied initial-condition data is not recorded in this run.",
            outdir / "node_counts_by_K.png",
            title="Final node counts by initial K",
        )
    by_k: dict[int, Counter[int]] = defaultdict(Counter)
    for row in rows:
        if row.get("klass") != TRUE_CLASS:
            continue
        kval = maybe_int(row.get(k_column), default=-1)
        n_fin = maybe_int(row.get("n_fin"), default=-1)
        if kval >= 0 and n_fin >= 0:
            by_k[kval][n_fin] += 1
    if not by_k:
        return blank_figure(
            "No TRUE_SATURATED rows with saved K values were found.",
            outdir / "node_counts_by_K.png",
            title="Final node counts by initial K",
        )
    ks = sorted(by_k)
    node_counts = sorted({count for dist in by_k.values() for count in dist})
    bottoms = np.zeros(len(ks), dtype=float)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    palette = plt.cm.tab20(np.linspace(0, 1, max(1, len(node_counts))))
    for color, node_count in zip(palette, node_counts):
        vals = np.asarray([by_k[k][node_count] for k in ks], dtype=float)
        ax.bar([str(k) for k in ks], vals, bottom=bottoms, color=color, label=f"n_fin={node_count}")
        bottoms += vals
    ax.set_xlabel("initial blob count K")
    ax.set_ylabel("TRUE_SATURATED configs")
    ax.set_title("Stored final node-count distribution by initial K")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    return save_figure(fig, outdir / "node_counts_by_K.png")


def _plot_slope_scatter(rows: list[dict[str, str]], outdir: Path) -> Path:
    labels = sorted({row.get("klass", "UNKNOWN") for row in rows})
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for klass in labels:
        subset = [row for row in rows if row.get("klass", "UNKNOWN") == klass]
        xvals = [maybe_float(row.get("er_fin")) for row in subset]
        yvals = [abs(maybe_float(row.get("late_slope"))) for row in subset]
        ax.scatter(xvals, yvals, s=28, alpha=0.75, label=klass, color=CLASS_COLORS.get(klass, "#666666"))
    ax.set_yscale("log")
    ax.set_xlabel("final energy ratio er_fin")
    ax.set_ylabel("|late_slope|")
    ax.set_title("Stored saturation slope scatter")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    return save_figure(fig, outdir / "saturation_slope_scatter.png")


def _plot_summary_panel(
    rows: list[dict[str, str]],
    summary: dict[str, object],
    analysis: dict[str, object],
    outdir: Path,
) -> Path:
    counts = class_counts(rows)
    labels = ordered_class_labels(counts)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].bar(
        range(len(labels)),
        [counts[label] for label in labels],
        color=[CLASS_COLORS.get(label, "#666666") for label in labels],
    )
    axes[0, 0].set_xticks(range(len(labels)))
    axes[0, 0].set_xticklabels(labels, rotation=35, ha="right")
    axes[0, 0].set_title("Class counts")
    axes[0, 0].grid(axis="y", alpha=0.25)

    true_rows = [row for row in rows if row.get("klass") == TRUE_CLASS]
    axes[0, 1].hist(
        [maybe_int(row.get("n_fin")) for row in true_rows if maybe_int(row.get("n_fin")) >= 0],
        bins=np.arange(-0.5, 10.5, 1),
        color=CLASS_COLORS[TRUE_CLASS],
    )
    axes[0, 1].set_xlabel("n_fin")
    axes[0, 1].set_ylabel("TRUE count")
    axes[0, 1].set_title("TRUE final node counts")
    axes[0, 1].grid(axis="y", alpha=0.25)

    for klass in labels:
        vals = [abs(maybe_float(row.get("late_slope"))) for row in rows if row.get("klass") == klass]
        vals = [value for value in vals if np.isfinite(value) and value > 0]
        if vals:
            axes[1, 0].scatter([klass] * len(vals), vals, s=14, alpha=0.55, color=CLASS_COLORS.get(klass, "#666666"))
    axes[1, 0].set_yscale("log")
    axes[1, 0].tick_params(axis="x", labelrotation=35)
    axes[1, 0].set_title("|late_slope| by stored class")
    axes[1, 0].grid(axis="y", alpha=0.25)

    metadata = [
        f"run: {analysis['run_name']}",
        f"status: {analysis['run_status']}",
        f"N: {summary.get('N', 'unknown')}",
        f"T: {summary.get('T', 'unknown')}",
        f"n_eval: {analysis['n_eval']}",
        f"elapsed_h: {summary.get('elapsed_h', 'unknown')}",
        f"k_column: {analysis.get('k_column') or 'not recorded'}",
        f"TRUE: {counts.get(TRUE_CLASS, 0)}",
        f"NEAR: {counts.get(NEAR_CLASS, 0)}",
    ]
    axes[1, 1].text(0.03, 0.97, "\n".join(metadata), va="top", family="monospace", fontsize=11)
    axes[1, 1].set_axis_off()
    axes[1, 1].set_title("Saved metadata")

    fig.suptitle("Phase C saved-result summary", fontsize=14)
    return save_figure(fig, outdir / "summary_panel.png")


def _best_candidates_table(
    rows: list[dict[str, str]],
    outdir: Path,
    *,
    limit: int = 40,
    candidate: str | None = None,
) -> tuple[Path, list[dict[str, object]]]:
    keep = [row for row in rows if row.get("klass") in {TRUE_CLASS, NEAR_CLASS}]
    keep.sort(key=_candidate_sort_key)
    if candidate:
        selected = [
            row
            for row in keep
            if str(row.get("idx", "")) == candidate or str(row.get("label", "")) == candidate
        ]
        if selected:
            keep = selected + [row for row in keep if row not in selected]
    keep = keep[:limit]
    fieldnames = [
        "rank",
        "idx",
        "label",
        "klass",
        "ic_blobs",
        "er_fin",
        "er_max",
        "late_slope",
        "n_mid",
        "n_fin",
        "core_fin",
        "param_D",
        "param_eta",
        "param_rho_vac",
        "param_omega0",
        "param_a_coupling",
        "param_s",
        "param_f",
        "param_a",
    ]
    table_rows: list[dict[str, object]] = []
    for rank, row in enumerate(keep, start=1):
        enriched = {"rank": rank}
        for field in fieldnames[1:]:
            enriched[field] = row.get(field, "")
        table_rows.append(enriched)
    path = outdir / "best_candidates_table.csv"
    write_csv(path, table_rows, fieldnames)
    return path, table_rows


def _build_analysis(
    run_dir: Path,
    rows: list[dict[str, str]],
    summary: dict[str, object],
    *,
    k_column: str | None,
    best_rows: list[dict[str, object]],
    candidate: str | None,
) -> dict[str, object]:
    counts = class_counts(rows)
    analysis: dict[str, object] = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "run_status": "RUN_COMPLETE",
        "n_eval": len(rows),
        "counts": dict(counts),
        "k_column": k_column,
        "k_varied_data_present": bool(k_column),
        "summary_json_present": bool(summary),
        "sat_node_counts": dict(
            Counter(
                str(maybe_int(row.get("n_fin")))
                for row in rows
                if row.get("klass") == TRUE_CLASS and maybe_int(row.get("n_fin")) >= 0
            )
        ),
        "best_true_candidates": [row for row in best_rows if row.get("klass") == TRUE_CLASS][:5],
        "best_near_candidates": [row for row in best_rows if row.get("klass") == NEAR_CLASS][:5],
    }
    if candidate:
        selected = [
            row
            for row in best_rows
            if str(row.get("idx", "")) == candidate or str(row.get("label", "")) == candidate
        ]
        analysis["selected_candidate"] = selected[0] if selected else None

    if not k_column:
        analysis["note"] = "This run does not store initial blob-count K values in all_evals.csv."
        return analysis

    by_k_counts: dict[int, Counter[str]] = defaultdict(Counter)
    by_k_nodes: dict[int, Counter[int]] = defaultdict(Counter)
    best_true_by_k: dict[int, dict[str, str]] = {}
    best_near_by_k: dict[int, dict[str, str]] = {}
    for row in rows:
        kval = maybe_int(row.get(k_column), default=-1)
        if kval < 0:
            continue
        by_k_counts[kval][row.get("klass", "UNKNOWN")] += 1
        if row.get("klass") == TRUE_CLASS:
            n_fin = maybe_int(row.get("n_fin"), default=-1)
            if n_fin >= 0:
                by_k_nodes[kval][n_fin] += 1
            current = best_true_by_k.get(kval)
            if current is None or _candidate_sort_key(row) < _candidate_sort_key(current):
                best_true_by_k[kval] = row
        elif row.get("klass") == NEAR_CLASS:
            current = best_near_by_k.get(kval)
            if current is None or _candidate_sort_key(row) < _candidate_sort_key(current):
                best_near_by_k[kval] = row

    analysis["class_counts_by_K"] = {str(k): dict(counts) for k, counts in sorted(by_k_counts.items())}
    analysis["true_count_by_K"] = {str(k): counts.get(TRUE_CLASS, 0) for k, counts in sorted(by_k_counts.items())}
    analysis["near_count_by_K"] = {str(k): counts.get(NEAR_CLASS, 0) for k, counts in sorted(by_k_counts.items())}
    analysis["failure_modes_by_K"] = {
        str(k): {klass: count for klass, count in counts.items() if klass not in {TRUE_CLASS, NEAR_CLASS}}
        for k, counts in sorted(by_k_counts.items())
    }
    analysis["final_node_count_distribution_by_K"] = {
        str(k): {str(node_count): count for node_count, count in sorted(nodes.items())}
        for k, nodes in sorted(by_k_nodes.items())
    }
    analysis["best_true_candidates_per_K"] = {str(k): best_true_by_k[k] for k in sorted(best_true_by_k)}
    analysis["best_near_candidates_per_K"] = {str(k): best_near_by_k[k] for k in sorted(best_near_by_k)}
    return analysis


def render(
    run_dir: str | None,
    *,
    outdir: str | None,
    overwrite: bool,
    latest: bool,
    candidate: str | None,
) -> list[Path]:
    resolved = resolve_run_dir(run_dir, latest=latest, patterns=PATTERNS)
    output_dir = Path(outdir).resolve() if outdir else resolved
    rows, summary = _load_run(resolved)
    guard_outputs(output_dir, OUTPUTS, overwrite)
    k_column = _detect_k_column(rows)

    best_path, best_rows = _best_candidates_table(rows, output_dir, candidate=candidate)
    analysis = _build_analysis(resolved, rows, summary, k_column=k_column, best_rows=best_rows, candidate=candidate)
    outputs = [
        _plot_summary_panel(rows, summary, analysis, output_dir),
        _plot_class_histogram(rows, output_dir),
        _plot_class_counts_by_k(rows, output_dir, k_column),
        _plot_node_counts_by_k(rows, output_dir, k_column),
        _plot_slope_scatter(rows, output_dir),
        best_path,
    ]
    analysis_path = output_dir / "analysis_summary.json"
    write_json(analysis_path, analysis)
    outputs.append(analysis_path)
    return outputs
