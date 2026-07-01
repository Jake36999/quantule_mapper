"""Core basin and refine saved-result renderer."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ..io import IncompleteRunError, guard_outputs, load_csv, load_json, maybe_float, resolve_run_dir, write_json
from ..plots import CLASS_COLORS, blank_figure, save_figure

PATTERNS = ["CORE_BASIN_REFINE_*", "CORE_BASIN_*"]
BASE_OUTPUTS = ["basin_pairwise.png", "basin_hist.png", "basin_summary.json"]
REFINE_OUTPUTS = ["refine_stability_curve.png", "basin_summary.json"]
PARAMS = ["param_eta", "param_a", "param_s", "param_f", "param_D", "param_a_coupling"]


def _label(name: str) -> str:
    return name.replace("param_", "")


def _load_rows(run_dir: Path) -> list[dict[str, str]]:
    csv_path = run_dir / "all_evals.csv"
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        raise IncompleteRunError(f"{run_dir} is incomplete: missing or empty all_evals.csv")
    rows = load_csv(csv_path)
    if not rows:
        raise IncompleteRunError(f"{run_dir} is incomplete: all_evals.csv has no rows")
    return rows


def _is_refine(run_dir: Path) -> bool:
    return "REFINE" in run_dir.name or (run_dir / "refine_curve.json").exists()


def _plot_pairwise(rows: list[dict[str, str]], outdir: Path) -> Path:
    klass = np.asarray([row.get("klass", "UNKNOWN") for row in rows], dtype=object)
    params = {name: np.asarray([maybe_float(row.get(name)) for row in rows], dtype=float) for name in PARAMS}
    pairs = [
        ("param_eta", "param_a"),
        ("param_eta", "param_s"),
        ("param_eta", "param_f"),
        ("param_a", "param_s"),
        ("param_a", "param_f"),
        ("param_s", "param_f"),
        ("param_eta", "param_D"),
        ("param_a", "param_a_coupling"),
        ("param_D", "param_a_coupling"),
    ]
    order = ["COLLAPSE", "BLOWUP", "FRAGMENT", "VIABLE_NO_NODES", "SPIN_DOWN", "SUSTAIN"]
    fig, axes = plt.subplots(3, 3, figsize=(15, 14))
    for axis, (px, py) in zip(axes.ravel(), pairs):
        for klass_name in order:
            mask = klass == klass_name
            if not np.any(mask):
                continue
            axis.scatter(
                params[px][mask],
                params[py][mask],
                s=22 if klass_name == "SUSTAIN" else 7,
                c=CLASS_COLORS.get(klass_name, "k"),
                alpha=0.9 if klass_name == "SUSTAIN" else 0.35,
                edgecolors="k" if klass_name == "SUSTAIN" else "none",
                linewidths=0.3,
                label=klass_name,
            )
        axis.set_xlabel(_label(px))
        axis.set_ylabel(_label(py))
        axis.grid(alpha=0.2)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, fontsize=9)
    fig.suptitle("Core basin parameter map (stored classifications)", fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return save_figure(fig, outdir / "basin_pairwise.png")


def _plot_hist(rows: list[dict[str, str]], outdir: Path) -> Path:
    klass = np.asarray([row.get("klass", "UNKNOWN") for row in rows], dtype=object)
    sustain = klass == "SUSTAIN"
    params = {name: np.asarray([maybe_float(row.get(name)) for row in rows], dtype=float) for name in PARAMS}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, name in zip(axes.ravel(), PARAMS):
        values = params[name][np.isfinite(params[name])]
        axis.hist(values, bins=30, color="0.8", label="all")
        if np.any(sustain):
            axis.hist(params[name][sustain], bins=30, color=CLASS_COLORS["SUSTAIN"], alpha=0.8, label="SUSTAIN")
        axis.set_title(_label(name))
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle("Per-parameter distributions from saved basin sweep", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return save_figure(fig, outdir / "basin_hist.png")


def _plot_refine_curve(run_dir: Path, outdir: Path) -> Path:
    curve_path = run_dir / "refine_curve.json"
    if not curve_path.exists():
        return blank_figure(
            "refine_curve.json is missing for this refine run.",
            outdir / "refine_stability_curve.png",
            title="Refine stability curve",
        )
    curve = load_json(curve_path)
    eta = np.asarray(curve["eta_grid"], dtype=float)
    probabilities = curve["P"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for klass_name in ("SUSTAIN", "SPIN_DOWN", "COLLAPSE", "BLOWUP"):
        axes[0].plot(eta, probabilities[klass_name], "-o", color=CLASS_COLORS.get(klass_name, "#666666"), lw=2, label=klass_name)
    axes[0].set_xlabel("param_eta")
    axes[0].set_ylabel(f"P(class | eta) [{curve['n_per_eta']}/eta]")
    axes[0].set_title("Empirical refine stability curve")
    axes[0].axhline(0.5, color="0.7", ls="--", lw=0.8)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    order = ["BLOWUP", "COLLAPSE", "SPIN_DOWN", "SUSTAIN"]
    axes[1].stackplot(
        eta,
        *[probabilities[klass_name] for klass_name in order],
        colors=[CLASS_COLORS.get(klass_name, "#666666") for klass_name in order],
        labels=order,
        alpha=0.9,
    )
    axes[1].set_xlabel("param_eta")
    axes[1].set_ylabel("fraction")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Outcome composition vs eta")
    axes[1].legend(loc="lower center", ncol=4, fontsize=8)

    fig.suptitle("Core basin refine run", fontsize=13)
    return save_figure(fig, outdir / "refine_stability_curve.png")


def _write_summary(
    output_root: Path,
    run_dir: Path,
    rows: list[dict[str, str]],
    *,
    refine: bool,
) -> Path:
    counts = Counter(row.get("klass", "UNKNOWN") for row in rows)
    summary: dict[str, object] = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "run_kind": "refine" if refine else "basin",
        "n_eval": len(rows),
        "counts": dict(counts),
    }
    if not refine:
        sustain_rows = [row for row in rows if row.get("klass") == "SUSTAIN"]
        percentiles: dict[str, list[float]] = {}
        for name in PARAMS + ["er_fin", "swirl_fin", "vt_ratio", "cd_ratio"]:
            values = np.asarray([maybe_float(row.get(name)) for row in sustain_rows], dtype=float)
            values = values[np.isfinite(values)]
            if len(values):
                percentiles[name] = [float(np.percentile(values, q)) for q in (10, 50, 90)]
        summary["sustain_percentiles_10_50_90"] = percentiles
    path = output_root / "basin_summary.json"
    write_json(path, summary)
    return path


def render(
    run_dir: str | None,
    *,
    outdir: str | None,
    overwrite: bool,
    latest: bool,
) -> list[Path]:
    resolved = resolve_run_dir(run_dir, latest=latest, patterns=PATTERNS, exclude_parts=("CALIB", "/core_characterize"))
    output_dir = Path(outdir).resolve() if outdir else resolved
    rows = _load_rows(resolved)
    refine = _is_refine(resolved)
    guard_outputs(output_dir, REFINE_OUTPUTS if refine else BASE_OUTPUTS, overwrite)

    if refine:
        outputs = [_plot_refine_curve(resolved, output_dir)]
    else:
        outputs = [_plot_pairwise(rows, output_dir), _plot_hist(rows, output_dir)]
    outputs.append(_write_summary(output_dir if outdir else resolved, resolved, rows, refine=refine))
    return outputs
