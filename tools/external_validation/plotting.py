from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .io_utils import repo_path


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def line_plot(path: str | Path, series: Sequence[tuple[Sequence[float], Sequence[float], str]], title: str, xlabel: str, ylabel: str) -> bool:
    try:
        plt = _plt()
    except Exception:
        return False
    p = repo_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
    for x, y, label in series:
        ax.plot(np.asarray(x, dtype=float), np.asarray(y, dtype=float), marker="o", label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(series) > 1:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return True


def scatter_line_plot(path: str | Path, x: Sequence[float], y: Sequence[float], fit_y: Sequence[float], title: str, xlabel: str, ylabel: str) -> bool:
    try:
        plt = _plt()
    except Exception:
        return False
    p = repo_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
    ax.scatter(x, y, label="measured")
    ax.plot(x, fit_y, label="linear fit", color="tab:orange")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return True


def heatmap_plot(path: str | Path, grid: np.ndarray, xlabels: list[str], ylabels: list[str], title: str) -> bool:
    try:
        plt = _plt()
    except Exception:
        return False
    p = repo_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
    im = ax.imshow(grid, aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=45, ha="right")
    ax.set_yticks(range(len(ylabels)), ylabels)
    fig.colorbar(im, ax=ax, label="outcome code")
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return True

