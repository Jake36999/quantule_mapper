"""Read-only loaders and filesystem helpers for Quantule Mapper artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


class IncompleteRunError(RuntimeError):
    """Raised when a run directory does not contain the required saved artifacts."""


def load_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_npz(path: str | Path) -> np.lib.npyio.NpzFile:
    return np.load(Path(path), allow_pickle=True)


def maybe_float(value: object, default: float = float("nan")) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def maybe_int(value: object, default: int = -1) -> int:
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def latest_run(
    root: str | Path,
    patterns: Sequence[str],
    *,
    exclude_parts: Sequence[str] = (),
) -> Path:
    root_path = Path(root)
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(path for path in root_path.glob(pattern) if path.is_dir())
    filtered = [
        path
        for path in matches
        if not any(part and part in path.as_posix() for part in exclude_parts)
    ]
    ordered = sorted(filtered, key=lambda path: path.stat().st_mtime)
    if not ordered:
        joined = ", ".join(patterns)
        raise FileNotFoundError(f"No run directories matched {root_path} with patterns: {joined}")
    return ordered[-1]


def resolve_run_dir(
    run_dir: str | None,
    *,
    latest: bool,
    patterns: Sequence[str],
    exclude_parts: Sequence[str] = (),
    root: str | Path = "sweep_runs",
) -> Path:
    if latest:
        return latest_run(root, patterns, exclude_parts=exclude_parts).resolve()
    if not run_dir:
        raise ValueError("run_dir is required unless --latest is supplied")
    path = Path(run_dir)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def write_csv(path: str | Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, payload: dict[str, object]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def guard_outputs(outdir: str | Path, filenames: Iterable[str], overwrite: bool) -> None:
    if overwrite:
        return
    outdir_path = Path(outdir)
    existing = [str((outdir_path / name).resolve()) for name in filenames if (outdir_path / name).exists()]
    if existing:
        joined = "\n".join(existing)
        raise FileExistsError(
            "Refusing to overwrite existing outputs without --overwrite:\n"
            f"{joined}"
        )


def choose_npz_key(bundle: np.lib.npyio.NpzFile, preferred: Sequence[str] = ()) -> str:
    for key in preferred:
        if key in bundle.files:
            return key
    if not bundle.files:
        raise KeyError("NPZ bundle contains no arrays")
    return str(bundle.files[0])
