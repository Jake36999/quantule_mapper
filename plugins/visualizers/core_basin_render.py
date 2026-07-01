"""Deprecated compatibility wrapper for the centralized quantule_viz core-basin renderer."""

from __future__ import annotations

from pathlib import Path
import sys

from quantule_viz.cli import main as quantule_viz_main


def main() -> int:
    print("Deprecated: use `python -m quantule_viz core-basin <run_dir>`.")
    root = Path(__file__).resolve().parents[2] / "sweep_runs"
    matches = sorted(
        path
        for path in root.glob("CORE_BASIN_*")
        if path.is_dir() and "REFINE" not in path.name and "CALIB" not in path.name
    )
    if not matches:
        raise SystemExit("No CORE_BASIN_* run directory found.")
    argv = ["core-basin", str(matches[-1]), *sys.argv[1:]]
    return int(quantule_viz_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
