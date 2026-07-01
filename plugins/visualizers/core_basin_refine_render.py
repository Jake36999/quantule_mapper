"""Deprecated compatibility wrapper for the centralized quantule_viz core-basin renderer."""

from __future__ import annotations

from pathlib import Path
import sys

from quantule_viz.cli import main as quantule_viz_main


def main() -> int:
    print("Deprecated: use `python -m quantule_viz core-basin <run_dir>`.")
    root = Path(__file__).resolve().parents[2] / "sweep_runs"
    matches = sorted(path for path in root.glob("CORE_BASIN_REFINE_*") if path.is_dir())
    if not matches:
        raise SystemExit("No CORE_BASIN_REFINE_* run directory found.")
    argv = ["core-basin", str(matches[-1]), *sys.argv[1:]]
    return int(quantule_viz_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
