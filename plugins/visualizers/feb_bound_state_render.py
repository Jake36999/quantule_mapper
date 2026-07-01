"""Deprecated compatibility wrapper for the centralized quantule_viz feb-bound-state renderer."""

from __future__ import annotations

import sys

from quantule_viz.cli import main as quantule_viz_main


def main() -> int:
    print("Deprecated: use `python -m quantule_viz feb-bound-state <run_dir>`.")
    argv = ["feb-bound-state", "--latest", *sys.argv[1:]]
    return int(quantule_viz_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
