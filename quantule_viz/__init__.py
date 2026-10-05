"""Centralized read-only visualization/report tools for saved Quantule Mapper artifacts."""

__all__ = ["__version__", "main"]
__version__ = "0.3.0"


def main(argv: list[str] | None = None) -> int:
    from .cli import main as cli_main

    return cli_main(argv)
