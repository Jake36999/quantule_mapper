"""
orchestrator/path_utils.py

Pure path builders for the DC-v1.0 output hierarchy (docs/OUTPUT_HIERARCHY.md §1):

    runs/{hunt}/{YYYY-MM-DD}/{solver_contract}/{variant}/gen_{NNNN}/{hash12}_{seed}/
        artifact.h5
        params.json
        provenance.json

No cupy / h5py / MCP dependency — importable and testable anywhere, shared by
the MCP staging tool and (later) the solver/orchestrator write paths.
"""
from __future__ import annotations

import os
import re
from typing import Optional

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize(component: str) -> str:
    """Make a single path component filesystem-safe (no separators / odd chars)."""
    s = str(component).strip().replace(os.sep, "_")
    if os.altsep:
        s = s.replace(os.altsep, "_")
    s = _SAFE.sub("_", s)
    return s or "unknown"


def run_dir_name(config_hash: str, seed: int) -> str:
    """Per-(params x seed) directory name: first 12 of config_hash + seed."""
    return f"{str(config_hash)[:12]}_{int(seed)}"


def build_run_dir(
    root: str,
    hunt_name: str,
    utc_date: str,
    solver_contract: str,
    variant_label: str,
    generation: int,
    config_hash: str,
    seed: int,
) -> str:
    """Directory holding one run's artifact + params + provenance."""
    return os.path.join(
        root,
        _sanitize(hunt_name),
        _sanitize(utc_date),
        _sanitize(solver_contract),
        _sanitize(variant_label),
        f"gen_{int(generation):04d}",
        run_dir_name(config_hash, seed),
    )


def build_artifact_path(
    root: str,
    hunt_name: str,
    utc_date: str,
    solver_contract: str,
    variant_label: str,
    generation: int,
    config_hash: str,
    seed: int,
    filename: str = "artifact.h5",
) -> str:
    """Full path to the HDF5 artifact for a run (OUTPUT_HIERARCHY.md §6)."""
    return os.path.join(
        build_run_dir(root, hunt_name, utc_date, solver_contract, variant_label,
                      generation, config_hash, seed),
        filename,
    )


def build_params_path(run_dir: str) -> str:
    return os.path.join(run_dir, "params.json")


def build_provenance_path(run_dir: str) -> str:
    return os.path.join(run_dir, "provenance.json")


def staging_dir(root: str) -> str:
    """Directory for staged (reviewed-before-run) manifests."""
    return os.path.join(root, "runs", "_staged")


def staged_manifest_path(root: str, config_hash: str, seed: int, run_id: Optional[str] = None) -> str:
    suffix = f"_{str(run_id)[:8]}" if run_id else ""
    return os.path.join(
        staging_dir(root),
        f"staged_{str(config_hash)[:12]}_{int(seed)}{suffix}.json",
    )


def smoke_output_path(root: str, config_hash: str, seed: int) -> str:
    """Smoke runs always live under runs/_smoke/ and never enter the hierarchy."""
    return os.path.join(root, "runs", "_smoke", f"{run_dir_name(config_hash, seed)}.h5")
