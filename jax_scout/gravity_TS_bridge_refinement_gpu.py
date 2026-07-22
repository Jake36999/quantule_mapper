"""Numerical refinement pass for the temporal-spatial bridge audit.

Runs representative bridge arms at higher spatial resolution and timestep:

    N=128, L=40, dt=0.0005

The driver imports the validated bridge audit routines and keeps the same
decomposition semantics: N_t is a clock observable; A_s is the spatial
divergence-form coefficient.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_FOR_IMPORT))

from jax_scout.gravity_D_neutral_probe_gpu import ROOT, build_grid, preflight  # noqa: E402
from jax_scout.gravity_TS_temporal_spatial_bridge_gpu import (  # noqa: E402
    ARMS,
    SOURCE_MODES,
    arm_comparisons,
    common_field_rows,
    environment_record,
    falsification_rows,
    hash_artifacts,
    labels_from,
    manifest_base,
    relational_scale_for,
    run_case,
    source_ladder_metadata,
    summarize_rows,
    write_csv,
    write_handoffs,
    write_json,
)


def git_state() -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    except Exception as exc:
        return f"git status failed: {exc}\n"


def command_line() -> str:
    return " ".join([sys.executable, *sys.argv])


def base_refined_cfg(arm: str, source_mode: str, rel_scale: float, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "run_id": f"refined_{arm.lower()}_{source_mode}",
        "arm": arm,
        "source_mode": source_mode,
        "duration_name": "refined_T",
        "matrix_role": "four_arm_decomposition",
        "N": int(args.N),
        "L": float(args.L),
        "dt": float(args.dt),
        "T": float(args.T),
        "sample_dt": float(args.sample_dt),
        "D": 0.3,
        "source_width": 1.5,
        "beta_t": 1.0,
        "beta_s": 1.0,
        "probe_position": (4.0, 0.0, 0.0),
        "probe_width": 1.0,
        "probe_amplitude": 1.0,
        "normalize_probe": True,
        "target_norm": 1.0,
        "carrier_vector": (0.0, 0.0, 0.0),
        "relational_scale": rel_scale,
    }


def build_matrix(args: argparse.Namespace, rel_scale: float) -> list[dict[str, Any]]:
    source_modes = SOURCE_MODES if args.source == "all" else tuple(s.strip() for s in args.source.split(",") if s.strip())
    arms = ARMS if args.arms == "all" else tuple(a.strip() for a in args.arms.split(",") if a.strip())
    matrix = []
    for source_mode in source_modes:
        for arm in arms:
            matrix.append(base_refined_cfg(arm, source_mode, rel_scale, args))
    return matrix


def write_summary(outdir: Path, labels: list[str], arm_rows: list[dict[str, Any]]) -> None:
    summary = {
        "labels": labels,
        "purpose": "Higher-resolution bridge decomposition refinement",
        "grid": "N=128,L=40,dt=0.0005 by default",
        "bounded_interpretation": "Refinement evidence only; no gravity/geodesic/equivalence-principle claim.",
        "arm_rows": arm_rows,
    }
    write_json(outdir / "REFINEMENT_SUMMARY.json", summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--L", type=float, default=40.0)
    parser.add_argument("--dt", type=float, default=0.0005)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--sample-dt", type=float, default=0.05)
    parser.add_argument("--source", default="objective,relational")
    parser.add_argument("--arms", default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = args.out or (ROOT / "sweep_runs" / f"GRAVITY_TS_BRIDGE_REFINEMENT_GPU_{timestamp}")
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "trajectories").mkdir(exist_ok=True)

    preflight_record = preflight()
    write_json(outdir / "gpu_preflight.json", preflight_record)
    write_json(outdir / "environment_versions.json", {**environment_record(preflight_record), "command_line": command_line()})
    (outdir / "git_state_before.txt").write_text(git_state(), encoding="utf-8")

    rel_scale = relational_scale_for(build_grid(int(args.N), float(args.L)))
    matrix = build_matrix(args, rel_scale)
    write_json(outdir / "preregistered_matrix.json", matrix)

    results = []
    for cfg in matrix:
        print(f"running {cfg['run_id']}", flush=True)
        results.append(run_case(cfg, outdir))

    clock_rows, force_rows, traj_rows = summarize_rows(results)
    arm_rows, pass_flags = arm_comparisons(results)
    # No weak-probe rows in this refinement pass; only four-arm decomposition falsifiers apply.
    falsifications = falsification_rows(arm_rows, [])
    labels = labels_from(pass_flags, falsifications)
    common_rows = common_field_rows(results)
    manifest_rows = [manifest_base(result) for result in results]

    write_csv(outdir / "run_manifest.csv", manifest_rows)
    write_csv(outdir / "clock_metrics.csv", clock_rows)
    write_csv(outdir / "force_metrics.csv", force_rows)
    write_csv(outdir / "trajectory_metrics.csv", traj_rows)
    write_csv(outdir / "source_ladder_metadata.csv", source_ladder_metadata(results))
    write_csv(outdir / "arm_comparison.csv", arm_rows)
    write_csv(outdir / "falsification_results.csv", falsifications)
    write_handoffs(outdir, labels, arm_rows, falsifications, common_rows)
    write_summary(outdir, labels, arm_rows)
    (outdir / "git_status_short.txt").write_text(git_state(), encoding="utf-8")
    diff_names = subprocess.check_output(["git", "diff", "--name-only"], cwd=ROOT, text=True)
    cached_names = subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True)
    (outdir / "git_diff_name_only.txt").write_text(diff_names, encoding="utf-8")
    (outdir / "git_diff_cached_name_only.txt").write_text(cached_names, encoding="utf-8")
    write_csv(outdir / "artifact_hashes.csv", hash_artifacts(outdir))
    print(f"labels: {', '.join(labels)}", flush=True)
    print(f"outdir: {outdir}", flush=True)


if __name__ == "__main__":
    main()
