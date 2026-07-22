from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_hashes(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    return rows


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def run_streaming(command: list[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        proc = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert proc.stdout is not None
        for chunk in iter(proc.stdout.readline, b""):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            log.write(chunk)
            log.flush()
        return proc.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TG-B1S-D reproduction then compare against a frozen baseline.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--simulation-entrypoint", default="jax_scout/gravity_TG_B1S_drift_decomposition_gpu.py")
    parser.add_argument("--analysis-entrypoint", default="colab_jobs/tg_b1s_d_reproduction_compare.py")
    parser.add_argument("--matrix", default="primary")
    parser.add_argument("--sample-dt", default="1.0")
    args = parser.parse_args()

    root = Path.cwd()
    out_dir = Path(args.out)
    simulation_dir = out_dir / "simulation"
    logs_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    simulation_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    baseline_manifest = Path(args.baseline_manifest)
    simulation_command = [
        sys.executable,
        "-u",
        args.simulation_entrypoint,
        "--out",
        str(simulation_dir),
        "--matrix",
        args.matrix,
        "--sample-dt",
        str(args.sample_dt),
    ]
    analysis_command = [
        sys.executable,
        "-u",
        args.analysis_entrypoint,
        "--baseline-manifest",
        str(baseline_manifest),
        "--reproduction-run",
        str(simulation_dir),
        "--out",
        str(out_dir),
    ]
    write_json(out_dir / "simulation_command.json", {"argv": simulation_command, "started_utc": now_iso()})
    sim_rc = run_streaming(simulation_command, cwd=root, log_path=logs_dir / "simulation_stdout_stderr.log")
    write_json(out_dir / "simulation_command.json", {"argv": simulation_command, "return_code": sim_rc, "finished_utc": now_iso()})
    if sim_rc != 0:
        status = {"status": "REPRODUCTION_INCOMPLETE", "stage": "simulation", "return_code": sim_rc}
        write_json(out_dir / "completion_status.json", status)
        raise SystemExit(sim_rc)

    write_json(out_dir / "analysis_command.json", {"argv": analysis_command, "started_utc": now_iso()})
    analysis_rc = run_streaming(analysis_command, cwd=root, log_path=logs_dir / "analysis_stdout_stderr.log")
    write_json(out_dir / "analysis_command.json", {"argv": analysis_command, "return_code": analysis_rc, "finished_utc": now_iso()})

    copy_if_exists(root / "gpu_preflight.json", out_dir / "gpu_preflight_before.json")
    copy_if_exists(root / "gpu_preflight_after_dependencies.json", out_dir / "gpu_preflight_final.json")
    if not (out_dir / "gpu_preflight_before.json").exists():
        copy_if_exists(simulation_dir / "gpu_preflight.json", out_dir / "gpu_preflight_before.json")
    if not (out_dir / "gpu_preflight_final.json").exists():
        copy_if_exists(simulation_dir / "gpu_preflight.json", out_dir / "gpu_preflight_final.json")

    write_json(out_dir / "artifact_hashes.json", {"files": artifact_hashes(out_dir)})
    if analysis_rc != 0:
        raise SystemExit(analysis_rc)


if __name__ == "__main__":
    main()
