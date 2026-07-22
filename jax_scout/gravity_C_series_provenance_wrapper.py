"""Unified provenance wrapper for Claude's Gravity C.1-C.3 clock scripts.

The wrapper does not change the C-series implementations.  It reruns them,
captures exact commands/stdout/stderr, copies summary JSON files, records
environment/git state, and hashes the resulting evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def git_state() -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    except Exception as exc:
        return f"git status failed: {exc}\n"


def config_hash(cfg: dict[str, Any]) -> str:
    data = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def run_command(label: str, cmd: list[str], outdir: Path) -> dict[str, Any]:
    started = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    elapsed = time.time() - started
    (outdir / "logs").mkdir(exist_ok=True)
    (outdir / "logs" / f"{label}.stdout.txt").write_text(proc.stdout, encoding="utf-8", errors="replace")
    (outdir / "logs" / f"{label}.stderr.txt").write_text(proc.stderr, encoding="utf-8", errors="replace")
    return {
        "label": label,
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "stdout_path": f"logs/{label}.stdout.txt",
        "stderr_path": f"logs/{label}.stderr.txt",
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def copy_summary(label: str, source_path: Path, outdir: Path) -> dict[str, Any]:
    target_dir = outdir / "summaries"
    target_dir.mkdir(exist_ok=True)
    target = target_dir / f"{label}_summary.json"
    if source_path.exists():
        shutil.copy2(source_path, target)
        payload = json.loads(target.read_text(encoding="utf-8"))
        return {"summary_path": str(target.relative_to(outdir)).replace("\\", "/"), "summary": payload}
    return {"summary_path": "", "summary": {"error": f"missing summary: {source_path}"}}


def latest_c3_dir(before: set[Path]) -> Path | None:
    candidates = set((ROOT / "sweep_runs").glob("GRAVITY_C3_BACKREACT_*")) - before
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def hash_artifacts(outdir: Path) -> list[dict[str, str]]:
    rows = []
    for path in sorted(outdir.rglob("*")):
        if path.is_file() and path.name != "artifact_hashes.csv":
            rows.append({"path": str(path.relative_to(outdir)).replace("\\", "/"), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return rows


def parse_c3_out(stdout: str) -> str:
    matches = re.findall(r"out=([^=\n]+?)\s*=+", stdout)
    return matches[-1].strip() if matches else ""


def interpretation_overlay(summaries: dict[str, Any]) -> dict[str, Any]:
    c1 = summaries.get("C1", {})
    c2 = summaries.get("C2", {})
    c3 = summaries.get("C3", {})
    c3_rows = c3.get("rows", [])
    c3_freqs = [float(row.get("freq_frac", float("nan"))) for row in c3_rows]
    c3_increasing = all(c3_freqs[i + 1] >= c3_freqs[i] - 1e-12 for i in range(len(c3_freqs) - 1))
    return {
        "C1_current_interpretation": (
            "SHARED_FROZEN_LAPSE_CLOCK_CONSISTENCY; historical script language "
            "says B_O emergence, but both mechanisms were supplied the same lapse."
        ),
        "C2_current_interpretation": (
            "RELATIONAL_SOURCE_PROBE_DEPENDENCE_AND_WEAK_PROBE_NONUNIVERSALITY; "
            "source-layer gravity success not established."
        ),
        "C3_current_interpretation": (
            "STABLE_SELF_RELIEVING_TEMPORAL_BACKREACTION; not a gravity verdict."
        ),
        "C3_freq_frac_monotone_increasing_recomputed": c3_increasing,
        "C3_original_freq_monotone_with_lam_field": c3.get("findings", {}).get("freq_monotone_with_lam"),
        "C3_monotonicity_note": (
            "The original C3 script checks for decreasing frequency with lambda; "
            "the rows increase from the frozen value toward the self-relieved value."
        ),
        "C1_original_B_O_flag": c1.get("findings", {}).get("B_O_emerges_two_mechanisms_agree"),
        "C2_own_N_converging_flag": c2.get("emergence", {}).get("converging_toward_point_value"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--phi-iso", default="sweep_runs/PHASE_D_C2_8_TWONODE_CODEX_20260709_123611/phi_iso.npy")
    parser.add_argument("--L", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = args.out or (ROOT / "sweep_runs" / f"GRAVITY_C_SERIES_PROVENANCE_{timestamp}")
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    environment = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "git_commit": git_commit(),
        "cwd": str(ROOT),
        "command_line": " ".join([sys.executable, *sys.argv]),
    }
    write_json(outdir / "environment_versions.json", environment)
    (outdir / "git_state_before.txt").write_text(git_state(), encoding="utf-8")

    phi_iso = args.phi_iso
    c1_out = outdir / "C1"
    c2_out = outdir / "C2"
    commands = [
        (
            "C1",
            [
                sys.executable,
                "jax_scout/gravity_C1_relational_clocks.py",
                "--phi-iso",
                phi_iso,
                "--L",
                str(args.L),
                "--out",
                str(c1_out),
            ],
        ),
        (
            "C2",
            [
                sys.executable,
                "jax_scout/gravity_C2_source_map.py",
                "--phi-iso",
                phi_iso,
                "--L",
                str(args.L),
                "--out",
                str(c2_out),
            ],
        ),
    ]
    manifest = []
    summaries = {}
    for label, cmd in commands:
        row = run_command(label, cmd, outdir)
        row["config_hash"] = config_hash({"label": label, "cmd": cmd})
        summary_info = copy_summary(label, (c1_out if label == "C1" else c2_out) / "summary.json", outdir)
        summaries[label] = summary_info["summary"]
        row["summary_path"] = summary_info["summary_path"]
        manifest.append({k: v for k, v in row.items() if k not in ("stdout", "stderr")})
        if row["returncode"] != 0:
            write_csv(outdir / "command_manifest.csv", manifest)
            write_json(outdir / "C_SERIES_PROVENANCE_SUMMARY.json", {"error": f"{label} failed", "manifest": manifest})
            raise SystemExit(row["returncode"])

    before_c3 = set((ROOT / "sweep_runs").glob("GRAVITY_C3_BACKREACT_*"))
    c3_cmd = [sys.executable, "jax_scout/gravity_C3_backreaction.py"]
    c3_row = run_command("C3", c3_cmd, outdir)
    c3_row["config_hash"] = config_hash({"label": "C3", "cmd": c3_cmd})
    c3_dir = latest_c3_dir(before_c3)
    parsed = parse_c3_out(c3_row["stdout"])
    if c3_dir is None and parsed:
        c3_dir = Path(parsed)
        if not c3_dir.is_absolute():
            c3_dir = ROOT / c3_dir
    summary_info = copy_summary("C3", c3_dir / "summary.json" if c3_dir else Path("__missing__"), outdir)
    summaries["C3"] = summary_info["summary"]
    c3_row["summary_path"] = summary_info["summary_path"]
    c3_row["raw_run_path"] = str(c3_dir.relative_to(ROOT)).replace("\\", "/") if c3_dir and c3_dir.exists() else ""
    manifest.append({k: v for k, v in c3_row.items() if k not in ("stdout", "stderr")})
    write_csv(outdir / "command_manifest.csv", manifest)

    labels = []
    if summaries.get("C1", {}).get("verdict") == "C1_FROZEN_PASS":
        labels.append("C1_SHARED_FROZEN_LAPSE_CLOCK_CONSISTENCY_REPRODUCED")
    if summaries.get("C2", {}).get("verdict") == "C2_SOURCE_MAP_DONE":
        labels.append("C2_SOURCE_MAP_REPRODUCED")
    if summaries.get("C3", {}).get("verdict") == "C3_BACKREACTION_STABLE":
        labels.append("C3_BACKREACTION_REPRODUCED")
    overlay = interpretation_overlay(summaries)
    write_json(
        outdir / "C_SERIES_PROVENANCE_SUMMARY.json",
        {
            "labels": labels,
            "bounded_interpretation": "C-series temporal throttling evidence reproduced with provenance capture; not a gravity verdict.",
            "current_interpretation_overlay": overlay,
            "summaries": summaries,
            "manifest": manifest,
        },
    )
    handoff = [
        "# Gravity C-Series Provenance Rerun",
        "",
        "This wrapper reran Claude's C.1-C.3 scripts without changing their implementations.",
        "The interpretation overlay corrects historical wording without editing the original scripts.",
        "",
        "## Labels",
        "",
        *[f"- `{label}`" for label in labels],
        "",
        "## Boundary",
        "",
        "These are temporal-throttling and source-map mirror results. They do not validate gravity, geodesics, equivalence principle behaviour, or production geometry.",
    ]
    (outdir / "TECHNICAL_HANDOFF.md").write_text("\n".join(handoff) + "\n", encoding="utf-8")
    (outdir / "git_status_short.txt").write_text(git_state(), encoding="utf-8")
    (outdir / "git_diff_name_only.txt").write_text(subprocess.check_output(["git", "diff", "--name-only"], cwd=ROOT, text=True), encoding="utf-8")
    (outdir / "git_diff_cached_name_only.txt").write_text(subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True), encoding="utf-8")
    write_csv(outdir / "artifact_hashes.csv", hash_artifacts(outdir))
    print(f"labels: {', '.join(labels)}", flush=True)
    print(f"outdir: {outdir}", flush=True)


if __name__ == "__main__":
    main()
