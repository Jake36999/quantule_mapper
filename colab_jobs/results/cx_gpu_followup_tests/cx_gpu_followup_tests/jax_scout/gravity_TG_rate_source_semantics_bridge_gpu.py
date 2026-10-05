"""Wrapper for the rate-source semantics bridge gate.

This script deliberately does not run temporal/geometric feedback. It reruns
the TG-S source-semantics event suite and extracts whether `R_relax` or
`L_lock` are admissible for a future feedback branch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_hashes(outdir: Path) -> list[dict[str, str]]:
    return [{"path": str(p.relative_to(outdir)).replace("\\", "/"), "sha256": sha256(p)} for p in sorted(outdir.rglob("*")) if p.is_file() and p.name != "artifact_hashes.csv"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "sweep_runs" / f"TG_RATE_SOURCE_SEMANTICS_BRIDGE_{time.strftime('%Y%m%d_%H%M%S')}"))
    parser.add_argument("--N", type=int, default=48)
    parser.add_argument("--sample-dt", type=float, default=0.05)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.out)
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    semantics_out = outdir / "tg_s_semantics"
    write_json(
        outdir / "bridge_scope.json",
        {
            "scope": "source semantics only; no T/G feedback evolution",
            "future_admissible_sources": ["R_relax", "L_lock"],
            "comparison_only": ["P_threshold"],
        },
    )
    cmd = [
        args.python,
        "-u",
        str(ROOT / "jax_scout" / "gravity_TG_S_source_semantics_gpu.py"),
        "--out",
        str(semantics_out),
        "--N",
        str(args.N),
        "--sample-dt",
        str(args.sample_dt),
    ]
    if args.dry_run:
        write_json(outdir / "RUN_COMPLETE.json", {"status": "DRY_RUN_ONLY", "command": " ".join(cmd)})
        print(json.dumps({"status": "DRY_RUN_ONLY", "outdir": str(outdir)}, indent=2))
        return
    log_path = outdir / "tg_s_semantics_stdout_stderr.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    class_rows = read_csv(semantics_out / "source_semantics_classification.csv")
    bridge_rows = []
    for row in class_rows:
        source = row.get("source_family", "")
        if source in {"R_relax", "L_lock", "S_state", "P_threshold"}:
            bridge_rows.append(
                {
                    "source_family": source,
                    "classification": row.get("classification", ""),
                    "passes_semantics_gate": row.get("passes_semantics_gate", ""),
                    "supported_label": row.get("supported_label", ""),
                    "admissible_for_future_feedback": source in {"R_relax", "L_lock"} and str(row.get("passes_semantics_gate")) == "True",
                    "reason": row.get("reason", ""),
                }
            )
    admissible = [row["source_family"] for row in bridge_rows if row["admissible_for_future_feedback"]]
    status = "TG_RATE_SOURCE_ADMISSIBLE_FOR_DESIGN_REVIEW" if admissible else "TG_RATE_SOURCE_NOT_READY_FOR_FEEDBACK"
    write_csv(outdir / "rate_source_admissibility.csv", bridge_rows)
    write_json(outdir / "rate_source_bridge_summary.json", {"status": status, "admissible_sources": admissible, "tg_s_returncode": proc.returncode, "log": str(log_path)})
    (outdir / "TECHNICAL_HANDOFF.md").write_text(
        "# TG Rate Source Semantics Bridge\n\n"
        f"Status: `{status}`.\n\n"
        "No feedback loop was run. This only determines whether a rate-like source can be selected for a future branch.\n",
        encoding="utf-8",
    )
    (outdir / "OPEN_QUESTIONS.md").write_text("# Open Questions\n\n- Which single admissible source, if any, should be selected for the next feedback branch?\n", encoding="utf-8")
    write_csv(outdir / "artifact_hashes.csv", artifact_hashes(outdir))
    write_json(outdir / "RUN_COMPLETE.json", {"status": status, "outdir": str(outdir)})
    print(json.dumps({"status": status, "outdir": str(outdir)}, indent=2))


if __name__ == "__main__":
    main()
