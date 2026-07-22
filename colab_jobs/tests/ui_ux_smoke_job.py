"""Lightweight Colab UI/UX smoke job.

This is not a scientific simulation. It exists to exercise the Colab capsule
control panel, stage logging, expected-output validation, checkpointing, and
archive flow with a tiny workload.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import sys
import time


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def progress(label: str, steps: int, sleep_seconds: float) -> list[dict]:
    events = []
    for index in range(1, steps + 1):
        event = {
            "label": label,
            "step": index,
            "steps": steps,
            "timestamp_utc": utc_now(),
        }
        print(f"[{label}] step {index}/{steps} at {event['timestamp_utc']}", flush=True)
        events.append(event)
        time.sleep(sleep_seconds)
    return events


def phase_prepare(out_dir: Path, steps: int, sleep_seconds: float) -> None:
    print("UI smoke prepare phase starting.", flush=True)
    events = progress("prepare", steps, sleep_seconds)
    write_json(
        out_dir / "preflight_marker.json",
        {
            "phase": "prepare",
            "status": "PREPARE_PASS",
            "events": events,
            "python": sys.version,
            "platform": platform.platform(),
            "cwd": os.getcwd(),
            "timestamp_utc": utc_now(),
        },
    )
    print(f"UI smoke prepare phase wrote {out_dir / 'preflight_marker.json'}", flush=True)


def phase_run(out_dir: Path, steps: int, sleep_seconds: float) -> None:
    print("UI smoke run phase starting.", flush=True)
    events = progress("run", steps, sleep_seconds)
    metrics_path = out_dir / "ui_smoke_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerow({"metric": "events_emitted", "value": len(events)})
        writer.writerow({"metric": "sleep_seconds", "value": sleep_seconds})
        writer.writerow({"metric": "stdout_flush_test", "value": "pass"})
        writer.writerow({"metric": "expected_output_test", "value": "pass"})

    write_json(
        out_dir / "ui_event_log.json",
        {
            "phase": "run",
            "events": events,
            "timestamp_utc": utc_now(),
        },
    )
    write_json(
        out_dir / "completion_status.json",
        {
            "status": "UI_UX_SMOKE_PASS",
            "purpose": "Control-panel and archive workflow smoke test only.",
            "timestamp_utc": utc_now(),
            "outputs": [
                "preflight_marker.json",
                "ui_smoke_metrics.csv",
                "ui_event_log.json",
                "completion_status.json",
            ],
        },
    )
    print(f"UI smoke run phase wrote {metrics_path}", flush=True)
    print("UI_UX_SMOKE_PASS", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a tiny Colab UI/UX smoke phase.")
    parser.add_argument("--phase", choices=["prepare", "run"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    if args.steps < 1:
        raise SystemExit("--steps must be at least 1")
    if args.sleep < 0:
        raise SystemExit("--sleep must be non-negative")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.phase == "prepare":
        phase_prepare(out_dir, args.steps, args.sleep)
    else:
        phase_run(out_dir, args.steps, args.sleep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
