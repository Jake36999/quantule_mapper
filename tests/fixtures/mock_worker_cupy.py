import argparse
import json
import os
import sys
import time
from pathlib import Path


def _append_trace(event: dict) -> None:
    trace_path = os.environ.get("ASTE_TRACE_FILE")
    if not trace_path:
        return
    trace_file = Path(trace_path)
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    with open(trace_file, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    sleep_seconds = float(os.environ.get("ASTE_MOCK_SLEEP_SECONDS", "0"))
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as handle:
        handle.write(b"MOCK_H5_ARTIFACT")

    sidecar_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    with open(sidecar_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "config_hash": manifest.get("config_hash"),
                "job_id": manifest.get("job_id"),
                "generation": manifest.get("generation"),
                "origin": manifest.get("origin"),
                "params": manifest.get("params", {}),
            },
            handle,
            indent=2,
        )

    _append_trace(
        {
            "stage": "mock_worker",
            "pid": os.getpid(),
            "argv": sys.argv,
            "manifest": str(Path(args.manifest).resolve()),
            "output": str(output_path.resolve()),
            "config_hash": manifest.get("config_hash"),
        }
    )

    if os.environ.get("ASTE_MOCK_FORCE_FAIL", "0") == "1":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
