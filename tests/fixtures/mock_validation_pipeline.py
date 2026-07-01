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
    parser.add_argument("--input", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    params_path = Path(args.params)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        return 3

    with open(params_path, "r", encoding="utf-8") as handle:
        params = json.load(handle)

    sleep_seconds = float(os.environ.get("ASTE_MOCK_VALIDATION_SLEEP_SECONDS", "0"))
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    config_hash = str(params.get("config_hash", "unknown"))
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = output_dir / f"provenance_{config_hash}.json"

    payload = {
        "spectral_fidelity": {
            "log_prime_sse": float(params.get("mock_sse", 0.75)),
            "bragg_peaks_detected": 3,
            "n_bragg_peaks": 3,
            "bragg_prime_sse": 0.21,
            "collapse_event_count": 1,
        },
        "aletheia_metrics": {
            "pcs": float(params.get("mock_pcs", 0.33)),
            "ic": float(params.get("mock_ic", 0.11)),
        },
    }

    with open(provenance_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    _append_trace(
        {
            "stage": "mock_validation",
            "pid": os.getpid(),
            "argv": sys.argv,
            "input": str(input_path.resolve()),
            "params": str(params_path.resolve()),
            "provenance": str(provenance_path.resolve()),
            "config_hash": config_hash,
        }
    )

    if os.environ.get("ASTE_MOCK_VALIDATION_FAIL", "0") == "1":
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
