from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_ROOT = Path("E:/quantule_mapper_external_data")
OUT_ROOT = REPO_ROOT / "docs/external_validation/external_data/ingestion_smoke"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys or ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def inspect_zip(path: Path, sample: int = 20) -> dict[str, Any]:
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        suffix_counts = Counter(Path(i.filename).suffix.lower() or "<dir>" for i in infos)
        return {
            "zip_path": str(path),
            "member_count": len(infos),
            "compressed_size_bytes": path.stat().st_size,
            "uncompressed_size_bytes": sum(i.file_size for i in infos),
            "suffix_counts": dict(sorted(suffix_counts.items())),
            "sample_members": [
                {"filename": i.filename, "size_bytes": i.file_size}
                for i in infos[:sample]
            ],
        }


def ingest_nlse(root: Path) -> dict[str, Any]:
    zip_path = root / "raw/Archive.zip"
    result = {
        "dataset_id": "DATA-NUM-001",
        "adapter_status": "INGESTED_SOURCE_ARCHIVE",
        "direct_level3_comparison_ready": False,
        "comparison_role": "Reduced-NLS numerical-method sanity candidate only.",
        "queued_followup": "Run external NLSE package in an isolated environment on the same reduced flat-NLS boost identity used for C2 V2, then compare v=2Dk slope and mass/norm retention.",
        "created_utc": utc_now(),
    }
    if not zip_path.exists():
        result.update({"adapter_status": "MISSING_ARCHIVE", "reason": str(zip_path)})
        return result
    result["archive_inventory"] = inspect_zip(zip_path)
    result["comparison_fields_available_now"] = []
    result["project_data_available"] = [
        "docs/external_validation/generated_metrics/v2/summary.csv",
        "docs/external_validation/generated_metrics/v2/result.json",
        "sweep_runs/C27_REDERIVE/r3_n96.json",
    ]
    return result


def read_nist_roster(data_info_zip: Path) -> list[dict[str, str]]:
    with zipfile.ZipFile(data_info_zip) as z:
        text = z.read("data_info/data_roster.csv").decode("utf-8", errors="replace")
    return list(csv.DictReader(io.StringIO(text)))


def sample_npy_shape(data_zip: Path, member_name: str) -> dict[str, Any]:
    with zipfile.ZipFile(data_zip) as z:
        with z.open(member_name) as f:
            try:
                arr = np.load(f, allow_pickle=False)
            except ValueError as exc:
                return {
                    "member": member_name,
                    "load_status": "SKIPPED_UNSAFE_OBJECT_ARRAY",
                    "reason": str(exc),
                }
    return {
        "member": member_name,
        "load_status": "LOADED_NUMERIC_ARRAY",
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
    }


def ingest_nist(root: Path) -> dict[str, Any]:
    data_zip = root / "raw/data_files.zip"
    info_zip = root / "raw/data_info.zip"
    train_list = root / "raw/train_data_2101-05404.txt"
    test_list = root / "raw/test_data_2101-05404.txt"
    result = {
        "dataset_id": "DATA-BEC-001",
        "adapter_status": "INGESTED_MORPHOLOGY_DATASET",
        "direct_level3_comparison_ready": False,
        "comparison_role": "Morphology/image tooling candidate only; not C2/C3 bright-collision or Q-ball dynamics validation.",
        "queued_followup": "If useful, create a morphology-only benchmark that compares image-processing/node-detection robustness on NIST dark-soliton labels; do not map it to C2/C3 dynamics.",
        "created_utc": utc_now(),
    }
    missing = [str(p) for p in [data_zip, info_zip, train_list, test_list] if not p.exists()]
    if missing:
        result.update({"adapter_status": "MISSING_FILES", "missing": missing})
        return result
    roster = read_nist_roster(info_zip)
    labels = Counter(r.get("label_v3") or r.get("label_v2") or r.get("label_v1") or "" for r in roster)
    class_paths = Counter(Path(r.get("file_name", "")).parts[-2] if r.get("file_name") else "" for r in roster)
    train_count = len([line for line in train_list.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()])
    test_count = len([line for line in test_list.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()])
    data_inventory = inspect_zip(data_zip, sample=12)
    info_inventory = inspect_zip(info_zip, sample=20)
    first_npy = next(
        (m["filename"] for m in data_inventory["sample_members"] if m["filename"].endswith(".npy")),
        None,
    )
    if first_npy is None:
        with zipfile.ZipFile(data_zip) as z:
            first_npy = next((i.filename for i in z.infolist() if i.filename.endswith(".npy")), "")
    result.update(
        {
            "roster_rows": len(roster),
            "label_counts": dict(sorted(labels.items())),
            "class_path_counts": dict(sorted(class_paths.items())),
            "train_list_count": train_count,
            "test_list_count": test_count,
            "data_zip_inventory": data_inventory,
            "info_zip_inventory": info_inventory,
            "sample_npy": sample_npy_shape(data_zip, first_npy) if first_npy else None,
            "comparison_fields_available_now": [
                "image array",
                "class label",
                "excitation position for labelled dark soliton rows",
                "train/test split lists",
            ],
            "project_data_gap": "No project-side dark-soliton image/morphology dataset exists. Current Quantule Mapper external metrics are trajectory/field summaries, not labelled absorption images.",
        }
    )
    return result


def build_run_queue(nlse: dict[str, Any], nist: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "queue_id": "Q-EXT-NLSE-REDUCED-V2",
            "priority": "medium",
            "trigger": "DATA-NUM-001 ingested source archive but no external benchmark outputs exist yet.",
            "proposed_action": "Create an isolated adapter that runs the NLSE package on a reduced flat-NLS boost identity and compares v=2Dk against project V2 outputs.",
            "requires_new_quantule_simulation": False,
            "requires_external_package_execution": True,
            "claim_boundary": "method sanity only, not empirical validation",
        },
        {
            "queue_id": "Q-EXT-NIST-MORPHOLOGY-ADAPTER",
            "priority": "low",
            "trigger": "DATA-BEC-001 is ingestible but only morphology/image-label comparable.",
            "proposed_action": "If desired, build a morphology-only reader/preview and node/dark-notch detector benchmark using NIST labels. Do not compare to C2/C3 bright collision dynamics.",
            "requires_new_quantule_simulation": False,
            "requires_external_package_execution": False,
            "claim_boundary": "image tooling only, not dynamics validation",
        },
        {
            "queue_id": "Q-EXT-V1-DIGITIZE-MITSCHKE",
            "priority": "high for Level-3 V1",
            "trigger": "V1 needs experimental phase-force data, but Mitschke/Mollenauer is digitization-required.",
            "proposed_action": "Digitise or obtain tabulated optical soliton force/separation data before Level-3 V1 correlation.",
            "requires_new_quantule_simulation": False,
            "requires_external_package_execution": False,
            "claim_boundary": "external empirical correlation only after digitisation and units review",
        },
        {
            "queue_id": "Q-EXT-V5-DIGITIZE-NGUYEN",
            "priority": "medium for Level-3 V5",
            "trigger": "V5 needs phase/speed collision data, but Nguyen/Hulet data are not machine-readable locally.",
            "proposed_action": "Digitise paper plots or request author data before comparing C3 collision phase grid to BEC matter-wave observations.",
            "requires_new_quantule_simulation": False,
            "requires_external_package_execution": False,
            "claim_boundary": "close analogue only, not exact equation matching",
        },
    ]


def write_report(nlse: dict[str, Any], nist: dict[str, Any], queue: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "# External Dataset Ingestion Smoke Report",
            "",
            "Status: provisional until Claude review.",
            "",
            "Final decision: `EXTERNAL_DATA_INGESTION_READY_WITH_CANDIDATES`",
            "",
            "## Summary",
            "",
            "- Downloaded external datasets were inspected read-only.",
            "- No Quantule Mapper simulations were run.",
            "- No physical-correspondence claim is made.",
            "- No Level-3 dynamics correlation is ready from the downloaded datasets alone.",
            "",
            "## DATA-NUM-001 NLSE",
            "",
            f"- Adapter status: `{nlse.get('adapter_status')}`",
            f"- Role: {nlse.get('comparison_role')}",
            f"- Archive members: {nlse.get('archive_inventory', {}).get('member_count')}",
            f"- Uncompressed size: {nlse.get('archive_inventory', {}).get('uncompressed_size_bytes')} bytes",
            f"- Follow-up: {nlse.get('queued_followup')}",
            "",
            "## DATA-BEC-001 NIST Dark Solitons",
            "",
            f"- Adapter status: `{nist.get('adapter_status')}`",
            f"- Role: {nist.get('comparison_role')}",
            f"- Roster rows: {nist.get('roster_rows')}",
            f"- Label counts: `{nist.get('label_counts')}`",
            f"- Train/test counts: {nist.get('train_list_count')} / {nist.get('test_list_count')}",
            f"- Sample NPY: `{nist.get('sample_npy')}`",
            f"- Project data gap: {nist.get('project_data_gap')}",
            "",
            "## Queued Follow-Ups",
            "",
            *[
                f"- `{item['queue_id']}` ({item['priority']}): {item['proposed_action']}"
                for item in queue
            ],
            "",
            "## Guardrails",
            "",
            "- NIST dark solitons are not treated as direct C2/C3 collision validation.",
            "- NLSE package is not treated as empirical validation.",
            "- IRER formulations remain primary.",
            "- External datasets are comparison instruments only.",
        ]
    ) + "\n"


def run(args: argparse.Namespace) -> int:
    archive_root = Path(args.archive_root)
    out = Path(args.out)
    nlse = ingest_nlse(archive_root / "DATA-NUM-001_nlse_package")
    nist = ingest_nist(archive_root / "DATA-BEC-001_nist_dark_solitons_v2")
    queue = build_run_queue(nlse, nist)
    write_json(out / "nlse_ingestion_result.json", nlse)
    write_json(out / "nist_ingestion_result.json", nist)
    write_csv(out / "queued_followups.csv", queue)
    write_text(out / "EXTERNAL_DATA_INGESTION_SMOKE_REPORT.md", write_report(nlse, nist, queue))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only ingestion smoke tests for downloaded external datasets.")
    parser.add_argument("--archive-root", default=str(ARCHIVE_ROOT))
    parser.add_argument("--out", default=str(OUT_ROOT))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
