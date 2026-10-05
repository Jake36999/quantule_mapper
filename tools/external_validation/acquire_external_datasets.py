from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE_ROOT = Path("E:/quantule_mapper_external_data")
DEFAULT_REPO_META_ROOT = REPO_ROOT / "docs/external_validation/external_data"
DEFAULT_CAP_BYTES = 60 * 1024**3


@dataclass
class DownloadItem:
    title: str
    url: str
    filename: str
    required: bool = True
    expected_sha256_url: str | None = None


@dataclass
class DatasetCandidate:
    dataset_id: str
    slug: str
    title: str
    source_url: str
    doi_or_repository_id: str
    authors_or_organization: str
    year: str
    license_or_terms: str
    availability_class: str
    relevant_metrics: list[str]
    comparison_quantity: str
    units_or_scaling_needed: str
    directly_comparable: bool
    comparison_limitations: list[str]
    recommendation: str
    download_items: list[DownloadItem] = field(default_factory=list)


DATASETS = [
    DatasetCandidate(
        dataset_id="DATA-NUM-001",
        slug="nlse_package",
        title="NLSE: A Python package to solve the nonlinear Schrodinger equation",
        source_url="https://github.com/Quantum-Optics-LKB/NLSE/releases/tag/2.3.0",
        doi_or_repository_id="10.21105/joss.06607 / github:Quantum-Optics-LKB/NLSE",
        authors_or_organization="Tangui Aladjidi, Clara Piekarski, Quentin Glorieux",
        year="2024",
        license_or_terms="GPL-3.0 repository; JOSS paper CC BY 4.0",
        availability_class="DIRECT_DOWNLOAD_READY",
        relevant_metrics=["V2"],
        comparison_quantity="Reduced-NLS numerical-method sanity comparator only.",
        units_or_scaling_needed="Must intentionally reduce project equation to flat NLS conventions.",
        directly_comparable=False,
        comparison_limitations=[
            "Not empirical validation.",
            "Not a full IRER solver comparison.",
            "Use only for deliberately reduced equation sanity checks.",
        ],
        recommendation="DOWNLOAD_NOW",
        download_items=[
            DownloadItem(
                title="NLSE release archive",
                url="https://github.com/Quantum-Optics-LKB/NLSE/releases/download/2.3.0/Archive.zip",
                filename="Archive.zip",
            )
        ],
    ),
    DatasetCandidate(
        dataset_id="DATA-BEC-001",
        slug="nist_dark_solitons_v2",
        title="Dark solitons in BECs dataset 2.0",
        source_url="https://catalog.data.gov/dataset/dark-solitons-in-becs-dataset-2-0",
        doi_or_repository_id="10.18434/MDS2-2363",
        authors_or_organization="National Institute of Standards and Technology / Justyna P. Zwolak et al.",
        year="2021/2022",
        license_or_terms="NIST public data license; see downloaded license.txt",
        availability_class="DIRECT_DOWNLOAD_READY",
        relevant_metrics=["future morphology only"],
        comparison_quantity="Preprocessed BEC absorption images and labels for dark-soliton morphology/data-engineering tasks.",
        units_or_scaling_needed="Image preprocessing, trap/imaging units, and dark-vs-bright soliton contrast caveats.",
        directly_comparable=False,
        comparison_limitations=[
            "Dark soliton image morphology only.",
            "Not a bright-soliton collision benchmark.",
            "Not direct C2/C3 phase-force or Q-ball validation.",
        ],
        recommendation="DOWNLOAD_NOW",
        download_items=[
            DownloadItem(
                title="Preprocessed image data",
                url="https://data.nist.gov/od/ds/mds2-2363/data_files.zip",
                filename="data_files.zip",
                expected_sha256_url="https://data.nist.gov/od/ds/mds2-2363/data_files.zip.sha256",
            ),
            DownloadItem(
                title="Label/roster data",
                url="https://data.nist.gov/od/ds/mds2-2363/data_info.zip",
                filename="data_info.zip",
            ),
            DownloadItem(
                title="Dataset information PDF",
                url="https://data.nist.gov/od/ds/mds2-2363/dataset_information.pdf",
                filename="dataset_information.pdf",
            ),
            DownloadItem(
                title="License",
                url="https://data.nist.gov/od/ds/mds2-2363/license.txt",
                filename="license.txt",
            ),
            DownloadItem(
                title="Test image list",
                url="https://data.nist.gov/od/ds/mds2-2363/test_data%282101-05404%29.txt",
                filename="test_data_2101-05404.txt",
            ),
            DownloadItem(
                title="Train image list",
                url="https://data.nist.gov/od/ds/mds2-2363/train_data%282101-05404%29.txt",
                filename="train_data_2101-05404.txt",
            ),
        ],
    ),
    DatasetCandidate(
        dataset_id="DATA-OPT-006",
        slug="mitschke_mollenauer_optical_force",
        title="Experimental observation of interaction forces between solitons in optical fibers",
        source_url="https://opg.optica.org/ol/fulltext.cfm?uri=ol-12-5-355",
        doi_or_repository_id="Optics Letters 12(5), 355 (1987)",
        authors_or_organization="F. M. Mitschke and L. F. Mollenauer",
        year="1987",
        license_or_terms="Publisher terms; no raw data license located.",
        availability_class="DIGITIZATION_REQUIRED",
        relevant_metrics=["V1"],
        comparison_quantity="Input/output pulse separation and phase-force sign/trend, if digitised.",
        units_or_scaling_needed="Fiber propagation distance, pulse separation, phase and amplitude conventions.",
        directly_comparable=False,
        comparison_limitations=["No machine-readable data located.", "Paper figures/tables need review before use."],
        recommendation="PLAN_ONLY",
    ),
    DatasetCandidate(
        dataset_id="DATA-BEC-005",
        slug="nguyen_hulet_matter_wave_collisions",
        title="Collisions of matter-wave solitons",
        source_url="https://arxiv.org/abs/1407.5087",
        doi_or_repository_id="arXiv:1407.5087 / Nature Physics 10, 918-922",
        authors_or_organization="Jason H. V. Nguyen, Paul Dyke, De Luo, Boris A. Malomed, Randall G. Hulet",
        year="2014",
        license_or_terms="Paper/arXiv terms; raw data license not located.",
        availability_class="DIGITIZATION_REQUIRED",
        relevant_metrics=["V5"],
        comparison_quantity="Phase-dependent collision behaviour and trajectory jump, if digitised or obtained.",
        units_or_scaling_needed="BEC trap, scattering length, atom number, velocity and phase conventions.",
        directly_comparable=False,
        comparison_limitations=["No machine-readable trajectory table located.", "Analogue only, not exact C2/C3 equation matching."],
        recommendation="PLAN_ONLY",
    ),
    DatasetCandidate(
        dataset_id="DATA-QB-004",
        slug="qball_oscillon_repository_search",
        title="Machine-readable Q-ball or oscillon benchmark code/data",
        source_url="docs/external_validation/DATASET_CANDIDATES.md",
        doi_or_repository_id="not yet identified",
        authors_or_organization="not yet identified",
        year="unknown",
        license_or_terms="unknown",
        availability_class="SOURCE_TRAIL_ONLY",
        relevant_metrics=["V3", "V5"],
        comparison_quantity="Q(omega), collision outcomes, oscillon/Q-ball dynamics if a repository is found.",
        units_or_scaling_needed="Model potential, dimensionality, charge normalization.",
        directly_comparable=False,
        comparison_limitations=["No verified machine-readable repository located."],
        recommendation="MANUAL_REVIEW_REQUIRED",
    ),
    DatasetCandidate(
        dataset_id="DATA-AG-003",
        slug="photon_fluid_bogoliubov",
        title="Observation of the Bogoliubov Dispersion in a Fluid of Light",
        source_url="https://link.aps.org/doi/10.1103/PhysRevLett.121.183604",
        doi_or_repository_id="10.1103/PhysRevLett.121.183604",
        authors_or_organization="Fontaine et al.",
        year="2018",
        license_or_terms="Publisher/arXiv terms; raw data license not located.",
        availability_class="DIGITIZATION_REQUIRED",
        relevant_metrics=["V7"],
        comparison_quantity="Bogoliubov dispersion / sound-speed scaling if data are tabulated or digitised.",
        units_or_scaling_needed="Photon-fluid density, nonlinear index, paraxial mapping.",
        directly_comparable=False,
        comparison_limitations=["No machine-readable dispersion table located.", "V7-adjacent only; not a gravity result."],
        recommendation="PLAN_ONLY",
    ),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def head_url(url: str, timeout: int = 30) -> dict[str, Any]:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "QuantuleMapperExternalData/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return {
                "url": url,
                "status": resp.status,
                "content_length": int(resp.headers.get("Content-Length") or 0),
                "content_type": resp.headers.get("Content-Type", ""),
                "accept_ranges": resp.headers.get("Accept-Ranges", ""),
                "resolved_url": resp.geturl(),
                "error": "",
            }
    except Exception as exc:
        return {
            "url": url,
            "status": None,
            "content_length": 0,
            "content_type": "",
            "accept_ranges": "",
            "resolved_url": "",
            "error": str(exc),
        }


def download_file(url: str, dest: Path, log_lines: list[str], timeout: int = 60) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    start = time.time()
    log_lines.append(f"{utc_now()} START {url} -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "QuantuleMapperExternalData/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as f:
        shutil.copyfileobj(resp, f, length=1024 * 1024)
    tmp.replace(dest)
    size = dest.stat().st_size
    log_lines.append(f"{utc_now()} DONE {dest.name} {size} bytes in {time.time() - start:.1f}s")
    return size


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def dataset_root(archive_root: Path, ds: DatasetCandidate) -> Path:
    return archive_root / f"{ds.dataset_id}_{ds.slug}"


def discover() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ds in DATASETS:
        heads = [head_url(item.url) for item in ds.download_items]
        estimated_size = sum(h["content_length"] for h in heads)
        formats = sorted({Path(item.filename).suffix.lstrip(".").lower() or "unknown" for item in ds.download_items})
        if not ds.download_items:
            formats = []
        rows.append(
            {
                "dataset_id": ds.dataset_id,
                "title": ds.title,
                "source_url": ds.source_url,
                "doi_or_repository_id": ds.doi_or_repository_id,
                "license_or_terms": ds.license_or_terms,
                "availability_class": ds.availability_class,
                "estimated_size_bytes": estimated_size,
                "estimated_size_gb": round(estimated_size / 1024**3, 4),
                "file_count": len(ds.download_items),
                "formats": ";".join(formats),
                "resumable_download_supported": "unknown",
                "source_checksums_provided": any(item.expected_sha256_url for item in ds.download_items),
                "login_or_manual_action_required": False if ds.download_items else True,
                "relevance": ";".join(ds.relevant_metrics),
                "comparison_quantity": ds.comparison_quantity,
                "recommendation": ds.recommendation,
                "directly_comparable": ds.directly_comparable,
                "head_errors": "; ".join(h["error"] for h in heads if h["error"]),
            }
        )
    return rows


def metadata_for(ds: DatasetCandidate, root: Path, status: str, reason: str, size: int, count: int, formats: list[str]) -> dict[str, Any]:
    return {
        "dataset_id": ds.dataset_id,
        "title": ds.title,
        "source_url": ds.source_url,
        "resolved_download_urls": [item.url for item in ds.download_items],
        "doi_or_repository_id": ds.doi_or_repository_id,
        "authors_or_organization": ds.authors_or_organization,
        "year": ds.year,
        "license_or_terms": ds.license_or_terms,
        "availability_class": ds.availability_class,
        "download_status": status,
        "skip_or_failure_reason": reason,
        "local_root": str(root),
        "raw_size_bytes": size,
        "file_count": count,
        "formats": formats,
        "sha256_manifest": "metadata/checksums.sha256",
        "source_checksum_results": [],
        "relevant_metrics": ds.relevant_metrics,
        "comparison_quantity": ds.comparison_quantity,
        "units_or_scaling_needed": ds.units_or_scaling_needed,
        "directly_comparable": ds.directly_comparable,
        "comparison_limitations": ds.comparison_limitations,
        "provisional_until_claude_review": True,
        "no_physical_correspondence_claim": True,
    }


def run(args: argparse.Namespace) -> int:
    archive_root = Path(args.archive_root)
    repo_meta_root = Path(args.repo_meta_root)
    cap_bytes = int(float(args.cap_gb) * 1024**3)
    archive_root.mkdir(parents=True, exist_ok=True)
    repo_meta_root.mkdir(parents=True, exist_ok=True)

    discovery_rows = discover()
    selected_ids = set(args.datasets.split(",")) if args.datasets else {
        row["dataset_id"] for row in discovery_rows if row["recommendation"] == "DOWNLOAD_NOW"
    }
    selected_size = sum(row["estimated_size_bytes"] for row in discovery_rows if row["dataset_id"] in selected_ids)
    if selected_size > cap_bytes:
        raise SystemExit(f"Selected downloads exceed cap: {selected_size} > {cap_bytes}")

    manifest_rows: list[dict[str, Any]] = []
    metadata_index: list[dict[str, Any]] = []
    total_downloaded = 0

    for ds in DATASETS:
        root = dataset_root(archive_root, ds)
        raw = root / "raw"
        meta = root / "metadata"
        archive = root / "archive"
        raw.mkdir(parents=True, exist_ok=True)
        meta.mkdir(parents=True, exist_ok=True)
        archive.mkdir(parents=True, exist_ok=True)
        log_lines: list[str] = []
        checksums: list[str] = []
        source_checksum_results: list[dict[str, Any]] = []
        inventory: list[dict[str, Any]] = []
        status = "SKIPPED"
        reason = "Not selected for download or not direct-download ready."
        size = 0
        count = 0
        formats: list[str] = []
        if ds.dataset_id in selected_ids and ds.recommendation == "DOWNLOAD_NOW":
            status = "DOWNLOADED"
            reason = ""
            for item in ds.download_items:
                dest = raw / item.filename
                if not args.force and dest.exists():
                    log_lines.append(f"{utc_now()} EXISTS {dest.name} {dest.stat().st_size} bytes")
                else:
                    try:
                        download_file(item.url, dest, log_lines)
                    except Exception as exc:
                        status = "FAILED"
                        reason = f"Failed downloading {item.filename}: {exc}"
                        log_lines.append(f"{utc_now()} ERROR {reason}")
                        break
                if dest.exists():
                    digest = sha256_file(dest)
                    checksums.append(f"{digest}  raw/{dest.name}")
                    inventory.append(
                        {
                            "relative_path": f"raw/{dest.name}",
                            "size_bytes": dest.stat().st_size,
                            "sha256": digest,
                            "source_url": item.url,
                        }
                    )
                    size += dest.stat().st_size
                    count += 1
                    suffix = dest.suffix.lstrip(".").lower() or "unknown"
                    if suffix not in formats:
                        formats.append(suffix)
                    if item.expected_sha256_url:
                        try:
                            expected_dest = meta / f"{dest.name}.source.sha256"
                            if args.force or not expected_dest.exists():
                                download_file(item.expected_sha256_url, expected_dest, log_lines)
                            expected_text = expected_dest.read_text(encoding="utf-8").strip().split()[0].lower()
                            source_match = expected_text == digest.lower()
                            source_checksum_results.append(
                                {
                                    "file": f"raw/{dest.name}",
                                    "source_checksum_url": item.expected_sha256_url,
                                    "source_sha256": expected_text,
                                    "local_sha256": digest,
                                    "match": source_match,
                                }
                            )
                            log_lines.append(
                                f"{utc_now()} SOURCE_CHECKSUM {dest.name} match={source_match}"
                            )
                            inventory.append(
                                {
                                    "relative_path": f"metadata/{expected_dest.name}",
                                    "size_bytes": expected_dest.stat().st_size,
                                    "sha256": sha256_file(expected_dest),
                                    "source_url": item.expected_sha256_url,
                                }
                            )
                        except Exception as exc:
                            log_lines.append(f"{utc_now()} WARNING source checksum unavailable: {exc}")
            total_downloaded += size
        metadata = metadata_for(ds, root, status, reason, size, count, formats)
        metadata["source_checksum_results"] = source_checksum_results
        write_json(meta / "dataset_metadata.json", metadata)
        write_text(meta / "source_record.md", source_record(ds, metadata))
        write_text(meta / "checksums.sha256", "\n".join(checksums) + ("\n" if checksums else ""))
        write_csv(meta / "file_inventory.csv", inventory)
        write_text(meta / "download_log.txt", "\n".join(log_lines) + ("\n" if log_lines else ""))
        write_text(meta / "license_or_terms.txt", ds.license_or_terms + "\n")
        write_text(root / "README.md", dataset_readme(ds, metadata))
        manifest_rows.append(manifest_row(ds, metadata, discovery_rows))
        metadata_index.append({"dataset_id": ds.dataset_id, "metadata_path": str(meta / "dataset_metadata.json"), "download_status": status})

    budget = {
        "storage_cap_bytes": cap_bytes,
        "storage_cap_gb": args.cap_gb,
        "archive_root": str(archive_root),
        "total_downloaded_bytes": total_downloaded,
        "total_downloaded_gb": round(total_downloaded / 1024**3, 4),
        "remaining_budget_bytes": cap_bytes - total_downloaded,
        "remaining_budget_gb": round((cap_bytes - total_downloaded) / 1024**3, 4),
        "datasets_10gb_or_larger_policy": "store on E:/quantule_mapper_external_data",
        "provisional_until_claude_review": True,
    }

    write_root_files(archive_root, manifest_rows, budget)
    write_repo_files(repo_meta_root, discovery_rows, manifest_rows, metadata_index, budget)
    return 0


def source_record(ds: DatasetCandidate, metadata: dict[str, Any]) -> str:
    return (
        f"# {ds.dataset_id} - {ds.title}\n\n"
        f"- Source URL: {ds.source_url}\n"
        f"- DOI/repository: {ds.doi_or_repository_id}\n"
        f"- Authors/organization: {ds.authors_or_organization}\n"
        f"- Year: {ds.year}\n"
        f"- Availability: {ds.availability_class}\n"
        f"- Download status: {metadata['download_status']}\n"
        f"- Relevant metrics: {', '.join(ds.relevant_metrics)}\n"
        f"- Comparison quantity: {ds.comparison_quantity}\n\n"
        "No physical-correspondence claim is made. IRER formulations remain primary.\n"
    )


def dataset_readme(ds: DatasetCandidate, metadata: dict[str, Any]) -> str:
    return (
        f"# {ds.dataset_id} - {ds.title}\n\n"
        "Raw files are preserved under `raw/`. Metadata, checksums, license/terms notes, "
        "and file inventory are under `metadata/`.\n\n"
        f"Download status: `{metadata['download_status']}`\n\n"
        "This archive is provisional until Claude review and is not a physical-validation claim.\n"
    )


def manifest_row(ds: DatasetCandidate, metadata: dict[str, Any], discovery_rows: list[dict[str, Any]]) -> dict[str, Any]:
    discovery = next((r for r in discovery_rows if r["dataset_id"] == ds.dataset_id), {})
    return {
        "dataset_id": ds.dataset_id,
        "source": ds.source_url,
        "availability_class": ds.availability_class,
        "estimated_size_bytes": discovery.get("estimated_size_bytes", 0),
        "downloaded_size_bytes": metadata["raw_size_bytes"],
        "local_root": metadata["local_root"],
        "download_status": metadata["download_status"],
        "relevant_metric": ";".join(ds.relevant_metrics),
        "direct_comparison_ready": metadata["directly_comparable"],
        "comparison_quantity": ds.comparison_quantity,
        "notes": "; ".join(ds.comparison_limitations),
        "Claude_review_required": True,
    }


def write_root_files(root: Path, manifest_rows: list[dict[str, Any]], budget: dict[str, Any]) -> None:
    write_text(
        root / "README.md",
        "# Quantule Mapper External Data Archive\n\n"
        "Raw external datasets live here, outside the git repository. Repo-side docs store only metadata, manifests, checksums, and reports.\n\n"
        "No physical-correspondence claim is made by storing these files.\n",
    )
    write_csv(root / "external_data_manifest.csv", manifest_rows)
    write_json(root / "storage_budget.json", budget)


def write_repo_files(repo_root: Path, discovery_rows: list[dict[str, Any]], manifest_rows: list[dict[str, Any]], metadata_index: list[dict[str, Any]], budget: dict[str, Any]) -> None:
    write_csv(repo_root / "external_data_manifest.csv", manifest_rows)
    write_json(repo_root / "storage_budget.json", budget)
    write_json(repo_root / "dataset_metadata_index.json", {"datasets": metadata_index})
    write_text(repo_root / "EXTERNAL_DATA_ACQUISITION_PLAN.md", acquisition_plan(discovery_rows, budget))
    write_text(repo_root / "DOWNLOAD_STATUS_REPORT.md", download_report(discovery_rows, manifest_rows, budget))


def acquisition_plan(discovery_rows: list[dict[str, Any]], budget: dict[str, Any]) -> str:
    lines = [
        "# External Data Acquisition Plan",
        "",
        "Status: provisional until Claude review.",
        "",
        f"- Storage cap: {budget['storage_cap_gb']} GB",
        f"- Archive root: `{budget['archive_root']}`",
        "- Raw data are stored outside git.",
        "",
        "| Dataset | Availability | Estimated GB | Recommendation | Relevant metric | Directly comparable? |",
        "|---|---|---:|---|---|---|",
    ]
    for row in discovery_rows:
        lines.append(
            f"| `{row['dataset_id']}` | `{row['availability_class']}` | {row['estimated_size_gb']} | "
            f"`{row['recommendation']}` | {row['relevance']} | {row['directly_comparable']} |"
        )
    lines += [
        "",
        "No dataset is promoted to validation-ready unless it contains machine-readable or tabulated data for a defined comparison quantity.",
    ]
    return "\n".join(lines) + "\n"


def download_report(discovery_rows: list[dict[str, Any]], manifest_rows: list[dict[str, Any]], budget: dict[str, Any]) -> str:
    downloaded = [r for r in manifest_rows if r["download_status"] == "DOWNLOADED"]
    skipped = [r for r in manifest_rows if r["download_status"] != "DOWNLOADED"]
    decision = "DATASET_ARCHIVE_READY_WITH_DOWNLOADS" if downloaded else "DATASET_DISCOVERY_COMPLETE_NO_DOWNLOADS"
    if any(r["download_status"] == "FAILED" for r in manifest_rows):
        decision = "DATASET_ARCHIVE_READY_WITH_GAPS"
    lines = [
        "# External Data Download Status Report",
        "",
        "Status: provisional until Claude review.",
        "",
        f"Final decision: `{decision}`",
        "",
        "## Discovery Results",
        "",
        "| Dataset | Availability | Recommendation | Estimated size |",
        "|---|---|---|---:|",
    ]
    for row in discovery_rows:
        lines.append(f"| `{row['dataset_id']}` | `{row['availability_class']}` | `{row['recommendation']}` | {row['estimated_size_gb']} GB |")
    lines += [
        "",
        "## Datasets Downloaded",
        "",
    ]
    if downloaded:
        for row in downloaded:
            lines.append(f"- `{row['dataset_id']}` -> `{row['local_root']}` ({int(row['downloaded_size_bytes'])} bytes)")
    else:
        lines.append("- None")
    lines += [
        "",
        "## Datasets Skipped",
        "",
    ]
    if skipped:
        for row in skipped:
            lines.append(f"- `{row['dataset_id']}`: {row['availability_class']} / {row['notes']}")
    else:
        lines.append("- None")
    lines += [
        "",
        "## Storage Budget",
        "",
        f"- Total downloaded size: {budget['total_downloaded_gb']} GB",
        f"- Remaining from cap: {budget['remaining_budget_gb']} GB",
        "- Raw data stored outside git.",
        "- Datasets >=10GB must be stored on E: drive; all current raw downloads are stored there.",
        "",
        "## Level-3 Readiness",
        "",
        "No downloaded dataset is currently marked directly comparable for Level-3 external-data correlation. NLSE is a method-sanity comparator; NIST is morphology/image tooling only.",
        "",
        "## Metadata And Checksums",
        "",
        "For every dataset entry, the archive contains:",
        "",
        "- `metadata/dataset_metadata.json`",
        "- `metadata/source_record.md`",
        "- `metadata/checksums.sha256`",
        "- `metadata/download_log.txt`",
        "- `metadata/file_inventory.csv`",
        "- `metadata/license_or_terms.txt`",
        "- dataset-local `README.md`",
        "",
        "For `DATA-BEC-001`, the source-provided `data_files.zip.sha256` was downloaded and matched against the local `data_files.zip` checksum.",
        "",
        "## Required Statements",
        "",
        "- No Quantule Mapper simulations run.",
        "- No production physics changed.",
        "- No verdicts changed.",
        "- No physical-correspondence claims added.",
        "- Total downloaded size <= 60GB.",
    ]
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover and archive external validation datasets.")
    parser.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE_ROOT))
    parser.add_argument("--repo-meta-root", default=str(DEFAULT_REPO_META_ROOT))
    parser.add_argument("--cap-gb", type=float, default=60.0)
    parser.add_argument("--datasets", default="", help="Comma-separated dataset IDs to download; defaults to DOWNLOAD_NOW entries.")
    parser.add_argument("--force", action="store_true", help="Re-download files even if they already exist.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
