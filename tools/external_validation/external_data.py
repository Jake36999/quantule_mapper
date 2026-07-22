from __future__ import annotations

from pathlib import Path

from .io_utils import write_csv, write_text


EXTERNAL_DATA_ROWS = [
    {
        "candidate": "JOSS NLSE package",
        "classification": "MACHINE_READABLE_NOW",
        "allowed_use": "reduced-NLS numerical sanity only",
        "comparison_status": "SKIPPED_WITH_REASON",
        "reason": "External package comparison is method sanity, not empirical validation; not run in this existing-data pass.",
    },
    {
        "candidate": "NIST dark soliton dataset",
        "classification": "MACHINE_READABLE_NOW",
        "allowed_use": "morphology/image tasks only",
        "comparison_status": "SKIPPED_WITH_REASON",
        "reason": "Dark-soliton image morphology is not directly comparable to current bright-collision/Q-ball dynamics metrics.",
    },
    {
        "candidate": "Nguyen/Hulet BEC bright soliton collisions",
        "classification": "DIGITIZATION_REQUIRED",
        "allowed_use": "future phase-dependent collision comparison",
        "comparison_status": "SKIPPED_WITH_REASON",
        "reason": "No machine-readable trajectory/outcome table is present locally.",
    },
    {
        "candidate": "Mitschke/Mollenauer optical soliton-force experiment",
        "classification": "DIGITIZATION_REQUIRED",
        "allowed_use": "future V1 force-law companion",
        "comparison_status": "SKIPPED_WITH_REASON",
        "reason": "No digitised separation/phase data are present locally.",
    },
    {
        "candidate": "Q-ball/oscillon repositories",
        "classification": "SOURCE_TRAIL_ONLY",
        "allowed_use": "future benchmark search",
        "comparison_status": "SKIPPED_WITH_REASON",
        "reason": "No verified repository or machine-readable benchmark was located in the registry.",
    },
    {
        "candidate": "Photon-fluid Bogoliubov dispersion",
        "classification": "DIGITIZATION_REQUIRED",
        "allowed_use": "future analogue-fluid density/dispersion comparison",
        "comparison_status": "SKIPPED_WITH_REASON",
        "reason": "No tabulated or machine-readable dispersion dataset is present locally.",
    },
]


def write_external_data_manifest(root: str | Path = "docs/external_validation/external_data") -> dict:
    out = Path(root)
    write_text(
        out / "README.md",
        "# External Data Readiness\n\n"
        "This folder records whether external machine-readable or digitised datasets are available for Level-3 comparison.\n\n"
        "No external empirical correlation is forced here. If data are not machine-readable or digitised, the comparison is skipped with reason.\n",
    )
    write_csv(out / "external_data_manifest.csv", EXTERNAL_DATA_ROWS)
    return {
        "status": "SKIPPED_WITH_REASON",
        "reason": "No direct external machine-readable dynamics dataset is currently ready for these metrics.",
        "rows": EXTERNAL_DATA_ROWS,
    }

