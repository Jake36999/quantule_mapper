# External Data Download Status Report

Status: provisional until Claude review.

Final decision: `DATASET_ARCHIVE_READY_WITH_DOWNLOADS`

## Discovery Results

| Dataset | Availability | Recommendation | Estimated size |
|---|---|---|---:|
| `DATA-NUM-001` | `DIRECT_DOWNLOAD_READY` | `DOWNLOAD_NOW` | 0.0527 GB |
| `DATA-BEC-001` | `DIRECT_DOWNLOAD_READY` | `DOWNLOAD_NOW` | 4.0328 GB |
| `DATA-OPT-006` | `DIGITIZATION_REQUIRED` | `PLAN_ONLY` | 0.0 GB |
| `DATA-BEC-005` | `DIGITIZATION_REQUIRED` | `PLAN_ONLY` | 0.0 GB |
| `DATA-QB-004` | `SOURCE_TRAIL_ONLY` | `MANUAL_REVIEW_REQUIRED` | 0.0 GB |
| `DATA-AG-003` | `DIGITIZATION_REQUIRED` | `PLAN_ONLY` | 0.0 GB |

## Datasets Downloaded

- `DATA-NUM-001` -> `E:\quantule_mapper_external_data\DATA-NUM-001_nlse_package` (56628688 bytes)
- `DATA-BEC-001` -> `E:\quantule_mapper_external_data\DATA-BEC-001_nist_dark_solitons_v2` (4330179725 bytes)

## Datasets Skipped

- `DATA-OPT-006`: DIGITIZATION_REQUIRED / No machine-readable data located.; Paper figures/tables need review before use.
- `DATA-BEC-005`: DIGITIZATION_REQUIRED / No machine-readable trajectory table located.; Analogue only, not exact C2/C3 equation matching.
- `DATA-QB-004`: SOURCE_TRAIL_ONLY / No verified machine-readable repository located.
- `DATA-AG-003`: DIGITIZATION_REQUIRED / No machine-readable dispersion table located.; V7-adjacent only; not a gravity result.

## Storage Budget

- Total downloaded size: 4.0855 GB
- Remaining from cap: 55.9145 GB
- Raw data stored outside git.
- Datasets >=10GB must be stored on E: drive; all current raw downloads are stored there.

## Level-3 Readiness

No downloaded dataset is currently marked directly comparable for Level-3 external-data correlation. NLSE is a method-sanity comparator; NIST is morphology/image tooling only.

## Metadata And Checksums

For every dataset entry, the archive contains:

- `metadata/dataset_metadata.json`
- `metadata/source_record.md`
- `metadata/checksums.sha256`
- `metadata/download_log.txt`
- `metadata/file_inventory.csv`
- `metadata/license_or_terms.txt`
- dataset-local `README.md`

For `DATA-BEC-001`, the source-provided `data_files.zip.sha256` was downloaded and matched against the local `data_files.zip` checksum.

## Required Statements

- No Quantule Mapper simulations run.
- No production physics changed.
- No verdicts changed.
- No physical-correspondence claims added.
- Total downloaded size <= 60GB.
