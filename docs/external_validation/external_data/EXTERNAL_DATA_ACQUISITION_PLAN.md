# External Data Acquisition Plan

Status: provisional until Claude review.

- Storage cap: 60.0 GB
- Archive root: `E:\quantule_mapper_external_data`
- Raw data are stored outside git.

| Dataset | Availability | Estimated GB | Recommendation | Relevant metric | Directly comparable? |
|---|---|---:|---|---|---|
| `DATA-NUM-001` | `DIRECT_DOWNLOAD_READY` | 0.0527 | `DOWNLOAD_NOW` | V2 | False |
| `DATA-BEC-001` | `DIRECT_DOWNLOAD_READY` | 4.0328 | `DOWNLOAD_NOW` | future morphology only | False |
| `DATA-OPT-006` | `DIGITIZATION_REQUIRED` | 0.0 | `PLAN_ONLY` | V1 | False |
| `DATA-BEC-005` | `DIGITIZATION_REQUIRED` | 0.0 | `PLAN_ONLY` | V5 | False |
| `DATA-QB-004` | `SOURCE_TRAIL_ONLY` | 0.0 | `MANUAL_REVIEW_REQUIRED` | V3;V5 | False |
| `DATA-AG-003` | `DIGITIZATION_REQUIRED` | 0.0 | `PLAN_ONLY` | V7 | False |

No dataset is promoted to validation-ready unless it contains machine-readable or tabulated data for a defined comparison quantity.
