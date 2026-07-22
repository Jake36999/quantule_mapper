# External Dataset Ingestion Smoke Report

Status: provisional until Claude review.

Final decision: `EXTERNAL_DATA_INGESTION_READY_WITH_CANDIDATES`

## Summary

- Downloaded external datasets were inspected read-only.
- No Quantule Mapper simulations were run.
- No physical-correspondence claim is made.
- No Level-3 dynamics correlation is ready from the downloaded datasets alone.

## DATA-NUM-001 NLSE

- Adapter status: `INGESTED_SOURCE_ARCHIVE`
- Role: Reduced-NLS numerical-method sanity candidate only.
- Archive members: 373
- Uncompressed size: 83993322 bytes
- Follow-up: Run external NLSE package in an isolated environment on the same reduced flat-NLS boost identity used for C2 V2, then compare v=2Dk slope and mass/norm retention.

## DATA-BEC-001 NIST Dark Solitons

- Adapter status: `INGESTED_MORPHOLOGY_DATASET`
- Role: Morphology/image tooling candidate only; not C2/C3 bright-collision or Q-ball dynamics validation.
- Roster rows: 16478
- Label counts: `{'0': 1130, '1': 3212, '2': 1036, '8': 879, '9': 10221}`
- Train/test counts: 5617 / 640
- Sample NPY: `{'member': 'data_files/class-2/2019-07-15_0011_20190523_BEC_F1_NewODT_DMD_217.npy', 'load_status': 'SKIPPED_UNSAFE_OBJECT_ARRAY', 'reason': 'Object arrays cannot be loaded when allow_pickle=False'}`
- Project data gap: No project-side dark-soliton image/morphology dataset exists. Current Quantule Mapper external metrics are trajectory/field summaries, not labelled absorption images.

## Queued Follow-Ups

- `Q-EXT-NLSE-REDUCED-V2` (medium): Create an isolated adapter that runs the NLSE package on a reduced flat-NLS boost identity and compares v=2Dk against project V2 outputs.
- `Q-EXT-NIST-MORPHOLOGY-ADAPTER` (low): If desired, build a morphology-only reader/preview and node/dark-notch detector benchmark using NIST labels. Do not compare to C2/C3 bright collision dynamics.
- `Q-EXT-V1-DIGITIZE-MITSCHKE` (high for Level-3 V1): Digitise or obtain tabulated optical soliton force/separation data before Level-3 V1 correlation.
- `Q-EXT-V5-DIGITIZE-NGUYEN` (medium for Level-3 V5): Digitise paper plots or request author data before comparing C3 collision phase grid to BEC matter-wave observations.

## Guardrails

- NIST dark solitons are not treated as direct C2/C3 collision validation.
- NLSE package is not treated as empirical validation.
- IRER formulations remain primary.
- External datasets are comparison instruments only.
