---
run_id: "GRAVITY_C_SERIES_PROVENANCE_20260714_092539"
date: 2026-07-14
family: "Gravity-D"
sector: "gravity"
verdict: "C1_FROZEN_TEMPORAL_RESPONSE_REPRODUCED"
complete: false
source: derived
n_csv: 2
n_plots: 1
tags: [run, gravity, Gravity_D]
---
# GRAVITY_C_SERIES_PROVENANCE_20260714_092539

*Spatial-geometry gravity mirror (non-Newtonian)* &middot; **Gravity-D** &middot; `2026-07-14`

> [!abstract] Verdict
> `C1_FROZEN_TEMPORAL_RESPONSE_REPRODUCED`

> [!info] Derived note - no `summary.json`
> This run predates or bypasses the `summary.json` convention. Fields below are reconstructed from: `TECHNICAL_HANDOFF.md`, `POST_REVIEW_CORRECTION.md`.
> The verdict shown is `RUN_COMPLETE.json.status` where present, otherwise the first verdict-shaped token found in the run's own report - **treat it as indicative and confirm against the source document.**

## From `TECHNICAL_HANDOFF.md`

This wrapper reran Claude's C.1-C.3 scripts without changing their implementations.


- `C1_FROZEN_TEMPORAL_RESPONSE_REPRODUCED`
- `C2_SOURCE_MAP_REPRODUCED`
- `C3_BACKREACTION_REPRODUCED`

**Boundary**

These are temporal-throttling and source-map mirror results. They do not validate gravity, geodesics, equivalence principle behaviour, or production geometry.

## From `POST_REVIEW_CORRECTION.md`

C.1 is interpreted as shared frozen-lapse clock consistency, not source-layer objective-lapse emergence. C.3 has a reporting-condition bug in `freq_monotone_with_lam`; the frequency rows increase monotonically from the frozen value toward the self-relieved value.

## Plots

*Generated from this run's CSVs by `tools/build_run_catalogue.py`. Regenerable — delete and rebuild to refresh.*

![[_plots/GRAVITY_C_SERIES_PROVENANCE_20260714_092539/command_manifest.png]]

*command manifest*

## Artifacts

- **Run directory** (gitignored, local only): `sweep_runs/GRAVITY_C_SERIES_PROVENANCE_20260714_092539/`
- **CSV data** (2): `artifact_hashes.csv`, `command_manifest.csv`

## Previous experiments

*Auto-derived: the preceding runs in the same family (`Gravity-D`). Correct by hand if the lineage is wrong — edits here survive rebuilds only if you move the link below the Review notes marker.*

- [[GRAVITY_C3_BACKREACT_20260714_092551]] &middot; `2026-07-14`
- [[GRAVITY_D_ROBUSTNESS_GPU_20260713_195713]] &middot; `2026-07-13`
- [[GRAVITY_D_ROBUSTNESS_CLOSURE_GPU_20260713_204559]] &middot; `2026-07-13`

## Associated docs

- [[Branch - Gravity - Index]]
- [[EXPERIMENT_TRACKER]]
- [[GRAVITY_TS_TEMPORAL_SPATIAL_BRIDGE_AUDIT]]
- [[external_validation/GRAVITY_TS_BRIDGE_EXTERNAL_VALIDATION_PACKAGE]]

## Next experiment

- [[GRAVITY_MATURITY_G1_CLOCK_20260714_100352]] &middot; `2026-07-14`

## Branches

- [[Main branch]] &larr; via [[Branch - Gravity - Index|gravity sector index]]
- Sector: `gravity` &middot; family: `Gravity-D`

---

## Review notes

*(Preserved across catalogue rebuilds - everything above this line is regenerated.)*

> [!important] Paired reading - fill your block before reading the other one.
> A disagreement here is the point, not a problem. Record the resolution; do not
> overwrite the disagreement.

### Reading - Claude

- **Observation:**
- **Reading:**
- **Confidence:**
- **What would change my mind:**

### Reading - Jake

- **Observation:**
- **Reading:**
- **Confidence:**
- **What would change my mind:**

### Comparison

**Agree on:**

**Disagree on:**

**Resolution:**

