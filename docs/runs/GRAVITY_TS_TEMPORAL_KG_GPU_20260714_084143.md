---
run_id: "GRAVITY_TS_TEMPORAL_KG_GPU_20260714_084143"
date: 2026-07-14
family: "Gravity-D"
sector: "gravity"
verdict: "TEMPORAL_KG_LAPSE_EQUATION_IMPLEMENTED"
complete: false
source: derived
n_csv: 3
n_plots: 1
tags: [run, gravity, Gravity_D]
---
# GRAVITY_TS_TEMPORAL_KG_GPU_20260714_084143

*Spatial-geometry gravity mirror (non-Newtonian)* &middot; **Gravity-D** &middot; `2026-07-14`

> [!abstract] Verdict
> `TEMPORAL_KG_LAPSE_EQUATION_IMPLEMENTED`

> [!info] Derived note - no `summary.json`
> This run predates or bypasses the `summary.json` convention. Fields below are reconstructed from: `TECHNICAL_HANDOFF.md`, `POST_REVIEW_CORRECTION.md`.
> The verdict shown is `RUN_COMPLETE.json.status` where present, otherwise the first verdict-shaped token found in the run's own report - **treat it as indicative and confirm against the source document.**

## From `TECHNICAL_HANDOFF.md`

This audit evolves a true static-lapse KG equation rather than only measuring a clock factor.


- `TEMPORAL_KG_LAPSE_EQUATION_IMPLEMENTED`
- `OBJECTIVE_TEMPORAL_LAPSE_KG_SCOUT_SUPPORTED`
- `RELATIONAL_TEMPORAL_SOURCE_WEAK_PROBE_WARNING_REPRODUCED`

**Boundary**

This is a temporal-equation mirror scout. It does not validate gravity, geodesics, equivalence principle behaviour, or production geometry.

## From `POST_REVIEW_CORRECTION.md`

The static-lapse KG equation is implemented and produces a measured near/far frequency differential. The constructed weighted lapse is not yet validated as the measured packet clock rate; temporal clock calibration remains open. Relational weak-probe collapse is established at the constructed field level, while measured KG rates are nonmonotonic.

## Plots

*Generated from this run's CSVs by `tools/build_run_catalogue.py`. Regenerable — delete and rebuild to refresh.*

![[_plots/GRAVITY_TS_TEMPORAL_KG_GPU_20260714_084143/kg_clock_metrics.png]]

*kg clock metrics*

## Artifacts

- **Run directory** (gitignored, local only): `sweep_runs/GRAVITY_TS_TEMPORAL_KG_GPU_20260714_084143/`
- **CSV data** (3): `artifact_hashes.csv`, `falsification_results.csv`, `kg_clock_metrics.csv`

## Previous experiments

*Auto-derived: the preceding runs in the same family (`Gravity-D`). Correct by hand if the lineage is wrong — edits here survive rebuilds only if you move the link below the Review notes marker.*

- [[GRAVITY_TS_BRIDGE_REFINEMENT_GPU_20260714_085110]] &middot; `2026-07-14`
- [[GRAVITY_TS_BRIDGE_GPU_20260714_065422]] &middot; `2026-07-14`
- [[GRAVITY_MATURITY_G1_CLOCK_20260714_100527]] &middot; `2026-07-14`

## Associated docs

- [[Branch - Gravity - Index]]
- [[EXPERIMENT_TRACKER]]
- [[GRAVITY_TS_TEMPORAL_SPATIAL_BRIDGE_AUDIT]]
- [[external_validation/GRAVITY_TS_BRIDGE_EXTERNAL_VALIDATION_PACKAGE]]

## Next experiment

*None yet — this is the most recent run in the `Gravity-D` family.*

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

