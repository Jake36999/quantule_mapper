---
run_id: "GRAVITY_TS_BRIDGE_REFINEMENT_GPU_20260714_085110"
date: 2026-07-14
family: "Gravity-D"
sector: "gravity"
verdict: "TEMPORAL_THROTTLING_REPRODUCED"
complete: false
source: derived
n_csv: 8
n_plots: 3
tags: [run, gravity, Gravity_D]
---
# GRAVITY_TS_BRIDGE_REFINEMENT_GPU_20260714_085110

*Spatial-geometry gravity mirror (non-Newtonian)* &middot; **Gravity-D** &middot; `2026-07-14`

> [!abstract] Verdict
> `TEMPORAL_THROTTLING_REPRODUCED`

> [!info] Derived note - no `summary.json`
> This run predates or bypasses the `summary.json` convention. Fields below are reconstructed from: `TECHNICAL_HANDOFF.md`, `DOCUMENTATION_INPUTS.md`, `BRIDGE_SUMMARY.json`.
> The verdict shown is `RUN_COMPLETE.json.status` where present, otherwise the first verdict-shaped token found in the run's own report - **treat it as indicative and confirm against the source document.**

## From `TECHNICAL_HANDOFF.md`

This mirror audit separates Claude's temporal clock factor `N_t` from the Gravity D spatial coefficient `A_s`.
Field evolution used the validated GPU spatial operator `H=-D div(A_s grad)`. Temporal factors were recorded as clock observables, not silently inserted as a metric equation.


- `TEMPORAL_THROTTLING_REPRODUCED`
- `SPATIAL_MEDIUM_FORCE_REPRODUCED`
- `TEMPORAL_SPATIAL_DECOMPOSITION_PASSED`


| source | duration | flat force | temporal clock | spatial force | combined force delta |
| --- | --- | ---: | ---: | ---: | ---: |
| objective | refined_T | 0.000000e+00 | 4.486118e-03 | 6.211596e-03 | 0.000000e+00 |
| relational | refined_T | 0.000000e+00 | 1.676835e-02 | 4.308223e-03 | 0.000000e+00 |


| test | passed | measured value |
| --- | --- | --- |
| `flat_clean_objective_refined_T` | True | `{"clock_shift": 0.0, "force_norm": 0.0}` |
| `temporal_only_objective_refined_T` | True |

**Boundary**

No gravity, geodesic, equivalence-principle, IRER-gravity-validation, or production-readiness label is assigned.

## From `DOCUMENTATION_INPUTS.md`

- `TEMPORAL_THROTTLING_REPRODUCED`
- `SPATIAL_MEDIUM_FORCE_REPRODUCED`
- `TEMPORAL_SPATIAL_DECOMPOSITION_PASSED`


The bridge audit reproduces temporal clock throttling and spatial effective-medium force as separable mirror mechanisms. The objective source shows a weak-probe clock limit at matched position, while the relational mutual source weakens toward no temporal field in the raw weak-probe limit, keeping it unsuitable as a universal gravity source.


- `N_t` was measured as a clock factor; this is not a metric-consistent KG or scalar-field equation.
- `A_s` remains the spatial kinetic coefficient in the validated divergence-form operator.
- `N_t=A_s` is only a restricted common-field check.


- `run_manifest.csv`
- `clock_metrics.csv`
- `force_metrics.csv`
- `arm_comparison.csv`
- `weak_probe_limit.csv`
- `falsification_results.csv`

## Summary fields

| field | value |
|---|---|
| `bounded_interpretation` | Temporal throttling and spatial effective-medium response are separable mirror observab... |

## Plots

*Generated from this run's CSVs by `tools/build_run_catalogue.py`. Regenerable — delete and rebuild to refresh.*

![[_plots/GRAVITY_TS_BRIDGE_REFINEMENT_GPU_20260714_085110/clock_metrics.png]]

*clock metrics*

![[_plots/GRAVITY_TS_BRIDGE_REFINEMENT_GPU_20260714_085110/force_metrics.png]]

*force metrics*

![[_plots/GRAVITY_TS_BRIDGE_REFINEMENT_GPU_20260714_085110/trajectory_metrics.png]]

*trajectory metrics*

## Artifacts

- **Run directory** (gitignored, local only): `sweep_runs/GRAVITY_TS_BRIDGE_REFINEMENT_GPU_20260714_085110/`
- **CSV data** (8): `arm_comparison.csv`, `artifact_hashes.csv`, `clock_metrics.csv`, `falsification_results.csv`, `force_metrics.csv`, `run_manifest.csv`, `source_ladder_metadata.csv`, `trajectory_metrics.csv`

## Previous experiments

*Auto-derived: the preceding runs in the same family (`Gravity-D`). Correct by hand if the lineage is wrong — edits here survive rebuilds only if you move the link below the Review notes marker.*

- [[GRAVITY_TS_BRIDGE_GPU_20260714_065422]] &middot; `2026-07-14`
- [[GRAVITY_MATURITY_G1_CLOCK_20260714_100527]] &middot; `2026-07-14`
- [[GRAVITY_MATURITY_G1_CLOCK_20260714_100352]] &middot; `2026-07-14`

## Associated docs

- [[Branch - Gravity - Index]]
- [[EXPERIMENT_TRACKER]]
- [[GRAVITY_TS_TEMPORAL_SPATIAL_BRIDGE_AUDIT]]
- [[external_validation/GRAVITY_TS_BRIDGE_EXTERNAL_VALIDATION_PACKAGE]]

## Next experiment

- [[GRAVITY_TS_TEMPORAL_KG_GPU_20260714_084143]] &middot; `2026-07-14`

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

