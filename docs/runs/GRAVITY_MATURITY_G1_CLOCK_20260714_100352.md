---
run_id: "GRAVITY_MATURITY_G1_CLOCK_20260714_100352"
date: 2026-07-14
family: "Gravity-D"
sector: "gravity"
verdict: "TEMPORAL_CLOCK_CALIBRATION_FAILED"
complete: false
source: derived
n_csv: 6
n_plots: 4
tags: [run, gravity, Gravity_D]
---
# GRAVITY_MATURITY_G1_CLOCK_20260714_100352

*Spatial-geometry gravity mirror (non-Newtonian)* &middot; **Gravity-D** &middot; `2026-07-14`

> [!abstract] Verdict
> `TEMPORAL_CLOCK_CALIBRATION_FAILED`

> [!info] Derived note - no `summary.json`
> This run predates or bypasses the `summary.json` convention. Fields below are reconstructed from: `TECHNICAL_HANDOFF.md`, `OPEN_QUESTIONS.md`, `metrics.json`.
> The verdict shown is `RUN_COMPLETE.json.status` where present, otherwise the first verdict-shaped token found in the run's own report - **treat it as indicative and confirm against the source document.**

## From `TECHNICAL_HANDOFF.md`

This run tests the eigenfrequency contract for supplied objective temporal fields.


- `TEMPORAL_CLOCK_CALIBRATION_FAILED`


| test | passed | measured |
| --- | --- | --- |
| `eigen_time_contract_core` | False | `0.010632168830970514` |
| `near_far_shift_detected` | True | `{"far": 6.466264160562619, "near": 6.436298238072359}` |
| `near_flat_shift_detected` | True | `{"flat": 12.03422898751764, "near": 6.436298238072359}` |
| `measured_amplitude_invariance` | True | `8.881784197001252e-16` |
| `two_clock_family_shift_same_direction` | False | `{"eigen_far_minus_near": 0.029965922490259977, "packet_far_minus_near": -0.015835496489161827}` |

**Boundary**

This is G1 clock calibration only. It does not implement a source-to-geometry law, combined metric dynamics, or a gravity verdict.

## From `OPEN_QUESTIONS.md`

- Does the calibrated local rate approach `m N_t(R)` under a stronger localized/high-mass limit?
- Which clock family provides the cleanest external rerun benchmark?
- Should G2 use `a_s` directly or test mappings from the previously validated spatial coefficient `A_s`?

## Plots

*Generated from this run's CSVs by `tools/build_run_catalogue.py`. Regenerable — delete and rebuild to refresh.*

![[_plots/GRAVITY_MATURITY_G1_CLOCK_20260714_100352/clock_family_comparison.png]]

*clock family comparison*

![[_plots/GRAVITY_MATURITY_G1_CLOCK_20260714_100352/eigenfrequency_metrics.png]]

*eigenfrequency metrics*

![[_plots/GRAVITY_MATURITY_G1_CLOCK_20260714_100352/numerical_refinement.png]]

*numerical refinement*

![[_plots/GRAVITY_MATURITY_G1_CLOCK_20260714_100352/time_domain_clock_metrics.png]]

*time domain clock metrics*

## Artifacts

- **Run directory** (gitignored, local only): `sweep_runs/GRAVITY_MATURITY_G1_CLOCK_20260714_100352/`
- **CSV data** (6): `artifact_hashes.csv`, `clock_family_comparison.csv`, `eigenfrequency_metrics.csv`, `falsification_results.csv`, `numerical_refinement.csv`, `time_domain_clock_metrics.csv`

## Previous experiments

*Auto-derived: the preceding runs in the same family (`Gravity-D`). Correct by hand if the lineage is wrong — edits here survive rebuilds only if you move the link below the Review notes marker.*

- [[GRAVITY_C_SERIES_PROVENANCE_20260714_092539]] &middot; `2026-07-14`
- [[GRAVITY_C3_BACKREACT_20260714_092551]] &middot; `2026-07-14`
- [[GRAVITY_D_ROBUSTNESS_GPU_20260713_195713]] &middot; `2026-07-13`

## Associated docs

- [[Branch - Gravity - Index]]
- [[EXPERIMENT_TRACKER]]

## Next experiment

- [[GRAVITY_MATURITY_G1_CLOCK_20260714_100527]] &middot; `2026-07-14`

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

