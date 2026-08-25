---
run_id: "TG_B1S_STATE_LOAD_GPU_20260714_135157"
date: 2026-07-14
family: "TG-B1S"
sector: "gravity"
verdict: "TG_STATE_LOAD_GEOMETRIC_BACKREACTION_DETECTED"
complete: false
source: derived
n_csv: 22
n_plots: 5
tags: [run, gravity, TG_B1S]
---
# TG_B1S_STATE_LOAD_GPU_20260714_135157

*Dual-substrate single-node / static baseline* &middot; **TG-B1S** &middot; `2026-07-14`

> [!abstract] Verdict
> `TG_STATE_LOAD_GEOMETRIC_BACKREACTION_DETECTED`

> [!info] Derived note - no `summary.json`
> This run predates or bypasses the `summary.json` convention. Fields below are reconstructed from: `TECHNICAL_HANDOFF.md`, `OPEN_QUESTIONS.md`, `source_spec.json`.
> The verdict shown is `RUN_COMPLETE.json.status` where present, otherwise the first verdict-shaped token found in the run's own report - **treat it as indicative and confirm against the source document.**

## From `TECHNICAL_HANDOFF.md`

Status: `TG_STATE_LOAD_GEOMETRIC_BACKREACTION_DETECTED`.
Run directory: `/mnt/f/quantule_mapper/sweep_runs/TG_B1S_STATE_LOAD_GPU_20260714_135157`.


- `TG_STATE_LOAD_FEEDFORWARD_CHAIN_SUPPORTED`
- `TG_STATE_LOAD_GEOMETRIC_BACKREACTION_DETECTED`


Do not treat this as gravity, photon emission, objective time dilation, geodesic dynamics, universal free fall, or IRER validation.

## From `OPEN_QUESTIONS.md`

- Should the accepted state-load feedback be rerun for longer to test bounded stability?
- Should TG-B1R now test phase-tension relaxation separately?

## Summary fields

| field | value |
|---|---|
| `S0_authoritative_TG_S` | 135.686219 |
| `casewise_normalization` | `false` |
| `enabled_source` | S_state |
| `global_normalization` | S_hat = local_definition / S0 |
| `local_definition` | 0.5*max(E_phi_density,0)/E_ref_max + 0.5*abs(charge_density)/Q_ref_max |
| `reference_charge_max` | 0.780786 |
| `reference_energy_max` | 1.399532 |
| `reference_spatial_integral` | 67.843111 |
| `source_global_norm_S0` | 135.686219 |
| `source_hypothesis` | STATE_LOAD_FEEDBACK |

## Plots

*Generated from this run's CSVs by `tools/build_run_catalogue.py`. Regenerable — delete and rebuild to refresh.*

![[_plots/TG_B1S_STATE_LOAD_GPU_20260714_135157/S0_baseline_trajectory.png]]

*S0 baseline trajectory*

![[_plots/TG_B1S_STATE_LOAD_GPU_20260714_135157/S1_temporal_trajectory.png]]

*S1 temporal trajectory*

![[_plots/TG_B1S_STATE_LOAD_GPU_20260714_135157/S2_feed_forward_trajectory.png]]

*S2 feed forward trajectory*

![[_plots/TG_B1S_STATE_LOAD_GPU_20260714_135157/S3_feedback_off_trajectory.png]]

*S3 feedback off trajectory*

![[_plots/TG_B1S_STATE_LOAD_GPU_20260714_135157/S4_full_loop_trajectory.png]]

*S4 full loop trajectory*

## Artifacts

- **Run directory** (gitignored, local only): `sweep_runs/TG_B1S_STATE_LOAD_GPU_20260714_135157/`
- **CSV data** (22): `S0_baseline_trajectory.csv`, `S1_temporal_trajectory.csv`, `S2_feed_forward_trajectory.csv`, `S3_feedback_off_trajectory.csv`, `S4_full_loop_trajectory.csv`, `S5_source_off_trajectory.csv`, `S6_temporal_off_trajectory.csv`, `S7_geometric_off_trajectory.csv`, `S8_global_phase_trajectory.csv`, `S9_translated_trajectory.csv` ...

## Previous experiments

*Auto-derived: the preceding runs in the same family (`TG-B1S`). Correct by hand if the lineage is wrong — edits here survive rebuilds only if you move the link below the Review notes marker.*

- [[TG_B1S_LONG_TIME_GPU_20260714_154323]] &middot; `2026-07-14`
- [[TG_B1S_LONG_TIME_GPU_20260714_152138]] &middot; `2026-07-14`
- [[TG_B1S_DRIFT_DECOMPOSITION_GPU_20260714_184736]] &middot; `2026-07-14`

## Associated docs

- [[Branch - Gravity - Index]]
- [[EXPERIMENT_TRACKER]]

## Next experiment

- [[TG_B1S_STATE_LOAD_GPU_20260714_140157]] &middot; `2026-07-14`

## Branches

- [[Main branch]] &larr; via [[Branch - Gravity - Index|gravity sector index]]
- Sector: `gravity` &middot; family: `TG-B1S`

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

