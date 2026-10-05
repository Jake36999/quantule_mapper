---
run_id: "TG_B1S_DRIFT_DECOMPOSITION_GPU_20260714_184736"
date: 2026-07-14
family: "TG-B1S"
sector: "gravity"
verdict: "TG_STATE_LOAD_LONG_TIME_DRIFT_UNRESOLVED"
complete: false
source: derived
n_csv: 11
n_plots: 3
tags: [run, gravity, TG_B1S]
---
# TG_B1S_DRIFT_DECOMPOSITION_GPU_20260714_184736

*Dual-substrate single-node / static baseline* &middot; **TG-B1S** &middot; `2026-07-14`

> [!abstract] Verdict
> `TG_STATE_LOAD_LONG_TIME_DRIFT_UNRESOLVED`

> [!info] Derived note - no `summary.json`
> This run predates or bypasses the `summary.json` convention. Fields below are reconstructed from: `TECHNICAL_HANDOFF.md`, `source_spec.json`, `preregistered_matrix.json`.
> The verdict shown is `RUN_COMPLETE.json.status` where present, otherwise the first verdict-shaped token found in the run's own report - **treat it as indicative and confirm against the source document.**

## From `TECHNICAL_HANDOFF.md`

Status: `TG_STATE_LOAD_LONG_TIME_DRIFT_UNRESOLVED`.
Run directory: `/mnt/f/quantule_mapper/sweep_runs/TG_B1S_DRIFT_DECOMPOSITION_GPU_20260714_184736`.


- `TG_STATE_LOAD_LONG_TIME_DRIFT_UNRESOLVED`


Frozen state-load model only; no R_relax, L_lock, P_threshold, photon, radiative, gravity, objective-time, geodesic or production claim.


The D0-D3 primary run separates phase drift from profile drift. Through 100 periods, `delta_theta` is linear-secular, while phase-aligned structural channels are bounded/oscillatory or below numerical floor.

Measured primary classification: `STABLE_FREQUENCY_SHIFT`.

Formal status remains `TG_STATE_LOAD_LONG_TIME_DRIFT_UNRESOLVED` because D4 numerical validation and D5 basin-orbital stability were not run in this primary matrix.

Do not promote `TG_STATE_LOAD_ORBITALLY_BOUNDED_FREQUENCY_SHIFT_SUPPORTED` or `TG_STATE_LOAD_BOUNDED_FEEDBACK_SUPPORTED` until

## Summary fields

| field | value |
|---|---|
| `S0_authoritative_TG_S` | 135.686219 |
| `casewise_normalization` | `false` |
| `enabled_source` | S_state |
| `source_hypothesis` | STATE_LOAD_FEEDBACK |
| `D0_phase_aligned_profile_comparison` | `true` |
| `D1_separate_drift_channels` | `true` |
| `D2_asymptotic_frequency_contract` | `true` |
| `D3_100_period_run` | `true` |
| `D4_numerical_validation` | `false` |
| `D5_basin_test` | `false` |
| `matrix` | primary |
| `state_load_only` | `true` |

## Plots

*Generated from this run's CSVs by `tools/build_run_catalogue.py`. Regenerable — delete and rebuild to refresh.*

![[_plots/TG_B1S_DRIFT_DECOMPOSITION_GPU_20260714_184736/drift_channel_fits.png]]

*drift channel fits*

![[_plots/TG_B1S_DRIFT_DECOMPOSITION_GPU_20260714_184736/orbital_distance.png]]

*orbital distance*

![[_plots/TG_B1S_DRIFT_DECOMPOSITION_GPU_20260714_184736/phase_alignment.png]]

*phase alignment*

## Artifacts

- **Run directory** (gitignored, local only): `sweep_runs/TG_B1S_DRIFT_DECOMPOSITION_GPU_20260714_184736/`
- **CSV data** (11): `artifact_hashes.csv`, `asymptotic_frequency.csv`, `baseline_node_contract.csv`, `basin_orbital_stability.csv`, `drift_channel_fits.csv`, `energy_ledger.csv`, `falsification_results.csv`, `long_time_100_period.csv`, `numerical_validation.csv`, `orbital_distance.csv` ...

## Previous experiments

*Auto-derived: the preceding runs in the same family (`TG-B1S`). Correct by hand if the lineage is wrong — edits here survive rebuilds only if you move the link below the Review notes marker.*

- [[TG_B1S_DRIFT_DECOMPOSITION_GPU_20260714_184247]] &middot; `2026-07-14`
- [[TG_B1S_BACKREACTION_ROBUSTNESS_GPU_20260714_142319]] &middot; `2026-07-14`

## Associated docs

- [[Branch - Gravity - Index]]
- [[EXPERIMENT_TRACKER]]
- [[RUN_QUEUE]]
- [[gravity_maturity/CLAUDE_HANDOFF_20260715_TG_B1S_AND_RUNTIME]]
- [[gravity_maturity/TG_B1S_D_DOCUMENTATION_INPUTS]]
- [[gravity_maturity/TG_B1S_D_RESULTS]]

## Next experiment

- [[TG_B1S_LONG_TIME_GPU_20260714_152138]] &middot; `2026-07-14`

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

