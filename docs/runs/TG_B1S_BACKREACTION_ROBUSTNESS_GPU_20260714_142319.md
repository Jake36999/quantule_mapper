---
run_id: "TG_B1S_BACKREACTION_ROBUSTNESS_GPU_20260714_142319"
date: 2026-07-14
family: "TG-B1S"
sector: "gravity"
verdict: "TG_STATE_LOAD_BACKREACTION_ROBUST_BUT_WEAK"
complete: false
source: derived
n_csv: 16
n_plots: 5
tags: [run, gravity, TG_B1S]
---
# TG_B1S_BACKREACTION_ROBUSTNESS_GPU_20260714_142319

*Dual-substrate single-node / static baseline* &middot; **TG-B1S** &middot; `2026-07-14`

> [!abstract] Verdict
> `TG_STATE_LOAD_BACKREACTION_ROBUST_BUT_WEAK`

> [!info] Derived note - no `summary.json`
> This run predates or bypasses the `summary.json` convention. Fields below are reconstructed from: `TECHNICAL_HANDOFF.md`, `OPEN_QUESTIONS.md`, `source_spec.json`, `preregistered_matrix.json`.
> The verdict shown is `RUN_COMPLETE.json.status` where present, otherwise the first verdict-shaped token found in the run's own report - **treat it as indicative and confirm against the source document.**

## From `TECHNICAL_HANDOFF.md`

Status: `TG_STATE_LOAD_BACKREACTION_ROBUST_BUT_WEAK`.
Run directory: `/mnt/f/quantule_mapper/sweep_runs/TG_B1S_BACKREACTION_ROBUSTNESS_GPU_20260714_142319`.


- `TG_STATE_LOAD_BACKREACTION_ROBUST`
- `TG_STATE_LOAD_BACKREACTION_ROBUST_BUT_WEAK`


State-load feedback only. No R_relax, L_lock, P_threshold, photon, gravity, objective-time or production claim.

## From `OPEN_QUESTIONS.md`

- Should the long-duration gate be extended to 25 and 50 node periods?
- Should TG-B1R phase-tension relaxation remain deferred until this result is reviewed?

## Summary fields

| field | value |
|---|---|
| `S0_authoritative_TG_S` | 135.686219 |
| `casewise_normalization` | `false` |
| `enabled_source` | S_state |
| `new_control_parameter` | lambda_fb multiplies only G->phi feedback coefficient |
| `source_hypothesis` | STATE_LOAD_FEEDBACK |
| `bounded_feedback_promotion_requires_25_period_or_longer_gate` | `true` |
| `matrix` | bounded |
| `state_load_only` | `true` |

## Plots

*Generated from this run's CSVs by `tools/build_run_catalogue.py`. Regenerable — delete and rebuild to refresh.*

![[_plots/TG_B1S_BACKREACTION_ROBUSTNESS_GPU_20260714_142319/basin_tests.png]]

*basin tests*

![[_plots/TG_B1S_BACKREACTION_ROBUSTNESS_GPU_20260714_142319/coupling_fit_holdout.png]]

*coupling fit holdout*

![[_plots/TG_B1S_BACKREACTION_ROBUSTNESS_GPU_20260714_142319/coupling_response.png]]

*coupling response*

![[_plots/TG_B1S_BACKREACTION_ROBUSTNESS_GPU_20260714_142319/energy_ledger.png]]

*energy ledger*

![[_plots/TG_B1S_BACKREACTION_ROBUSTNESS_GPU_20260714_142319/field_replay_controls.png]]

*field replay controls*

## Artifacts

- **Run directory** (gitignored, local only): `sweep_runs/TG_B1S_BACKREACTION_ROBUSTNESS_GPU_20260714_142319/`
- **CSV data** (16): `artifact_hashes.csv`, `attractor_classification.csv`, `baseline_node_contract.csv`, `basin_tests.csv`, `coupling_fit_holdout.csv`, `coupling_response.csv`, `energy_ledger.csv`, `falsification_results.csv`, `field_replay_controls.csv`, `long_time_metrics.csv` ...

## Previous experiments

*None — this is the first catalogued run in the `TG-B1S` family.*

## Associated docs

- [[Branch - Gravity - Index]]
- [[EXPERIMENT_TRACKER]]
- [[gravity_maturity/CLAUDE_HANDOFF_20260715_TG_B1S_AND_RUNTIME]]
- [[gravity_maturity/TG_B1S_R_DOCUMENTATION_INPUTS]]
- [[gravity_maturity/TG_B1S_R_RESULTS]]

## Next experiment

- [[TG_B1S_DRIFT_DECOMPOSITION_GPU_20260714_184247]] &middot; `2026-07-14`

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

