---
run_id: "TG_SOURCE_SEMANTICS_GPU_20260714_133513"
date: 2026-07-14
family: "TG-S"
sector: "gravity"
verdict: "TG_NODE_STATE_LOAD_SOURCE_SUPPORTED"
complete: false
source: derived
n_csv: 10
n_plots: 5
tags: [run, gravity, TG_S]
---
# TG_SOURCE_SEMANTICS_GPU_20260714_133513

*Temporal-substrate scalar rung* &middot; **TG-S** &middot; `2026-07-14`

> [!abstract] Verdict
> `TG_NODE_STATE_LOAD_SOURCE_SUPPORTED`

> [!info] Derived note - no `summary.json`
> This run predates or bypasses the `summary.json` convention. Fields below are reconstructed from: `TECHNICAL_HANDOFF.md`, `OPEN_QUESTIONS.md`.
> The verdict shown is `RUN_COMPLETE.json.status` where present, otherwise the first verdict-shaped token found in the run's own report - **treat it as indicative and confirm against the source document.**

## From `TECHNICAL_HANDOFF.md`

Status: `TG_NODE_STATE_LOAD_SOURCE_SUPPORTED`.
Run directory: `/mnt/f/quantule_mapper/sweep_runs/TG_SOURCE_SEMANTICS_GPU_20260714_133513`.


- `TG_NODE_STATE_LOAD_SOURCE_SUPPORTED`
- `TG_PHASE_TENSION_RELAXATION_SOURCE_SUPPORTED`


Do not rerun TG-B1 feedback unless a reviewer accepts one supported source family and chooses exactly one feedback hypothesis.

## From `OPEN_QUESTIONS.md`

- Should a future TG-B1 rerun test `STATE_LOAD_FEEDBACK`, `PHASE_LOCKING_FEEDBACK`, or `PHASE_TENSION_RELAXATION_FEEDBACK` first?
- Are the source-family thresholds strict enough for external review?
- Should the two-packet locking event be repeated in the larger validated C3 two-Q-ball box before feedback use?

## Plots

*Generated from this run's CSVs by `tools/build_run_catalogue.py`. Regenerable — delete and rebuild to refresh.*

![[_plots/TG_SOURCE_SEMANTICS_GPU_20260714_133513/density_contamination.png]]

*density contamination*

![[_plots/TG_SOURCE_SEMANTICS_GPU_20260714_133513/derivative_convergence.png]]

*derivative convergence*

![[_plots/TG_SOURCE_SEMANTICS_GPU_20260714_133513/event_suite.png]]

*event suite*

![[_plots/TG_SOURCE_SEMANTICS_GPU_20260714_133513/forward_reverse_comparison.png]]

*forward reverse comparison*

![[_plots/TG_SOURCE_SEMANTICS_GPU_20260714_133513/invariance_results.png]]

*invariance results*

## Artifacts

- **Run directory** (gitignored, local only): `sweep_runs/TG_SOURCE_SEMANTICS_GPU_20260714_133513/`
- **CSV data** (10): `artifact_hashes.csv`, `baseline_node_contract.csv`, `density_contamination.csv`, `derivative_convergence.csv`, `event_suite.csv`, `falsification_results.csv`, `forward_reverse_comparison.csv`, `invariance_results.csv`, `local_source_metrics.csv`, `source_semantics_classification.csv`

## Previous experiments

*Auto-derived: the preceding runs in the same family (`TG-S`). Correct by hand if the lineage is wrong — edits here survive rebuilds only if you move the link below the Review notes marker.*

- [[TG_SOURCE_SEMANTICS_GPU_20260714_133207]] &middot; `2026-07-14`

## Associated docs

- [[Branch - Gravity - Index]]
- [[EXPERIMENT_TRACKER]]
- [[gravity_maturity/CLAUDE_HANDOFF_20260715_TG_B1S_AND_RUNTIME]]
- [[gravity_maturity/TG_S_DOCUMENTATION_INPUTS]]
- [[gravity_maturity/TG_S_RESULTS]]

## Next experiment

*None yet — this is the most recent run in the `TG-S` family.*

## Branches

- [[Main branch]] &larr; via [[Branch - Gravity - Index|gravity sector index]]
- Sector: `gravity` &middot; family: `TG-S`

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

