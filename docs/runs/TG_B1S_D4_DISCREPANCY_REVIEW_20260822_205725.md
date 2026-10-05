---
run_id: "TG_B1S_D4_DISCREPANCY_REVIEW_20260822_205725"
date: 2026-08-22
family: "TG-B1S"
sector: "gravity"
verdict: "D4_LARGER_BOX_DISCREPANCY_REVIEWED"
complete: true
source: derived
n_csv: 4
n_plots: 2
tags: [run, gravity, TG_B1S]
---
# TG_B1S_D4_DISCREPANCY_REVIEW_20260822_205725

*Dual-substrate single-node / static baseline* &middot; **TG-B1S** &middot; `2026-08-22`

> [!abstract] Verdict
> `D4_LARGER_BOX_DISCREPANCY_REVIEWED`

> [!info] Derived note - no `summary.json`
> This run predates or bypasses the `summary.json` convention. Fields below are reconstructed from: `TECHNICAL_HANDOFF.md`, `OPEN_QUESTIONS.md`, `recommendation.json`.
> The verdict shown is `RUN_COMPLETE.json.status` where present, otherwise the first verdict-shaped token found in the run's own report - **treat it as indicative and confirm against the source document.**

## From `TECHNICAL_HANDOFF.md`

Run directory: `/mnt/f/quantule_mapper/sweep_runs/TG_B1S_D4_DISCREPANCY_REVIEW_20260822_205725`.

This is a CPU post-processing review. No field evolution was run.

**Finding**

- `D4_larger_box_50P` remains a real D4 gate failure on frequency scale.
- The row is structurally clean: high overlap, bounded T/G classes, low leakage, and no structural-channel fail.
- The grid-refined and absorber-wider rows pass, so the failure is specific to the larger-box geometry/branch/transient comparison.
- FC-box predicts the local fixed-profile mechanism should increase in the larger box, while the measured D4 shift decreases.

**Boundary**

This does not falsify the TG-B1S feed-forward chain or robust-but-weak backreaction. It blocks D5-style promotion until a targeted same-geometry or relief/boundary audit is run.

## From `OPEN_QUESTIONS.md`

- Is the larger-box row selecting a different Petviashvili branch/profile despite same nominal Q-ball target?
- Does absorber radius alter T/G tail ring-in or boundary energy accounting over 50 periods?
- Would an L=12 row compared only to a local L=12 reference clear the frequency-scale gate?

## Summary fields

| field | value |
|---|---|
| `absorber_wider_passed` | `true` |
| `analytic_contradicts_measured_drop` | `true` |
| `analytic_larger_box_over_baseline` | 1.319468 |
| `d5_should_remain_blocked` | `true` |
| `grid_refined_passed` | `true` |
| `larger_box_failed_only_frequency_scale` | `true` |
| `measured_larger_box_over_baseline` | 0.394152 |
| `primary_inference` | D4 larger-box failure is not explained by resolution or the local fixed-profile frequen... |
| `recommended_next` | Queue a targeted same-geometry L=12 baseline/off/full rerun or absorber/relief isolatio... |
| `status` | D4_LARGER_BOX_DISCREPANCY_REVIEWED |
| `outdir` | /mnt/f/quantule_mapper/sweep_runs/TG_B1S_D4_DISCREPANCY_REVIEW_20260822_205725 |

## Plots

*Generated from this run's CSVs by `tools/build_run_catalogue.py`. Regenerable — delete and rebuild to refresh.*

![[_plots/TG_B1S_D4_DISCREPANCY_REVIEW_20260822_205725/d4_case_comparison.png]]

*d4 case comparison*

![[_plots/TG_B1S_D4_DISCREPANCY_REVIEW_20260822_205725/fc_box_mechanism_comparison.png]]

*fc box mechanism comparison*

## Artifacts

- **Run directory** (gitignored, local only): `sweep_runs/TG_B1S_D4_DISCREPANCY_REVIEW_20260822_205725/`
- **CSV data** (4): `artifact_hashes.csv`, `d4_case_comparison.csv`, `falsification_results.csv`, `fc_box_mechanism_comparison.csv`

## Previous experiments

*Auto-derived: the preceding runs in the same family (`TG-B1S`). Correct by hand if the lineage is wrong — edits here survive rebuilds only if you move the link below the Review notes marker.*

- [[TG_B1S_D4_DISCREPANCY_REVIEW_CODEX_PREP_20260717]] &middot; `2026-07-17`
- [[TG_B1S_FREQ_CONTRACT_20260715_181825]] &middot; `2026-07-15`
- [[TG_B1S_D4_ROWS_GPU_20260715_091558]] &middot; `2026-07-15`

## Associated docs

- [[Branch - Gravity - Index]]
- [[EXPERIMENT_TRACKER]]
- [[RUN_QUEUE]]

## Next experiment

*None yet — this is the most recent run in the `TG-B1S` family.*

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

