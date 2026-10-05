---
run_id: "GRAVITY_D_EFFECTIVE_MEDIUM_CODEX_20260713"
date: 2026-07-13
family: "Gravity-D"
sector: "gravity"
verdict: "D_EFFECTIVE_MEDIUM_ATTRACTION_UNCLEAR"
complete: false
source: derived
n_csv: 1
n_plots: 1
tags: [run, gravity, Gravity_D]
---
# GRAVITY_D_EFFECTIVE_MEDIUM_CODEX_20260713

*Spatial-geometry gravity mirror (non-Newtonian)* &middot; **Gravity-D** &middot; `2026-07-13`

> [!abstract] Verdict
> `D_EFFECTIVE_MEDIUM_ATTRACTION_UNCLEAR`

> [!info] Derived note - no `summary.json`
> This run predates or bypasses the `summary.json` convention. Fields below are reconstructed from: `replication_report.md`.
> The verdict shown is `RUN_COMPLETE.json.status` where present, otherwise the first verdict-shaped token found in the run's own report - **treat it as indicative and confirm against the source document.**

## From `replication_report.md`

Verdict: `D_EFFECTIVE_MEDIUM_ATTRACTION_UNCLEAR`

Operator: `H = -D div(N grad)`, with momentum contract
`d<P_x>/dt = -D int (d_x N) |grad psi|^2 dV` for the implemented periodic spectral operator.

| label | drift_x | net_drift_x | F_analytic | F_rhs | F_fd | rhs_rel | norm_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| main_well | -8.930284e-05 | -8.930284e-05 | -3.458820e-02 | -3.458820e-02 | -3.458822e-02 | 4.312e-08 | 1.000000000 |
| flat_null | +2.220446e-15 | +0.000000e+00 | -0.000000e+00 | -2.555649e-16 | +1.674639e-14 | 2.556e+14 | 1.000000000 |
| reversed_hill | +1.041040e-04 | +1.041040e-04 | +3.870449e-02 | +3.870449e-02 | +3.870450e-02 | 6.454e-15 | 1.000000000 |
| wide_probe | -4.148335e-05 | -4.151142e-05 | -7.520812e-02 | -7.520812e-02 | -7.520812e-02 | 5.536e-16 | 1.000000000 |
| larger_box | -8.929749e-05 | -8.929755e-05 | -3.459023e-02 | -3.458800e-02 | -3.458801e-

## Plots

*Generated from this run's CSVs by `tools/build_run_catalogue.py`. Regenerable — delete and rebuild to refresh.*

![[_plots/GRAVITY_D_EFFECTIVE_MEDIUM_CODEX_20260713/summary.png]]

*summary*

## Artifacts

- **Run directory** (gitignored, local only): `sweep_runs/GRAVITY_D_EFFECTIVE_MEDIUM_CODEX_20260713/`
- **CSV data** (1): `summary.csv`

## Previous experiments

*Auto-derived: the preceding runs in the same family (`Gravity-D`). Correct by hand if the lineage is wrong — edits here survive rebuilds only if you move the link below the Review notes marker.*

- [[GRAVITY_D_DYNAMICS_CHARACTERIZATION_GPU_E0_SMOKE_20260713_220905]] &middot; `2026-07-13`
- [[GRAVITY_D_DYNAMICS_CHARACTERIZATION_GPU_20260713_221727]] &middot; `2026-07-13`
- [[GRAVITY_C_CLOCK_20260713_001146]] &middot; `2026-07-13`

## Associated docs

- [[Branch - Gravity - Index]]
- [[EXPERIMENT_TRACKER]]
- [[GRAVITY_D_EFFECTIVE_MEDIUM_CODEX_REPLICATION]]
- [[external_validation/artifact_bundles/gravity_d_effective_medium_characterization/docs/GRAVITY_D_EFFECTIVE_MEDIUM_CODEX_REPLICATION]]

## Next experiment

- [[GRAVITY_D_EXISTING_REPRO_CODEX_SHORT_20260713]] &middot; `2026-07-13`

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

