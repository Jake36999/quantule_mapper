---
run_id: "TG_B2_CHARACTERIZATION_20260718_121223"
date: 2026-07-18
family: "TG-B2"
sector: "gravity"
verdict: "TG_B2_CHARACTERIZATION_COMPLETE"
complete: true
source: summary.json
elapsed_h: 8.921437927484511
n_csv: 8
n_plots: 5
tags: [run, gravity, TG_B2]
---
# TG_B2_CHARACTERIZATION_20260718_121223

*Dual-substrate two-node force (A-well/A-hill)* &middot; **TG-B2** &middot; `2026-07-18`

> [!abstract] Verdict
> `TG_B2_CHARACTERIZATION_COMPLETE`

**Note:** F(sep) falloff/screening + F(mass) scaling exponent for the in-phase A-well body force. Fmass_power p: F~M^p; gravity(equal masses) would give p~2.

**Boundary:** mirror-only; frozen modules; no gravity/UFF/IRER claim

## Results (`rows`, 8 rows)

| row | sep | w | node_mass | node_amp | F_R_well_avg | F_R_well_std | attracts | charge_ok | sep_min | nodes_distinct | gates_pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fsep_sep2.5 | 2.5 | 0.964 | 52.061179 | 0.987733 | -5.5974e-05 | 1.0091e-05 | `true` | `true` | 3.25121 | `true` | `true` |
| Fsep_sep3.0 | 3 | 0.964 | 52.061179 | 0.987733 | -5.4501e-05 | 7.9503e-06 | `true` | `true` | 2.982445 | `true` | `true` |
| Fsep_sep4.0 | 4 | 0.964 | 52.061179 | 0.987733 | -4.8526e-05 | 8.3831e-06 | `true` | `true` | 2.663336 | `true` | `true` |
| Fsep_sep5.0 | 5 | 0.964 | 52.061179 | 0.987733 | -4.2134e-05 | 1.0278e-05 | `true` | `true` | 2.650108 | `true` | `true` |
| Fmass_w0.945 | 3 | 0.945 | 105.926346 | 1.047936 | -4.9989e-05 | 4.0509e-05 | `true` | `true` | 3.4766 | `true` | `true` |
| Fmass_w0.955 | 3 | 0.955 | 69.916557 | 1.027398 | -6.1072e-05 | 1.8131e-05 | `true` | `true` | 3.441226 | `true` | `true` |
| Fmass_w0.972 | 3 | 0.972 | 42.830841 | 0.926792 | -5.1975e-05 | 6.8128e-06 | `true` | `true` | 2.781189 | `true` | `true` |
| Fmass_w0.980 | 3 | 0.98 | 38.676374 | 0.825216 | -5.8493e-05 | 8.9664e-06 | `true` | `true` | 2.9079 | `true` | `true` |

## Summary fields

| field | value |
|---|---|
| `fits` | `screening_length`=8.627394, `Fsep_power`=-0.41241, `Fmass_power`=-0.06728 |
| `elapsed_hours` | 8.921438 |

## Plots

*Generated from this run's CSVs by `tools/build_run_catalogue.py`. Regenerable — delete and rebuild to refresh.*

![[_plots/TG_B2_CHARACTERIZATION_20260718_121223/scalars_Fmass_w0.945.png]]

*scalars Fmass w0.945*

![[_plots/TG_B2_CHARACTERIZATION_20260718_121223/scalars_Fmass_w0.955.png]]

*scalars Fmass w0.955*

![[_plots/TG_B2_CHARACTERIZATION_20260718_121223/scalars_Fmass_w0.972.png]]

*scalars Fmass w0.972*

![[_plots/TG_B2_CHARACTERIZATION_20260718_121223/scalars_Fmass_w0.980.png]]

*scalars Fmass w0.980*

![[_plots/TG_B2_CHARACTERIZATION_20260718_121223/scalars_Fsep_sep2.5.png]]

*scalars Fsep sep2.5*

## Artifacts

- **Run directory** (gitignored, local only): `sweep_runs/TG_B2_CHARACTERIZATION_20260718_121223/`
- **CSV data** (8): `scalars_Fmass_w0.945.csv`, `scalars_Fmass_w0.955.csv`, `scalars_Fmass_w0.972.csv`, `scalars_Fmass_w0.980.csv`, `scalars_Fsep_sep2.5.csv`, `scalars_Fsep_sep3.0.csv`, `scalars_Fsep_sep4.0.csv`, `scalars_Fsep_sep5.0.csv`

## Previous experiments

*Auto-derived: the preceding runs in the same family (`TG-B2`). Correct by hand if the lineage is wrong — edits here survive rebuilds only if you move the link below the Review notes marker.*

- [[TG_B2_ALIGNMENT_20260718_004234]] &middot; `2026-07-18`
- [[TG_B2_ALIGNMENT_20260718_004229]] &middot; `2026-07-18`
- [[TG_B2_ALIGNMENT_20260718_004221]] &middot; `2026-07-18`

## Associated docs

- [[Branch - Gravity - Index]]
- [[EXPERIMENT_TRACKER]]
- [[RUN_QUEUE]]
- [[gravity_maturity/TG_B2_CHARACTERIZATION_RESULTS]]

## Next experiment

- [[TG_B2_DYN_ALIGN_20260718_004440]] &middot; `2026-07-18`

## Branches

- [[Main branch]] &larr; via [[Branch - Gravity - Index|gravity sector index]]
- Sector: `gravity` &middot; family: `TG-B2`

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

