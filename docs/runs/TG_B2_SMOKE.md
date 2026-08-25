---
run_id: "TG_B2_SMOKE"
date: 2026-07-16
family: "TG-B2"
sector: "gravity"
verdict: "TG_B2_DYNAMICAL_AWELL_ATTRACTION_SECULAR_CONFIRMED"
complete: true
source: summary.json
N: 64
L: 16.0
T: 6.0
dt: 0.002
elapsed_h: 0.11259055190616184
n_csv: 3
n_plots: 0
tags: [run, gravity, TG_B2]
---
# TG_B2_SMOKE

*Dual-substrate two-node force (A-well/A-hill)* &middot; **TG-B2** &middot; `2026-07-16`

> [!abstract] Verdict
> `TG_B2_DYNAMICAL_AWELL_ATTRACTION_SECULAR_CONFIRMED`

**Note:** secular loop force = slope of J=P_R_full-P_R_off over settled window; inward calibrated from bare-off drift slope; A-well/A-hill antisymmetry = control

**Boundary:** mirror-only dynamical two-node force; frozen TG-B1S untouched; no gravity/UFF/IRER claim

## Results (`rows`, 1 rows)

| sep | n_samples | t_end | slope_J_well | slope_J_hill | off_drift_slope | inward_sign | well_inward | sign_reverses | charge_ok | P_tot_max | sep_min | nodes_distinct | gates_pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3.5 | 13 | 6 | -8.6005e-06 | 8.6223e-06 | -0.589655 | -1 | `true` | `true` | `true` | 4.2454e-15 | 3.793368 | `true` | `true` |

## Summary fields

| field | value |
|---|---|
| `elapsed_hours` | 0.112591 |
| `seps` | 3.5 |

## Configuration

| parameter | value |
|---|---|
| `out` | sweep_runs/TG_B2_SMOKE |
| `seps` | 3.5 |
| `N` | 64 |
| `L` | 16 |
| `T` | 6 |
| `dt` | 0.002 |
| `sample_dt` | 0.5 |
| `f_discard` | 0.4 |
| `merge_thresh` | 1.2 |
| `q_tol` | 0.01 |
| `target_hours` | 8 |
| `c` | 0.5477 |
| `m` | 1 |
| `a` | 0.8 |
| `s` | -0.5 |
| `f` | -0.1 |
| `w` | 0.964 |
| `alpha_T` | 0.35 |
| `omega_T` | 1.25 |
| `omega_G` | 0.85 |
| `gamma_T` | 0.08 |
| `gamma_G` | 0.06 |
| `kappa_TG` | 0.55 |
| `epsilon_G` | 0.06 |
| `cT` | 0.7 |
| `cG` | 0.55 |
| `absorb_width` | 1.6 |
| `absorb_strength` | 0.02 |
| `core_radius` | 2 |

## Artifacts

- **Run directory** (gitignored, local only): `sweep_runs/TG_B2_SMOKE/`
- **CSV data** (3): `scalars_sep3.50_hill.csv`, `scalars_sep3.50_off.csv`, `scalars_sep3.50_well.csv`

## Previous experiments

*Auto-derived: the preceding runs in the same family (`TG-B2`). Correct by hand if the lineage is wrong — edits here survive rebuilds only if you move the link below the Review notes marker.*

- [[TG_B2_OVERNIGHT_20260716_010733]] &middot; `2026-07-16`
- [[TG_B2_DYNAMICAL_FORCE_20260716_001134]] &middot; `2026-07-16`
- [[TG_B2_TWO_NODE_AWELL_20260715_233906]] &middot; `2026-07-15`

## Associated docs

- [[Branch - Gravity - Index]]
- [[EXPERIMENT_TRACKER]]

## Next experiment

- [[TG_B2_COOLED_PAIR_DRY_RUN_CODEX_PREP_20260717]] &middot; `2026-07-17`

## Branches

- [[Main branch]] &larr; via [[Branch - Gravity - Index|gravity sector index]]
- Sector: `gravity` &middot; family: `TG-B2`

---

## Review notes

*(Preserved across catalogue rebuilds - everything above this line is regenerated.)*

