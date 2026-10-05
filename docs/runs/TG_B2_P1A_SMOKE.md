---
run_id: "TG_B2_P1A_SMOKE"
date: 2026-08-25
family: "TG-B2"
sector: "gravity"
verdict: "TG_B2_MIDPLANE_FLUX_INCONCLUSIVE"
complete: true
source: summary.json
N: 48
L: 12.0
T: 8.0
dt: 0.002
elapsed_h: 0.037457582553227745
n_csv: 2
n_plots: 2
tags: [run, gravity, TG_B2]
---
# TG_B2_P1A_SMOKE

*Dual-substrate two-node force (A-well/A-hill)* &middot; **TG-B2** &middot; `2026-08-25`

> [!abstract] Verdict
> `TG_B2_MIDPLANE_FLUX_INCONCLUSIVE`

**Note:** F_flux is an INDEPENDENT estimator: it samples a 2-D plane and carries |pi|^2, m^2 rho and U(rho), none of which appear in the body force F_R. Agreement is a genuine cross-check; disagreement means the two measure different quantities.

**Boundary:** mirror-only; frozen TG-B1S dynamics untouched; no gravity/UFF/IRER claim

## Summary fields

| field | value |
|---|---|
| `gates` | `G1_off_ledger_closes`=`true`, `G2_live_ledger_closes`=`true`, `G3_estimators_agree`=`true`, `G4_flux_sign_reverses`= |
| `tolerances` | `resid_rel`=0.05, `estimator_rel_gap`=0.05 |
| `sep` | 3 |
| `N` | 48 |
| `L` | 12 |
| `T` | 8 |
| `dt` | 0.002 |
| `elapsed_hours` | 0.037458 |

## Configuration

| parameter | value |
|---|---|
| `out` | sweep_runs/TG_B2_P1A_SMOKE |
| `sep` | 3 |
| `N` | 48 |
| `L` | 12 |
| `T` | 8 |
| `dt` | 0.002 |
| `sample_dt` | 0.05 |
| `f_discard` | 0.4 |
| `arms` | off,well |
| `reanalyze` | `false` |
| `resid_tol` | 0.05 |
| `agree_tol` | 0.05 |
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

## Plots

*Generated from this run's CSVs by `tools/build_run_catalogue.py`. Regenerable — delete and rebuild to refresh.*

![[_plots/TG_B2_P1A_SMOKE/stress_off.png]]

*stress off*

![[_plots/TG_B2_P1A_SMOKE/stress_well.png]]

*stress well*

## Artifacts

- **Run directory** (gitignored, local only): `sweep_runs/TG_B2_P1A_SMOKE/`
- **CSV data** (2): `stress_off.csv`, `stress_well.csv`

## Previous experiments

*Auto-derived: the preceding runs in the same family (`TG-B2`). Correct by hand if the lineage is wrong — edits here survive rebuilds only if you move the link below the Review notes marker.*

- [[TG_B2_MID_conv_0.25]] &middot; `2026-08-25`
- [[TG_B2_MID_conv_0.10]] &middot; `2026-08-25`
- [[TG_B2_MID_conv_0.05]] &middot; `2026-08-25`

## Associated docs

- [[gravity_maturity/TG_P1A_ENERGY_OBSERVABLE]]

## Next experiment

*None yet — this is the most recent run in the `TG-B2` family.*

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

