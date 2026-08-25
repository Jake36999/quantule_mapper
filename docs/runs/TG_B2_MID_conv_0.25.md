---
run_id: "TG_B2_MID_conv_0.25"
date: 2026-08-25
family: "TG-B2"
sector: "gravity"
verdict: "TG_B2_MIDPLANE_FLUX_STRESS_TENSOR_INVALID__OFF_ARM_LEDGER_OPEN"
complete: true
source: summary.json
N: 48
L: 12.0
T: 6.0
dt: 0.002
elapsed_h: 0.013859030538135106
n_csv: 1
n_plots: 1
tags: [run, gravity, TG_B2]
---
# TG_B2_MID_conv_0.25

*Dual-substrate two-node force (A-well/A-hill)* &middot; **TG-B2** &middot; `2026-08-25`

> [!abstract] Verdict
> `TG_B2_MIDPLANE_FLUX_STRESS_TENSOR_INVALID__OFF_ARM_LEDGER_OPEN`

**Note:** F_flux is an INDEPENDENT estimator: it samples a 2-D plane and carries |pi|^2, m^2 rho and U(rho), none of which appear in the body force F_R. Agreement is a genuine cross-check; disagreement means the two measure different quantities.

**Boundary:** mirror-only; frozen TG-B1S dynamics untouched; no gravity/UFF/IRER claim

## Summary fields

| field | value |
|---|---|
| `gates` | `G1_off_ledger_closes`=`false`, `G2_live_ledger_closes`=`true`, `G3_estimators_agree`=, `G4_flux_sign_reverses`= |
| `tolerances` | `resid_rel`=0.05, `estimator_rel_gap`=0.25 |
| `sep` | 3 |
| `N` | 48 |
| `L` | 12 |
| `T` | 6 |
| `dt` | 0.002 |
| `elapsed_hours` | 0.013859 |

## Configuration

| parameter | value |
|---|---|
| `out` | sweep_runs/TG_B2_MID_conv_0.25 |
| `sep` | 3 |
| `N` | 48 |
| `L` | 12 |
| `T` | 6 |
| `dt` | 0.002 |
| `sample_dt` | 0.25 |
| `f_discard` | 0.4 |
| `arms` | off |
| `resid_tol` | 0.05 |
| `agree_tol` | 0.25 |
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

![[_plots/TG_B2_MID_conv_0.25/stress_off.png]]

*stress off*

## Artifacts

- **Run directory** (gitignored, local only): `sweep_runs/TG_B2_MID_conv_0.25/`
- **CSV data** (1): `stress_off.csv`

## Previous experiments

*Auto-derived: the preceding runs in the same family (`TG-B2`). Correct by hand if the lineage is wrong — edits here survive rebuilds only if you move the link below the Review notes marker.*

- [[TG_B2_MID_conv_0.10]] &middot; `2026-08-25`
- [[TG_B2_MID_conv_0.05]] &middot; `2026-08-25`
- [[TG_B2_MID_S25]] &middot; `2026-08-25`

## Associated docs

- [[Branch - Gravity - Index]]
- [[EXPERIMENT_TRACKER]]

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

