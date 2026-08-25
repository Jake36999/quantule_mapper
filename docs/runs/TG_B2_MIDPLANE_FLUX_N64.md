---
run_id: "TG_B2_MIDPLANE_FLUX_N64"
date: 2026-08-25
family: "TG-B2"
sector: "gravity"
verdict: "TG_B2_MIDPLANE_FLUX_CONFIRMS_BODY_FORCE"
complete: true
source: summary.json
N: 64
L: 16.0
T: 40.0
dt: 0.002
elapsed_h: 1.1474953757392036e-05
n_csv: 3
n_plots: 3
tags: [run, gravity, TG_B2]
---
# TG_B2_MIDPLANE_FLUX_N64

*Dual-substrate two-node force (A-well/A-hill)* &middot; **TG-B2** &middot; `2026-08-25`

> [!abstract] Verdict
> `TG_B2_MIDPLANE_FLUX_CONFIRMS_BODY_FORCE`

**Note:** F_flux is an INDEPENDENT estimator: it samples a 2-D plane and carries |pi|^2, m^2 rho and U(rho), none of which appear in the body force F_R. Agreement is a genuine cross-check; disagreement means the two measure different quantities.

**Boundary:** mirror-only; frozen TG-B1S dynamics untouched; no gravity/UFF/IRER claim

## Summary fields

| field | value |
|---|---|
| `gates` | `G1_off_ledger_closes`=`true`, `G2_live_ledger_closes`=`true`, `G3_estimators_agree`=`true`, `G4_flux_sign_reverses`=`true` |
| `tolerances` | `resid_rel`=0.05, `estimator_rel_gap`=0.05 |
| `sep` | 3 |
| `N` | 64 |
| `L` | 16 |
| `T` | 40 |
| `dt` | 0.002 |
| `elapsed_hours` | 1.1475e-05 |

## Configuration

| parameter | value |
|---|---|
| `out` | sweep_runs/TG_B2_MIDPLANE_FLUX_N64 |
| `sep` | 3 |
| `N` | 64 |
| `L` | 16 |
| `T` | 40 |
| `dt` | 0.002 |
| `sample_dt` | 0.05 |
| `f_discard` | 0.4 |
| `arms` | off,well,hill |
| `reanalyze` | `true` |
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

![[_plots/TG_B2_MIDPLANE_FLUX_N64/stress_hill.png]]

*stress hill*

![[_plots/TG_B2_MIDPLANE_FLUX_N64/stress_off.png]]

*stress off*

![[_plots/TG_B2_MIDPLANE_FLUX_N64/stress_well.png]]

*stress well*

## Artifacts

- **Run directory** (gitignored, local only): `sweep_runs/TG_B2_MIDPLANE_FLUX_N64/`
- **CSV data** (3): `stress_hill.csv`, `stress_off.csv`, `stress_well.csv`

## Previous experiments

*Auto-derived: the preceding runs in the same family (`TG-B2`). Correct by hand if the lineage is wrong — edits here survive rebuilds only if you move the link below the Review notes marker.*

- [[TG_B2_CONV_SEP3_20260824_221915]] &middot; `2026-08-24`
- [[TG_B2_FARFIELD_20260822_205537]] &middot; `2026-08-22`
- [[TG_B2_CHARGE_AUDIT_20260719_005326]] &middot; `2026-07-19`

## Associated docs

- [[IRER_MASTER_HYPOTHESIS_CATALOG]]
- [[gravity_maturity/TG_P2_MIDPLANE_STRESS_FLUX_RESULTS]]

## Next experiment

- [[TG_B2_MIDPLANE_SMOKE]] &middot; `2026-08-25`

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

