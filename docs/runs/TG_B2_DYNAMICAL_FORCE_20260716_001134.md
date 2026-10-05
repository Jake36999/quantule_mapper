---
run_id: "TG_B2_DYNAMICAL_FORCE_20260716_001134"
date: 2026-07-16
family: "TG-B2"
sector: "gravity"
verdict: "TG_B2_DYNAMICAL_AWELL_ATTRACTION_CONFIRMED"
complete: false
source: summary.json
N: 64
L: 16.0
T: 12.0
dt: 0.002
n_csv: 0
n_plots: 0
tags: [run, gravity, TG_B2]
---
# TG_B2_DYNAMICAL_FORCE_20260716_001134

*Dual-substrate two-node force (A-well/A-hill)* &middot; **TG-B2** &middot; `2026-07-16`

> [!abstract] Verdict
> `TG_B2_DYNAMICAL_AWELL_ATTRACTION_CONFIRMED`

> [!warning] Completion sentinel missing
> This run was expected to write a `RUN_COMPLETE.json` and did not - the summary may be partial.

**Note:** J = P_R(full)-P_R(off); inward calibrated from bare OFF drift. Momentum observable (not COM).

**Boundary:** mirror-only dynamical two-node force; frozen TG-B1S untouched; no gravity/UFF/IRER claim

## Summary fields

| field | value |
|---|---|
| `sep` | 3.5 |
| `dphi` | 0 |
| `P_R_off_end` | -7.743232 |
| `inward_sign` | -1 |
| `J_well_end` | -1.7617e-04 |
| `J_hill_end` | 1.7625e-04 |
| `well_inward` | `true` |
| `sign_flip_reverses` | `true` |
| `P_total_conservation_max` | 2.3355e-14 |
| `mass_imbalance_max` | `off`=0.069739, `well`=0.069739, `hill`=0.069738 |
| `caveats` | short-range Yukawa -> NOT gravity/1-over-r^2 (long-range = LR-1), weak effect (eps_G=0.06); near-field separations, half-space momentum: dP_R/dt = body force + midplane stress flux; differential isolates... |

## Configuration

| parameter | value |
|---|---|
| `out` |  |
| `N` | 64 |
| `L` | 16 |
| `sep` | 3.5 |
| `T` | 12 |
| `dt` | 0.002 |
| `sample_dt` | 0.5 |
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

- **Run directory** (gitignored, local only): `sweep_runs/TG_B2_DYNAMICAL_FORCE_20260716_001134/`

## Previous experiments

*Auto-derived: the preceding runs in the same family (`TG-B2`). Correct by hand if the lineage is wrong — edits here survive rebuilds only if you move the link below the Review notes marker.*

- [[TG_B2_TWO_NODE_AWELL_20260715_233906]] &middot; `2026-07-15`
- [[TG_B2_TWO_NODE_AWELL_20260715_232649]] &middot; `2026-07-15`
- [[TG_B2_STATIC_FORCE_20260715_235154]] &middot; `2026-07-15`

## Associated docs

- [[Branch - Gravity - Index]]
- [[EXPERIMENT_TRACKER]]
- [[RUN_QUEUE]]
- [[gravity_maturity/TG_B2_TWO_NODE_FORCE_RESULTS]]

## Next experiment

- [[TG_B2_OVERNIGHT_20260716_010733]] &middot; `2026-07-16`

## Branches

- [[Main branch]] &larr; via [[Branch - Gravity - Index|gravity sector index]]
- Sector: `gravity` &middot; family: `TG-B2`

---

## Review notes

*(Preserved across catalogue rebuilds - everything above this line is regenerated.)*

