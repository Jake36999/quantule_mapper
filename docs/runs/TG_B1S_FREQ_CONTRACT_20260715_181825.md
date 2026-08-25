---
run_id: "TG_B1S_FREQ_CONTRACT_20260715_181825"
date: 2026-07-15
family: "TG-B1S"
sector: "gravity"
verdict: "TG_B1S_FREQUENCY_CONTRACT_SIGN_MATCH_ONLY"
complete: false
source: summary.json
N: 48
L: 10.0
dt: 0.002
n_csv: 0
n_plots: 0
tags: [run, gravity, TG_B1S]
---
# TG_B1S_FREQ_CONTRACT_20260715_181825

*Dual-substrate single-node / static baseline* &middot; **TG-B1S** &middot; `2026-07-15`

> [!abstract] Verdict
> `TG_B1S_FREQUENCY_CONTRACT_SIGN_MATCH_ONLY`

> [!warning] Completion sentinel missing
> This run was expected to write a `RUN_COMPLETE.json` and did not - the summary may be partial.

## Summary fields

| field | value |
|---|---|
| `S0` | 135.686219 |
| `qball_residual` | 1.1855e-09 |
| `T_peak` | 0.001444 |
| `G_peak` | 8.0095e-04 |
| `T_node` | 4.2890e-04 |
| `G_node` | -2.8663e-04 |
| `A_minus_1_max` | 4.8058e-05 |
| `A_minus_1_min` | 1.1211e-06 |
| `leg1_fixed_profile_d_omega` | 5.3979e-07 |
| `leg2_fixed_Q_d_omega` | 4.6897e-06 |
| `dQ_dw` | -618.340258 |
| `ddE_dw` | -0.0029 |
| `measured_delta_omega_infty_D3_convention` | -2.1509e-06 |
| `measured_physical_shift` | 2.1509e-06 |
| `convention_note` | D3 delta_omega_infty is the slope of (theta_full - theta_off); theta ~ -omega t, so phy... |
| `caveats` | quasi-static (ignores T/G transient + oscillatory dynamics), first-order in (A-1), absorber neglected in the static solve (fields screened), steady S from unperturbed node (second-order neglected) |

## Configuration

| parameter | value |
|---|---|
| `N` | 48 |
| `L` | 10 |
| `c` | 0.5477 |
| `m` | 1 |
| `a` | 0.8 |
| `s` | -0.5 |
| `f` | -0.1 |
| `w` | 0.964 |
| `dt` | 0.002 |
| `alpha_T` | 0.35 |
| `omega_T` | 1.25 |
| `omega_G` | 0.85 |
| `gamma_T` | 0.08 |
| `gamma_G` | 0.06 |
| `kappa_TG` | 0.55 |
| `epsilon_G` | 0.06 |
| `cT` | 0.7 |
| `cG` | 0.55 |
| `core_radius` | 2 |
| `lambda_fb` | 1 |

## Artifacts

- **Run directory** (gitignored, local only): `sweep_runs/TG_B1S_FREQ_CONTRACT_20260715_181825/`

## Previous experiments

*Auto-derived: the preceding runs in the same family (`TG-B1S`). Correct by hand if the lineage is wrong — edits here survive rebuilds only if you move the link below the Review notes marker.*

- [[TG_B1S_D4_ROWS_GPU_20260715_091558]] &middot; `2026-07-15`
- [[TG_B1S_D4_ROWS_GPU_20260715_080446]] &middot; `2026-07-15`
- [[TG_B1S_D4_D5_CLOSURE_GPU_20260715_003411]] &middot; `2026-07-15`

## Associated docs

- [[Branch - Gravity - Index]]
- [[EXPERIMENT_TRACKER]]
- [[RUN_QUEUE]]
- [[gravity_maturity/TG_B1S_EQUATION_AUDIT_ADDENDUM_20260715]]
- [[gravity_maturity/TG_B1S_FC1_AND_RUN_QUEUE_BANKING_NOTE_20260715]]
- [[gravity_maturity/TG_B1S_FREQUENCY_CONTRACT_RESULTS]]

## Next experiment

- [[TG_B1S_D4_DISCREPANCY_REVIEW_CODEX_PREP_20260717]] &middot; `2026-07-17`

## Branches

- [[Main branch]] &larr; via [[Branch - Gravity - Index|gravity sector index]]
- Sector: `gravity` &middot; family: `TG-B1S`

---

## Review notes

*(Preserved across catalogue rebuilds - everything above this line is regenerated.)*

