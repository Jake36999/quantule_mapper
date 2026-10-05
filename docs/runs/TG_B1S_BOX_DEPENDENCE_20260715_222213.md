---
run_id: "TG_B1S_BOX_DEPENDENCE_20260715_222213"
date: 2026-07-15
family: "TG-B1S"
sector: "gravity"
verdict: "BOX_DEPENDENCE_PARTIAL__LOCAL_MECHANISM_EXPLAINS_SOME_NOT_ALL"
complete: false
source: summary.json
n_csv: 0
n_plots: 0
tags: [run, gravity, TG_B1S]
---
# TG_B1S_BOX_DEPENDENCE_20260715_222213

*Dual-substrate single-node / static baseline* &middot; **TG-B1S** &middot; `2026-07-15`

> [!abstract] Verdict
> `BOX_DEPENDENCE_PARTIAL__LOCAL_MECHANISM_EXPLAINS_SOME_NOT_ALL`

> [!warning] Completion sentinel missing
> This run was expected to write a `RUN_COMPLETE.json` and did not - the summary may be partial.

## Summary fields

| field | value |
|---|---|
| `measured` | `baseline`=-2.1627e-06, `grid_refined`=-2.1627e-06, `larger_box`=-8.5243e-07, `reference`=-2.1509e-06, `measured_larger_box_over_baseline`=0.394152 |
| `analytic_larger_box_over_baseline` | 1.319468 |
| `analytic_flat_across_L` | `false` |
| `local_mechanism_reproduces_measured_drop` | `false` |
| `dx_effect_larger_box_fine_minus_coarse` | -8.8718e-10 |
| `screening` | `naive_cG_over_omegaG`=0.647059, `light_coupled_mode_range`=0.81946 |
| `caveats` | fixed-profile (Leg 1) analytic shift only; absorber neglected in the static T/G solve, first order in (A-1); quasi-static (ignores T/G transient/oscillatory dynamics), the D4 measurement compares each row vs a FIXED N48/L10 reference (not a same-geometry ... |

## Artifacts

- **Run directory** (gitignored, local only): `sweep_runs/TG_B1S_BOX_DEPENDENCE_20260715_222213/`

## Previous experiments

*Auto-derived: the preceding runs in the same family (`TG-B1S`). Correct by hand if the lineage is wrong — edits here survive rebuilds only if you move the link below the Review notes marker.*

- [[TG_B1S_STATE_LOAD_GPU_20260714_140157]] &middot; `2026-07-14`
- [[TG_B1S_STATE_LOAD_GPU_20260714_135157]] &middot; `2026-07-14`
- [[TG_B1S_LONG_TIME_GPU_20260714_154323]] &middot; `2026-07-14`

## Associated docs

- [[Branch - Gravity - Index]]
- [[EXPERIMENT_TRACKER]]
- [[RUN_QUEUE]]
- [[gravity_maturity/TG_B1S_BOX_DEPENDENCE_RESULTS]]
- [[gravity_maturity/TG_B1S_D4_BOX_DISCREPANCY_CODEX_NOTE_20260715]]

## Next experiment

- [[TG_B1S_D4_D5_CLOSURE_GPU_20260715_003411]] &middot; `2026-07-15`

## Branches

- [[Main branch]] &larr; via [[Branch - Gravity - Index|gravity sector index]]
- Sector: `gravity` &middot; family: `TG-B1S`

---

## Review notes

*(Preserved across catalogue rebuilds - everything above this line is regenerated.)*

