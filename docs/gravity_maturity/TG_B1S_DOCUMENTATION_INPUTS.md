# TG-B1S Documentation Inputs

- Status: `TG_STATE_LOAD_GEOMETRIC_BACKREACTION_DETECTED`.
- Labels: `TG_STATE_LOAD_FEEDFORWARD_CHAIN_SUPPORTED, TG_STATE_LOAD_GEOMETRIC_BACKREACTION_DETECTED`.
- Artifacts: `/mnt/f/quantule_mapper/sweep_runs/TG_B1S_STATE_LOAD_GPU_20260714_140157`.
- Source: `STATE_LOAD_FEEDBACK` only.
- Preserve prior TG-B1/TG-S labels; do not update master catalogue until review.

## Strongest Supporting Tables

- `baseline_node_contract.csv`: reproduced the TG-S Q-ball baseline.
- `intervention_results.csv`: source-off, temporal-off, geometric-off, global-phase and translation controls passed.
- `temporal_response.csv` / `geometric_response.csv`: persistent S-state load produced T, then G.
- `backreaction_effects.csv`: full-loop minus feedback-off deltas were nonzero.
- `numerical_validation.csv`: `dt/2`, grid, box and absorber checks completed, with backreaction deltas recorded.

## Caveats

- Backreaction magnitude is small; this is not yet a bounded-feedback or radiation result.
- `R_relax`, `L_lock` and `P_threshold` were intentionally disabled.
- No gravity, photon, objective-time, geodesic, universal-free-fall or IRER-validation claim is made.
