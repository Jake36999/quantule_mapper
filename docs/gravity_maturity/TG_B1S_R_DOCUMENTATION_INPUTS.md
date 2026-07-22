# TG-B1S-R Documentation Inputs

- Status: `TG_STATE_LOAD_BACKREACTION_ROBUST_BUT_WEAK`.
- Labels: `TG_STATE_LOAD_BACKREACTION_ROBUST, TG_STATE_LOAD_BACKREACTION_ROBUST_BUT_WEAK`.
- Artifacts: `/mnt/f/quantule_mapper/sweep_runs/TG_B1S_BACKREACTION_ROBUSTNESS_GPU_20260714_142319`.
- Source: `STATE_LOAD_FEEDBACK` only.
- `R_relax`, `L_lock` and `P_threshold` disabled.
- `lambda_fb` multiplies only the existing G-to-phi coefficient.

## Strongest Supporting Tables

- `numerical_floor.csv`
- `coupling_response.csv`
- `modal_phase_metrics.csv`
- `field_replay_controls.csv`
- `long_time_metrics.csv`
- `falsification_results.csv`

## Caveats

- This is still a state-load branch; no radiation is required or claimed.
- Long intervals are bounded scouts unless the 25/50-period gate is explicitly present.
- Current run reached 10 node periods, not 25 or 50; do not promote `TG_STATE_LOAD_BOUNDED_FEEDBACK_SUPPORTED`.
- The translated basin row used the original modal reference for leakage, making that specific leakage value conservative.
- Do not update the master catalogue until review.
