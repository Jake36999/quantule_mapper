# TG-B1S-D Documentation Inputs

- Status: `TG_STATE_LOAD_LONG_TIME_DRIFT_UNRESOLVED`.
- Labels: `TG_STATE_LOAD_LONG_TIME_DRIFT_UNRESOLVED`.
- Artifacts: `/content/qm_job/results/tg_b1s_d_100p_reproduction/simulation`.
- Source: `STATE_LOAD_FEEDBACK` only.
- Frozen model: no coefficient, damping, coupling, source or absorber redesign.

## Strongest Tables

- `orbital_distance.csv`
- `phase_alignment.csv`
- `drift_channel_fits.csv`
- `asymptotic_frequency.csv`
- `long_time_100_period.csv`
- `numerical_validation.csv`
- `basin_orbital_stability.csv`
- `falsification_results.csv`

## Caveats

- Relative phase is allowed to grow under a stable frequency shift.
- Phase-aligned profile distance, amplitude, width and leakage determine whether the drift is structural.
- `TG_STATE_LOAD_BOUNDED_FEEDBACK_SUPPORTED` remains unpromoted unless validation and basin gates close.