# TG-B1S-LT Documentation Inputs

- Status: `TG_STATE_LOAD_CONTINUOUS_SLOW_DRIFT`.
- Labels: `TG_STATE_LOAD_CONTINUOUS_SLOW_DRIFT`.
- Artifacts: `/mnt/f/quantule_mapper/sweep_runs/TG_B1S_LONG_TIME_GPU_20260714_154323`.
- Source: `STATE_LOAD_FEEDBACK` only.
- Preserved labels: `TG_STATE_LOAD_FEEDFORWARD_CHAIN_SUPPORTED`, `TG_STATE_LOAD_GEOMETRIC_BACKREACTION_DETECTED`, `TG_STATE_LOAD_BACKREACTION_ROBUST`, `TG_STATE_LOAD_BACKREACTION_ROBUST_BUT_WEAK`.

## Strongest Tables

- `translated_modal_reference.csv`
- `long_time_pairs.csv`
- `phase_drift_fits.csv`
- `field_boundedness.csv`
- `basin_tests.csv`
- `numerical_validation.csv`
- `falsification_results.csv`

## Caveats

- This branch does not require radiation.
- Any failing numerical-validation row blocks bounded-feedback promotion.
- Do not update the master catalogue until review.
- The 25-period pair was stable-shift positive, but the 50-period pair classified as `CONTINUOUS_DRIFT`.
- T/G fields remained bounded and approached fixed profiles; the failed gate is unresolved long-time node drift, not field blow-up.
- `TG_STATE_LOAD_BOUNDED_FEEDBACK_SUPPORTED` is not supported by this pass.

## Suggested Wording

The state-load feedback branch shows a real, robust and weak G-to-node backreaction, but the preregistered long-time primary pass did not show a bounded shifted-node attractor. The node remains localized and the T/G response fields remain bounded over 50 node periods, yet full-loop minus feedback-off phase, core and width observables continue drifting. The result should therefore be recorded as `TG_STATE_LOAD_CONTINUOUS_SLOW_DRIFT`, with bounded-feedback support left unpromoted.

## Recommended Next Decision

Do not start the phase-tension radiative branch from this result alone. Either audit the source of the slow drift under the frozen state-load model, or begin a separate model-design branch that explicitly changes the feedback equations and treats the current drift result as a falsification boundary for the frozen model.
