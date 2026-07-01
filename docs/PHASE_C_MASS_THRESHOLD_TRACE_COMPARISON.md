# Phase C Mass Threshold Trace Comparison

## Purpose

This comparison turns the threshold pilot into a small mechanism pack.

Question:

`What does a low-K survivor, a low-K above-threshold failure, and a same-mass K=6 survivor actually do in time?`

The goal is descriptive:

- separate evidence from inference
- keep the accepted labels disciplined
- avoid upgrading any failure trace into stronger language than the metrics support

## Compared Cases

Generated bundle:

- [CORE_SAT_TRACE_COMPARE_20260623_185450](F:/quantule_mapper/sweep_runs/CORE_SAT_TRACE_COMPARE_20260623_185450)
- [mass_threshold_trace_comparison.csv](F:/quantule_mapper/sweep_runs/CORE_SAT_TRACE_COMPARE_20260623_185450/mass_threshold_trace_comparison.csv)
- [trace_overlay.png](F:/quantule_mapper/sweep_runs/CORE_SAT_TRACE_COMPARE_20260623_185450/trace_overlay.png)

Candidate panels:

- [K=1 below-threshold survivor](F:/quantule_mapper/sweep_runs/CORE_SAT_TRACE_COMPARE_20260623_185450/CORE_SAT_HUNT_20260623_171758_idx_4/trace_panel.png)
- [K=1 above-threshold failure](F:/quantule_mapper/sweep_runs/CORE_SAT_TRACE_COMPARE_20260623_185450/CORE_SAT_HUNT_20260623_173417_idx_2/trace_panel.png)
- [K=6 same-mass survivor](F:/quantule_mapper/sweep_runs/CORE_SAT_TRACE_COMPARE_20260623_185450/CORE_SAT_HUNT_20260623_173417_idx_10/trace_panel.png)
- [ref_feb56dc7 control](F:/quantule_mapper/sweep_runs/CORE_SAT_TRACE_COMPARE_20260623_185450/ref_feb56dc7/trace_panel.png)

Replay command:

```powershell
wsl.exe -d Ubuntu -- bash -lc 'source ~/jax_irer/bin/activate && cd /mnt/f/quantule_mapper && stdbuf -oL -eL python jax_scout/core_saturation_collapse_diag.py --mode trace-comparison --n-snap 60 --failure-n-snap 120 --outdir /mnt/f/quantule_mapper/sweep_runs/CORE_SAT_TRACE_COMPARE_20260623_185450'
```

Render command:

```powershell
python jax_scout/core_saturation_collapse_diag.py --mode render-trace-plots --outdir sweep_runs/CORE_SAT_TRACE_COMPARE_20260623_185450
```

## Evidence

### 1. K=1 Survivor Below Threshold

Source:

- idx `4` from [CORE_SAT_HUNT_20260623_171758](F:/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260623_171758)

Saved classification:

- `K = 1`
- `target_initial_mass = 500`
- `class = TRUE_SATURATED_BOUND_STATE`
- diagnostic label = `SATURATED_BOUND_STATE`

Key metrics:

- `er_final_or_last ~= 0.6944`
- `late_energy_slope ~= 6.72e-05`
- `rho_peak_max ~= 1.716`
- `core_radius_min ~= 4.129`
- `compactness_max ~= 59.754`
- `omega2_min_min ~= 0.313`
- `grad_log_omega_max ~= 17.885`
- `node_count_mid = 2`
- `node_count_last = 2`

Interpretation:

- this row does not stay single-core; it branches into a `2`-node late state
- but it remains bounded and passes the saved saturation classifier
- this is the below-threshold low-K survivor branch in the threshold pilot

### 2. K=1 Failure Above Threshold

Source:

- idx `2` from [CORE_SAT_HUNT_20260623_173417](F:/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260623_173417)

Saved classification:

- `K = 1`
- `target_initial_mass = 1200`
- `class = LATE_BLOWUP_REJECT`
- diagnostic label = `FRAGMENTING_BLOWUP`

Key metrics:

- `time_to_failure ~= 833.3`
- `time_to_blowup ~= 833.3`
- `er_final_or_last ~= 29.782`
- `late_energy_slope ~= 0.01064`
- `rho_peak_max ~= 4.118`
- `core_radius_min ~= 4.029`
- `compactness_max ~= 191.588`
- `omega2_min_min ~= 73.841`
- `grad_log_omega_max ~= 9.765`
- `node_count_mid = 2`
- `node_count_last = 2`

Interpretation:

- this is not a clean collapse-like signature
- it remains a split-first failure trace under the current criteria
- compactness is much higher than in the same-mass `K=6` survivor

### 3. K=6 Survivor At The Same Mass

Source:

- idx `10` from [CORE_SAT_HUNT_20260623_173417](F:/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260623_173417)

Saved classification:

- `K = 6`
- `target_initial_mass = 1200`
- `class = TRUE_SATURATED_BOUND_STATE`
- diagnostic label = `SATURATED_BOUND_STATE`

Key metrics:

- `er_final_or_last ~= 0.6515`
- `late_energy_slope ~= -3.75e-05`
- `rho_peak_max ~= 0.9875`
- `core_radius_min ~= 18.170`
- `compactness_max ~= 17.209`
- `omega2_min_min ~= 401.443`
- `grad_log_omega_max ~= 9.699`
- `node_count_mid = 5`
- `node_count_last = 5`

Interpretation:

- at the same target mass where the selected `K=1` row fails, this `K=6` row remains a distributed `5`-node bounded state
- compactness is far lower than the `K=1` failure
- the state remains distributed rather than compressing into the same trace family

### 4. feb56dc7 Control

Saved classification:

- `class = TRUE_SATURATED_BOUND_STATE`
- diagnostic label = `SATURATED_BOUND_STATE`

Key metrics:

- `N = 96`
- `T = 6000`
- `er_final_or_last ~= 1.5781`
- `late_energy_slope ~= -2.16e-06`
- `rho_peak_max ~= 1.063`
- `core_radius_min ~= 27.449`
- `compactness_max ~= 62.972`
- `node_count_last = 4`

Interpretation:

- the reference control remains a late stable multi-node attractor
- it provides a stable comparison surface for the threshold cases without changing the solver path

## Cross-Comparison

At target mass `1200`:

- selected `K=1` failure compactness: about `191.588`
- selected `K=6` survivor compactness: about `17.209`

At target mass `1200`:

- selected `K=1` failure peak density: about `4.118`
- selected `K=6` survivor peak density: about `0.987`

At target mass `1200`:

- selected `K=1` failure core-radius minimum: about `4.029`
- selected `K=6` survivor core-radius minimum: about `18.170`

Reading:

- the low-K failure is denser, more compact, and shorter-lived
- the same-mass high-K survivor distributes mass over a wider, lower-compactness late state

## Inference

Supported provisional label:

`DISTRIBUTED_MASS_STABILIZATION_SUPPORTED`

Meaning here:

- the selected high-mass `K=6` branch remains distributed and bounded
- the selected same-mass `K=1` row fails without producing a clean collapse-like signature
- the threshold contrast is consistent with a mass-distribution stabilization hypothesis

## Caveat

- the threshold pilot candidates are still `N=48 / T=4000`
- the `ref_feb56dc7` control is `N=96 / T=6000`
- the low-K survivor below threshold is a `2`-node late state, not a persistent single-node control
- one comparison pack does not define the whole threshold surface

Current disciplined reading remains:

- `K_AND_MASS_EFFECT_SUPPORTED`
- `NO_CLEAN_COLLAPSE_LIKE_SIGNATURE_YET`
- `DISTRIBUTED_MASS_STABILIZATION_SUPPORTED`

## Proposed Action

Best tiny validation set after this comparison:

1. the `K=1` survivor below threshold
2. the `K=1` failure above threshold
3. the same-mass `K=6` survivor
4. the highest-mass `K=6` survivor
5. `ref_feb56dc7`

Those should be replayed at `N=96 / T=6000` before any larger new threshold sweep.
