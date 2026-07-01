# Phase C Collapse/Runaway Diagnostic

## Purpose

This pass asks a narrow post-run question:

`Does high-mass low-K blowup look like localized collapse/runaway, or more generic failure such as splitting, delocalized growth, or early non-finite loss?`

It is a replay-backed analysis layer only. The PDE, solver path, and `gravity/unified_omega.py` behavior were not modified.

## Scope

Analyzed candidate groups:

1. High-target `K=1` blowup rows from [CORE_SAT_HUNT_20260623_123527](F:/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260623_123527)
2. High-target `K=6` TRUE rows from [CORE_SAT_HUNT_20260623_123527](F:/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260623_123527)
3. Low-target `K=1` TRUE / SPIN rows from [CORE_SAT_HUNT_20260623_120758](F:/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260623_120758)
4. Baseline per-blob `K=1` and `K=6` rows from [CORE_SAT_HUNT_20260623_113318](F:/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260623_113318)
5. `ref_feb56dc7`

Generated outputs:

- diagnostic bundle: [CORE_SAT_COLLAPSE_DIAG_20260623_160000](F:/quantule_mapper/sweep_runs/CORE_SAT_COLLAPSE_DIAG_20260623_160000)
- table: [collapse_runaway_diagnostics.csv](F:/quantule_mapper/sweep_runs/CORE_SAT_COLLAPSE_DIAG_20260623_160000/collapse_runaway_diagnostics.csv)

## Method

Replay path:

- same saved parameter vector
- same IC family
- same reconstructed `K`
- same replay seed path
- same solver path used by the search
- sparse snapshot capture only

Per-snapshot derived diagnostics:

- total mass / energy proxy
- peak density
- half-mass core-radius proxy
- mass inside fixed radius
- compactness proxy
- `Omega^2` minimum
- `|grad log Omega|`
- curvature proxy from saved geometry contract
- node count
- participation ratio
- high-k spectral fraction

Allowed output labels only:

- `COLLAPSE_LIKE_RUNAWAY`
- `FRAGMENTING_BLOWUP`
- `DELOCALIZED_GROWTH`
- `HIGH_K_NUMERICAL_ARTIFACT_SUSPECT`
- `SATURATED_BOUND_STATE`
- `SPIN_DOWN_DECAY`
- `INCONCLUSIVE_FAILURE_TRACE`

## Run Command

Executed in the WSL JAX environment:

```powershell
wsl.exe -d Ubuntu -- bash -lc 'source ~/jax_irer/bin/activate && cd /mnt/f/quantule_mapper && stdbuf -oL -eL python jax_scout/core_saturation_collapse_diag.py --outdir /mnt/f/quantule_mapper/sweep_runs/CORE_SAT_COLLAPSE_DIAG_20260623_160000'
```

## Evidence

Total analyzed candidates: `31`

Diagnostic label counts:

- `SATURATED_BOUND_STATE`: `9`
- `SPIN_DOWN_DECAY`: `7`
- `FRAGMENTING_BLOWUP`: `6`
- `INCONCLUSIVE_FAILURE_TRACE`: `9`
- `COLLAPSE_LIKE_RUNAWAY`: `0`
- `DELOCALIZED_GROWTH`: `0`
- `HIGH_K_NUMERICAL_ARTIFACT_SUSPECT`: `0`

By group:

- High-target `K=1` blowup (`8` rows):
  - `7` `INCONCLUSIVE_FAILURE_TRACE`
  - `1` `FRAGMENTING_BLOWUP`
- High-target `K=6` TRUE (`3` rows):
  - `3` `SATURATED_BOUND_STATE`
- Low-target `K=1` TRUE / SPIN (`3` rows):
  - `1` `SATURATED_BOUND_STATE`
  - `2` `SPIN_DOWN_DECAY`
- Baseline `K=1` and `K=6` (`16` rows):
  - `4` `SATURATED_BOUND_STATE`
  - `5` `SPIN_DOWN_DECAY`
  - `5` `FRAGMENTING_BLOWUP`
  - `2` `INCONCLUSIVE_FAILURE_TRACE`
- `ref_feb56dc7`:
  - `1` `SATURATED_BOUND_STATE`

## High-Target K=1 Failures

### Evidence

- The high-target `K=1` slice began from a much larger initial peak density than the corresponding `K=6` high-target slice:
  - `K=1` initial max density about `7.04`
  - `K=6` initial max density about `1.03`
- Most high-target `K=1` failures went non-finite too early to provide a reliable localization history. Those remain `INCONCLUSIVE_FAILURE_TRACE`.
- One informative failure, idx `2`, was not single-core runaway. It was:
  - `FRAGMENTING_BLOWUP`
  - `time_to_blowup = 1200`
  - `node_count_mid = 2`
  - `node_count_last = 2`
  - `split_before_blowup = True`
  - `high_k_fraction_max ~= 0.055`

### Inference

The current pass does **not** support a clean claim that high-target `K=1` failures are predominantly localized collapse-like runaway.

### Caveat

Most of the high-target `K=1` failures became non-finite before enough finite snapshots accumulated to support a stronger geometric diagnosis. So the right reading is not "not collapse," but "not yet shown to be collapse-like."

## High-Target K=6 TRUE Cases

### Evidence

All three selected high-target `K=6` TRUE cases remained `SATURATED_BOUND_STATE`.

Typical diagnostic ranges among those three:

- `rho_peak_max`: about `1.03` to `1.58`
- `core_radius_min`: about `6.19` to `14.87`
- `compactness_max`: about `33.8` to `83.5`
- `high_k_fraction_max`: about `0.009`

Example high-target `K=6` stable row, idx `32`:

- `late_energy_slope ~= -9.13e-05`
- `node_count_mid = 5`
- `node_count_last = 5`
- `split_before_blowup = False`

### Inference

At the same total target mass where `K=1` often failed, the `K=6` branch can still distribute that mass across a stable multi-node state with much lower peak density and lower high-k content.

## Low-Target K=1 Branch

### Evidence

For the low-target normalized `K=1` subset:

- idx `4`: `SATURATED_BOUND_STATE`
- idx `6`: `SPIN_DOWN_DECAY`
- idx `7`: `SPIN_DOWN_DECAY`

The stable low-target case remained a modest 2-node branch rather than a high-compactness failure trace.

### Inference

Lower target mass clearly shifts some `K=1` behavior from failure toward spin-down or bounded saturation, which supports a genuine mass effect.

## Baseline Comparison

### Evidence

Baseline per-blob `K=1` and `K=6` both show mixed outcomes:

- `K=1` baseline includes:
  - `SATURATED_BOUND_STATE`
  - `SPIN_DOWN_DECAY`
  - `FRAGMENTING_BLOWUP`
  - `INCONCLUSIVE_FAILURE_TRACE`
- `K=6` baseline includes:
  - `SATURATED_BOUND_STATE`
  - `SPIN_DOWN_DECAY`
  - `FRAGMENTING_BLOWUP`

### Inference

"High K" is not a uniform stability region, and "low K" is not a uniform collapse region. The failure/stability landscape is branched in both directions.

## feb56dc7 Control

The control remained `SATURATED_BOUND_STATE`.

Selected diagnostic features:

- `node_count_mid = 4`
- `node_count_last = 4`
- `late_energy_slope ~= -2.09e-06`
- `high_k_fraction_max ~= 0.0079`

This is consistent with using `ref_feb56dc7` as the stable comparison trace.

## Provisional Reading

### Evidence

- No candidate in this pass reached `COLLAPSE_LIKE_RUNAWAY`.
- No candidate triggered `HIGH_K_NUMERICAL_ARTIFACT_SUSPECT`.
- High-target `K=1` failures were mostly too abrupt/non-finite for a strong geometric verdict, with one explicit split-first trace.
- High-target `K=6` TRUE cases remained stably multi-node.

### Inference

Current cautious reading:

`NO_CLEAN_COLLAPSE_LIKE_SIGNATURE_YET`

Expanded prose:

- the current high-target `K=1` failures do **not yet** show a clean, repeated collapse-like runaway signature under the diagnostic rules used here;
- at least one informative high-mass `K=1` case is better described as `FRAGMENTING_BLOWUP`;
- the remaining high-mass `K=1` failures are best kept as `INCONCLUSIVE_FAILURE_TRACE`;
- meanwhile, high-target `K=6` stable cases remain consistent with a distributed multi-node branch at the same total mass.

Combined with the normalization pilot, the current disciplined labels are:

- `NO_CLEAN_COLLAPSE_LIKE_SIGNATURE_YET`
- `K_AND_MASS_EFFECT_SUPPORTED`

### Caveat

This is still a snapshot-based replay diagnostic on the current `N=48 / T=4000` pilot regime for most selected rows. It is useful for triage, not proof.

## Conclusion

The pilot does not yet support a localized collapse-like `K=1` runaway signature. It supports a mass/structure interaction: high mass destabilizes low-`K` cases, while `K=6` can distribute the same target mass into stable multi-node states.

## Next Actions

Most justified next steps, in order:

1. `N=96 / T=6000` replay of the most informative high-target `K=1` failures
   - start with idx `2` from [CORE_SAT_HUNT_20260623_123527](F:/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260623_123527)
   - compare against stable high-target `K=6` rows such as idx `32`, `33`, and `39`
2. targeted mass-threshold scan for `K=1` and `K=6`
   - this now looks better motivated than a broad balanced hunt if the goal is specifically to localize the `K=1` failure boundary
3. only after that, a larger balanced normalized pilot

If later we want to probe numerical sensitivity with artificial limiters, that should be a separate explicitly marked experiment:

`NON_PHYSICAL_LIMITER_SENSITIVITY_TEST`

It was not implemented here.
