# Phase C Method Parity Audit

## 1. Purpose

This audit checks whether the completed K-varied Phase C hunt tested the same long-time saturation
question Claude calibrated in the pilot, with only one intended science variable changed:
initial blob count `K`.

Current scope:

- confirm the completed run contract
- confirm classifier parity with Claude's calibrated Phase C pilot
- confirm initial-condition parity and document the important non-parity
- confirm how replayable the saved rows really are
- verify the new row-targeted replay path against controls before broader `N=96/T=6000` validation

## 2. Completed Run Contract

Target run directory:

`F:/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260623_004605`

Supporting runtime log:

`F:/quantule_mapper/runtime_logs/core_sat_hunt_20260623_004559.log`

### Audit Table

| Field | Value |
| --- | --- |
| script used | `jax_scout/core_saturation_search.py` |
| exact command | `python jax_scout/core_saturation_search.py --hours 6 --batch 64 --ic-counts 1,2,3,4,6` |
| git commit | not stamped in run outputs; launch time places the run after `3ce05e5cf`, but the K-varied code path lived in an uncommitted dirty working tree matching the current diff in `jax_scout/core_saturation_search.py` |
| output directory | `F:/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260623_004605` |
| N | `48` |
| T | `4000` |
| batch size | `64` |
| K values | `{1,2,3,4,6}` |
| total rows | `832` |
| elapsed time | `6.30349309987492 h` |
| parameter bounds | base bounds from `jax_scout/gain_bounds.json`; pilot/hunt tighten `param_eta` to `[-0.02, 0.15]` and `param_a` to `[0.20, 0.50]`; all other parameters remain at the gain-bounds ranges |
| random seed policy | parameter RNG `np.random.default_rng(20260622)`; IC seed fixed at `20260619` |
| IC generator function | `jax_scout.afield_current_coupled.multiseed_ic(N, 20260619, K)` |
| classifier function | `classify(...)` in `jax_scout/core_saturation_search.py` |
| TRUE / NEAR thresholds | `SAT_SLOPE = 1.5e-4`, `NEAR_SLOPE = 3.5e-4`, plus `0.5 <= er_fin <= 2.5` |
| blow-up rejection | `not finite` or non-finite `er` or `er_max > 3.0` |
| spin-down rejection | `er_fin < 0.3`, or negative slope outside TRUE/NEAR band |
| fragmentation rejection | `n_mid > 8` or `n_fin > 8` |
| localization rule | reject if `n_fin < 1` or `core_fin < 0.15` |
| ref_feb56dc7 included as control | no; this K-varied run began on `K=1`, so the script branch that injects `ref_feb56dc7` on batch 1 only would not fire |
| ref_feb56dc7 same classification as Claude's smoke calibration | yes under the replay path at `N=96/T=6000`: `TRUE_SATURATED_BOUND_STATE`, `n_mid=4`, `n_fin=4`, `late_slope=-2.316e-06`, `er_fin=1.5781` |

### Operational Integrity

Verdict:

`RUN_COMPLETE_VALID`

Evidence:

- `all_evals.csv` contains `832` non-empty rows with contiguous `idx=0..831`
- `summary.json` reports `n_eval = 832`
- runtime log ends with `HUNT EXIT 0`
- all intended `K` values appear
- class counts from CSV and `summary.json` agree

Warnings:

- K sampling is only approximately balanced because the 6-hour cutoff ended after 13 full batches:
  - `K=1`: `192`
  - `K=2`: `192`
  - `K=3`: `192`
  - `K=4`: `128`
  - `K=6`: `128`
- the run does not stamp its own code commit, command, classifier version, or IC seed into `summary.json`
- the K-varied code path was launched from a dirty working tree, so exact code provenance is inferred, not recorded by the artifact itself

## 3. Classifier Parity

Verdict:

`CLASSIFIER_PARITY_CONFIRMED`

### What Matches Claude's Calibrated Pilot

The current K-varied hunt uses the same saturation classifier Claude calibrated around feb56dc7:

- `TRUE_SATURATED_BOUND_STATE` if:
  - `abs(late_slope) <= 1.5e-4`
  - `0.5 <= er_fin <= 2.5`
  - not already rejected by blow-up / spin-down / fragmentation / delocalization gates
- `NEAR_SATURATED_BOUND_STATE` if:
  - `abs(late_slope) <= 3.5e-4`
  - `0.5 <= er_fin <= 2.5`
  - not already rejected by earlier gates
- `TRANSIENT_GROWER_REJECT` if `late_slope > 3.5e-4`
- `LATE_BLOWUP_REJECT` if non-finite or `er_max > 3.0`
- `SPIN_DOWN_REJECT` if `er_fin < 0.3`, or if the late slope is negative but outside the TRUE/NEAR band
- `FRAGMENTATION_REJECT` if `n_mid > 8` or `n_fin > 8`
- `DELOCALIZED_HALO_REJECT` if `n_fin < 1` or `core_fin < 0.15`

### Specific Parity Checks

| Check | Result |
| --- | --- |
| TRUE slope threshold | same calibrated `1.5e-4` |
| NEAR slope threshold | same calibrated `3.5e-4` |
| spin-down rejection rule | same `er_fin < 0.3`, plus late negative slope fallback |
| blow-up rejection rule | same `er_max > 3.0` or non-finite |
| node-count stability rule | `n_mid == n_fin` is not required; both values are recorded only |
| localization rule | present through `core_fin < 0.15` and `n_fin < 1` |
| negative late slopes can be TRUE | yes, if within slope and energy band |
| low-energy decayed remnants can be TRUE | no; TRUE requires `er_fin >= 0.5` |

### Important Clarification

The classifier does **not** require `n_mid == n_fin`. Claude treated matching mid/final node counts as a
useful preference when ranking candidates, but not as a hard TRUE gate in the search classifier itself.

## 4. Initial-Condition Parity

Verdict:

`IC_ENERGY_SCALES_WITH_K`

### What Stayed the Same

All tested `K` values use the same IC generator family:

- function: `multiseed_ic`
- domain: same `L=10.0` box
- placement rule: blob centers sampled uniformly in `[-L/2, +L/2]`
- blob width: fixed `w = L/12`
- per-blob amplitude: fixed at `1`
- phase/noise family: real positive Gaussians plus `0.01` complex Gaussian noise
- IC seed: fixed `20260619`

So the intended family is consistent. Only the number of summed blobs changes.

### What Did Not Stay the Same

The total initial field strength is **not** normalized across `K`.

Because `multiseed_ic` literally sums `K` blobs of fixed width and unit amplitude, the total initial
density / mass proxy increases strongly with `K`.

#### IC Diagnostic Table (`N=48`, seed `20260619`)

| K | mass proxy `sum(|psi0|^2)` | max density | mean density | peak sanity count |
| --- | ---: | ---: | ---: | ---: |
| `1` | `291.88` | `1.0015` | `0.002639` | `1` |
| `2` | `687.24` | `1.0082` | `0.006214` | `2` |
| `3` | `911.36` | `1.0206` | `0.008241` | `3` |
| `4` | `1362.85` | `1.0317` | `0.012323` | `4` |
| `6` | `2050.29` | `1.0325` | `0.018539` | `6` |

Interpretation:

- peak amplitude stays roughly fixed because each blob has fixed unit amplitude
- total injected density / mass proxy scales sharply with `K`
- therefore low-K versus high-K outcome differences can reflect both multiplicity and injected initial energy budget

This means the completed hunt is **not** a pure "change K only, hold total initial mass fixed" experiment.

## 5. Replayability of Saved Rows

Verdict:

`ROWS_NOT_FULLY_REPLAYABLE`

### Why Not Fully

The current `all_evals.csv` does save:

- parameter vector
- `ic_blobs`
- row `idx`
- saved classification
- saved `N=48`, `T=4000` indirectly through the run context

But it does **not** save, inside the artifact itself:

- IC seed
- parameter RNG seed
- classifier version string
- exact command
- code commit

### What Can Still Be Reconstructed

Under the present repo state, rows are replayable because:

- the IC seed is a fixed code constant: `SEED = 20260619`
- the parameter RNG seed is a fixed code constant: `20260622`
- the IC is not row-specific; it is determined by `(N, K, SEED)`
- the parameter vector is already saved per row, so replay does not need to regenerate the random draw

That makes the rows practically reproducible **given the current codebase**, but not fully self-describing as
archival artifacts.

## 6. Control Replays

### Control A — feb56dc7 reference

Replay path:

```powershell
python jax_scout/core_saturation_replay.py --ref feb56dc7 --N 96 --T 6000 --out <dir>
```

Observed result:

`FEB_CONTROL_REPRODUCED`

Saved summary:

- class: `TRUE_SATURATED_BOUND_STATE`
- `n_mid = 4`
- `n_fin = 4`
- `late_slope = -2.316e-06`
- `er_fin = 1.5781`
- `er_max = 1.5956`

This reproduces the known disciplined label:

`LONG_TIME_STABLE_4_NODE_ATTRACTOR`

### Control B — hunt row `idx 623`

Replay path:

```powershell
python jax_scout/core_saturation_replay.py --csv <all_evals.csv> --idx 623 --N 96 --T 6000 --out <dir>
```

Observed result:

- class: `TRUE_SATURATED_BOUND_STATE`
- `n_mid = 4`
- `n_fin = 4`
- `late_slope = -2.951e-05`
- `er_fin = 0.8313`
- `er_max = 0.9918`

This confirms the new replay path works for ordinary saved hunt rows, not only the built-in reference.

## 7. Current Audit Summary

### Earned Findings

- classifier parity with Claude's calibrated Phase C pilot is confirmed
- IC family parity is confirmed, but IC normalization is **not**: total injected mass/energy scales with `K`
- saved rows are only partially reproducible as artifacts because provenance fields were not stamped into the original run outputs
- the new replay helper reproduces both the feb56dc7 control and the best `K=6` feb-like hunt row at `N=96/T=6000`

### Consequence For Interpretation

The fair current reading remains:

`K_DEPENDENCE_UNCLEAR`

Reason:

- low-K TRUE candidates exist at `N=48/T=4000`
- high-K feb-like candidates also exist
- but the hunt did **not** normalize initial injected mass across `K`
- therefore the completed hunt alone cannot cleanly distinguish multiplicity preference from IC-energy scaling
