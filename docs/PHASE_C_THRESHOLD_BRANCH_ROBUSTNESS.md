# Phase C Threshold Branch Robustness

## Purpose

Test whether the restored N96/T6000 threshold shortlist remains stable under small local perturbations in:

- target mass,
- IC seed,
- and nearby parameter values,

without launching a broad search.

This is a branch-local robustness pass, not a new population hunt.

## Method

- Resolution-fair replay grid: `N=96`, `T=6000`
- Total rows: `37`
- Trace capture: `32` sparse snapshots per row
- Parameter perturbation: multiplicative Gaussian jitter with `sigma = 1%`
- Jitter seed: `314159`
- IC seeds: `20260619`, `20260620`
- Cross-resolution rows used explicit raw-target overrides through `jax_scout/core_saturation_replay.py`

Groups tested:

1. `k1_low_mass_branch`
2. `k1_high_mass_failure`
3. `k6_distributed_branch`
4. `k6_high_mass_branch`
5. `feb56dc7_control`

## Overall Results

- Overall class counts: `19 TRUE_SATURATED_BOUND_STATE`, `18 LATE_BLOWUP_REJECT`
- Overall diagnostic labels:
  - `19 SATURATED_BOUND_STATE`
  - `10 DELOCALIZED_GROWTH`
  - `8 FRAGMENTING_BLOWUP`

## Group Verdicts

- `k1_low_mass_branch` -> `K1_LOW_MASS_BRANCH_FRAGILE`
- `k1_high_mass_failure` -> `K1_FAILURE_BOUNDARY_ROBUST`
- `k6_distributed_branch` -> `K6_DISTRIBUTED_BRANCH_ROBUST`
- `k6_high_mass_branch` -> `K6_DISTRIBUTED_BRANCH_ROBUST`
- `feb56dc7_control` -> `FEB_CONTROL_REPRODUCED`

## Evidence

### 1. K=1 low-mass branch

Rows: `12`

- Class counts: `2 TRUE`, `10 LATE_BLOWUP_REJECT`
- Diagnostic labels: `2 SATURATED_BOUND_STATE`, `10 DELOCALIZED_GROWTH`
- Final node counts: `2` for the survivors, `0` for the failures

Mass sensitivity:

- `4000`: `2 TRUE`, `2 BLOWUP`
- `4800`: `4 BLOWUP`
- `5600`: `4 BLOWUP`

Seed sensitivity:

- seed `20260619`: `2 TRUE`, `4 BLOWUP`
- seed `20260620`: `6 BLOWUP`

Interpretation at this stage: the restored low-mass `K=1` branch is real, but narrow. It did not remain stable under nearby seed and mass perturbations.

### 2. K=1 high-mass failure branch

Rows: `8`

- Class counts: `8 LATE_BLOWUP_REJECT`
- Diagnostic labels: `8 FRAGMENTING_BLOWUP`
- Final node counts: all `0`

Mass sensitivity:

- `7200`: `4 BLOWUP`
- `9600`: `4 BLOWUP`

Seed sensitivity:

- seed `20260619`: `4 BLOWUP`
- seed `20260620`: `4 BLOWUP`

This branch remained stable as a failure branch under both seed and jitter perturbations.

### 3. K=6 distributed branch

Rows: `8`

- Class counts: `8 TRUE_SATURATED_BOUND_STATE`
- Diagnostic labels: `8 SATURATED_BOUND_STATE`
- Final node counts: `4` rows with `5` nodes, `4` rows with `6` nodes

Mass sensitivity:

- `9600`: `4 TRUE`
- `12800`: `4 TRUE`

Seed sensitivity:

- seed `20260619`: `4 TRUE`
- seed `20260620`: `4 TRUE`

This branch remained fully stable under the tested perturbations.

### 4. K=6 high-mass branch

Rows: `8`

- Class counts: `8 TRUE_SATURATED_BOUND_STATE`
- Diagnostic labels: `8 SATURATED_BOUND_STATE`
- Final node counts: `4` rows with `5` nodes, `4` rows with `6` nodes

Mass sensitivity:

- `12800`: `4 TRUE`
- `16402.349616`: `4 TRUE`

Seed sensitivity:

- seed `20260619`: `4 TRUE`
- seed `20260620`: `4 TRUE`

This higher-mass distributed branch also remained fully stable under the tested perturbations.

### 5. feb56dc7 control

Rows: `1`

- Class: `TRUE_SATURATED_BOUND_STATE`
- Diagnostic label: `SATURATED_BOUND_STATE`
- Final node count: `4`

The control reproduced cleanly.

## Inference

This robustness layer supports three clean points:

1. The high-mass `K=1` failure branch is robust in the tested neighborhood.
2. The distributed `K=6` branches at matched and higher target mass are robust in the tested neighborhood.
3. The low-mass `K=1` survivor branch is not comparably robust; it appears to be a much smaller local basin.

That keeps the current disciplined interpretation intact:

- `THRESHOLD_PATTERN_N96_SUPPORTED`
- `K1_FAILURE_THRESHOLD_N96_SUPPORTED`
- `K6_DISTRIBUTED_BRANCH_N96_SUPPORTED`

and adds a more local basin statement:

- `K1_LOW_MASS_BRANCH_FRAGILE`
- `K1_FAILURE_BOUNDARY_ROBUST`
- `K6_DISTRIBUTED_BRANCH_ROBUST`

## Caveats

- This is not a population law.
- This is not a new search over broad parameter space.
- The robustness layer only probes a small local neighborhood around shortlisted candidates.
- The low-mass `K=1` branch may still exist as a legitimate branch while being practically fragile.
- No PDE, solver, classifier thresholds, or `gravity/unified_omega.py` behavior were changed for this pass.

## Proposed Action

The next Phase C move should now be structured around the asymmetry revealed here:

1. treat the distributed `K=6` branches as locally robust,
2. treat the high-mass `K=1` branch as a robust failure boundary,
3. treat the low-mass `K=1` survivor branch as real but fragile,
4. and only then decide whether the next discovery layer should emphasize:
   - distributed high-mass branches,
   - tighter local refinement around the low-mass `K=1` pocket,
   - or a balanced follow-up with explicit branch-diversity preservation.
