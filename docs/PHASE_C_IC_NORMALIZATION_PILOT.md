# Phase C IC Normalization Pilot

## Purpose

This pilot tested whether the Phase C K-varied saturation picture was being shaped by:

- initial blob count `K`,
- total initial mass,
- or both together.

The goal was not to establish a final population law. The goal was to separate a `K` effect from the obvious confound that the original `per_blob_fixed` IC family injects more total mass as `K` increases.

## Why IC Normalization Was Needed

The original K-varied hunt used the same per-blob amplitude for all `K`. That means total initial mass scales upward with blob count.

Under `per_blob_fixed`:

- `K=1` starts at much lower total mass
- `K=6` starts at much higher total mass

So any low-K failure or low-K survival result can be confounded by mass scale, not just multiplicity.

This pilot introduced an explicit normalization control:

- `ic_norm = per_blob_fixed`
- `ic_norm = total_mass_fixed`

so we could compare branch behavior when total mass is held constant across `K`.

## Three Arms

All three arms used:

- `N = 48`
- `T = 4000`
- `batch = 8`
- `K in {1,2,3,4,6}`
- `max_batches = 5`
- same Phase C parameter regime:
  - `eta in [-0.02, 0.15]`
  - `a in [0.2, 0.5]`

### Arm A: baseline `per_blob_fixed`

Run:

- [CORE_SAT_HUNT_20260623_113318](F:/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260623_113318)

Contract:

- fixed amplitude per blob
- total mass increases with `K`

### Arm B: `total_mass_fixed` at K=1 mass

Run:

- [CORE_SAT_HUNT_20260623_120758](F:/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260623_120758)

Contract:

- all `K` values rescaled to the `K=1` total initial mass

### Arm C: `total_mass_fixed` at K=6/feb mass

Run:

- [CORE_SAT_HUNT_20260623_123527](F:/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260623_123527)

Contract:

- all `K` values rescaled to the `K=6` / `ref_feb56dc7` total initial mass

## Target Masses

- `K=1` mass: `291.882452`
- `K=6` / feb mass: `2050.293702`

Measured under the current IC family:

- `K=1` initial max density under its own mass: about `1.0015`
- `K=6` initial max density under its own mass: about `1.0325`
- when `K=1` is rescaled to the high target mass, its initial max density rises to about `7.0352`

## Results By Arm

### Arm A: baseline `per_blob_fixed`

Overall counts:

- `TRUE_SATURATED_BOUND_STATE`: `9`
- `NEAR_SATURATED_BOUND_STATE`: `3`
- `LATE_BLOWUP_REJECT`: `17`
- `SPIN_DOWN_REJECT`: `10`
- `TRANSIENT_GROWER_REJECT`: `1`

TRUE final node counts:

- `1`: `1`
- `2`: `5`
- `4`: `2`
- `5`: `1`

Interpretation:

- low-K TRUE cases exist in this small pilot
- high-K still carries the feb-like `4-5` node branch
- because mass scales with `K`, this baseline alone cannot separate multiplicity from mass

### Arm B: `total_mass_fixed` at K=1 mass

Overall counts:

- `TRUE_SATURATED_BOUND_STATE`: `3`
- `NEAR_SATURATED_BOUND_STATE`: `1`
- `LATE_BLOWUP_REJECT`: `12`
- `SPIN_DOWN_REJECT`: `22`
- `TRANSIENT_GROWER_REJECT`: `2`

TRUE final node counts:

- `2`: `1`
- `3`: `1`
- `5`: `1`

Interpretation:

- reducing all cases to the low target mass suppresses many earlier TRUE outcomes
- spin-down becomes much more common
- but TRUE outcomes do not vanish entirely, including one `K=6` five-node branch

### Arm C: `total_mass_fixed` at K=6/feb mass

Overall counts:

- `TRUE_SATURATED_BOUND_STATE`: `6`
- `NEAR_SATURATED_BOUND_STATE`: `2`
- `LATE_BLOWUP_REJECT`: `25`
- `SPIN_DOWN_REJECT`: `5`
- `TRANSIENT_GROWER_REJECT`: `2`

TRUE final node counts:

- `1`: `1`
- `2`: `2`
- `4`: `2`
- `5`: `1`

Interpretation:

- the high target mass does not force all branches into the same outcome
- `K=1` lost all TRUE outcomes in this arm
- `K=6` kept the feb-like `4-5` node branch
- some intermediate `K` values still produced lower-node TRUE cases

## Results By K

### Baseline `per_blob_fixed`

- `K=1`: `1` TRUE, `1` NEAR, `4` BLOWUP, `2` SPIN
- `K=2`: `2` TRUE, `4` BLOWUP, `2` SPIN
- `K=3`: `1` TRUE, `1` NEAR, `1` GROWER, `4` BLOWUP, `1` SPIN
- `K=4`: `2` TRUE, `1` NEAR, `3` BLOWUP, `2` SPIN
- `K=6`: `3` TRUE, `2` BLOWUP, `3` SPIN

TRUE node counts:

- `K=1 -> 2`
- `K=2 -> 1,2`
- `K=3 -> 2`
- `K=4 -> 2,2`
- `K=6 -> 4,4,5`

### `total_mass_fixed` at K=1 mass

- `K=1`: `1` TRUE, `1` NEAR, `4` BLOWUP, `2` SPIN
- `K=2`: `4` BLOWUP, `4` SPIN
- `K=3`: `1` TRUE, `2` BLOWUP, `1` GROWER, `4` SPIN
- `K=4`: `2` BLOWUP, `1` GROWER, `5` SPIN
- `K=6`: `1` TRUE, `7` SPIN

TRUE node counts:

- `K=1 -> 2`
- `K=3 -> 3`
- `K=6 -> 5`

### `total_mass_fixed` at K=6/feb mass

- `K=1`: `8` BLOWUP
- `K=2`: `1` TRUE, `1` NEAR, `5` BLOWUP, `1` GROWER
- `K=3`: `1` TRUE, `1` NEAR, `5` BLOWUP, `1` SPIN
- `K=4`: `1` TRUE, `5` BLOWUP, `1` GROWER, `1` SPIN
- `K=6`: `3` TRUE, `2` BLOWUP, `3` SPIN

TRUE node counts:

- `K=2 -> 1`
- `K=3 -> 2`
- `K=4 -> 2`
- `K=6 -> 4,4,5`

## Verdict

`K_AND_MASS_EFFECT_SUPPORTED`

Why:

- mass clearly matters:
  - low target mass shifts many cases toward spin-down
  - high target mass wipes out `K=1` TRUE cases in this small sample
- `K` still matters after normalization:
  - at the same high target mass, `K=6` still supports feb-like `4-5` node TRUE branches
  - at the same high target mass, lower-`K` cases do not simply become copies of the `K=6` branch

This is the disciplined summary of the pilot:

- `K_AND_MASS_EFFECT_SUPPORTED`

## Caveats

- small `N=48 / T=4000` pilot
- only `8` rows per `K` per arm
- not a final population law
- the three pilot runs were launched before the later `git_dirty` fallback patch, so their `summary.json` files still show `git_dirty: null`
- the pilot is strong enough to motivate targeted follow-up, not to close the question

## Next Actions

1. targeted mass-threshold pilot with:
   - `K in {1,6}`
   - `ic_norm = total_mass_fixed`
   - masses spanning `291.882452` to `2050.293702`
2. mechanism / trace comparison:
   - one `K=1` survivor below threshold
   - one `K=1` failure above threshold
   - one `K=6` survivor at the same or higher target mass
   - `ref_feb56dc7`
3. tiny `N=96 / T=6000` shortlist after the threshold scan, not before
