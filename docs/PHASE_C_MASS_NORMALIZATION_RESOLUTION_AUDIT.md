# Phase C Mass Normalization Resolution Audit

## Purpose

This audit checked whether the earlier `N=96 / T=6000` threshold replays used a mass target that is resolution-fair relative to the `N=48 / T=4000` threshold pilot.

Concern:

- the threshold pilot stored `target_initial_mass` as a raw grid sum
- doubling grid resolution roughly octuples that raw sum for the same continuous-mass field
- if the same raw target is replayed at `N=96`, the field is under-massed relative to the `N=48` continuous target

## Contract Audit

### Search

In [core_saturation_search.py](</F:/quantule_mapper/jax_scout/core_saturation_search.py:111>):

- `measure_ic()` defines:
  - `initial_mass = float(np.sum(rho))`
- no `dx^3` factor is applied

In [core_saturation_search.py](</F:/quantule_mapper/jax_scout/core_saturation_search.py:129>):

- `build_ic()` rescales by:
  - `sqrt(target_initial_mass / raw_mass)`
- again, no `dx^3` factor appears

### Replay

In [core_saturation_replay.py](</F:/quantule_mapper/jax_scout/core_saturation_replay.py:121>):

- the replay path passes the saved `target_initial_mass` through unchanged unless an explicit override is supplied
- before this audit, that meant the `N=96` threshold replays used the raw `N=48` target values directly

### Diagnostic Replay

In [core_saturation_collapse_diag.py](</F:/quantule_mapper/jax_scout/core_saturation_collapse_diag.py:526>):

- the diagnostic replay path rebuilds the IC through the same `build_ic()` path
- therefore it inherits the same raw-target convention

## Measured IC Masses

Same IC family and same seed:

```text
N48 K1 raw mass = 291.88245200678637
N48 K6 raw mass = 2050.293702162646
N96 K1 raw mass = 2406.422573896147
N96 K6 raw mass = 16534.010659945343

N48 K1 dx-weighted mass = 2.6392727503507163
N48 K6 dx-weighted mass = 18.539258736279717
N96 K1 dx-weighted mass = 2.7199329222458988
N96 K6 dx-weighted mass = 18.688072667943146

raw_mass_ratio_K1 = 8.24449211438102
raw_mass_ratio_K6 = 8.064215698709557
expected_ratio = 8
```

## Interpretation

The raw-mass ratios are very close to the resolution-volume ratio:

- doubling linear resolution from `48` to `96` multiplies voxel count by `8`
- the raw `sum(|psi|^2)` mass proxy rises by about the same factor
- the `dx^3` weighted mass stays nearly constant, which is the expected continuous-mass behavior

That means:

- `target_initial_mass` is a raw grid-sum target
- it is **not** a resolution-invariant integral

## Consequence For The Earlier N96 Threshold Replay

The earlier `N=96` threshold validation reused the saved raw `N=48` targets:

- `500`
- `1200`
- `2050.293702`

at `N=96` without scaling them upward by the resolution factor.

That made the earlier `N=96` threshold replay mechanically valid but not mass-resolution fair.

## Corrected Resolution-Scaled Targets

Using:

- `dx48 = L / 48`
- `dx96 = L / 96`
- `target_integral = target_raw_N48 * dx48^3`
- `target_raw_N96 = target_integral / dx96^3`

the corrected raw `N=96` targets are:

```text
N48 target 500 -> N96 raw target 4000.0
N48 target 1200 -> N96 raw target 9600.0
N48 target 2050.293702 -> N96 raw target 16402.349616
```

## Conclusion

Earned labels:

- `RAW_MASS_SCALING_CONFIRMED`
- `N96_UNSCALED_TARGETS_UNDERMASS_CONFIRMED`

Interim threshold label after this audit:

- `THRESHOLD_PATTERN_N96_UNCLEAR_PENDING_SCALED_REPLAY`

The high-mass low-K failure result remains separately supported:

- `K1_FAILURE_THRESHOLD_N96_SUPPORTED`
