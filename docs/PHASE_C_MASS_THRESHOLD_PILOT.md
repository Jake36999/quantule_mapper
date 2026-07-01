# Phase C Mass Threshold Pilot

## Purpose

This pilot tested a narrow question:

`Does K=6 shift the high-mass failure threshold upward relative to K=1 when total initial mass is fixed?`

The goal was not discovery volume. The goal was to map a small threshold slice under:

- `N = 48`
- `T = 4000`
- `ic_norm = total_mass_fixed`
- `K in {1, 6}`

using the same Phase C parameter regime as the recent normalization work.

## Method

Shared regime:

- `param_eta in [-0.02, 0.15]`
- `param_a in [0.2, 0.5]`
- `batch = 8`
- `8` samples per `K` at each target mass

Target masses:

- `291.882452`
- `500`
- `800`
- `1200`
- `1600`
- `2050.293702`

Balanced launch:

```powershell
$masses = 291.882452,500,800,1200,1600,2050.293702
foreach ($mass in $masses) {
  wsl.exe -d Ubuntu -- bash -lc "source ~/jax_irer/bin/activate && cd /mnt/f/quantule_mapper && stdbuf -oL -eL python jax_scout/core_saturation_search.py --hours 1 --N 48 --T 4000 --batch 8 --ic-counts '1,6' --max-batches 2 --ic-norm total_mass_fixed --target-initial-mass $mass"
}
```

Run directories:

- [CORE_SAT_HUNT_20260623_170944](F:/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260623_170944) for `291.882452`
- [CORE_SAT_HUNT_20260623_171758](F:/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260623_171758) for `500`
- [CORE_SAT_HUNT_20260623_172609](F:/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260623_172609) for `800`
- [CORE_SAT_HUNT_20260623_173417](F:/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260623_173417) for `1200`
- [CORE_SAT_HUNT_20260623_174215](F:/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260623_174215) for `1600`
- [CORE_SAT_HUNT_20260623_175018](F:/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260623_175018) for `2050.293702`

Replay-backed post-run diagnostic bundle:

- [CORE_SAT_THRESHOLD_DIAG_20260623_180519](F:/quantule_mapper/sweep_runs/CORE_SAT_THRESHOLD_DIAG_20260623_180519)
- [threshold_diagnostics.csv](F:/quantule_mapper/sweep_runs/CORE_SAT_THRESHOLD_DIAG_20260623_180519/threshold_diagnostics.csv)
- [threshold_summary.json](F:/quantule_mapper/sweep_runs/CORE_SAT_THRESHOLD_DIAG_20260623_180519/threshold_summary.json)

Diagnostic command:

```powershell
wsl.exe -d Ubuntu -- bash -lc 'source ~/jax_irer/bin/activate && cd /mnt/f/quantule_mapper && stdbuf -oL -eL python jax_scout/core_saturation_collapse_diag.py --mode threshold-pilot --n-snap 24 --outdir /mnt/f/quantule_mapper/sweep_runs/CORE_SAT_THRESHOLD_DIAG_20260623_180519'
```

## Overall Results

Across `96` configs:

- `TRUE_SATURATED_BOUND_STATE`: `8`
- `NEAR_SATURATED_BOUND_STATE`: `3`
- `SPIN_DOWN_REJECT`: `27`
- `LATE_BLOWUP_REJECT`: `57`
- `TRANSIENT_GROWER_REJECT`: `1`

Replay-backed diagnostic labels across the same `96` configs:

- `SATURATED_BOUND_STATE`: `8`
- `SPIN_DOWN_DECAY`: `27`
- `FRAGMENTING_BLOWUP`: `16`
- `DELOCALIZED_GROWTH`: `33`
- `INCONCLUSIVE_FAILURE_TRACE`: `12`
- `HIGH_K_NUMERICAL_ARTIFACT_SUSPECT`: `0`
- `COLLAPSE_LIKE_RUNAWAY`: `0`

## Results By K And Target Mass

### Class counts

| K | target mass | TRUE | NEAR | SPIN | BLOWUP | GROWER |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 291.882452 | 1 | 1 | 2 | 4 | 0 |
| 6 | 291.882452 | 0 | 1 | 6 | 1 | 0 |
| 1 | 500 | 1 | 1 | 1 | 5 | 0 |
| 6 | 500 | 0 | 0 | 5 | 2 | 1 |
| 1 | 800 | 0 | 0 | 1 | 7 | 0 |
| 6 | 800 | 0 | 0 | 4 | 4 | 0 |
| 1 | 1200 | 0 | 0 | 1 | 7 | 0 |
| 6 | 1200 | 2 | 0 | 2 | 4 | 0 |
| 1 | 1600 | 0 | 0 | 1 | 7 | 0 |
| 6 | 1600 | 2 | 0 | 2 | 4 | 0 |
| 1 | 2050.293702 | 0 | 0 | 0 | 8 | 0 |
| 6 | 2050.293702 | 2 | 0 | 2 | 4 | 0 |

### Diagnostic labels

| K | target mass | SAT | FRAG | DELOC | SPIN | INCONCLUSIVE |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 291.882452 | 1 | 3 | 2 | 2 | 0 |
| 6 | 291.882452 | 0 | 0 | 0 | 6 | 2 |
| 1 | 500 | 1 | 3 | 3 | 1 | 0 |
| 6 | 500 | 0 | 1 | 0 | 5 | 2 |
| 1 | 800 | 0 | 2 | 5 | 1 | 0 |
| 6 | 800 | 0 | 2 | 0 | 4 | 2 |
| 1 | 1200 | 0 | 2 | 5 | 1 | 0 |
| 6 | 1200 | 2 | 1 | 1 | 2 | 2 |
| 1 | 1600 | 0 | 1 | 6 | 1 | 0 |
| 6 | 1600 | 2 | 0 | 2 | 2 | 2 |
| 1 | 2050.293702 | 0 | 1 | 7 | 0 | 0 |
| 6 | 2050.293702 | 2 | 0 | 2 | 2 | 2 |

## Final Node-Count Pattern

Search-level TRUE outcomes show the clearest branch split:

- `K=1`
  - `291.882452`: one TRUE, final node count `2`
  - `500`: one TRUE, final node count `2`
  - `800` and above: no TRUE outcomes
- `K=6`
  - `291.882452`, `500`, `800`: no TRUE outcomes
  - `1200`: two TRUE outcomes, final node counts `5, 5`
  - `1600`: two TRUE outcomes, final node counts `5, 4`
  - `2050.293702`: two TRUE outcomes, final node counts `5, 4`

This is not IC tracking in the simple sense. It is a branch split:

- low-mass `K=1` can hold a low-node saturated branch
- higher-mass `K=6` can hold a distributed `4-5` node branch
- intermediate masses do not simply interpolate into the same outcome family

## Compactness / High-k / Time-To-Failure

Median compactness proxy by `K x target_mass`:

- `K=1`: about `56.6`, `94.5`, `191.5`, `191.0`, `191.2`, `245.0`
- `K=6`: about `4.09`, `7.00`, `13.33`, `19.69`, `29.75`, `37.67`

Median time-to-failure where a failure time exists:

- `K=1`: about `416.7`, `166.7`, `166.7`, `166.7`, `166.7`, `166.7`
- `K=6`: about `3000`, `1916.7`, `2250`, `1583.3`, `1583.3`, `1666.7`

High-k spectral fraction stayed nearly constant inside each IC family:

- `K=1`: about `0.05475`
- `K=6`: about `0.00901`

That means the high-k fraction is informative as a branch-family contrast here, but not yet a sharp intra-family threshold discriminator across target mass.

## Provisional Threshold Verdicts

Supported in this pilot:

- `K1_LOW_MASS_SATURATION_SUPPORTED`
- `K1_HIGH_MASS_FRAGMENTATION_SUPPORTED`
- `K6_DISTRIBUTED_STABILIZATION_SUPPORTED`
- `K6_THRESHOLD_SHIFT_SUPPORTED`

Why:

- `K=1` shows TRUE survivors at `291.882452` and `500`, then loses them by `800`
- `K=1` high-mass failures include recurring `FRAGMENTING_BLOWUP` traces, although `DELOCALIZED_GROWTH` is numerically more common overall
- `K=6` shows no TRUEs in the low-mass corner, then gains stable distributed TRUE branches at `1200`, `1600`, and `2050.293702`
- the `K=6` stable branch carries much lower compactness than the `K=1` failures at the same or higher target mass

Not supported in this pilot:

- `NUMERICAL_ARTIFACT_SUSPECT`
- `COLLAPSE_LIKE_RUNAWAY`

## Interpretation

Disciplined summary:

- `K_AND_MASS_EFFECT_SUPPORTED`
- `NO_CLEAN_COLLAPSE_LIKE_SIGNATURE_YET`
- `DISTRIBUTED_MASS_STABILIZATION_SUPPORTED`

Threshold-specific reading:

At fixed total initial mass, the current pilot supports an upward shift in stable high-mass behavior for `K=6` relative to `K=1`. In this `N=48 / T=4000` slice:

- `K=1` supports a low-node saturated branch only in the low-mass corner
- `K=6` supports distributed `4-5` node saturated branches only in the higher-mass corner
- the transition is not symmetric, and it is not well described by a single universal node-count law

## Caveats

- small pilot: `96` total configs
- only `8` samples per `K x target_mass`
- post-run diagnostics used `24` replay snapshots per row, which is enough for branch mapping but not a fine-grained instability atlas
- this is still `N=48 / T=4000` evidence
- `K=1` high-mass failures are mixed between `FRAGMENTING_BLOWUP` and `DELOCALIZED_GROWTH`, so the failure mechanism is not one clean trace family

## Next Action

Best next validation shortlist after this pilot:

1. one `K=1` survivor just below the apparent threshold
2. one `K=1` failure just above the threshold
3. one `K=6` survivor at the same mass as the `K=1` failure
4. one highest-mass `K=6` survivor
5. `ref_feb56dc7`

Those should be replayed at `N=96 / T=6000`, not expanded into a broad new hunt yet.
