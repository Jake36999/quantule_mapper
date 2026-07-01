# Phase C Mass Threshold N96 Scaled Validation

## Purpose

This validation reran the earlier `N=96 / T=6000` threshold shortlist using resolution-scaled raw targets, so the replayed ICs match the `N=48` continuous-mass targets rather than the raw `N=48` grid sums.

This is a corrected replay family, not an exact raw-row replay.

Earned replay label:

- `N96_TARGET_SCALING_CORRECTED`
- replay kind: `RESOLUTION_SCALED_TARGET_REPLAY`

## Corrected Targets

```text
N48 target 500 -> N96 raw target 4000.0
N48 target 1200 -> N96 raw target 9600.0
N48 target 2050.293702 -> N96 raw target 16402.349616
```

## Output Bundle

Generated bundle:

- [CORE_SAT_MASS_THRESHOLD_N96_SCALED_20260623_222546](F:/quantule_mapper/sweep_runs/CORE_SAT_MASS_THRESHOLD_N96_SCALED_20260623_222546)
- [mass_threshold_n96_scaled_validation.csv](F:/quantule_mapper/sweep_runs/CORE_SAT_MASS_THRESHOLD_N96_SCALED_20260623_222546/mass_threshold_n96_scaled_validation.csv)
- [validation_summary.json](F:/quantule_mapper/sweep_runs/CORE_SAT_MASS_THRESHOLD_N96_SCALED_20260623_222546/validation_summary.json)

## Exact Scaled Replay Commands

```powershell
wsl.exe -d Ubuntu -- bash -lc 'source ~/jax_irer/bin/activate && cd /mnt/f/quantule_mapper && python jax_scout/core_saturation_replay.py --csv /mnt/f/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260623_171758/all_evals.csv --idx 4 --N 96 --T 6000 --target-initial-mass-override 4000.0 --out /mnt/f/quantule_mapper/sweep_runs/CORE_SAT_MASS_THRESHOLD_N96_SCALED_20260623_222546/k1_below_threshold_survivor'
```

```powershell
wsl.exe -d Ubuntu -- bash -lc 'source ~/jax_irer/bin/activate && cd /mnt/f/quantule_mapper && python jax_scout/core_saturation_replay.py --csv /mnt/f/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260623_173417/all_evals.csv --idx 2 --N 96 --T 6000 --target-initial-mass-override 9600.0 --out /mnt/f/quantule_mapper/sweep_runs/CORE_SAT_MASS_THRESHOLD_N96_SCALED_20260623_222546/k1_above_threshold_failure'
```

```powershell
wsl.exe -d Ubuntu -- bash -lc 'source ~/jax_irer/bin/activate && cd /mnt/f/quantule_mapper && python jax_scout/core_saturation_replay.py --csv /mnt/f/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260623_173417/all_evals.csv --idx 10 --N 96 --T 6000 --target-initial-mass-override 9600.0 --out /mnt/f/quantule_mapper/sweep_runs/CORE_SAT_MASS_THRESHOLD_N96_SCALED_20260623_222546/k6_same_mass_survivor'
```

```powershell
wsl.exe -d Ubuntu -- bash -lc 'source ~/jax_irer/bin/activate && cd /mnt/f/quantule_mapper && python jax_scout/core_saturation_replay.py --csv /mnt/f/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260623_175018/all_evals.csv --idx 10 --N 96 --T 6000 --target-initial-mass-override 16402.349616 --out /mnt/f/quantule_mapper/sweep_runs/CORE_SAT_MASS_THRESHOLD_N96_SCALED_20260623_222546/k6_highest_mass_survivor'
```

```powershell
wsl.exe -d Ubuntu -- bash -lc 'source ~/jax_irer/bin/activate && cd /mnt/f/quantule_mapper && python jax_scout/core_saturation_replay.py --ref feb56dc7 --N 96 --T 6000 --out /mnt/f/quantule_mapper/sweep_runs/CORE_SAT_MASS_THRESHOLD_N96_SCALED_20260623_222546/ref_feb56dc7_control'
```

## Side-By-Side Results

| candidate | K | saved N48 target | corrected N96 raw target | N48 class | previous unscaled N96 class | corrected scaled N96 class | diagnostic label | final nodes | restores N48 mode |
|---|---:|---:|---:|---|---|---|---|---:|---|
| K6 same-mass survivor | 6 | 1200 | 9600 | `TRUE_SATURATED_BOUND_STATE` | `SPIN_DOWN_REJECT` | `TRUE_SATURATED_BOUND_STATE` | `SATURATED_BOUND_STATE` | 5 | yes |
| K6 highest-mass survivor | 6 | 2050.293702 | 16402.349616 | `TRUE_SATURATED_BOUND_STATE` | `SPIN_DOWN_REJECT` | `TRUE_SATURATED_BOUND_STATE` | `SATURATED_BOUND_STATE` | 5 | yes |
| K1 below-threshold survivor | 1 | 500 | 4000 | `TRUE_SATURATED_BOUND_STATE` | `DELOCALIZED_HALO_REJECT` | `TRUE_SATURATED_BOUND_STATE` | `SATURATED_BOUND_STATE` | 2 | yes |
| K1 above-threshold failure | 1 | 1200 | 9600 | `LATE_BLOWUP_REJECT` | `LATE_BLOWUP_REJECT` | `LATE_BLOWUP_REJECT` | `FRAGMENTING_BLOWUP` | 0 | yes |
| `ref_feb56dc7` control | 6 | control | control | `REFERENCE_CONTROL` | `TRUE_SATURATED_BOUND_STATE` | `TRUE_SATURATED_BOUND_STATE` | `SATURATED_BOUND_STATE` | 4 | yes |

## Per-Candidate Results

### K6 Same-Mass Survivor

- source run: `CORE_SAT_HUNT_20260623_173417`
- idx `10`
- `K = 6`
- saved `N48` target: `1200`
- corrected `N96` raw target: `9600`
- dx-weighted target mass: `10.850694444444446`
- previous unscaled `N96` class: `SPIN_DOWN_REJECT`
- corrected scaled `N96` class: `TRUE_SATURATED_BOUND_STATE`
- diagnostic label: `SATURATED_BOUND_STATE`
- final node count: `5`
- late slope: `-2.7455526825268815e-05`
- `er_final ~ 0.59199`
- `er_max ~ 0.99182`
- peak density max: `~ 0.87263`
- compactness max: `~ 27.82596`
- core radius min: `~ 30.49912`
- high-k fraction max: `~ 0.0079203`

Reading:

- `SCALED_REPLAY_RESTORES_K6_BRANCH`
- `K6_DISTRIBUTED_BRANCH_N96_SUPPORTED`

### K6 Highest-Mass Survivor

- source run: `CORE_SAT_HUNT_20260623_175018`
- idx `10`
- `K = 6`
- saved `N48` target: `2050.293702`
- corrected `N96` raw target: `16402.349616`
- dx-weighted target mass: `18.53925873480903`
- previous unscaled `N96` class: `SPIN_DOWN_REJECT`
- corrected scaled `N96` class: `TRUE_SATURATED_BOUND_STATE`
- diagnostic label: `SATURATED_BOUND_STATE`
- final node count: `5`
- late slope: `-2.0111227789063404e-05`
- `er_final ~ 0.81633`
- `er_max ~ 0.99195`
- peak density max: `~ 1.05484`
- compactness max: `~ 54.23094`
- core radius min: `~ 30.22264`
- high-k fraction max: `~ 0.0079203`

Reading:

- the high-mass K6 distributed branch also restores under the scaled replay
- this strengthens `K6_DISTRIBUTED_BRANCH_N96_SUPPORTED`

### K1 Below-Threshold Survivor

- source run: `CORE_SAT_HUNT_20260623_171758`
- idx `4`
- `K = 1`
- saved `N48` target: `500`
- corrected `N96` raw target: `4000`
- dx-weighted target mass: `4.521122685185186`
- previous unscaled `N96` class: `DELOCALIZED_HALO_REJECT`
- corrected scaled `N96` class: `TRUE_SATURATED_BOUND_STATE`
- diagnostic label: `SATURATED_BOUND_STATE`
- final node count: `2`
- late slope: `1.0475208162483371e-04`
- `er_final = er_max ~ 0.94492`
- peak density max: `~ 1.68423`
- compactness max: `~ 94.01662`
- core radius min: `~ 8.28287`
- high-k fraction max: `~ 0.0506500`

Reading:

- `K1_LOW_MASS_BRANCH_N96_SUPPORTED`
- the below-threshold low-K branch restores under mass-fair replay

### K1 Above-Threshold Failure

- source run: `CORE_SAT_HUNT_20260623_173417`
- idx `2`
- `K = 1`
- saved `N48` target: `1200`
- corrected `N96` raw target: `9600`
- dx-weighted target mass: `10.850694444444446`
- previous unscaled `N96` class: `LATE_BLOWUP_REJECT`
- corrected scaled `N96` class: `LATE_BLOWUP_REJECT`
- diagnostic label: `FRAGMENTING_BLOWUP`
- final node count: `0`
- late slope: `1.652711817164762e-02`
- `er_final = er_max ~ 64.67298`
- peak density max: `~ 4.04214`
- compactness max: `~ 225.63990`
- core radius min: `~ 8.17966`
- high-k fraction max: `~ 0.0506500`
- time-to-failure: `~ 900`

Reading:

- `K1_FAILURE_THRESHOLD_N96_SUPPORTED`
- the high-mass low-K failure persists under the corrected replay as well

### feb56dc7 Control

- corrected scaled `N96` class: `TRUE_SATURATED_BOUND_STATE`
- diagnostic label: `SATURATED_BOUND_STATE`
- final node count: `4`
- late slope: `-2.3162181715396658e-06`

Reading:

- the control remains stable under the same replay path

## Provisional Interpretation

The corrected replay reverses the earlier under-mass conclusion for the saturating branches.

Earned labels from the scaled replay:

- `N96_TARGET_SCALING_CORRECTED`
- `SCALED_REPLAY_RESTORES_K6_BRANCH`
- `K6_DISTRIBUTED_BRANCH_N96_SUPPORTED`
- `K1_LOW_MASS_BRANCH_N96_SUPPORTED`
- `K1_FAILURE_THRESHOLD_N96_SUPPORTED`
- `THRESHOLD_PATTERN_N96_SUPPORTED`

What changed:

- the earlier unscaled `N96` threshold replay was mechanically valid but under-massed
- once the raw target is scaled by the resolution factor, the shortlisted low-mass K1 survivor and both K6 distributed survivors all restore their `N48` mode
- the high-mass K1 failure still remains a failure

This means the earlier label:

- `THRESHOLD_PATTERN_WEAKENS_AT_N96`

is **not** supported as the fair interpretation of the threshold shortlist.

The fair corrected reading is:

- `THRESHOLD_PATTERN_N96_SUPPORTED`

## Caveats

- this is still a tiny shortlist, not a large new `N96` population survey
- the corrected replay establishes resolution-fair support for these shortlisted branches, not a global law over the full search space
- no stronger language is warranted
