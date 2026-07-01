# Phase C Mass Threshold N96 Validation

## Status

This document records the original higher-resolution replay pass:

- `UNSCALED_RAW_TARGET_N96_REPLAY`

It remains mechanically valid as a replay-path check, but it is **not** the final mass-resolution-fair interpretation of the threshold shortlist.

See also:

- [PHASE_C_MASS_NORMALIZATION_RESOLUTION_AUDIT.md](F:/quantule_mapper/docs/PHASE_C_MASS_NORMALIZATION_RESOLUTION_AUDIT.md)
- [PHASE_C_MASS_THRESHOLD_N96_SCALED_VALIDATION.md](F:/quantule_mapper/docs/PHASE_C_MASS_THRESHOLD_N96_SCALED_VALIDATION.md)

## Purpose

This validation checked whether the small `N=48 / T=4000` mass-threshold pattern survives a higher-fidelity replay path at:

- `N = 96`
- `T = 6000`

It used the existing saved-row replay helper and the existing post-run diagnostic layer. The PDE, solver, `gravity/unified_omega.py`, and Omega-squared behavior were not modified.

At the time of this replay, the saved `N48` raw targets were passed through unchanged at `N96`. The later audit showed that this under-masses the `N96` IC relative to the `N48` continuous-mass target.

## Validation Shortlist

The validated shortlist was:

1. `K=1` below-threshold survivor
   - source run: [CORE_SAT_HUNT_20260623_171758](F:/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260623_171758)
   - idx `4`
   - target mass `500`
2. `K=1` above-threshold failure
   - source run: [CORE_SAT_HUNT_20260623_173417](F:/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260623_173417)
   - idx `2`
   - target mass `1200`
3. `K=6` same-mass survivor
   - source run: [CORE_SAT_HUNT_20260623_173417](F:/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260623_173417)
   - idx `10`
   - target mass `1200`
4. highest-mass `K=6` survivor
   - source run: [CORE_SAT_HUNT_20260623_175018](F:/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260623_175018)
   - idx `10`
   - chosen because it had the lowest `|late_slope|` among the `2050.293702` TRUE rows and stable `5 -> 5` node count at `N=48`
5. `ref_feb56dc7` control

## Output Bundle

Generated validation bundle:

- [CORE_SAT_MASS_THRESHOLD_N96_20260623_202328](F:/quantule_mapper/sweep_runs/CORE_SAT_MASS_THRESHOLD_N96_20260623_202328)
- [mass_threshold_n96_validation.csv](F:/quantule_mapper/sweep_runs/CORE_SAT_MASS_THRESHOLD_N96_20260623_202328/mass_threshold_n96_validation.csv)
- [validation_summary.json](F:/quantule_mapper/sweep_runs/CORE_SAT_MASS_THRESHOLD_N96_20260623_202328/validation_summary.json)

Each candidate directory contains:

- replay `summary.json`
- `probe_data.npz`
- `diagnostic_summary.json`
- `diagnostic_frames.npz`

## Commands

Durable public replay commands used for this validation family:

```powershell
wsl.exe -d Ubuntu -- bash -lc 'source ~/jax_irer/bin/activate && cd /mnt/f/quantule_mapper && python jax_scout/core_saturation_replay.py --csv /mnt/f/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260623_171758/all_evals.csv --idx 4 --N 96 --T 6000 --out /mnt/f/quantule_mapper/sweep_runs/CORE_SAT_MASS_THRESHOLD_N96_20260623_202328/k1_below_threshold_survivor'
```

```powershell
wsl.exe -d Ubuntu -- bash -lc 'source ~/jax_irer/bin/activate && cd /mnt/f/quantule_mapper && python jax_scout/core_saturation_replay.py --csv /mnt/f/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260623_173417/all_evals.csv --idx 2 --N 96 --T 6000 --out /mnt/f/quantule_mapper/sweep_runs/CORE_SAT_MASS_THRESHOLD_N96_20260623_202328/k1_above_threshold_failure'
```

```powershell
wsl.exe -d Ubuntu -- bash -lc 'source ~/jax_irer/bin/activate && cd /mnt/f/quantule_mapper && python jax_scout/core_saturation_replay.py --csv /mnt/f/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260623_173417/all_evals.csv --idx 10 --N 96 --T 6000 --out /mnt/f/quantule_mapper/sweep_runs/CORE_SAT_MASS_THRESHOLD_N96_20260623_202328/k6_same_mass_survivor'
```

```powershell
wsl.exe -d Ubuntu -- bash -lc 'source ~/jax_irer/bin/activate && cd /mnt/f/quantule_mapper && python jax_scout/core_saturation_replay.py --csv /mnt/f/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260623_175018/all_evals.csv --idx 10 --N 96 --T 6000 --out /mnt/f/quantule_mapper/sweep_runs/CORE_SAT_MASS_THRESHOLD_N96_20260623_202328/k6_highest_mass_survivor'
```

```powershell
wsl.exe -d Ubuntu -- bash -lc 'source ~/jax_irer/bin/activate && cd /mnt/f/quantule_mapper && python jax_scout/core_saturation_replay.py --ref feb56dc7 --N 96 --T 6000 --out /mnt/f/quantule_mapper/sweep_runs/CORE_SAT_MASS_THRESHOLD_N96_20260623_202328/ref_feb56dc7_control'
```

Post-run diagnostic metrics in this bundle were computed from the same saved candidates using the existing `jax_scout/core_saturation_collapse_diag.py` replay path. The one-off batch wrapper used to assemble the five-case table was analysis-only and was not kept as a durable repo tool.

## Side-By-Side Results

| candidate | K | target mass | N48 class | N48 diagnostic | N96 class | N96 diagnostic | final nodes | mode matched |
|---|---:|---:|---|---|---|---|---:|---|
| below-threshold survivor | 1 | 500 | `TRUE_SATURATED_BOUND_STATE` | `SATURATED_BOUND_STATE` | `DELOCALIZED_HALO_REJECT` | `FRAGMENTING_BLOWUP` | 2 | no |
| above-threshold failure | 1 | 1200 | `LATE_BLOWUP_REJECT` | `FRAGMENTING_BLOWUP` | `LATE_BLOWUP_REJECT` | `FRAGMENTING_BLOWUP` | 0 | yes |
| same-mass survivor | 6 | 1200 | `TRUE_SATURATED_BOUND_STATE` | `SATURATED_BOUND_STATE` | `SPIN_DOWN_REJECT` | `SPIN_DOWN_DECAY` | 0 | no |
| highest-mass survivor | 6 | 2050.293702 | `TRUE_SATURATED_BOUND_STATE` | `SATURATED_BOUND_STATE` | `SPIN_DOWN_REJECT` | `SPIN_DOWN_DECAY` | 0 | no |
| `ref_feb56dc7` control | 6 | control | `REFERENCE_CONTROL` | `SATURATED_BOUND_STATE` | `TRUE_SATURATED_BOUND_STATE` | `SATURATED_BOUND_STATE` | 4 | yes |

## Per-Candidate Details

### 1. K=1 Below-Threshold Survivor

- source: `CORE_SAT_HUNT_20260623_171758`, idx `4`
- `K = 1`
- target mass `500`
- `N48` class: `TRUE_SATURATED_BOUND_STATE`
- `N48` diagnostic label: `SATURATED_BOUND_STATE`
- `N96` class: `DELOCALIZED_HALO_REJECT`
- `N96` diagnostic label: `FRAGMENTING_BLOWUP`
- `n_mid = 2`, `n_fin = 2`
- late slope `~ -2.27e-05`
- `er_final ~ 0.7713`
- `er_max ~ 0.9491`
- peak density max `~ 0.2112`
- compactness max `~ 11.7521`
- core radius min `~ 8.0210`
- high-k fraction max `~ 0.05065`

Reading:

- this row no longer satisfies the localization criterion at `N=96 / T=6000`
- it does not support `K1_LOW_MASS_BRANCH_N96_SUPPORTED`

### 2. K=1 Above-Threshold Failure

- source: `CORE_SAT_HUNT_20260623_173417`, idx `2`
- `K = 1`
- target mass `1200`
- `N48` class: `LATE_BLOWUP_REJECT`
- `N48` diagnostic label: `FRAGMENTING_BLOWUP`
- `N96` class: `LATE_BLOWUP_REJECT`
- `N96` diagnostic label: `FRAGMENTING_BLOWUP`
- `n_mid = 0`, `n_fin = 0`
- late slope `~ +2.88e-02`
- `er_final = er_max ~ 97.0505`
- peak density max `~ 2.8869`
- compactness max `~ 191.7305`
- core radius min `~ 4.0833`
- high-k fraction max `~ 0.05065`
- time-to-failure `~ 2250`

Reading:

- the high-mass low-K failure persists cleanly at higher fidelity
- this does support `K1_FAILURE_THRESHOLD_N96_SUPPORTED`
- it remains `FRAGMENTING_BLOWUP`, not `COLLAPSE_LIKE_RUNAWAY`

### 3. K=6 Same-Mass Survivor

- source: `CORE_SAT_HUNT_20260623_173417`, idx `10`
- `K = 6`
- target mass `1200`
- `N48` class: `TRUE_SATURATED_BOUND_STATE`
- `N48` diagnostic label: `SATURATED_BOUND_STATE`
- `N96` class: `SPIN_DOWN_REJECT`
- `N96` diagnostic label: `SPIN_DOWN_DECAY`
- `n_mid = 0`, `n_fin = 0`
- late slope `~ -1.81e-05`
- `er_final ~ 0.00383`
- `er_max ~ 0.9912`
- peak density max `~ 0.07717`
- compactness max `~ 3.4782`
- core radius min `~ 30.2758`
- high-k fraction max `~ 0.00792`

Reading:

- this row does not remain on the distributed saturated branch at `N=96 / T=6000`
- it spins down instead
- this does not support `K6_DISTRIBUTED_BRANCH_N96_SUPPORTED`

### 4. K=6 Highest-Mass Survivor

- source: `CORE_SAT_HUNT_20260623_175018`, idx `10`
- `K = 6`
- target mass `2050.293702`
- `N48` class: `TRUE_SATURATED_BOUND_STATE`
- `N48` diagnostic label: `SATURATED_BOUND_STATE`
- `N96` class: `SPIN_DOWN_REJECT`
- `N96` diagnostic label: `SPIN_DOWN_DECAY`
- `n_mid = 0`, `n_fin = 0`
- late slope `~ -2.01e-05`
- `er_final ~ 0.00427`
- `er_max ~ 0.9913`
- peak density max `~ 0.1328`
- compactness max `~ 5.9429`
- core radius min `~ 30.2953`
- high-k fraction max `~ 0.00792`

Reading:

- this higher-mass `K=6` candidate also weakens at `N=96 / T=6000`
- it does not preserve the `N48` distributed saturated mode

### 5. feb56dc7 Control

- source: `ref_feb56dc7`
- `N96` class: `TRUE_SATURATED_BOUND_STATE`
- `N96` diagnostic label: `SATURATED_BOUND_STATE`
- `n_mid = 4`, `n_fin = 4`
- late slope `~ -2.32e-06`
- `er_final ~ 1.5781`
- `er_max ~ 1.5956`
- peak density max `~ 1.0633`
- compactness max `~ 62.9716`
- core radius min `~ 27.4488`
- high-k fraction max `~ 0.00792`

Reading:

- the control reproduces cleanly
- the replay path itself remains trustworthy for this validation step

## Provisional Validation Verdict

Supported:

- `K1_FAILURE_THRESHOLD_N96_SUPPORTED`

Historical unscaled replay label:

- `THRESHOLD_PATTERN_N96_UNCLEAR_PENDING_SCALED_REPLAY`

Why:

- the validated low-mass `K=1` survivor does not remain a TRUE saturated branch at `N=96 / T=6000`
- the validated `K=6` distributed survivors at `1200` and `2050.293702` both spin down at `N=96 / T=6000`
- the only shortlisted pattern that clearly survives is the high-mass `K=1` failure branch
- the reference control still reproduces, which argues against a broken replay path

But this replay is now known to be under-massed at `N=96`.

Not earned from this shortlist:

- `K1_LOW_MASS_BRANCH_N96_SUPPORTED`
- `K6_DISTRIBUTED_BRANCH_N96_SUPPORTED`
- `THRESHOLD_PATTERN_N96_SUPPORTED`

## Interpretation

Current disciplined reading after the shortlist:

- `K_AND_MASS_EFFECT_SUPPORTED`
- `NO_CLEAN_COLLAPSE_LIKE_SIGNATURE_YET`
- `DISTRIBUTED_MASS_STABILIZATION_SUPPORTED` remains supported as an `N=48 / T=4000` threshold-pilot inference
- but this specific replay is now understood as an under-massed `N96` test

The fair update for this document alone is:

- `THRESHOLD_PATTERN_N96_UNCLEAR_PENDING_SCALED_REPLAY`

That does **not** erase the earlier pilot. It means the unscaled raw-target replay should not be treated as the final resolution comparison.

The corrected scaled-target replay is documented separately in:

- [PHASE_C_MASS_THRESHOLD_N96_SCALED_VALIDATION.md](F:/quantule_mapper/docs/PHASE_C_MASS_THRESHOLD_N96_SCALED_VALIDATION.md)

## Caveats

- this is a tiny shortlist, not a broad new N96 campaign
- the low-mass `K=1` row still ends with `2` nodes, but fails the localization criterion
- the two `K=6` rows weaken by spin-down rather than by high-growth failure
- this replay used unscaled raw `N48` targets at `N96`
- no stronger language is warranted from this validation step

## Next Actions

Most justified next steps now are small and targeted:

1. search locally around the `K=1`, mass `500` neighborhood for more robust localized low-mass survivors
2. search locally around the `K=6`, masses `1200` to `2050.293702` for higher-fidelity distributed survivors
3. keep using `ref_feb56dc7` as the stable control in every validation batch

What does **not** look justified yet:

- a broad new threshold hunt
- any change to the PDE or solver path
- any non-physical limiter or cap
