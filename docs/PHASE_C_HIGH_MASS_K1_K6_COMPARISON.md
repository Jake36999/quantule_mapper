# Phase C High-Mass K1/K6 Comparison

## Purpose

This comparison asks a narrow matched question:

`At the same target initial mass, does low-K failure persist while high-K cases remain distributed and saturated?`

It is a replay-backed comparison only. The PDE, solver path, and `gravity/unified_omega.py` behavior were not modified.

## Compared Cases

All replays used:

- `N = 96`
- `T = 6000`
- the same saved solver path

Candidates:

- high-target `K=1` failure: idx `2` from [CORE_SAT_HUNT_20260623_123527](F:/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260623_123527)
- high-target `K=6` stable rows: idx `32`, `33`, `39` from [CORE_SAT_HUNT_20260623_123527](F:/quantule_mapper/sweep_runs/CORE_SAT_HUNT_20260623_123527)
- `ref_feb56dc7`

Generated outputs:

- bundle: [CORE_SAT_HIGH_MASS_COMPARE_20260623_170000](F:/quantule_mapper/sweep_runs/CORE_SAT_HIGH_MASS_COMPARE_20260623_170000)
- table: [high_mass_k1_k6_comparison.csv](F:/quantule_mapper/sweep_runs/CORE_SAT_HIGH_MASS_COMPARE_20260623_170000/high_mass_k1_k6_comparison.csv)

## Run Command

```powershell
wsl.exe -d Ubuntu -- bash -lc 'source ~/jax_irer/bin/activate && cd /mnt/f/quantule_mapper && stdbuf -oL -eL python jax_scout/core_saturation_collapse_diag.py --mode high-mass-comparison --N-override 96 --T-override 6000 --n-snap 60 --failure-n-snap 120 --outdir /mnt/f/quantule_mapper/sweep_runs/CORE_SAT_HIGH_MASS_COMPARE_20260623_170000'
```

## Evidence

### High-Target K=1 Failure: idx 2

- `K = 1`
- `ic_norm = total_mass_fixed`
- `target_initial_mass = 2050.293702162646`
- original class: `LATE_BLOWUP_REJECT`
- replay diagnostic label: `FRAGMENTING_BLOWUP`
- `time_to_failure = 750`
- `time_to_blowup = 750`
- `node_count_last = 2`
- `rho_peak_max ~= 2.896`
- `core_radius_min ~= 5.574`
- `compactness_max ~= 191.697`
- `omega2_min_min ~= 121.865`
- `grad_log_omega_max ~= 15.546`
- `high_k_fraction_max ~= 0.05065`

Interpretation:

- the high-mass low-K failure persisted at higher fidelity;
- it did **not** promote to `COLLAPSE_LIKE_RUNAWAY`;
- it remained a split-first failure trace, so the disciplined label stays `FRAGMENTING_BLOWUP`.

### High-Target K=6 Stable Rows

All three replays remained `SATURATED_BOUND_STATE`.

#### idx 32

- `K = 6`
- `ic_norm = total_mass_fixed`
- `target_initial_mass = 2050.293702162646`
- `node_count_last = 4`
- `rho_peak_max ~= 0.132`
- `core_radius_min ~= 28.619`
- `compactness_max ~= 5.943`
- `high_k_fraction_max ~= 0.00792`

#### idx 33

- `K = 6`
- `ic_norm = total_mass_fixed`
- `target_initial_mass = 2050.293702162646`
- `node_count_last = 6`
- `rho_peak_max ~= 0.488`
- `core_radius_min ~= 27.929`
- `compactness_max ~= 29.655`
- `high_k_fraction_max ~= 0.00792`

#### idx 39

- `K = 6`
- `ic_norm = total_mass_fixed`
- `target_initial_mass = 2050.293702162646`
- `node_count_last = 5`
- `rho_peak_max ~= 0.132`
- `core_radius_min ~= 29.128`
- `compactness_max ~= 5.943`
- `high_k_fraction_max ~= 0.00792`

### feb56dc7 Control

- replay diagnostic label: `SATURATED_BOUND_STATE`
- `node_count_last = 4`
- `rho_peak_max ~= 1.063`
- `core_radius_min ~= 27.449`
- `compactness_max ~= 62.972`
- `high_k_fraction_max ~= 0.00792`

## Cross-Comparison

At the same target mass:

- the low-K failure case had much higher peak density than the K=6 stable rows
- the low-K failure case had a much smaller minimum core-radius proxy
- the low-K failure case had much larger compactness
- the low-K failure case had substantially larger high-k spectral fraction
- the K=6 rows remained distributed multi-node states rather than collapsing into the same failure trace

Compactness comparison:

- idx `2` (`K=1`): about `191.697`
- idx `32` (`K=6`): about `5.943`
- idx `33` (`K=6`): about `29.655`
- idx `39` (`K=6`): about `5.943`

Peak-density comparison:

- idx `2` (`K=1`): about `2.896`
- idx `32` (`K=6`): about `0.132`
- idx `33` (`K=6`): about `0.488`
- idx `39` (`K=6`): about `0.132`

Core-radius comparison:

- idx `2` (`K=1`): about `5.574`
- idx `32` (`K=6`): about `28.619`
- idx `33` (`K=6`): about `27.929`
- idx `39` (`K=6`): about `29.128`

High-k fraction comparison:

- idx `2` (`K=1`): about `0.05065`
- stable `K=6` rows: about `0.00792`

## Provisional Interpretation

Matched-comparison label:

`DISTRIBUTED_MASS_STABILIZATION_SUPPORTED`

Meaning:

- at the same target initial mass, the selected low-K case remains a failure trace;
- selected high-K cases remain distributed saturated states;
- the high-K branch carries the mass at much lower compactness and much lower high-k fraction.

This does **not** upgrade the failure label to collapse-like runaway. The disciplined combined reading remains:

- `NO_CLEAN_COLLAPSE_LIKE_SIGNATURE_YET`
- `K_AND_MASS_EFFECT_SUPPORTED`
- `DISTRIBUTED_MASS_STABILIZATION_SUPPORTED`

## Caveats

- This is still a targeted comparison, not a broad new hunt.
- One low-K failure case does not prove a universal low-K rule.
- Stable `K=6` rows vary in final node count (`4`, `5`, `6`), so the distributed branch itself is not a single rigid endpoint.
- Any future limiter experiment would need to be explicitly separated as:

`NON_PHYSICAL_LIMITER_SENSITIVITY_TEST`

That was not implemented here.

## Next Actions

Most justified next options now:

1. targeted high-mass threshold scan for `K=1` and `K=6`
2. more low-K `N=96 / T=6000` matched replays at the same target mass
3. larger balanced normalized pilot only after the threshold picture is clearer
