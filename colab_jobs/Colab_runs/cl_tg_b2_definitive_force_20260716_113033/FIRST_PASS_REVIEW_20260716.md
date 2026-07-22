# CL TG-B2 Definitive Force First-Pass Review

## Scope

This is a first-pass review of the returned Colab capsule at:

```text
F:\quantule_mapper\colab_jobs\Colab_runs\cl_tg_b2_definitive_force_20260716_113033
```

It checks capsule completion, archive integrity, environment, and whether the returned CSVs support the reported `summary.json` values. It is not yet a full scientific write-up.

## Capsule Status

- Capsule status: `RUN_COMPLETE_CAPSULE`
- Completed row: `tg_b2_definitive_force_seps_3_4`
- Failed rows: none
- Script return code: `0`
- Runtime: `2718.65 s` (`0.755 h`, about 45.3 minutes)
- Archive size: `199447 bytes`
- Archive SHA-256 sidecar matched local archive hash:
  - `f678d77ec5e02382005148ccab5a84d5fae4df3c6dc8a71482f433514ab4180e`
- `result_integrity_manifest.json` verification: `0` mismatches

## Environment

- Backend: JAX GPU
- Device: `NVIDIA A100-SXM4-40GB`
- JAX/JAXLIB: `0.7.2` / `0.7.2`
- Python: `3.12.13`
- NumPy: `2.0.2`
- x64 enabled: yes
- float64 kernel dtype: `float64`
- Runtime memory policy applied:
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false`
  - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`

## Packaged Command

```text
/usr/bin/python3 -u jax_scout/gravity_TG_B2_definitive_force.py --seps 3.0,4.0 --T 180 --out /content/qm_job/results/cl_tg_b2_definitive
```

Configuration:

- `N=64`
- `L=16`
- `T=180`
- `dt=0.002`
- `sample_dt=0.5`
- first `40%` discarded for the settled-window average
- `cool_T=0.0`

## Reported Verdict

`summary.json` reports:

```text
TG_B2_DEFINITIVE_AWELL_ATTRACTION_CONFIRMED
```

This should be read as the script-level verdict for the configured mirror test. It is not, by itself, a broader physics promotion.

## CSV-Recomputed Observable Summary

The first-pass recomputation from the six scalar CSV files matches `summary.json`.

| sep | n | t_end | <F_R_well> | <F_R_hill> | <F_R_off> | well/hill antisym rel | charge drift max | sep_min | gates |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 3.0 | 361 | 180.0 | `-5.4501234166498155e-05` | `+5.45001593440529e-05` | `0.0` | `1.97e-05` | `2.13e-13` | `2.9824448062083118` | pass |
| 4.0 | 361 | 180.0 | `-4.8525703120004155e-05` | `+4.8524721724240265e-05` | `0.0` | `2.02e-05` | `8.40e-14` | `2.6633355426968413` | pass |

Sign convention:

```text
F_R < 0 means attraction.
```

First-pass reading:

- A-well arm is negative at both separations.
- A-hill arm is positive at both separations.
- Off arm is exactly zero in the returned CSV recomputation.
- Well/hill antisymmetry is extremely tight at about `2e-5` relative.
- Charge drift is far below the declared `1e-2` tolerance.
- Nodes remain distinct under the declared `sep > 1.2` gate.

## Cautions For Full Review

- The configured force observable passes cleanly, but the separation histories still show substantial dynamical motion/breathing. For example, the `sep=4.0` arms reach minima around `2.66`, while remaining safely above the merge threshold.
- The first-pass result supports the clean body-force observable for this configured A-well/A-hill mirror test. It does not automatically settle broader long-range, gravity, UFF, or production-model claims.
- Full review should inspect the time series plots or statistics for settled-window stability, not only the means.

## First-Pass Conclusion

The returned Colab capsule is technically valid and internally consistent. The archive verifies, the A100/JAX/x64 environment is recorded, all declared outputs are present, and independent CSV recomputation supports the script verdict:

```text
TG_B2_DEFINITIVE_AWELL_ATTRACTION_CONFIRMED
```

Recommended next step: proceed to a deeper scientific review focused on time-series stability, breathing sensitivity, and whether this result closes or only narrows the TG-B2 dynamical sign question.
