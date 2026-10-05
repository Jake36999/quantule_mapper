# CL_TG_R_ROBUSTNESS_CAPSULE_V2 First-Pass Review

Date: 2026-07-17

Returned results folder: `F:\quantule_mapper\colab_jobs\results\cl_tg_r_robustness_capsule_v2`

## Verdict

`TG_R_ROBUSTNESS_PASS`

The v2 Colab capsule completed successfully. It preserved the two successful v1 rows as frozen evidence, ran the missing epsilon rows, ran the repaired larger-box row, and produced no failed rows.

This is a first-pass infrastructure and numerical-result review. The result supports Phase R robustness for the TG-B2 definitive A-well attraction branch under the declared sentinel checks, pending any deeper science-review wording Claude/Jake want to bank in the main results docs.

## Integrity

Folder-level integrity was verified from `result_integrity_manifest.json`.

- Manifest entries checked: 73
- Hash/size mismatches: 0
- Capsule row return code: 0
- Capsule runtime: 6365.63 seconds, about 1h46m
- `RUN_COMPLETE_CAPSULE.json`: present
- `ROBUSTNESS_RUN_COMPLETE.json`: present
- Original `.tar.gz` archive and Drive sidecar were not present in the local results parent, so no archive-file SHA was recorded in this review.

Protected scientific source verification passed:

| file | status |
|---|---|
| `jax_scout/phase_d_c3_wave.py` | hash match |
| `jax_scout/gravity_TG_B1S_state_load_feedback_gpu.py` | hash match |
| `jax_scout/gravity_TG_B2_two_node_awell.py` | hash match |
| `jax_scout/gravity_TG_B2_definitive_force.py` | hash match |

Runtime preflight:

- JAX backend: `gpu`
- Device: `cuda:0`
- JAX x64: enabled
- Float64 kernel dtype: `float64`
- JAX/JAXLIB: `0.7.2` / `0.7.2`
- Resource profile: `a100_aggressive`
- `XLA_PYTHON_CLIENT_MEM_FRACTION`: `0.90`
- `XLA_PYTHON_CLIENT_PREALLOCATE`: `false`

## Rows

Baseline sep 3.0 reference:

```text
<F_R_well> = -5.4501234166498155e-05
```

### Preserved v1 Evidence

| row | source | <F_R_well> | drift vs baseline | gate status |
|---|---|---:|---:|---|
| `r1_grid_N96_sep3` | v1 returned capsule | `-5.452931236310377e-05` | `0.0005151848662203447` | PASS |
| `r1_dt_half_sep3` | v1 returned capsule | `-5.450123416655863e-05` | `1.1094980403251824e-12` | PASS |

### v2 Rows

| row | runtime | <F_R_well> | <F_R_hill> | <F_R_off> | status |
|---|---:|---:|---:|---:|---|
| `r2_eps_003_sep3` | 1364.71s | `-2.7250482735304782e-05` | `+2.725021402968578e-05` | `0.0` | PASS |
| `r2_eps_012_sep3` | 1373.91s | `-0.00010900354307761069` | `+0.00010899924378803952` | `0.0` | PASS |
| `r1_box_L20_N96_sep3` | 3626.93s | `-5.316540398870711e-05` | `+5.316427664538773e-05` | `0.0` | PASS |

All v2 rows reported:

- `well_negative = true`
- `sign_reverses = true`
- `off_null_ok = true`
- `charge_ok = true`
- `nodes_distinct = true`
- `gates_pass = true`

## Epsilon Scaling

The epsilon scaling check used:

| epsilon_G | <F_R_well> |
|---:|---:|
| `0.03` | `-2.7250482735304782e-05` |
| `0.06` | `-5.4501234166498155e-05` |
| `0.12` | `-0.00010900354307761069` |

Through-origin fit:

```text
slope = -0.0009083605132995932
R^2   = 0.9999999998998101
SSE   = 3.47213579526618e-19
```

Assessment:

- epsilon signs are all attractive;
- linearity gate passed;
- well/hill antisymmetry remains clean;
- off-null remains exact.

## Repaired Larger-Box Sentinel

The failed v1 larger-box row was:

```text
L=20, N=64
```

It failed before force measurement during validated Q-ball setup.

The repaired v2 row was:

```text
L=20, N=96
```

It completed and measured:

```text
<F_R_well> = -5.316540398870711e-05
drift_vs_baseline_sep3 = 0.024510090426762856
```

Assessment:

- attraction sign preserved;
- drift is about `2.45%`, within the declared `20%` sentinel tolerance;
- setup failure from v1 is resolved for the repaired row;
- this is still a sentinel, not a full box-convergence campaign.

## Science Boundary

This result strengthens the TG-B2 definitive A-well attraction result under:

- grid refinement (`N=96`, v1);
- timestep halving (`dt=0.001`, v1);
- repaired larger-box sentinel (`L=20,N=96`, v2);
- epsilon scaling (`0.03`, `0.06`, `0.12`, v2).

It does not by itself establish gravity, UFF, IRER validation, long-range behavior, or a full continuum/box-convergence theorem. It is a compact Phase R robustness pass for the already-reviewed mirror TG-B2 body-force observable.

## Recommended Queue Update

Move `CL_TG_R_ROBUSTNESS_CAPSULE` from `QUEUED(v2 packaged; awaiting Colab upload)` to `REVIEW(v2 returned; first-pass PASS)` or `DONE` after primary science-review wording is banked in the gravity maturity docs.
