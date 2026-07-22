# CL_TG_R_ROBUSTNESS_CAPSULE First-Pass Review

Date: 2026-07-17

Returned archive: `C:\Users\jakem\Downloads\cl_tg_r_robustness_capsule_20260717_112558.tar.gz`

Extracted review location: `F:\quantule_mapper\colab_jobs\Colab_runs\cl_tg_r_robustness_capsule_20260717_112558`

## Verdict

`TG_R_ROBUSTNESS_INCOMPLETE`

The Colab deployment and returned archive are valid, but the robustness matrix did not complete. This is not a notebook-packager or UI failure. Cell 5 failed closed because the main robustness driver returned code `2` after one scientific row failed.

The completed rows are useful partial evidence:

- `r1_grid_N96_sep3`: PASS; attraction sign preserved under grid refinement.
- `r1_dt_half_sep3`: PASS; attraction sign preserved under timestep halving.

The failed row is not evidence against the force result:

- `r1_box_L20_sep3`: failed during Q-ball construction with `RuntimeError: validated Q-ball solve failed`, before force evolution or force measurement.

The epsilon rows were not reached, so Phase R epsilon scaling was not assessed.

## Integrity

Archive-local result integrity was verified from `result_integrity_manifest.json`.

- Manifest entries checked: 61
- Hash/size mismatches: 0
- Archive SHA-256 as received locally: `b679d5dde653aa767be923264a4b25284758b01289835a765908893920604cce`

The returned artifacts are internally trustworthy for first-pass review.

## Cell 5 Failure

The notebook reported:

```text
RuntimeError: Row tg_r_robustness_matrix failed in stage main with return code 2
```

The actual scientific subprocess failure occurred in:

```text
results/cl_tg_r_robustness/logs/r1_box_L20_sep3.log
```

Traceback summary:

```text
gravity_TG_B2_definitive_force.py -> b1s.solve_qball(cfg)
RuntimeError: validated Q-ball solve failed
```

This happened for the larger-box row:

```text
--seps 3.0 --L 20 --T 180
```

Because the row failed at the validated Q-ball solve, it produced no force observable and no `summary.json`.

## Completed Row Evidence

Baseline sep 3.0 reference:

```text
<F_R_well> = -5.4501234166498155e-05
```

### Grid Refinement Sentinel

Row: `r1_grid_N96_sep3`

Command:

```text
python jax_scout/gravity_TG_B2_definitive_force.py --seps 3.0 --N 96 --T 180
```

Result:

```text
<F_R_well> = -5.452931236310377e-05
<F_R_hill> = +5.45282371341913e-05
<F_R_off>  = 0.0
```

Assessment:

- attraction sign preserved;
- A-well/A-hill sign reversal preserved;
- off-null exact;
- gates pass;
- relative drift vs baseline is approximately `5.15e-4` (`0.0515%`), well inside the declared `20%` sentinel tolerance.

### Timestep-Halved Sentinel

Row: `r1_dt_half_sep3`

Command:

```text
python jax_scout/gravity_TG_B2_definitive_force.py --seps 3.0 --dt 0.001 --T 180
```

Result:

```text
<F_R_well> = -5.450123416655863e-05
<F_R_hill> = +5.450015934412122e-05
<F_R_off>  = 0.0
```

Assessment:

- attraction sign preserved;
- A-well/A-hill sign reversal preserved;
- off-null exact;
- gates pass;
- relative drift vs baseline is approximately `1.11e-12`, effectively identical at the reported precision.

## What Was Not Tested

The capsule did not reach:

- `r2_eps_003_sep3`;
- `r2_eps_012_sep3`;
- epsilon through-origin linear fit;
- final `TG_R_ROBUSTNESS_PASS` or `TG_R_ROBUSTNESS_FAIL` decision.

The larger-box question also remains unresolved because `L=20,N=64` failed setup rather than measurement.

## Recommended Repair

Do not rerun the same notebook unchanged.

Prepare a v2 robustness capsule with these changes:

1. Continue independent rows after a row-level subprocess failure, while still marking the final capsule verdict as partial or incomplete if any required row fails.
2. Move the larger-box row after the epsilon rows so a setup failure cannot block epsilon scaling evidence.
3. Replace or split the larger-box sentinel. The likely issue is that `L=20,N=64` changes the Q-ball solve geometry/coarseness enough to fail validation. A candidate repaired larger-box row should couple the larger box to an adequate grid, for example `L=20,N=80` or `L=20,N=96`, but this should be treated as a preregistered v2 change rather than silently substituted into the completed capsule.
4. Preserve the successful `N=96` and `dt=0.001` results as already-returned evidence; do not spend A100 time rerunning them unless the v2 comparison contract requires a single archive containing every row.

## Science Boundary

This partial run supports first-order numerical robustness of the TG-B2 definitive A-well attraction under grid refinement and timestep halving at `sep=3.0`.

It does not yet establish the full Phase R robustness claim because box robustness and epsilon scaling remain unresolved.
