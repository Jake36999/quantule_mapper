# CL TG-R Robustness Capsule Local Validation Report

## Capsule

- Queue row: `CL_TG_R_ROBUSTNESS_CAPSULE`
- Job id: `cl_tg_r_robustness_capsule`
- Row id: `tg_r_robustness_matrix`
- Manifest: `F:\quantule_mapper\colab_jobs\cl_tg_r_robustness_capsule_manifest.json`
- Generated notebook: `F:\quantule_mapper\colab_jobs\cl_tg_r_robustness_capsule.ipynb`
- Operator instructions: `F:\quantule_mapper\colab_jobs\cl_tg_r_robustness_capsule_operator_instructions.md`
- Driver: `F:\quantule_mapper\colab_jobs\tg_r_robustness_driver.py`
- Baseline summary: `F:\quantule_mapper\colab_jobs\baselines\tg_b2_definitive_force_20260716_baseline_summary.json`
- Drive results path: `/content/drive/MyDrive/QuantuleMapperRuns/cl_tg_r_robustness_capsule`

## Purpose

This capsule runs a compact Phase R robustness matrix for the reviewed TG-B2 A-well attraction result. It imports protected TG-B2/TG-B1S modules unchanged, executes sentinel N/dt/box/epsilon rows, and writes a consolidated robustness verdict.

## Scope Contract

- Run class: `full-campaign`
- Execution scope: `full-fidelity simulation`
- Fidelity requirement: protected modules imported unchanged
- Estimated A100 runtime: `180-300 minutes`
- Operator stop rule: if `J1` shows no new log output for 30 minutes, or total runtime exceeds 6 hours on A100 without `ROBUSTNESS_RUN_COMPLETE.json`/archive progress, stop and return logs for review.
- Boundary: mirror-only TG-B2 robustness; not gravity, UFF, or IRER validation.

## Matrix

| row | purpose |
|---|---|
| `r1_grid_N96_sep3` | grid-refinement sentinel |
| `r1_dt_half_sep3` | timestep-halved sentinel |
| `r1_box_L20_sep3` | larger-box sentinel |
| `r2_eps_003_sep3` | epsilon half-scale point |
| `r2_eps_012_sep3` | epsilon double-scale point |

The reviewed `epsilon_G=0.06`, `N=64`, `L=16`, `dt=0.002`, `sep=3.0` A100 definitive row is used as the baseline reference.

## Local Validation Performed

- Manifest JSON parsed with `python -m json.tool`.
- Baseline JSON parsed with `python -m json.tool`.
- Protected registry JSON parsed with `python -m json.tool`.
- Packager compiled with `python -m py_compile`.
- Driver and packaged source entrypoints/helpers compiled with `python -m py_compile`.
- Driver protected-hash verification passed locally for the four frozen modules it requires.
- Packager dry-run passed.
- Generated notebook was written successfully.
- Generated notebook JSON parsed.
- All generated notebook code cells compiled successfully.
- Control-panel Markdown table was present.
- Control-panel bridge generation code was present.
- `T1`, `T2`, `T3`, and `J1` were present in the dry-run control-panel registry.
- `J1` label includes the runtime estimate: `Run TG-R Robustness (~3-5h)`.
- Runtime env override preserved: `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`.
- No old `0.45` memory cap was found in the generated notebook.
- No `sys.stdout.buffer` usage was found in the generated notebook.

## Packager Dry-Run Summary

- Packager version: `4.5`
- Bundle SHA-256: `8ea2d53d69753b2edbd8b2ccec98bb0741eb9ff15660e7cdd24afe767981bf44`
- Embedded files: `8`
- Control panel actions: `T1`, `T2`, `T3`, `J1`
- A100 required by manifest: yes
- JAX x64 requested by manifest: yes

## Live Checks Not Yet Performed

These require the manual Colab run and returned archive:

- A100 availability in the selected Colab runtime.
- Google Drive mount/write behavior.
- Full Phase R robustness runtime.
- Returned archive integrity.
- Review of `robustness_summary.json`, `robustness_matrix.csv`, per-row `summary.json` files, convergence drift, epsilon linearity, and all gate statuses.
