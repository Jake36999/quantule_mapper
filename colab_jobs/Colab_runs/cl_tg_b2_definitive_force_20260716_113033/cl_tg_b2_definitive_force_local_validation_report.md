# CL TG-B2 Definitive Force Local Validation Report

## Capsule

- Queue row: `CL_TG_B2_DEFINITIVE_FORCE`
- Job id: `cl_tg_b2_definitive_force`
- Row id: `tg_b2_definitive_force_seps_3_4`
- Manifest: `F:\quantule_mapper\colab_jobs\cl_tg_b2_definitive_force_manifest.json`
- Generated notebook: `F:\quantule_mapper\colab_jobs\cl_tg_b2_definitive_force.ipynb`
- Operator instructions: `F:\quantule_mapper\colab_jobs\cl_tg_b2_definitive_force_operator_instructions.md`
- Drive results path: `/content/drive/MyDrive/QuantuleMapperRuns/cl_tg_b2_definitive_force`

## Purpose

This capsule runs the queued TG-B2 definitive dynamical body-force campaign on Colab A100. It is intended to settle the attract/repel sign using the clean Gravity-D body-force observable, not the previous momentum/COM proxy labels.

## Scope Contract

- Run class: `full-campaign`
- Execution scope: `full-fidelity simulation`
- Fidelity requirement: `scientific source unchanged; A-well branch; frozen TG-B1S untouched; single-node norm`
- Estimated A100 runtime: `150-210 minutes`
- Operator stop rule: if `J1` shows no new log output for 30 minutes, or total runtime exceeds 5 hours on A100 without `RUN_COMPLETE.json`/archive progress, stop and return logs for review.
- Boundary: do not substitute proxy momentum/COM observables, shorten the run, relax gates, or modify scientific source for Colab convenience.

## Command

```text
python jax_scout/gravity_TG_B2_definitive_force.py --seps 3.0,4.0 --T 180 --out /content/qm_job/results/cl_tg_b2_definitive
```

## Packaged Source Allowlist

- `jax_scout/__init__.py`
- `jax_scout/phase_d_c3_wave.py`
- `jax_scout/gravity_TG_B1S_state_load_feedback_gpu.py`
- `jax_scout/gravity_TG_B2_two_node_awell.py`
- `jax_scout/gravity_TG_B2_definitive_force.py`

## Expected Scientific Outputs

- `results/cl_tg_b2_definitive/config.json`
- `results/cl_tg_b2_definitive/RUN_COMPLETE.json`
- `results/cl_tg_b2_definitive/summary.json`
- `results/cl_tg_b2_definitive/ROW_sep3.00_STARTED.json`
- `results/cl_tg_b2_definitive/ROW_sep3.00_COMPLETE.json`
- `results/cl_tg_b2_definitive/ROW_sep4.00_STARTED.json`
- `results/cl_tg_b2_definitive/ROW_sep4.00_COMPLETE.json`
- `results/cl_tg_b2_definitive/scalars_sep3.00_{off,well,hill}.csv`
- `results/cl_tg_b2_definitive/scalars_sep4.00_{off,well,hill}.csv`

## Local Validation Performed

- Manifest JSON parsed with `python -m json.tool`.
- Packager compiled with `python -m py_compile`.
- Packaged source entrypoints/helpers compiled with `python -m py_compile`.
- Packager dry-run passed.
- Generated notebook was written successfully.
- Generated notebook JSON parsed.
- All generated notebook code cells compiled successfully.
- Control-panel Markdown table was present.
- Control-panel bridge generation code was present.
- `T1`, `T2`, `T3`, and `J1` were present in the dry-run control-panel registry.
- `J1` label includes the runtime estimate: `Run TG-B2 Definitive Force (~2.5-3.5h)`.
- Runtime env override preserved: `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`.
- No old `0.45` memory cap was found in the generated notebook.
- No `sys.stdout.buffer` usage was found in the generated notebook.

## Packager Dry-Run Summary

- Packager version: `4.5`
- Bundle SHA-256: `31177fb86b6eeefdd3e172d6a099f5dcbff7c3f91af1d3b17beb88eaddb3d543`
- Embedded files: `5`
- Control panel actions: `T1`, `T2`, `T3`, `J1`
- A100 required by manifest: yes
- JAX x64 requested by manifest: yes

## Live Checks Not Yet Performed

These require the manual Colab run and returned archive:

- A100 availability in the selected Colab runtime.
- Google Drive mount/write behavior.
- Full TG-B2 definitive-force runtime.
- Returned archive integrity.
- Review of `summary.json`, especially `<F_R>` sign, well/hill antisymmetry, off-null behavior, gates, and cross-separation consistency.
