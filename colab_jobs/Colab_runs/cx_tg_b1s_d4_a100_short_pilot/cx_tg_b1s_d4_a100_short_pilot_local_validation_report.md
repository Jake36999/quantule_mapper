# CX TG-B1S D4 A100 Short Pilot Local Validation Report

## Capsule

- Queue row: `CX_TG_B1S_D4_A100_SHORT_PILOT`
- Run class: `short-scientific-pilot`
- Job id: `cx_tg_b1s_d4_a100_short_pilot`
- Row id: `d4_dt_half_2p_n32`
- Manifest: `F:\quantule_mapper\colab_jobs\cx_tg_b1s_d4_a100_short_pilot_manifest.json`
- Generated notebook: `F:\quantule_mapper\colab_jobs\cx_tg_b1s_d4_a100_short_pilot.ipynb`
- Operator instructions: `F:\quantule_mapper\colab_jobs\cx_tg_b1s_d4_a100_short_pilot_operator_instructions.md`
- Drive result path: `/content/drive/MyDrive/QuantuleMapperRuns/cx_tg_b1s_d4_a100_short_pilot`

## Intended J1 Command

```text
python -u jax_scout/gravity_TG_B1S_D4_rows_gpu.py --out /content/qm_job/results/cx_tg_b1s_d4_a100_short_pilot --reference-run /content/qm_job/sweep_runs/TG_B1S_DRIFT_DECOMPOSITION_GPU_20260714_184736 --cases dt_half --d4-periods 2 --N 32
```

The packager invokes this through `run_registered_job` with row id `d4_dt_half_2p_n32`.

## Runtime Contract

- Estimated A100 runtime: 5-30 minutes.
- Basis: one real D4 `dt_half` row shortened to 2 periods at `N=32`, much smaller than the prior TG-B1S-D 100P reproduction that took 6250.04 seconds on A100.
- Stop rule: if no new `J1` log output appears for 15 minutes, or total `J1` runtime exceeds 30 minutes without `D4_RUN_COMPLETE.json` or archive progress, stop and return logs for review.

## Local Validation Performed

- Manifest JSON parsed with `python -m json.tool`.
- Packager and packaged Python entrypoints compiled with `python -m py_compile`.
- Packager dry-run passed.
- Generated notebook was written successfully.
- Generated notebook JSON parsed.
- All generated notebook code cells compiled successfully.
- Embedded control panel includes `T1`, `T2`, `T3`, and `J1`.

## Packager Dry-Run Summary

- Packager version: `4.5`
- Bundle SHA-256: `391a88f74cad5aa9bb0fdece87d3f531187335423f38f9276a01f71148373eed`
- Control panel actions: `T1`, `T2`, `T3`, `J1`
- `J1` label: `Run D4 dt_half Short Pilot (<30m)`
- A100 required by manifest: yes
- JAX x64 requested by manifest: yes
- Runtime env override: `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`
- External inputs: none; the frozen reference run files are packaged explicitly.

## Notebook Compile And Control Check

```json
{
  "notebook": "F:\\quantule_mapper\\colab_jobs\\cx_tg_b1s_d4_a100_short_pilot.ipynb",
  "code_cells": 9,
  "syntax_errors": [],
  "action_ids": ["T1", "T2", "T3", "J1"],
  "missing_required_actions": [],
  "j1_label": "Run D4 dt_half Short Pilot (<30m)"
}
```

## Live Checks Not Yet Performed

These require Jake's manual Colab run and returned archive:

- A100 availability in the selected Colab runtime.
- Google Drive mount/write behavior.
- JAX GPU backend and x64 behavior in Colab.
- D4 short-pilot runtime and output generation.
- Archive copy/hash workflow against real Drive.
- Returned `completion_ledger.json`, `result_integrity_manifest.json`, `D4_RUN_COMPLETE.json`, `D4_SUMMARY.json`, and output CSV review.
