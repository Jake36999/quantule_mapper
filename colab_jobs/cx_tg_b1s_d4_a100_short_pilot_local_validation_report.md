# CX TG-B1S D4 A100 Short Pilot Local Validation Report

## Capsule

- Queue row: `CX_TG_B1S_D4_A100_SHORT_PILOT`
- Run class: `short-scientific-pilot`
- Job id: `cx_tg_b1s_d4_a100_short_pilot`
- Row id: `d4_dt_half_2p_n32`
- Manifest: `F:\quantule_mapper\colab_jobs\cx_tg_b1s_d4_a100_short_pilot_manifest.json`
- Generated notebook: `F:\quantule_mapper\colab_jobs\cx_tg_b1s_d4_a100_short_pilot.ipynb`
- Operator instructions: `F:\quantule_mapper\colab_jobs\cx_tg_b1s_d4_a100_short_pilot_operator_instructions.md`
- Provenance report: `F:\quantule_mapper\colab_jobs\cx_tg_b1s_d4_a100_short_pilot_provenance_report.md`
- Drive result path: `/content/drive/MyDrive/QuantuleMapperRuns/cx_tg_b1s_d4_a100_short_pilot`

## Intended J1 Command

```text
python -u jax_scout/gravity_TG_B1S_D4_rows_gpu.py --out /content/qm_job/results/cx_tg_b1s_d4_a100_short_pilot --reference-run /content/qm_job/sweep_runs/TG_B1S_DRIFT_DECOMPOSITION_GPU_20260714_184736 --cases dt_half --d4-periods 2 --N 32
```

The packager invokes this through `run_registered_job` with row id `d4_dt_half_2p_n32`.

## Runtime Contract

- Estimated A100 runtime: `5-30 minutes`.
- Basis: one real D4 `dt_half` row shortened to 2 periods at `N=32`, much smaller than the prior TG-B1S-D 100P reproduction that took 6250.04 seconds on A100.
- Stop rule: if no new `J1` log output appears for 15 minutes, or total `J1` runtime exceeds 30 minutes without `D4_RUN_COMPLETE.json` or archive progress, stop and return logs for review.

## Local Validation Performed

- Manifest JSON parsed with `python -m json.tool`.
- Packager compiled with `python -m py_compile`.
- Packaged Python source files compiled with `python -m py_compile`.
- Packaged reference run directory exists and selected files total about 791 KB.
- Packager dry-run passed.
- Generated notebook was written successfully.
- Generated notebook JSON parsed.
- All generated notebook code cells compiled successfully: `9` code cells.
- Embedded control panel includes `T1`, `T2`, `T3`, and `J1`.
- `terminal_J1` is present.
- Control-panel Markdown table is present.
- Runtime env override preserved: `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`.
- No old `0.45` memory cap was found in the generated notebook.
- No `sys.stdout.buffer` usage was found in the generated notebook.

## Packager Dry-Run Summary

- Packager version: `4.6`
- Bundle SHA-256: `391a88f74cad5aa9bb0fdece87d3f531187335423f38f9276a01f71148373eed`
- Embedded files: `24`
- Control panel actions: `T1`, `T2`, `T3`, `J1`
- `J1` label: `Run D4 dt_half Short Pilot (<30m)`
- A100 required by manifest: yes
- JAX x64 requested by manifest: yes
- Resource profile: `a100_aggressive`
- XLA memory fraction: `0.90`
- External inputs: none; the frozen reference run files are packaged explicitly.

## Artifact Hashes

| Artifact | SHA-256 |
|---|---|
| `cx_tg_b1s_d4_a100_short_pilot_manifest.json` | `9be4b8eb44cd519eab97637d3989bd67423654ba67795bb5b5709caf054b28f9` |
| `cx_tg_b1s_d4_a100_short_pilot.ipynb` | `6f5e1967b93416e80654ad2da569fbd7915878edd0a33638be9696659e285a06` |

## Local Limitation

Directly executing `gravity_TG_B1S_D4_rows_gpu.py --help` in the local Windows Python failed because this local interpreter does not have `jax` installed:

```text
ModuleNotFoundError: No module named 'jax'
```

This does not affect the Colab handoff. Syntax compilation succeeded locally, and the live GPU/JAX/x64 behavior is tested by `T1` and `T2` inside Colab.

## Live Checks Not Yet Performed

These require Jake's manual Colab run and returned archive:

- A100 availability in the selected Colab runtime.
- Google Drive mount/write behavior.
- JAX GPU backend and x64 behavior in Colab.
- D4 short-pilot runtime and output generation.
- Archive copy/hash workflow against real Drive.
- Returned `completion_ledger.json`, `result_integrity_manifest.json`, `D4_RUN_COMPLETE.json`, `D4_SUMMARY.json`, and output CSV review.
