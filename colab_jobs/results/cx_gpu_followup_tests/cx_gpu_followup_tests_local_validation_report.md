# CX GPU Follow-up Tests Local Validation Report

## Capsule

- Job id: `cx_gpu_followup_tests`
- Manifest: `F:\quantule_mapper\colab_jobs\cx_gpu_followup_tests_manifest.json`
- Generated notebook: `F:\quantule_mapper\colab_jobs\cx_gpu_followup_tests.ipynb`
- Operator instructions: `F:\quantule_mapper\colab_jobs\cx_gpu_followup_tests_operator_instructions.md`
- Provenance report: `F:\quantule_mapper\colab_jobs\cx_gpu_followup_tests_provenance_report.md`
- Drive results path: `/content/drive/MyDrive/QuantuleMapperRuns/cx_gpu_followup_tests`

## Purpose

Package four GPU-facing queued follow-up tests into one Colab control-panel notebook. The CPU-only D4 discrepancy review script is deliberately excluded.

## Rows

| button | row id | script |
|---|---|---|
| `J1` | `cx_tg_b2_cooled_pair_secular_force` | `jax_scout/gravity_TG_B2_cooled_pair_secular_force_gpu.py` |
| `J2` | `cx_tg_clock_migration_characterization` | `jax_scout/gravity_TG_clock_migration_characterization_gpu.py` |
| `J3` | `cx_tg_rate_source_semantics_bridge` | `jax_scout/gravity_TG_rate_source_semantics_bridge_gpu.py` |
| `J4` | `cx_gravity_d_load_capacity_yield_map` | `jax_scout/gravity_D_load_capacity_yield_map_gpu.py` |

## Local Validation Performed

- Manifest JSON parsed with `python -m json.tool`.
- Packager compiled with `python -m py_compile`.
- All bundled Python source files compiled with `python -m py_compile`.
- Packager dry-run passed.
- Generated notebook was written successfully.
- Generated notebook JSON parsed.
- All generated notebook code cells compiled successfully: `9` code cells.
- Control-panel Markdown table was present.
- Control-panel bridge generation code was present.
- `T1`, `T2`, `T3`, `J1`, `J2`, `J3`, and `J4` were present.
- `terminal_J1`, `terminal_J2`, `terminal_J3`, and `terminal_J4` were present.
- Runtime env override preserved: `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`.
- No old `0.45` memory cap was found in the generated notebook.
- No `sys.stdout.buffer` usage was found in the generated notebook.

## Packager Dry-Run Summary

- Packager version: `4.6`
- Bundle SHA-256: `3b8d5ed1acd7cf68f5807f38e3fb7ac46b5bd431bfbf4c849adaa5cfc08cfe54`
- Embedded files: `13`
- Control panel actions: `T1`, `T2`, `T3`, `J1`, `J2`, `J3`, `J4`
- A100 requested by manifest: yes
- JAX x64 requested by manifest: yes
- Resource profile: `a100_aggressive`
- XLA memory fraction: `0.90`

## Artifact Hashes

| Artifact | SHA-256 |
|---|---|
| `cx_gpu_followup_tests_manifest.json` | `3af61e8ab1e0761b81f091064d46a0ac63d4ac6f4ef19a34fba2551cdb6f9524` |
| `cx_gpu_followup_tests.ipynb` | `d7e7de301a4b9562d1a809877805c9e746d99a1907e45bd5dd241f744b7936e1` |

## Live Checks Not Yet Performed

These require manual Colab execution and returned archives:

- A100 availability in the selected Colab runtime.
- Google Drive mount/write behavior.
- Actual runtime for each selected job button.
- Returned archive integrity.
- Review of each row's `RUN_COMPLETE.json`, summary JSON/CSV outputs, logs, and machine verdict.
