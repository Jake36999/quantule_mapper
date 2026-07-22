# CX Colab Fast-Lane Handoff Test Local Validation Report

## Capsule

- Queue row: `CX_COLAB_FAST_LANE_QUEUE_HANDOFF_TEST`
- Job id: `cx_colab_fast_lane_handoff_test`
- Row id: `tg_b1s_d_100p_reproduction`
- Manifest: `F:\quantule_mapper\colab_jobs\cx_colab_fast_lane_handoff_test_manifest.json`
- Generated notebook: `F:\quantule_mapper\colab_jobs\cx_colab_fast_lane_handoff_test.ipynb`
- Operator instructions: `F:\quantule_mapper\colab_jobs\cx_colab_fast_lane_handoff_test_operator_instructions.md`
- Drive results path: `/content/drive/MyDrive/QuantuleMapperRuns/cx_colab_fast_lane_handoff_test`

## Purpose

This capsule runs a real TG-B1S-D 100-period reproduction on Colab A100 and compares it against the frozen `tg_b1s_d_20260714_184736` baseline. It is an infrastructure and numerical-reproduction handoff test only; no new scientific claim is inferred from dashboard state.

## Scope Contract

- Execution scope: `full-fidelity simulation`
- Fidelity requirement: `exact reproduction; scientific source unchanged`
- Run class: `full-reproduction`
- Estimated A100 runtime: `90-120 minutes`, based on the prior TG-B1S-D 100P Colab reproduction runtime of `6250.04 seconds` (~1h44m).
- Stop rule: if `J1` shows no new log output for 30 minutes, or total runtime exceeds 2.5 hours without checkpoint/archive progress, stop and return logs for review.
- Boundary: this is a queue-handoff workflow test, but the workload is intentionally real. Do not substitute a toy smoke test or shortened procedural job for performance reasons.

## Frozen Baseline And Contract

- Baseline manifest: `F:\quantule_mapper\colab_jobs\baselines\tg_b1s_d_20260714_184736_baseline_manifest.json`
- Target run: `D3_100P_lam1`
- Simulation entrypoint: `jax_scout/gravity_TG_B1S_drift_decomposition_gpu.py`
- Comparison entrypoint: `colab_jobs/tg_b1s_d_reproduction_compare.py`
- Declared tolerances: existing TG-B1S-D/D4 gates recorded in the baseline manifest and comparison script.
- Expected final status on successful reproduction: `REPRODUCTION_WITHIN_DECLARED_TOLERANCES`

## Local Validation Performed

- Manifest JSON parsed with `python -m json.tool`.
- Packager and packaged Python entrypoints compiled with `python -m py_compile`.
- Packager dry-run passed.
- Generated notebook was written successfully.
- Generated notebook JSON parsed.
- All generated notebook code cells compiled successfully.

## Packager Dry-Run Summary

- Packager version: `4.5`
- Bundle SHA-256: `be64e031db8f907b93bfa5e826d9c5ff1f6085fedc5ef9a40b51494d2833015c`
- Control panel actions: `T1`, `T2`, `T3`, `J1`
- `J1` label: `Run TG-B1S-D 100P Reproduction (~1.5-2h)`
- A100 required by manifest: yes
- JAX x64 requested by manifest: yes
- Runtime env override: `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`
- Memory policy note: `0.90` is intentional for the Colab fast lane so the A100 session is not artificially restricted by the older conservative cap. This changes environment resource policy only; it does not alter TG-B1S-D equations or comparison gates.

## Notebook Compile Check

```json
{
  "notebook": "F:\\quantule_mapper\\colab_jobs\\cx_colab_fast_lane_handoff_test.ipynb",
  "code_cells": 9,
  "syntax_errors": []
}
```

## Live Checks Not Yet Performed

These require the manual Colab run and returned archive:

- A100 availability in the selected Colab runtime.
- Google Drive mount/write behavior.
- Full TG-B1S-D 100-period runtime.
- Returned archive integrity.
- `completion_ledger.json`, `completion_status.json`, `comparison_summary.json`, and `observable_comparison.csv` review.
