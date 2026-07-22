# CL TG-R Robustness Capsule v2 Local Validation Report

## Capsule

- Queue row: `CL_TG_R_ROBUSTNESS_CAPSULE`
- Job id: `cl_tg_r_robustness_capsule_v2`
- Row id: `tg_r_robustness_v2_completion`
- Manifest: `F:\quantule_mapper\colab_jobs\cl_tg_r_robustness_capsule_v2_manifest.json`
- Generated notebook: `F:\quantule_mapper\colab_jobs\cl_tg_r_robustness_capsule_v2.ipynb`
- Operator instructions: `F:\quantule_mapper\colab_jobs\cl_tg_r_robustness_capsule_v2_operator_instructions.md`
- Driver: `F:\quantule_mapper\colab_jobs\tg_r_robustness_driver_v2.py`
- Baseline summary: `F:\quantule_mapper\colab_jobs\baselines\tg_b2_definitive_force_20260716_baseline_summary.json`
- V1 evidence reference: `F:\quantule_mapper\colab_jobs\baselines\tg_r_robustness_v1_partial_evidence_20260717.json`
- Drive results path: `/content/drive/MyDrive/QuantuleMapperRuns/cl_tg_r_robustness_capsule_v2`

## Purpose

This capsule completes the Phase R robustness matrix after v1 returned partial evidence. It preserves successful v1 `N=96` and `dt/2` rows as frozen returned evidence, runs epsilon rows before the box row, and attempts a repaired larger-box sentinel `L=20,N=96`.

## Scope Contract

- Run class: `full-campaign`
- Execution scope: `full-fidelity simulation`
- Fidelity requirement: protected modules imported unchanged
- Estimated A100 runtime: `120-240 minutes`
- Operator stop rule: if `J1` shows no new log output for 30 minutes, or total runtime exceeds 5 hours on A100 without `ROBUSTNESS_RUN_COMPLETE.json`/archive progress, stop and return logs for review.
- Boundary: mirror-only TG-B2 robustness; not gravity, UFF, or IRER validation.

## V2 Matrix

| row | purpose |
|---|---|
| `r2_eps_003_sep3` | epsilon half-scale point |
| `r2_eps_012_sep3` | epsilon double-scale point |
| `r1_box_L20_N96_sep3` | repaired larger-box sentinel replacing failed v1 `L=20,N=64` |

The reviewed `epsilon_G=0.06`, `N=64`, `L=16`, `dt=0.002`, `sep=3.0` A100 definitive row is used as the baseline reference.

## Local Validation Performed

- Manifest JSON parsed with `python -m json.tool`.
- V1 evidence JSON parsed with `python -m json.tool`.
- Baseline JSON parsed with `python -m json.tool`.
- Protected registry JSON parsed with `python -m json.tool`.
- Packager compiled with `python -m py_compile`.
- V2 driver and packaged source entrypoints/helpers compiled with `python -m py_compile`.
- Driver protected-hash verification passed locally for the four frozen modules it requires.
- V2 driver CLI help executed successfully.
- Packager dry-run passed.
- Generated notebook was written successfully.
- Generated notebook JSON parsed.
- All generated notebook code cells compiled successfully: `9` code cells.
- Control-panel Markdown table was present.
- Control-panel bridge generation code was present.
- `T1`, `T2`, `T3`, and `J1` were present in the generated control registry.
- `terminal_J1` was present as the v2 job output channel.
- `J1` label includes the runtime estimate: `Run TG-R v2 Completion (~2-4h)`.
- Runtime env override preserved: `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`.
- No old `0.45` memory cap was found in the generated notebook.
- No `sys.stdout.buffer` usage was found in the generated notebook.

## Packager Dry-Run Summary

- Packager version: `4.6`
- Bundle SHA-256: `6b25a9d05592d184f9ca2c958e92b306177a852fb39b0e3b905e04dd0ef7cc31`
- Embedded files: `9`
- Control panel actions: `T1`, `T2`, `T3`, `J1`
- A100 required by manifest: yes
- JAX x64 requested by manifest: yes
- Resource profile: `a100_aggressive`
- XLA memory fraction: `0.90`

## Artifact Hashes

| Artifact | SHA-256 |
|---|---|
| `cl_tg_r_robustness_capsule_v2_manifest.json` | `a502b39e6d4839ccc7ef250ac9042ccf9987cf6fc4763077d3054ed3ffd3eaf1` |
| `tg_r_robustness_driver_v2.py` | `957f776ebbcc38e645f64666f8690e7e8993be32fc2939d8b45fcbfa2d7dfd32` |
| `tg_r_robustness_v1_partial_evidence_20260717.json` | `b641fbffe0bd38553cb7a0899fb56e2918003a922f53da4b3c50d7bc0894ff21` |
| `cl_tg_r_robustness_capsule_v2.ipynb` | `8ec750d92a63930c7e92de1761666c0c904319b3809dfaff506584a688e98500` |

## Live Checks Not Yet Performed

These require the manual Colab run and returned archive:

- A100 availability in the selected Colab runtime.
- Google Drive mount/write behavior.
- Full v2 robustness runtime.
- Returned archive integrity.
- Review of `robustness_summary.json`, `robustness_matrix.csv`, `robustness_matrix_v2_only.csv`, per-row `summary.json` files, epsilon linearity, repaired box drift, and all gate statuses.
