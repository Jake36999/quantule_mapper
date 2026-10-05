# TG-B1S-D Colab Success Example

This document records the first successful live Colab deployment and validation of the Quantule Mapper Colab capsule lane.

It is an infrastructure and numerical-reproduction result only. It does not create or promote a new physics conclusion.

## Summary

```text
job_id: tg_b1s_d_100p_reproduction
row_id: tg_b1s_d_100p_primary
target_run: D3_100P_lam1
baseline_id: tg_b1s_d_20260714_184736
baseline_source_run: F:\quantule_mapper\sweep_runs\TG_B1S_DRIFT_DECOMPOSITION_GPU_20260714_184736
returned_status: REPRODUCTION_WITHIN_DECLARED_TOLERANCES
colab_runtime: NVIDIA A100-SXM4-40GB
jax_backend: gpu
jax_x64_enabled: true
row_runtime_seconds: 6250.04
archive_size_bytes: 446637
archive_sha256: c5297b822983ebc95d1e2b5740537cb0db0a704cb4b5c0804dbb29dbe872892a
```

The user uploaded the generated notebook to Google Colab, ran the capsule, downloaded the returned archive, and placed it into:

```text
F:\quantule_mapper\colab_jobs\results
```

Local review verified that the archive was complete and internally consistent.

## Files

Manifest:

```text
F:\quantule_mapper\colab_jobs\tg_b1s_d_reproduction_manifest.json
```

Generated notebook:

```text
F:\quantule_mapper\colab_jobs\tg_b1s_d_100p_reproduction.ipynb
```

Returned archive:

```text
F:\quantule_mapper\colab_jobs\results\tg_b1s_d_100p_reproduction_20260715_132529.tar.gz
```

Extracted review folder:

```text
F:\quantule_mapper\colab_jobs\results\tg_b1s_d_100p_reproduction_20260715_132529_extracted\tg_b1s_d_100p_reproduction
```

Key returned artifacts:

```text
results\tg_b1s_d_100p_reproduction\completion_status.json
results\tg_b1s_d_100p_reproduction\comparison_summary.json
results\tg_b1s_d_100p_reproduction\observable_comparison.csv
results\tg_b1s_d_100p_reproduction\simulation\environment_versions.json
completion_ledger.json
result_integrity_manifest.json
```

## Colab Terminal Evidence

Cell 5 completed the registered row:

```text
Running stage main: /usr/bin/python3 -u colab_jobs/tg_b1s_d_reproduce_and_compare.py ...
backend: gpu
devices: [CudaDevice(id=0)]
{
  "status": "TG_STATE_LOAD_LONG_TIME_DRIFT_UNRESOLVED",
  "labels": [
    "TG_STATE_LOAD_LONG_TIME_DRIFT_UNRESOLVED"
  ],
  "outdir": "/content/qm_job/results/tg_b1s_d_100p_reproduction/simulation"
}
Wrote row checkpoint: /content/drive/MyDrive/QuantuleMapperRuns/tg_b1s_d_100p_reproduction/checkpoints/tg_b1s_d_100p_reproduction_tg_b1s_d_100p_primary_checkpoint.json
{
  "status": "RUN_COMPLETE_CAPSULE",
  "completed_rows": [
    "tg_b1s_d_100p_primary"
  ],
  "finished_utc": "2026-07-15T13:16:45Z"
}
```

Cell 6 archived and copied results to Drive:

```text
archive_path: /content/tg_b1s_d_100p_reproduction_20260715_132529.tar.gz
archive_size_bytes: 446637
over_ideal: false
over_hard: false
Copied archive to /content/drive/MyDrive/QuantuleMapperRuns/tg_b1s_d_100p_reproduction/tg_b1s_d_100p_reproduction_20260715_132529.tar.gz
Copied archive integrity sidecar to /content/drive/MyDrive/QuantuleMapperRuns/tg_b1s_d_100p_reproduction/tg_b1s_d_100p_reproduction_20260715_132529.tar.gz.sha256.json
```

## Local Review Results

`completion_status.json`:

```json
{
  "status": "REPRODUCTION_WITHIN_DECLARED_TOLERANCES",
  "completed_utc": "2026-07-15T13:16:45Z"
}
```

`comparison_summary.json`:

```text
compared_observable_count: 17
failed_observables: []
informational_observables_without_declared_tolerances:
  - final_delta_theta
  - final_orbital_distance
  - final_phase_aligned_distance
scientific_claim_boundary: Infrastructure/numerical reproduction only; no new scientific conclusion is inferred.
```

Integrity:

```text
result_integrity_manifest files checked: 65
hash failures: 0
```

Runtime:

```text
completion_ledger row duration_seconds: 6250.036316156387
approx_runtime: 1h 44m 10s
```

## Observable Comparison Highlights

All 17 rows in `observable_comparison.csv` reported `PASS`.

```text
delta_omega_infty
  baseline:   -2.150936882840006e-06
  reproduced: -2.150936882856951e-06
  abs_diff:    1.6944894109822278e-17
  rel_diff:    7.877913222314299e-12

full_profile_overlap
  baseline:    0.9999999987895385
  reproduced:  0.9999999987895382
  abs_diff:    3.3306690738754696e-16

full_modal_leakage_max
  baseline:    9.15831330478709e-05
  reproduced:  9.158313304799448e-05
  abs_diff:    1.2357194260903537e-16

ledger_residual_abs
  baseline:    1.0817746070755172e-05
  reproduced:  1.081774605654435e-05
  abs_diff:    1.4210820833884114e-14

early_late_relative_difference
  baseline:    0.0584210752588697
  reproduced:  0.0584210752527291
  abs_diff:    6.140601915838317e-12

final_delta_theta
  baseline:   -0.0013329967322306402
  reproduced: -0.0013329967322306402
  abs_diff:    0.0
```

## Environment

The returned Colab preflight recorded:

```text
python: 3.12.13
platform: Linux-6.6.122+-x86_64-with-glibc2.35
gpu: NVIDIA A100-SXM4-40GB
driver: 580.82.07
cuda reported by nvidia-smi: 13.0
jax: 0.7.2
jaxlib: 0.7.2
numpy: 2.0.2
jax_backend: gpu
jax_devices: cuda:0
jax_x64_enabled: true
float64_kernel_dtype: float64
```

The returned simulation environment recorded:

```text
XLA_PYTHON_CLIENT_PREALLOCATE: false
XLA_PYTHON_CLIENT_MEM_FRACTION: 0.45
git_commit: UNKNOWN
```

The `git_commit: UNKNOWN` value is expected for the capsule because the generated notebook does not include `.git`. Source identity is instead established by the embedded source manifest and SHA-256 file hashes.

## Lessons

- The manual Colab lane worked end to end: package, upload, extract, verify, run, checkpoint, archive, Drive copy, local review.
- Runtime was about 1h 44m for a run remembered locally as roughly 3.5h, so the first observed gain is useful but not "few minutes" fast.
- The first live notebook used packager `4.4` and script memory defaults. Packager `4.5` now supports manifest runtime environment overrides such as `XLA_PYTHON_CLIENT_MEM_FRACTION`.
- The `fatal: not a git repository` messages are provenance noise, not execution failure. Future capsules should rely on source hashes unless repository metadata is intentionally packaged.
- The result archive was very small, well below both the 1 GB ideal and 5 GB hard cap.

## Reuse Pattern

For the next reproduction capsule:

1. choose a recent completed local baseline;
2. freeze baseline values and existing tolerances in a baseline manifest;
3. write one JSON manifest with explicit `source_files`, `config_files`, control panel actions, and `rows[].stages[]`;
4. package with `notebook_packager_v4.py`;
5. locally validate dry-run, JSON parse, notebook code-cell compilation, and path containment;
6. hand the notebook to the user for Colab;
7. verify the returned archive exactly as done here.
