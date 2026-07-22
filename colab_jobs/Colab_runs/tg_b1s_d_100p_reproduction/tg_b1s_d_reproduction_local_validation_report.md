# TG-B1S-D Colab Capsule Validation Report

## Live Colab Outcome

This capsule was successfully run in Google Colab on July 15, 2026.

- Returned archive: `F:\\quantule_mapper\\colab_jobs\\results\\tg_b1s_d_100p_reproduction_20260715_132529.tar.gz`
- Extracted review folder: `F:\\quantule_mapper\\colab_jobs\\results\\tg_b1s_d_100p_reproduction_20260715_132529_extracted\\tg_b1s_d_100p_reproduction`
- Final comparison status: `REPRODUCTION_WITHIN_DECLARED_TOLERANCES`
- Compared observables: `17`
- Failed observables: `0`
- Integrity manifest check: `65 / 65` files verified
- Row runtime: `6250.036316156387` seconds, about `1h 44m 10s`
- Archive size: `446637` bytes
- Archive SHA-256: `c5297b822983ebc95d1e2b5740537cb0db0a704cb4b5c0804dbb29dbe872892a`
- Colab GPU: `NVIDIA A100-SXM4-40GB`
- Colab JAX backend: `gpu`
- Colab JAX x64: enabled, with float64 kernel verified

This is a numerical reproduction/infrastructure success only. It does not infer a new scientific conclusion.

## Baseline

- Selected baseline: `F:\\quantule_mapper\\sweep_runs\\TG_B1S_DRIFT_DECOMPOSITION_GPU_20260714_184736`
- Target run: `D3_100P_lam1`
- Baseline status: `TG_STATE_LOAD_LONG_TIME_DRIFT_UNRESOLVED`
- Baseline git commit: `9fc7785df19d24f4be29e5d90bad59ed993da87b`
- Baseline manifest: `F:\\quantule_mapper\\colab_jobs\\baselines\\tg_b1s_d_20260714_184736_baseline_manifest.json`

## Packaged Files

See `F:\\quantule_mapper\\colab_jobs\\tg_b1s_d_reproduction_allowlist.json`.

The source bundle hash from packager dry-run was:

`be64e031db8f907b93bfa5e826d9c5ff1f6085fedc5ef9a40b51494d2833015c`

## Commands

Simulation command inside the capsule:

```text
python jax_scout/gravity_TG_B1S_drift_decomposition_gpu.py --out /content/qm_job/results/tg_b1s_d_100p_reproduction/simulation --matrix primary --sample-dt 1.0
```

Analysis command inside the capsule:

```text
python colab_jobs/tg_b1s_d_reproduction_compare.py --baseline-manifest colab_jobs/baselines/tg_b1s_d_20260714_184736_baseline_manifest.json --reproduction-run /content/qm_job/results/tg_b1s_d_100p_reproduction/simulation --out /content/qm_job/results/tg_b1s_d_100p_reproduction
```

## Declared Tolerances

The comparison uses only existing TG-B1S-D/D4 gate thresholds from `jax_scout/gravity_TG_B1S_drift_decomposition_gpu.py`:

- `delta_omega_relative_difference_max = 0.50`
- `profile_overlap_min = 0.999`
- `modal_leakage_max = 1e-3`
- `ledger_residual_abs_max = 1e-4`
- `constant_slope_early_late_relative_difference_max = 0.35`
- structural channels must not classify as `LINEAR_SECULAR` or `QUADRATIC_OR_ACCELERATING`
- phase channel must classify as `LINEAR_SECULAR`

No new cross-device tolerance was introduced.

## Local Tests Performed

- Packager compiled.
- Helper scripts compiled.
- All JSON parsed.
- Packager dry-run validation passed.
- Generated notebook compiled all code cells.
- Generated constants, runtime setup, extraction and dashboard cells executed locally.
- Embedded bytes were extracted and hash-verified locally.
- Control-panel bridge contained isolated `T1`, `T2`, `T3` and `J1` outputs.
- Path traversal manifest was rejected.
- Missing/stale project root was rejected.
- Comparison analysis passed when run against the frozen baseline directory as a fixture.
- Missing output case failed closed.
- Intentionally changed `delta_omega_infty` produced `REPRODUCTION_OUTSIDE_DECLARED_TOLERANCES`.
- Local comparison archive was reopened and all member hashes were verified.

## Notebook

Generated notebook:

`F:\\quantule_mapper\\colab_jobs\\tg_b1s_d_100p_reproduction.ipynb`

Expected Drive result path:

`/content/drive/MyDrive/QuantuleMapperRuns/tg_b1s_d_100p_reproduction`

## Live Runtime Checks

- Google Drive mount and Drive write: passed in the live Colab run.
- A100 availability: passed.
- JAX GPU backend on Colab: passed.
- JAX x64 behavior on Colab: passed.
- Full 100-period TG-B1S-D simulation runtime and output generation: passed.
- Archive copy/hash workflow against real Drive: passed from notebook terminal evidence and local archive integrity review.

See `F:\\quantule_mapper\\colab_jobs\\TG_B1S_D_COLAB_SUCCESS_EXAMPLE.md` for the full live evidence summary.
