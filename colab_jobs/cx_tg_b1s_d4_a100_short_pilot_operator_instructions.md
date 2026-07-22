# CX TG-B1S D4 A100 Short Pilot Operator Instructions

This notebook runs the queued `CX_TG_B1S_D4_A100_SHORT_PILOT` Colab fast-lane job. It is a short scientific pilot, not a UI smoke test and not the full D4 campaign.

## Run Contract

- Run class: `short-scientific-pilot`
- Execution scope: `full-fidelity simulation`
- Fidelity requirement: scientific source unchanged; deliberately shortened pilot arguments only
- Expected A100 runtime: `5-30 minutes`
- Stop rule: if `J1` shows no new log output for 15 minutes, or total `J1` runtime exceeds 30 minutes without `D4_RUN_COMPLETE.json` or archive progress, stop and return the logs for review.

## Scientific Scope

- Entrypoint: `jax_scout/gravity_TG_B1S_D4_rows_gpu.py`
- Case: `dt_half`
- Duration: `--d4-periods 2`
- Grid: `--N 32`
- Reference run: packaged inside the capsule at `/content/qm_job/sweep_runs/TG_B1S_DRIFT_DECOMPOSITION_GPU_20260714_184736`
- Scientific source is unchanged. Only the pilot arguments shorten the run.

The script's case label remains `D4_dt_half_50P` even in this 2-period pilot, because that label is hard-coded by the D4 runner.

## Manual Steps

1. Upload `F:\quantule_mapper\colab_jobs\cx_tg_b1s_d4_a100_short_pilot.ipynb` to Google Colab.
2. Select an A100 GPU runtime.
3. Run the notebook from the top until the control panel appears.
4. Click `T1: Run Preflight`.
5. Click `T2: Run Smoke Test`.
6. Click `T3: Test Persistence`.
7. Only after all three tests pass, click `J1: Run D4 dt_half Short Pilot (<30m)`.
8. When the archive copy completes, download the archive from `/content/drive/MyDrive/QuantuleMapperRuns/cx_tg_b1s_d4_a100_short_pilot`.
9. Place the downloaded archive in `F:\quantule_mapper\colab_jobs\results`.
10. Ask Codex/Claude to review the returned `CX_TG_B1S_D4_A100_SHORT_PILOT` capsule.

The returned archive, not the dashboard color alone, is the review evidence.
