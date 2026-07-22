# CL TG-B2 Definitive Force Operator Instructions

This notebook runs the queued `CL_TG_B2_DEFINITIVE_FORCE` Colab fast-lane job. It is a full-fidelity dynamical body-force campaign, not a UI smoke test.

Scope contract:

- Run class: `full-campaign`
- Execution scope: `full-fidelity simulation`
- Fidelity requirement: `scientific source unchanged; A-well branch; frozen TG-B1S untouched; single-node norm`
- Expected A100 runtime: about `2.5-3.5 hours`
- Operator stop rule: if `J1` shows no new log output for 30 minutes, or total runtime exceeds 5 hours on A100 without `RUN_COMPLETE.json`/archive progress, stop and return logs for review.
- Script-level stop rule: each arm stops if the node separation falls below `1.2` or becomes invalid.

Manual Colab steps:

1. Upload `F:\quantule_mapper\colab_jobs\cl_tg_b2_definitive_force.ipynb` to Google Colab.
2. Select an A100 GPU runtime.
3. Run the notebook from the top until the control panel appears.
4. Click `T1: Run Preflight`.
5. Click `T2: Run Smoke Test`.
6. Click `T3: Test Persistence`.
7. Only after all tests pass, click `J1: Run TG-B2 Definitive Force (~2.5-3.5h)`.
8. Keep the Colab tab active until the archive copy completes.
9. Download the archive from `/content/drive/MyDrive/QuantuleMapperRuns/cl_tg_b2_definitive_force`.
10. Place the archive in `F:\quantule_mapper\colab_jobs\results`.
11. Ask Codex/Claude to review the returned `CL_TG_B2_DEFINITIVE_FORCE` capsule.

Review focus after download:

- Do not trust the auto-label alone.
- Scrutinize `summary.json`, especially `<F_R>` sign, well/hill antisymmetry, off-null behavior, and cross-separation consistency.
- Primary sign convention: `F_R < 0` means attraction.
- Expected key artifacts include `RUN_COMPLETE.json`, `summary.json`, `config.json`, per-separation row markers, and `scalars_sep*_{off,well,hill}.csv`.
- The bounded notebook terminal is only a live tail. Full logs are archived separately.
