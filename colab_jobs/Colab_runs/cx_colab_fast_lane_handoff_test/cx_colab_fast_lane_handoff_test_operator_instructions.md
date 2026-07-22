# CX Colab Fast-Lane Handoff Test Operator Instructions

This notebook runs a real TG-B1S-D 100-period reproduction on Colab A100 and compares it against the frozen `tg_b1s_d_20260714_184736` baseline. It is an infrastructure and numerical-reproduction handoff test, not a new physics promotion.

Scope contract:

- Execution scope: `full-fidelity simulation`
- Fidelity requirement: `exact reproduction; scientific source unchanged`
- Run class: `full-reproduction`
- Expected A100 runtime: about `1.5-2 hours`; prior matching run took `6250.04 seconds` (~1h44m).
- Stop rule: if `J1` shows no new log output for 30 minutes, or total runtime exceeds 2.5 hours without checkpoint/archive progress, stop and return the logs for review.
- Do not replace `J1` with a toy smoke test or shortened procedural workload. The A100 lane is the performance optimization.
- If the goal is only to test Colab UI/UX or handoff plumbing, use the `ui_ux_smoke_test` capsule instead of this notebook.

1. Upload `F:\quantule_mapper\colab_jobs\cx_colab_fast_lane_handoff_test.ipynb` to Google Colab.
2. Select an A100 GPU runtime.
3. Run the notebook from the top until the control panel appears.
4. Click `T1`, then `T2`, then `T3`.
5. Only after all three tests pass, click `J1: Run TG-B1S-D 100P Reproduction (~1.5-2h)`.
6. Leave the tab active while the 100-period run executes. The prior matching Colab run took about 1h 44m on A100.
7. When `J1` reaches `DRIVE_COMMITTED`, download the archive from:
   `/content/drive/MyDrive/QuantuleMapperRuns/cx_colab_fast_lane_handoff_test`
8. Place the downloaded archive in:
   `F:\quantule_mapper\colab_jobs\results`
9. Ask Codex to review the returned `CX_COLAB_FAST_LANE_QUEUE_HANDOFF_TEST` capsule.

Notes:

- `T1` writes to `terminal_T1`.
- `T2` writes to `terminal_T2`.
- `T3` writes to `terminal_T3`.
- `J1` writes to `terminal_J1`.
- The visible terminal is a bounded readout. Full logs are archived separately.
- Dashboard green states are not the final evidence. The review should use the returned archive, `completion_ledger.json`, `result_integrity_manifest.json`, `completion_status.json`, `comparison_summary.json`, and `observable_comparison.csv`.
- Expected final comparison status, if the reproduction is within the declared existing gates, is `REPRODUCTION_WITHIN_DECLARED_TOLERANCES`.
