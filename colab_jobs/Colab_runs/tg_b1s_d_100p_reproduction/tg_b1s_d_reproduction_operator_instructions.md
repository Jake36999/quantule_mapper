# TG-B1S-D Colab Reproduction Operator Instructions

1. Upload `F:\\quantule_mapper\\colab_jobs\\tg_b1s_d_100p_reproduction.ipynb` to Google Colab.
2. Select an A100 GPU runtime if available.
3. Run the notebook from the top until the control panel appears.
4. Click `T1`, then `T2`, then `T3`.
5. Only after all three tests pass, click `J1: Run TG-B1S-D Reproduction`.
6. Leave the tab active while the 100-period run executes.
7. When `J1` reaches `DRIVE_COMMITTED`, download the archive from:
   `/content/drive/MyDrive/QuantuleMapperRuns/tg_b1s_d_100p_reproduction`
8. Place the downloaded archive in:
   `F:\\quantule_mapper\\colab_jobs\\results`
9. Ask Codex to review the returned TG-B1S-D reproduction capsule.

Notes:

- The notebook shows a Markdown control-panel table first. Use it to confirm that each button has its own output terminal:
  - `T1` writes to `terminal_T1`.
  - `T2` writes to `terminal_T2`.
  - `T3` writes to `terminal_T3`.
  - `J1` writes to `terminal_J1`.
- The visible terminal is a bounded readout. The full logs are archived separately.
- Dashboard green states are infrastructure/numerical reproduction statuses only.
- They do not prove or promote a scientific interpretation.
- Full logs are written under `/content/qm_colab/jobs/tg_b1s_d_100p_reproduction/run/logs`.
- The bounded UI terminals are only rolling views.
