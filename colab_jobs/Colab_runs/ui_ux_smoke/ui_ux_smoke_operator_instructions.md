# Colab UI/UX Smoke Test Operator Instructions

Use this notebook to test the Colab control-panel experience without running a scientific simulation.

1. Upload `F:\\quantule_mapper\\colab_jobs\\ui_ux_smoke_test.ipynb` to Google Colab.
2. Any normal Colab runtime should work. GPU is optional for this UI test.
3. Run notebook cells from the top until the Markdown control-panel table and button panel appear.
4. Confirm the table is readable and maps:
   - `T1` to `terminal_T1`
   - `T2` to `terminal_T2`
   - `T3` to `terminal_T3`
   - `J1` to `terminal_J1`
5. Click `T1`, then `T2`, then `T3`.
6. Click `J1: Run UI Smoke`.
7. Watch `terminal_J1`; it should show two stages, `prepare` and `run`, with short progress lines.
8. After archive completion, download the result archive from:
   `/content/drive/MyDrive/QuantuleMapperRuns/ui_ux_smoke_test`
9. Place the downloaded archive in:
   `F:\\quantule_mapper\\colab_jobs\\results`
10. Ask Codex to review the returned UI/UX smoke capsule.

Expected behavior:

- Total `J1` runtime should be only a few seconds.
- The full log should be preserved even though the visible terminal is bounded.
- The final job output should include `completion_status.json` with `UI_UX_SMOKE_PASS`.
- This capsule does not test scientific reproducibility.
