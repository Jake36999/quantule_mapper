# CL TG-R Robustness Capsule v2 Operator Instructions

This notebook runs the queued `CL_TG_R_ROBUSTNESS_CAPSULE` v2 completion job. It is a Phase R robustness completion capsule for the reviewed TG-B2 A-well attraction result, not a UI smoke test.

Scope contract:

- Run class: `full-campaign`
- Execution scope: `full-fidelity simulation`
- Fidelity requirement: protected TG-B2/TG-B1S modules imported unchanged
- Expected A100 runtime: about `2-4 hours`
- Operator stop rule: if `J1` shows no new log output for 30 minutes, or total runtime exceeds 5 hours on A100 without `ROBUSTNESS_RUN_COMPLETE.json`/archive progress, stop and return logs for review.
- Scientific boundary: mirror-only, near-field/screened TG-B2 robustness; not gravity, UFF, or IRER validation.

The v2 capsule does not rerun the two successful v1 rows. It bundles the returned v1 evidence as a frozen reference and runs only the missing/repaired rows.

## V1 Evidence Preserved

| row | v1 result |
|---|---|
| `r1_grid_N96_sep3` | PASS; `<F_R_well>=-5.452931236310377e-05`; drift vs baseline ~`0.0515%` |
| `r1_dt_half_sep3` | PASS; `<F_R_well>=-5.450123416655863e-05`; effectively identical to baseline |

## V2 Rows To Run

| row | purpose |
|---|---|
| `r2_eps_003_sep3` | epsilon half-scale point, `epsilon_G=0.03`, `sep=3.0` |
| `r2_eps_012_sep3` | epsilon double-scale point, `epsilon_G=0.12`, `sep=3.0` |
| `r1_box_L20_N96_sep3` | repaired larger-box sentinel, `L=20,N=96`, `sep=3.0` |

Manual Colab steps:

1. Upload `F:\quantule_mapper\colab_jobs\cl_tg_r_robustness_capsule_v2.ipynb` to Google Colab.
2. Select an A100 GPU runtime.
3. Run the notebook from the top until the control panel appears.
4. Click `T1: Run Preflight`.
5. Click `T2: Run Smoke Test`.
6. Click `T3: Test Persistence`.
7. Only after all tests pass, click `J1: Run TG-R v2 Completion (~2-4h)`.
8. Keep the Colab tab active until the archive copy completes.
9. Download the archive from `/content/drive/MyDrive/QuantuleMapperRuns/cl_tg_r_robustness_capsule_v2`.
10. Place the archive in `F:\quantule_mapper\colab_jobs\results` or `F:\quantule_mapper\colab_jobs\Colab_runs`.
11. Ask Codex/Claude to review the returned `CL_TG_R_ROBUSTNESS_CAPSULE_V2`.

Review focus after download:

- `robustness_summary.json`
- `robustness_matrix.csv`
- `robustness_matrix_v2_only.csv`
- `ROBUSTNESS_RUN_COMPLETE.json`
- protected hash verification
- epsilon through-origin linear fit and `R^2`
- repaired larger-box row status and drift
- sign, off-null, well/hill antisymmetry, charge, and node-distinctness gates

The driver intentionally returns a successful infrastructure code once it has completed the v2 matrix and written the summary. The scientific status is in the machine verdict:

```text
TG_R_ROBUSTNESS_PASS
TG_R_ROBUSTNESS_PARTIAL
TG_R_ROBUSTNESS_FAIL
```

Treat that verdict as a machine label pending primary science review.
