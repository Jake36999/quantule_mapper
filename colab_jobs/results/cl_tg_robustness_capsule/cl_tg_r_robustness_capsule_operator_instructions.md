# CL TG-R Robustness Capsule Operator Instructions

This notebook runs the queued `CL_TG_R_ROBUSTNESS_CAPSULE` Colab fast-lane job. It is a compact Phase R robustness campaign for the reviewed TG-B2 A-well attraction result, not a UI smoke test.

Scope contract:

- Run class: `full-campaign`
- Execution scope: `full-fidelity simulation`
- Fidelity requirement: protected TG-B2/TG-B1S modules imported unchanged
- Expected A100 runtime: about `3-5 hours`
- Operator stop rule: if `J1` shows no new log output for 30 minutes, or total runtime exceeds 6 hours on A100 without `ROBUSTNESS_RUN_COMPLETE.json`/archive progress, stop and return logs for review.
- Scientific boundary: mirror-only, near-field/screened TG-B2 robustness; not gravity, UFF, or IRER validation.

The packaged matrix is:

| row | purpose |
|---|---|
| `r1_grid_N96_sep3` | grid-refinement sentinel, `N=96`, `sep=3.0` |
| `r1_dt_half_sep3` | timestep-halved sentinel, `dt=0.001`, `sep=3.0` |
| `r1_box_L20_sep3` | larger-box sentinel, `L=20`, `sep=3.0` |
| `r2_eps_003_sep3` | epsilon half-scale point, `epsilon_G=0.03`, `sep=3.0` |
| `r2_eps_012_sep3` | epsilon double-scale point, `epsilon_G=0.12`, `sep=3.0` |

The reviewed A100 definitive run supplies the `epsilon_G=0.06`, `N=64`, `L=16`, `dt=0.002`, `sep=3.0` baseline.

Manual Colab steps:

1. Upload `F:\quantule_mapper\colab_jobs\cl_tg_r_robustness_capsule.ipynb` to Google Colab.
2. Select an A100 GPU runtime.
3. Run the notebook from the top until the control panel appears.
4. Click `T1: Run Preflight`.
5. Click `T2: Run Smoke Test`.
6. Click `T3: Test Persistence`.
7. Only after all tests pass, click `J1: Run TG-R Robustness (~3-5h)`.
8. Keep the Colab tab active until the archive copy completes.
9. Download the archive from `/content/drive/MyDrive/QuantuleMapperRuns/cl_tg_r_robustness_capsule`.
10. Place the archive in `F:\quantule_mapper\colab_jobs\results` or `F:\quantule_mapper\colab_jobs\Colab_runs`.
11. Ask Codex/Claude to review the returned `CL_TG_R_ROBUSTNESS_CAPSULE`.

Review focus after download:

- `robustness_summary.json`
- `robustness_matrix.csv`
- per-row `summary.json` and scalar CSVs
- protected hash verification
- convergence drift against the sep=3 baseline
- epsilon through-origin linear fit and `R^2`
- sign, off-null, well/hill antisymmetry, charge, and node-distinctness gates

The driver verdict may be:

```text
TG_R_ROBUSTNESS_PASS
TG_R_ROBUSTNESS_PARTIAL
TG_R_ROBUSTNESS_FAIL
TG_R_ROBUSTNESS_INCOMPLETE
```

Treat that verdict as a machine label pending primary science review.
