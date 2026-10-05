# CL TG-R Robustness Capsule IPYNB Provenance Report

## Capsule Identity

- Queue row: `CL_TG_R_ROBUSTNESS_CAPSULE`
- Job name: `CL TG-R Robustness Capsule - TG-B2 A-well Attraction`
- Job id: `cl_tg_r_robustness_capsule`
- Row id: `tg_r_robustness_matrix`
- Notebook: `F:\quantule_mapper\colab_jobs\cl_tg_r_robustness_capsule.ipynb`
- Manifest: `F:\quantule_mapper\colab_jobs\cl_tg_r_robustness_capsule_manifest.json`
- Operator instructions: `F:\quantule_mapper\colab_jobs\cl_tg_r_robustness_capsule_operator_instructions.md`
- Local validation report: `F:\quantule_mapper\colab_jobs\cl_tg_r_robustness_capsule_local_validation_report.md`
- Generated for manual Google Colab upload and A100 execution.

## Bundle Hashes

| Artifact | SHA-256 |
|---|---|
| `cl_tg_r_robustness_capsule_manifest.json` | `be59f1d413a0ffd0f17ff888545b34c41c7e787f86757521ddd4625cb9e04b24` |
| `tg_r_robustness_driver.py` | `f0423516512ea7e5fc3a25715776e5f1f5340a1adffb82664c7af5f4088d0521` |
| `baselines/tg_b2_definitive_force_20260716_baseline_summary.json` | `bcb6436e0fede58f466d9f6f289678ca3a67f7ba83f0908a2578f753acba8a1d` |
| Packager dry-run source bundle | `8ea2d53d69753b2edbd8b2ccec98bb0741eb9ff15660e7cdd24afe767981bf44` |

## Bundled Files

The notebook embeds exactly these files. It does not embed the whole repository.

| Bundled path | Role |
|---|---|
| `colab_jobs/tg_r_robustness_driver.py` | Phase R matrix runner and aggregator |
| `colab_jobs/baselines/tg_b2_definitive_force_20260716_baseline_summary.json` | Reviewed definitive A100 baseline |
| `docs/gravity_maturity/TG_PROTECTED_REGISTRY.json` | Protected-module hash contract |
| `jax_scout/__init__.py` | Package marker/import support |
| `jax_scout/phase_d_c3_wave.py` | Protected Q-ball/KG machinery |
| `jax_scout/gravity_TG_B1S_state_load_feedback_gpu.py` | Protected frozen TG-B1S helper/scaffold |
| `jax_scout/gravity_TG_B2_two_node_awell.py` | Protected TG-B2 A-well dynamics helper |
| `jax_scout/gravity_TG_B2_definitive_force.py` | Protected definitive body-force entrypoint |

## What Test Is Being Run

The packaged `J1` action runs:

```text
python colab_jobs/tg_r_robustness_driver.py --out /content/qm_job/results/cl_tg_r_robustness --baseline-summary colab_jobs/baselines/tg_b2_definitive_force_20260716_baseline_summary.json --protected-registry docs/gravity_maturity/TG_PROTECTED_REGISTRY.json
```

The driver first verifies protected file hashes, then invokes `jax_scout/gravity_TG_B2_definitive_force.py` unchanged for each matrix row.

Matrix:

| row | command modifiers |
|---|---|
| `r1_grid_N96_sep3` | `--seps 3.0 --N 96 --T 180` |
| `r1_dt_half_sep3` | `--seps 3.0 --dt 0.001 --T 180` |
| `r1_box_L20_sep3` | `--seps 3.0 --L 20 --T 180` |
| `r2_eps_003_sep3` | `--seps 3.0 --epsilon-G 0.03 --T 180` |
| `r2_eps_012_sep3` | `--seps 3.0 --epsilon-G 0.12 --T 180` |

The baseline point is the reviewed `epsilon_G=0.06`, `N=64`, `L=16`, `dt=0.002`, `sep=3.0` Colab A100 definitive result:

```text
<F_R_well> = -5.4501234166498155e-05
```

## Reason

The TG-B2 definitive body-force run confirmed the A-well attraction sign for the configured mirror test. Phase R asks whether that result is robust under the first numerical and parameter perturbations:

- grid refinement;
- timestep halving;
- larger box;
- epsilon scaling through the reviewed baseline point.

The capsule intentionally uses a compact one-separation sentinel matrix so it can fit a realistic Colab A100 session. It is a first Phase R robustness capsule, not the entire future robustness/characterization program.

## Expected Outputs

The capsule fails closed if the declared outputs are missing, including:

- `baseline_reference.json`
- `protected_hash_verification.json`
- `robustness_matrix_plan.json`
- `robustness_matrix.csv`
- `robustness_summary.json`
- `ROBUSTNESS_RUN_COMPLETE.json`
- per-row logs
- per-row `summary.json` and `RUN_COMPLETE.json`

The final result archive should also contain capsule-level `completion_ledger.json`, `result_integrity_manifest.json`, control-panel status files, and full `J1` logs.

## Results

Status: `PENDING_COLAB_RUN`

When the archive is returned, update this section with:

- archive filename and SHA-256;
- Colab runtime and GPU details;
- `ROBUSTNESS_RUN_COMPLETE.json` verdict;
- convergence drift table;
- epsilon fit and `R^2`;
- sign/gate failures, if any;
- final review status, using factual language only.
