# CL TG-R Robustness Capsule v2 IPYNB Provenance Report

## Capsule Identity

- Queue row: `CL_TG_R_ROBUSTNESS_CAPSULE`
- Job name: `CL TG-R Robustness Capsule v2 - Complete Missing Evidence`
- Job id: `cl_tg_r_robustness_capsule_v2`
- Row id: `tg_r_robustness_v2_completion`
- Notebook: `F:\quantule_mapper\colab_jobs\cl_tg_r_robustness_capsule_v2.ipynb`
- Manifest: `F:\quantule_mapper\colab_jobs\cl_tg_r_robustness_capsule_v2_manifest.json`
- Operator instructions: `F:\quantule_mapper\colab_jobs\cl_tg_r_robustness_capsule_v2_operator_instructions.md`
- Local validation report: `F:\quantule_mapper\colab_jobs\cl_tg_r_robustness_capsule_v2_local_validation_report.md`
- Generated for manual Google Colab upload and A100 execution.

## Bundle Hashes

| Artifact | SHA-256 |
|---|---|
| `cl_tg_r_robustness_capsule_v2_manifest.json` | `a502b39e6d4839ccc7ef250ac9042ccf9987cf6fc4763077d3054ed3ffd3eaf1` |
| `tg_r_robustness_driver_v2.py` | `957f776ebbcc38e645f64666f8690e7e8993be32fc2939d8b45fcbfa2d7dfd32` |
| `baselines/tg_r_robustness_v1_partial_evidence_20260717.json` | `b641fbffe0bd38553cb7a0899fb56e2918003a922f53da4b3c50d7bc0894ff21` |
| `cl_tg_r_robustness_capsule_v2.ipynb` | `8ec750d92a63930c7e92de1761666c0c904319b3809dfaff506584a688e98500` |
| Packager dry-run source bundle | `6b25a9d05592d184f9ca2c958e92b306177a852fb39b0e3b905e04dd0ef7cc31` |

## Bundled Files

The notebook embeds exactly these files. It does not embed the whole repository.

| Bundled path | Role |
|---|---|
| `colab_jobs/tg_r_robustness_driver_v2.py` | Phase R v2 completion runner and aggregator |
| `colab_jobs/baselines/tg_b2_definitive_force_20260716_baseline_summary.json` | Reviewed definitive A100 baseline |
| `colab_jobs/baselines/tg_r_robustness_v1_partial_evidence_20260717.json` | Frozen returned v1 partial evidence |
| `docs/gravity_maturity/TG_PROTECTED_REGISTRY.json` | Protected-module hash contract |
| `jax_scout/__init__.py` | Package marker/import support |
| `jax_scout/phase_d_c3_wave.py` | Protected Q-ball/KG machinery |
| `jax_scout/gravity_TG_B1S_state_load_feedback_gpu.py` | Protected frozen TG-B1S helper/scaffold |
| `jax_scout/gravity_TG_B2_two_node_awell.py` | Protected TG-B2 A-well dynamics helper |
| `jax_scout/gravity_TG_B2_definitive_force.py` | Protected definitive body-force entrypoint |

## What Test Is Being Run

The packaged `J1` action runs:

```text
python colab_jobs/tg_r_robustness_driver_v2.py --out /content/qm_job/results/cl_tg_r_robustness_v2 --baseline-summary colab_jobs/baselines/tg_b2_definitive_force_20260716_baseline_summary.json --v1-evidence colab_jobs/baselines/tg_r_robustness_v1_partial_evidence_20260717.json --protected-registry docs/gravity_maturity/TG_PROTECTED_REGISTRY.json
```

The driver first verifies protected file hashes, loads the reviewed baseline and returned v1 partial evidence, then invokes `jax_scout/gravity_TG_B2_definitive_force.py` unchanged for each v2 row.

V2 matrix:

| row | command modifiers |
|---|---|
| `r2_eps_003_sep3` | `--seps 3.0 --epsilon-G 0.03 --T 180` |
| `r2_eps_012_sep3` | `--seps 3.0 --epsilon-G 0.12 --T 180` |
| `r1_box_L20_N96_sep3` | `--seps 3.0 --L 20 --N 96 --T 180` |

Preserved v1 rows:

| row | source |
|---|---|
| `r1_grid_N96_sep3` | returned v1 capsule; PASS |
| `r1_dt_half_sep3` | returned v1 capsule; PASS |

The baseline point is the reviewed `epsilon_G=0.06`, `N=64`, `L=16`, `dt=0.002`, `sep=3.0` Colab A100 definitive result:

```text
<F_R_well> = -5.4501234166498155e-05
```

## Reason

The v1 robustness capsule returned trustworthy partial evidence, but stopped before epsilon scaling after the original larger-box row `L=20,N=64` failed during Q-ball setup. V2 avoids rerunning successful rows, runs epsilon scaling first, and replaces the failed larger-box setup with a preregistered repaired sentinel `L=20,N=96`.

The capsule intentionally separates infrastructure completion from scientific verdict. If one independent row fails, the result archive should still return for review with the failure recorded in `robustness_summary.json`.

## Expected Outputs

The capsule fails closed if core outputs are missing, including:

- `baseline_reference.json`
- `v1_partial_evidence_reference.json`
- `protected_hash_verification.json`
- `robustness_matrix_plan.json`
- `robustness_matrix.csv`
- `robustness_matrix_v2_only.csv`
- `robustness_summary.json`
- `ROBUSTNESS_RUN_COMPLETE.json`

Per-row summaries are reviewed when present, but they are not manifest-level required outputs because v2 must return controlled row failures for review.

## Results

Status: `PENDING_COLAB_RUN`

When the archive is returned, update this section with:

- archive filename and SHA-256;
- Colab runtime and GPU details;
- `ROBUSTNESS_RUN_COMPLETE.json` verdict;
- epsilon fit and `R^2`;
- repaired box row status and drift;
- sign/gate failures, if any;
- final review status, using factual language only.
