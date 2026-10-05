# CX TG-B1S D4 A100 Short Pilot Provenance Report

## Capsule Identity

- Queue row: `CX_TG_B1S_D4_A100_SHORT_PILOT`
- Job name: `CX TG-B1S D4 A100 Short Scientific Pilot`
- Job id: `cx_tg_b1s_d4_a100_short_pilot`
- Row id: `d4_dt_half_2p_n32`
- Notebook: `F:\quantule_mapper\colab_jobs\cx_tg_b1s_d4_a100_short_pilot.ipynb`
- Manifest: `F:\quantule_mapper\colab_jobs\cx_tg_b1s_d4_a100_short_pilot_manifest.json`
- Operator instructions: `F:\quantule_mapper\colab_jobs\cx_tg_b1s_d4_a100_short_pilot_operator_instructions.md`
- Local validation report: `F:\quantule_mapper\colab_jobs\cx_tg_b1s_d4_a100_short_pilot_local_validation_report.md`
- Generated for manual Google Colab upload and A100 execution.

## Bundle Hashes

| Artifact | SHA-256 |
|---|---|
| `cx_tg_b1s_d4_a100_short_pilot_manifest.json` | `9be4b8eb44cd519eab97637d3989bd67423654ba67795bb5b5709caf054b28f9` |
| `cx_tg_b1s_d4_a100_short_pilot.ipynb` | `6f5e1967b93416e80654ad2da569fbd7915878edd0a33638be9696659e285a06` |
| Packager dry-run source bundle | `391a88f74cad5aa9bb0fdece87d3f531187335423f38f9276a01f71148373eed` |

## Bundled Files

The notebook embeds exactly the explicit source and reference files declared in the manifest. It does not embed the whole repository.

### Source Allowlist

| Bundled path | Role |
|---|---|
| `jax_scout/__init__.py` | Package marker/import support |
| `jax_scout/phase_d_c3_wave.py` | Q-ball/KG support |
| `jax_scout/gravity_TG_B1S_state_load_feedback_gpu.py` | Frozen TG-B1S state-load feedback model |
| `jax_scout/gravity_TG_B1S_backreaction_robustness_gpu.py` | Required B1S helper dependency |
| `jax_scout/gravity_TG_B1S_long_time_gpu.py` | Required B1S helper dependency |
| `jax_scout/gravity_TG_B1S_drift_decomposition_gpu.py` | Drift decomposition machinery |
| `jax_scout/gravity_TG_B1S_D4_D5_closure_gpu.py` | D4/D5 gate and closure helpers |
| `jax_scout/gravity_TG_B1S_D4_rows_gpu.py` | D4 row-wise pilot entrypoint |

### Packaged Reference Run

The capsule includes selected files from:

```text
sweep_runs/TG_B1S_DRIFT_DECOMPOSITION_GPU_20260714_184736
```

These are packaged to let the notebook run entirely from `/content/qm_job` without Drive-side scientific inputs.

## What Test Is Being Run

The packaged `J1` action runs:

```text
python -u jax_scout/gravity_TG_B1S_D4_rows_gpu.py --out /content/qm_job/results/cx_tg_b1s_d4_a100_short_pilot --reference-run /content/qm_job/sweep_runs/TG_B1S_DRIFT_DECOMPOSITION_GPU_20260714_184736 --cases dt_half --d4-periods 2 --N 32
```

This is a short scientific pilot of the real D4 `dt_half` pathway. It is not the full 50-period D4 campaign and must not be interpreted as D4 closure.

## Reason

The queue asks for a Colab A100 short pilot that exercises a real TG-B1S D4 numerical-validation row without launching the full active D4 campaign. This tests the fast-lane deployment path for the D4 row runner while keeping runtime appropriate for a first Colab trial.

## Expected Outputs

The capsule requires the D4 row runner's standard outputs, including:

- `environment_versions.json`
- `model_spec.json`
- `preregistered_gates.json`
- `preregistered_matrix.json`
- `baseline_reproduction.csv`
- `numerical_validation.csv`
- `asymptotic_frequency.csv`
- `structural_trends.csv`
- `field_boundedness.csv`
- `energy_ledger.csv`
- `boundary_flux.csv`
- `orbital_distance.csv`
- `phase_alignment.csv`
- `attractor_classification.csv`
- `falsification_results.csv`
- `artifact_hashes.csv`
- `ROW_D4_dt_half_50P_COMPLETE.json`
- `D4_RUN_COMPLETE.json`
- `D4_SUMMARY.json`
- `TECHNICAL_HANDOFF.md`

The `ROW_D4_dt_half_50P_COMPLETE.json` label is expected even for the 2-period pilot because the D4 runner uses fixed case names.

## Results

Status: `PENDING_COLAB_RUN`

When the archive is returned, update this section with:

- archive filename and SHA-256;
- Colab runtime and GPU details;
- `D4_RUN_COMPLETE.json` status;
- whether the short pilot generated every expected artifact;
- observed D4 row outcome;
- final review status, using pilot-only language.
