# CX GPU Follow-up Tests IPYNB Provenance Report

## Capsule Identity

- Job name: `CX GPU Follow-up Tests Capsule`
- Job id: `cx_gpu_followup_tests`
- Notebook: `F:\quantule_mapper\colab_jobs\cx_gpu_followup_tests.ipynb`
- Manifest: `F:\quantule_mapper\colab_jobs\cx_gpu_followup_tests_manifest.json`
- Operator instructions: `F:\quantule_mapper\colab_jobs\cx_gpu_followup_tests_operator_instructions.md`
- Local validation report: `F:\quantule_mapper\colab_jobs\cx_gpu_followup_tests_local_validation_report.md`

## Included Queue Rows

The notebook packages these GPU-facing tests:

| button | queue row / run |
|---|---|
| `J1` | `CX_TG_B2_COOLED_PAIR_SECULAR_FORCE` |
| `J2` | `CX_TG_CLOCK_MIGRATION_CHARACTERIZATION` |
| `J3` | `CX_TG_RATE_SOURCE_SEMANTICS_BRIDGE` |
| `J4` | `CX_GRAVITY_D_LOAD_CAPACITY_YIELD_MAP` |

Excluded by request:

| script | reason |
|---|---|
| `jax_scout/gravity_TG_B1S_D4_discrepancy_review.py` | CPU-only review script; not included in this Colab GPU capsule |

## Bundle Hashes

| Artifact | SHA-256 |
|---|---|
| `cx_gpu_followup_tests_manifest.json` | `3af61e8ab1e0761b81f091064d46a0ac63d4ac6f4ef19a34fba2551cdb6f9524` |
| `cx_gpu_followup_tests.ipynb` | `d7e7de301a4b9562d1a809877805c9e746d99a1907e45bd5dd241f744b7936e1` |
| Packager dry-run source bundle | `3b8d5ed1acd7cf68f5807f38e3fb7ac46b5bd431bfbf4c849adaa5cfc08cfe54` |

## Bundled Files

The notebook embeds exactly these source files:

| path | role |
|---|---|
| `jax_scout/__init__.py` | package marker/import support |
| `jax_scout/phase_d_c3_wave.py` | KG/Q-ball helper dependency |
| `jax_scout/gravity_TG_B1S_state_load_feedback_gpu.py` | TG-B1S helper dependency |
| `jax_scout/gravity_TG_B2_two_node_awell.py` | TG-B2 A-well helper dependency |
| `jax_scout/gravity_TG_B2_definitive_force.py` | definitive TG-B2 body-force runner used by `J1` |
| `jax_scout/gravity_TG_B2_cooled_pair_secular_force_gpu.py` | `J1` wrapper |
| `jax_scout/gravity_G1_clock_calibration_gpu.py` | G1 clock machinery used by `J2` |
| `jax_scout/gravity_TG_clock_migration_characterization_gpu.py` | `J2` wrapper |
| `jax_scout/gravity_TG_S_source_semantics_gpu.py` | TG-S semantics runner used by `J3` |
| `jax_scout/gravity_TG_rate_source_semantics_bridge_gpu.py` | `J3` wrapper |
| `jax_scout/gravity_D_neutral_probe_gpu.py` | Gravity-D helper dependency |
| `jax_scout/gravity_D_dynamics_characterization_gpu.py` | Gravity-D characterization helper used by `J4` |
| `jax_scout/gravity_D_load_capacity_yield_map_gpu.py` | `J4` wrapper |

No external Drive inputs are declared.

## Commands

`J1`:

```text
python jax_scout/gravity_TG_B2_cooled_pair_secular_force_gpu.py --out /content/qm_job/results/cx_tg_b2_cooled_pair_secular_force --seps 3.0,4.0 --cool-T-values 0,40 --T 180
```

`J2`:

```text
python jax_scout/gravity_TG_clock_migration_characterization_gpu.py --out /content/qm_job/results/cx_tg_clock_migration_characterization --matrix pilot --time-domain selected
```

`J3`:

```text
python jax_scout/gravity_TG_rate_source_semantics_bridge_gpu.py --out /content/qm_job/results/cx_tg_rate_source_semantics_bridge
```

`J4`:

```text
python jax_scout/gravity_D_load_capacity_yield_map_gpu.py --out /content/qm_job/results/cx_gravity_d_load_capacity_yield_map --matrix pilot
```

## Reason

These rows were newly queued as standalone follow-up runners. The notebook lets Jake dispatch the GPU-capable rows through the Colab fast lane while keeping each row independently controlled and separately archived from the central panel.

## Results

Status: `PENDING_COLAB_RUN`

When results return, update this section with:

- which job button was run;
- archive filename and SHA-256 if available;
- runtime and GPU details;
- row-specific machine verdict;
- integrity check result;
- first-pass scientific/infrastructure review status.
