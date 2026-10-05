# CX GPU Follow-up Tests Operator Instructions

This notebook packages four GPU-facing follow-up tests from the queue. The CPU-only D4 discrepancy review script is intentionally excluded.

Notebook:

```text
F:\quantule_mapper\colab_jobs\cx_gpu_followup_tests.ipynb
```

Drive result path:

```text
/content/drive/MyDrive/QuantuleMapperRuns/cx_gpu_followup_tests
```

## Scope

- Run class: `full-campaign`, but each test is exposed as an independent job button.
- Execution scope: full-fidelity GPU scripts with the declared pilot/full arguments.
- Fidelity requirement: scientific source unchanged; wrapper scripts only.
- Resource profile: A100 aggressive, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- Archive policy: 1 GB preferred, 5 GB hard cap.

Do not run all four jobs by habit. Use the button for the test you intend to dispatch.

## Buttons

Run the fixed tests first:

| button | purpose |
|---|---|
| `T1` | runtime/GPU/JAX/x64 preflight |
| `T2` | numeric smoke test |
| `T3` | Drive persistence test |

Then choose one or more job buttons:

| button | row | notes |
|---|---|---|
| `J1` | `cx_tg_b2_cooled_pair_secular_force` | Longest row. Runs `seps=3.0,4.0`, `cool-T=0,40`, `T=180`; may take several A100 hours. |
| `J2` | `cx_tg_clock_migration_characterization` | Pilot matrix, selected time-domain rows. |
| `J3` | `cx_tg_rate_source_semantics_bridge` | TG-S semantics bridge only; no feedback loop. |
| `J4` | `cx_gravity_d_load_capacity_yield_map` | Gravity-D instantaneous yield-map pilot. |

## Manual Colab Steps

1. Upload `F:\quantule_mapper\colab_jobs\cx_gpu_followup_tests.ipynb` to Colab.
2. Select an A100 GPU runtime.
3. Run cells from the top until the control panel appears.
4. Click `T1`, then `T2`, then `T3`.
5. Click the job button for the row you want to run.
6. Wait for the row to finish and archive to Drive.
7. Download the returned archive from Drive.
8. Place the archive or extracted folder under `F:\quantule_mapper\colab_jobs\results`.
9. Ask Codex/Claude to review the returned capsule.

## Stop Rules

For any job button:

- stop and return logs if no new output appears for 30 minutes;
- stop if runtime exceeds the declared row expectation without `RUN_COMPLETE.json` or archive progress;
- do not reinterpret a failed dashboard state as a scientific failure until logs and returned artifacts are reviewed.

## Expected Outputs

`J1` cooled pair:

- `cooled_pair_metrics.csv`
- `secular_force_assessment.json`
- `run_manifest.csv`
- `RUN_COMPLETE.json`

`J2` clock migration:

- `clock_migration_metrics.csv`
- `eigenfrequency_metrics.csv`
- `time_domain_clock_metrics.csv`
- `migration_summary.json`
- `RUN_COMPLETE.json`

`J3` rate-source semantics:

- `rate_source_admissibility.csv`
- `rate_source_bridge_summary.json`
- `RUN_COMPLETE.json`

`J4` Gravity-D yield map:

- `instantaneous_force_atlas.csv`
- `yield_classification.csv`
- `yield_summary.json`
- `RUN_COMPLETE.json`

Every returned archive should also contain `completion_ledger.json`, `result_integrity_manifest.json`, control-panel logs/status files, and capsule provenance.
