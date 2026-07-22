# External Validation Behavioural Comparison Report

All results remain `PROVISIONAL_UNTIL_CLAUDE_REVIEW`.

Final decision: `BEHAVIOURAL_VALIDATION_READY_WITH_GAPS`

## Depth Audit Result

The harness distinguishes formalism alignment from measured-data behavioural comparison. IRER formulations remain primary; external models are comparison instruments, not replacements.

## Numeric Summary

| Metric | Status | Depth | Behavioural result | Key numeric fields |
|---|---|---|---|---|
| `v1` | `RAN` | `INTERNAL_DATA_ANALYTIC_FIT` | `PASS` | phase_sign_accuracy=1.0 |
| `v2` | `RAN` | `INTERNAL_DATA_BEHAVIOUR_CHECK` | `PASS` | slope_percent_error=0.010901254911721558, r_squared=1.0 |
| `v3` | `RAN` | `INTERNAL_DATA_BEHAVIOUR_CHECK` | `PASS` | dQ_domega_fit=-849.138404192748 |
| `v5` | `RAN` | `INTERNAL_DATA_BEHAVIOUR_CHECK` | `PASS` | anti_phase_transmission_cells=3 |
| `v6` | `RAN` | `INTERNAL_DATA_BEHAVIOUR_CHECK` | `PASS` | corrected_v_over_2Dk=0.9998909278496895 |
| `v7` | `RAN` | `INTERNAL_DATA_BEHAVIOUR_CHECK` | `PARTIAL` | fitted_effective_exponent=2.9259806516213103, cliff_detected=True |

## External-Data Readiness

- External data mode status: `SKIPPED_WITH_REASON`
- Reason: No direct external machine-readable dynamics dataset is currently ready for these metrics.

## Limitations

- V1 uses measured separation tracks and free analytic-law fits; it does not assert canonical pure-NLS constants.
- V5 uses existing collision labels and telemetry only; it does not reinterpret outcomes.
- V7 is a design-constraint and geometry-law diagnostic, not a gravity result.
- No external dataset is promoted beyond the readiness labels in `DATASET_CANDIDATES.md`.

## Claude Review Checklist

- Review whether each behavioural threshold is appropriate.
- Review V1 acceleration-window choice and whether later fits should use more robust trajectory models.
- Review whether any external dataset should be digitised before Level-3 comparison.

Required statements:

- No simulations were run.
- No production physics changed.
- No verdicts changed.
- No physical-correspondence claims added.
- IRER formulations were preserved as primary.
