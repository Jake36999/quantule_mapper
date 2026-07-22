# External Validation Automation Report

Status: provisional until Claude review.

Final decision: `VALIDATION_HARNESS_READY_WITH_GAPS`

## Summary

- Metrics run successfully: 6
- Metrics skipped with reason: 0
- Metrics failed: 0
- No simulations run.
- No production physics changed.
- No verdicts changed.
- No physical correspondence claims added.

## Metric Outcomes

| Metric | Status | Depth | Behavioural result | Match-level candidate | Main caveat |
|---|---|---|---|---|---|
| `v1` | `RAN` | `INTERNAL_DATA_ANALYTIC_FIT` | `PASS` | `SAME FAMILY` | Constants C/lambda are freely fit; canonical NLS constants are secondary context only. |
| `v2` | `RAN` | `INTERNAL_DATA_BEHAVIOUR_CHECK` | `PASS` | `EXACT EQUATION` | sweep_runs/C27_REDERIVE/r3_n96.json lacks explicit D; using project C2.7 convention D=1.0. |
| `v3` | `RAN` | `INTERNAL_DATA_BEHAVIOUR_CHECK` | `PASS` | `SAME FAMILY` | VK sign depends on branch variable convention; this report uses omega/w. |
| `v5` | `RAN` | `INTERNAL_DATA_BEHAVIOUR_CHECK` | `PASS` | `CLOSE ANALOGUE` | Uses existing outcome labels only; no collision reinterpretation is performed. |
| `v6` | `RAN` | `INTERNAL_DATA_BEHAVIOUR_CHECK` | `PASS` | `EXACT EQUATION` | V6 is an instrument regression/overlay check, not an empirical validation. |
| `v7` | `RAN` | `INTERNAL_DATA_BEHAVIOUR_CHECK` | `PARTIAL` | `CLOSE ANALOGUE` | This is a design constraint and soft-clip diagnosis, not a gravity result. |

## Files Read

- `sweep_runs/C27_REDERIVE/r3_n96.json`
- `sweep_runs/C29_ROBUST/summary.json`
- `sweep_runs/C29_ROBUST/track_static_0.00.npz`
- `sweep_runs/C29_ROBUST/track_static_0.79.npz`
- `sweep_runs/C29_ROBUST/track_static_1.57.npz`
- `sweep_runs/C29_ROBUST/track_static_2.36.npz`
- `sweep_runs/C29_ROBUST/track_static_3.14.npz`
- `sweep_runs/C29_VALIDATE/summary.json`
- `sweep_runs/C3_ANTIPHASE_LADDER/summary.json`
- `sweep_runs/C3_COLLISION_LADDER_FULL/summary.json`
- `sweep_runs/C3_EXACT_VK2/summary.json`
- `sweep_runs/C3_PHASE_3pi4/summary.json`
- `sweep_runs/C3_PHASE_7pi8/summary.json`
- `sweep_runs/C3_PHASE_pi2/summary.json`
- `sweep_runs/C3_TWOQBALL_FULL/pair_static_0.00.npz`
- `sweep_runs/C3_TWOQBALL_FULL/pair_static_0.79.npz`
- `sweep_runs/C3_TWOQBALL_FULL/pair_static_1.57.npz`
- `sweep_runs/C3_TWOQBALL_FULL/pair_static_2.36.npz`
- `sweep_runs/C3_TWOQBALL_FULL/pair_static_3.14.npz`
- `sweep_runs/C3_TWOQBALL_FULL/summary.json`
- `sweep_runs/GRAVITY_AD_bg0/summary.json`
- `sweep_runs/GRAVITY_DESAT_PILOT/desat_pilot.json`
- `sweep_runs/PHASE_D_C2_6_CODEX_AUDIT_20260709/c2_6_independent_audit_summary.json`
- `sweep_runs/PHASE_D_CODEX_REPRODUCTION_20260709_233433/c2_6_reproduction_summary.json`
- `sweep_runs/PHASE_D_CODEX_REPRODUCTION_20260709_233433/gravity_geometry/radial_profiles.csv`

## Commands Run

- `F:\quantule_mapper\.venv\Scripts\python.exe tools/external_validation/run_external_validation.py --metric all --mode internal_analytic`

## Protected Diff Status

Expected protected diff is empty.

```text
<empty>
```

## Output Root

`docs/external_validation/generated_metrics`

## Guardrails

All results are first-pass numerical summaries for external legibility. They do not validate IRER experimentally, do not change project verdicts, and do not claim matter, gravity, or unification.
