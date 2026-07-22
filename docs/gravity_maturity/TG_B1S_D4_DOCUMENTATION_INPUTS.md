# TG-B1S D4 Documentation Inputs

Author: Codex  
Timestamp: 2026-07-15  

## Final D4 Label

```text
TG_STATE_LOAD_D4_NUMERICAL_VALIDATION_FAILED
```

## Artifact Paths

```text
sweep_runs/TG_B1S_D4_ROWS_GPU_20260715_091558/D4_RUN_COMPLETE.json
sweep_runs/TG_B1S_D4_ROWS_GPU_20260715_091558/D4_SUMMARY.json
sweep_runs/TG_B1S_D4_ROWS_GPU_20260715_091558/numerical_validation.csv
sweep_runs/TG_B1S_D4_ROWS_GPU_20260715_091558/asymptotic_frequency.csv
sweep_runs/TG_B1S_D4_ROWS_GPU_20260715_091558/attractor_classification.csv
sweep_runs/TG_B1S_D4_ROWS_GPU_20260715_091558/structural_trends.csv
sweep_runs/TG_B1S_D4_ROWS_GPU_20260715_091558/energy_ledger.csv
sweep_runs/TG_B1S_D4_ROWS_GPU_20260715_091558/boundary_flux.csv
sweep_runs/TG_B1S_D4_ROWS_GPU_20260715_091558/artifact_hashes.csv
```

## Strongest Supporting Facts

- D4 completed all rows.
- GPU preflight passed.
- Five of six rows passed.
- Timestep halving, output cadence, grid refinement and absorber widening preserved the reference frequency shift within about `0.55%`.
- The larger-box row failed frequency-scale tolerance with relative difference `0.6036934864398061`, above the preregistered `0.60` threshold.

## Caveat

The failed row remained clean in structural and boundedness diagnostics. Therefore the failure should be documented as a frequency-scale/box-comparability discrepancy, not a runaway or structural-drift failure.

## Recommended Wording

```text
D4 numerical validation did not close. The state-load frequency shift remained stable under timestep, output cadence, grid refinement and absorber checks, but the larger-box row fell just outside the preregistered frequency-scale tolerance. D5 should remain blocked until this discrepancy is reviewed.
```

