# RK4 Stepper Diagnostic Final Report

Final decision: `RK4_STEPPER_DIAGNOSTIC_PROMISING`
Secondary flags: `no_stability_claim, no_production_change, no_amplitude_normalization, long_campaign_not_run, explicit_rk4_diagnostic_only, profile_mismatch_untested`

## Files Added Or Modified

- `tools/conservative_rk4_stepper_diagnostic.py`
- `tests/test_conservative_rk4_stepper_diagnostic.py`
- Diagnostic outputs under `quantule_viz/outputs/conservative_geometry_campaign/rk4_stepper_diagnostic/`

## Environment

- Python path: `F:\quantule_mapper\.venv\Scripts\python.exe`
- CuPy version: `14.0.1`
- GPU: `NVIDIA GeForce GTX 1080`
- JAX/JAXLIB absent: `True`

## Commands Run

- `smoke`

## Summary

- Conservative C2 spectral contract reused from prior audit.
- RK4 is diagnostic-only and not promoted as production.
- One-step RK4 gate passed: `True`
- Multi-step RK4 gate passed: `True`
- Geometry replay classification: `None`
- No stability claim is made.
- No longer conservative campaign was run.