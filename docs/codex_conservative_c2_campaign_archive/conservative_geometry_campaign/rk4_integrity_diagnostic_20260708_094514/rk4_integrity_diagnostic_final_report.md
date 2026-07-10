# RK4 Integrity Diagnostic Final Report

Final decision: `RK4_RHS_FLUX_FAIL`
Secondary flags: `no_stability_claim, no_production_change, no_amplitude_normalization, no_jax, long_campaign_not_run, rk4_diagnostic_only, budget_respected`
Total wallclock seconds: `77.998`

## Environment

- Python: `F:\quantule_mapper\.venv\Scripts\python.exe`
- CuPy: `14.0.1`
- GPU: `NVIDIA GeForce GTX 1080`
- JAX/JAXLIB absent: `True`
- Protected diff empty: `True`

## Results

- DT integrity passed: `True`
- RHS flux failed: `True`
- N64 replay rows: `0`
- N64 optional T2 skipped due to budget: `False`

No stability claim is made. RK4 remains diagnostic-only. No longer campaign was run.