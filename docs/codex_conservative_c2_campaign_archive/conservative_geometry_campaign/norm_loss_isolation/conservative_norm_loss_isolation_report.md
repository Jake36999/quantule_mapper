# Conservative C2 Norm-Loss Isolation

Standalone CuPy-only diagnostic. No production solver, Hunter, validation, config, or JAX scout reference files were modified.

## Summary

Recommendation: **timestep-sensitive numerical loss**.

Native C2 profile test: not run; no existing native C2 soliton NPZ artifact with psi-like field was found

## Results

| case | group | N | dt | nonlinear | mask | norm loss | rho max change | high-k final | recomb mask mean | nodes |
|---|---|---:|---:|:---:|---|---:|---:|---:|---:|---:|
| linear_current_N48_dt0.001 | linear_only | 48 | 0.001 | False | current | 2.63353e-14 | -0.927952 | 0 | 0 | 8 |
| linear_none_N48_dt0.001 | linear_only | 48 | 0.001 | False | none | 2.63353e-14 | -0.927952 | 3.41186e-18 | 0 | 8 |
| mask_current_N48_dt0.001 | mask_isolation | 48 | 0.001 | True | current | -0.186933 | -0.510102 | 0 | 0 | 3 |
| mask_nonlinear_only_N48_dt0.001 | mask_isolation | 48 | 0.001 | True | nonlinear_only | -0.186933 | -0.510102 | 0 | 0 | 3 |
| mask_no_recombination_N48_dt0.001 | mask_isolation | 48 | 0.001 | True | no_recombination | -0.186933 | -0.510102 | 0 | 0 | 3 |
| mask_none_N48_dt0.001 | mask_isolation | 48 | 0.001 | True | none | -0.186933 | -0.510102 | 2.05243e-17 | 0 | 3 |
| dt_0.001_N48 | timestep | 48 | 0.001 | True | current | -0.186933 | -0.510102 | 0 | 0 | 3 |
| dt_0.0005_N48 | timestep | 48 | 0.0005 | True | current | -0.109925 | -0.329815 | 0 | 0 | 3 |
| dt_0.00025_N48 | timestep | 48 | 0.00025 | True | current | -0.0596741 | -0.157439 | 0 | 0 | 3 |
| resolution_N48_dt0.001 | resolution | 48 | 0.001 | True | current | -0.186933 | -0.510102 | 0 | 0 | 3 |
| resolution_N64_dt0.001 | resolution | 64 | 0.001 | True | current | -0.186933 | -0.510279 | 0 | 0 | 3 |

## Interpretation Notes

- Linear-only cases test the conservative spectral unitary operator with nonlinear RHS disabled.
- `current` keeps both nonlinear-transform and recombination masks.
- `nonlinear_only` and `no_recombination` keep nonlinear-transform masking but disable the recombination mask.
- `none` is diagnostic-only: the dealias mask is replaced with ones, so nonlinear and recombination masks are disabled.
- Recombination-mask removal is measured directly; nonlinear-transform mask removal is inferred by mode contrasts.
- No stability claim is made from these isolation tests.