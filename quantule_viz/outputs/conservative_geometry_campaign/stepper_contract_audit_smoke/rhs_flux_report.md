# Conservative C2 RHS Flux Report

CuPy-only standalone diagnostic. No stability claim is made.

Native C2 profile: `not_run_no_artifact`
Hard stop triggered: `False`

Thresholds: `<=1e-10` numerical zero; `>1e-8` warning; `>1e-6` hard stop unless C2 contract says norm is not conserved.

| case | status | diagnostic norm | physical-grid norm | fractional flux | d_norm_dt_raw | nodes |
|---|---|---:|---:|---:|---:|---:|
| uniform_constant_control | numerical_zero | 32768 | 1000 | 0 | 0 | 0 |
| single_gaussian_node | numerical_zero | 105.59199757 | 3.2224120352 | 2.53239063169e-16 | 2.67400185427e-14 | 1 |
| ablated_triangle_spacing_0.45 | numerical_zero | 211.328143115 | 6.44922311753 | 7.86505236431e-17 | 1.66210691165e-14 | 2 |
| triangle_spacing_0.45 | numerical_zero | 317.21232172 | 9.6805518103 | -1.52668557651e-16 | -4.84283476262e-14 | 3 |