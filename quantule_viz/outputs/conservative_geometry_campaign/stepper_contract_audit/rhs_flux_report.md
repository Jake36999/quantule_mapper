# Conservative C2 RHS Flux Report

CuPy-only standalone diagnostic. No stability claim is made.

Native C2 profile: `not_run_no_artifact`
Hard stop triggered: `False`

Thresholds: `<=1e-10` numerical zero; `>1e-8` warning; `>1e-6` hard stop unless C2 contract says norm is not conserved.

| case | status | diagnostic norm | physical-grid norm | fractional flux | d_norm_dt_raw | nodes |
|---|---|---:|---:|---:|---:|---:|
| uniform_constant_control | numerical_zero | 110592 | 1000 | 0 | 0 | 0 |
| single_gaussian_node | numerical_zero | 356.372991797 | 3.2224120352 | 4.93097026301e-17 | 1.75726462509e-14 | 1 |
| ablated_triangle_spacing_0.45 | numerical_zero | 713.232483016 | 6.44922311755 | -5.05190644913e-17 | -3.60318378068e-14 | 2 |
| triangle_spacing_0.45 | numerical_zero | 1070.59158576 | 9.68055180987 | 3.12049024093e-16 | 3.34077059537e-13 | 3 |