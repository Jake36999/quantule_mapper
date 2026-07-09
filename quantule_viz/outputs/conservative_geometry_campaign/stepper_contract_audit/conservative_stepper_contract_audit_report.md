# Conservative C2 Nonlinear Stepper Contract Audit

Standalone CuPy-only diagnostic. No production solver, worker, Hunter, validation, config, or JAX scout reference files were modified.

## Summary

- Primary classification: `etdrk4_timestep_error`
- Secondary flags: `geometry_persistence_short_horizon, long_campaign_not_justified, explicit_rk4_not_conservative_benchmark, split_step_not_supported_by_rhs_structure, profile_mismatch_untested`
- Final decision: `ETDRK4_ERROR_CONFIRMED`
- Stopped after: `completed_available_audit`
- Native C2 profile: `not_run_no_artifact`
- One-step norm-defect dt slope: `2.0002283882853304`

No stability claim is made. Geometry persistence is separate from invariant preservation. No longer geometry campaign was run. No amplitude normalization was applied. No JAX was used or installed.

## Contract Snapshot

- RHS boundary: `{'reference_expects': 'spectral_psi_k', 'reference_returns': 'spectral_rhs', 'wrapper_expects': 'spectral_psi_k', 'wrapper_returns': 'spectral_rhs'}`
- Linear operator max abs diff: `0.0`
- param_eta inactive max abs diff: `0.0`
- Source contract ok: `True`

## Split-Step Feasibility

- `split_step_not_supported_by_rhs_structure`: The nonlinear RHS uses covariant/flat Laplacian terms and is not safely expressible as dpsi/dt = i*F(rho,geometry)*psi with real multiplicative F.

## Results

| case | group | mode | dt | steps | norm defect/flux | rho max change | status | finite |
|---|---|---|---:|---:|---:|---:|---|:---:|
| uniform_constant_control | rhs_flux | instantaneous_conservative_nonlinear_rhs | 0.001 | 0 | 0.0 | 0.0 | numerical_zero | True |
| single_gaussian_node | rhs_flux | instantaneous_conservative_nonlinear_rhs | 0.001 | 0 | 4.930970263009903e-17 | 0.0 | numerical_zero | True |
| ablated_triangle_spacing_0.45 | rhs_flux | instantaneous_conservative_nonlinear_rhs | 0.001 | 0 | -5.0519064491347044e-17 | 0.0 | numerical_zero | True |
| triangle_spacing_0.45 | rhs_flux | instantaneous_conservative_nonlinear_rhs | 0.001 | 0 | 3.1204902409264404e-16 | 0.0 | numerical_zero | True |
| one_step_etdrk4_dt_0.001 | one_step | etdrk4_current | 0.001 | 1 | -5.95392293287088e-05 | -0.00020890884261281118 |  | True |
| one_step_etdrk4_dt_0.0005 | one_step | etdrk4_current | 0.0005 | 1 | -1.4881017116593284e-05 | -5.224231242714182e-05 |  | True |
| one_step_etdrk4_dt_0.00025 | one_step | etdrk4_current | 0.00025 | 1 | -3.719709943278738e-06 | -1.3064704537542839e-05 |  | True |
| one_step_etdrk4_dt_0.000125 | one_step | etdrk4_current | 0.000125 | 1 | -9.29855031819395e-07 | -3.2668291037145896e-06 |  | True |
| etdrk4_current_dt_0.001_T_4 | multistep | etdrk4_current | 0.001 | 4000 | -0.18693344798183317 | -0.5101020634822734 |  | True |
| etdrk4_no_mask_dt_0.001_T_4 | multistep | etdrk4_no_mask | 0.001 | 4000 | -0.1869334490746488 | -0.5101020365822386 |  | True |
| rk4_full_rhs_dt_0.001_T_0.5 | multistep | rk4_full_rhs | 0.001 | 500 | -9.203073115357175e-05 | -0.016215469539479718 |  | True |
| rk4_nonlinear_only_dt_0.001_T_0.5 | multistep | rk4_nonlinear_only | 0.001 | 500 | 0.00021865078622997146 | -0.9334567863963624 |  | True |
| etdrk4_current_dt_0.0005_T_4 | multistep | etdrk4_current | 0.0005 | 8000 | -0.10992507369730657 | -0.3298154607040295 |  | True |
| etdrk4_no_mask_dt_0.0005_T_4 | multistep | etdrk4_no_mask | 0.0005 | 8000 | -0.1099250844975598 | -0.3298143986539953 |  | True |
| rk4_full_rhs_dt_0.0005_T_0.5 | multistep | rk4_full_rhs | 0.0005 | 1000 | -9.203073115335937e-05 | -0.016215469539479044 |  | True |
| rk4_nonlinear_only_dt_0.0005_T_0.5 | multistep | rk4_nonlinear_only | 0.0005 | 1000 | 0.0002186508459998148 | -0.9334567902640747 |  | True |
| etdrk4_current_dt_0.00025_T_4 | multistep | etdrk4_current | 0.00025 | 16000 | -0.05967407055812564 | -0.15743901791148604 |  | True |
| etdrk4_no_mask_dt_0.00025_T_4 | multistep | etdrk4_no_mask | 0.00025 | 16000 | -0.059674075045822145 | -0.15744054344923836 |  | True |
| rk4_full_rhs_dt_0.00025_T_0.5 | multistep | rk4_full_rhs | 0.00025 | 2000 | -9.203073115357175e-05 | -0.016215469539479718 |  | True |
| rk4_nonlinear_only_dt_0.00025_T_0.5 | multistep | rk4_nonlinear_only | 0.00025 | 2000 | 0.0002186508478585763 | -0.9334567904909306 |  | True |
| etdrk4_current_dt_0.000125_T_4 | multistep | etdrk4_current | 0.000125 | 32000 | -0.033187769428700886 | -0.06063792995309749 |  | True |
| etdrk4_no_mask_dt_0.000125_T_4 | multistep | etdrk4_no_mask | 0.000125 | 32000 | -0.03318612207027538 | -0.06090821138991286 |  | True |
| rk4_full_rhs_dt_0.000125_T_0.5 | multistep | rk4_full_rhs | 0.000125 | 4000 | -9.203073115166032e-05 | -0.01621546953947871 |  | True |
| rk4_nonlinear_only_dt_0.000125_T_0.5 | multistep | rk4_nonlinear_only | 0.000125 | 4000 | 0.00021865084791804307 | -0.9334567905045916 |  | True |