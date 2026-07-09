# Conservative C2 Nonlinear Stepper Contract Audit

Standalone CuPy-only diagnostic. No production solver, worker, Hunter, validation, config, or JAX scout reference files were modified.

## Summary

- Primary classification: `unclear`
- Secondary flags: `geometry_persistence_short_horizon, long_campaign_not_justified, explicit_rk4_not_conservative_benchmark, split_step_not_supported_by_rhs_structure, profile_mismatch_untested`
- Final decision: `UNCLEAR_BLOCKED`
- Stopped after: `completed_available_audit`
- Native C2 profile: `not_run_no_artifact`
- One-step norm-defect dt slope: `None`

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
| single_gaussian_node | rhs_flux | instantaneous_conservative_nonlinear_rhs | 0.001 | 0 | 2.53239063169037e-16 | 0.0 | numerical_zero | True |
| ablated_triangle_spacing_0.45 | rhs_flux | instantaneous_conservative_nonlinear_rhs | 0.001 | 0 | 7.865052364307719e-17 | 0.0 | numerical_zero | True |
| triangle_spacing_0.45 | rhs_flux | instantaneous_conservative_nonlinear_rhs | 0.001 | 0 | -1.5266855765116642e-16 | 0.0 | numerical_zero | True |
| one_step_etdrk4_dt_0.001 | one_step | etdrk4_current | 0.001 | 1 | -5.9539173080213106e-05 | -0.0002166957074976708 |  | True |
| etdrk4_current_dt_0.001_T_0.02 | multistep | etdrk4_current | 0.001 | 20 | -0.001188085781128688 | -0.004194388945599935 |  | True |
| etdrk4_no_mask_dt_0.001_T_0.02 | multistep | etdrk4_no_mask | 0.001 | 20 | -0.0011881104419861977 | -0.004008132533551569 |  | True |
| rk4_full_rhs_dt_0.001_T_0.02 | multistep | rk4_full_rhs | 0.001 | 20 | -1.3905899947093994e-07 | -1.0587676254232465e-05 |  | True |
| rk4_nonlinear_only_dt_0.001_T_0.02 | multistep | rk4_nonlinear_only | 0.001 | 20 | 1.429529359277501e-05 | -0.032743301708206496 |  | True |