# Conservative C2 Diagnostic Handover Summary

Date: 2026-07-08

This note summarizes the current Conservative C2 diagnostic status for the main workflow. It is a handover document only. No stability claim is made.

## Scope

- Regime: Conservative C2 / moving substrate diagnostics only.
- Runtime used: `F:\quantule_mapper\.venv\Scripts\python.exe`.
- GPU runtime: CuPy on `NVIDIA GeForce GTX 1080`.
- JAX/JAXLIB were not used or installed.
- No production solver, worker, Hunter, validation, production config, or `jax_scout` reference files were modified.
- No amplitude normalization was applied.
- No longer geometry campaign or N64 replay should proceed until the invariant/contract issue is reviewed.

## Tests Run

Latest diagnostic test command:

```powershell
F:\quantule_mapper\.venv\Scripts\python.exe -m pytest tests/test_weighted_invariant_audit.py tests/test_rhs_term_flux_decomposition.py tests/test_rhs_flux_source_isolation.py tests/test_rk4_integrity_batch.py tests/test_conservative_rk4_stepper_diagnostic.py tests/test_conservative_stepper_contract_audit.py tests/test_conservative_geometry_campaign.py -q
```

Latest result:

```text
64 passed in 2.10s
```

Earlier focused test command:

```powershell
F:\quantule_mapper\.venv\Scripts\python.exe -m pytest tests/test_rhs_term_flux_decomposition.py tests/test_rhs_flux_source_isolation.py tests/test_rk4_integrity_batch.py tests/test_conservative_rk4_stepper_diagnostic.py tests/test_conservative_stepper_contract_audit.py tests/test_conservative_geometry_campaign.py -q
```

Earlier result:

```text
58 passed in 1.80s
```

Protected-file diff check remained empty for:

- `solver/core.py`
- `solver/run.py`
- `worker_cupy.py`
- `aste_hunter.py`
- `validation_pipeline.py`
- `config_utils.py`
- `tools/production_h7_revalidation.py`
- `jax_scout/physics.py`
- `jax_scout/phase_d_c2_transport.py`
- `jax_scout/phase_d_c2_soliton_scout.py`
- `jax_scout/phase_d_c2_2_loss_source.py`

## Diagnostic Chain

1. Conservative C2 stepper contract audit

   Output folder: `stepper_contract_audit`

   Result:
   - Linear conservative evolution preserved ordinary norm to numerical precision.
   - Nonlinear ETDRK4 norm loss was timestep-sensitive.
   - Initial nonlinear RHS norm flux was numerical zero for symmetric controls.
   - ETDRK4 was not suitable as the current invariant-preserving diagnostic stepper at the tested timestep.

2. Diagnostic full-RHS RK4 stepper

   Output folder: `rk4_stepper_diagnostic`

   Result:
   - RK4 was implemented as diagnostic-only.
   - RK4 improved ordinary norm preservation over short horizons.
   - Short RK4 geometry replay preserved node counts through T=4.0 for triangle, ablated triangle, tetrahedron, and triangular prism.
   - This was not a stability result.

3. RK4 integrity batch

   Output folder: `rk4_integrity_diagnostic_20260708_094514`

   Result:
   - DT, step count, physical time, and state/history hashes passed integrity checks.
   - The previous concern that RK4 outputs were identical across dt did not reproduce as a caching/reuse bug.
   - Trajectory nonlinear RHS flux failed after t=0:
     - t=0.25: about `-1.93e-4`
     - t=0.5: about `-3.34e-4`
     - t=0.75: about `-3.89e-4`
     - t=1.0: about `-3.49e-4`
   - N64 replay was correctly blocked.

4. RHS flux source isolation

   Output folder: `rhs_flux_source_isolation`

   Result:
   - Full flux matched nonlinear flux.
   - Linear flux was numerical zero.
   - Physical/spectral flux convention checks passed.
   - Reconstruction checks passed.
   - Final decision: `RHS_NONLINEAR_ALGEBRAIC_FLUX_CONFIRMED`.

5. RHS term-flux decomposition and operator audit

   Output folder: `rhs_term_flux_decomposition_20260708_113818`

   Key reports:
   - `rhs_term_flux_decomposition_final_report.md`
   - `rhs_term_flux_decomposition_results.csv`
   - `rhs_operator_adjoint_audit_results.csv`
   - `symmetry_flux_controls_results.csv`
   - `c2_conservative_contract_implication_review.md`

   Result:
   - Term recombination matched original `N_op`.
     - max abs error: `3.68e-12`
     - relative L2 error: `2.72e-16`
   - Dominant flux term: `geometry_covariant_correction`
   - Dominant flux: about `-3.8858e-4` at t=0.75
   - Local density terms contributed only numerical noise.
   - Flat Laplacian adjoint test passed.
   - Frozen conformal/covariant operator proxy failed adjointness.
   - Final decision: `DISCRETE_OPERATOR_ADJOINT_FAILURE`.

6. Weighted invariant audit

   Output folder: `weighted_invariant_audit_20260708_120726`

   Key reports:
   - `weighted_invariant_audit_report.md`
   - `weighted_invariant_audit_results.csv`
   - `weighted_operator_adjoint_results.csv`

   Result:
   - Ordinary norm max drift over T=1.0: about `2.78e-4`.
   - Best short-trajectory drift reducer: `omega_power_p6_exploratory`, max drift about `1.05e-5`.
   - However, its frozen geometry-operator adjoint mismatch was worse: about `0.0188`.
   - `sqrt_g_weighted_norm = Omega^3 |psi|^2` reduced frozen full flux but also failed adjointness.
   - Recommendation: `NO_WEIGHTED_INVARIANT_FOUND`.

## Current Interpretation

The conservative C2 linear operator behaves as expected:

```text
L_k = -1j * D * k^2
```

The nonlinear density terms appear pointwise phase-like and do not materially explain the observed ordinary norm flux.

The observed trajectory flux localizes to the conformal/covariant geometry correction:

```text
i * D * (lap_cov - lap_flat)
```

The strongest current evidence is that the nonlinear geometry correction is not adjoint-neutral under the tested ordinary or Omega-weighted inner products. The issue is therefore not currently a dt-integrity bug, cache/reuse bug, physical/spectral measurement bug, or local polynomial density-term issue.

## Main Workflow Recommendation

Do not run longer Conservative C2 geometry campaigns yet.

Recommended next action:

Ask Jake/Claude to review the intended C2 contract for the nonlinear geometry correction:

```text
Should Conservative C2 `kinetic_mode='conservative'` conserve total sum(|psi|^2)
for the full nonlinear geometry-corrected RHS, or only for the linear dispersive
substrate / special symmetric states?
```

If the full nonlinear C2 branch is intended to conserve ordinary norm or a specific weighted norm, the conformal/covariant operator implementation needs theory-guided review for the correct discrete adjoint measure.

If the full nonlinear C2 branch is intentionally quasi-conservative, then future reports should label it that way and avoid treating ordinary norm drift as a solver failure.

## Blocked Until Review

- N64 replay.
- T=8000 or longer conservative campaigns.
- Overnight conservative geometry campaigns.
- Any claim of Conservative C2 node stability.

## Safe To Continue

- Read-only report analysis.
- Additional diagnostic-only contract probes.
- Documentation and handover notes.
- Small synthetic algebra/operator probes that do not patch solver physics.
