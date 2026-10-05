---
tags: [record, validation, instrument-integrity]
date: 2026-10-04
branch: Branch - Validation - Index
status: complete
---

# ETDRK4 integrator bugs — October 2026

This is a bug report and the record of its fix. It covers two defects in the ETDRK4 time-stepper that
made the Phase C / C1 / C2 integrator converge at about **order 0.6 instead of 4**. Both bugs are fixed
in `e270cdc`. Results that depend on them are listed for re-validation; none is retracted yet.

> [!info] Sources
> - Fix commit: `e270cdc` (branch `fix/etdrk4-integrator-order`)
> - Code: `solver/etdrk4_coeffs.py` (new), `solver/core.py`, `solver/kernels.py:78`,
>   `jax_scout/physics.py`, `tools/conservative_geometry_campaign.py`,
>   `tools/conservative_stepper_contract_audit.py`
> - Regression gate: `tests/test_etdrk4_order.py`
> - Evidence scripts and outputs: `docs/instrument_integrity/evidence/etdrk4_2026-10/`

## Summary
- **Bug 1:** the Kassam–Trefethen coefficients took `real()` of an upper-half-circle mean. That
  shortcut is only valid when the linear operator L is real, and L is complex everywhere in this
  project.
- **Bug 2:** stage c used `N(a)` where Cox–Matthews requires `N(u_n)`.

Together they cut the measured order from 4 to about 0.6. At the production dt=0.005, ψ was wrong by
8.5% after T=2 (N=48, feb parameters, a\*); after the fix the error is 0.27%.

Neither bug could be seen by the existing checks:
- The identity tests check conservation laws, not accuracy.
- CuPy↔JAX parity compared two copies of the same code.
- [[BASELINE_AUDIT_NUMERICAL]] §1 had marked the integrator `CONFIRMED`.

**Not affected:** C3 KG and the whole TG/gravity stack, which use the split-step in
`jax_scout/phase_d_c3_wave.py`.

## Detail

### Bug 1 — contour coefficients (`solver/core.py`, `jax_scout/physics.py`, campaign tool)
The original code evaluated the φ-functions Q, f1, f2, f3 on 64 points over the **upper half** of a
unit circle around each `w = L·dt`, then kept `real()` of the mean. Kassam & Trefethen use that trick
for real L, where conjugate symmetry makes the half-circle real part equal the full-circle mean.

Here `L_k = −D k² − η + iω0`, or `−i D k²` on the conservative branch, so the trick silently discarded
the imaginary part of every coefficient.

Measured coefficient errors (`t1_etdrk4_coeffs.py`):

| L | max rel. error, Q / f1 / f2 / f3 |
|---|---|
| Phase C baseline | 1.5e-3 / 6.9e-3 / 3.2e-3 / 4.8e-4 |
| NLS-like `−i k²` | 0.24 / **0.95** / 0.49 / 0.056 |
| real L (control) | ~1e-14 |

The fix is `solver/etdrk4_coeffs.py`: 128 points over the full circle, with complex results kept. For
real L it reproduces the old coefficients to ~1e-14.

### Bug 2 — stage c (`solver/kernels.py:compute_kt_stage_c`, JAX `step`, two tools)
The code used `c = E2·a + Q·(2·N_b − N_a)`. The Cox–Matthews / Kassam–Trefethen scheme is
`c = E2·a + Q·(2·N_b − N_n)`, where `N_n` is the nonlinearity at the start of the step.

This alone makes the scheme second-order. It was found because, after Bug 1 was fixed, a smooth
non-stiff test still converged at exactly order 2.00 (`isolate.py`, mode A).

### Measured effect on the real solver
From `convergence_gpu.py` (CuPy, N=48, feb parameters with `param_a = 0.5522`, T=2, reference
dt = T/6400), output in `convergence_N48.txt`:

| dt | before (both bugs) | after |
|---|---|---|
| 0.02 | 1.9e-1 | 5.4e-2 |
| 0.01 | 1.3e-1 | 1.7e-2 |
| **0.005** | **8.5e-2** | **2.7e-3** |
| 0.0025 | 5.5e-2 | 2.5e-4 |

The observed order goes from about 0.6 to about 3.4 and is still rising as dt shrinks. A smooth
cubic test gives exactly 4.00.

The A-field update (`update_field_of_affect`) is first-order symplectic Euler, but it is a one-way
observer: `A_real` never feeds back into N(ψ). It does not limit ψ's accuracy.

## What this changes

| status | item |
|---|---|
| **Unchanged** | C3 KG results, TG-A/B1S/B2, H2 box ladder, S1–S3 — none of them use ETDRK4. |
| **Pending re-validation** | a\* location and bracket; C2.7 `v = 2Dk`; the C2 "quasi-conservative" norm drift; C1 headline numbers. See [[REVALIDATION_E270CDC_RESULTS]]. |
| **Newly explained** | Codex's C2 note that "nonlinear ETDRK4 norm loss was timestep-sensitive" fits these bugs. |
| **Downgraded** | CuPy↔JAX parity is not evidence of correctness; it shows only that the two copies match. |

## Recommendations
1. Give every stepper an order gate and a manufactured-solution gate ([[IMPLEMENTATION_PLAN_2026-10]] Phase A).
2. Count parity only when the two implementations share no code.
3. Flag affected runs automatically when a component is fixed (Phase B).

## Issues raised

| # | issue | status |
|---|---|---|
| 1 | Coefficients invalid for complex L | RESOLVED (`e270cdc`) |
| 2 | Stage c uses N_a | RESOLVED (`e270cdc`) |
| 3 | JAX mirror edit not yet executed | OPEN → Phase B3 |
| 4 | Pre-fix ETDRK4 results not re-validated | OPEN → Phase B4 |

## Associated docs
- [[SOLVER_AND_RUNTIME_CHANGELOG]] · [[SEARCH_STACK_AUDIT_2026-10]] · [[BASELINE_AUDIT_NUMERICAL]]
- [[PROCESS_PLAN_2026-10]] · [[IMPLEMENTATION_PLAN_2026-10]] · [[IRER_MASTER_HYPOTHESIS_CATALOG]]

## Branches
- [[Branch - Validation - Index]]
