---
tags: [result, validation, instrument-integrity, revalidation]
date: 2026-10-04
branch: Branch - Validation - Index
status: running
---

# Re-validation after the ETDRK4 fix (e270cdc) — full replay

This is a result note covering Phase B of [[IMPLEMENTATION_PLAN_2026-10]]. Every quantitative result
that ran on the ETDRK4 path before `e270cdc` is replayed with its original harness and original
arguments on the fixed solver.

**The decision rules below were written and committed before the first replay step started.**
They are not edited after the results arrive. If a rule turns out to be badly posed, that is
recorded as an issue and the rule stays as written.

> [!info] Sources
> - Driver: `tools/revalidation/reval_e270cdc.sh` (WSL `~/jax_irer`, GTX 1080, sequential)
> - Harnesses, unchanged except for the additive `--dt` / `--cells` / provenance options from Phase B3:
>   `jax_scout/phase_d_c2_7_rederivation.py`, `jax_scout/phase_d_c1_transport.py`,
>   `jax_scout/feb_gain_ladder_longt.py`, `jax_scout/feb_astar_confirm.py`
> - Originals: `sweep_runs/C27_REDERIVE`, `sweep_runs/c1_longT_confirm.json`,
>   `sweep_runs/FEB_GAIN_LADDER_LONGT_T72000_20260701_175708`, `sweep_runs/FEB_ASTAR_CONFIRM_20260702_003055`
> - Replays: `sweep_runs/*_REVAL_e270cdc`; logs in `sweep_runs/REVAL_e270cdc_*.log`

## Pre-replay gates (B3) — all PASSED, 2026-10-04

| gate | result |
|---|---|
| JAX ETDRK4 order (first execution of the mirror fix) | order 3.94–4.08 ([[STEPPER_ORDER_GATES]]) |
| CuPy ↔ JAX parity, fixed code, `omega0 = 0` (the historical setting) | rel-L2 1.73e-12, `PARITY_WITHIN_TOL` |
| CuPy ↔ JAX parity, fixed code, `omega0 = rho_vac` (complex L, production default) | rel-L2 2.13e-12, `PARITY_WITHIN_TOL` |
| `--dt` smoke test (N=32, ×1.15, 200 steps vs 400 at dt/2) | `er_fin` agrees to 1.4e-6 |

> [!warning] Why the historical parity check could not have caught Bug 1
> `tools/solver_parity_check.py` pins `param_omega0 = 0.0`. That makes L real, which is exactly the case
> where the half-circle `real()` shortcut is valid. So even two independent implementations would have
> agreed there. The check now also passes with complex L. Any future parity evidence should use complex L.

## Pre-registered decision rules
1. **a\* unchanged** if all of the following hold:
   - the late-slope (50% window) zero crossing stays between ×1.15 and ×1.16;
   - ×1.15 at T=144000 has |slope| < 0.001 per 1k steps;
   - seeds 620 and 621 at ×1.15 both have |slope| < 0.001 per 1k;
   - the ×1.15 dt/2 cell agrees with the base-dt ×1.15 ladder cell in sign of slope and to within
     0.0005 per 1k.

   Otherwise the verdict is **`A_STAR_SHIFTED`**, with the new bracket reported.
2. **C2.7 unchanged** if R3 boosts n = 1, 2 give v/2Dk ∈ [0.999, 1.001] and mass ≥ 0.999, and R3's
   verdict is `CLEAN_TRANSPORT_N96_CONFIRMED`. Otherwise **`C2_7_SHIFTED`**.
3. **C1 unchanged** if the verdict string matches the original and, at D_imag = 0, |v_x| < 1e-3
   (the Phase C null). Otherwise **`C1_SHIFTED`**.
4. **C2 norm drift:** re-measured from R0's `mass_ret_T1`. The expectation is that drift shrinks
   relative to the pre-fix runs; no pass/fail threshold is set.

## Results
*Pending — the replay was launched on 2026-10-04 and takes about 18 h.*

## What changed as a result
*Pending.*

## Issues raised

| # | issue | status |
|---|---|---|
| 1 | `C27_REDERIVE` has no `summary.json`, so the results index does not list it | OPEN — tracked by hand here |
| 2 | C1's initial state (`DEFAULT_STATE`) is a saved pre-fix a\* field; it is used only as the starting state | NOTED |
| 3 | `feb_breathing_longt` (×1.0, ×1.05 at T=72000) is not replayed; the a\* rules do not depend on it | NOTED |

## Associated docs
- [[ETDRK4_INTEGRATOR_BUGS_2026-10]] · [[STEPPER_ORDER_GATES]] · [[SOLVER_AND_RUNTIME_CHANGELOG]]
- [[PHASE_C_GAIN_LADDER_RESULTS]] · [[IRER_MASTER_HYPOTHESIS_CATALOG]]

## Branches
- [[Branch - Validation - Index]]
