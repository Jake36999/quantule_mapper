---
tags: [record, validation, instrument-integrity, ci]
date: 2026-10-04
branch: Branch - Validation - Index
status: complete
---

# Stepper order gates (Phase A)

This is the build record for Phase A of [[IMPLEMENTATION_PLAN_2026-10]]. Every time-stepper on an
active research path now has a convergence-order gate. ETDRK4 and TG RK4 also have
manufactured-solution (MMS) gates. All of these run in CI beside the physics identities.

**The headline:** identity tests alone catch **0 of 6** stepper-bug shapes; with the order gates,
**6 of 6** are caught.

> [!info] Sources
> - Tests: `tests/test_stepper_order_jax.py` (new, 10 tests); `tests/test_etdrk4_order.py` (CuPy +
>   NumPy, from `e270cdc`)
> - Mutation probe: `tools/mutation_probe.py` (6 new `stepper` mutations, `--tests` option)
> - CI: `.github/workflows/physics-identities.yml`
> - Calibration script: `docs/instrument_integrity/evidence/phase_a/calibrate.py`
> - Probe outputs (copied from the gitignored `runtime_logs/`):
>   `docs/instrument_integrity/evidence/phase_a/mutation_probe_stepper.json`,
>   `docs/instrument_integrity/evidence/phase_a/mutation_probe_identities_only.json`

## Summary
The ETDRK4 bugs ([[ETDRK4_INTEGRATOR_BUGS_2026-10]]) survived because the project's guard tests
check identities (conservation, nulls, operator algebra), and an inaccurate integrator still
satisfies them. Phase A adds the missing class of test and proves it works.

## Gated steppers — measured orders (WSL `~/jax_irer`, CPU, x64)

| stepper | code | designed | measured | gate |
|---|---|---|---|---|
| ETDRK4 dissipative (feb, geometry on) | `jax_scout/physics.py:step` | 4 | 3.94 / 4.07 | order > 3.7 + MMS |
| ETDRK4 conservative (NLS branch) | same, `kinetic_mode="conservative"` | 4 | 3.94 | order > 3.7 + MMS |
| KG Strang split | `jax_scout/phase_d_c3_wave.py:kg_evolve` | 2 | 2.00 / 2.00 / 2.00 | order > 1.9; energy drift ratio > 3 per dt halving |
| TG-B1S RK4 | `gravity_TG_B1S_state_load_feedback_gpu.py:rk4_step` | 4 | 4.07 / 4.03 | order > 3.7 + MMS |
| TG-B2 RK4 | `gravity_TG_B2_two_node_awell.py:rk4_2n` | 4 | 4.07 / 4.03 | order > 3.7 |
| Gravity-D RK4 | `gravity_D_neutral_probe_gpu.py:rk4_step` | 4 | > 3.7 | order > 3.7 |
| CuPy ETDRK4 (production) | `solver/core.py` | 4 | 4.00 | `tests/test_etdrk4_order.py`, local `.venv` only (no GPU in CI) |

This is also the **first execution of the JAX-mirror ETDRK4 fix**. It converges at order 4.0–4.08 on
the full feb physics (closes issue 3 of [[ETDRK4_INTEGRATOR_BUGS_2026-10]]).

### How the MMS gates work
- An exact smooth ψ\*(t) is chosen. For ETDRK4 it is a single Fourier mode, so dealiasing is exact.
- The stepper's nonlinear function (`physics.n_op`, `b1s.rhs`) is wrapped, via monkeypatch, to add
  S = ∂ₜψ\* − Lψ\* − N(ψ\*) at the correct stage time (0, ½, ½, 1).
- The stepper runs unmodified under `jax.disable_jit()`, and its error against ψ\* must converge
  at order 4. No reference run is involved, so the gate cannot share a bug with a reference.
- **Not done for KG Strang.** The kick has no clean source hook (it multiplies `g(ρ)ψ`), so the
  Strang gate is order plus energy-drift scaling instead. This deviates from the plan and is noted
  here.

## Mutation probe

| run | suite | caught | survived |
|---|---|---|---|
| full | identities + order gates | **16 / 16** | 0 |
| identities only | `tests/test_physics_identities.py` | 10 / 16 | **all 6 stepper mutations** |

The stepper mutations:
- `etdrk4_stage_c_uses_Na` — the October bug 2
- `etdrk4_contour_real_part` — the shape of bug 1
- `kg_strang_kick_asymmetric` — the second half-kick evaluated at the old ψ
- `tg_b1s_rk4_stage3_uses_k1`
- `tg_b2_rk4_stage3_uses_k1`
- `gravity_d_rk4_stage3_uses_k1`

## Not gated (legacy or inactive steppers)
These have their own integrators and are not on an active research path. They are listed so the gap
is visible:
- `gravity_TG_B0_feedback_exchange.py:rk4_step` (NumPy)
- `gravity_TG_B1_feedback_scout_gpu.py:step`
- `gravity_TG_B1S_backreaction_robustness_gpu.py:rk4_lambda`
- `gravity_D_effective_medium_replicate.py:rk4_step` (NumPy)
- `gravity_TS_temporal_kg_audit_gpu.py:kg_step`
- `gravity_G1_clock_calibration_gpu.py:kg_evolve_local`
- `gravity_A1_predictive.py:evolve_coupled`

Harnesses that call `physics.step` (for example `feb_*`, `core_characterize`, `phase_d_c1_*`,
`phase_d_c2_*`) are covered by the ETDRK4 gate.

## What this changes

| status | item |
|---|---|
| **Newly explained** | Why the October bugs survived: the identity suite cannot see accuracy (0/6). |
| **Unchanged** | No solver output changes; this phase adds tests and tooling only. |

## Issues raised

| # | issue | status |
|---|---|---|
| 1 | JAX mirror fix unexecuted | RESOLVED (measured order 4.0–4.08) |
| 2 | Legacy steppers ungated | OPEN — gate one if it is reactivated |
| 3 | No MMS for KG Strang | WONTFIX for now (order + energy-scaling gates cover it) |

## Associated docs
- [[ETDRK4_INTEGRATOR_BUGS_2026-10]] · [[SOLVER_AND_RUNTIME_CHANGELOG]] · [[IMPLEMENTATION_PLAN_2026-10]]
- [[gravity_maturity/H3_MUTATION_PROBE_RESULTS]]

## Branches
- [[Branch - Validation - Index]]
