---
tags: [result, method, instrument]
date: 2026-09-16
branch: Branch - Validation - Index
status: complete
verdict: H3_IDENTITY_TESTS_PINNED_SYMMETRIES_NOT_VALUES__7_OF_10_BUG_SHAPES_SURVIVED__NOW_0
---

# H3 — The Identity Tests Caught 3 of 10 Bug Shapes. Now They Catch 10.

Author: Claude (primary), 2026-09-16. Plan item **H3**.
Probe `tools/mutation_probe.py`; tests `tests/test_physics_identities.py`; raw result
`runtime_logs/mutation_probe.json`.

Method work, not physics. **No gravity / UFF / IRER claim.**

> [!abstract] Verdict
> `H3_IDENTITY_TESTS_PINNED_SYMMETRIES_NOT_VALUES__7_OF_10_BUG_SHAPES_SURVIVED__NOW_0`
>
> The physics-identity CI, built in August specifically to guard against the C2.6 / C2.8b / C3
> failure classes, **missed seven of ten mutations shaped like those failures** — including a
> global flip of the coupling polarity, in the sector whose single largest open problem is a sign.
> Six new tests close all seven.

---

## 1. Why not `mutmut`

The plan named `mutmut`, and `mutmut` is the right tool for *"what fraction of arbitrary edits does
the suite notice?"* That is not the question worth asking here. The question is whether the identity
tests would catch **the bugs this project actually had** — three of them, each costing months, each
found by chasing a contradiction rather than by a test. Blind mutation answers that only by
accident, buried in thousands of irrelevant mutants.

So the ten mutations are shaped like entries in the instrument-integrity ledger:

| class | the original failure | injected as |
|---|---|---|
| **C2.6** | `Ops.geom_fac` / `param_geom_off` left `D_eff = D/151`; five campaigns of "pinning/drag" verdicts retracted | a flag dropped from the `A` expression; the coupling divided by 151 |
| **C2.8b** | a bespoke peak-tracker failed on overlapping cores → elasticity 3.21 and an apparent energy violation that was never in the physics | dynamics left intact, an **observable** miscomputed |
| **C3** | boost-IC bug surfaced through charge drift | mass term dropped from the KG dispersion; propagator detuned |
| **sign** | `a_sign` is a runtime flag and the sector's #1 open problem | polarity flipped, in three different places |

## 2. The result

| | caught | survived |
|---|---:|---:|
| **before** | 3 | **7** |
| **after** | **10** | 0 |

The survivors were not a random scatter. **Every one of them preserved a symmetry while changing the
physics** — which is exactly what the suite was checking.

## 3. The diagnosis: the suite pinned symmetries, never values

Every original test asserted a *relationship*: off-is-off, antisymmetry in `a_sign`, charge
conservation, finiteness. Each is invariant under precisely the mutations that mattered.

The clearest case is `test_force_reverses_sign_with_coupling_polarity`, which asserts

```python
assert well * hill < 0.0
assert abs(well + hill) <= 1e-3 * max(abs(well), abs(hill))
```

Both statements compare the two arms **against each other**. Flipping the coupling globally swaps
which arm is which, so both still hold. The test pins the *antisymmetry* and says nothing about the
*polarity* — and polarity is the open problem.

Two further structural findings, neither visible without the probe:

> [!warning] No test ever set the geometry flag to zero
> `OFF = [1.0, 1.0, 1.0, 0.0]` disables **feedback** and leaves `geom_en = 1`. Nothing in the suite
> exercised `geom_en = 0`, so deleting `geom_en` from the `A` expression changed no result. That is
> the C2.6 shape exactly: a switch that does not switch.

> [!warning] `A` is written twice, and the observer watches its own copy
> `A = exp(a_sign * eps_G * G * ...)` appears in **`rhs_2n`** (the dynamics) and again in
> **`stress_diag`** (the observer). The first round of new tests pinned values through
> `stress_diag` and *still* left `coupling_scaled_by_151` and `a_sign_polarity_flipped` alive,
> because those mutations are in the dynamics. An observer computing its own copy of a quantity
> cannot report that the dynamics changed. Tests now pin both.

## 4. What was added

Six tests, each pinning a value against an **independently derived expression** rather than a second
implementation — the design rule for this file:

| test | pins | kills |
|---|---|---|
| `test_positive_polarity_is_a_well_and_negative_is_a_hill` | `a_sign=+1` gives `A ≤ 1` everywhere, `a_sign=-1` gives `A ≥ 1` | absolute polarity |
| `test_geometry_flag_off_keeps_polarity_out_of_the_dynamics` | `geom_en=0` ⟹ dynamics independent of `a_sign`, and `G` does not evolve | C2.6 shape |
| `test_force_matches_an_independent_linear_response_estimate` | `F_R` against `−c²·a·ε∫(∂ₓG)|∇φ|²` — no exponential anywhere | magnitude, in the observer |
| `test_dynamics_A_coupling_matches_linear_response` | the A-mediated part of `kg_force` against `c²∇·(εG∇φ)` | magnitude **and** polarity, in the dynamics |
| `test_free_kg_force_matches_the_closed_form_operator` | with `a=s=f=0`, force `= c²∇²φ − m²φ` | mass-term sign |
| `test_T_G_coupling_is_reciprocal` | `∂V̇_T/∂G = ∂V̇_G/∂T = −κ`, by finite difference | one-sided κ sign flip |

The linear-response checks are legitimate independent identities rather than re-implementations:
`A_well_min ≈ 0.99993` puts this sector deep in linear response, so `A = 1 + a·ε·G` to five digits,
and the resulting expression contains no exponential. Measured agreement is **0.03%**, the size of
the neglected quadratic term.

`test_T_G_coupling_is_reciprocal` is worth singling out. Both `V̇_T ⊃ −κG` and `V̇_G ⊃ −κT` descend
from the single potential term `κTG`, which is why the coefficients match. Flipping one leaves a
system that still conserves U(1) charge and still stays finite — every pre-existing dynamics test
passes — while no longer deriving from any potential at all. Given that **GAP-4** (the loop is
non-variational) is a live open problem, a test that detects loss of a variational structure is
worth more than its mutation score.

## 5. What this does and does not establish

**Does:** the CI now detects all ten bug shapes, including every sign mutation. The claim in the
record that the identity CI guards the C2.6/C2.8b/C3 classes is true *as of now*; it was mostly
false before.

**Does not:** ten mutations is not a coverage measure. A survivor is evidence of a gap; the absence
of survivors across ten hand-picked mutations is not evidence of completeness. The probe is cheap
(≈50 s) and should grow whenever a new failure class is found — that, not the current score, is
the point of keeping it.

**Not run:** `mutmut` proper. If a broad coverage number is ever wanted, it remains the right tool;
it answers a different question from this one.

---

## Issues raised

- `A` is duplicated between `rhs_2n` and `stress_diag`. Tests now pin both, but the duplication is
  the underlying defect and a shared helper would remove the failure mode rather than detect it.
- The probe's mutation list is hand-maintained. It should be extended whenever the ledger gains an
  entry, and nothing currently enforces that.

## Associated docs

- [[../IRER_MASTER_HYPOTHESIS_CATALOG]] · [[../INTEGRATED_PLAN_2026-09]]
- [[../SYSTEM_PRESSURE_TEST_AND_METHOD_REVIEW]] — threat **T10**, reviewer scarcity

## Branches

- [[../Branch - Validation - Index]]
