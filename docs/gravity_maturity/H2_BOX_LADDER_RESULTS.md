---
tags: [result, gravity]
date: 2026-09-17
branch: Branch - Gravity - Index
status: complete
verdict: H2_DRIFT_CONVERGES_TO_A_NONZERO_ASYMPTOTE__70PCT_BOX_30PCT_REAL
---

# H2 — The Long-Time Drift Is Real. Most of What Was Measured Was Not.

Author: Claude (primary), 2026-09-17. Plan item **H2**, closing
`TG_STATE_LOAD_LONG_TIME_DRIFT_UNRESOLVED`.
Harness `jax_scout/gravity_TG_B1S_H2_box_ladder.py`; run
[[../runs/TG_H2_BOX_LADDER]] (WSL/JAX GPU, 12.4 h, 50 periods per row).

Observation-only; the frozen D4 dynamics are imported, not modified.
**No gravity / UFF / IRER claim.**

> [!abstract] Verdict
> `H2_DRIFT_CONVERGES_TO_A_NONZERO_ASYMPTOTE__70PCT_BOX_30PCT_REAL`
>
> The drift is **numerically converged and physically real**, but ~70% of the value the project has
> been quoting was finite-box contamination.

---

## 1. The dt/dx half was already answered, and the plan's prior was wrong

The plan framed H2 as numerical-versus-physical and proposed dt-halving and grid-doubling. The
existing D4 rows already contain both, and at full precision:

| knob varied | change in `Δω∞` |
|---|---|
| **dt halved** | 4 parts in 10¹² |
| **grid N 48→56** (dx 0.208→0.179) | 4 parts in 10⁹ |
| absorber ×1.25 | 7 parts in 10⁷ |
| **box L 10→12** | **factor 2.54** |

RK4 is fourth order, so a time-integration artefact would shrink ~16× when dt is halved. It moves in
the twelfth digit. **The plan's stated prior — "RK4 is not structure-preserving, so there is a real
prior for numerical" — is refuted.** The values also differ (11th and 9th digit), confirming the
flags applied rather than silently no-op'ing.

## 2. The ladder, and two bit-exact controls

| L | N | dx | `Δω∞` | `boundary_flux` | early/late |
|---:|---:|---:|---:|---:|---:|
| 10 | 48 | 0.2083 | **−2.162695983559886e-06** | 2.2623e-07 | 0.3008 |
| 12 | 56 | 0.2143 | **−8.524302969263542e-07** | 6.3290e-08 | 0.00142 |
| 14 | 64 | 0.2188 | −6.893675932988515e-07 | 3.0684e-08 | 0.00080 |
| 16 | 72 | 0.2222 | −6.378083964933548e-07 | 1.5677e-08 | 0.00689 |

> [!important] The first two rows are controls and they reproduce D4 **bit-for-bit**
> L=10/N=48 is D4's `baseline_50P` and L=12/N=56 is D4's `larger_box_50P`. Both reproduce to **all
> 16 digits**, from a separately-written harness that reads the frozen module's own parameter
> defaults by parsing its source. Any difference in rows 3–4 is therefore the box, not the method.

## 3. It does not go to zero

The drift falls **3.39×** across the ladder while the boundary flux falls **14.4×**, and the ratio
`drift/flux` *rises* monotonically 9.6 → 13.5 → 22.5 → 40.7. **The drift does not track the
boundary**, which it would if it were purely a boundary effect.

Fitting both hypotheses (deterministically — see §5):

| model | asymptote | rel-rms |
|---|---:|---:|
| **`a + b·exp(−cL)`** | **6.4535e-07** | **0.88%** |
| `b·L^p` → 0 | 0 by construction | 22.09% |

**The asymptotic model is 25× better.** Residuals against measured values: +2.2e-10, −3.4e-09,
+1.5e-08, −1.2e-08. Dropping the most contaminated row (L=10) and refitting the remaining three
gives **6.14e-07** — a 5% shift, so the asymptote is robust to the point that most influences it.

$$\boxed{\;|\Delta\omega_\infty|(L\to\infty) \approx 6.1\text{–}6.5\times10^{-7}\;\approx\;30\%\text{ of the }L=10\text{ value}\;}$$

**A consistency check that was not fitted for:** the recovered decay rate is **c = 0.9875**, against
S1's mediator inverse-range $\mu_-=1.1424$ — a ratio of **0.86**. The box contamination decays at
essentially the mediator's screening rate, which is what a node seeing its own periodic images
*through the screened mediator* should do.

## 4. What this changes

**The blocker does not dissolve — it resolves the other way.**

- The drift is **converged in dt, in dx, and now in L**. It is a real property of the model, not an
  artefact of the integrator, the grid, or the boundary.
- **But ~70% of the magnitude the project has been quoting is box contamination.** Every drift number
  measured at L=10 — including D4's reference `−2.15e-06` — is roughly 3.4× too large.
- `early_late_relative_difference` collapses from 0.30 at L=10 to ~0.001 at L≥12. At L=10 the drift
  is not even constant-slope; the non-uniformity was the boundary, and in a larger box the drift is
  clean.

> [!warning] Consequence for D4's gate
> D4 failed on the `larger_box` row, whose relative frequency difference (0.6037) exceeded the 0.60
> gate. That gate compares against a reference measured **at L=10** — i.e. against a value now known
> to be ~70% contamination. **The `larger_box` row was closer to the truth than the reference it was
> being judged against.** D4's failure is an artefact of the reference, not of the row, and
> `TG_STATE_LOAD_D4_NUMERICAL_VALIDATION_FAILED` should be re-examined on that basis.

## 5. Method note — a verdict my own harness got wrong

The harness's first verdict rule was *"shrank a lot and has a negative log-log slope ⟹ finite-box
effect"*. It fired, and it was wrong: a 3.4× decrease that is **levelling off at 30%** looks
identical to a power-law slope, and the two are opposite physics — one makes the drift an artefact,
the other makes it real.

Worse, the replacement using `scipy.optimize.curve_fit` was **starting-guess sensitive**: three
parameters against four points, and a plausible-looking `p0` landed in a local minimum giving
`c = 37.6` with a 58% residual, against 0.88% at the true optimum. That would have produced a
confident wrong answer.

The fit is now **deterministic**: for fixed `c` the model is linear in `(a,b)`, so the harness scans
`c` in one dimension and solves `(a,b)` exactly. No starting guess, no local minima.

---

## Issues raised

- Four points against a three-parameter fit is thin. The asymptote is stable (6.45e-07 on four
  points, 6.14e-07 on three), but an L=20 row would settle it and is ~7 h.
- Every drift magnitude in the record measured at L=10 needs the 3.4× factor applied before being
  compared with anything. That affects more documents than this one.
- `early_late_relative_difference` at L=16 (0.0069) is ~9× its L=14 value (0.0008) — non-monotonic.
  Small in absolute terms, but it is the one quantity here that does not behave smoothly, and I have
  not explained it.

## Previous experiments

- [[TG_B1S_D4_FINAL_ANALYSIS]] — the rows that answered the dt/dx half
- [[S1_STATIC_TG_GREENS_FUNCTION]] — the mediator range the decay rate matches

## Associated docs

- [[../IRER_MASTER_HYPOTHESIS_CATALOG]] · [[../INTEGRATED_PLAN_2026-09]]

## Branches

- [[../Branch - Gravity - Index]]
