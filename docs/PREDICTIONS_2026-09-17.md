---
tags: [prediction, main]
date: 2026-09-17
branch: Main branch
status: open
author: Claude
---

# Predictions — 2026-09-17

Stated **in advance**, from the equations in [[WHAT_THE_SIMULATION_ACTUALLY_DOES]] plus what has been
measured. Each is falsifiable, most are cheap, and each carries a **confidence** and an explicit
**what would falsify it**.

Written to be scored. The point is not to be right; it is to be *checkable*, and to find out whether
deriving-before-measuring actually pays. I have marked the two I most expect to be wrong.

> [!important] Scoring rule, fixed now so it can't drift
> A prediction counts as **WRONG** if the stated number falls outside the stated tolerance, or the
> stated direction reverses. "Roughly right for a different reason" counts as wrong. Fill the
> **Outcome** column when each is run; do not edit the prediction.

---

## The headline one

### P1 — The variational force scales as `ε_G²`, the coded force as `ε_G`

The sharpest available discriminator between the model as coded and the model as
[[gravity_maturity/S3_VARIATIONAL_SIGN_DERIVATION|S3]] would have it.

- **Coded:** `G` comes from `S_state`, which has no `ε_G` in it. Then `A−1 ≈ a·ε_G·G`. **`F ∝ ε_G¹`.**
- **Variational:** the source for `G` is itself `∝ ε_G`, so `G* ∝ ε_G`, then `A−1 ∝ ε_G²`. **`F ∝ ε_G²`.**

| `ε_G` scaled by | coded force | variational force |
|---|---|---|
| ×0.5 | ×0.50 | ×0.25 |
| ×2 | ×2.0 | ×4.0 |
| ×4 | ×4.0 | ×16.0 |

**Prediction:** run the *existing* harness at `ε_G` ∈ {0.03, 0.06, 0.12, 0.24} and fit `F ∝ ε_G^n`.
**n = 1.00 ± 0.05.**

**Why it matters:** it is a clean test of *which model the code is*, and if the variational term is
ever added, the same measurement becomes n → 2 and confirms the change took effect. **Confidence:
very high (>95%)** — this is close to arithmetic.

**Falsified by:** n outside 0.95–1.05 in the current code. That would mean `ε_G` is entering
somewhere I have not accounted for.

| Outcome | |
|---|---|
| measured n | |
| verdict | |

---

## The one I think is most interesting

### P2 — The force's falloff length tracks the **node size**, not the mediator range

The mediator's ranges are **0.875 and 0.484**, fixed by the couplings and independent of `w`. But
nodes are ~3 wide, so they sit *inside* each other's mediator field, and the measured falloff should
be set by how fast the **source** dies away, not the kernel.

A Q-ball's tail goes as $e^{-\kappa_q r}$ with $\kappa_q=\sqrt{m^2-\omega^2}$, so the energy-like
source falls at $2\kappa_q$:

| `w` | κ_q | **predicted force falloff length 1/(2κ_q)** | mediator range |
|---:|---:|---:|---:|
| 0.945 | 0.3271 | **1.53** | 0.875 |
| 0.955 | 0.2966 | **1.69** | 0.875 |
| 0.964 | 0.2659 | **1.88** | 0.875 |
| 0.972 | 0.2350 | **2.13** | 0.875 |
| 0.980 | 0.1990 | **2.51** | 0.875 |

**Prediction:** measure `exp_lambda` for the two-node force at each `w`. It will **increase
monotonically from ~1.5 to ~2.5**, tracking `1/(2κ_q)` within **±25%**, and will *not* sit near 0.875
at any `w`.

**Partial support already in the record:** the far-field run measured `exp_lambda → ~1.6` at
`w = 0.964`. Predicted source scale 1.88; mediator scale 0.875. **1.6 is far closer to the node scale.**
That was measured before this reasoning existed.

**Why it matters:** if true, the "screened mediation" language is misleading — what is being measured
is mostly the *shape of the nodes*, not a property of the mediator. It would also mean the falloff is
tunable by `w` alone, which is a strong internal handle.

**Confidence: high (~80%).** **Falsified by:** `exp_lambda` flat in `w`, or clustering near 0.875.

| Outcome | |
|---|---|
| measured λ(w) | |
| verdict | |

---

## The rest

### P3 — Freezing `e_ref`/`q_ref` raises the force's mass exponent from −0.07 to ≈ +0.29

Predicted from linearity in [[gravity_maturity/P1B_FROZEN_REFERENCE_REANALYSIS|the P1-b reanalysis]]:
`+0.293`, tolerance **±0.08**. The "force is flat in mass" reading is mostly the normaliser.

**Confidence: high (~85%)**, since it is a rescaling of measured rows. **Falsified by:** an exponent
still below +0.15 with frozen references.

> **Secondary, and I am less sure:** it will **still not be +1.0**, so freezing the reference will
> *not* rescue a gravity-like mass scaling. Confidence ~75%.

### P4 — The saturation cliff will not survive a frozen reference

If the cliff is the peak normalisation saturating by construction, then with frozen `e_ref` the
load→geometry response becomes **linear with no threshold**, and the cliff disappears.

**Confidence: moderate (~65%).** This is the one I would most like to be wrong about, because a real
threshold is worth far more than a normalisation artefact. Evidence for it: the T/G map is exactly
linear and cannot hold a threshold (S1); the mass axis shows clean power laws throughout (P1-b); and
the φ-sector threshold does not survive N48→N96.

**Falsified by:** a threshold that stays put with frozen references — which would mean it is in the
φ sector after all and **D1 becomes worth running**.

### P5 — D1's critical exponent will not converge ⚠️ *(expect I may be wrong)*

Following P4: if the cliff is a normalisation artefact there is no critical point, so a fitted
exponent will drift with box size and resolution instead of converging, the way a real critical
exponent does.

**Confidence: moderate (~60%)** — and flagged because it is a prediction of *absence*, which is the
easiest kind to be wrong about and the least informative when right. **Falsified by:** an exponent
stable to ±10% across two box sizes and two resolutions. That would be a genuine find.

### P6 — H2's residual drift is dissipative, and scales with `γ_T`, `γ_G`

The drift converges to `6.45e-07` rather than zero. The natural culprit is the damping in the
mediator sector (`γ_T = 0.08`, `γ_G = 0.06`) plus the absorber: the node leaks energy into fields
that dissipate it, and a dissipative coupling shifts frequency.

**Prediction:** set `γ_T = γ_G = 0` and `absorb_strength = 0`, re-run the L=16 row. The asymptotic
drift falls by **more than 5×**.

**Confidence: moderate (~60%).** I have *not* verified this mechanism — it is the obvious candidate,
not a derived result, and I am flagging that rather than dressing it up. **Falsified by:** the drift
surviving undamped at more than half its present value, which would point at the `κ` coupling or the
node itself.

### P7 — A single-Yukawa fit to the force curve fails systematically at small separation

From the two-scale kernel (S1). The shorter range (0.484) matters only where nodes are close, so a
one-scale fit should show **structured** residuals — same sign across the small-separation points —
rather than scatter.

**Confidence: moderate (~55%)**, lowered by P2: if the falloff is source-dominated, the mediator's
two scales may be buried. **These two predictions partly conflict, deliberately** — P2 says the
node's shape dominates, P7 says the kernel's structure should still be visible. If P2 is right and P7
fails, that is coherent, and the pair is more informative than either alone.

### P8 — Adding the variational term flips the `a_sign = −1` arm from repulsion to attraction

With the variational term present, the coded channel supplies only ~26–34% of `G`, and it is the only
part carrying a free sign. So the `a_sign = −1` arm should stop repelling.

**Prediction:** with the term added, `F_R(a_sign = −1)` changes sign relative to the present code and
ends within **a factor of 3** of `F_R(a_sign = +1)`.

**Confidence: moderate-high (~70%).** **Falsified by:** the `−1` arm still repelling, which would
mean the coded channel dominates more than the static estimate suggests.

---

## Scorecard

| # | prediction | confidence | outcome |
|---|---|---|---|
| **P1** | `F ∝ ε_G¹` in current code, n = 1.00 ± 0.05 | >95% | |
| **P2** | falloff length tracks `1/(2κ_q)`: 1.5 → 2.5 across `w`, ±25% | ~80% | |
| **P3** | frozen-ref mass exponent ≈ +0.29 ± 0.08 | ~85% | |
| **P3b** | …and still not +1.0 | ~75% | |
| **P4** | cliff disappears with frozen references | ~65% | |
| **P5** | D1 exponent fails to converge ⚠️ | ~60% | |
| **P6** | undamped mediator cuts the residual drift >5× | ~60% | |
| **P7** | single-Yukawa fit shows structured residuals | ~55% | |
| **P8** | variational term flips the `a_sign = −1` arm | ~70% | |

**Aggregate:** across nine predictions at these confidences, roughly **6–7 should land**. If I score
9/9 the confidences were dishonest; if I score below 4 the derivation-first approach is not earning
its keep and should be argued for differently.

> [!warning] What none of these are
> Every prediction here is about **this model's behaviour**, not about nature. They test whether
> deriving before measuring produces reliable foresight *within the model*. None of them is a
> discriminating prediction, and the project still has zero of those. That gap is unchanged.

---

## Associated docs

- [[WHAT_THE_SIMULATION_ACTUALLY_DOES]] — the maths these follow from
- [[SESSION_SYNTHESIS_2026-09-17]] · [[INTEGRATED_PLAN_2026-09]]

## Branches

- [[Main branch]]
