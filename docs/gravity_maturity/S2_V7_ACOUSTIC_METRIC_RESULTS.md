---
tags: [result, gravity, theory]
date: 2026-09-17
branch: Branch - Gravity - Index
status: complete
verdict: V7_ACOUSTIC_MAPPING_EXACT_FOR_KINETIC_SECTOR__RESPONSE_LAW_DERIVED__SOURCE_SIGN_STILL_NOT_DERIVED
---

# S2 / V7 — The Acoustic-Metric Mapping Is Exact. The Sign Still Is Not.

Author: Claude (primary), 2026-09-17. Plan item **S2**, worked alongside **S1** as the plan directs.
Analysis `tools/v7_acoustic_metric.py` (symbolic via sympy + a numerical check against the coded force).

Observation-only; nothing simulated, nothing changed. **No gravity / UFF / IRER claim.**

> [!abstract] Verdict
> `V7_ACOUSTIC_MAPPING_EXACT_FOR_KINETIC_SECTOR__RESPONSE_LAW_DERIVED__SOURCE_SIGN_STILL_NOT_DERIVED`
>
> Three of the four things V7 was hoped to deliver, and a clean negative on the fourth.

---

## 1. The mapping is exact, and the metric is *determined* rather than fitted

The coded operator (`gravity_TG_B2_two_node_awell.rhs_2n`) is

$$\partial_t^2\phi = \nabla\cdot\!\left(c^2A\,\nabla\phi\right) - m^2\phi + U'(\rho)\phi$$

so $f^{\mu\nu} = \mathrm{diag}(-1,\;c^2A,\;c^2A,\;c^2A)$. The analogue-gravity identification
(Unruh 1981; Visser 1998) is $f^{\mu\nu} = \sqrt{-g}\,g^{\mu\nu}$. In 3+1 dimensions taking
determinants gives $\det f = g$ **with no freedom left** — this is why the result is a derivation and
not a fit:

$$\det f = -c^6A^3 \;\Rightarrow\; \sqrt{-g} = c^3A^{3/2}$$

$$\boxed{\;g_{\mu\nu} = \mathrm{diag}\!\left(-c^3A^{3/2},\; c\sqrt{A},\; c\sqrt{A},\; c\sqrt{A}\right)}$$

Three independent checks, all in closed form:

| check | result |
|---|---|
| reconstructs $f^{\mu\nu}$ exactly | **True** |
| determinant self-consistent | **True** |
| null condition gives local speed $c\sqrt{A}$, matching the operator | **True** |

**The plan listed "is the mapping exact?" as unverified. It is exact.**

## 2. The response law is derived, not postulated — the structure transfer is real

For the static metric, $N = \sqrt{-g_{tt}} = c^{3/2}A^{3/4}$, and a slow particle obeys

$$a = -\nabla\ln N = -\tfrac{3}{4}\,\frac{\nabla A}{A}$$

**Acceleration points toward smaller $A$.** The coded body force says the same thing —
$dP/dt = -c^2\!\int\!\nabla A\,|\nabla\phi|^2dV$, i.e. "a node is pushed toward smaller $A$".

Checked numerically on an explicit two-node configuration:

| `a_sign` | A at the node | coded $F_\text{body}$ | geodesic prediction | |
|---|---:|---:|---:|---|
| +1 | 0.941765 | −1.8643e-04 | −8.6904e-05 | **same direction** |
| −1 | 1.061837 | +1.8664e-04 | +8.6918e-05 | **same direction** |

> [!important] This is the clean-win pattern the research asked for
> The Gravity-D force law has been carried as a **postulate** since it was introduced. It is not one:
> it is what a geodesic of the reconstructed metric does. The project now has a genuine
> **structure transfer** — the thing the research's Request-1 pattern identifies as how comparable
> programmes made first contact, and which the project previously had *zero* candidates for.
>
> The magnitudes differ by ~2× because the two weight the probe differently ($|\nabla\phi|^2$ against
> $\phi^2$); the direction, which is the claim, is what agrees.

## 3. The sign is still not derived — and now we know exactly where it lives

The metric fixes how a packet **responds** to $A$. It says nothing about which sign of $A$ a
state-load **sources**. That is the `a_sign` flag, and it sits in the T→G chain:

```
Vdot_T  contains  + alpha_T * S_state     a positive load drives T UP
Vdot_G  contains  - kappa   * T           positive T drives G DOWN
```

so a load gives $G<0$, and `a_sign=+1` then gives $A<1$ — a well, hence attraction. (Measured: with
`a_sign=+1`, $A_\text{max}\le 1$ everywhere.)

In metric language `a_sign=+1` is the statement *"a state-load lowers the effective lapse"*, which is
the standard gravitational sign. **That is a restatement, not a derivation.** Choosing the flag
because it reproduces attraction is fitting the phenomenology and carries no evidential weight.

> [!danger] GAP-4, sharpened into something actionable
> The $\phi$ sector **is** variational — it follows from
> $\mathcal{L} = |\pi|^2 - c^2A|\nabla\phi|^2 - m^2\rho + U(\rho)$, which is *precisely why the
> acoustic metric exists for it at all*. The T,G back-coupling is **not** obtained by varying that
> Lagrangian.
>
> So the sign problem lives **exactly in the non-variational part of the loop**. Any derivation of
> `a_sign` must come from making the T–G coupling variational. The wave operator has now given
> everything it has, and **S3 (`cadabra2`, GAP-4 with a Q-ball ansatz) is the only remaining route** —
> it is no longer the fallback the plan listed it as.

## 4. A bonus, and it is a rejection: D3 is answered negatively

The mass term is where exactness stops:

$$\Box_g\phi = m_\text{eff}^2\phi, \qquad m_\text{eff}^2 = \frac{m^2 - U'(\rho)}{c^3A^{3/2}}$$

The matter field acquires a **position-dependent effective mass**. It is tempting to call that a
chameleon. It is not, and the research request's global disqualifier applies — *resemblance is a
rejection*:

- **Not chameleon.** A chameleon varies the **mediator's** mass with ambient density. Here the
  mediator is $G$, whose mass is $\omega_G$ — a **constant** of the model
  (`gravity_TG_B2_two_node_awell.py:74`). What varies is the **matter** field's mass. The assignment
  is the other way round, so the mechanism does not transfer.
- **Not a conformal/scalar-tensor mass variation either.** $-g_{tt}/g_{xx} = c^2A$ depends on $A$, so
  the metric is **not conformally flat**. Time and space carry different powers of $A$ ($3/2$ against
  $1/2$) — the acoustic signature proper.
- **Not symmetron or Vainshtein**: no density-dependent coupling, no derivative self-interaction.

**D3's honest answer: none of the three.** This agrees with the plan's own suspicion that $\Omega$ is
a *kinetic* coefficient matching none of them cleanly — now established rather than suspected, and at
zero compute cost.

## 5. What changed

**Promoted:**
- The acoustic mapping from "solid reading" to **exact, verified**.
- The Gravity-D force law from postulate to **derived consequence**.
- **S3** from "if S1/S2 are inconclusive" to **the only remaining route to the sign**.

**Demoted:**
- The hope that V7 would deliver the sign. It does not, and the reason is structural rather than
  a matter of effort.

**Closed at no cost:** D3, negatively.

> [!warning] What this is still not
> An exact acoustic metric is a statement about **this model's own equations**, not about nature. It
> makes the model legible and its force law derived; it is not evidence for IRER. The discriminating
> quantity remains unfound, and **D1** (the critical exponent) is still the primary candidate.

---

## Issues raised

- The ~2× magnitude gap between the geodesic estimate and the coded force is a *weighting* difference
  ($|\nabla\phi|^2$ vs $\phi^2$) and has not been reconciled analytically. Worth doing: a proper
  stress-tensor derivation of the force from the metric would give the exact coefficient and would
  be a second, sharper check.
- If the T–G coupling is made variational (S3) the resulting $A$ may differ from `exp(a_sign·ε_G·G)`.
  That would change the metric, so §1 would need re-deriving rather than assuming.

## Previous experiments

- [[TG_P2_MIDPLANE_STRESS_FLUX_RESULTS]] — the force estimator this now reinterprets
- [[TG_P1A_ENERGY_OBSERVABLE]]

## Associated docs

- [[../INTEGRATED_PLAN_2026-09]] · [[../IRER_MASTER_HYPOTHESIS_CATALOG]]
- [[../FUTURE_WORK_AND_EXTERNAL_VALIDATION_PLAN]] — where V7 was first named

## Branches

- [[../Branch - Gravity - Index]]
