---
tags: [explainer, main]
date: 2026-09-17
branch: Main branch
status: complete
audience: Jake
---

# What the Simulation Actually Does

Every equation the TG sector runs, what each piece is for, and a plain-language picture of what is
happening — written so the physics can be checked against the theory it is supposed to be exploring.

**This is a description of the code, not an argument for it.** Where the implementation and the
intent look like they have parted company, §7 says so.

---

## 1. The cast — six fields, three jobs

The simulation evolves six arrays on a periodic 3-D grid.

| symbol | what it is | job |
|---|---|---|
| `φ` | complex field | **the stuff.** Lumps of it are "nodes" — the particle-analogues |
| `π` | `∂ₜφ` | φ's velocity, carried separately so the solver is first-order in time |
| `T` | real field | **temporal field.** Driven by how much φ is present |
| `V_T` | `∂ₜT` | T's velocity |
| `G` | real field | **geometric field.** Driven by T. Sets the local stiffness |
| `V_G` | `∂ₜG` | G's velocity |

The whole model is a chain:

```
   φ  ──[S_state]──►  T  ──[κ]──►  G  ──[A = exp(a·ε·G)]──►  back into φ's wave speed
   ▲                                                                    │
   └────────────────────────────────────────────────────────────────────┘
```

**The analogy to hold onto:** φ is ripples on a pond. `A` is how *stiff* the pond is at each point —
and stiffness sets wave speed. The chain says: *where there is stuff, the pond changes stiffness, and
that changed stiffness bends the stuff's own ripples.* That loop is the whole theory.

---

## 2. The φ equation — waves in a medium of varying stiffness

$$\partial_t^2\phi \;=\; \underbrace{c^2\,\nabla\cdot\!\big(A\,\nabla\phi\big)}_{\text{wave term}}
\;-\; \underbrace{m^2\phi}_{\text{mass}}
\;+\; \underbrace{\big(a\rho + s\rho^2 + f\rho^3\big)\phi}_{\text{self-interaction}},
\qquad \rho = |\phi|^2$$

**Wave term.** Ordinary wave equation, except the wave speed is not constant: it is $c\sqrt{A}$,
varying from place to place. Set `A = 1` everywhere and this is the textbook Klein–Gordon equation.

> **Analogy.** Light in glass. A region of higher refractive index is a region where light goes
> slower — and light *bends toward* it. `A < 1` is exactly "denser glass". Everything about the
> "gravity" in this model is that one idea.

**Mass term.** Makes waves of frequency $\omega$ require $\omega^2 \ge m^2$ — below that, waves can't
propagate and instead decay exponentially. This is why lumps stay lumpy instead of dispersing: a lump
oscillating at $\omega < m$ is *forbidden* from radiating away, so it sits there.

**Self-interaction.** A cubic/quintic/septic nonlinearity. Without it a lump would spread out.
With it, spreading costs energy and the lump settles at a preferred size — a **Q-ball**.

> **Analogy.** A soap bubble. Surface tension wants it small, internal pressure wants it big, and it
> settles where they balance. Here the mass term wants the lump compact and the gradient term wants
> it spread; the nonlinearity sets where they balance.

**What a "node" is, precisely:** $\phi = e^{-i\omega t}\,f(r)$ — a lump whose *shape* is frozen but
whose *phase* spins. Because the phase spins, the lump carries a conserved charge (like a conserved
particle number), and that conservation is what keeps it alive. **Derrick's theorem** says a purely
static lump in 3-D cannot be stable; the spinning phase is the loophole.

---

## 3. The mediator — T and G, and why there are two

$$\partial_t^2T = c_T^2\nabla^2T - \omega_T^2T - \gamma_T\partial_tT + \alpha_T S - \kappa G$$
$$\partial_t^2G = c_G^2\nabla^2G - \omega_G^2G - \gamma_G\partial_tG - \kappa T$$

Each is a **damped, massive wave field**: $\omega_T$, $\omega_G$ give them mass (so they are
short-range), $\gamma_T$, $\gamma_G$ damp them, and $-\kappa T G$ couples them to each other.

> **Analogy.** Two pendulums joined by a spring. Push the first (that's `S`, the stuff). It swings;
> the spring passes the motion to the second; the second is what you actually read out (`G`).
>
> Two coupled pendulums have **two normal modes** — a fast one and a slow one. That is not decoration:
> it is exactly why [[gravity_maturity/S1_STATIC_TG_GREENS_FUNCTION|S1]] found the mediator is a
> **difference of two Yukawas** rather than one.

Solving the static case exactly:

$$\hat G(k) = \frac{-\alpha_T\,\kappa\,\hat S(k)}{\big(c_T^2k^2+\omega_T^2\big)\big(c_G^2k^2+\omega_G^2\big) - \kappa^2}
\qquad\Longrightarrow\qquad G(r)\sim\frac{e^{-\mu_- r}-e^{-\mu_+ r}}{r}$$

with **ranges 0.875 and 0.484**, both fixed by the couplings — no freedom.

**Two consequences worth holding:**
1. The mediator is **short-range**. It dies off within about one unit of distance.
2. $\alpha_T, \kappa > 0$ and the denominator never changes sign, so **a positive source always makes
   `G` negative**. That part of the sign chain is derived, not chosen.

---

## 4. The source — and this is where to look hardest for drift

$$S_{\text{state}} = \frac{1}{S_0}\left[\frac{1}{2}\frac{\max(e,0)}{e_{\text{ref}}} + \frac{1}{2}\frac{|{\rm Im}(\phi^*\pi)|}{q_{\text{ref}}}\right]$$

where $e$ is the energy density and the second term is charge density. Half energy, half charge.

**But `e_ref` and `q_ref` are the peak values of *that configuration's own initial state*.**

> **Analogy, and it matters.** Imagine measuring how heavy people are, but weighing everyone *in
> units of their own body weight*. Everyone comes out at 1.0. A heavier person does not register as
> heavier.
>
> That is what the peak normalisation does. A node with 2.74× the mass produces a source only
> **1.91×** larger, and a geometric response only **1.47×** larger. About **60% of the load's mass
> dependence is removed before it reaches the geometry**
> ([[gravity_maturity/P1B_FROZEN_REFERENCE_REANALYSIS|measured here]]).

This is the single biggest candidate for conceptual drift, and §7 returns to it.

---

## 5. The feedback, and the force

$$A = \exp\big(a_{\text{sign}}\cdot\varepsilon_G\cdot G\big)$$

With $\varepsilon_G = 0.06$ and $|G| \sim 10^{-3}$, this is $A = 1 \pm 7\times10^{-5}$. **The
exponential is irrelevant** — the model lives entirely in the linear regime $A \approx 1 + a\varepsilon G$.

The force actually measured:

$$F_R = -c^2\!\!\int_{x>0}\!(\partial_x A)\,|\nabla\phi|^2\,dV$$

Read it as: **stuff is pushed toward smaller `A`** — toward the slow patch.

### Two things that were discovered about this, both recent

**(a) It is not a postulate.** [[gravity_maturity/S2_V7_ACOUSTIC_METRIC_RESULTS|S2]] showed the wave
operator *is* an effective curved spacetime:

$$g_{\mu\nu} = \mathrm{diag}\big(-c^3A^{3/2},\;c\sqrt{A},\;c\sqrt{A},\;c\sqrt{A}\big)$$

and a slow particle in that geometry obeys $a = -\tfrac34\nabla A/A$ — motion toward smaller `A`.
**The force law is a geodesic.** It was derived from the equations, not added to them.

> **Analogy.** You don't need a "force" to explain why light bends in a lens. The lens changes how
> long each path takes, and light takes the quickest one. Same here: the node isn't pushed, the
> geometry it moves through is tilted.

**(b) It only couples to 4.5% of a node's energy.** $F_R$ weights $|\nabla\phi|^2$ — the *gradient*
part. The kinetic, mass and potential parts (95.5%) are invisible to it
([[gravity_maturity/TG_P1A_ENERGY_OBSERVABLE|P1-a]]). So this is not a force on mass-energy. It is a
force on *how sharply the field varies*.

---

## 6. The sign problem, and what dissolved it

`a_sign = ±1` was a **runtime flag**. `+1` makes a node dig a *dent* in the medium (attraction);
`−1` makes it raise a *bump* (repulsion). Both ran, both converged. The project's largest open
problem was that the model did not decide.

[[gravity_maturity/S3_VARIATIONAL_SIGN_DERIVATION|S3]] found the reason and the fix.

Real physics comes from an **action** — a single energy bookkeeping from which every equation is
derived by asking "what arrangement costs least?" The Lagrangian here contains $-c^2A(G)|\nabla\phi|^2$.
Vary it with respect to $\phi$ and you get the coded wave equation. **Vary the same term with respect
to `G` and you get a source for `G` that the code does not have.**

Restore it and the sign cancels, because `a_sign` now appears **twice** — once deciding how φ digs
into G, once deciding how G changes A — so it enters squared.

> **The physical version, which is the one to remember.** A node's energy contains
> $c^2A|\nabla\phi|^2$. To lower its energy, the node wants `A` **small** exactly where its own
> gradients are **large**. So *a node digs its own dent, because sitting in a dent is cheaper than
> sitting on a bump.* It cannot choose to build a bump — that would cost energy.
>
> And two nodes that each dig a dent sit in each other's dents, which is cheaper still when they are
> close. **That is the attraction.** It is the same reason any scalar-mediated force between like
> sources is attractive.

The freedom existed only because the code routes φ→G through a **separate channel** (`S_state`) that
an action does not authorise — and a separate channel can carry its own independent sign.

---

## 7. Where the implementation may have drifted from the theory

Flagged for your judgement — these are readings of the code, not verdicts on the theory.

| # | what the code does | what to check against the theory |
|---|---|---|
| **1** | **The source is normalised by its own peak** (§4), so a heavier node barely sources more | If the theory says informational load *accumulates* with the amount of structure, the code contradicts it. "Load" here is a **shape** measure, near-blind to quantity. |
| **2** | The mediator's range (0.875) is **shorter than the nodes are wide** (~3) | Gravity is long-range; this is the opposite regime. The nodes mostly sit **inside** each other's mediator field, so what is measured is an overlap effect, not a field-at-a-distance effect. |
| **3** | The force couples to **gradient energy only** — 4.5% of a node | If the theory intends coupling to total energy or to mass, the implemented observable is not it. |
| **4** | `A` deviates from 1 by **7×10⁻⁵**, so the exponential never matters | If the theory's content is in the *nonlinear* response of geometry to load, none of it has been tested. Everything measured is linear response. |
| **5** | The loop is **non-variational** — no action generates it | This is what made the sign free. If the theory intends a genuinely non-variational (dissipative, information-driven) loop, that is a deliberate departure from field theory and should be stated as one. |
| **6** | `S_state` is half **energy** and half **charge** — both ordinary mechanical quantities | Nothing in it is informational in any sense a physicist would recognise. It is an observable built from the field, not a new kind of quantity. |
| **7** | `T` is called "temporal" but is an ordinary damped massive scalar | Nothing in its equation involves time differently from any other field. The name carries theory content the maths does not. |

**Item 6 and 7 together are the sharpest question:** the theory is about *information* and *temporal
resolution*, and the implementation contains neither — it contains energy, charge, and two ordinary
scalar fields. That may be a legitimate modelling reduction, but it should be a *decision*, not an
accident.

---

## 8. What was actually established, stripped of interpretation

- The dynamics are **verified**: parity between two independent backends to 1.7e-12; the momentum
  ledger closes at exactly second order; charge conserved to ~1e-13.
- The force estimator is **confirmed** against a fully independent one to 0.34%.
- The wave operator **is** an acoustic metric, exactly, and the force law is its geodesic.
- The mediator's falloff is a **derived** two-scale screened form, with no free parameters.
- The long-time drift is **real** and converged — but ~70% of every magnitude quoted for it was
  contamination from a too-small box ([[gravity_maturity/H2_BOX_LADDER_RESULTS|H2]]).
- The force's **direction** is derivable, and the answer is attraction — in a *modified*, variational
  model that has not been simulated.

**Not established:** anything about nature. Every result above is a statement about this model's own
equations. There is still **no parameter-free prediction checked against measurement**, which remains
the load-bearing gap.

---

## Associated docs

- [[SESSION_SYNTHESIS_2026-09-17]] · [[INTEGRATED_PLAN_2026-09]] · [[IRER_MASTER_HYPOTHESIS_CATALOG]]
- [[PREDICTIONS_2026-09-17]] — the companion: what I expect to happen next, stated in advance

## Branches

- [[Main branch]]
