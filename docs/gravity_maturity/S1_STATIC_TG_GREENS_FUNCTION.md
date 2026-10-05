---
tags: [result, gravity, theory]
date: 2026-09-17
branch: Branch - Gravity - Index
status: complete
verdict: S1_STATIC_TG_SOLVED_EXACTLY__MEDIATOR_IS_A_TWO_SCALE_SCREENED_FIELD__G_SIGN_DERIVED
---

# S1 — The Static T/G Sector, Solved Exactly

Author: Claude (primary), 2026-09-17. Plan item **S1**, worked with
[[S2_V7_ACOUSTIC_METRIC_RESULTS|S2]] as the plan directs.
Analysis `tools/s1_static_tg_greens_function.py` (sympy + a numerical Green's function check).

Observation-only; nothing simulated. **No gravity / UFF / IRER claim.**

> [!abstract] Verdict
> `S1_STATIC_TG_SOLVED_EXACTLY__MEDIATOR_IS_A_TWO_SCALE_SCREENED_FIELD__G_SIGN_DERIVED`
>
> No perturbation series was needed. The T/G sector is **linear**, so the static problem has a
> closed-form solution — this is the whole answer, not the leading term of one.

---

## 1. Why the "perturbative hand derivation" turned out not to be perturbative

The plan scoped S1 as *"linearise `A = 1 + ε_G G`, solve the screened T/G sector for a static source"*.
The first half is right and necessary: `A_well_min ≈ 0.99993` puts the loop in strict linear response.

The second half needed no approximation at all. **The only nonlinearity in the model lives in the φ
sector** (through `U(ρ)` and through `S_state`). Read off `rhs_2n` with $\partial_t=0$, $V=0$:

$$0 = c_T^2\nabla^2 T - \omega_T^2 T + \alpha_T S - \kappa G, \qquad
0 = c_G^2\nabla^2 G - \omega_G^2 G - \kappa T$$

Two coupled **linear** equations. In Fourier space:

$$\hat G(k) = \frac{-\alpha_T\,\kappa\,\hat S(k)}
{\left(c_T^2k^2+\omega_T^2\right)\left(c_G^2k^2+\omega_G^2\right) - \kappa^2}$$

## 2. The mediator is a difference of two Yukawas

The denominator is **quartic** in $k$, not quadratic, so the mediator is not a single Yukawa. Its two
poles, at the frozen couplings ($c_T{=}0.7$, $c_G{=}0.55$, $\omega_T{=}1.25$, $\omega_G{=}0.85$,
$\kappa{=}0.55$):

| | $\mu^2$ | $\mu$ | range $1/\mu$ |
|---|---:|---:|---:|
| long | 1.305041 | 1.1424 | **0.8754** |
| short | 4.272164 | 2.0669 | **0.4838** |

Both real and positive, so

$$G(r) \;\sim\; \frac{e^{-\mu_- r} - e^{-\mu_+ r}}{r}$$

> [!important] The recorded falloff is derivable, not merely empirical
> The project has carried "the force is short-range and exponential-like rather than a power law;
> the far-field is not yet reached" as an **observation**, and listed it as one of three candidate
> discriminators. It is a **consequence of the coded couplings**, with both scales fixed. The
> "far field" is simply the longer range, $1/\mu_-\approx 0.88$ — which at the box sizes used
> (L = 10–16) is why it was never reached.

Confirmed numerically. Fitting $\ln|rG(r)|$ on a Green's function computed by FFT, with the window
pushed outward until the slope stops moving:

| fit window | $\mu_-$ |
|---|---:|
| $r\ge2$ | 1.1182 |
| $r\ge3$ | 1.1309 |
| $r\ge4$ | 1.1366 |
| $r\ge5$ | 1.1393 |
| $r\ge6$ | **1.1406** |

against the analytic **1.1424** — **0.155%**, converging monotonically as the near-field and the
second Yukawa drop out.

## 3. The sign of G is derived — `a_sign` is the *only* remaining freedom

At $k=0$ the denominator is $\omega_T^2\omega_G^2 - \kappa^2 = 0.826 > 0$, and it only grows with
$k^2$, so it never changes sign. Therefore $\operatorname{sign}\hat G = \operatorname{sign}(-\alpha_T\kappa\hat S)$, and with
$\alpha_T = 0.35 > 0$, $\kappa = 0.55 > 0$:

**A positive state-load gives $G<0$. Derived from the coded couplings, not chosen.**
(Confirmed numerically: $G$ at the source $=-2.05\times10^{-2}$, and $G\le0$ everywhere.)

Combining with [[S2_V7_ACOUSTIC_METRIC_RESULTS|S2]], the whole chain is now accounted for:

| link | status |
|---|---|
| load → $T$ | **derived** ($\alpha_T>0$ ⟹ $T>0$) |
| $T$ → $G$ | **derived** ($\kappa>0$ ⟹ $G<0$) |
| $G$ → $A$ | **`a_sign` — the one free flag** |
| $A$ → motion | **derived** (geodesic: toward smaller $A$, S2) |

> [!danger] Three of four links are derived. The residual freedom is one sign in one place.
> This is a sharper statement of the sign problem than the project has had. It is *not* progress
> toward solving it: `a_sign` remains exactly as free as before. But it rules out the possibility
> that the sign was hiding somewhere else in the chain, and it confirms S2's conclusion that the
> only route is **S3 — make the T–G coupling variational**.

## 3a. Verified against the code, exactly — and it corrects a recorded number

`gravity_TG_B2_static_force.static_G` implements

```python
D  = (cT**2 * k2 + omega_T**2) * (cG**2 * k2 + omega_G**2) - kappa_TG**2
Gk = -kappa_TG * alpha_T * Sk / D
```

which is **term for term** the $\hat G(k)$ derived in §1. The derivation is not merely consistent with
the implementation; it is the implementation.

That makes one recorded number checkable. [[../runs/TG_B2_FARFIELD_20260822_205537]] reports
`kernel_screening_length = 0.9221`, obtained by fitting $\ln|G(r)|$ over $r\in(3,\,0.42L)$ on a
solved Q-ball source. Recomputing it with the real source reproduces **0.9221 to four decimals**, and
pushing the fit band outward shows what it is made of:

| fit band | $\lambda$ |
|---|---:|
| $r\in(2,11.76)$ | 0.9275 |
| $r\in(3,11.76)$ | **0.9221** ← the recorded value |
| $r\in(5,11.76)$ | 0.9137 |
| $r\in(7,11.00)$ | 0.9099 |

> [!warning] 0.9221 is not the mediator's range, and should not be read as one
> The kernel has **two exact ranges, 0.8754 and 0.4838**. The recorded 0.9221 is a *single*-exponential
> fit to a *two*-scale kernel convolved with an *extended* source, over a finite band — it drifts with
> the band and never reaches the pole. It sits ~4% above 0.8754 because the Q-ball's own tail (decay
> length 1.88 for the energy-like source, against the mediator's 0.875) is still contributing at these
> radii.
>
> The number is reproducible and useful as an effective length. It is not a constant of the model,
> and the constants of the model are now known exactly.

## 4. A parameter-free prediction that is checkable now

With $G$ a two-scale screened field and $F_R = -c^2 a_\text{sign}\varepsilon_G\int_{x>0}(\partial_x G)|\nabla\phi|^2dV$
(the linear-response form S2 verified to 0.03%), the force-versus-separation curve must fit a
**two-scale screened form with both ranges fixed by the couplings and no free parameters**.

A single-Yukawa fit should fail systematically at small separation, where the second term is not
negligible. The project already has force-versus-separation runs; this is a re-analysis, not new
compute.

> [!warning] This discriminates the model against itself, not against nature
> It is a **verification** target of the sharpest kind available — a parameter-free prediction the
> harness can be checked against — but it is not the discriminating quantity Phase 2 needs. **D1**
> (the critical exponent) remains the primary candidate for that.

## 5. What changed

- The screened-falloff "candidate discriminator" is **closed as derived model behaviour**. It was
  never going to discriminate: it is a consequence of the couplings, so any theory with these
  couplings predicts it.
- The sign problem is **localised to one flag**, with the other three links derived.
- A new, cheap verification target: the two-scale falloff fit.

---

## 6. A consequence for the saturation cliff — it cannot live where it was being looked for

The load→geometry map is **exactly linear**: given $S$, the field $G$ is its convolution with the
Green's function above. Doubling the load doubles $G$, at every load, with no threshold available
anywhere in the T/G sector.

**A linear map cannot produce a sharp threshold.** So the saturation cliff — recorded as
`the load→geometry response is a sharp threshold, not graded`, currently pausing the gravity ladder
and named as a candidate discriminator — **cannot originate in the T→G chain**. It must be upstream,
in $S_\text{state}(\phi,\pi)$ or in the φ sector that produces it.

[[TG_P1_EVIDENCE_RECONCILIATION|P1 §2]] already established the relevant property of that upstream
part, and this is its consequence rather than a new finding: `e_ref` and `q_ref` are **peak
normalisers recomputed per configuration**, so each node is normalised by *its own* peak energy, and
≈0.36 of the source's mass exponent is removed by the normalisation rather than by physics.

Putting the two together narrows the cliff to two candidates, and they are distinguishable:

1. **The self-normalisation.** A source divided by its own peak cannot grow indefinitely with load,
   which is a saturation by construction rather than a physical yield point. P1-b (the frozen-`e_ref`
   mass sweep) is the existing, cheap test — and it is now better motivated than when P1 proposed it.
2. **The φ sector's own stability boundary** — whether a coherent node survives the load at all.
   That is a genuine threshold and would be physics.

> [!important] This redirects D1
> **D1** proposes measuring the cliff's critical exponent as the primary discriminating quantity.
> If the cliff is (1), the exponent measures a normalisation choice and discriminates nothing.
> **P1-b should run before D1**, because it separates the two at a fraction of the cost, and D1's
> value depends entirely on which answer it gives.

---

## Issues raised

- The two ranges (0.88 and 0.48) are **comparable to the core radius** (2.0) and much smaller than
  the boxes used (L = 10–16). Worth checking whether a point-source Green's function is the right
  idealisation when the source is wider than the mediator's range — it may not be.
- $\kappa^2$ enters the denominator with a minus sign, so as $\kappa^2 \to \omega_T^2\omega_G^2$ the
  static response **diverges**. At the frozen couplings $\kappa^2 = 0.3025$ against $0.8264$ — a
  factor 2.7 below, closer than is comfortable for a parameter that has been swept. Worth knowing the
  boundary exists: $\kappa_\text{crit} = \omega_T\omega_G = 1.0625$.

## Previous experiments

- [[S2_V7_ACOUSTIC_METRIC_RESULTS]] — the response law, and the linear-response force form used here
- [[TG_P2_MIDPLANE_STRESS_FLUX_RESULTS]]

## Associated docs

- [[../INTEGRATED_PLAN_2026-09]] · [[../IRER_MASTER_HYPOTHESIS_CATALOG]]

## Branches

- [[../Branch - Gravity - Index]]
