---
tags: [result, gravity, theory]
date: 2026-09-17
branch: Branch - Gravity - Index
status: complete
verdict: S3_VARIATIONAL_COMPLETION_FORCES_ATTRACTION__a_sign_FREEDOM_IS_AN_ARTEFACT_OF_GAP4
---

# S3 / GAP-4 — A Variational Loop Has No Free Sign

Author: Claude (primary), 2026-09-17. Plan item **S3**, promoted to the critical path by
[[S2_V7_ACOUSTIC_METRIC_RESULTS|S2]].
Analysis `tools/s3_variational_sign.py` (sympy + numerical check).

Analysis of a **modified** model. The frozen dynamics are untouched and nothing was simulated.
**No gravity / UFF / IRER claim.**

> [!abstract] Verdict
> `S3_VARIATIONAL_COMPLETION_FORCES_ATTRACTION__a_sign_FREEDOM_IS_AN_ARTEFACT_OF_GAP4`
>
> The project's single largest open problem — that the force's direction is a runtime flag — is a
> consequence of the loop not deriving from an action. Restore the missing variation and the sign
> cancels identically.

---

## 1. The term the coded G equation is missing

The Lagrangian contains $-c^2A(G)|\nabla\phi|^2$. Varying it with respect to $\phi$ gives
$\nabla\cdot(c^2A\nabla\phi)$ — the coded operator, which is why the acoustic metric of S2 exists.

**Varying the same term with respect to $G$ gives a source for $G$ that the coded equation does not
have:**

$$\frac{\partial\mathcal{L}}{\partial G} = -a_\text{sign}\,c^2\varepsilon_G\,A\,|\nabla\phi|^2$$

so a variational G-equation must read

$$\ddot G = c_G^2\nabla^2G - \omega_G^2G - \kappa T \;\underbrace{-\;c^2 a_\text{sign}\varepsilon_G A|\nabla\phi|^2}_{\textbf{absent from the code}}$$

**The coded $\ddot G$ contains no $|\nabla\phi|^2$ term at all.** That is GAP-4, stated exactly rather
than as "the loop is non-variational".

## 2. The sign cancels — `a_sign` enters squared

In the static local limit (justified by [[S1_STATIC_TG_GREENS_FUNCTION|S1]]: the mediator range 0.875
is short next to the source):

$$G^* = -\frac{a_\text{sign}c^2\varepsilon_G|\nabla\phi|^2}{\omega_G^2}$$

$G^*$ flips with the polarity — but $A$ carries $a_\text{sign}$ a **second** time:

$$A^* = \exp\!\left(a_\text{sign}\varepsilon_G G^*\right)
= \exp\!\left(-\frac{a_\text{sign}^2\,c^2\varepsilon_G^2|\nabla\phi|^2}{\omega_G^2}\right)$$

$a_\text{sign}^2 = 1$. Both polarities give the identical exponent
$-c^2\varepsilon_G^2|\nabla\phi|^2/\omega_G^2$, which is **negative definite**. So $A<1$ near a node —
an **A-well** — either way, and by S2's geodesic result motion is always toward smaller $A$.
**Attraction, with no free sign.**

Confirmed numerically on an explicit Gaussian node:

| route | `a_sign = +1` | `a_sign = −1` | |
|---|---|---|---|
| **variational** | `G = −4.674513e-03`, `A_min = 0.999694685` | `G = +4.674513e-03`, `A_min = 0.999694685` | **WELL both** |
| **coded** (via `α_T·S_state`) | `A_min = 0.969053` | `A_max = 1.031935` | WELL / **HILL** |

The coded route flips well↔hill with the flag. The variational route cannot.

## 3. Why this is not an accident of the exponential

Nothing above used the specific form $A=\exp(a\varepsilon G)$. The structure is general: for
$\mathcal{L}\supset f(G)\,X[\phi]$ with $X = -c^2|\nabla\phi|^2$,

- $\phi$'s equation gets $f(G)$ — how $G$ modulates $\phi$;
- $G$'s equation gets $f'(G)X$ — how $\phi$ sources $G$ (the missing term);
- static response $G^*\sim -f'(G)X/\omega_G^2$, so the induced change in the modulator is
  $\delta f \sim -f'(G)^2X/\omega_G^2$.

$f'(G)^2\ge0$ always, so the sign is set by $X$ alone, and $X$ is the $\phi$ kinetic term, which is
sign-definite. **The mediator's polarity cancels.**

This is the standard reason **scalar exchange between like sources is attractive**, and why a vector
mediator — whose source is a current rather than a square — can repel. The project is not exempt
from it; the coded loop escapes it only by routing $\phi\to G$ through a **separate, non-variational
channel** (`α_T·S_state` → T → G), which permits the source polarity to be chosen independently of
the modulation polarity. **Two independent signs is one more than an action allows.**

## 4. What must actually change

> [!danger] Adding the term is not sufficient for a *clean* derivation -- but it is far from marginal
> If the `α_T·S_state → T → G` channel is retained alongside the restored variation, then
> $G = G_\text{variational} + G_\text{coded}$, and the coded part does **not** carry `a_sign`, so
> residual sign freedom survives in proportion to its share of $G$.
>
> **Measured on solved Q-ball profiles**, using the coupled static system -- a source entering G's
> equation propagates with $d_T/D$, not $1/d_G$:
>
> | `w` | \|G\| coded | \|G\| variational | variational / coded |
> |---|---:|---:|---:|
> | 0.945 | 1.108e-03 | 2.802e-03 | **2.53** |
> | 0.955 | 9.396e-04 | 2.669e-03 | **2.84** |
> | 0.964 | 8.293e-04 | 2.418e-03 | **2.92** |
> | 0.972 | 7.685e-04 | 2.041e-03 | **2.66** |
> | 0.980 | 7.552e-04 | 1.459e-03 | **1.93** |
>
> The coded column reproduces the recorded `|G_min|` of the charge-audit rows (8.293e-04 at
> w = 0.964), so the comparison is anchored to measured data rather than to a toy.
>
> **The variational term is the LARGER of the two, by 1.9-2.9x.** Simply adding it makes the loop
> majority-variational: attraction becomes the dominant behaviour, with roughly a **26-34% residual
> contamination** from the non-variational channel rather than the flag deciding the outcome.
>
> For a derivation with *no* free sign, the state-load channel must still be **replaced**. But the
> distance to "attraction dominates" is one added term, not a rewrite.

> [!warning] Correction to an earlier figure in this document
> A first pass put this ratio at **~110x the other way** -- that the variational coupling was
> negligible and would make an already-tiny effect ~100x tinier. That was wrong in both magnitude
> and direction. It compared an arbitrarily normalised Gaussian `phi**2` against the real
> `S_state`, which carries a division by `S0 = 135.686` and by per-configuration peak normalisers,
> and it used the wrong propagator for a source entering G's equation. A ratio taken from a toy
> profile is worth nothing when the real source carries normalisers the toy does not.

The deeper reason the loop is not variational is worth stating plainly: **`S_state` is an
*observable*, not a field-theoretic source.** It is built from energy and charge densities and then
divided by *that configuration's own peak values*
([[TG_P1_EVIDENCE_RECONCILIATION|P1 §2]]). No Lagrangian variation produces a peak-normalised
quantity. Anything normalised by its own maximum cannot be $\partial\mathcal{L}/\partial G$ for any
$\mathcal{L}$.

## 5. What this establishes, and what it does not

**Establishes:**
- `a_sign`'s freedom is **not** a physical ambiguity in the theory. It is a consequence of GAP-4, and
  it disappears under variational completion.
- The variational answer is **attraction**, derived rather than chosen — and it agrees with the
  polarity the project has been calling "theory-faithful".
- The argument is structural, not model-specific: it is scalar exchange between like sources.

**Does not establish:**
- That the variational model reproduces anything else the project has measured. **This changes the
  equations of motion**, and the modified model has not been simulated.
- That the effect survives at a useful size -- though here the news is *good*: the variational
  coupling is **2-3x stronger** than the coded one on real profiles, so a variational loop would
  make the effect larger rather than smaller. It stays small in absolute terms
  ($A-1 \approx 7\times10^{-5}$ at present).
- Anything about nature. Deriving the sign removes the project's worst internal problem; it does not
  supply a discriminating prediction, and falsification condition **F2** is about whether the sector
  is structural, which this addresses, not about evidence.

> [!important] The decision point in the plan has been reached
> The plan states: *"If S1/S2 derive the sign → the TG sector is structural."* S1 and S2 did not, and
> located the problem. S3 does, conditionally: **the sign is derivable, and the price is replacing
> the state-load source with a variational one.** That is a different model, and whether it is still
> *this project's* model is Jake's call, not mine.

---

## Issues raised

- The variational replacement removes `S_state`, and with it the "informational state load" that
  motivates the IRER reading. A variational source is $|\nabla\phi|^2$ — gradient energy, not
  information. **That may be the more important consequence than the sign.**
- The T field has no variational role in the replacement as analysed. If `α_T·S_state` goes, it is
  worth asking what T is for.
- ~~The ~110x weakening should be checked properly rather than from one Gaussian.~~ **Done, and it reversed the conclusion** (§4 correction box).

## Previous experiments

- [[S2_V7_ACOUSTIC_METRIC_RESULTS]] — located the sign in the non-variational part
- [[S1_STATIC_TG_GREENS_FUNCTION]] — the static/local limit used here

## Associated docs

- [[../IRER_MASTER_HYPOTHESIS_CATALOG]] · [[../INTEGRATED_PLAN_2026-09]]

## Branches

- [[../Branch - Gravity - Index]]
