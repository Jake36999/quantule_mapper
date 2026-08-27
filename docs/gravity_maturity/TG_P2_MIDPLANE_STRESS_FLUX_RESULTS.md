---
tags: [result, gravity]
date: 2026-08-25
branch: Branch - Gravity - Index
status: complete
verdict: TG_B2_MIDPLANE_FLUX_CONFIRMS_BODY_FORCE
---

# TG-P2 — Midplane Stress-Flux Estimator: the Body Force is Independently Confirmed

Author: Claude (primary), 2026-08-25. Campaign item **P2**, the item
[[TG_P1_EVIDENCE_RECONCILIATION|P1]] identified as the single highest-leverage gap.
Harness `jax_scout/gravity_TG_B2_midplane_stress_flux.py`; run
[[../runs/TG_B2_MIDPLANE_FLUX_N64]] (WSL/JAX GPU, GTX 1080, 0.69 h, x64, N=64 L=16 T=40 sep=3.0).

Mirror-only; the frozen TG-B1S dynamics are untouched — this module only **observes**.
**No gravity / UFF / IRER claim.**

> [!abstract] Verdict
> `TG_B2_MIDPLANE_FLUX_CONFIRMS_BODY_FORCE` — all four preregistered gates pass.

---

## 1. Why this was the priority

Every force number in the gravity branch — roughly 75 runs — came from **one** estimator, the
Gravity-D body force

$$F_R = -c^2 \int_{x>0} (\partial_x A)\,|\nabla\phi|^2\, dV$$

P1 established that this is a half-space total body force weighted by the **gradient energy only**,
one of four terms in the KG energy density, and that it had never been cross-checked against
anything. A second estimator was the only way to find out whether the number meant what it was
being read to mean.

## 2. The construction — an identity, not just a second number

Rather than invent another observable, the estimator is derived from the momentum ledger of the
system exactly as coded (`gravity_TG_B2_two_node_awell.rhs_2n`). That system follows from

$$\mathcal{L} = |\pi|^2 - c^2 A|\nabla\phi|^2 - m^2\rho + U(\rho), \qquad
U(\rho) = \tfrac{a}{2}\rho^2 + \tfrac{s}{3}\rho^3 + \tfrac{f}{4}\rho^4$$

Differentiating the momentum density $p_x = -2\,\mathrm{Re}(\pi^*\partial_x\phi)$ and substituting the
equations of motion gives the exact local conservation law

$$\partial_t p_x = \partial_x S + (\text{transverse}) - c^2(\partial_x A)|\nabla\phi|^2, \qquad
S = c^2A|\nabla\phi|^2 - 2c^2A|\partial_x\phi|^2 - |\pi|^2 + m^2\rho - U(\rho)$$

Integrating over the half-space, the transverse divergences drop and

$$\boxed{\;\frac{d}{dt}P_x(x>0) = \big(S_\text{out} - S_\text{in}\big) + F_R\;}$$

**The last term is exactly the existing body force.** So the flux gives (i) a second estimator
$F_\text{flux} = dP_x/dt - (S_\text{out}-S_\text{in})$ and (ii) — more valuably — a **falsifiable
identity** the two must jointly satisfy.

It is genuinely independent: $F_\text{flux}$ samples a **2-D plane** and carries $|\pi|^2$, $m^2\rho$
and $U(\rho)$, none of which appear anywhere in $F_R$. Not a rearrangement of the same quantity.

## 3. Validation of the tensor — the OFF-arm null (G1)

With feedback off, $A=1$, so $F_R = 0$ *identically*. The ledger must then close on the flux terms
alone — a test of the stress tensor with **no involvement of $A$ at all**. Residual, relative to the
scale of the individual terms:

| `sample_dt` | plane variant | divergence variant | ratio (div) |
|---|---:|---:|---:|
| 0.25 | 6.05e-02 | 6.05e-02 | — |
| 0.10 | 8.94e-03 | 9.74e-03 | 6.2 *(expect 6.25)* |
| 0.05 | 8.85e-03 | **2.44e-03** | 4.0 *(expect 4.00)* |

**Exactly second order in `sample_dt`.** The residual is the centred-difference error in $dP_x/dt$,
not a physics error. The identity holds and the derivation is confirmed. The plane variant floors at
~9e-03, which is the O(dx) interface-interpolation limit.

## 4. The comparison (G3, G4) — and why it must be differential

$F_R$ is a **differential** quantity: identically zero when $A=1$, so it measures only the
A-mediated force. $F_\text{flux}$ is a **total**: it also contains any bare KG two-body force, which
$F_R$ cannot see. Comparing them directly is a category error. The correct comparison — and the
convention the two-node harness already uses ("isolate the loop force as `F_full - F_off`") — is

$$F_\text{flux}(\text{arm}) - F_\text{flux}(\text{off}) \quad\text{against}\quad F_R(\text{arm})$$

differenced **sample by sample**, since the arms share initial conditions.

| arm | $\langle F_\text{flux}-F_\text{off}\rangle$ | $\langle F_R\rangle$ | gap | stat. uncertainty |
|---|---:|---:|---:|---:|
| **well** | −5.69130e-05 ± 8.06e-07 | −5.71074e-05 | **0.340%** | ±1.42% |
| **hill** | +5.69136e-05 ± 8.06e-07 | +5.71078e-05 | **0.340%** | ±1.42% |

**The two estimators agree to 0.34%, which is well inside the 1.4% statistical uncertainty.** The
differential is significant at 71σ, the sign reverses correctly between well and hill (G4), and the
per-sample correlation between the two estimators is **r = 0.89** — they track each other
sample-by-sample, not merely on average.

> [!important] The body force is confirmed as an estimator of the A-mediated force
> `F_R` measures what it was believed to measure. This closes the largest open instrument question
> in the gravity branch and satisfies goal **I1** in [[../Main branch]].

## 5. What is *not* resolved — the bare two-body force

The absolute $F_\text{flux}$ is a **~1700:1 cancellation** between $dP_x/dt$ (1.597e-01) and the plane
flux (1.598e-01). Its per-sample scatter is 1.89e-02, roughly 205× its own mean:

| quantity | value | significance |
|---|---:|---|
| bare KG two-body force, $F_\text{flux}(\text{off})$ | −9.21e-05 ± 8.62e-04 | **0.11σ — NOT MEASURED** |
| A-mediated force (differential) | −5.6913e-05 ± 8.06e-07 | 71σ |

> [!warning] 0.11σ means "not measured", not "measured to be zero"
> Nothing can be concluded about the bare (A-independent) two-body force from this run. An earlier
> reading of mine took the −9.21e-05 mean at face value and inferred a bare force 1.6× larger than
> the A-mediated one; that was wrong — the mean is statistically indistinguishable from zero at this
> run length. Resolving it needs variance reduction or ~4×10⁴ times the samples, not a longer run of
> the same kind.

The differential is resolvable precisely *because* this enormous scatter is common mode across arms:
differencing reduces the standard deviation by a factor of **1069** (1.89e-02 → 1.77e-05). That is
also why the paired, sample-by-sample difference matters — averaging the arms separately and then
subtracting would not have worked.

## 6. What this changes

**Confirmed:**
- `F_R` is a valid estimator of the A-mediated inter-node force, to 0.34% against a fully independent
  method, with correct sign reversal.
- The momentum ledger of the coded system closes at second order.
- The converged A-well attraction ([[TG_B2_CONVERGENCE_N80_L20_RESULTS]]) survives this check.

**Unchanged — P1's downgrades all still stand:**
- `F/M_p` is still not an acceleration; this result says nothing about mass normalization.
- The mass axis is still confounded with morphology.
- The force's *direction* is still an input (`a_sign` is a runtime flag), not a derived result.
- The falloff is still exponential-like and short range.

**Newly available:** a second observable for every future force claim, and an identity that will
catch any future error in the momentum sector the way the C2.6 identity check caught `D_eff = D/151`.

## 7. Method notes (two errors caught in development)

- **Sign convention.** The flux term is $S_\text{out}-S_\text{in}$, from
  $\int_a^b \partial_x S\,dx = S(b)-S(a)$. My first implementation had it reversed; caught because
  the two flux variants disagreed with each other while agreeing in magnitude.
- **Plane placement.** The region is the mask $X>0$, i.e. cells $n/2{+}1 \dots n{-}1$, whose faces lie
  at the cell *interfaces* $x=dx/2$ and $x=L/2-dx/2$ — **not** on grid planes. Sampling $S$ at index
  $n/2$ and index 0 misplaces one face by half a cell and the seam by a whole one, which opens the
  ledger at O(1) (residual 3.37 vs 0.06). Both variants are retained so the placement error stays
  visible.
- **Gate specification.** G3/G4 were initially written to compare $F_\text{flux}$ against $F_R$
  directly, producing a spurious `..._DISAGREES_WITH_BODY_FORCE` verdict. The estimators were never
  in disagreement; the gate was comparing a total against a differential. Corrected in the harness,
  with `--reanalyze` added so a completed run can be re-scored without re-simulating.

## 8. Next

- **P1-a** (per-half-space energy observable) is now the remaining P1 blocker; `F/E_R` would be the
  first dimensionally coherent acceleration-like quantity.
- Re-run this estimator at the converged N=80/L=20 configuration alongside the sep-4.0 row.
- The **non-variational loop (GAP-4)** is now the sharpest open problem: with the estimator confirmed,
  the fact that the force's sign remains a free flag is no longer maskable as instrument uncertainty.

---

## Previous experiments

- [[TG_P1_EVIDENCE_RECONCILIATION]] — the audit that made this the priority
- [[TG_B2_CONVERGENCE_N80_L20_RESULTS]] · [[TG_B2_FARFIELD_KERNEL_RESULTS]]

## Associated docs

- [[../IRER_MASTER_HYPOTHESIS_CATALOG]] · [[../RUN_QUEUE]] · [[../META_ANALYSIS_BRANCH_PROGRESS]]
- [[../runs/TG_B2_MIDPLANE_FLUX_N64]]

## Next experiment

- [[TG_P1A_ENERGY_OBSERVABLE]]

## Branches

- [[../Branch - Gravity - Index]]

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `5a19e60` (2026-08-25) — *P2 CONFIRMED: midplane stress-flux estimator validates the body force to 0.34%*
**Revised since:** 3 commit(s), most recently `9517d9f` (2026-08-26)

**Harness code changed since it was written:** 4 commit(s) to `jax_scout/`.
  - `06edae5` 2026-08-27 — Fix run provenance: 30% -> 86% of runs now dated
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Also referenced by (same date or earlier):** [[ACTION_PLAN_2026-08]], [[IRER_MASTER_HYPOTHESIS_CATALOG]], [[RUN_QUEUE]], [[SYSTEM_PRESSURE_TEST_AND_METHOD_REVIEW]], [[gravity_maturity/TG_P1A_ENERGY_OBSERVABLE]]

**The master catalog references this document** — the catalog is the authority on whether its verdict is still live:

> | **TG P2 — midplane stress-flux estimator** | **CONFIRMED (2026-08-25)** | `docs/gravity_maturity/TG_P2_MIDPLANE_STRESS_FLUX_RESULTS.md`, run `TG_B2_MIDPLANE_FLUX_N64`. Verdict `TG_B2_MIDPLANE_FLUX_CONFIRMS_BODY_FORCE`; all 4 preregistered gates pass. Derived the exact momentum ledger of the coded …

<!-- LINEAGE:END -->
