---
tags: [record, plan, index]
date: 2026-09-12
status: active
supersedes: ACTION_PLAN_2026-08
---

# Integrated Plan — September 2026

Author: Claude, 2026-09-12. **This is the operative plan.** It folds four documents and one external
research pass into one ordered list:

- [[ACTION_PLAN_2026-08]] — phases 0–4 (the scientific spine; still correct, now better supported)
- [[RESOURCE_LIBRARY_ASSESSMENT_2026-09]] — R1–R8 (tooling triage)
- [[VISUAL_HUD_SCOPE_RFC]] — items 6.1–6.5 (6.1 and 6.2 built)
- [[gravity_maturity/DERRICK_SCALING_AND_TARGET_TRIAGE]] — two items closed 2026-09-12
- The literature pass at `D:\Resource-Library\RESEARCH FINDINGS - Quantule Mapper Discriminating Evidence.md`

The earlier documents remain valid as records of *why*. This one says *what next*.

---

## 1. What the research changed

| item | before | after |
|---|---|---|
| **Phase 1a method** | hand derivation, assumed to be a fallback | **Confirmed correct.** No soliton-perturbation software exists — searched exhaustively across Hirota, reductive and adiabatic branches. `sympy` + Jakobsen 2013 is the realistic path. |
| **Phase 1a prior** | sign freedom, cause unknown | **In analogue gravity — the nearest known relative — the metric's sign is *derived* from the equations of motion.** No case found where a sign was legitimately free after a proper derivation. The project's `a_sign` is **anomalous relative to its closest analogue**, not merely untested. |
| **Phase 1b (GAP-4)** | open | **Constrained.** Derrick applies to C2′ unchanged; `Ω(ρ)` contributes no dilation factor. **Q-ball ansatz required; static ruled out.** |
| **Phase 2 primary candidate** | π/2 (retired), then nothing concrete | **Critical exponent of the saturation cliff vs Choptuik universality.** |
| **Phase 2 secondary** | Townes `p_cr ≈ 1.86` | **Retired** — all 33 runs at `s = −0.5`, the cubic-quintic *stable* branch, not a marginal threshold. |
| **Secular drift** | unexplained blocker | **Two named candidates and a cheap test.** Reclassified — see §2. |
| **Screening falloff** | "is it known?" | **Known: three distinct non-Yukawa mechanisms.** New, sharper question: which does `Ω²(ρ)` resemble? |

---

## 2. The filter that should govern Phase 2 from now on

The research's central deliverable is a **pattern**: a first quantitative contact in this class of
programme takes one of exactly two forms.

1. **Mathematical-structure transfer** — the new theory's governing equation is shown to take
   *literally the same form* as an already-tested theory. **This is the clean-win case** (analogue
   gravity → Hawking temperature, subsequently measured by Steinhauer in a BEC).
2. **Discreteness/counting argument** — yields an order-of-magnitude number that the field itself
   labels heuristic (causal sets → Λ). Real, made in advance, but weaker.

And three failure modes, all of which the project can now recognise: **resemblance without
discrimination** (already π/2's fate), **confound-blocking** (DSR's fate), and **fitted agreement**.

> [!important] Applying that filter reclassifies our own candidates
>
> | candidate | which pattern? | verdict |
> |---|---|---|
> | **Saturation cliff → critical exponent** | **structure transfer** — universality-class membership is exactly a structural claim | **Primary Phase-2 candidate** |
> | **V7 / acoustic metric** (already in `FUTURE_WORK`) | **structure transfer — and the clean-win form** | **Promoted.** See below. |
> | Screening falloff shape | structure transfer | Secondary; needs the mechanism-matching question answered first |
> | **Secular drift** | **neither** | **Not a Phase-2 candidate at all.** It is hygiene. Reclassified. |
> | Townes critical power | — | Retired |

### The promotion that matters: V7 is the structure-transfer route, and it was already on the books

The TG wave operator, exactly as coded
(`gravity_TG_B2_two_node_awell.py:69`, `div_A_grad`):

$$\partial_t^2\phi \;=\; \nabla\cdot\!\left(c^2 A\,\nabla\phi\right) - m^2\phi + U'(\rho)\phi$$

`c²A(x,t)` is a **position- and time-dependent squared propagation speed**. That is the acoustic-metric
form for a non-flowing medium — and analogue gravity is the one programme in the research's survey
whose first contact **discriminated cleanly and was subsequently confirmed by measurement.**

**This is not a new idea, and I should not present it as one.** The project's own
[[FUTURE_WORK_AND_EXTERNAL_VALIDATION_PLAN]] already names "Analogue-gravity acoustic metrics
(Unruh; Visser 1998); polytropic `a = (γ−3)/2` → **V7 target**."

**What is new is the connection nobody has drawn:** V7 has been filed as an *external-legibility*
item — a way to make IRER readable to physicists. The research's Request-1 pattern says it is
something considerably more useful than that:

- It is a **structure transfer**, the clean-win pattern, in a project that currently has no
  structure-transfer candidate at all.
- In that framework the effective metric is **derived from the equations of motion**, which is
  precisely what Phase 1a needs. **The acoustic-metric route may derive the force sign directly** —
  one piece of work serving both the #1 priority and the load-bearing Phase-2 goal.

> [!warning] Stated as a hypothesis, not a result
> That the operator *has* the acoustic form is a reading of the code and is solid. That the mapping is
> **exact** (that `c²A` is a genuine effective metric rather than merely a similar-looking coefficient),
> and that the sign falls out of it, are **unverified**. Both are checkable analytically in about the
> same effort as Phase 1a itself, which is why they should be done *together* rather than sequenced.

---

## 3. The plan

### Tier 0 — hygiene, this week (~half a day, near-zero install)

Each closes an item that has been open longer than it should have been, and none is research.

| # | action | install | closes |
|---|---|---|---|
| **H1** | **Backend-parity check.** Save one state array from CuPy and one from the JAX mirror on a matched short run; compare with `cupy.testing.assert_array_almost_equal_nulp` and `jnp.allclose`. | **none** | A residual open since the baseline audit. Verified present. |
| **H2** | **Secular-drift discriminator.** Does the drift rate scale down under dt-halving and resolution-doubling (→ numerical) or not (→ physical phase-detuning)? **The integrator is RK4, which is not structure-preserving, so there is a real prior for "numerical."** | none | `TG_STATE_LOAD_LONG_TIME_DRIFT_UNRESOLVED` — reclassified out of Phase 2 and into hygiene |
| **H3** | **`mutmut` on `tests/test_physics_identities.py`.** Do C2.6/C2.8b/C3-class mutations survive? | `mutmut` | Tests a claim already in the record. |
| **H4** | **HUD 6.3** — centroid overlay via `skimage.feature.peak_local_max`. | none (installed) | The RFC item most likely to have caught C2.8b. |

### Tier 1 — the sign problem, and its newly-paired partner

**This is the scientific priority and it now has two routes that should be worked together.**

| # | action | install | notes |
|---|---|---|---|
| **S1** | **Phase 1a — perturbative hand derivation.** Linearise `A = 1 + ε_G G`, solve the screened T/G sector for a static source, evaluate `F_R`. | `sympy` | Confirmed by the literature as the only available path. Jakobsen 2013 is the method reference. `A_well_min ≈ 0.99993` means linear response is essentially exact here. |
| **S2** | **V7 — test whether the acoustic-metric mapping is exact.** Can `c²A` be written as a genuine effective metric? If so, does the force sign follow from the metric gradient? | none | **Do alongside S1, not after.** Same algebra, and it is the project's only structure-transfer candidate. |
| **S3** | If S1/S2 are inconclusive → **GAP-4 via `cadabra2`**, **with a Q-ball ansatz** (Derrick rules out static). | `cadabra2` | The Derrick constraint is now a design input, not a surprise. |
| **S4** | Once a reduced ODE exists → integrate with `diffrax`, batched over separations and phases. | `diffrax` | JAX-native; shares the substrate. |

> [!danger] The decision point that governs everything after
> **If S1/S2 derive the sign** → the TG sector is structural, Phase 2 has somewhere to live, and V7
> becomes a live discriminating-prediction route.
> **If they cannot** → falsification condition **F2** fires: the sector's distinguishing feature is a
> chosen input, and the project's defensible content is the conservative-substrate verification work.
> That is a real, publishable negative result and should be treated as one, not as a setback.

### Tier 2 — the discriminating quantity

| # | action | install | notes |
|---|---|---|---|
| **D1** | **Critical-exponent measurement of the saturation cliff.** Does the approach to threshold follow a power law `M ~ \|p − p*\|^γ` with a measurable exponent, or is it genuinely discontinuous? Compare the *shape* against Choptuik universality (γ≈0.37 for a massless real scalar — the project's field content differs, so an exact match is not expected and is not the test). | none | **A critical exponent is dimensionless and parameter-free by construction**, satisfying two of Phase 2's four criteria for free. This is the primary candidate. |
| **D2** | **SALib Sobol analysis inside the confirmed a\* basin**, where the response is smooth. Converts threat **T2** from a count into a measured ranking *with interactions*. | `SALib` | The research's own caution applies: sampling is unreliable near sharp thresholds, and this project has one. **Basin only.** |
| **D3** | **Mechanism-matching for the screening falloff** — is `Ω²(ρ)` structurally a chameleon (density-dependent *mass*), a symmetron (density-dependent *coupling*), or Vainshtein (*derivative* self-interaction)? | none | One of the two items the research explicitly left open as needing judgement. Note that `Ω` is a **kinetic coefficient**, which matches none of the three cleanly — that may itself be the answer. |
| **D4** | **PySR + SISSO** as a blind cross-check of whatever S1 derives. | `pysr` | Two algorithmically distinct methods agreeing with a hand derivation is strong. |

### Tier 3 — deferred, with named triggers

| item | trigger |
|---|---|
| `Hypothesis` property tests | after H3 reports |
| MMS / dt-convergence via `sympy` | when `sympy` arrives for S1 |
| `svirl` / `exponax` / `py-pde` solver cross-check | if H1 shows a parity gap |
| HUD 6.4 (PyVista interactive viewer) | when a genuinely 3-D question needs it — e.g. D3 |
| `cplot` / `complexplorer` domain colouring | opportunistic |
| `dynesty` / `UltraNest` | only once Phase 2 names a measurable to form a likelihood against |
| `pixi`, `Orbax` | if reproducibility or run-resilience becomes blocking again |

---

## 4. Retired — do not re-propose

| item | why |
|---|---|
| **π/2 crossover as a discriminator** | Karpman–Solov'ev / Gordon predict it. **Verification benchmark, filed beside `v = 2Dk`.** |
| **Townes critical power `p_cr ≈ 1.86`** | All 33 runs at `s = −0.5` — cubic-quintic stable branch, not a marginal threshold. |
| **Static-ansatz GAP-4** | Derrick applies unchanged; `Ω(ρ)` gives no evasion. |
| **Secular drift as a Phase-2 candidate** | Neither structure transfer nor counting argument. It is hygiene (H2). |
| `Aim`, `Sacred`, `Hydra`, `dynaconf`, `psutil`+`Supervisor`, `pyMultiobjective`, `MASA` | Per [[RESOURCE_LIBRARY_ASSESSMENT_2026-09]] §3. |
| **More characterisation runs** | The instrument is trusted. This would be avoidance. |

**One caution carried forward:** the BEC critical-atom-number target (`N_cr·\|a\|/a_ho ≈ 0.55–0.57`
mean-field vs `0.461 ± 0.012 ± 0.054` measured) is a **known beyond-mean-field discrepancy**, not an
open mystery. Using it as a target means competing with an explanation that already exists.

---

## 5. Ordering rationale

```mermaid
flowchart TD
    H["Tier 0 — hygiene<br/><i>~half a day, near-zero install</i>"]:::now
    S12["S1 + S2 together<br/>perturbative derivation<br/>+ acoustic-metric mapping"]:::crit
    S3["S3 — GAP-4 via cadabra2<br/><i>Q-ball ansatz, static ruled out</i>"]:::opt
    D1["D1 — critical exponent<br/>vs Choptuik universality"]:::crit
    F2["F2 fires:<br/>sign is an input<br/><b>real negative result</b>"]:::fail
    VER["Defensible content =<br/>conservative-substrate<br/>verification work"]:::soon

    H --> S12
    S12 -->|"sign derived → TG is structural"| D1
    S12 -->|inconclusive| S3
    S12 -->|"cannot be fixed"| F2
    S3 --> D1
    F2 -.-> VER

    classDef now  fill:#2d4a5e,stroke:#7fb0cc,color:#fff
    classDef crit fill:#7a2d2d,stroke:#e08080,color:#fff
    classDef opt  fill:#5e5426,stroke:#ccbe6a,color:#fff
    classDef soon fill:#2d5e3d,stroke:#7fcc95,color:#fff
    classDef fail fill:#4a3d6b,stroke:#8a7db8,color:#fff
```

**Tier 0 first** because it is half a day and removes ambiguity that would contaminate later work.
**S1 and S2 together** because they are the same algebra and S2 is the project's only structure-transfer
candidate. **D1 after S1/S2** because whether the TG sector is structural determines what a critical
exponent would even mean.

---

## 6. The standing test, unchanged

> [!danger] Verification is still A, validation is still F
> The research pass did not change the underlying position: **16 free physics parameters against one
> external comparison yielding a binary sign law.** What it changed is that the project now has
> *named, filtered candidates* rather than an open search.
>
> Every item above must still answer: **does this advance validation, or add machinery?** Tier 0 is
> explicitly hygiene and is budgeted as such. Tiers 1 and 2 are the science. Tier 3 is deferred
> precisely so it does not quietly become the work.

---

## What changed as a result

- **Code / model changes:** none yet — this is a plan.
- **Verdicts changed:** none. Two open questions closed on 2026-09-12 (Derrick, Townes).
- **What was done next, and why:** pending Jake's decision on Tier 0.

## Issues raised

| # | issue | status | resolved by |
|---|---|---|---|
| 1 | Force sign is an input (T3 / GAP-4 / F2) | OPEN | S1 + S2 |
| 2 | No discriminating dimensionless quantity | OPEN | D1 |
| 3 | Is the acoustic-metric mapping exact? | OPEN | S2 |
| 4 | Which screening mechanism does `Ω²(ρ)` match? | OPEN | D3 |
| 5 | Backend-parity residual | OPEN | H1 |
| 6 | Identity-CI bug-catching power unmeasured | OPEN | H3 |
| 7 | Is the secular drift numerical or physical? | OPEN | H2 |
| 8 | No independent human reviewer (T10) | OPEN | unchanged — cannot be fixed internally |

## Associated docs

- [[ACTION_PLAN_2026-08]] (superseded by this document) · [[SYSTEM_PRESSURE_TEST_AND_METHOD_REVIEW]]
- [[RESOURCE_LIBRARY_ASSESSMENT_2026-09]] · [[VISUAL_HUD_SCOPE_RFC]] · [[FUTURE_WORK_AND_EXTERNAL_VALIDATION_PLAN]] — V7
- [[gravity_maturity/DERRICK_SCALING_AND_TARGET_TRIAGE]] · [[gravity_maturity/TG_P1_EVIDENCE_RECONCILIATION]]
- [[Main branch]] · [[IRER_MASTER_HYPOTHESIS_CATALOG]]

## Branches

- [[Main branch]]

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** unknown — this document predates the clean-slate commit `909e6e2` (2026-07-01), so git carries no history for it. Use the citation and succession signals below instead.

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Also referenced by (same date or earlier):** [[ACTION_PLAN_2026-08]]

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
