---
tags: [record, meta, index]
date: 2026-08-25
status: complete
---

# Session Synthesis — what changed, what it revealed, and where the project actually stands

Author: Claude, 2026-08-25. Eleven commits, 918 files, +34,043 lines. This document is the
synthesis: what was built, **what building it made visible that nobody knew before**, the physics as
we now honestly understand it, and how the overall picture of the project has shifted.

Readable standalone. Nothing here changes a verdict.

---

## 1. What changed

| commit | what |
|---|---|
| `292fc96` | Obsidian vault: run catalogue (182 notes), experiment tracker, documentation methodology, 486 image embeds |
| `96b324c` | Meta-analysis of branch progress against Jake's own stated goals |
| `eeb8f6d` | Midplane stress-flux estimator; stability review checklist; refined goal structure |
| `5a19e60` | **P2 confirmed** — the body force validated to 0.34% against an independent estimator |
| `b4e0613` | **P1-a** — per-half-space energy observable; closes the P1 checklist |
| `616fe31` | Document lineage — what changed after each document |
| `3eb93af` | System pressure test: dependency graph, threat register T1–T10 |
| `9517d9f` | Infrastructure assessment: build three small things, not a platform |
| `050a832` | Physics-identity CI + manifest results index |
| `06edae5` | Run provenance: 30% → 86% of runs dated |
| `60093e5` | Action plan |

Three campaign items closed (**P1**, **P1-a**, **P2**); three infrastructure items built
(identity CI, results index, provenance); five assessments written.

---

## 2. What the changes revealed

Almost everything of value this session came from **building an instrument and then looking through
it**, not from planning. These are things nobody knew before the work was done.

### About the physics observable

| discovery | how it surfaced |
|---|---|
| **`F_R` couples to just 4.5% of a node's energy** — `E_kin` is 48.6%, `E_mass` 54.6%, `E_pot` −7.7%. P1 said "one of four terms"; the term is one twenty-second. | Only visible once P1-a added an energy observable that had never existed |
| **That coupling fraction varies 29.1% NON-MONOTONICALLY across the mass axis** (0.0354→0.0483, peaking at M≈70) | Computed from *existing* charge-audit rows — the data had been sitting there since July |
| **`M_R ≡ E_mass_R` exactly** at `m=1`, so the historical `F/M_p` was force ÷ the mass-energy term alone | Fell out of the decomposition |
| **The `F ~ M^-0.067` exponent has R² = 0.11** — it was never a power law, just flat scatter | Refitting five points that had been quoted as an exponent for a month |
| **The whole effect is a 7e-5 perturbation** (`A_well_min ≈ 0.99993`) — strictly linear response | Reading `A_well_min` at full precision instead of 3 significant figures |
| **The body force is correct** — 0.34% against a fully independent estimator, 71σ, r=0.89 | P2 |
| **The bare two-body force is NOT measured** (0.11σ) — I briefly claimed otherwise from a mean without its uncertainty | Checking the sem before publishing |

### About the record

| discovery | significance |
|---|---|
| **The master catalog names only 6 documents.** It refers to results by verdict string and prose, never by document. | **This is the structural reason a document cannot tell you its own status** — the link from evidence to status was never made |
| **188 of 213 working documents have no later document citing them** | Flag, not proof — cross-referencing is sparse (1.6 links/doc) — but it is a real review queue |
| **0 of 213 documents record their own consequences** | The gap Jake identified while working the checklist |
| **Sectors close by burst-exhaustion, not by decision** — Stability was 35 runs in 3 days; Transport 44 in 8 days then silent for six weeks with three OPEN threads | Nobody decided to stop; the burst just ended |
| **128 runs were undated — and 102 of them were recoverable** | I had asserted they could not be retro-dated. Wrong. |

### About the system

| discovery | significance |
|---|---|
| **There is no Redis or Docker anywhere** — only in 2025 transcripts | "Expanding it" would have been building it |
| **A results database already existed and was abandoned** — `queue_runtime.db`, 1,850 rows, all pointing at a dead `G:\` drive since March | The right question was *why it died*, not whether to build one |
| **`core_saturation_search` has 38 dependents**; 11 modules define their own energy/force observable | The mechanism behind C2.8b and the `F_R` confusion |
| **Two long-term goals were already being worked on, unrecognised** | The T↔G long-time drift **is** the sequential-time goal; the two-body binding results **are** the co-operation primitives |

---

## 3. The physics, as we now honestly understand it

### Dissipative substrate (S-NCGL) — **closed, negative**

A real attractor exists at ×1.15 cubic gain, bracketed ±0.5%, confirmed across seed, N=128 and
T=144k, and explained as a **gain/loss balance** rather than a topological object. Nodes are
rotational cores, not vortices.

It **does not move** (mobility negative across three morphologies) and it is **not structurally
predictable** — five independent predictors returned NULL (log-prime, TDA, tensor routing,
bridge/void, Payan). That is a substantive negative result about the dissipative substrate, and it
is why the pivot to conservative substrates was correct.

*Caveat: closed on the strength of 35 runs over three days whose ~200 figures have never had a
second reading. That review is [[Branch - Stability - Review Checklist|open]].*

### Conservative substrates (NLS, KG) — **the strongest physics in the project**

- Solitons translate inertially at **`v = 2Dk`, agreement 0.9999** — a Galilean identity, not a fit.
- KG Q-balls are VK-stable (`dQ/dω = −849`) and transport with **E and U(1) charge conserved to ~1e-13**.
- A **phase-dependent two-body law**: static force crosses over at **Δφ = π/2** — attractive below,
  repulsive above.
- Collisions **bind rather than scatter**. Capture is generic; transmission occurs *only* at exact
  anti-phase and `v ≤ 0.45c`.
- **The same two-body law appears in both NLS and KG.**

> [!important] Cross-substrate universality is the most physically interesting thing here
> A result that reproduces across two structurally different conservative field theories is the
> signature of something **structural rather than fitted**. This is the project's best evidence that
> it has found a property of a *class* of nonlinear field theories, not an artefact of one.

### TG dual-substrate — **real, converged, and its direction is an input**

The chain `S_state → T → G → A(G) → δφ` produces a genuine short-range attraction:
`⟨F_R⟩ = −5.28e-05` at N=80/L=20, **3.15% shift** under a 1.95× grid refinement and 25% larger box,
an exactly-zero `off` null, working sign control, and now **independent confirmation to 0.34%**.

But:

- The force couples to **4.5%** of the node's energy.
- Its **mass dependence is not measurable** — the axis is confounded with morphology.
- The whole effect is **linear response** at `εG ~ 7e-5`.
- **Its direction is a runtime flag** (`a_sign = +1`, labelled "theory"). Flip it, get equally
  converged repulsion.
- The `T ⇄ G` loop **does not close at long times** (`TG_STATE_LOAD_LONG_TIME_DRIFT_UNRESOLVED`).

### Gravity-D spatial mirror — **non-Newtonian, demoted**

No 1/r², no shell theorem, no universal free fall; later demoted to a finite-width wave force.

---

## 4. The new holistic picture

### The perception before this session

*"We have a converged emergent gravity-like attraction. We are characterising it and climbing a
maturity ladder toward a gravity analogue."*

### The perception now

**The project's scientific asset is not the gravity work.**

What it actually has is (a) an **exceptionally well-verified simulation** of a 16-parameter nonlinear
field theory, and (b) a genuinely interesting **emergent-structure result in the conservative
substrates** — localized objects that move inertially, interact, bind, and obey a two-body law that
is *the same in two different substrates*.

The gravity branch, which has absorbed the most effort (75 of 183 runs), has produced a real
converged force whose **direction is assumed rather than derived** and whose mass dependence is
**not currently measurable**. That is a well-posed open problem, not a result about gravity.

Three shifts follow:

1. **Verification is excellent (A); validation is absent (F).** 16 free parameters against one
   external comparison yielding a binary sign law. Verification work *feels* like progress and the
   project is very good at it — which is exactly why it can absorb unlimited effort while the
   evidential position does not move. **You can verify forever and learn nothing about the universe.**

2. **The sequencing was wrong, not the goals.** The temporal↔geometric long-time stability problem is
   not an obstacle on the road to a gravity analogue — **it is a goal in its own right, and it is that
   analogue's precondition.** Promoting it makes closing the drift a result rather than a chore.

3. **The most valuable unexploited asset is in the transport branch, which has been dormant for six
   weeks.** The **π/2 phase-force crossover** is dimensionless, parameter-free, substrate-independent,
   and has published experimental data against it (Mitschke & Mollenauer 1987, sign law 7/7 on the
   usable series). It is the strongest candidate the project has for a *parameter-free quantitative
   prediction* — the one thing that would convert any of this into evidence about nature.

> [!important] The single sentence that captures the change
> The project spent this session discovering that its instrument is sound, its most-worked branch is
> less conclusive than it looked, and its least-worked recent branch contains the best shot at a real
> external test.

### What would honestly be claimable today

> A self-consistent, carefully verified 16-parameter nonlinear field model exhibiting emergent
> localized structures, inertial transport, a substrate-independent phase-dependent two-body law, and
> a short-range attraction whose direction is a modelling input — with no established quantitative
> contact with measurement.

That is a real contribution to computational field theory. **It is not yet evidence about the
universe**, and no further simulation of the same kind would make it so.

### What is genuinely working and should not change

The instrument-integrity ledger; the willingness to retract (four verdicts after C2.6); preregistering
unwelcome outcomes (P2 preregistered its own disagreement case); the standing posture enforced across
~200 documents; mirror-first with a frozen baseline. **That discipline is why this record was worth
analysing at all** — most projects at this stage cannot be audited because nothing was recorded
honestly enough.

---

## 5. Where that leaves the next move

Per [[ACTION_PLAN_2026-08]]: close Phase 0 (~1 day), then **the sign problem** — a perturbative
hand-derivation in the `ε_G → 0` limit, which is cheap, needs no GPU, and is essentially exact
because the system is strictly in linear response. Then **the π/2 crossover as a parameter-free
prediction**. And in parallel, the one threat that cannot be fixed from inside: **an external
reviewer**.

---

## What changed as a result

- **Code / model changes:** none from this document; it is a synthesis of eleven commits listed in §1.
- **Verdicts changed:** none. P1, P1-a and P2 changed how existing verdicts must be *read*; no verdict
  string was altered.
- **What was done next, and why:** [[ACTION_PLAN_2026-08]].

## Issues raised

| # | issue | status | resolved by |
|---|---|---|---|
| 1 | Force sign is an input, not derived | OPEN | Action plan Phase 1 |
| 2 | No parameter-free dimensionless prediction | OPEN | Action plan Phase 2 |
| 3 | Stability sector closed without a second reading | OPEN | [[Branch - Stability - Review Checklist]] |
| 4 | Catalog does not reference documents (evidence→status link missing) | OPEN | |
| 5 | No independent human reviewer | OPEN | Action plan Phase 4 |

## Associated docs

- [[ACTION_PLAN_2026-08]] — what to do next
- [[SYSTEM_PRESSURE_TEST_AND_METHOD_REVIEW]] · [[META_ANALYSIS_BRANCH_PROGRESS]] · [[INFRASTRUCTURE_AND_ORCHESTRATION_ASSESSMENT]]
- [[gravity_maturity/TG_P1_EVIDENCE_RECONCILIATION]] · [[gravity_maturity/TG_P1A_ENERGY_OBSERVABLE]] · [[gravity_maturity/TG_P2_MIDPLANE_STRESS_FLUX_RESULTS]]
- [[IRER_MASTER_HYPOTHESIS_CATALOG]] — authority on live status · [[Main branch]] · [[HOME]]

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

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
