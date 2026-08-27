---
tags: [branch, main, index]
branch_kind: main
date: 2026-08-25
---

# Main branch

## The overarching goal

Use the IRER theory to build a digital model of an empty pocket of the universe and simulate whether,
by following first principles as described by IRER, **emergent fields, forces and structures can
emerge** — and whether those emergent results follow dynamics observed in the real world and in
empirical experiment.

A **side branch** ([[Side branch]]) is work that does not directly serve that question: exploratory
probes, robustness testing, or scoping which part of the system an algebraic function governs.
**Only a branch index links back to this note** — see [[DOCUMENTATION_METHODOLOGY]] §3.

> [!info] Provenance of the goals below
> Jake wrote a first pass as **examples**, off the top of his head, to illustrate the shape of the
> thing. This is a refinement of those examples by Claude (2026-08-25), informed by
> [[META_ANALYSIS_BRANCH_PROGRESS]]. **Jake retains conceptual authority** — treat anything here that
> misstates the intent as a drafting error, not a decision. Original wording is preserved in §5.

---

## 1. Instrument goals — what has to be true for any result to mean anything

These are not physics. They are the conditions under which physics claims are believable, and the
project has learned each of them the hard way.

| id | goal | done when | status |
|---|---|---|---|
| **I1** | **Every headline observable has at least two independent estimators.** | No claim rests on a single measurement path. | **IN PROGRESS** — the gravity force had exactly one estimator for ~75 runs ([[gravity_maturity/TG_P1_EVIDENCE_RECONCILIATION\|P1]]); the midplane stress-flux second estimator is now implemented and running. |
| **I2** | **Every observable's physical dimension is stated before it is used.** | No quantity is reported whose units are unestablished. | **PARTIAL** — P1 had to establish retroactively that `F_R` is a gradient-energy-weighted half-space force and that `F/M_p` is not an acceleration. |
| **I3** | **Every null is checked against an identity before being believed.** | A null that contradicts a symmetry is treated as instrument fault until proven physical. | **HELD** — this rule caught all three bugs in the integrity ledger. Keep it. |
| **I4** | **Every visual conclusion has two independent readings.** | The paired-reading queue is not chronically empty on one side. | **CAPABILITY BUILT, UNUSED** — [[EXPERIMENT_TRACKER]]: 96 runs with figures, 0 second readings. |
| **I5** | **Every branch has a completion criterion.** | No sector goes quiet without a recorded decision. | **NOT MET** — transport has been silent for six weeks with three OPEN threads. |

*Rationale: the single most valuable artifact this project has produced is the instrument-integrity
ledger ([[IRER_MASTER_HYPOTHESIS_CATALOG]] §10). Three bugs, each of which had already changed
published verdicts. I1–I5 are that lesson written as goals.*

## 2. Core goals — the actual physics questions, in dependency order

| id | goal | done when | status |
|---|---|---|---|
| **C1** | **A stable, characterised field configuration.** | A configuration exists, is bracketed in parameter space, survives seed/resolution/long-time variation, and its stability mechanism is understood. | **ACHIEVED** — the a\* attractor at ×1.15 cubic gain, bracketed ±0.5%, seed/N128/T144k confirmed, explained as gain/loss balance. |
| **C2** | **Structures that move, interact and bind.** | Localized structures translate inertially, obey a stated two-body law, and the law is substrate-independent. | **ACHIEVED** — NLS solitons at `v = 2Dk` (0.9999); KG Q-balls transport with E and charge conserved ~1e-13; the same two-body law on both substrates; collision diagram mapped. **This is the project's strongest result.** |
| **C3** | **Sequential time: a temporal↔geometric feedback loop that is stable at long times.** | The T↔G loop runs without secular drift, and long-time behaviour is resolution- and box-independent. | **ACTIVE, BLOCKING** — `TG_STATE_LOAD_LONG_TIME_DRIFT_UNRESOLVED`; the D4 larger-box failure is real and specific. **C3 gates C4** — see the note below. |
| **C4** | **A gravity analogue that is verified rather than asserted.** | A force whose *direction is derived, not chosen*; a stated falloff law; behaviour under mass variation that is measurable on an unconfounded axis. | **EARLY** — converged short-range attraction, but its sign is a runtime flag (`a_sign`), its mass dependence is not currently measurable, and the falloff is exponential-like, not a power law. |
| **C5** | **Composite behaviour — structures whose combination has properties the parts do not.** | A bound composite persists longer, or behaves differently, than either constituent alone, and the mechanism is stated. | **PRIMITIVES EXIST, UNRECOGNISED** — two-body binding, capture-dominated collisions, and the anti-phase transmission window are exactly this class of result. Nobody has pursued them as such. |

> [!important] C3 is a precondition for C4, and this was not previously stated
> A gravity analogue cannot be verified on a substrate whose temporal–geometric loop drifts at long
> times. The long-time drift is currently filed as an *obstacle* in the gravity branch. It is a goal
> in its own right, and it comes first. Promoting it changes what "progress" means: closing the drift
> is a result, not a chore.

## 3. Long-horizon goals

| id | goal | what would count |
|---|---|---|
| **L1** | **Locate our scale.** | The model has no units. Scale identification means finding a **dimensionless ratio** the model predicts that can be compared with a measured dimensionless ratio in nature. Until such a ratio is named, "what scale are we at" is not yet a well-posed question. |
| **L2** | **External legibility.** | The IRER core is expressed in standard formalism well enough that a physicist outside the project can check it. Partly underway — [[FUTURE_WORK_AND_EXTERNAL_VALIDATION_PLAN]] maps onto Karpman–Solov'ev/Gordon, Coleman/VK, Madelung, Visser acoustic metrics. |
| **L3** | **A quantitative match to something measured.** | Not "resembles" — a number from the model reproducing a number from an experiment or dataset, with the comparison stated in advance. |

## 4. What would make us stop

A goal set without this is a wish list. Explicit falsification conditions for the main branch:

- **F1** — If C3 cannot be made stable at long times under any admissible parameter choice, the
  dual-substrate construction is wrong and C4 is unreachable through it.
- **F2** — If the emergent force's direction can never be *derived* — if it remains a chosen sign
  under every formulation — then it is a modelling input, not an emergent phenomenon, and C4 fails on
  its own terms.
- **F3** — If no dimensionless ratio can be named for L1, the project cannot connect to measurement,
  however internally consistent it becomes.

*None of these is currently triggered.* F2 is the one to watch: a variational formulation would fix
the sign by construction, which is why the non-variational loop (GAP-4) matters more than it looks.

## 5. Original wording (Jake, 2026-08-25) — preserved

> The practical short term goals, as far as i can tell are as follows.
> 1) create a stable field configuration that we can visually analysis the dynamics of, measure the
>    systems behaviour and find a way to predict them by testing a large sample group filtered through
>    the ASTE hunter to optimise results.
> 2) Create a long term way to visually analyse, track and understand these dynamics so we can
>    theorise further than the first principles with more conviction.
> 3) create a consolidated library of validation methods to test experiments.
>
> long term goals
> 1) simulate a verified gravity analogue
> 2) simulate emergent co-operation
> 3) identify the scale of which we are working at in respect to known structures.
> 4) simulate sequential time by simulating a stable feedback loop between temporal and geometric fields.
> and so on... these are just of top of my head.

**What changed and why.** Short-term 1 was split: the *stable configuration* half succeeded (C1), the
*prediction* half failed comprehensively — five independent structural predictors returned NULL, which
is a finding rather than an incomplete task, and is the subject of
[[Branch - Stability - Review Checklist]]. Short-term 2 and 3 became **I4** and **I1–I2**, because they
are conditions on believability rather than physics questions. Long-term 4 was promoted to **C3** and
placed *before* long-term 1 (**C4**), since it gates it. Long-term 2 became **C5** and was moved out of
"long-term" because its primitives already exist in the transport results. Long-term 3 became **L1**
with a sharper success criterion. §4 is new.

---

## Branches

| branch | kind | state |
|---|---|---|
| [[Branch - Gravity - Index]] | main | active — C3/C4 |
| [[Branch - Transport - Index]] | main | C2 achieved; three threads OPEN and dormant |
| [[Branch - Stability - Index]] | main | C1 achieved; closed — [[Branch - Stability - Review Checklist\|under review]] |
| [[Branch - Validation - Index]] | side | under-invested relative to I1–I2 |
| [[Branch - Unsorted - Index]] | side | needs triage |

## Associated docs

- [[META_ANALYSIS_BRANCH_PROGRESS]] — the analysis this refinement is based on
- [[IRER_MASTER_HYPOTHESIS_CATALOG]] — authority on live status
- [[DOCUMENTATION_METHODOLOGY]] · [[EXPERIMENT_TRACKER]] · [[HOME]]

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `292fc96` (2026-08-25) — *Obsidian vault: run catalogue, experiment tracker, documentation methodology*
**Revised since:** 4 commit(s), most recently `9517d9f` (2026-08-26)

**Harness code changed since it was written:** 4 commit(s) to `jax_scout/`.
  - `06edae5` 2026-08-27 — Fix run provenance: 30% -> 86% of runs now dated
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Also referenced by (same date or earlier):** [[ACTION_PLAN_2026-08]], [[INFRASTRUCTURE_AND_ORCHESTRATION_ASSESSMENT]], [[SYSTEM_PRESSURE_TEST_AND_METHOD_REVIEW]], [[gravity_maturity/TG_P2_MIDPLANE_STRESS_FLUX_RESULTS]]

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
