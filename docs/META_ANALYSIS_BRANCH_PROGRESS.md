---
tags: [record, meta, index]
date: 2026-08-25
status: complete
---

# Meta-Analysis — Branch Progress & Goal Validation

Author: Claude, 2026-08-25. Cross-branch synthesis over the 176 catalogued runs in
[[runs/_INDEX|the run catalogue]], the [[IRER_MASTER_HYPOTHESIS_CATALOG|master catalog]], and the
sector documents. **Desk analysis — no new simulation.**

Purpose: measure progress against the goals Jake states in [[Main branch]], not against the project's
own internal ladders, and say plainly whether those goals still look right.

> [!warning] Standing posture
> No matter claim, no gravity claim, no emergent-physics claim. Nothing below upgrades any verdict.

---

## 1. The evidence base, measured

| | count |
|---|---:|
| catalogued runs | 176 |
| carrying a verdict | 98 |
| **positive** | **21** |
| **negative / null** | **30** |
| mixed | 1 |
| neutral / descriptive | 46 |
| no verdict recorded | 78 |

**A 21:30 positive-to-negative ratio is a healthy sign, not a worrying one.** A project that mostly
confirms itself is usually measuring its own assumptions. The nulls here are load-bearing: they closed
the stability sector honestly and they caught three instrument bugs.

The 78 no-verdict runs are mostly parameter hunts whose output is a distribution rather than a claim.
That is legitimate, but it means **44% of the run corpus cannot be audited by reading a verdict** —
you have to open the note.

---

## 2. The shape of the work — a finding in its own right

| month | gravity | transport | stability | other |
|---|---:|---:|---:|---:|
| 2026-06 | 0 | 0 | **35** | 20 |
| 2026-07 | **72** | **44** | 0 | 2 |
| 2026-08 | 3 | 0 | 0 | 0 |

Each sector is a **short, intense burst that then stops dead**:

- **Stability:** 35 runs across **three days** (Jun 22–24). Never revisited.
- **Transport:** 44 runs across **eight days** (Jul 4–12). Never revisited — its last run is six weeks old.
- **Gravity:** 72 runs in July, then **3 in all of August**.

> [!important] Sectors are closing by burst-exhaustion, not by a completion criterion
> When a burst ends, whatever was still open stays open — and the branch reads as "done" because
> nothing new appears in it. The master catalog still lists **three OPEN transport threads**
> (C3 asymmetric-velocity energy budget, C3 captured-remnant long-time fate, the C2 bounce→transmit
> boundary) on a branch that has had no run since 12 July. Nobody decided to stop; the burst simply ended.

This is the mechanism behind the disorientation at the start of this session. It is fixable with a
completion criterion per branch — see §7.

---

## 3. Baseline (instrument & infrastructure)

**Status: the strongest part of the project, and still the one with the biggest single hole.**

What is solid:
- Frozen Phase C dissipative operator, byte-identical; mirror-first discipline held throughout.
- DC-v1.0 data contract, 16/16 compliant on audit.
- The **instrument-integrity ledger** — three bugs caught by chasing contradictions against known
  identities rather than accepting convenient nulls (C2.6 `D_eff=D/151`; C2.8b peak-tracking giving
  `e=3.21`; C3 boost IC missing the carrier phase). Each reshaped verdicts. Two further classifier
  thresholds were caught in flight during RUN-2.
- Cross-platform determinism evidence (TG-B1S-D reproduced to 7.9e-12 on A100 vs local).

The hole, found this session by [[gravity_maturity/TG_P1_EVIDENCE_RECONCILIATION|P1]]:

> [!danger] The headline observable has never been independently cross-checked
> Every force number across ~75 gravity runs comes from **one estimator**, `F_R`. P1 established it is
> a half-space total body force weighted by **gradient energy only**, and that the mass it would be
> divided by is a `∫|φ|²` charge integral — so `F/M_p` is not an acceleration. The independent
> midplane stress-flux force has been on the plan since the TG contracts were written and is still
> unimplemented.

A validation library that lacks a second estimator for the project's headline quantity is not yet
doing its job. This is the single highest-leverage gap in the baseline.

---

## 4. Branch by branch

### Stability — `main` · [[Branch - Stability - Index]] · 35 runs · **CLOSED, honestly**

**Progress.** A real attractor at ×1.15 cubic gain (`param_a≈0.55`), bracketed to ±0.5%, confirmed
across seed, N=128 and T=144k. Established as a gain/loss balance, not a topological object. The
hi-fi continuation corrected an earlier misreading: nodes are **rotational cores**, not topological
vortices, and stable-sustains/unstable-dissipates is energy balance rather than missing spin.

**The nulls, which are the substance.** Every structural *predictor* failed:

| predictor | outcome |
|---|---|
| log-prime resonance | NULL (0/60) |
| TDA / topological signature | NULL |
| tensor-geometry routing | NO_SUPPORT — the proxy effect was a flat-form artifact |
| bridge/void ratio | invalid (no-bridge = 949) |
| Payan / phase-alignment | NO_SIGNAL (gap −0.08) |
| mobility (kick + static well) | NEGATIVE across 3 morphologies |

**Implication.** We can *find* the attractor and *hold* it. We cannot *predict* it from structure, and
it does not *move*. That is a genuine, well-supported, negative result about the dissipative substrate —
and it is why the project pivoted to conservative substrates. The pivot was correct.

**Risk.** 35 runs in three days, zero verdict strings, and no second reading of any of its 39+36
visual-analysis figures. The sector is closed on the strength of analysis that was never visually
reviewed by a second party.

---

### Transport — `main` · [[Branch - Transport - Index]] · 44 runs · **ANSWERED POSITIVE, then abandoned mid-thread**

**Progress — and it is the project's cleanest success.** After the C2.6 geometry-off bug was found
(`a_coupling=0` did not disable geometry; `D_eff = D/151`), four "pinning/drag" verdicts were retracted
and the re-derivation gave clean results on **two independent conservative substrates**:

- **NLS (C2.7):** true solitons translate at `v = 2Dk` — agreement 0.9999, a Galilean identity check.
- **KG (C3):** VK-stable Q-balls (`dQ/dω = −849`) transport inertially; energy and U(1) charge
  conserved to ~1e-13.
- **Two-body law (C2.9):** static force shows a **π/2 phase crossover** (attract below, repel above);
  collisions **bind rather than scatter**.
- **Cross-substrate universality:** NLS and KG give the same two-body law.
- **Collision diagram (C3, C2.10):** capture is generic; transmission occurs **only** at exact
  anti-phase `Δφ=π` and `v ≤ 0.45c`. The static π/2 force does *not* govern collisions.

**Implication.** The conservative substrates do what the dissipative one could not: they support
localized structures that *move*, *interact*, and *bind*. This is the strongest evidence in the project
that IRER-style first principles produce anything dynamically interesting.

**The problem.** Three threads are still OPEN and the branch has been silent for six weeks. Worse, the
verdict strings visible in the catalogue read as descriptive fragments (`C2_ANTIPHASE_CAPTURE_DOMINATED`,
`C3_CAPTURE_DOMINATED_UP_TO_0.45c`); the **synthesis that turns eight runs into a closed question exists
only as prose in the master catalog.** There is no experiment note in between (see §6).

---

### Gravity — `main` · [[Branch - Gravity - Index]] · 75 runs · **ACTIVE, and weaker than its verdict strings suggest**

**Progress.** Two sub-lines. The Gravity-D spatial mirror characterized a body force
`F = −D ∫ ∇N|∇ψ|² dV` that is explicitly **non-Newtonian** — no 1/r², no shell theorem, no universal
free fall — and was later demoted to a finite-width wave force. The TG dual-substrate ladder
(`S_state → T → G → A(G) → δφ`) reached a converged A-well attraction at N=80/L=20 (3.15% shift over a
1.95× grid refinement and 25% larger box), with an exact `off` null and working sign control.

**Read the verdict sequence, though:**

```
2026-07-16  TG_B2_DYNAMICAL_AWELL_ATTRACTION_CONFIRMED
2026-07-16  TG_B2_DYNAMICAL_SIGN_INCONSISTENT_OR_NULL_ACROSS_SEP     <- same day
2026-07-18  TG_B2_CHARACTERIZATION_COMPLETE
2026-07-19  SOURCE_SCALES_BUT_FORCE_FLAT__RECEIVER_OR_OBSERVABLE_NORMALIZATION
2026-08-24  TG_B2_DEFINITIVE_AWELL_ATTRACTION_CONFIRMED
```

**The same question returns "confirmed" and "null" on the same day.** P1 explains why: the observable
was under-specified, so different framings of the same data gave different answers. That is not
dishonesty in the record — it is an instrument that was not pinned down before it was used.

**What P1 established (2026-08-25):**
- `F_R` is a half-space total body force on **gradient energy only** — one of four terms in the KG
  energy density. `F/M_p` is **not** an acceleration.
- `e_ref`/`q_ref` are **peak normalizers recomputed per mass row**, removing ~0.36 of the source
  exponent by construction (`E_tot ~ M^0.935` → `∫S_state ~ M^0.585`).
- The mass axis is **confounded with morphology** — source width varies 25% non-monotonically.
- The `F ~ M^-0.067` fit has **R² = 0.11**: flat scatter, not a power law.
- The whole effect is a **7e-5 perturbation** (`A_well_min ≈ 0.99993`) — strictly linear response.

**Implication.** The gravity branch has a converged, sign-controlled, short-range attraction **whose
direction is an input rather than an output** (`a_sign = +1` is a runtime flag labelled "theory") and
whose mass dependence is not currently measurable. That is a real result and a well-posed next problem.
It is not a gravity analogue.

**Also unresolved:** `TG_STATE_LOAD_LONG_TIME_DRIFT_UNRESOLVED` still blocks D5 promotion, and the
2026-08-22 D4 review confirmed the larger-box failure is real and specific to that geometry.

---

### Validation — `side` · [[Branch - Validation - Index]] · **thin**

Parity, diagnostics and reproduction runs. Cross-platform determinism is evidenced. But per §3 the
validation *library* Jake set as short-term goal 3 does not yet include a second estimator for the
headline observable. This branch is under-invested relative to its importance.

### Unsorted — `side` · [[Branch - Unsorted - Index]] · 22 runs · **needs triage**

Mostly June-era `CORRECTED_PHYSICS_JAX_SCOUT`, `SUBSTRATE_HUNT`, `FEB_*` and `ADAPTIVE_HUNT` runs that
predate the naming conventions. 45 of the vault's figures live here. Cheap to classify; worth doing
because `SUBSTRATE_HUNT` is where the rotational-core basin work lives.

---

## 5. Progress against Jake's stated goals

Measured against [[Main branch]] directly.

### Short-term

| # | goal | status |
|---|---|---|
| 1 | stable field configuration; visually analyse, measure, **and predict** behaviour via ASTE-hunter-filtered sampling | **HALF.** Stable configuration ✅ (bracketed, multi-seed, multi-N). Prediction ❌ — *every* structural predictor returned NULL. The hunter can find the attractor; nothing predicts it from structure. |
| 2 | long-term way to visually analyse, track and understand dynamics | **CAPABILITY DELIVERED, PRACTICE NOT STARTED.** 486 figures now in-vault with a chronological tracker and paired-reading protocol — but **96 runs await a first reading**. |
| 3 | consolidated library of validation methods | **PARTIAL — the weakest link.** DC-v1.0, the 3-bug ledger and the evidence package exist. The headline force estimator has no independent cross-check. |

### Long-term

| # | goal | status |
|---|---|---|
| 1 | simulate a verified gravity analogue | **ACTIVE, EARLY.** Converged short-range attraction with a sign that is an input. No 1/r², no UFF, no shell theorem. |
| 2 | simulate emergent co-operation | **NOT STARTED — but the primitives already exist, unlabelled.** See below. |
| 3 | identify our scale relative to known structures | **NOT STARTED.** |
| 4 | simulate sequential time via a stable feedback loop between temporal and geometric fields | **ACTIVE — and misfiled.** See below. |

> [!important] Two goals are being worked on without being recognised as such
>
> **Goal 4 (sequential time via a stable T↔G feedback loop) *is* the dual-substrate work.** TG-B1S is
> literally a temporal–geometric feedback loop, and its blocking status —
> `TG_STATE_LOAD_LONG_TIME_DRIFT_UNRESOLVED` — is precisely the statement *"the loop is not yet stable
> at long times."* It is currently framed as an obstacle on the road to goal 1. **It is goal 4, and it
> is a precondition for goal 1**: a gravity analogue cannot be verified on a substrate whose temporal
> loop drifts at long times.
>
> **Goal 2 (emergent co-operation) has its first primitives in the transport branch.** Two-body
> binding, capture-dominated collisions, and the phase×speed diagram (transmission only at exact
> anti-phase) are multi-body interaction results. Nobody has framed them as co-operation work.

---

## 6. The structural gap: a missing middle layer

Three documentation layers exist and one is absent:

| layer | records | exists |
|---|---|---|
| Run note | what one run concluded | ✅ 176 |
| **Experiment note** | **what several runs jointly settled** | ❌ **zero** |
| Catalog row | project-wide status | ✅ |

The synthesis that turned eight C3/C2 runs into "the collision diagram is complete, capture is generic,
transmission only at anti-phase" lives **only as prose inside a catalog table cell**. So does the
transport closure, the stability closure, and the C2.6 retraction chain.

That is why the frontier is hard to reconstruct after a break, and why P4 work got done while P1 sat
open: the reasoning that orders the work is not written down at the level the work happens.
`_templates/experiment-note.md` exists for exactly this and is currently unused.

---

## 7. Do the goals still look right?

**Yes — the goals are sound. The sequencing is not.** Three specific corrections:

1. **Promote goal 4 (sequential time / stable T↔G loop) to an explicit target, ahead of goal 1.**
   It is already being worked on as a blocker. Naming it as a goal makes long-time loop stability
   something to *achieve and measure* rather than something to get past. It is also the honest
   precondition for a verified gravity analogue.

2. **Close short-term goal 3 before extending goal 1.** The midplane stress-flux estimator is the
   validation library's missing keystone *and* the gravity branch's strongest outstanding check. One
   piece of work serves both. This agrees with the campaign's own P2, and with the standing diagnosis
   that *numerical robustness leads theory fidelity*.

3. **Give every branch a completion criterion.** Transport has been silent for six weeks with three
   OPEN threads and no decision to stop. Either close them explicitly (a documented "not pursuing,
   because…") or schedule them. Burst-exhaustion should not be allowed to masquerade as completion.

**What I would not change.** The mirror-first discipline, the frozen-baseline rule, the preregistration
habit, and the willingness to retract. Those produced the 3-bug ledger and the honest nulls, and they
are the reason this record is worth analysing at all.

**One caution.** Short-term goal 1's prediction half has failed comprehensively — five independent
structural predictors, all NULL. That is strong evidence, and it deserves a decision rather than
drift: either declare structural prediction of the dissipative attractor **closed-negative**, or state
what sixth predictor would be different in kind from the five that failed.

---

## 8. Recommended order

| | work | serves |
|---|---|---|
| 1 | **P2 midplane stress-flux estimator** | ST-3 (validation library) + LT-1 (gravity) — one job, two goals |
| 2 | **P1-a energy observable** (small; unblocks remaining P1 rows) | ST-3, LT-1 |
| 3 | **Reframe the T↔G long-time drift as goal LT-4** and give it its own gates | LT-4, unblocks LT-1 |
| 4 | **Write experiment notes** for the three closed arcs (stability, transport, C2.6 retraction) | ST-2, and fixes §6 |
| 5 | **Work the visual review queue** — 96 runs, newest first | ST-2 |
| 6 | **Decide** on structural prediction: closed-negative, or name the sixth predictor | ST-1 |
| 7 | Triage the 22 Unsorted runs | hygiene |

P1-b (frozen-reference mass sweep) stays queued but is no longer top of the list: with `F_R` shown to
be gradient-only and the mass axis confounded, a second estimator is worth more than a better fit on
the first one.

---

## Associated docs

- [[IRER_MASTER_HYPOTHESIS_CATALOG]] — authority on live status
- [[gravity_maturity/TG_P1_EVIDENCE_RECONCILIATION]] — the P1 audit this analysis leans on
- [[gravity_maturity/TG_RECOVERED_CONCEPTS_INTEGRATION_AND_REPRIORITIZATION]] — set the current campaign
- [[EXPERIMENT_TRACKER]] · [[runs/_INDEX]] · [[DOCUMENTATION_METHODOLOGY]]

## Branches

- [[Main branch]] — this document is a cross-branch synthesis and links to it deliberately
- [[Branch - Gravity - Index]] · [[Branch - Transport - Index]] · [[Branch - Stability - Index]]
- [[Branch - Validation - Index]] · [[Branch - Unsorted - Index]]

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `96b324c` (2026-08-25) — *Meta-analysis: branch progress and goal validation*
**Revised since:** 11 commit(s), most recently `bd93089` (2026-09-12)

**Harness code changed since it was written:** 5 commit(s) to `jax_scout/`.
  - `caf61af` 2026-09-10 — Build HUD items 1+2: snapshot writer and universal field renderer
  - `06edae5` 2026-08-27 — Fix run provenance: 30% -> 86% of runs now dated
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Also referenced by (same date or earlier):** [[ACTION_PLAN_2026-08]], [[INFRASTRUCTURE_AND_ORCHESTRATION_ASSESSMENT]], [[Main branch]], [[SESSION_SYNTHESIS_2026-08]], [[SYSTEM_PRESSURE_TEST_AND_METHOD_REVIEW]], [[gravity_maturity/TG_P2_MIDPLANE_STRESS_FLUX_RESULTS]]

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
