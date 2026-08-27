---
tags: [record, plan, index]
date: 2026-08-25
status: proposed
---

# Action Plan — from instrument work to validation

Author: Claude, 2026-08-25. Synthesises [[SYSTEM_PRESSURE_TEST_AND_METHOD_REVIEW|the pressure
test]], [[META_ANALYSIS_BRANCH_PROGRESS|the meta-analysis]],
[[INFRASTRUCTURE_AND_ORCHESTRATION_ASSESSMENT|the infrastructure assessment]] and the P1/P1-a/P2
closures into one ordered plan.

---

## The situation in one paragraph

**The instrument phase is finished.** P1 classified the observable, P1-a supplied the energy
normalisation, P2 confirmed the force against a fully independent estimator to 0.34%, and the CI,
results index and provenance fixes now protect all of it. There is no remaining instrument question
blocking the science. **That means more instrument work would now be avoidance**, because the
pressure test established the real position: 16 free parameters, one external comparison yielding a
binary sign law, and a force whose *direction is a runtime flag rather than a derived result*.

Everything below is ordered by that.

---

## Phase 0 — close what is open (this week, ~1 day total)

Small, and all of it removes ambiguity that would otherwise contaminate later work.

| # | action | why now | effort |
|---|---|---|---|
| 0.1 | Work the [[Branch - Stability - Review Checklist]] | Already started (A1 ticked). The entire dissipative-substrate closure rests on 35 runs over three days with **no second reading**. If any of the five NULLs is wrong, later branch choices were made on bad information. | 2–4 h |
| 0.2 | **Decide the orchestrator's fate** | `orchestrator/` + `queue_runtime.db` (1,850 rows, all pointing at a dead `G:\` drive) will mislead the next reader into thinking it is live. Revive deliberately or delete. Either is fine; leaving it is not. | 15 min |
| 0.3 | Add *What changed as a result* to the ~10 load-bearing documents | **0 of 213 documents have one.** Start with the ones the catalog cites. Query: `SELECT * FROM v_docs_needing_consequences` | 1–2 h |
| 0.4 | Delete the stale second virtualenv (`venv/` alongside `.venv/`) if unused | A live footgun given the numpy-ABI breakage already seen | 5 min |

**Decision point:** if 0.1 overturns any stability NULL, stop and re-plan — the branch structure
would be wrong.

---

## Phase 1 — the sign problem (the scientific priority)

> [!danger] This is the single question that decides whether there is a result here
> `a_sign = +1` is a **runtime flag**, labelled "theory" in the source. Flip it, get repulsion,
> equally converged, equally antisymmetric. Until the direction is *derived*, the attraction is a
> modelling input elaborated by a very well-verified simulation — which is threat **T3**, and
> falsification condition **F2** in [[Main branch]].
>
> P2 removed the last excuse: the estimator is now trusted, so the free sign can no longer be
> attributed to instrument uncertainty.

### 1a. Perturbative hand-derivation of the two-body force — **start here**

Derive `F(sep)` analytically in the `ε_G → 0` limit and compare to the measured value.

**Why this is the right first move:** P1 measured `A_well_min ≈ 0.99993`, i.e. `ε_G·G ~ 7e-5`. The
system is **strictly in linear response** — the regime where perturbation theory is not an
approximation but essentially exact. This is the cheapest possible route to a *derived* sign, and it
needs no GPU at all.

| | |
|---|---|
| Method | Linearise `A = 1 + ε_G G`, solve the screened `T`/`G` sector for a static source, evaluate `F_R = −c²∫(∂ₓA)|∇φ|²` against the known Q-ball profile |
| Success | The sign falls out of the algebra rather than the flag, **and** the magnitude matches the measured −5.7e-05 to within the linear-response error |
| Failure that teaches something | The derived sign is ambiguous → the model genuinely does not fix it → **F2 is triggered** and that is a real, publishable negative result |
| Effort | 1–2 days of analysis; no compute |

### 1b. Variational reformulation (GAP-4) — only if 1a is inconclusive

Fixes the sign by construction and yields Noether conservation laws as free independent checks.
High value, high effort. **Do not start this before 1a**: if the perturbative derivation settles the
sign, the reformulation becomes a nice-to-have rather than a necessity.

---

## Phase 2 — name a dimensionless ratio (L1)

The pressure test promoted this from long-horizon to **the load-bearing goal of the programme**,
because it is the only thing that converts simulation into evidence about nature.

> [!important] There is already a strong candidate, and it has been hiding in plain sight
> **The π/2 phase-force crossover.**
>
> - It is **dimensionless** — a pure angle.
> - It is **parameter-free** — π/2 is not tuned, it falls out of the two-body structure.
> - It is **substrate-independent** — the catalog records the same two-body law in *both* NLS and
>   KG, which is exactly the signature of something structural rather than fitted.
> - **There is published data.** [[external_validation/empirical_comparison_branch/EMP-V1-MM_CLAUDE_REVIEW|Mitschke & Mollenauer 1987]]
>   measured the phase-dependent soliton interaction and the wide series follows the sign law 7/7.
>
> That comparison currently carries `WEAK_CLOSE_ANALOGUE_SIGN_LAW_SUPPORTED_NO_QUANTITATIVE_CLAIM`.
> The upgrade path is to stop asking "does the sign match" and start asking **"where exactly is the
> crossover, in our model and in their data, and do the two numbers agree?"** A crossover *angle*
> is a quantitative, parameter-free prediction. That is a different class of claim from a sign law.

**Concrete steps:**

| # | action | effort |
|---|---|---|
| 2.1 | Enumerate every dimensionless, parameter-free quantity the model produces. Candidates: the π/2 crossover; the anti-phase transmission threshold (`v ≤ 0.45c`); `E_grad/E` at the attractor; the VK boundary `dQ/dω = 0`; the capture threshold bracket (0.565, 0.660) | half a day |
| 2.2 | For each, ask: *is there a measured counterpart, and can we predict it without tuning?* Rank by riskiness — the most falsifiable first | half a day |
| 2.3 | Run the crossover measurement precisely in both substrates, with error bars | 1 short GPU run |
| 2.4 | Re-open the M&M comparison against the crossover **angle**, not the sign | 1 day |

**This is the highest-value work in the entire plan.** One parameter-free number that matches is
worth more than every characterisation run to date.

---

## Phase 3 — close the transport branch honestly

Three threads are OPEN and the branch has been silent since 12 July: C3 asymmetric-velocity energy
budget, C3 captured-remnant long-time fate, the C2 bounce→transmit boundary.

Per the meta-analysis, sectors here close by **burst-exhaustion rather than by decision**. Either
schedule these or write an explicit *"not pursuing, because…"* into
[[Branch - Transport - Index]]. **Either is acceptable; drift is not.** Note that Phase 2 may
reopen this branch anyway, since the crossover lives in it.

Effort: 30 min to decide; days if pursued.

---

## Phase 4 — external review (T10)

The most under-mitigated critical threat, and **the only one that cannot be fixed from inside**.
Every reviewer to date is Jake or an AI agent, and AI agents share priors and failure modes.

Options, roughly in order of accessibility:

1. **University supervisor or a physics postgrad** — the cheapest real adversary available, and
   this is a student project with access to one.
2. **Publish the verification work.** The conservative-substrate results (`v = 2Dk` to 0.9999,
   cross-substrate two-body universality, the collision diagram, the momentum-ledger estimator)
   are a legitimate computational-field-theory contribution **independent of IRER**. Submitting
   them gets genuine referees onto the method without requiring anyone to accept the theory.
3. **Independent re-implementation** from the equations alone.

Effort: hours to ask; months for route 2.

---

## What NOT to do next

Stated explicitly, because each is tempting and each would be avoidance:

- **More characterisation runs** (separation sweeps, phase maps, resolution studies). The
  instrument is trusted now; more of these do not address the sign or the validation gap.
- **P1-b (frozen-reference mass sweep).** Still queued and still reasonable, but P1-a showed the
  coupling fraction itself varies 29% non-monotonically — so a cleaner mass exponent would still
  not be interpretable. Deprioritised, not cancelled.
- **More infrastructure.** A1–A3 are built. The rest of the A-list is nice-to-have.
- **The gravity analogue as a target in itself.** Even a perfect emergent 1/r² from a 16-parameter
  model would not separate *"the theory is right"* from *"a flexible field theory can be made to do
  this."*

---

## Ordering rationale

```mermaid
flowchart TD
    P0["Phase 0 — close what is open<br/><i>~1 day</i>"]:::now
    P1A["1a — perturbative sign derivation<br/><i>1–2 days, no compute</i>"]:::crit
    P1B["1b — variational reformulation<br/><i>only if 1a inconclusive</i>"]:::opt
    P2["Phase 2 — dimensionless ratio (L1)<br/><i>the pi/2 crossover</i>"]:::crit
    P3["Phase 3 — close transport<br/><i>decide, do not drift</i>"]:::soon
    P4["Phase 4 — external reviewer<br/><i>ask this week</i>"]:::soon
    F2["F2 triggered:<br/>sign is an input<br/><b>real negative result</b>"]:::fail

    P0 --> P1A
    P1A -->|sign derived| P2
    P1A -->|ambiguous| P1B
    P1A -->|cannot be fixed| F2
    P1B --> P2
    P2 --> P3
    P0 --> P4

    classDef now  fill:#2d4a5e,stroke:#7fb0cc,color:#fff
    classDef crit fill:#7a2d2d,stroke:#e08080,color:#fff
    classDef opt  fill:#5e5426,stroke:#ccbe6a,color:#fff
    classDef soon fill:#2d5e3d,stroke:#7fcc95,color:#fff
    classDef fail fill:#4a3d6b,stroke:#8a7db8,color:#fff
```

**Phase 1a before Phase 2** because if the sign turns out to be an unavoidable input, that changes
what the crossover comparison *means*. **Phase 4 in parallel** because asking costs an hour and the
answer may take weeks.

---

## What changed as a result

- **Code / model changes:** none yet — this is a plan.
- **Verdicts changed:** none.
- **What was done next, and why:** pending Jake's decision on Phase 0 and 1a.

## Issues raised

| # | issue | status | resolved by |
|---|---|---|---|
| 1 | Force sign is an input (T3 / GAP-4 / F2) — Phase 1 | OPEN | |
| 2 | No parameter-free dimensionless prediction (L1) — Phase 2 | OPEN | |
| 3 | Transport branch OPEN and dormant — Phase 3 | OPEN | |
| 4 | No independent human reviewer (T10) — Phase 4 | OPEN | |
| 5 | Orchestrator dormant, 1,850 stale rows — Phase 0.2 | OPEN | |
| 6 | 0 of 213 documents record their consequences — Phase 0.3 | OPEN | |

## Associated docs

- [[SYSTEM_PRESSURE_TEST_AND_METHOD_REVIEW]] · [[META_ANALYSIS_BRANCH_PROGRESS]]
- [[INFRASTRUCTURE_AND_ORCHESTRATION_ASSESSMENT]] · [[Main branch]] · [[RUN_QUEUE]]
- [[gravity_maturity/TG_P1_EVIDENCE_RECONCILIATION]] · [[gravity_maturity/TG_P1A_ENERGY_OBSERVABLE]] · [[gravity_maturity/TG_P2_MIDPLANE_STRESS_FLUX_RESULTS]]

## Branches

- [[Main branch]]

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `60093e5` (2026-08-27) — *Action plan: instrument phase closed, the sign problem is next*

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Also referenced by (same date or earlier):** [[SESSION_SYNTHESIS_2026-08]]

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
