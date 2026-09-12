---
tags: [index, home]
---

# IRER Quantule Mapper — Documentation Vault

Vault root is `docs/`. Everything below is a note; wikilinks and backlinks work across all layers.

> [!warning] Standing posture
> **No matter claim. No gravity claim. No emergent-physics claim.**
> Mirror-first: `jax_scout/` is the experimental surface; the frozen Phase C dissipative operator
> stays byte-identical. Verdicts here are historical unless
> [[IRER_MASTER_HYPOTHESIS_CATALOG|the master catalog]] says otherwise.

## Start here

| | what it is |
|---|---|
| [[IRER_MASTER_HYPOTHESIS_CATALOG]] | **The authority on live status.** Every tested hypothesis across the project — CONFIRMED / FALSIFIED / NULL / RETRACTED / OPEN / PAUSED — plus the instrument-integrity ledger and the open frontier. |
| [[RUN_QUEUE]] | What is queued, running, and done. Lane assignment and launch notes. |
| [[runs/_INDEX\|Run Catalogue]] | One note per simulation run, with plots and figures. Generated from `sweep_runs/`. |
| [[EXPERIMENT_TRACKER]] | **Chronological experiment → result tracker**, and the visual-review queue. |
| [[DOCUMENTATION_METHODOLOGY]] | How this vault is built: note kinds, branch discipline, backlink categories, image rules. |
| [[DOC_LINEAGE]] | **What changed after each document** — review queue of dead-end and untracked docs. |
| [[SYSTEM_PRESSURE_TEST_AND_METHOD_REVIEW]] | **Adversarial review of the method** — dependency graph, threat register, what the data could mean. |
| [[INTEGRATED_PLAN_2026-09]] | **THE OPERATIVE PLAN** — folds the action plan, resource triage, HUD RFC and research findings into one ordered list. |
| [[SESSION_SYNTHESIS_2026-08]] | **Where the project stands** — what changed, what it revealed, and the physics as now understood. |
| [[RESOURCE_LIBRARY_ASSESSMENT_2026-09]] | External resource sprint reviewed — what to take, decline, and in what order. |
| [[ACTION_PLAN_2026-08]] | **What to do next, in order** — the instrument phase is closed; the sign problem is the priority. |
| [[INFRASTRUCTURE_AND_ORCHESTRATION_ASSESSMENT]] | Orchestration, CI, results storage and RAG — what to build and what not to. |
| [[META_ANALYSIS_BRANCH_PROGRESS]] | **Where every branch stands and whether the goals still hold** (2026-08-25). |

## Current campaign

**TG dual-substrate — factor apart & close contracts.** Three independent review audits converged on
one diagnosis: *numerical robustness leads theory fidelity.* Do these **before** any new source branch
or broad sweep.

- **P1 — reconcile existing evidence.** Audit fixed-vs-recomputed `e_ref`/`q_ref` across all mass/sep
  rows; expose per-row source integral, mediator amplitude/gradient, probe susceptibility, raw force,
  `F/M_p`, breathing amplitude, uncertainty. **Classify what `F_R_well` actually is** (total force /
  density integral / average / acceleration) — this is currently undetermined.
- **P2 — close the planned contracts.** Validate dynamical T/G profiles against the analytic screened
  solution; implement the **independent midplane stress-flux** force; close the non-variational
  energy/work ledger.
- **P3 — factor source vs probe.** Asymmetric 2×2 sweep → `F ∝ M_s^α M_p^β`. Do *not* retune coupling
  to improve an exponent.
- **P4 — only then** targeted characterization: far-field, phase-isolated alignment, binding energy,
  separate lapse/yield gates.

> [!tip] Where the campaign actually stands (2026-08-25)
> **P1 is substantially closed** — see [[gravity_maturity/TG_P1_EVIDENCE_RECONCILIATION]].
> The audit questions are answered; two instrumentation items are queued (`P1-a` energy observable,
> `P1-b` frozen-reference mass sweep). P1 downgraded several earlier framings: `F/M_p` is **not** an
> acceleration, and the mass-scaling exponents are **not interpretable as mass scaling** (the mass axis
> also varies node morphology by 25%).
> **P2 is CLOSED (2026-08-25):** the midplane stress-flux estimator is implemented and confirms the
> body force to **0.340%** — see [[gravity_maturity/TG_P2_MIDPLANE_STRESS_FLUX_RESULTS]]. `F_R`
> measures what it was believed to measure. Every P1 downgrade still stands, and with the instrument
> now trusted, **GAP-4 (the non-variational loop) is the sharpest open problem**: the force's sign is
> still a runtime flag rather than a derived result.

## The document layers

1. **Catalog** — [[IRER_MASTER_HYPOTHESIS_CATALOG]]. Status of everything.
2. **Theory synthesis** — [[theory_synthesis/README|theory_synthesis/]]. Concept translation
   IRER→standard physics, maths traceback, proof ledger, architecture rationale, results synthesis.
   Includes the [[theory_synthesis/irer_archive/00_ARCHIVE_MASTER|IRER archive]] (May–Aug 2025
   transcript re-extraction) and the v9→v10 delta report.
3. **Evidence package** — [[evidence_package/README|evidence_package/]]. Claims → artifacts →
   verdicts, traceable. Two-layer: tracked indexes point at local gitignored run data.
4. **Future work** — [[FUTURE_WORK_AND_EXTERNAL_VALIDATION_PLAN]]. External legibility and
   quantitative validation, V1–V7.
5. **Run catalogue** — [[runs/_INDEX]]. Per-run notes with embedded plots, figures and renders,
   built for secondary visual review. Chronological view: [[EXPERIMENT_TRACKER]].

## Branches

Per [[Side branch]], **only a branch index links back to [[Main branch]]** — run and experiment notes
link to their branch index instead. See [[DOCUMENTATION_METHODOLOGY]] §3.

| branch | kind |
|---|---|
| [[Branch - Gravity - Index]] | main |
| [[Branch - Transport - Index]] | main |
| [[Branch - Stability - Index]] | main |
| [[Branch - Validation - Index]] | side |
| [[Branch - Unsorted - Index]] | side |

Templates for new notes live in `_templates/`: [[_templates/experiment-note|experiment]],
[[_templates/result-note|result]], [[_templates/visual-review|visual review]],
[[_templates/branch-index|branch index]], [[_templates/doc-of-record|doc of record]].

## Sectors

- `gravity_maturity/` — the TG ladder (G0 → G1 → TG-S → TG-B1S → TG-B2), 51 notes. Includes
  [[gravity_maturity/CONCEPT_IMPLEMENTATION_AND_GAP_ANALYSIS|the gap analysis]] (GAP-1…GAP-5) and
  [[gravity_maturity/TG_RECOVERED_CONCEPTS_INTEGRATION_AND_REPRIORITIZATION|the reprioritization]]
  that set the current campaign.
- `external_validation/` — empirical comparison, generated metrics, artifact bundles.
- `codex_conservative_c2_campaign_archive/` — ~50 promoted Codex campaign reports.
- `past_variants/`, `legacy_colab_variants_pre-deployment_ R&d/` — superseded, kept for the record.

## Reading rules

- **A verdict in a result note is historical.** Only the master catalog says what is currently live.
- **Nulls and retractions are the map.** Whole branches (prime-resonance, TDA, routing, Payan) came
  back NULL, and that is what closed the stability sector honestly.
- **One bug rewrote a sector.** `PHASE_D_C2_6_GEOMETRY_OFF_BUG_REPORT` found that `a_coupling=0` did
  not disable geometry (`D_eff = D/151`); it retracted four "pinning" verdicts and forced the C2.7
  re-derivation that confirmed clean transport at `v = 2Dk`.
- **RFCs are intentions, not results.** A design contract may never have been executed.

## Rebuilding the run catalogue

```bash
.venv/Scripts/python.exe tools/build_run_catalogue.py
```

Safe to re-run: hand-written **Review notes** sections in each run note are preserved.

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `292fc96` (2026-08-25) — *Obsidian vault: run catalogue, experiment tracker, documentation methodology*
**Revised since:** 13 commit(s), most recently `bd93089` (2026-09-12)

**Harness code changed since it was written:** 5 commit(s) to `jax_scout/`.
  - `caf61af` 2026-09-10 — Build HUD items 1+2: snapshot writer and universal field renderer
  - `06edae5` 2026-08-27 — Fix run provenance: 30% -> 86% of runs now dated
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
