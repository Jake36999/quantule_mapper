# Recovered-Concepts Integration & Reprioritization

Author: Claude (primary review), 2026-07-17. Integrates the parallel concept re-extraction workflow (the
`docs/theory_synthesis/irer_archive/` corpus, `CROSSMAP_CONCEPTS_VS_RESULTS.md`, `CONCEPT_IMPLEMENTATION_AND_GAP_ANALYSIS.md`,
and its Codex continuation) into the main consolidation plan (`TG_CONSOLIDATION_AND_VALIDATION_FRAMEWORK.md`,
`TG_B3_FOUNDATIONS_DESIGN_CONTRACT.md`). Jake retains conceptual authority; this is review + reconciliation, not a
verdict change. Posture unchanged: no gravity/matter/time-dilation claim.

## 1. Independent verification (I checked the load-bearing claim, not just the extraction)

MC-1 — "the Gravity-D 'failures' are IRER's *predicted* force behaviour" — is the claim everything downstream leans on.
I verified it against the **raw** May-2025 transcript (`D:\memory_bank\bank 1\2025_May.txt`), not the re-extraction:
- L30533: *"Force Saturation Limits — Could a high-RD region 'cap out' and temporarily resist further field coupling?
  Tied to field rigidity."*
- L30629: *"Is there a critical RD value beyond which additional gradient no longer increases force? (Analogous to
  elastic yield points...)"*

These predate the 2026 Gravity-D runs and match, point for point, what those runs measured (saturation cliff;
gradient-energy-weighted force; ~54.6% probe-structure dependence; finite range). **The reframe is real and grounded,
not post-hoc.** The extraction workflow is faithful.

## 2. Assessment of the parallel workflow

**Quality: high.** It is checkable against our own catalog and code, cites specific provenance, keeps verdicts and
posture intact, and separates confidence tiers. The Codex continuation's evidence-class table and failure register are
genuinely useful guardrails. Three items are load-bearing and change the plan; the rest are good reframes that earn
their keep by generating new tests.

**Load-bearing (changes what we do next):**
1. **χ_out is the likely CAUSE of the active D4 blocker** (GAP 3). My own box-dependence analysis concluded the L=12
   failure was "absorber/comparability/dynamical, not physical box-fragility." The gap analysis supplies the mechanism:
   with **no physical radiation channel, the box boundary is the only energy sink**, so a long-time result *must* be
   box-dependent. This promotes χ_out from a foundations nicety to **the candidate fix for the D4 blocker.**
2. **The gravity re-entry target is miscalibrated** (MC-1, verified §1). The paused production-gravity gate demands a
   *graded/Newtonian* load response — a theory IRER never claimed. The faithful test characterizes the **saturation
   cliff as a critical-RD yield-point observable**, not a bug to remove (and de-saturating made it *worse*, consistent
   with a yield point). This reframes the entire paused sector.
3. **Evidence-class separation** (Codex continuation) — adopt as standing rules (§4).

**Good reframes that generate tests (moderate; must stay tests, not relabels):** MC-2 (DII = stability⟂transport),
MC-3 (π/2 = Payan alignment; add an alignment diagnostic), MC-4 (anti-phase node = Mirrored-Collapse), MC-5 (FMIA
routing nulled in the wrong sector), MC-6 (c_emergent = temporal limb, dated to May-2025).

**Labeling discipline (Jake + Claude, binding).** The recovered-concept docs occasionally state matches too definitely
("the failures *are* IRER's predicted behaviour"), which risks overfitting. Corrected framing, applied throughout:
**an observed behaviour or failure point is a *candidate* predicted mode — a potential/predicted failure, not a
confirmed prediction.** The value of a match is diagnostic: *does the theory suggest a way to avoid the failure, or to
develop the system past it?* MC-1's reframe is legitimate **only because it generates falsifiable tests** (yield-point
onset RD + scaling; does the 54.6% probe-dependence *track* Payan/∇φ alignment, MC-3) that could still fail. Never let
"IRER predicted non-universality" become an all-purpose excuse to relabel a null as a success. A one-line correction
note has been added at the top of `CROSSMAP_CONCEPTS_VS_RESULTS.md`; the MC-section prose should be read through it.

## 3. Reconciliation with the existing plan (what's new vs what overlaps)

| item | already in my plan? | recovered-concept change |
|---|---|---|
| R_coh rate source (GAP 1) | yes — B3 contract F1 (re-enable R_relax) | **confirmed independently** — gap analysis also identifies `R_relax` as the closest rate-of-resolution proxy. No change; higher confidence. |
| N_t lapse (GAP 2) | yes — B3 contract F2 | **added constraint**: fix the G1 clock instrument *first* (clock modes migrate toward source); and the migration may be signal, not just failure — deliberate test. |
| χ_out (GAP 3) | yes — B3-2 emission experiment | **PROMOTED**: a smaller *diagnostic-first* version (shell-flux + relief accounting on the D4 config) is now a near-term D4-attack, ahead of the full merger/emission experiment. |
| ledger / VAR-1 (GAP 4) | yes — B3 ledger + Phase F | unchanged; fallback suspect if D4 drift survives χ_out. |
| node formation (GAP 5) | noted as inserted-assumption | **elevated**: identified as *the same missing piece* as the paused production-gravity re-entry gate. |
| saturation-as-yield (MC-1) | not in plan | **NEW strategic reframe** of the paused sector — retarget re-entry from "graded/Newtonian" to "critical-RD yield-point." |
| Payan-alignment diagnostic (MC-3) | not in plan | **NEW cheap Phase-C add**: alignment readout on the two-node/collision harnesses; test if probe-dependence tracks it. |
| A-hill vs A-well branch convergence | flagged in framework | **elevated**: boundedness is validated on the A-hill control branch, attraction on the A-well branch — a hardened, box-independent A-well campaign to converge them is the real milestone. |

**Net:** the parallel workflow does not overturn the framework — it (a) confirms the B3 foundations direction from an
independent angle, (b) reprioritizes χ_out toward the active D4 blocker, (c) adds a strategic reframe of the paused
gravity sector, and (d) adds cheap in-harness characterization tests (alignment, yield-point, phase-mirroring).

## 4. Standing guardrails adopted (from the Codex continuation — now binding)

- **Body-force robustness ≠ secular orbital binding.** The TG-B2 body-force pass does not establish a settled two-node
  orbit/capture; the secular-trajectory observable remains a separate, open method problem.
- **A source is not "resolution activity" until it passes its own semantics gate.** `S_state` = state load (do not
  rename); `R_relax` must re-pass TG-S semantics on the B3 branch before being called a resolution-rate source.
- **A-hill boundedness is not A-well theory validation.** Keep the branch labels explicit; harden A-well separately.
- **No direct temporal-language for a geometry-only frequency shift.** The +2.15e-6 shift is geometric until a direct
  N_t limb passes.
- **Static nulls stay local to the static sector.** FMIA routing is NULL-in-static, UNTESTED-in-transport — not
  globally refuted.
- **Predicted non-Newtonian / finite-range / saturating / structure-dependent responses may be IRER-native signatures**
  — but each must be tied to a falsifiable test, and ordinary-physics claims stay bounded.

## 5. Updated priority stack (supersedes the framework's ordering for the near term)

1. **χ_out diagnostic-first, on the D4 config** (attacks the active blocker): add named shell-flux `chi_out_{φ,T,G}` +
   exported-energy ledger to a B1S larger-box row; does relief accounting reduce/explain the box-dependence? → queue
   `CL_TG_CHIOUT_D4_ISOLATION`.
2. **Phase R robustness** (already queued, `CL_TG_R_ROBUSTNESS_CAPSULE`) — proceeds in parallel; cheap.
3. **MC-3 alignment diagnostic** — add Payan/∇φ readout to the TG-B2 two-node harness; test probe-dependence↔alignment
   (cheap, in-harness) → queue `CL_TG_MC3_ALIGNMENT_DIAGNOSTIC`.
4. **B3 foundations** (R_coh + N_t + full emission) — the design contract stands; B3-0 polarity derivation next, with
   χ_out now informed by item 1.
5. **Strategic (design-only): re-entry retarget** — rewrite the gravity re-entry gate around the yield-point/finite-range
   force structure (MC-1) rather than Newtonian benchmarks; a note under the re-entry plan, no runs.

## 5b. BLOCKER CORRECTION (2026-07-17, Jake caught a conflation)

The priority stack above put "χ_out on the D4 config" at #1 to "attack the active D4 blocker." **On verification that
is misdirected — there are two distinct blockers and they were conflated:**

1. **Production saturation cliff** — the *first* gravity-ladder attempt's blocker (production `Ω²(ρ)`, PAUSED sector).
   This is the OLD blocker; MC-1 reframes *this* (cliff = yield-point). It is **not** the dual-substrate system.
2. **D4 larger-box frequency-scale failure** — on the **TG-B1S A-*hill* scaffold** branch, frequency/momentum
   observable. Real, but it validates the *control-sign scaffold*, which the semantic baseline reclassified as
   "not the model." **Not the live A-well direction.**

**Verified state of the *current* (A-well, TG-B2) direction** (Phase R capsule
`cl_tg_r_robustness_capsule_20260717_112558`): the A-well body force is **grid-converged (N96 ✓) and dt-converged ✓**;
the **larger-box sentinel CRASHED for a trivial config reason** — L=20 at N=64 → dx=0.31 → Q-ball Petviashvili solve
fails (`validated Q-ball solve failed`, return code 1, 3.8 s). So the live direction's **box-robustness is UNTESTED,
not passed** (the Codex-continuation "passes larger-box" was overstated) and **not** scientifically failed either.

**Consequence:** the χ_out-attacks-D4 run targets a stale scaffold-branch issue. The correct immediate check was the
A-well larger-box sentinel — **which the robustness v2 capsule (`cl_tg_r_robustness_capsule_v2`, 2026-07-17) has since
run and PASSED:** `r1_box_L20_N96_sep3` gives `<F_R_well> = −5.32e-5` vs baseline `−5.45e-5` — **2.45% drift, box-
INDEPENDENT**; full Phase-R verdict `TG_R_ROBUSTNESS_PASS` (N96, dt/2, box-L20, and ε_G-linear 0.5×/1×/2× all pass).

**Resolution:** the current A-well direction has **no box blocker** — it is box-independent, grid/dt-converged, and
linear in ε_G. So χ_out is **not** a blocker fix; it stays a **B3 foundations item** (the theory's relief/radiation
limb), valuable for the long-time/formation story but not urgent. The two "blockers" (production cliff; A-hill D4
box-dependence) are both off the live path. `CL_TG_CHIOUT_*` demoted to foundations; `CL_TG_R_BOX_SENTINEL_RERUN`
marked DONE (already covered by v2). **Phase R is complete and PASSED** — the two-node A-well attraction is now a
hardened, robust result, and the next work is Phase C characterization (what the force *is*), not blocker-clearing.

## 6. Boundaries

Integration/reprioritization only. No verdict, solver, production, or frozen-registry change. All recovered-concept
connections are hypotheses-to-test or reframes-to-verify, not confirmations. Jake holds conceptual authority; the
extraction corpus is the source-discovery layer; this doc ties it into the run plan.

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `25020bf` (2026-07-22) — *dual substrate*
**Revised since:** 4 commit(s), most recently `60093e5` (2026-08-27)

**Harness code changed since it was written:** 4 commit(s) to `jax_scout/`.
  - `06edae5` 2026-08-27 — Fix run provenance: 30% -> 86% of runs now dated
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine

**Later documents that cite this one** — the downstream consequences:

- [[gravity_maturity/TG_P1_EVIDENCE_RECONCILIATION]] &middot; `2026-08-25`

**Also referenced by (same date or earlier):** [[RUN_QUEUE]]

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
