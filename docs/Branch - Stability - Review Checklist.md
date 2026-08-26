---
tags: [checklist, review, stability]
date: 2026-08-25
branch: Branch - Stability - Index
status: open
---

# Stability Sector — Review Checklist

> [!important] Why this review exists
> [[META_ANALYSIS_BRANCH_PROGRESS]] found the stability sector was closed on the strength of
> **35 runs across three days** (22–24 June), **none of which carries a verdict string**, and whose
> ~200 analysis figures **have never had a second reading**. The conclusions may well be right — but
> nothing here has been independently checked, and this is the sector everything downstream rests on.
>
> Tick an item only when you have *looked at the evidence*, not when you have read the summary.

- Branch: [[Branch - Stability - Index]] (55 runs) · Tracker: [[EXPERIMENT_TRACKER]]
- Protocol: [[DOCUMENTATION_METHODOLOGY]] §5 — fill your reading block **before** reading mine.

---

## A. The positive claim — does the attractor hold up?

The claim: a real attractor at ×1.15 cubic gain (`param_a ≈ 0.55`), bracketed to ±0.5%, confirmed
across seed, N=128 and T=144k, and explained as a **gain/loss balance** rather than a topological object.

- [x] **A1** — Read the closure argument end to end: [[PHASE_C_FINAL_DOSSIER]] (then [[PHASE_C_DOSSIER]] for the earlier state).
- [ ] **A2** — Check the bracketing actually brackets: [[PHASE_C_GAIN_LADDER_RESULTS]]. Is ±0.5% a resolved bracket or the sampling step?

> [!NOTE] Request
> need visual analysis including vector maps, density maps, heatmaps for energy, with different frames throughout the runs at different T for A2. also, we should schedual a longer run for this test as we need to confirm the breathing state and the spin down trend stopping isnt just a energy or time artefact. 
> also, we are working in 3d space, so top down visuals arent enough, please also render a 3d animation i can view and rotate of the run. 

- [ ] **A3** — Seed independence: [[PHASE_C_OPTION_B_N96_STAGE1_RESULTS]], [[PHASE_C_N96_OVERNIGHT_REVIEW]].
- [ ] **A4** — Resolution independence: [[PHASE_C_MASS_THRESHOLD_N96_VALIDATION]] and [[PHASE_C_MASS_THRESHOLD_N96_SCALED_VALIDATION]]. Do the scaled and unscaled N96 validations agree?
- [ ] **A5** — Long-time survival: [[PHASE_C_T24000_CORE_DELINEATION]], [[runs/PHASE_C_N96_LONGT_CONTROL_20260625_083731]] (7 figures).
- [ ] **A6** — Normalization sanity — this is where P1-style errors hide: [[PHASE_C_MASS_NORMALIZATION_RESOLUTION_AUDIT]], [[PHASE_C_IC_NORMALIZATION_PILOT]]. **Is the mass axis confounded with morphology here too, as it was in gravity?**
- [ ] **A7** — Gate calibration: [[PHASE_C_STABILITY_GATE_CALIBRATION]], [[PHASE_C_GATE_CALIBRATION_SUMMARY]], [[PHASE_C_GATE_V3_BREATHING_BOUND_STATE]]. Were the gates set before or after seeing the data?
- [ ] **A8** — Method parity: [[PHASE_C_METHOD_PARITY_AUDIT]], [[PHASE_C_VALIDATION_STACK_AUDIT]].

## B. The five NULLs — is each one really negative?

These closed the sector. A false null is more costly than a false positive, because it stops work.

- [ ] **B1 · log-prime resonance (0/60)** — [[PHASE_C_VALIDATION_STACK_AUDIT]], [[PHASE_C_RESEARCH_STRATEGY_AND_FALSIFICATION]]. Was the prime-SSE objective retired for being *wrong*, or for being *unmeasurable*? Those have different implications.
- [ ] **B2 · TDA / topological signature** — [[PHASE_C_VALIDATION_STACK_AUDIT]], [[VALIDATION_PATH_RECONCILIATION]].
- [ ] **B3 · tensor-geometry routing → NO_SUPPORT** — [[ANISOTROPIC_METRIC_TENSOR_RFC]], [[FMIA_TRANSFER_DIAGNOSTIC_FINDING]]. The proxy effect was a **flat-form artifact**; confirm the faithful anisotropic metric really was faithful.
- [ ] **B4 · bridge/void ratio invalid (no-bridge = 949)** — [[PAYAN_PHASE_ALIGNMENT_RFC]]. Is the denominator problem fatal, or fixable with a different normalization?
- [ ] **B5 · Payan / phase-alignment NO_SIGNAL (gap −0.08)** — [[PAYAN_DERIVATION_PASSIVE_DIAGNOSTIC]], [[PAYAN_PHASE_ALIGNMENT_RFC]]. The diagnostic was **passive**; a null from a passive probe does not rule out an active coupling.
- [ ] **B6 · mobility NEGATIVE across 3 morphologies** — [[PHASE_C_MOBILITY_ENDPOINT]], [[PHASE_C_KICK_INERTIA_AND_OPERATOR_FINDING]], [[PHASE_C_ADIABATIC_DRAG_MORPHOLOGY_RESULTS]], [[PHASE_C_ADIABATIC_DRAG_DESIGN]].

> [!warning] Cross-check against the instrument ledger
> The transport sector's "no transport" verdicts were **wrong**, caused by the C2.6 geometry-off bug
> (`D_eff = D/151`). Ask of each null above the question that caught that one: *does this null
> contradict a symmetry or identity we can state independently?* If it does, treat it as an instrument
> fault until proven physical. See [[IRER_MASTER_HYPOTHESIS_CATALOG]] §10.

## C. Visual review — the untouched evidence

~200 figures, no second reading. Highest figure-count runs first; each note has a paired-reading block
below its Review-notes marker.

- [ ] **C1** — [[runs/SUBSTRATE_HUNT_20260621_161557]] — **48 figures**. The rotational-core basin.
- [ ] **C2** — [[runs/PHASE_C_VISUAL_ANALYSIS_20260624_161650]] — **42 figures**. Also [[PHASE_C_VISUAL_ANALYSIS_PACK]].
- [ ] **C3** — [[runs/PHASE_C_VISUAL_ANALYSIS_V2_20260624_185426]] — **36 figures**. What changed between V1 and V2, and why?
- [ ] **C4** — [[runs/PHASE_C_N96_CURRENT_CLOSURE_20260625_001621]] — 14 figures. Compare with [[PHASE_C_N96_CURRENT_CLOSURE_ANALYSIS]].
- [ ] **C5** — [[runs/CORE_SAT_PILOT_20260622_190340]] (7) and [[runs/CORE_SAT_HUNT_20260623_004605]] (7).
- [ ] **C6** — [[runs/CORE_SAT_TRACE_COMPARE_20260623_185450]] (6) — trace comparison; see [[PHASE_C_MASS_THRESHOLD_TRACE_COMPARISON]].
- [ ] **C7** — [[runs/CORE_BASIN_20260622_023759]] (6) and [[runs/CORE_SAT_THRESHOLD_BRANCH_ROBUSTNESS_20260624_001934]] (6) — see [[PHASE_C_THRESHOLD_BRANCH_ROBUSTNESS]].
- [ ] **C8** — Gallery: [[runs/Gallery - k6_mid_mass_true_emergence]] (24 images) and [[runs/Gallery - triangle_cupy_screen]] (13). Compare against [[PHASE_C_HIGH_MASS_K1_K6_COMPARISON]].
- [ ] **C9** — FEB basin set: [[runs/FEB_BASIN_TOPOLOGY_20260625_154619]], [[runs/FEB_PARAM_BASIN_20260626_004039]], [[runs/FEB_JOINT_BASIN_20260626_224056]], [[runs/FEB_CORE_DELINEATION_T24000_20260627_175050]] — against [[PHASE_C_FEB_PARAMETER_BASIN_CONSOLIDATED_ANALYSIS]] and [[PHASE_C_FEB_BASIN_RESULTS]].

## D. Known corrections — confirm they propagated

- [ ] **D1** — Nodes are **rotational cores, not topological vortices**; stable-sustains / unstable-dissipates is energy balance, not missing spin. Confirm no surviving document still says "vortex". *(No document currently owns this correction — if it is only in the catalog and in memory, it needs a home.)*
- [ ] **D2** — The hunter was **re-aimed** from prime-SSE to a stability objective: [[HUNTER_REAIM_DESIGN_SPEC]] → [[HUNTER_REAIM_IMPLEMENTATION_NOTES]] → [[HUNTER_REAIM_OFFLINE_RESCORE]] → [[HUNTER_REAIM_REDISCOVERY_RESULTS]]. Rediscovery PASSED (a*×1.15 rank 0) — **but rediscovering a known answer is a weaker test than finding an unknown one.** Is there an unseen-target test?
- [ ] **D3** — Scope: [[PHASE_C_SCOPE_RECONCILIATION]] — does the sector's stated scope match what was actually run?
- [ ] **D4** — [[PHASE_C_COLLAPSE_RUNAWAY_DIAGNOSTIC]] — were the runaway/collapse readings superseded by the hi-fi continuation?

## E. The decision this review must produce

- [ ] **E1** — **Is structural prediction of the dissipative attractor closed-negative?** Five independent predictors failed (B1–B5). Either write that down as a closed negative result, or name a sixth predictor that is **different in kind** from the five — not a variant of them.
- [ ] **E2** — Does the mobility null (B6) mean the dissipative substrate is *finished* as a research object? The conservative substrates (transport) do everything it could not.
- [ ] **E3** — Record the outcome in [[Branch - Stability - Index]] under *Open threads* / *Closed*, and update [[IRER_MASTER_HYPOTHESIS_CATALOG]] if any status changes.
- [ ] **E4** — If any item above overturns a conclusion, add it to the instrument-integrity ledger — that ledger is the project's most valuable artifact and it should keep growing.

---

## Review log

*Record findings here as you work. Date each entry. Disagreements with the existing conclusions are
the point of this exercise — write them down even when unresolved.*

| date | item | finding | changes anything? |
|---|---|---|---|
| | | | |

---

## Associated docs

- [[META_ANALYSIS_BRANCH_PROGRESS]] — why this review was raised
- [[IRER_MASTER_HYPOTHESIS_CATALOG]] — authority on live status
- [[EXPERIMENT_TRACKER]] · [[runs/_INDEX]]

## Branches

- [[Branch - Stability - Index]]

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `eeb8f6d` (2026-08-25) — *P2: midplane stress-flux estimator, stability review checklist, refined goals*
**Revised since:** 2 commit(s), most recently `616fe31` (2026-08-25)

**Harness code changed since it was written:** 3 commit(s) to `jax_scout/`.
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Also referenced by (same date or earlier):** [[Main branch]], [[SYSTEM_PRESSURE_TEST_AND_METHOD_REVIEW]]

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
