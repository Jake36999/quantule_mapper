# TG Semantic Baseline & Postulate-Implementation Audit

Author: Claude (primary), 2026-07-16. Written after the TG-B2 direction settlement
(`TG_B2_DEFINITIVE_AWELL_ATTRACTION_CONFIRMED`). Two jobs: (1) state what changed *semantically* in how we reason
about the system; (2) audit Jake's claim — *"there aren't any foundational postulates within IRER that we haven't
implemented; everything after is emergent"* — **against the documentation**
(`docs/theory_synthesis/IRER_TEMPORAL_GEOMETRIC_FEEDBACK_LOOP.md`, the TG-S/TG-B1S/TG-B2 records), not against memory.

---

## 1. What changed semantically (the new baseline for reasoning)

1. **The polarity is now a theory commitment, not a knob.** Dense load → slower chronology → *lower* propagation
   coefficient (A-well) → attraction. This is fixed by IRER's own throttling logic and is now dynamically realized.
   The frozen TG-B1S A-hill is reclassified: it is the *anti-throttling numerical scaffold* — still a valid bounded-
   feedback existence proof, but no longer "the model." Sign choices downstream (any new coupling) must trace to
   this commitment.
2. **"Limbs" → a working loop.** Before: the sims implemented pieces (throttling here, medium force there). Now the
   chain **node state → T → G → A → force on another node** runs end-to-end dynamically. The sentence *"the model's
   geometric sector mediates a screened, attractive, inter-node interaction"* is now available and evidenced. This
   changes the register of every downstream question from "can the loop do X?" to "what does the loop's X look like?"
3. **Instrument semantics: measure forces with force observables.** For weak forces on breathing solitons,
   half-space momentum and masked-COM are *not* direction meters (proven: the bare-drift calibration flips sign
   between separations; COM is reshaping-contaminated). Force-density observables (body force, stress flux) are the
   meaningful ones. This is now a standing methods rule, not a lesson.
4. **Quasi-static reasoning is a directional guide, dynamics is the quantitative authority.** The static calc got
   the sign and falloff shape right and the magnitude wrong by ~2.3× ; adiabaticity demonstrably fails (T/G response
   time ~ node breathing time) yet the time-averaged dynamics still realizes the quasi-static direction. Use static
   estimates to design experiments, never to settle them.
5. **Emergence framing.** With the loop implemented, binding structure, stability basins, radiation/relief channels,
   and scaling laws are **emergent questions to be measured under fixed equations** — matching the theory doc's own
   rule ("attractor selection, not agency"; no hard-coded 'choose the stable route').
6. **Two-lane runtime reality.** The A100 ran the definitive campaign ~8× faster than the 1080. Heavy
   characterization belongs on the Colab lane; the 1080 is for scouts, validation, and overnight single campaigns.

---

## 2. Postulate-implementation audit (checked against the docs)

Verdict on the claim first: **mostly right, with two genuine foundational gaps, one half-gap, and one inserted
assumption.** The loop doc's "not yet explored" list is now partially stale — TG-B1S/TG-B2 closed several of its
items — but not all of them, and the doc itself names what is still open.

| # | IRER foundational element (per loop doc) | status in the dual-substrate model | evidence |
|---|---|---|---|
| 1 | Ψ_OIW substrate (node field) | **IMPLEMENTED** — validated KG Q-ball substrate (NLS mirror also exists) | C3 machinery; TG-B1S baseline gate dE/E~5e-14 |
| 2 | ρ resonance-density proxy | **IMPLEMENTED** | throughout |
| 3 | **R_res[Ψ] — resolution-activity source** | **GAP (proxied).** Implemented source is `S_state` = energy+charge *state load* — TG-S classified it `NODE_STATE_LOAD`, deliberately *not* a completed-resolution rate. The doc calls R_res "the unresolved central object" and warns a density-like source "only says field presence creates temporal load" — exactly what S_state is. The theory's candidate sources (continuous coherence-transition `R_coh=[−d_tK_phase]_+`; threshold `R_thr`) were **not operationalized**: `L_lock` failed its TG-S separation gate; `R_relax` passed semantics but is disabled | TG-S run + doc §Resolution Source Problem |
| 4 | T chronology-load field, dynamically sourced | **IMPLEMENTED** (2nd-order damped wave, α_T·source; matches the doc's scaffold minus λT³) | TG-B1S |
| 5 | G geometric field, dynamically sourced via T | **IMPLEMENTED** (symmetric −κTG conservative-exchange coupling; stable, κ²<ω_T²ω_G²) | TG-B1S + equation audit |
| 6 | **N_t(T) temporal factor acting on Ψ** | **GAP (absent).** Code-verified: the dual-substrate model has *no* temporal lapse on φ — T reaches φ **only through G→A_s**. The doc's KG scout prescribes `∂_tφ = N_t(T)Π`; it also warns `N_t = A_s` is a *separate* hypothesis. Temporal throttling was validated in separate mirrors (C-series; TS bridge) but is **not wired into the working loop**; the G1 clock-calibration attempt failed (`CHARACTERIZED_NEGATIVE`) | grep-verified; TS bridge; GRAVITY_CURRENT_STATUS |
| 7 | A_s(G) spatial coefficient acting on Ψ | **IMPLEMENTED** — now with the theory-faithful A-well polarity (TG-B2 branch) | TG-B2 definitive |
| 8 | Closed loop Ψ→source→T→G→Ψ | **IMPLEMENTED** (stale "not yet" item — now closed): bounded backreaction, frequency shift, two-node attraction | TG-B1S, TG-B2 |
| 9 | **χ_out — outgoing relief/radiation channel** | **HALF-GAP.** No *named* radiation field or relief diagnostic; the absorber is a boundary sponge, not a channel observable. The theory's attractor taxonomy (STABLE_NONRADIATING … RUNAWAY_COLLAPSE) is unexplored beyond the single-node STABLE_FREQUENCY_SHIFT classification | doc §Why the node does not become a radioactive core |
| 10 | Conservative-exchange class / loss ledger | **HALF-GAP.** T/G sector is action-derived (−κTG); the full φ/T/G chain is non-variational (audit finding 1) and the doc's rule "every loss term recorded" awaits LEDGER-1 | equation audit addendum |
| 11 | Positive coefficient maps | **IMPLEMENTED** (exp forms) | TG-B1S |
| 12 | Determinism (no stochastic collapse) | **IMPLEMENTED** — no stochastic term anywhere | throughout |
| 13 | Attractor selection without agency | **METHOD ADOPTED**, partially exercised (D3/D4 classifications) | D3/D4 |
| 14 | **Node *formation* from OIW overlap → RD/PAS → resolution** | **INSERTED, not demonstrated** (conservative sector): nodes are *placed* as ready-made Petviashvili Q-balls; the theory's genesis narrative (nodes forming out of OIW dynamics) has only dissipative-sector analogues (GL rotational cores) | C3/TG init paths |
| 15 | Downstream: FMIA bridges, photon-as-relaxation-mode, emission-as-inverse-image | **NOT implemented — correctly classified as emergent hypotheses**, consistent with the claim's second half | doc §Photon/Radiation Hypotheses |

### Bottom line

- **The claim's spirit is confirmed**: the loop's structural postulates (substrate, densities, dynamical T, dynamical
  G, positive maps, closed feedback, determinism, attractor methodology) are all implemented, and the downstream
  phenomena are correctly emergent territory.
- **Two foundational gaps remain, per the theory's own documents**: (i) the **resolution-rate source** — the most
  IRER-specific object in the theory — is currently a state-load proxy (`S_state`), and (ii) the **temporal lapse
  N_t(T) acting on the substrate** is absent from the working loop (temporal effects reach φ only via geometry).
  Plus the half-gaps: no named χ_out channel; non-variational ledger; and node *formation* is inserted, not emergent.
- **Consequence for the emergent program:** results from the current model are statements about the
  *state-load → geometry* limb with the theory-faithful polarity. Before claiming "IRER's loop" is fully implemented,
  gaps (i) and (ii) need either implementation (R_coh source; dual-coefficient N_t+A_s model) or an explicit,
  documented decision that S_state and geometry-only mediation are the intended reading. These are queued as
  foundations-completion items (Phase F in the companion framework doc) — they do not block robustness or
  characterization of the current model, but they bound its interpretation.

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `25020bf` (2026-07-22) — *dual substrate*
**Revised since:** 2 commit(s), most recently `3eb93af` (2026-08-26)

**Harness code changed since it was written:** 3 commit(s) to `jax_scout/`.
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Also referenced by (same date or earlier):** [[RUN_QUEUE]], [[gravity_maturity/CONCEPT_IMPLEMENTATION_AND_GAP_ANALYSIS]], [[gravity_maturity/CONCEPT_IMPLEMENTATION_AND_GAP_ANALYSIS_CODEX_CONTINUATION_20260717]], [[gravity_maturity/TG_B3_FOUNDATIONS_DESIGN_CONTRACT]], [[gravity_maturity/TG_CONSOLIDATION_AND_VALIDATION_FRAMEWORK]]

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
