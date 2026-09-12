# Concepts Applied vs Concepts Missing in the TG Gravity-Maturity Runs — and How the Gaps Could Move the Results

**Author:** Claude, 2026-07-17. **Scope:** reads the recent gravity-maturity runs — individual **kinetic** (G0 / Gravity-D
spatial medium force), individual **temporal** (G1 clock calibration), and the recent **dual substrate** (TG-B1S state-load
loop + TG-B2 two-node force) — and maps them against the IRER concept inventory (re-extracted in
`docs/theory_synthesis/irer_archive/` + `CROSSMAP_CONCEPTS_VS_RESULTS.md`). It answers two questions: **which concepts are
applied where and how**, and **which concepts are not yet implemented/tested and how they could change the results.**

**Posture (unchanged).** Nothing here is a gravity/matter/time-dilation claim or a verdict change. The runs' own bounded
labels stand. This is an interpretive gap analysis to steer the next steps. Sources are cited inline (all under
`docs/gravity_maturity/`).

**Codex continuation:** `CONCEPT_IMPLEMENTATION_AND_GAP_ANALYSIS_CODEX_CONTINUATION_20260717.md` adds an implementation-facing
follow-up focused on evidence-class separation, recovered predictions, and failure modes that may be theory-expected.

---

## 1. The three configurations, and the concept each embodies

| config | what it is (equation-level) | IRER concept embodied | current result |
|---|---|---|---|
| **Individual — KINETIC** (G0 / Gravity-D) | `i∂_tψ = −D∇·(N(x)∇ψ)`; force `F = −D∫∇N·|∇ψ|²dV` | **Gradient-derived informational force** (Concept 20: "gravity-like effects from cumulative RD gradients") acting through the **spatial coefficient** = informational-manifold-topology→propagation | **Frozen / reproduced.** Finite-width, gradient-energy-weighted, **probe-structure-dependent (54.6%)**, short-range. Explicitly **non-Newtonian** (no 1/r², no shell theorem, no universal free fall). `G0_FOUNDATION_FROZEN`. |
| **Individual — TEMPORAL** (G1) | supplied temporal field → local KG clock; test `ω_local ≈ m·N_t(R)` | **Time as chronology of resolution / c_emergent varies with load** (Concept 24: local clock rate depends on informational load) | **Characterized negative.** A real objective **temporal differential is detected** (near ω=6.436 vs far ω=6.466 vs flat ω=12.03), but the **local clock could not be calibrated**: the far clock eigenmode **drifted toward the low-lapse source region** (localization error 4.09) and two clock constructions disagreed on shift direction. `TEMPORAL_CLOCK_CALIBRATION_FAILED`. |
| **Dual substrate** (TG-B1S + TG-B2) | `state=(φ,π,T,V_T,G,V_G)`; chain **S_state → T → G → A(G) → δφ**; T,G are damped 2nd-order wave fields with symmetric `−κTG` exchange; `A=exp(∓ε_G G)` | **The full TG feedback loop**: node load → temporal load (T) → geometric response (G) → propagation coefficient (A) → force back on the field. Realizes "dense load → slower chronology → attraction." | **Robust-but-weak backreaction**; single-node modal **frequency shift +2.15e-6** (geometric stiffening + partial profile relaxation, FC-1); **D4 numerical validation FAILED** at larger box (5/6 rows pass; L=12 fails the frequency-scale gate). **Two-node (TG-B2): theory-faithful A-well produces confirmed dynamical inter-node ATTRACTION** (−5.45e-5, 100% of settled samples, sign-controlled). Status `TG_STATE_LOAD_LONG_TIME_DRIFT_UNRESOLVED`. |

**One structural nuance up front.** The polarity is now a **theory commitment**: dense load → slower chronology → *lower*
propagation coefficient (**A-well**) → attraction (`TG_SEMANTIC_BASELINE…` §1). The **A-well attraction lives in the TG-B2
branch**; the **frozen TG-B1S that the D4 campaign validates is the A-*hill* "anti-throttling numerical scaffold"** — a
bounded-feedback existence proof with the *control* sign. So today the *boundedness* is validated on the control-sign model
and the *gravity-relevant attraction* is on a separate, less-hardened branch. These need to converge.

---

## 2. Concept → implementation map (which concepts are applied, where)

From the postulate audit (`TG_SEMANTIC_BASELINE_AND_POSTULATE_AUDIT.md`), graded against the theory docs:

**Applied / implemented:**
- **Ψ_OIW substrate** → validated KG Q-ball (dE/E~5e-14). ✓
- **ρ resonance-density** proxy. ✓
- **T chronology-load field** (time-as-resolution) → dynamically sourced 2nd-order damped wave. ✓
- **G geometric field** (manifold deformation) → dynamically sourced via T, conservative symmetric `−κTG` exchange. ✓
- **A_s(G) spatial coefficient** (manifold topology → propagation) → with theory-faithful **A-well** polarity (TG-B2). ✓
- **Closed loop** Ψ→source→T→G→Ψ → runs end-to-end; bounded backreaction; two-node attraction. ✓ (was a "not-yet" item — now closed)
- **Gradient-derived force** (Concept 20) → the Gravity-D body-force law is the measured interaction. ✓
- Positive coefficient maps, determinism (no stochastic collapse), attractor-selection method. ✓

**How the recovered concepts show up in the *results* (cross-links to `CROSSMAP_CONCEPTS_VS_RESULTS.md`):**
- The A-well attraction *is* "dense→slower→attraction" = **Concept-20 gradient-derived force + throttling** realized dynamically.
- Its **short-range, screened, non-1/r² character** is exactly **MC-1**: IRER predicted a *finite-range, non-universal,
  probe-dependent* force, not Newtonian gravity. The Gravity-D 54.6% probe-dependence and the TG-B2 near-field falloff are
  the predicted behaviour, not failures.
- The single-node **frequency shift under load** is a **mass-like modal effect** — the flavour of **MC-2 (Dynamic
  Informational Inertia:** load changes the structure's inertial/modal signature).
- The G1 **clock eigenmode migrating toward the source** is an un-named "clock falls toward the load" behaviour — a
  gravitational-signature-shaped result that currently reads only as a *calibration failure*.

---

## 3. Concepts NOT yet implemented/tested — and how each could move the results

Five gaps (the audit's two foundational gaps + two half-gaps + one inserted assumption). For each: the concept, why it's
absent, and **the specific result it could change.**

### GAP 1 — Resolution-RATE source (R_coh). The source is a *load*, not a *rate*.  [could change the drift, the saturation, and the adiabaticity]

**Concept.** IRER's most specific object is the **resolution-activity source** `R_res` — the *rate* of coherence-transition/
collapse events ("time as chronology of **resolution**"; PAS = "readiness to **resolve**"). The theory's candidate is
`R_coh = [−∂_t K_phase]_+` (load from *becoming* more phase-locked) — a **dynamic rate**.

**What's implemented instead.** `S_state` = energy+charge **state load** (static field *presence*), classified by TG-S as
`NODE_STATE_LOAD` and explicitly *not* a completed-resolution rate. `R_coh` was never operationalized; `L_lock` failed its
gate, `R_relax` is disabled (`TG_SEMANTIC_BASELINE…` row 3; equation audit).

**How it could change the results:**
- **The saturation character (ties to MC-1).** A *load*-based source is exactly the object that **saturates** — it is the
  same design choice that produced the gravity **saturation cliff** in the production geometry. A *rate*-based source pulses
  with activity and does not saturate the same way. If the goal is to understand the cliff, R_coh is the alternative source
  that would tell you whether the cliff is intrinsic to load-sourcing or to the geometry.
- **The adiabaticity problem (the TG-B2 dynamical difficulty).** The two-node dynamical test was hard because the T/G
  response time ≈ the node **breathing** time, so the A-well never adiabatically forms (`TG_B2…` instrument-3). A **resolution-
  rate** source is *intrinsically tied to the breathing* (each breath = resolution activity), so it could **phase-lock T/G to
  the node's own rhythm** instead of fighting it — potentially converting the oscillating loop force into a clean secular one
  without needing to cool the pair.
- **The unresolved long-time drift.** A constant load sources a near-static T; the slow drift then has no natural dynamical
  driver and reads as numerical. A rate source makes T genuinely time-varying, which could either *explain* the drift as
  physical breathing-driven modulation or remove it — either way it decides the `TG_STATE_LOAD_LONG_TIME_DRIFT_UNRESOLVED`
  question that the current source cannot.

### GAP 2 — Temporal lapse N_t(T) on the substrate. There is no direct time-dilation channel.  [the temporal sector's whole result changes]

**Concept.** IRER Concept 24: **c_emergent varies with local load** — `∂_tφ = N_t(T)·Π`, i.e. temporal load directly
retimes the field. This is the actual "gravitational time-dilation" analogue.

**What's implemented instead.** Code-verified: the dual loop has **no N_t on φ**. T reaches the node **only through geometry**
(T→G→A_s). Temporal throttling was validated in *separate* mirrors (C-series, TS bridge) but is **not wired into the working
loop**; the doc warns `N_t = A_s` is a *separate* hypothesis, not an identity. The G1 attempt to calibrate a local clock in a
supplied temporal field **failed** (`TEMPORAL_CLOCK_CALIBRATION_FAILED`).

**How it could change the results:**
- **The frequency shift is currently *geometric*, not *temporal*.** The measured +2.15e-6 shift is A-stiffening (a *spatial*
  effect), bracketed between fixed-profile (0.25×) and relaxed-adiabatic (2.18×) analytic limits (FC-1). Wiring N_t would add
  a **direct temporal contribution** that could sit *outside* that bracket and even oppose or dominate it — the current shift
  is not the theory's time-dilation, only its geometric shadow.
- **The model cannot presently produce time dilation as a primary effect at all** — every "temporal" consequence is laundered
  through the spatial coefficient. Any claim about IRER's temporal sector is, right now, a claim about geometry wearing a
  temporal label.
- **The G1 failure is the warning.** Even in isolation the temporal substrate can't yet host a calibrated local clock (the
  probe migrates toward the source). Adding N_t to the *dual* loop inherits that unsolved measurement problem **plus** the
  dynamical coupling — so the temporal sector needs the G1 clock-instrument fixed *before* N_t wiring will yield a readable
  result. (The probe-migrates-toward-the-source behaviour may itself be the physical signal, mis-read as an instrument fault
  — worth testing deliberately.)

### GAP 3 — Outgoing relief/radiation channel (χ_out). No physical sink; only a boundary sponge.  [likely the *cause* of the D4 box-dependence failure]

**Concept.** The TG loop's final limb is an **outgoing perturbation / relaxation channel** — in the theory's language,
photon-as-minimal-geometric-relaxation-mode and emission-as-source-geometry-projection; relatedly, "novelty from instability"
as a re-seeding/release valve.

**What's implemented instead.** No named radiation field or relief diagnostic. The only energy sink is a **boundary absorber
(a box-edge sponge)**, not a physical channel (`TG_SEMANTIC…` row 9, HALF-GAP).

**How it could change the results — the strongest single point in this report:**
- **The D4 validation failed on box size** (L=12 fails the frequency-scale gate; failure isolated to L, not resolution; the
  box-dependence discriminator says it's "comparability/absorber/dynamical, not physical box-fragility"
  — `GRAVITY_CURRENT_STATUS.json`, `TG_B1S_D4_FINAL_ANALYSIS.md`). **A missing radiation channel is the natural cause of an
  absorber/box-dependent long-time result:** with no physical channel for the node to shed excess energy, the only outlet is
  the box boundary, so the long-time behaviour *must* depend on where that boundary is. The theory says there *should* be an
  outgoing relaxation channel — its absence forces box-dependent boundary dumping.
- **Concrete prediction:** implementing χ_out (a genuine radiating mode with an energy-accounting observable) could convert
  `TG_STATE_LOAD_LONG_TIME_DRIFT_UNRESOLVED` into a **bounded radiating attractor whose frequency shift is box-independent** —
  directly attacking the D4 blocker rather than working around it. This is the most actionable gap→open-problem link here.

### GAP 4 — Variational / least-action generation (VAR-1). The loop is phenomenological, not action-derived.  [bounds claim strength; small spurious-work risk]

**Concept.** FMIA — systems follow **least informational action**; and the theory's rule "every loss term recorded"
(conservation). The generating principle should be a single action.

**What's implemented instead.** The T/G sector is action-derived (`−κTG`), but the **full φ/T/G chain is non-variational**:
two reciprocal channels are missing — a back-force on φ from the S_state→T channel, and a direct G-source from the φ
gradient-energy (equation audit, Finding 1). The energy ledger is an *accounting* diagnostic, not an exact Noether contract.

**How it could change the results:** the missing reciprocal terms mean **energy isn't exactly conserved**, so a small
component of the measured frequency shift/drift could be **non-reciprocal spurious work** rather than physics. Weak-coupling
ledger closure makes this small, but it caps interpretation: no "mature feedback law" claim is defensible until the model is
either derived from one action or its non-reciprocal work is explicitly booked. This is a *claim-strength* gap more than a
result-moving one — but if the D4 drift survives a radiation channel, the next suspect is exactly these unbooked work terms.

### GAP 5 — Node *formation* from OIW dynamics. Nodes are placed, not born.  [this gap *is* the gravity re-entry condition]

**Concept.** Quantules **form** from OIW overlap → RD/PAS rise → resolution (the genesis narrative; the origin-story
"potentiality collapses on a central spot"; interference-peaks→reality-stability; the May-17 **load-capacity/saturation**
picture of what a stable dense load can and cannot hold).

**What's implemented instead.** Nodes are **inserted** as ready-made Petviashvili Q-balls; the formation narrative has only
dissipative-sector analogues (GL rotational cores) (`TG_SEMANTIC…` row 14, INSERTED-not-demonstrated).

**How it could change the results:**
- **The current model tests "how a *placed* load deforms geometry," not "how mass *emerges* and then deforms geometry."** It
  assumes the very thing (a stable overdense load) that the production-gravity **re-entry gate demands** — the paused
  production ladder's re-entry condition is literally "a *validated stable-overdense-load-on-ρ_vac-background* regime"
  (master catalog §8). **GAP 5 and the paused gravity ladder are the same missing piece.**
- Until formation/stability is demonstrated, the model cannot address whether the load itself is **saturation-limited** (the
  May-17 "manifold load capacity" — exceed it and it disperses), which is precisely the behaviour the gravity cliff hints at.
  So GAP 5 and GAP 1's saturation question are entangled: both are about what a *self-consistent* dense load does, not an
  imposed one.

---

## 4. Summary: what's proven, what's assumed, and the highest-leverage next moves

**Applied and working:** the OIW substrate, densities, dynamical T, dynamical G, the closed loop, the gradient-derived force,
and — newly — a **theory-faithful (A-well) dynamical inter-node attraction** (TG-B2). The interaction's **short-range,
non-universal, probe-dependent** character is not a failure; it is what IRER's own force concept predicts (MC-1).

**Assumed / proxied / absent (in priority order for moving the open results):**
1. **χ_out radiation channel [GAP 3]** — likely the *cause* of the D4 box-dependence / unresolved long-time drift; implementing
   it is the most direct attack on the current blocker.
2. **R_coh resolution-rate source [GAP 1]** — replacing the *load* proxy with a *rate* could fix the adiabaticity problem,
   decide the drift, and separate the saturation-from-sourcing question from the saturation-from-geometry question.
3. **N_t temporal lapse [GAP 2]** — without it the model has **no primary time-dilation channel**; the current frequency shift
   is geometric, not temporal. Needs the G1 clock instrument fixed first.
4. **Node formation [GAP 5]** — the same missing piece as the paused production-gravity re-entry gate; also where the
   saturation/load-capacity concept lives.
5. **Variational model [GAP 4]** — bounds claim strength; the fallback suspect for any drift that survives GAP-3.

**Two structural flags for the team:**
- The **D4-validated dual substrate (TG-B1S) carries the *control* (A-hill) sign**; the **gravity-relevant attraction is on the
  separate TG-B2 (A-well) branch.** Boundedness and attraction are currently validated on different branches — converging them
  (a hardened, box-independent, A-well dual-substrate campaign) is the real next milestone.
- The **G1 clock probe migrating toward the source** may be a mis-read *signal* (clock responding to load), not just an
  instrument fault — a deliberately-designed test of that would inform both the temporal gap and the re-entry story.

**Cross-references.** Recovered-concept detail and the six missed-connection analyses (MC-1 saturation-as-feature, MC-2 DII,
MC-3 π/2 = Payan alignment, MC-4 anti-phase node, MC-5 FMIA routing untested-in-transport, MC-6 c_emergent = temporal-lapse
limb) are in `docs/theory_synthesis/irer_archive/CROSSMAP_CONCEPTS_VS_RESULTS.md`. Run-level detail is in the cited
`docs/gravity_maturity/` files. No verdict, solver, or posture is changed by this analysis.

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `25020bf` (2026-07-22) — *dual substrate*
**Revised since:** 10 commit(s), most recently `e42b5bb` (2026-09-11)

**Harness code changed since it was written:** 6 commit(s) to `jax_scout/`.
  - `caf61af` 2026-09-10 — Build HUD items 1+2: snapshot writer and universal field renderer
  - `06edae5` 2026-08-27 — Fix run provenance: 30% -> 86% of runs now dated
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine
  - *…and 1 more.*

**Later documents that cite this one** — the downstream consequences:

- [[SYSTEM_PRESSURE_TEST_AND_METHOD_REVIEW]] &middot; `2026-08-25`
- [[gravity_maturity/DERRICK_SCALING_AND_TARGET_TRIAGE]] &middot; `2026-09-12`

**Also referenced by (same date or earlier):** [[RUN_QUEUE]], [[gravity_maturity/CONCEPT_IMPLEMENTATION_AND_GAP_ANALYSIS_CODEX_CONTINUATION_20260717]], [[gravity_maturity/TG_RECOVERED_CONCEPTS_INTEGRATION_AND_REPRIORITIZATION]]

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
