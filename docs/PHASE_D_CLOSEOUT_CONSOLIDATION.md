# Phase D — Closeout & Consolidation (Transport/Coupling Sector)

**Status in one paragraph.** Phase D asked whether stable IRER structures can move and interact. After correcting a
substrate instrument bug (C2.6) that had invalidated the earlier conservative-transport readings, the answer is
**positive across two distinct conservative substrate families**: first-order NLS solitons translate at exactly the
Galilean velocity, and second-order (Klein–Gordon) Q-balls co-move under a relativistic boost. A hardened two-node
diagnostic gives a trustworthy interaction law (a phase-dependent force with an attract→repel crossover at Δφ=π/2,
and capture-dominated collisions). **None of this is a matter, gravity, or emergent-physics claim** — "transport" and
"interaction" here mean specific measured observables (velocity vs prediction, mass/momentum conservation, separation
trajectories). The gravity-like target is deliberately *not* what Phase D measured; it is scoped separately in
`docs/IRER_GEOMETRY_DENSITY_GRAVITY_RFC.md`.

## 1. The instrument correction (C2.6) — everything downstream depends on it
**`docs/PHASE_D_C2_6_GEOMETRY_OFF_BUG_REPORT.md`.** The geometry soft-clip chain (`_soft_clip_log_with_derivative`,
mirroring `gravity/unified_omega.py`) is a **global log-space tanh squash**, not a boundary clamp: it maps the
nominally-flat Ω²=1 to ≈151, so `param_a_coupling=0` did **not** turn geometry off — the covariant correction
cancelled ~99.3% of the kinetic term, and every "pure NLS / geometry-off" run in C2.2–C2.5 silently simulated
**D_eff = D/151**. Quantitative closure: the universal "drag" μ≈0.036 equalled 2·D_eff exactly — the structures were
moving at the correct Galilean velocity of the *bugged* substrate; nothing was ever pinned.
- **Fix:** `Ops.geom_fac` (default 1.0, bitwise-identical; C1 parity re-PASS) + `param_geom_off=True` = true flat.
- **Retracted:** all C2.1–C2.5 *transport* verdicts (the "flow-through pinning," "quasi-soliton fountain,"
  "conservative arc closed negative"). **Independently re-audited by Codex** (RK4 ≡ ETDRK4 transport, D_eff ratio
  0.0065 ≈ 1/151, geometry-off flux ~1e-16) — the fix is confirmed by a second codebase.
- **Not retracted:** Phase C dissipative results (production-consistent), the existence machinery, and the Codex
  adjointness findings. **Flagged (not hot-patched):** the production soft-clip redefines Ω²(ρ) vs the documented
  law — a contract item for the geometry review, relevant to the gravity RFC (the geometry-response is what that
  squash shapes).

## 2. C2.7 — corrected conservative transport: solitons move at v = 2Dk
**`docs/PHASE_D_C2_7_REDERIVATION_RESULTS.md`.** On the fixed substrate: a true stationary soliton (found by
Petviashvili) **translates at exactly the Galilean velocity** v/2Dk = 0.9999 with mass retention 0.9999 at N=96 —
clean coherent transport. The feb/a\* coefficient family itself has **no** localized conservative structure (its
pure-NLS sector is structureless: g_max≈0.23 below the box binding floor). Moving families require **s<0**
(defocusing-quintic saturation) and a **box-compatible D** (soliton width ℓ=√(D/μ) must fit). So conservative
transport is answered positive *for the substrate class*, with a sharp coefficient criterion.

## 3. C2.9 — hardened two-node qualification (supersedes C2.8/C2.8b)
**`docs/PHASE_D_C2_9_TWONODE_ROBUST_RESULTS.md`.** The first two-node run (C2.8, executed by Codex) and its
elasticity follow-up (C2.8b) used **projected-density peak-tracking**, which failed exactly when cores are close or
breathing (unphysical elasticity e=3.21; a false-"repel" static reading). **C2.8b is inconclusive, not negative.**
The rebuild (C2.9) measures velocity from the **mass current**: `v = 2D·(∫Im ψ*∂ₓψ)/(∫ρ)` over a COM-following
window — an integral, no peak identification. Validated: 0.03% on a known soliton; tracks cleanly through mergers.
Result:
- **Static force law:** attract for Δφ<π/2, repel for Δφ>π/2, **crossover at Δφ=π/2** — the canonical cos(Δφ) NLS
  interaction. (This is category-1 "phase-force / coherence" behaviour — **not** gravity; see the RFC.)
- **Collisions:** at closing speeds up to 1.3, both asymmetric cases **capture** — cores merge into a bound state,
  mass and momentum conserved (non-radiative). Incoming speeds measured to sub-percent.
- **No elastic-scattering claim.** "Two cores survive"/"pass-through" language from C2.8 is superseded; the honest
  statement is capture-dominated binding at the tested speeds. Whether a critical velocity for transmission exists
  above this range is the one open collision question.

## 4. C3 — wave-kinetic (nonlinear Klein–Gordon) first pass
**`docs/PHASE_D_C3_WAVE_KINETIC_RESULTS.md`.** A standalone second-order substrate `ψ_tt = c²∇²ψ − m²ψ + g(ρ)ψ`
(its own state + stepper; touches no Phase C path). Gates: **G1** stepper exact (Strang ≡ analytic rotation,
max|Δ|=0); **G3** Q-ball **found** at exactly the C2.7-mapped point (the KG stationary equation *is* the C2 soliton
equation, c²↔D, μ↔m²−w²); **G2** energy + U(1)-charge conserved to **~1e-13** (cleaner than the C2 geometry-on
quasi-conservative branch); **G5** null drift 0. **G4 transport:** after correcting an IC bug (the naive kick lacked
the carrier phase that *is* the momentum), the Q-ball's **density co-moves at the imparted velocity** (v_frac ≈ 0.9,
mass ≈ 0.99) → `C3_INERTIAL_TRANSPORT_SUPPORTED` (qualitative). Loose fidelity = the IC omits the O((v/c)²) Lorentz
contraction — a fixable refinement, not a wall.
- **Reframe:** post-C2.6, C3 is no longer a "rescue" for a transport-incapable C2 (that motivation was the bug
  artifact). Both substrates transport; C3 stands on its own as a **distinct, genuinely-conservative relativistic
  substrate** with exact invariants and Lorentz-covariant boosts. (KG note: the conserved U(1) quantity is charge
  Q=Im∫ψ*π, not ∫ρ; ∫ρ breathes — the two are reported separately, never conflated.)

## 5. What is established vs. what is NOT claimed
**Established (as measured observables):**
- A single conservative structure can be transported (NLS: v=2Dk exact; KG: density co-moves under boost).
- Two structures interact with a phase-dependent force (crossover π/2) and capture at moderate speed.
- The corrected substrate is trustworthy (two-codebase parity; machine-precision conservation in KG).

**NOT claimed (explicitly):** no matter, no gravity, no emergent-physics, no "mass attracts mass." The two-node
phase-force is **nonlinear-wave relational dynamics**, not gravity. Transport fidelity has open refinements (C3
contraction; C2 critical velocity). The gravity-like mechanism IRER targets is a *different observable class* and has
**not** been tested — it is defined and laddered in the companion RFC.

## 6. The three interaction categories (pointer)
To keep future claims clean, `docs/IRER_GEOMETRY_DENSITY_GRAVITY_RFC.md` separates:
1. **Phase-force / coherence** (e.g. the C2.9 two-soliton phase law) — baseline nonlinear-wave dynamics, not gravity.
2. **Charge-like / EM-like** (phase gradients, winding, current, Payan alignment) — not gravity unless separately
   isolated.
3. **Geometry-density / IRER gravity-like** — a large coherent field load creating a *persistent tensor/geometric
   response*, with neutral/phase-averaged probes experiencing altered path availability / capture. **Interaction-
   density and tensor-response first, acceleration last.** This is the actual gravity target, and it is untested.

## 7. Open threads (none blocking, ranked)
1. **Gravity-density ladder** (the RFC) — the substantive next scientific question; the Ω²(ρ) + T_info + LOM
   machinery already exists.
2. **C3 refinement** — exact-contraction boost (v_frac→1), Q-ball stability slope, two-Q-ball interaction.
3. **C2 critical velocity** for collision transmission (higher-speed ladder).
4. **C2′ canonical geometry** (`docs/PHASE_D_C2PRIME_CANONICAL_GEOMETRY_RFC.md`) — resolves the geometry-conservation
   contract; relevant if the gravity ladder needs an exactly-conservative geometry-response.
5. **Codex hygiene/replication** — `quantule_viz/outputs/` gitignore; later re-run of the finalized harnesses.

## Provenance & guardrails
Mirror-only throughout; Phase C dissipative default byte-identical; no production/Hunter/validation changes; no
clipping to force outcomes; every transport/interaction claim tied to a measured observable with conservation
telemetry; instrument bugs (C2.6 geometry, C2.8b tracker, C3 boost IC) documented and their affected verdicts
retracted. Coefficients outside the feb family are labelled exploratory substrate variants, never merged into frozen
baselines.

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `c2f320e` (2026-07-09) — *Phase D closeout consolidation + IRER geometry-density gravity-target RFC*
**Revised since:** 8 commit(s), most recently `bc5b54c` (2026-08-31)

**Harness code changed since it was written:** 22 commit(s) to `jax_scout/`.
  - `06edae5` 2026-08-27 — Fix run provenance: 30% -> 86% of runs now dated
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine
  - `25020bf` 2026-07-22 — dual substrate
  - *…and 17 more.*

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
