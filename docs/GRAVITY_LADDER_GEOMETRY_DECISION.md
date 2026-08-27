# Gravity Ladder — Geometry-Contract Decision (post rung A+D + Codex characterization + de-saturation pilot)

**Decision: PAUSE the gravity ladder (Option 3) and return to C3/C2 transport follow-ups, with a documented
re-entry condition.** Neither de-saturation (Option 1) nor C2′ (Option 2) unblocks rung B; the pilot shows the
blocker is deeper than the soft-clip. Not an emergent-gravity claim; cautious language throughout.

## Why Option 1 (de-saturated soft-clip) is ruled out
Diagnostic pilot (`jax_scout/gravity_desaturation_pilot.py`, post-processing the saved rung-A load, no evolution):
recomputing the load's Ω²(ρ) under Codex's de-saturated candidate (β=0.75, window [1e-6,1e8]) removed the cap
(cap_fraction 0.86→0.00) but made the **spatial** cliff ~250× steeper (max radial jump 54.6→13390), not graded:
| geometry | Ω² core | Ω² at r≈1.4 | Ω² ambient |
|---|---|---|---|
| production (β=3, cap 1e6) | 535 | 375767 | ~929000 |
| de-sat (β=0.75, cap 1e8) | 2.1 | 71440 | ~8×10⁶ |
**Root cause:** the cliff is **not** the cap — it is the conformal law `Ω²=(ρ_vac/ρ)^a` diverging at the **near-zero
simulation vacuum** (load core ρ≈1, vacuum ρ≈1e-12 → Ω² spans ~27 orders across the core boundary). Removing the cap
unmasks the divergence. Codex's `graded_fraction=1.0` was for a *uniform ρ-sampling of the curve*, not the load's
actual spatial ρ-distribution — over the real load, the response still cliffs.

## Why Option 2 (C2′ canonical geometry) does not unblock it either
C2′ changes the *conservation form* of the kinetic coupling (divergence-form + metric-variation term) but uses the
**same** conformal factor Ω²(ρ). The saturation/cliff of Ω²(ρ) is unchanged in C2′. C2′ remains the right fix for the
*conservation contract* (a separate item), but it is orthogonal to the gravity-response geometry.

## The real blocker (what a graded gravity-like geometry actually requires)
A gravity-like response needs a **graded, extended Ω² well**, not a boundary cliff. That requires, together:
1. **Vacuum at ρ≈ρ_vac** (not ≈0), so ambient Ω²≈1 and the response is referenced to a real background.
2. **A load denser than ρ_vac** (an overdensity → Ω²<1 well). The rung-A load had ρ_core≈1 < ρ_vac=1.19 — an
   *underdensity*, which the (ρ_vac/ρ)^a law would render as a bump, not a well.
3. **A stable coherent load on that ρ_vac background.** The rung-A `bg=ρ_vac` attempt already failed here — the
   dissipative a\* attractor **dissolves** on a filled background (it lives on a low background by construction).
Satisfying 1–3 simultaneously is a **research sub-project** (a stable overdense structure on a ρ_vac background), not
a parameter tweak. The two zero-cost geometry knobs (de-sat, vacuum-reference) cannot manufacture it.

## Decision + re-entry condition
- **Pause the gravity ladder** (rung B stays blocked). Bank the honest rung-A+D finding: the geometry-density
  channel is *live* (a coherent load produces a geometry-dependent Ω²/T_info response) but the production regime
  yields a **saturation cliff, not a graded potential**, so it is not yet interpretable as gravity-like.
- **Return to C3/C2 transport follow-ups** (the solid, reproduced core): C3 exact-Lorentz-contraction boost
  (v_frac→1) + Q-ball stability; C2 higher-speed collision (critical velocity); optionally C2′ for the conservation
  contract.
- **Re-entry condition (documented for when we resume gravity):** a validated **stable-overdense-load-on-ρ_vac-
  background** regime — the load must (a) sit on a ρ≈ρ_vac ambient, (b) exceed ρ_vac in its core, (c) persist. Only
  then does Ω²(ρ) become a graded well and rung B (neutral probe) become interpretable. This is the prerequisite
  pilot to run before any further gravity rung.

## Codex reproduction pass — accepted, with follow-ups
`PHASE_D_REPRODUCTION_PARTIAL` is accepted. Solid confirmations (independent replication): C2.6 default parity exact,
true geometry-off flat, D/151 regression 0.006511, geometry-off flux ~1e-16, ETDRK4≡RK4 transport; C3 conservation
~1e-13, Q-ball existence, exact linear rotation. Bounded partials (C2.7 R3 / C2.9 quick exceeded time bounds → finalized
artifacts packaged) are fine — the finalized numbers match. Adopt Codex's two hygiene suggestions:
1. Add **short official smoke modes** to C2.7 (`--quick`: R0 + R2 only, skip the N96 R3 hold) and C2.9
   (cap V0 + one static) so future replication is bounded — small harness additions (Claude).
2. **Curate the unique handover Markdown** out of `quantule_viz/outputs/` (e.g. `CONSERVATIVE_C2_HANDOVER_SUMMARY.md`
   and the audit reports) into `docs/` *before* adding `quantule_viz/outputs/` to `.gitignore` — do not ignore the
   tree until the unique MDs are promoted (Codex correctly declined to auto-ignore).

## Guardrails
Read-only geometry diagnostics; no production/default/Hunter/validation changes; no probe run; no gravity claim. The
geometry-contract (soft-clip characterization + C2′) remains an open review item, now with concrete data
(`sweep_runs/PHASE_D_CODEX_REPRODUCTION_20260709_233433/gravity_geometry/`, `sweep_runs/GRAVITY_DESAT_PILOT/`).

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `489fb44` (2026-07-10) — *Gravity ladder geometry decision: PAUSE (de-sat ruled out; blocker is vacuum-rho*
**Revised since:** 4 commit(s), most recently `60093e5` (2026-08-27)

**Harness code changed since it was written:** 5 commit(s) to `jax_scout/`.
  - `06edae5` 2026-08-27 — Fix run provenance: 30% -> 86% of runs now dated
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine
  - `25020bf` 2026-07-22 — dual substrate

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Also referenced by (same date or earlier):** [[FUTURE_WORK_AND_EXTERNAL_VALIDATION_PLAN]]

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
