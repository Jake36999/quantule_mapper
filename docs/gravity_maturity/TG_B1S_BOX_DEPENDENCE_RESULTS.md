# TG-B1S Box-Dependence Discriminator — Results (SCREEN-1 + FC-box)

Author: Claude, 2026-07-15. Script: `jax_scout/gravity_TG_B1S_box_dependence.py` (CPU-only, read-only).
Run: `sweep_runs/TG_B1S_BOX_DEPENDENCE_20260715_222213`. Complements Codex's `CX_TG_B1S_D4_LARGER_BOX_DISCREPANCY_REVIEW`.

## Question

D4's `larger_box` row (N=56, L=12) failed the frequency-scale gate: measured Δω dropped `−2.16e-6 → −8.52e-7`
(to **39%** of the reference). The isolating variable is **L** (10→12): `grid_refined` already cleared N=56 at L=10.
Does the **local quasi-static screened stiffening mechanism** (FC-1) *predict* this box-size drop, or does the drop
come from something the local model omits?

## Method

Recompute the FC-1 fixed-profile analytic shift `Δω_fp = c²∫(A−1)|∇φ₀|² / (2ω₀∫ρ₀)` at five geometries, holding
**all** physics params + frozen `S0` fixed, changing only (N, L). Two of them are pure resolution controls (same L,
finer dx) to separate an L-effect from a dx-effect.

## Results

| geometry | N | L | dx | ∫(A−1)\|∇φ\|² | analytic Δω_fp | analytic / baseline | measured / baseline |
|---|---|---|---|---|---|---|---|
| baseline | 48 | 10 | 0.208 | 1.810e-4 | +5.398e-7 | 1.000 | 1.000 (PASS) |
| grid_refined | 56 | 10 | 0.179 | 1.810e-4 | +5.398e-7 | 1.000 | 1.000 (PASS) |
| **larger_box** | **56** | **12** | 0.214 | 2.396e-4 | +7.122e-7 | **1.319** | **0.394 (FAIL)** |
| larger_box_fine | 68 | 12 | 0.176 | 2.396e-4 | +7.122e-7 | 1.319 | — |
| baseline_fine | 68 | 10 | 0.147 | 1.810e-4 | +5.398e-7 | 1.000 | — |

## Two clean eliminations

**1. It is NOT resolution / under-resolution.** `larger_box` (dx=0.214) and `larger_box_fine` (dx=0.176) give
**identical** analytic shifts (dx_effect = −0.000); `baseline` and `baseline_fine` likewise identical. The Q-ball
and its gradient-energy overlap are fully grid-converged at these resolutions. So the failure is **purely the L
change**, not the marginally-coarser dx — matching D4's own `grid_refined` pass.

**2. It is NOT the local quasi-static mechanism.** The local mechanism predicts the shift should **rise ~32%**
in the bigger box (the Petviashvili Q-ball solved at L=12 has a slightly larger gradient-energy overlap with the
A-hill: ∫(A−1)|∇φ|² goes 1.81e-4 → 2.40e-4, while ∫ρ barely moves). The measurement **fell 61%**. **The analytic
prediction and the measurement move in opposite directions** — so the measured drop is not a property of the
FC-1 local screened stiffening picture (whether unrelaxed Leg-1 or, barring wild geometry-sensitivity of
relaxation, the relaxed blend).

## Interpretation (bounded)

Machine verdict `BOX_DEPENDENCE_PARTIAL...` is a fall-through label; the honest reading of the numbers is stronger:

> The larger-box failure is **not resolution and not the local quasi-static mechanism** — the local mechanism
> predicts the *opposite sign* of box-dependence. The measured −61% drop must originate in something the static
> local model omits.

**Correction (Codex catch, verified by Claude in code).** The D4 row's `delta_omega` is already a **same-geometry**
full-minus-off subtraction (`jax_scout/gravity_TG_B1S_D4_rows_gpu.py` `d4_row` lines 129–132: per-row config →
per-geometry Q-ball solve → full and off both evolved at L=12; the fixed L=10 reference enters *only* the gate's
relative-difference test). So the L=12 shift is a **genuine, same-geometry, box-dependent measurement**, not a
full-vs-off comparability artifact. Corrected candidate ranking for the omitted-physics cause, in order of
likelihood: **(a) absorber-position coupling** (the absorber shell moved r=3.5→4.2, changing the T/G tail boundary;
neglected in the static solve; `absorber_wider` tested width but not position), **(b) T/G transient dynamics** over
the 50-period window that a settled quasi-static estimate cannot see, **(c) modal-fit/branch differences** at L=12.
A separate **gate-design** question (is a fixed-L=10 box-invariance requirement fair for a mildly box-dependent
shift?) is secondary to finding the row-level cause.

This **redirects** the discrepancy review: it rules out the two most alarming readings (a real resolution failure;
a physically box-fragile effect in the modelled mechanism) and points the CX run-data review at absorber-position /
transient dynamics on the **existing** D4 artifacts (see `TG_B1S_D4_FINAL_ANALYSIS.md` §5). The single most
decisive next check is the **absorber-position isolation** — compare the larger_box row's late-time T/G steady
values and approach slopes against baseline; if the dynamical A(x) at the node is materially weaker at L=12,
absorber repositioning is the cause.

## SCREEN-1 (secondary)

Fitted G(r) tail length ≈ 1.4–1.7 (source-size-inflated); naive `c_G/ω_G = 0.65`, light coupled-mode range ≈ 0.82.
Mediation is **short-range** (order the coupled-mode range convolved with the finite node size), consistent with
Yukawa screening — which is *why* a purely local mechanism should be ~box-insensitive, exactly as the analytic
Δω would be if not for the mild box-sensitivity of the Petviashvili solve itself.

## Caveats & boundary

Leg-1 (fixed-profile) analytic shift only, computed per geometry; the fully-relaxed Leg-2 blend was not swept across
geometries (would only overturn the conclusion if relaxation were strongly geometry-dependent, which is not expected
for a localized node). Absorber neglected in the static T/G solve. Analytic-mechanism discriminator only — no
gravity, time-dilation, or IRER claim; no model/label/production change. D5 stays blocked pending the D4 review.

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `25020bf` (2026-07-22) — *dual substrate*
**Revised since:** 11 commit(s), most recently `bd93089` (2026-09-12)

**Harness code changed since it was written:** 6 commit(s) to `jax_scout/`.
  - `caf61af` 2026-09-10 — Build HUD items 1+2: snapshot writer and universal field renderer
  - `06edae5` 2026-08-27 — Fix run provenance: 30% -> 86% of runs now dated
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine
  - *…and 1 more.*

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Also referenced by (same date or earlier):** [[RUN_QUEUE]], [[gravity_maturity/CLAUDE_HANDOFF_20260715_TG_B1S_AND_RUNTIME]], [[gravity_maturity/CONCEPT_IMPLEMENTATION_AND_GAP_ANALYSIS_CODEX_CONTINUATION_20260717]], [[gravity_maturity/TG_B1S_D4_BOX_DISCREPANCY_CODEX_NOTE_20260715]], [[gravity_maturity/TG_B1S_D4_FINAL_ANALYSIS]]

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
