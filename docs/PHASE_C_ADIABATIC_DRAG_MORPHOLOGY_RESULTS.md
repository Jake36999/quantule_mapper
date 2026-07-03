# Phase C — Adiabatic Drag: Morphology Universality (static-well V0 ladder)

**Date:** 2026-07-02 · script `jax_scout/feb_adiabatic_drag.py` (default-off variant, `BASELINE_REPRODUCED`
bit-identical on both runs) · saved states only, no fresh hunt, no moving-well, no solver change.
**Wording:** site-pinning / relaxational resistance / local accretion — **not inertia**, not matter-motion.

## Question

Is the accretion-only / no-relocation result specific to the 4-node seed619 a×1.15 state, or general across
other stable a×1.15 morphologies? Tested the **same** static-well V0 ladder (0.075→0.40, w=1.0, offset=1.8) on
the two saved 6-node alternates.

## Result — no relocation in ANY morphology

| state | morphology | verdict | offset-cell labels across V0 = 0.075→0.40 |
|---|---|---|---|
| seed619 | 4-node | `ACCRETION_ONLY_NO_RELOCATION` | accretion at all V0; origin never depletes |
| seed620 | 6-node | `STATIC_WELL_NO_COUPLING` | NULL at all V0 (offset well landed in low density; barely accreted) |
| seed621 | 6-node | `ACCRETION_THEN_NUCLEATION` | NULL → accretion (V0 0.15–0.3) → **new 7th node** at V0=0.4 |

**seed620 (6-node), COM0=[−3.16, 2.65, 1.29]:** offset well NULL at every V0 — COM "bias" only 0.03→0.09, origin
mass flat (0.058→0.06), well mass ≈0 throughout, nodes 6→6. The structure ignores the offset well entirely.

**seed621 (6-node), COM0=[−2.84, 1.33, −1.45]:** origin mass stays flat (0.024→0.036) while well mass grows
(0.034→0.101) and total mass climbs (→1.14); at V0=0.4 a **new node forms at the well** (6→7). This is the
field growing *new* structure at the preference — nucleation — not the existing structure relocating.

**Cross-check (both 6-node states):** a strong *on-centre* well (V0=0.4) grows a 7th node (6→7) in both — a
strong local gain nucleates new structure rather than reshaping/moving the existing one.

## Interpretation

In none of the three morphologies does the existing node structure migrate toward the well: **origin regions
never deplete, node centroids do not shift coherently, and no node identity relocates.** The response is always
*local* — negligible coupling (seed620), local accretion (seed619), or nucleation of a new blob (seed621 at high
V0). The apparent COM shifts are mass-weighted artifacts of local growth at the well, exactly as in the 4-node
case.

## Verdict

`STATIC_WELL_ACCRETION_ONLY_GENERALIZED_ACROSS_MORPHOLOGIES` — the **no-relocation** result is general across
the 4-node and both 6-node a×1.15 stable states, over a 16× well-strength range. (Response *mode* varies with
morphology/well-placement: none → accretion → nucleation; relocation occurs in none.)

**Mobility arc (closed under the tested probes):**
- standing dissipative attractors exist and are robust;
- they show effective **site-pinning / relaxational resistance** under weak-to-moderate static gain wells;
- they respond by **local accretion (and nucleation of new structure at strong wells), not by migration**;
- combined with the operator-level inertial null (`docs/PHASE_C_KICK_INERTIA_AND_OPERATOR_FINDING.md`),
  **matter-like transport would require a different transport/relocation mechanism** (e.g. a dispersive/
  advective term) — a future formalism decision, not a patch to this run.

**Moving-well test: NOT run** (gate never passed — no static relocation in any morphology). No inertia claim, no
matter-motion claim, no solver change, no new hunt.

## Provenance
`sweep_runs/FEB_ADIABATIC_DRAG_V0LADDER_seed620_20260702/`, `..._seed621_20260702/`; states from
`FEB_ASTAR_CONFIRM_20260702_003055`; N96/L10/dt0.005; geometry frozen `e8d6a78ea`.
