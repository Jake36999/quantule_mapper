# Phase D.6 — Reduced Node-Network ("Zoom-Out") Model — Plan

**Goal.** A **minimal reduced simulator** built *only* from the measured Phase D dissipative-sector laws — **not** a
new PDE solver. It coarse-grains a field configuration to a **node graph** (descriptors + merge/couple rules) so we
can reason about many-node arrangements without full-resolution vacuum-field sims. Standalone (numpy), read-only
w.r.t. the PDE solver; validated against the harvested `PHASE_C_NODE_LIBRARY`.

## The measured laws it encodes (from D.1–D.5)
- **merge radius ≈ 0.3 box** (two settled nodes closer than this coalesce — D.5).
- **connectivity radius ≈ 0.5 box** (density-bridge coupling exists only within this — D.4).
- **no advective drift** in the dissipative substrate (nodes are site-pinned — C1, D.5): positions are fixed except
  by merging.
- **global phase-lock** = a stability signature (not a distance coupling — D.4).
- **current channel null** (drop from the node descriptor — D.3).
- **possible cooperative stabilisation** (multi-node layouts pack tighter than the isolated-pair merge radius, e.g.
  seed621 6-node at 0.21 box; 2 isolated blobs grew — D.5) → a phenomenon the model should *flag*, not assume.

## Node descriptor
`{position (box), mass, phase, node_class, stability_score, neighbour_count (within r_couple), coupling_edges}`.
Current/vorticity are **omitted** (null in stable nodes).

## Rules (dissipative baseline; no drift)
1. **merge** if pair separation `< r_merge` (default 0.3) → combine into a mass-weighted node.
2. **couple** (add an edge) if separation `< r_couple` (default 0.5).
3. **hold** position otherwise (no drift term — dissipative nodes are pinned).
4. **no drift** unless an *explicitly labelled hypothetical* extension is toggled on (default off).
5. **network stability** is a function of spacing, node count, and coupling topology (heuristic, calibrated to the
   library) — not an assumed constant.

## Validation against the Phase C node library (the real test)
1. **Node-count tendency** — feed random dense arrangements, apply merge-resolution, and find whether/what merge
   radius reproduces the library's **~4-node cap** (histogram peak).
2. **Stable spacing bands** — do the library's stable configs respect the `r_merge` floor? Where they sit above it
   is the model's predicted stable band.
3. **Over-merged / isolated prediction** — configs with all pairs `< r_merge` → over-merge to 1; all pairs
   `> r_couple` → isolated. Check against observed failures.
4. **Cooperative-stability layouts** — identify library configs that are **stable below the isolated-pair merge
   radius** (the seed621-type tight packings) — the model should *flag* these as cooperative exceptions the pairwise
   law under-predicts.

## Outputs
`jax_scout/reduced_node_model.py` (standalone numpy; reads `PHASE_C_NODE_LIBRARY.json`) +
`docs/PHASE_D6_REDUCED_NODE_MODEL_RESULTS.md` (validation + honest limits).

## Guardrails / non-goals (hard stops)
- **Do not alter the Phase C PDE solver.** This is a separate reduced model.
- **No macro/matter validation claims** — the model reproduces *coarse tendencies*, it does not prove macro physics.
- **No active stress forces**; **no drift** in the reduced model unless explicitly labelled hypothetical.
- After D.6: the transport-substrate branch (B / C2) is opened *separately*, with D.6 as the dissipative baseline.
