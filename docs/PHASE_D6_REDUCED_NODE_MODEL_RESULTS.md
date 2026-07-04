# Phase D.6 — Reduced Node-Network Model — Results

**A standalone coarse-graining model built only from the measured Phase D laws reproduces the Phase C library's
node-count tendency and spacing band, and correctly flags cooperative-stability exceptions.** `jax_scout/
reduced_node_model.py` (numpy, **no PDE solver, no jax**); validated against `PHASE_C_NODE_LIBRARY` (162 configs,
129 stable). No matter/macro claims; no solver change; pinned (no drift).

## The model
Nodes = points in the periodic box with `{position, mass, phase, class, stability, neighbours, edges}` (current
dropped — null). Rules: **merge if separation < r_merge**, **couple (edge) if < r_couple**, **hold otherwise** (no
drift). Defaults `r_merge=0.3`, `r_couple=0.5` (the measured D.5/D.4 radii).

## Validation
1. **Node-count tendency — reproduced.** Library stable configs peak at **4** (median 4, mean 4.1; hist
   `{2:4,3:4,4:106,5:3,6:10,7:1,8:1}`). Random dense ICs → merge-resolution:
   | r_merge | random-IC final count (median) |
   |---|---|
   | 0.3 | 7 |
   | **0.4** | **4** ✓ (matches library) |
   | 0.5 | 3 |
   | 0.6 | 2 |
   The **effective settling merge radius ≈ 0.4** reproduces the ~4-node cap — sitting between the settled-pair merge
   radius (0.3, D.5) and the coupling radius (0.5, D.4). So the "merges to ~4" tendency is quantitatively captured by
   a merge-resolution rule with a single calibrated radius.
2. **Cooperative-stability exceptions — flagged.** **7/129** stable configs pack *below* the pairwise merge radius
   (min spacing 0.21–0.23 < 0.3) — and they are **all the `seed621` morphology** (K5/K6/K8, joint-stage2, astar
   seed621). The pairwise law would (wrongly) merge these; observed stable → **cooperative stabilisation**, correctly
   identified as exceptions, and **seed-specific** (a real morphological finding, not universal).
3. **Library consistency.** 122/129 respect the merge floor; the 7 exceptions are the cooperative cases; **3** configs
   are fully isolated (all pairs > 0.5). So the pairwise model is consistent for ~95% of the stable library.
4. **Stable spacing ≈ coupling radius.** Mean node degree **0.74** at `r_couple=0.5` — stable configs settle *right
   at* the coupling-radius edge (min spacing ~0.485 ≈ 0.5): **marginally-coupled** arrangements, not dense cliques.

## What this establishes
The dissipative node sector coarse-grains to a **pinned/merging network** with a small set of measured constants:
- **settling merge radius ≈ 0.4**, **settled merge radius ≈ 0.3**, **coupling radius ≈ 0.5**;
- **no drift** (nodes pinned — the transport null);
- **~4-node attractor** as a packing/merge outcome;
- **stable spacing ≈ coupling-radius edge (~0.485)**;
- **cooperative stabilisation** as a seed-specific exception (tight ~0.22 packings) the pairwise law under-predicts.
This is the **D.6 "zoom-out" baseline** for the dissipative sector: a many-node arrangement's fate (over-merge /
stable-network / isolated) is predictable from spacing + the merge/couple radii, without a full-resolution PDE run.

## Honest limits (hard-stop compliant)
- **Tendency-level, not dynamical.** Because dissipative nodes are pinned, the model has no continuous evolution
  beyond merge-resolution + graph construction — it predicts *arrangements/fates*, not trajectories. **No macro or
  matter validation is claimed.**
- **Cooperative term missing.** The 7 exceptions show a single pairwise merge radius is incomplete for tight
  multi-node packings; a fuller model needs a cooperative/neighbour-count term (calibratable if more tight-packed
  stable families are catalogued — a D.3 extension).
- **No drift term** (dissipative baseline); any drift is a *hypothetical* extension for the separate transport branch
  (B / C2), for which this model is the dissipative comparison baseline.
