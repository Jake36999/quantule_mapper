# Phase C — joint basin Stage 2: boundary seed-repeat + interior T=24000

**Date:** 2026-06-27
**Run:** `sweep_runs/FEB_JOINT_STAGE2_20260627_123235/` (classifier v3, K=6/per-blob/N=96). 19/19 complete.
Follows [PHASE_C_JOINT_PARAM_BASIN_RESULTS.md](PHASE_C_JOINT_PARAM_BASIN_RESULTS.md). Analysis/replay only;
geometry frozen at `e8d6a78ea`. Scientific rejects = basin boundary.

> **Net:** the joint basin's **core is robust** (seed- and long-time-stable), but its **margins are
> T12000-window-marginal**: the boundary is mostly seed-robust (14/16) with one seed-fragile corner, and the
> **T=24000-confirmed interior is narrower than the T=12000 basin** (a low-drive corner that is TRUE at
> T12000 decays by T24000). Quote the basin at the **T24000-confirmed core**, with the T12000 extent flagged
> as window-marginal at the edges.

## 1. Boundary seed-repeat (8 transition cells × seeds 20260620/621, T12000) — 14/16 match

| cell (a,eta,rho ×factor) | ref (seed619) | seed620 | seed621 | |
|---|---|---|---|---|
| a1.0, e1.2, r0.85 | SPIN | SPIN | SPIN | ✓ |
| **a1.05, e1.2, r0.85** (gain-rescue) | TRUE | TRUE | TRUE | ✓ seed-robust |
| a0.95, e1.2, r1.0 | SPIN | SPIN | SPIN | ✓ |
| a1.0, e1.2, r1.0 | TRUE | TRUE | TRUE | ✓ |
| a0.9, e1.2, r1.0 | SPIN | SPIN | SPIN | ✓ |
| **a0.9, e1.2, r1.25** (rho-rescue, lowest gain) | TRUE | **SPIN** | **SPIN** | ✗ **seed-flip** |
| a1.0, e0.8, r1.25 | TRUE | TRUE | TRUE | ✓ |
| a1.05, e0.8, r1.25 (grower) | GROWER | GROWER | GROWER | ✓ |

- **The gain-rescue is seed-robust** (raising `param_a` to rescue a high-loss cell reproduces at all seeds).
- **The rho-rescue at the *lowest* gain is seed-fragile**: at `a×0.9, eta×1.2`, `rho×1.25` rescued the cell at
  seed-619 but not at 620/621 → SPIN. So the conformal-density rescue is marginal when gain is low; the
  basin's high-loss corner moves ~one cell with seed (consistent with the OAT/edge-confirm finding that
  upper boundaries are seed-sensitive).

## 2. Interior T=24000 confirmation (3 cells, seed 619)

| cell (a,eta,rho ×factor) | T24000 class | drift | breath | er_fin | reading |
|---|---|---|---|---|---|
| a1.05, e1.0, r1.0 | **TRUE** | −0.113 | True | 1.46 | genuine long-time breather ✓ |
| a1.1, e1.0, r1.25 (high-drive corner) | TRUE | +0.065 | False | 2.31 | marginal — high in-band, near grower edge |
| **a0.9, e1.0, r0.85** (low-drive corner) | **SPIN_DOWN** | −0.447 | False | 0.52 | **decays by T24000 — T12000 artifact** |

(feb-center `a1.0,e1.0,r1.0` already T24000-confirmed TRUE/breathing in the param-basin run.)

- The **core** (feb-center, a1.05_e1.0_r1.0) is genuinely long-time stable.
- The **low-drive corner** (low gain + low rho_vac) is a **slow decayer** that the T12000 window scored TRUE
  but reveals as spin-down at T24000 — the same window-artifact lesson, now at the basin's low-drive corner
  (less gain + less reference density vs the same loss → slow energy bleed).
- The **high-drive corner** (high gain + high rho) is TRUE at T24000 but sits high in-band (er_fin 2.31, not
  breathing) — near the grower edge.

**Implication:** the T=24000-validated interior is **narrower** than the T=12000 basin; the genuinely
long-time-stable region is the core around feb, contracting away from both the low-drive and high-drive
corners. The T12000 map remains the right *breadth* tool (with the v3 drift gate), but long-time promotion
should use T=24000 at the margins.

## 3. Matched off-basin controls (read-only on Stage 1; one-step flips under the identical gate)

| pair (one-parameter step) | stable | flipped |
|---|---|---|
| +5% gain at high loss | a1.0,e1.2,r.85 → SPIN (breath F, er_fin 0.82) | a1.05 → **TRUE** (breath T, 0.97) |
| +rho at high loss | a0.9,e1.2,r1.0 → SPIN (breath F, 0.68) | r1.25 → **TRUE** (breath T, 0.85) |
| +5% gain at low loss | a1.0,e0.8,r1.25 → TRUE (er_fin 2.30) | a1.05 → **GROWER** (er_fin 2.54) |
| feb vs high loss (eta+rho) | a1.0,e1.0,r1.0 → TRUE (breath T, 1.45) | a1.0,e1.2,r0.85 → SPIN (breath F, 0.82) |

For **every** pair: **n_fin = 4 nodes on both sides**, and **prime peaks = 0, persistent topology = 0 on
both sides**. So near the boundary, **morphology and spectra do NOT distinguish stable from failed — only
the dynamics do** (bounded breathing vs decay/grow). One parameter step flips the outcome under the
identical v3 gate — structured, predictable boundary behaviour.

## 4. Updated claim status (for the dossier)

- **Claim C (eta/rho_vac trade off with gain): SUPPORTED** — gain- and rho-rescues of high-loss cells are
  explicit; gain-rescue seed-robust.
- **Claim D (boundary seed-sensitive): SUPPORTED & quantified** — coupled boundary 14/16 seed-robust; the
  low-gain rho-rescue corner is the seed-fragile cell.
- **Claim F (bounded breathers): reinforced** — flips are dynamical (breath True↔False), not morphological
  or spectral.
- **New: the T=24000-validated basin is narrower than the T=12000 basin** (low-drive corner decays) — quote
  results at the T24000-confirmed core.

## 5. Where this leaves Phase 1

Internal-validation block essentially closed: resolution ✓ (N=128), seed robustness ✓ (interior +
boundary, with the one fragile corner identified), long-time ✓ (core confirmed; margins contract),
coupled basin mapped ✓, matched controls ✓, post-hoc diagnostics null ✓. Remaining optional rigor: a
couple more T=24000 interior cells to delineate the long-time core precisely. The dossier (claims A–G +
this) is ready to assemble for the wider discussion.

No charge / topology-proof / log-prime-proof / matter / ground-state / molecule / black-hole language.
