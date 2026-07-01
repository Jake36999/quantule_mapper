# Phase C — joint (param_a, param_eta, param_rho_vac) basin: Stage 1 results

**Date:** 2026-06-27
**Run:** `sweep_runs/FEB_JOINT_BASIN_20260626_224056/` (classifier **v3**, N=96, K=6, per-blob, seed 20260619,
T=12000, ω₀=0; `param_D`/`param_a_coupling`/`param_s`/`param_f` at feb). Map: `…/joint_basin_map.png`.
Analysis/replay only; geometry frozen at `e8d6a78ea`. Scientific rejects = the coupled basin boundary.

**Output root:** `FEB_JOINT_BASIN_20260626_224056` · **wallclock:** 7.11 h (ran in 2 parts — resumed after a
session-boundary teardown at 6/45; resumable `--out` skipped the 6 done) · **completion:** **45/45**.

**Counts:** TRUE 38 · NEAR 0 · reject 7 (SPIN_DOWN 5, GROWER 2). Basin is **large (84% TRUE)** in these
ranges with a thin, structured boundary.

## 3-D grid (T=TRUE, S=spin-down, G=grower; rows = `param_a`×factor, cols = (eta, rho_vac)×factor)

```
a\(eta,rho)  e.8 r.85  e.8 r1   e.8 r1.25 | e1 r.85  e1 r1   e1 r1.25 | e1.2 r.85  e1.2 r1  e1.2 r1.25
 a0.9          T        T        T        |  T        T        T       |   S          S         T
 a0.95         T        T        T        |  T        T        T       |   S          S         T
 a1.0          T        T        T        |  T        T        T       |   S          T         T
 a1.05         T        T        G        |  T        T        T       |   T          T         T
 a1.1          T        T        G        |  T        T        T       |   T          T         T
```

Reject cells: 5 spin-down all at **eta×1.2** (high loss); 2 growers at **eta×0.8 + rho×1.25 + a×{1.05,1.1}**.

Reject marginals: `param_eta` ×0.8/1.0/1.2 → 2/**0**/**5**; `param_a` ×0.9…1.1 → 2/2/1/1/1;
`param_rho_vac` ×0.85/1.0/1.25 → 3/2/2.

## Basin-boundary interpretation — a gain/loss balance surface

The boundary is **not a product of independent 1-D windows**; it is a **coupled gain/loss balance**:

- **The `eta×1.0` (feb-loss) plane is fully stable** — 0/15 reject across all `a` and `rho`.
- **High loss (`eta×1.2`) spins the state down — UNLESS compensated.** At `eta×1.2, rho×0.85` the low/mid-gain
  cells (`a×0.9/0.95/1.0`) spin down, but `a×1.05/1.1` stay TRUE: **raising the cubic gain rescues the
  high-loss cells.** And `eta×1.2, rho×1.25` is TRUE for *all* `a`: **raising the conformal reference
  density also rescues high loss.**
- **Low loss + high drive grows (`eta×0.8, rho×1.25, a×1.05/1.1` → GROWER) — UNLESS gain is lowered.**
  Dropping to `a≤1.0` at that corner returns to TRUE: **lowering gain rescues the over-driven cells.**

So the bound state exists where **gain ≈ loss** on the emergent geometry. `param_a` (gain) and
`param_rho_vac` (reference density) both **trade off against `param_eta`** (loss). The basin is a diagonal
region in `(a, eta, rho)`, and its boundary moves with the *balance*, not any single parameter.

## Does the OAT story survive coupling?

- **`param_eta`/`param_rho_vac` tradeoff: CONFIRMED** — clear, in both directions (`a` and `rho` each
  compensate for `eta`). The OAT 1-D windows would have *missed* this coupling.
- **Is `param_a` still the critical knob?** Refined: within ±10% of feb (the joint `a`-range, deliberately
  inside the OAT `a`-window) `param_a` alone drives few rejects — because the joint grid stays inside the
  `param_a` window established by OAT. The joint result shows the **real control is the gain/loss *balance*
  (`a` vs `eta`)**: `param_a` is critical *as the gain that must balance the loss*, not in isolation. In
  these ranges `eta×1.2` is the dominant single-axis reject driver (5/7), with `a` and `rho` as its
  compensators. So: `param_a` remains a critical control axis, but the joint map reframes the headline as
  **gain/loss balance**, deepening (not contradicting) the OAT read.
- **Joint basin not dramatically smaller than the OAT product** — 84% TRUE; the coupling sharpens the
  boundary into a diagonal rather than shrinking the basin wholesale.

## Post-hoc diagnostics (cheap, read-only)

`prime_log_sse` and TDA on all 45 final fields: **0/45 prime peaks, 0/45 persistent topology** — unchanged
across the coupled basin (smooth, spectrally featureless solitons; consistent with the earlier null).

## Candidate cells for Stage 2 (proposed — NOT launched)

**Interior cells for T=24000 confirmation** (deep, far from boundary; confirm coupled interior is long-time
stable, not a T=12000 artifact):
- `a1.0_e1.0_r1.0` (feb-center, anchor — already T24000-confirmed) · `a1.1_e1.0_r1.25` (high gain+high rho) ·
  `a0.9_e1.0_r0.85` (low gain+low rho).

**Boundary cells for seed-repeat** (TRUE↔reject transitions; the upper-edge seed-sensitivity finding makes
these the cells to multi-seed):
- a↔eta rescue: `a1.0_e1.2_r0.85` (S) vs `a1.05_e1.2_r0.85` (T); `a0.95_e1.2_r1.0` (S) vs `a1.0_e1.2_r1.0` (T).
- rho rescue: `a0.9_e1.2_r1.0` (S) vs `a0.9_e1.2_r1.25` (T).
- grower onset: `a1.0_e0.8_r1.25` (T) vs `a1.05_e0.8_r1.25` (G).

**Matched off-basin controls** (same grid, one/two-step perturbation flips — for the dossier):
- `a1.0_e1.0_r1.0` (TRUE feb) vs `a1.0_e1.2_r0.85` (SPIN) — differ in loss+rho.
- `a1.0_e0.8_r1.25` (TRUE) vs `a1.05_e0.8_r1.25` (GROWER) — differ in `a` by one step.

## Stage 2 — paused, for your decision (no auto-launch)

Proposed Stage 2 = boundary seed-repeat (the cells above, seeds 20260620/621) + interior T=24000
confirmation (3 cells) + the matched controls run through the gate *and* post-hoc diagnostics. Estimated
~14 boundary/interior runs at T12000 (~3h) + 3 at T24000 (~1.5h). Not launched.

No charge / topology-proof / log-prime-proof / matter / ground-state / molecule / black-hole language.
