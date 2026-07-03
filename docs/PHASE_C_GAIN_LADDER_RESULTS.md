# Phase C — Long-T Gain Ladder: the true long-time bound state (a\* ≈ ×1.15)

**Date:** 2026-07-01 (run `FEB_GAIN_LADDER_LONGT_T72000_20260701_175708`, script
`jax_scout/feb_gain_ladder_longt.py`). N96 / K6 / per-blob / seed 20260619 / v3 gate, geometry frozen `e8d6a78ea`.

## Context — this converts the T=72000 "falsifier" into a refinement

The long-T breathing run (`FEB_BREATHING_LONGT_T72000`, 2026-06-28) showed feb-center (param_a ×1.0)
and ×1.05 both **slowly decay** to T=72000 (single early peak, then a monotone `er(t)` decline,
`er_min == er_fin`, no plateau). Read in isolation that falsifies feb-center as a fixed long-time bound
state. But the decay was **gain-dependent** (×1.05 decayed slower than ×1.0), and the T24000 delineation
had `er_fin` rising monotonically with param_a — so the hypothesis was that the true gain/loss balance
sits at a gain **above** feb-center. This run tests that by climbing param_a and reading the
**late-window slope** of `er(t)` (fractional change per 1000 steps over the last half of the run).

## Result — late slope → 0 at a ≈ ×1.15

| param_a (×feb) | er peak | er_fin | late slope /1k (last 50%) | reading |
|---|--:|--:|--:|---|
| ×1.00 (feb-center) | 1.596 | 0.416 | −0.0126 | decays (breathing run) |
| ×1.05 | 1.734 | 0.812 | −0.0120 | decays (breathing run) |
| ×1.075 | 1.808 | 1.066 | −0.0105 | slow decay |
| ×1.10 | 1.888 | 1.358 | −0.0081 | slow decay |
| ×1.125 | 1.975 | 1.681 | −0.0047 | slow decay |
| **×1.15** | **2.075** | **2.038** | **−0.0006** | **flat — stationary** |

- The late slope is a **clean monotonic function of gain**, crossing ~0 at **a ≈ ×1.15** (param_a ≈ 0.55;
  feb param_a = 0.4802). Linear extrapolation of the last three points puts the exact zero at ≈ ×1.15–1.16.
- **a×1.15 is genuinely stationary**, not caught mid-plateau: `er(t)` rises to ~2.07 (peak at 18% of the
  run) then holds — 50%→100% is 2.061 → 2.038 (−1% total), slope −0.0006/1k over *both* the last 50% and
  last 10%. `floor_ratio` = 1.0 (never fell below start), `er_fin ≈ er_max`. All cells `n_fin = 4`.

## Interpretation

- **A real long-time bound state exists** — the Phase-C search was not wrong that a bound state exists; it
  was **anchored ~15% below the balance gain**. feb-center decays because it is loss-dominated relative to
  a\*; raising the cubic gain to a×1.15 closes the balance and the state holds to T=72000.
- **The late-window slope → 0 is the correct long-time-stability criterion.** The v3 "TRUE" verdict at
  T≤24000 is window-limited: cells with a small negative slope (×1.075…×1.125) still classify TRUE at
  T=72000 only because their higher gain lifts `er` so they haven't yet crossed `er0` — but their slope is
  negative, so at long enough T they would decay like feb-center did. Only slope≈0 is a true fixed point.
- **a\* is a knife-edge balance**: below it, adiabatic decay (rate ∝ distance from a\*, shape-preserving,
  n_fin held); above it, growth. This is the gain/loss balance made quantitative on the cubic-gain axis.

## Caveats (→ overnight confirmation run)

Single seed (20260619); a\* inferred from slope≈0 over [T/2, T] at a single T (72000); a\* sits just under
the growth edge. The overnight run `FEB_ASTAR_CONFIRM` tests all three:
- **Longer-T:** a×1.15 and a×1.125 at **T=144000** — does a×1.15 stay flat, and does the sub-a\* survivor
  (×1.125) reveal itself as a slow decayer (cross `er0`)?
- **Seed-robust:** a×1.15 at seeds 20260620 / 20260621 (T72000).
- **Bracket a\* + growth edge:** a×1.16, ×1.175, ×1.20 (T72000) — locate the zero-crossing precisely and
  confirm the slope goes positive (growth) above a\*.

## Downstream implications

- The **matter-likeness kick/inertia test** (endpoint sequence step 5) should target the **confirmed a\***
  state (a×1.15, param_a ≈ 0.55), **not** feb-center — feb-center is a decaying transient.
- Re-run the **node-interaction review** (`docs/PHASE_C_NODE_INTERACTION_REVIEW` TODO) on the four T72000
  gain-ladder fields: they carry real long-time labels (decay vs stationary) and include a genuine
  stationary example, fixing the snapshot/window confounds of the T24000 pass.
- Candidate **aste_hunter stability objective**: minimise |late-window er slope| (slope→0), on the
  param_a / eta / rho_vac gain-balance axes — the observable this run shows is the real discriminator.
