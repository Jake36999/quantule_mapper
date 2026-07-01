# Phase C — feb56dc7 basin focused search: results

**Date:** 2026-06-25
**Plan:** [PHASE_C_FEB_BASIN_SEARCH_PLAN.md](PHASE_C_FEB_BASIN_SEARCH_PLAN.md)
**Run:** `sweep_runs/FEB_BASIN_20260625_122824/` (N=96, T=12000, seed 20260619, **classifier v2** with the
long-time drift gate). 8/8 configs completed (ran in two parts — a daytime WSL teardown interrupted after
2 configs; resumed). Analysis/replay only; geometry frozen at `e8d6a78ea`.

> **Headline: feb56dc7's parameter regime is a genuine, robust bound-state basin.** All 8 configs —
> every node count K∈{3,4,5,6,8} and every IC-norm/mass variant — reach a **gate-validated**
> `TRUE_SATURATED_BOUND_STATE`. Bound-state formation is controlled by the **parameter regime**, not by
> IC norm, mass, or node count. This *supersedes* the earlier per-node-mass-window and IC-norm hypotheses.

## Results (all classifier-v2 gate-validated)

| config | class | n_fin | late_drift | er_fin | er_max | init mass → held |
|---|---|---|---|---|---|---|
| K3 per-blob | TRUE_SATURATED | 2 | −0.091 | 1.37 | 1.53 | 7463 → 10251 |
| K4 per-blob | TRUE_SATURATED | 2 | −0.074 | 1.47 | 1.60 | 11077 → 16311 |
| K5 per-blob | TRUE_SATURATED | 3 | −0.077 | 1.47 | 1.60 | 13927 → 20417 |
| K6 per-blob (=feb) | TRUE_SATURATED | 4 | −0.080 | 1.45 | 1.60 | 16534 → 24030 |
| K8 per-blob | TRUE_SATURATED | 4 | −0.068 | 1.53 | 1.66 | 22507 → 34520 |
| K6 total-mass 0.5× | TRUE_SATURATED | 5 | −0.121 | 1.38 | 1.60 | 8267 → 11437 |
| K6 total-mass 1× | TRUE_SATURATED | 4 | −0.080 | 1.45 | 1.60 | 16534 → 24030 |
| K6 total-mass 2× | TRUE_SATURATED | 4 | −0.061 | 1.16 | 1.25 | 33068 → 38413 |

All drifts within the gate (±0.15), all er_max bounded (≤1.66), all in-band. The fields grow early then
settle (mild negative late drift) — condensing bound states, not runaways.

## Interpretation

1. **Node-count basin (Grid A):** feb's param point supports stable bound states across the whole tested
   range; the final node count scales with the IC blob count (K3/4→2, K5→3, K6/8→4). feb's bound state is
   **not K-specific** — it is one member of a node-count family at that regime.
2. **IC-norm / mass is NOT the discriminator (Grid B) — hypothesis overturned.** At feb's params,
   total-mass-fixed at 0.5×/1×/2× the natural mass are all stable; even 2× (mass 33068) saturates with
   bounded energy. The Option-B failures used total-mass-fixed too, but with *different (random) params* —
   so IC norm and mass forcing are **not** why they failed. The pre-registered "total-mass destabilizes"
   read is **rejected by the data.**
3. **The controlling variable is the parameter regime.** feb's full parameter vector sits in a
   bound-state basin that is robust to IC details (node count, norm, mass). The broad Phase C search
   sampled random parameter vectors (even within the tightened eta/a band, the other six params were
   random) → mostly outside the basin → mostly transients. feb was the rare in-basin draw.
4. **The v2 gate is validated in both directions:** it rejected the Option-B transients (earlier) and it
   passed all 8 genuine bound states here without false negatives.

## Caveats

- **Single seed (20260619).** 8/8 TRUE is clean, but confirm the standout cases with 2 extra seeds before
  treating the node-count family as established (the discovery layer showed seed-sensitivity TRUE↔NEAR).
- **T=12000 gate-validated**, not T=24000. feb itself is confirmed stable to T=24000; the other seven are
  T=12000/drift-gated. A T=24000 confirmation of one or two (e.g. K3, K6 total-mass 2×) would add rigor.
- **One parameter point.** This maps the IC-robustness *at* feb's params; it does not yet map how far the
  parameters can move and stay in-basin (that is the next experiment).
- No charge / topology / proof / ground-state / black-hole / universal-law claim is made.

## Recommended next direction (the principled search axis)

The data points the search at **parameters, not ICs**. The next experiment is a **parameter-neighborhood
basin map around feb**: perturb feb's full param vector (one axis at a time, then small joint steps),
K=6/per-blob fixed, each N=96/T=12000 drift-gated, to find how wide the bound-state basin is in parameter
space and which params it is most sensitive to. That is the natural successor to this study — and unlike
the original broad hunt, it is anchored on a known attractor and validated by the calibrated gate.

Optional immediate rigor step first: 2-seed confirmation of the Grid-A node-count family + a couple of
T=24000 confirmations.

---

# Confirmation run — 2 seeds + T=24000 (2026-06-25)

`sweep_runs/FEB_BASIN_CONFIRM_20260625_154503/` (classifier v2). 12 configs.

## Seed robustness (T=12000, seeds 20260620 + 20260621 × K{3,4,5,6,8}) — CONFIRMED

All **10/10** → gate-validated `TRUE_SATURATED_BOUND_STATE` (drift −0.09 to −0.10). The node-count basin
is **seed-robust in stability**. One new wrinkle: the **final node count is seed-dependent**:

| seed | K3 | K4 | K5 | K6 | K8 |
|---|---|---|---|---|---|
| 20260619 (orig) | 2 | 2 | 3 | 4 | 4 |
| 20260620 | 3 | 4 | 5 | 6 | 7 |
| 20260621 | 3 | 4 | 5 | 6 | 8 |

For seeds 620/621 the blobs mostly persist (n_fin ≈ K); seed 619 merges more. So **stability is set by the
parameter regime; the node count is set by the IC blob geometry (placement)** — a clean separation, and
useful for the topology inspection.

## T=24000 rigor — exposed a GATE LIMITATION, not a bound-state failure

K3 and K6 (seed 619) at T=24000 both classified `SPIN_DOWN_REJECT` (drift −0.200, −0.177) — **but this is
the v2 gate over-rejecting a breathing bound state, not a real decay.** Evidence:
- K6_s20260619 **is feb's exact IC**. At T=24000: mass 16534 → 19787 (**held/init = 1.20, net growth**),
  `er_min = er[0] = 0.992` (energy **never** fell below its start), er_max 1.596, er_fin 1.197 — identical
  to the feb control we independently validated as a stable breathing bound state.
- K3_s20260619: held/init 1.10, er_min = er[0], bounded — same pattern.

The gate's normalized late-half drift over `[12000, 24000]` catches feb's **breathing downswing** from its
er_max peak (−0.18) and exceeds the 0.15 threshold. The threshold was calibrated at **Tv=12000** (where
feb's drift is 0.08); it does **not** transfer to Tv=24000 — exactly the "provisional, one stable exemplar,
breathing-bounded" caveat flagged in `PHASE_C_STABILITY_GATE_CALIBRATION.md`.

**Net:** the feb-basin bound states are real and seed-robust (T=12000 confirmed). The v2 gate is sound at
its calibrated T=12000 window but **over-rejects breathing bound states at longer windows** — it needs a
breathing-robust refinement (distinguish a breathing downswing — `er_min ≈ er[0]`, er_fin bounded, mass
net-grown — from a monotonic decay — `er_fin ≈ er_min → floor`) before being applied at T≥24000. Use the
gate at T=12000 for now.
