---
tags: [record, infra, telemetry, hud]
date: 2026-10-04
branch: Branch - Validation - Index
status: complete
---

# Telemetry stream (Phase C)

This is the build record for Phase C of [[IMPLEMENTATION_PLAN_2026-10]]. Harnesses can now stream
the scalars they already compute to `telemetry.jsonl`, and declare which of them are invariants and
with what tolerance. `tools/hud_monitor.py` plots them live and reports breaches. This answers
[[SESSION_SYNTHESIS_2026-09-17]] §5: until now you could not see an identity break at hour 3 of a
12-hour run.

> [!info] Sources
> - `jax_scout/snapshots.py`: `TelemetryWriter`, `read_telemetry`, `invariant_breaches`
> - `tools/hud_monitor.py`: `telemetry_dirs`, `render_telemetry`, breach reporting
> - Wired harnesses: `gravity_TG_B2_midplane_stress_flux.py` (live momentum-ledger residual) and
>   `gravity_TG_B1S_state_load_feedback_gpu.py` (diagnostic stream)
> - Tests: `tests/test_snapshots.py` (7 new)
> - Smoke run: midplane, default grid, T=3, arms off + well (`F:\Maths_exploration\phase_c\midplane_smoke`)

## Design
It follows the same contract as `SnapshotWriter`:
- The writer is a **pure observer** with a bounded queue and a background thread.
- It **never blocks**. When the queue is full, samples are dropped and counted.
- Every failure is swallowed and counted.

Each line is written with a single `write()`, so a reader sees at most one torn line, always the last,
and `read_telemetry` skips it. Multi-arm harnesses write one directory per arm
(`<run>/telemetry/<arm>/`). The monitor renders `rendered/telemetry_<arm>.png`, drawing every
declared invariant as |value| on a log axis with its tolerance line.

**Deviation from the plan:** the plan said "add `telemetry()` to `SnapshotWriter`". It is a separate
class instead, because telemetry is cheap enough to be on whenever a harness passes a directory,
while snapshots stay off by default.

## Invariants declared so far

| harness | invariant | tolerance | why |
|---|---|---|---|
| TG-B2 midplane | `ledger_resid_rel`: per-sample momentum-ledger residual, normalised by the running mean of the term magnitudes | the harness's own `--resid-tol` (0.05) | the same identity its G1/G2 gates check offline |
| TG-B1S | *none* | — | an open system (source, damping, absorbing boundary): neither charge nor energy is conserved, so nothing honest can be declared live |

### A design error caught by the smoke run
The first version normalised each sample's residual by the terms **at that sample**. Those terms
cross zero at turning points, so the residual spiked there: 8 false "breaches" on a run whose
offline gates all passed. The fix normalises by the running mean of |dP/dt|, |flux| and |F_R|, which
is the same scale `ledger()` uses offline. The rerun showed **0 breaches**, and the residual settles
to 1.5e-4.

This is why invariant tolerances belong in the harness, next to the identity they claim, and why
each one needs a smoke run before it is trusted.

## Not yet wired (deliberately)
`feb_gain_ladder_longt.py`, `feb_astar_confirm.py`, `phase_d_c2_7_rederivation.py` and
`core_saturation_search.py` are running in the Phase B4 replay right now. Editing them mid-replay
would change the code that later replay steps load. They will be wired after the replay finishes.

## What this changes

| status | item |
|---|---|
| **Unchanged** | Every harness's numerical output; the midplane gates are identical with telemetry on. |
| **Newly possible** | Watching invariant residuals of a long run live. |

## Issues raised

| # | issue | status |
|---|---|---|
| 1 | Wire feb / C2.7 / core_saturation harnesses | OPEN — after the B4 replay |
| 2 | TG-B1S has no live invariant (open system) | OPEN — candidate: a live energy-ledger residual once its closure order is measured |

## Associated docs
- [[IMPLEMENTATION_PLAN_2026-10]] · [[SOLVER_AND_RUNTIME_CHANGELOG]] · [[VISUAL_HUD_SCOPE_RFC]] · [[SESSION_SYNTHESIS_2026-09-17]]

## Branches
- [[Branch - Validation - Index]]
