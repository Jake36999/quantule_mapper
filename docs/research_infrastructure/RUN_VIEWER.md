---
tags: [record, infra, ui, viewer]
date: 2026-10-07
branch: Branch - Validation - Index
status: complete
---

# Run gallery and viewer

Goal: anyone can look at a run in roughly the ways an agent can, without writing a script. That means
browsing every run, seeing its final 3-D state from any angle and in any quantity, seeing its time
series, and getting a playable recording of the dynamic part.

```
.venv/Scripts/python.exe tools/serve_viewer.py          ->  http://127.0.0.1:8766/
python tools/render_queue_worker.py   (WSL ~/jax_irer)  ->  runs queued re-runs, one at a time
```

> [!info] Sources
> - `tools/viewer_data.py`: data layer, testable without HTTP
> - `tools/serve_viewer.py`: localhost server, standard library only
> - `web/viewer/index.html`: page, plotly from a CDN
> - `tools/render_queue_worker.py`: human-started runner
> - `tools/run_spec.py` + `irer_specs`: the `protocol.record` option
> - Tests: `tests/test_viewer.py` (13); `tests/test_spec_layer.py::test_record_window_writes_only_the_window`

## What it does

**Gallery.** Every run in the results index appears as a card:
- **Thumbnail**, when the run has a rendered figure.
- **Badge:**
  - `current` / `revalidated`: trustworthy on the fixed solver;
  - `stale`: a pre-fix ETDRK4 run;
  - `unaffected`: its stepper was never broken;
  - `unindexed`: written since the index was last rebuilt, such as a fresh re-run.
- **What it can show:** number of field files and number of recorded frames.
- **Filters:** substrate, branch, search, "has fields", "has history".

**Viewer.**
- **Final state.**
  - One tab per saved 3-D field.
  - Any quantity: |ψ|², |ψ|, arg ψ, Re, Im.
  - Three centre-plane slices plus a rotatable 3-D isosurface view, at 32/48/64 per side.
  - Fields are downsampled on the server, and only the array headers are read until a field is
    opened.
- **Time series.** Every 1-D series a run saved (for example `er(t)`, up to 144k points, downsampled
  for plotting) and every telemetry scalar, drawn with its declared tolerance.
- **Suggested windows.**
  - Shaded bands where the behaviour **changes**: transients, bends, oscillation.
  - These are found from curvature (d²y/dt²), not slope, so a steady drift produces no suggestions.
    The page then says to record the whole run at a low frame rate instead.
- **Recorded history.** A frame slider and play/pause, with slices and 3-D. The colour range is held
  fixed during playback so colours mean the same thing on every frame.
- **Re-run with visuals.** Choose either:
  - the **whole run at a low frame rate**, or
  - a **time window at a high frame rate**, by dragging across the time series or clicking a
    suggested window.

  Pick a frame count and volume resolution, preview the wall-time and disk estimate, then queue the
  job.

**Recording (`protocol.record`).** Available to every substrate through the spec layer:
- `every`: frame interval in physical time.
- `volume`: downsampled 3-D size per side.
- `window`: `[t0, t1]` for a close-up.
- `fields`: which fields to save.

Frames are written to `<run>/history/` by the same `SnapshotWriter` the harnesses use.

**A window re-run stops at the end of its window.** Integration must start at t=0, because the state
at t0 can only be reached by running there, but it does not need to continue afterwards. So a
close-up of the a\* start-up transient (t = 0–23 of 720) costs about 3% of the full run.

## The queue rule
The page **cannot run anything**. "Queue" writes `specs/queue/<job>.json` (gitignored), and the job runs
only when a person starts `tools/render_queue_worker.py`.

This keeps the lesson from the old HUD, where coupling the UI to the simulation broke both: a fault
in the page cannot reach a running simulation. Agents still have no launch path either; they can
`research_propose_spec`, and a person decides. A test asserts the server has no run endpoint.

## Which runs can be re-run
- **Spec runs** (`spec.json` present): any of them.
- **The a\* probe harnesses** (`feb_gain_ladder_longt`, `feb_astar_confirm`, from their
  `*_summary.json` rows): any cell. The spec is reconstructed as FEB params with
  `param_a = 0.4802 × a_factor`, the per-blob multiseed IC, and the recorded N, K, seed and steps.
- **Everything else:** the page says re-running is not supported yet. Adding a mapping is one function
  in `viewer_data.rerun_spec`.

## Verified (2026-10-07, in the browser and in tests)
- **Gallery:** 207 runs.
- **a\* final states:** 96³ fields shown at 48³ with slices and isosurfaces.
- **er(t) suggestions:** only the genuine start-up transient is suggested; the long steady decay
  gets none.
- **Recorded run (KG):** a 61-frame recording plays back.
- **End-to-end:** window picked → preview → queued → `render_queue_worker.py --once` → 46 steps
  (stopped at the window end, out of 600) and 47 frames → shown in the gallery as `unindexed`.

## Bugs found while building
| bug | effect | fix |
|---|---|---|
| `.gitignore` line `UI/*` (for the legacy `UI/` folder) matched `ui/` case-insensitively on Windows | **the spec editor page from Phase E was never committed**, so the pushed spec editor had no page | the pages moved to `web/`; `test_web_pages_are_not_gitignored` |
| plotly.js has no built-in "Magma" or "Phase" scale | slices silently fell back to a red-white scale | colour scales defined explicitly (magma, cyclic phase, diverging Re/Im) |
| 3-D plots captured the mouse wheel | scrolling the page zoomed the plot instead | `scrollZoom: false` |
| windows from \|dy/dt\| | arbitrary suggestions along a smooth decay | curvature, an absolute floor, and one window per contiguous episode |
| a window re-run integrated to the original T | paid for the whole run to see its start | stop at the window end |
| the frame estimate ignored step rounding | estimated 40 frames, wrote 47 | the estimate uses the step-rounded interval |

## Not done yet
- **Re-run mappings for other harness families** (TG, C2/C3): they need their IC reconstruction.
- **True volume ray-marching:** the 3-D view uses nested isosurfaces. Ray-marching would need a WebGL
  shader page, e.g. three.js.
- **Starting a recording mid-run from a saved state.** This would let the first t0 seconds be
  skipped, but no harness saves mid-run states yet.

## Associated docs
- [[IMPLEMENTATION_PLAN_2026-10]] · [[EXPERIMENT_SPECS_MCP_UI]] · [[TELEMETRY_STREAM]] · [[SOLVER_AND_RUNTIME_CHANGELOG]]

## Branches
- [[Branch - Validation - Index]]
