---
tags: [record, validation, changelog]
date: 2026-10-04
branch: Branch - Validation - Index
status: running
---

# Solver and runtime changelog

This is the single record of **every change to the solvers, steppers, provenance or run-time
plumbing**, starting October 2026. The newest entry goes at the top.

Each entry states:
- what changed, with commit and files
- why it changed
- whether it changes numerical output
- how the change was verified
- which earlier results it affects

**Rule:** a commit that touches `solver/`, a stepper in `jax_scout/`, `jax_scout/provenance.py`,
`jax_scout/snapshots.py` or a runtime tool is not done until it has an entry here.

**Output-changing?** means the change alters the numbers a run produces. "No" means it changes only
bookkeeping, telemetry or tests.

---

## Index

| date | id | summary | output-changing? | commit |
|---|---|---|---|---|
| 2026-10-08 | CL-015 | QD explorer `tools/qd_explore.py` (CMA-MAE + boundary sampler, background-safe); `BatchedETDRK4` compiled-function cache | no (cache returns the identical compiled function; explorer is a new driver) | qd branch |
| 2026-10-08 | CL-014 | `tools/screen_verify.py`: fp32 screen → choose boundary/outlier/representative points → fp64 verify spec → compare | no (analysis tool; never launches) | screen-verify branch |
| 2026-10-07 | CL-013 | `protocol.precision` (fp64 default \| fp32 screening, ETDRK4 only); `--precision` override; fp32 runs flagged in summary + gallery | no for fp64 (default path unchanged); fp32 is opt-in, ~0.1–0.5% drift | precision branch |
| 2026-10-07 | CL-012 | Batched (vmap) spec sweeps; `physics.operator_args`/`ops_from_args` refactor (byte-identical); provenance git state cached per process | no (batched ≡ single to 1e-10) | batch branch |
| 2026-10-07 | CL-011 | Run gallery + viewer; `protocol.record` (whole run or window); render queue + worker; pages moved `ui/`→`web/` (gitignore bug) | no (recording is opt-in) | viewer branch |
| 2026-10-05 | CL-010 | Continuation: jvp matvec replaces `jax.linearize` (N³ memory → constant) | no | `4a482df` |
| 2026-10-05 | CL-009 | Retired 3 launch-capable legacy MCP tools; pinned `mcp<2` | no | main |
| 2026-10-04 | CL-008 | Basin mapping: `state_descriptors`, `basin_cluster.py`, `continuation.py` (Newton–Krylov + arclength + Floquet), a\* pilot, post-replay queue | no (new analysis path) | phase-f |
| 2026-10-04 | CL-007 | MCP research tools (11, fenced writes, no launch); local spec editor; `prune_empty` | no | `4ddff4d` |
| 2026-10-04 | CL-006 | Spec layer: `irer_specs`, `jax_scout/registry.py`, `tools/run_spec.py`, `specs/`, index `specs` table | no (new path; harness-equivalent to <1e-10) | `a3cb757` |
| 2026-10-04 | CL-005 | Component-hash provenance; harness manifests + registry; content-level staleness | no (stamp + inert manifests) | `333f17f` |
| 2026-10-04 | CL-004 | Telemetry stream (`TelemetryWriter`), live HUD invariants, TG-B1S/B2-midplane wiring | no (observer only) | `2185d9f` |
| 2026-10-04 | CL-003 | Provenance `steppers`; stale-run view; `--dt`/`--cells` for the a\* harnesses; re-validation driver | no (default paths byte-identical) | `d3d87fd` |
| 2026-10-04 | CL-002 | Stepper order + MMS gates, stepper mutations, CI extension | no (tests/tooling) | `0419ff9` |
| 2026-10-02 | CL-001 | ETDRK4: complex-safe contour coefficients + stage-c `N_n` | **YES** | `e270cdc` |

---

## CL-015 — Quality-diversity explorer, and a batch compile cache
- **Branch:** `feat/qd-explorer`
- **Files:**
  - `tools/qd_explore.py`
    - Drives `run`, `status`, `pause`, `resume`, `stop` and `export`.
    - CMA-MAE (pyribs) fills a behaviour archive; an extra-trees boundary sampler adds points where
      the regime is uncertain; each evaluation gets a fresh IC seed.
    - State lives in one fsync'd `evals.jsonl`, with the archive rebuilt from it on resume.
    - Disables XLA preallocation before JAX can start.
  - `tools/qd_background.ps1` + `tools/qd_run_wsl.sh`: hidden background launch at `nice 10`.
  - `specs/qd/wide-net-v1.qd.json`: an 8-D box containing FEB/a\*.
  - `tests/test_qd_explore.py`. `tests/test_spec_layer.py` now skips `specs/qd/` (explorer configs,
    not specs).
  - `jax_scout/registry.py`: `BatchedETDRK4` reuses one compiled function per (grid, dt, static
    args, dtype), so a long-running driver does not recompile every generation.
- **Why:** basin discovery. See [[QD_EXPLORER]], which also records the review of the old search
  scripts.
- **Output-changing?** No.
  - The cache hands back the same `jax.jit` function, built from the same arguments.
  - The explorer is a new driver; existing paths are unchanged.
- **Verified:**
  - In WSL: 5 QD tests (including a real resumable run whose exported evaluations reproduce through
    `run_spec` to 1e-8), plus 26 spec-layer and screen-verify tests.
  - GPU smoke tests at N=48 and N=96: see [[QD_EXPLORER]] §4. The N=48 run found that resolution
    does not reproduce a\*, so the wide net runs at N=96. Diverged members now leave the batch, and
    `eval_chunk` bounds GPU memory; both are tested to leave survivors' results unchanged.
  - New WSL dependencies: `ribs` 0.12 and `scikit-learn` 1.9.

---

## CL-014 — Screen-then-verify helper
- **Branch:** `feat/screen-verify`
- **Files:**
  - `tools/screen_verify.py`
    - `plan` selects screen points by `boundary`, `ic_split`, `outlier`, `representative` and
      `screen_failed`, and writes one fp64 zip-sweep spec to `specs/proposed/` plus a
      `verify_plan.json`.
    - `compare` assigns each fp64 end state to a screen basin, and writes `VERIFY.md` and
      `verify.json`.
  - `tests/test_screen_verify.py`: point selection, exact rebuilding of the points, agreement and
    disagreement, the structure warning, and a real end-to-end fp32→fp64 run.
- **Why:** to make fp32 screening ([[BATCHED_RUNS]]) safe to use for basin maps.
- **Output-changing?** No. It is an analysis tool and never launches a run.
- **Verified:**
  - Tests: 4 in `.venv`, and the end-to-end test in WSL.
  - First real use on the a\* neighbourhood: **17/17 AGREE**. The fp32→fp64 descriptor distance was
    at most 0.008, against a basin radius of 0.2.
  - That run also found that N=32 does not resolve a\* states. See [[SCREEN_AND_VERIFY]].

---

## CL-013 — Precision option for screening runs
- **Branch:** `feat/precision-option`
- **Files:**
  - `irer_specs/__init__.py` + `schemas/experiment_spec.schema.json`: `protocol.precision`, enum
    `fp64` | `fp32`, default `fp64`. It can be a sweep axis (`protocol.precision`).
  - `jax_scout/registry.py`:
    - `DTYPES` and `PRECISIONS`; only `etdrk4-sncgl` offers fp32. `check_spec` reports, and
      `build` refuses, fp32 on KG/TG.
    - The ETDRK4 Sim passes the dtypes to the existing dtype-parametric `physics.build_operators`.
      **No solver code changed.**
    - `batch_key` includes precision, so fp32 and fp64 never share a vmapped batch.
    - `max_batch` counts 8 bytes per complex point for fp32.
  - `tools/run_spec.py`: `--precision` overrides every point; `summary.json` gains `precision`; a
    note is printed for fp32 runs.
  - `tools/viewer_data.py` + `web/viewer/index.html`: gallery chip "fp32 screen" or "mixed screen".
  - tests: KG rejects fp32; an fp32 sweep runs in complex64, is batched apart from fp64, and matches
    fp64 to 1e-3 over a short run; the viewer flag.
- **Why:** fp32 is 3.7–4.5× faster on the GTX 1080 (see [[BATCHED_RUNS]]), enough to screen many
  more configurations for basin mapping.
- **Output-changing?** No for any existing spec: the default is fp64 and that path is unchanged.
  fp32 is opt-in and drifts about 0.1–0.5% from fp64 over an a\* replay. **fp32 numbers are
  screening results only.** Brackets, continuation and catalogued values must come from fp64 runs.
- **Verified:** 21 spec-layer tests in WSL (CPU), plus viewer, spec-UI and MCP tests: 50 passed.

---

## CL-012 — Batched sweeps, and a provenance speed fix
- **Branch:** `batch/vmap-executor`
- **Files:**
  - `jax_scout/physics.py`: `BATCHABLE_PARAMS`, `operator_args`, `ops_from_args`. `build_operators`
    is now a two-line wrapper around them. **Solver-adjacent:** the operator construction code is
    unchanged, and all 60 `Ops` arrays are byte-identical before and after on both branches.
  - `jax_scout/registry.py`: `batch_key`, `BatchedETDRK4`, `max_batch`
  - `tools/run_spec.py`: the executor is split into `_setup_member` / `_run_members` / `_finalize`,
    with a single run being a batch of one. Adds `run_batch`, `plan_batches`, `schedule_key`,
    `--no-batch` and `--batch-size`.
  - `jax_scout/provenance.py`: `_git_state()` caches commit, status and branch once per process
  - tests: two new spec-layer tests (batched ≡ point-by-point to 1e-10; batch planning)
- **Why:** more configurations per GPU-hour for basin mapping. See [[BATCHED_RUNS]].
- **Output-changing?** No.
  - Batched members reproduce single runs to 1e-10.
  - Provenance records the same fields, read once instead of for every file. A run that edits the
    repo mid-run would now be stamped with its start-of-run state, which is the correct reading
    anyway.
- **Verified:**
  - 19/19 spec-layer tests in WSL; 62 related tests in `.venv`.
  - Full-suite failure set: no new failures, and 3 fewer than the baseline (MCP tests now pass).
  - Benchmarks are in [[BATCHED_RUNS]]: the profile showed 85% of a small sweep was git calls.

---

## CL-011 — Run gallery, viewer, recording, render queue
- **Branch:** `viewer/run-gallery`
- **Files:**
  - `tools/viewer_data.py`, `tools/serve_viewer.py`, `tools/render_queue_worker.py`,
    `web/viewer/index.html` (all new)
  - `irer_specs` schema: `protocol.record`
  - `tools/run_spec.py`: an event-scheduled loop that writes recorded frames to `<run>/history/`, and
    `summary.history`
  - `web/spec_editor/index.html`: moved from the never-committed `ui/spec_editor/`;
    `tools/serve_spec_ui.py` path updated
  - `.gitignore`: `specs/queue/`
  - tests: `tests/test_viewer.py` (13); a record-window test in `tests/test_spec_layer.py`
- **Why:** to let a user view runs the way an agent can, and to record the dynamic parts of a run.
  See [[RUN_VIEWER]].
- **Output-changing?** No. Without `record`, the executor visits the same sample steps and gives the
  same results. Recording only reads `sim.fields()`.
- **Also fixed:** the Phase E spec editor page was **never committed**, because `.gitignore` `UI/*`
  matched `ui/` on Windows. It is now in `web/`, with a regression test.
- **Verified:**
  - 13/13 viewer tests and 27/27 related UI/MCP tests in `.venv`; 35 spec/snapshot tests in WSL.
  - In the browser: gallery, final-state viewer, history playback, and window re-run →
    queue → worker → result.

---

## CL-010 — Continuation memory fix
- **Commit:** `4a482df`, merged to main
- **Files:** `jax_scout/continuation.py` (`direction` uses a `jax.jvp` matvec);
  `tests/test_continuation.py` (memory regression test)
- **Why:** `jax.linearize` through a 1000-step `fori_loop` keeps every step's intermediates. The
  compiled scratch memory was 1.3 GiB at N=8, 10.5 GiB at N=16 and about 86 GiB at N=32, which is
  why the a\* pilot hit an 81 GiB OOM. With the jvp matvec it is 0.8 MiB at N=8 and 6.2 MiB at N=16.
- **Output-changing?** No. It is the same Jacobian-vector product, computed by forward mode.
- **Verified:** 2/2 continuation tests pass. The a\* pilot at N=32 now runs at about 265 MiB of GPU
  memory.

---

## CL-009 — Legacy MCP launch tools retired; MCP dependency pinned
- **Branch:** `main` (after merge `c646dd5`)
- **Files:**
  - `mcp_server/server.py`: the `run_simulation_manifest`, `run_smoke_simulation` and
    `validate_artifact` MCP registrations are removed and replaced by a note explaining why
  - `tests/test_mcp_write_tools.py`: now asserts that no launch tool is registered (20 tools)
  - `requirements.txt`: `mcp>=1.20,<2`
- **Why:**
  - Agents may read and propose but never launch (PROCESS_PLAN P6). These tools could start GPU/CPU
    work, and they targeted the retired orchestrator pipeline.
  - mcp 2.x renamed `FastMCP` → `MCPServer`, which breaks `server.py`. The `.venv` now has mcp 1.30.0.
- **Output-changing?** No. The implementations remain in `write_tools.py` for direct human use.
- **Verified:** 50/50 MCP tests pass; the server imports and lists 20 tools.

---

## CL-008 — Basin mapping tools (Phase F)
- **Branch:** `phase-f/basin-mapping`
- **Files:**
  - `jax_scout/registry.py`: `state_descriptors` v2
  - `tools/run_spec.py`: `final_only`
  - `tools/basin_cluster.py` (new)
  - `jax_scout/continuation.py` (new)
  - `jax_scout/astar_continuation_pilot.py` (new, with a `HARNESS` manifest)
  - `specs/approved/astar-basin-ensemble.json`
  - `tools/revalidation/post_replay_queue.sh`, `tools/revalidation/compare_e270cdc.py`
  - tests: `test_basin_cluster.py` (3), `test_continuation.py` (1, now in CI)
- **Why:** to find distinct basins by forward ensembles and continuation instead of the blending
  GA. See [[BASIN_MAPPING]].
- **Output-changing?** No. These are new analysis tools, and the solver is untouched.
- **Verified:**
  - Continuation converges quadratically to the uniform S-NCGL state (ρ\* to 1e-12, θ to 1e-14).
  - The clusterer recovers the planted basins on synthetic data, and the KG demo gives the predicted
    2 basins.
  - Limits found and documented: trivial-solution attraction; a singular family Jacobian for
    conservative solitons.

---

## CL-007 — MCP research tools and spec editor (Phase E2–E3)
- **Commit:** `4ddff4d` · **Branch:** `phase-e/specs-mcp-ui`
- **Files:**
  - `mcp_server/research_tools.py` (new); `mcp_server/server.py` (11 `research_*` tools registered)
  - `jax_scout/registry.py` (`export()`, `__main__`); `docs/registry/components.json` (generated)
  - `irer_specs.prune_empty`
  - `tools/serve_spec_ui.py` (new); `web/spec_editor/index.html` (new)
  - `tests/test_mcp_research_tools.py` (10 tests); `tests/test_spec_ui.py` (5 tests)
- **Why:** agent oversight through a fenced channel, and a schema-driven editor. See
  [[EXPERIMENT_SPECS_MCP_UI]].
- **Output-changing?** No. There is no new compute path and no launch path.
- **Verified:**
  - 27 pass, 4 skipped (JAX) in `.venv`.
  - In-browser test: open, validate, new spec, save draft. It found and fixed the empty-default bug.

---

## CL-006 — Spec layer (Phase E1)
- **Commit:** `a3cb757` · **Branch:** `phase-e/specs-mcp-ui` (dev worktree)
- **Files:**
  - `irer_specs/__init__.py` (new): the schema, a dependency-free validator, sweep expansion and
    prediction scoring
  - `schemas/experiment_spec.schema.json` (generated)
  - `jax_scout/registry.py` (new): three substrates (`etdrk4-sncgl`, `kg-strang`, `tg-rk4`), ICs
    (`gaussian`, `multiseed`, `qball`, `nls_soliton`, `load_npz`) and observers (`mass`, `amp`,
    `centroid`, `nodes`, `kg_invariants`, `tg_diagnostics`, `energy_ratio`). All of them wrap
    existing gated code; nothing is re-implemented.
  - `tools/run_spec.py` (new): the generic executor
  - `specs/approved/{astar-probe-pilot, c27-r3-soliton-transport-n1}.json`
  - `tools/build_results_index.py`: `specs` table plus `instantiates` / `requires` edges
  - `tests/test_spec_layer.py` (new, 16 tests); the CI workflow runs it
- **Output-changing?** **No.** This is a new execution path. **Equivalence:** the a\* probe run as a
  spec reproduces `core_saturation_search.run_probe`'s er(t) to < 1e-10 (tested).
- **Verified:** 16/16 tests in WSL; 12 pass with 4 skipped in `.venv`, which has no JAX.

---

## CL-005 — Component-hash provenance and harness registry (Phase D)
- **Commit:** `333f17f` · **Branch:** `phase-d/provenance-manifest`, developed in a separate worktree (`F:\quantule_mapper_dev`)
  so the running B4 replay's tree stayed fixed.
- **Files:**
  - `jax_scout/provenance.py`: `git_blob_hash`, `component_hashes`; `stamp()` gains
    `component_hashes` / `dirty_components`; `flat_stamp()` gains `dirty_components` / `n_components`
  - `tools/stepper_staleness.py`: `blob_contains_fix`, `hash_verdict`, content-first staleness
  - `docs/registry/COMPONENT_FIXES.json`: `affects_files`
  - `tools/build_harness_registry.py` (new); `docs/registry/harness_registry.json` and
    `docs/research_infrastructure/HARNESS_REGISTRY.md` (generated)
  - `tools/build_results_index.py`: `harnesses` table, `runs.harness` = harness id, `produced_by` edges
  - `HARNESS = {...}` manifests added to 8 harnesses: `feb_gain_ladder_longt`, `feb_astar_confirm`,
    `core_saturation_search`, `phase_d_c1_transport`, `phase_d_c2_7_rederivation`,
    `gravity_TG_B2_midplane_stress_flux`, `gravity_TG_B1S_state_load_feedback_gpu`,
    `gravity_TG_B1S_H2_box_ladder`
  - `.github/workflows/harness-manifest.yml` (new)
  - `tests/test_harness_registry.py` (new); `tests/test_stepper_staleness.py` (git-history skip marker)
- **Why:** to pin runs to exact file contents and to give harnesses a lifecycle record. See
  [[PROVENANCE_AND_HARNESS_REGISTRY]].
- **Output-changing?** **No.** The manifests are inert literals. The stamp adds keys to the summary
  JSON: `component_hashes` (one entry per imported repo module) and `dirty_components`.
- **Verified:**
  - Blob hashes equal `git hash-object`.
  - 21/21 Phase D and staleness tests pass on Windows. In WSL, 62 pass across identities, order
    gates, snapshots, registry and staleness, with 6 git-history tests skipped.
  - The full-suite failure set is unchanged.
  - Every manifested script passes `compile()`.

---

## CL-004 — Telemetry stream (Phase C)
- **Commit:** `2185d9f` · branch `phase-c/telemetry`
- **Files:**
  - `jax_scout/snapshots.py`: `TelemetryWriter`, `read_telemetry`, `invariant_breaches`
  - `tools/hud_monitor.py`: renders `telemetry*.png` per arm and prints breaches with the harness note
  - `jax_scout/gravity_TG_B2_midplane_stress_flux.py`: `_live_ledger()`, `run_arm(..., tel=None)`,
    a per-arm writer in `main()`
  - `jax_scout/gravity_TG_B1S_state_load_feedback_gpu.py`: `run_arm(..., telemetry_dir=None)`, wired
    in `main()`
  - `tests/test_snapshots.py`: 7 new tests
- **Why:** to watch the invariants of long runs live. See [[TELEMETRY_STREAM]].
- **Output-changing?** **No.** It only records scalars that are already on the host. The midplane
  smoke run's gates (G1/G2/G3) are unchanged with telemetry on. Each harness run now also writes
  `telemetry/<arm>/telemetry.jsonl` and `telemetry_meta.json`.
- **Verified:**
  - `test_snapshots.py` 19/19, identities 36/36 (CPU).
  - Midplane GPU smoke run: 0 breaches with the running-mean scale. The per-sample scale gave 8
    false breaches and was replaced.

---

## CL-003 — Stepper provenance, stale-run view, re-validation options (Phase B1–B3)
- **Commit:** `d3d87fd` · branch `phase-b/stale-flags-revalidation`
- **Files:**
  - `jax_scout/provenance.py`: `STEPPER_MODULES` and `steppers_loaded()`. `stamp()` gains
    `steppers`; `flat_stamp()` gains `steppers` as a comma string.
  - `tools/stepper_staleness.py` (new): resolves each run's steppers (recorded → harness AST scan →
    substrate → run-id prefix) and its staleness per fix (git ancestry, falling back to date).
  - `tools/build_results_index.py`: `runs.steppers` / `runs.stepper_source`, the `component_fixes`
    and `run_staleness` tables, the `v_stale_runs` view and `revalidated_by` edges. It also now reads
    the commit from a nested `provenance` stamp.
  - `docs/registry/COMPONENT_FIXES.json` (new): one entry, `etdrk4-2026-10`.
  - `jax_scout/core_saturation_search.py`: `run_probe(..., dt=None)` and `dt_ratio()`.
  - `jax_scout/feb_gain_ladder_longt.py`, `jax_scout/feb_astar_confirm.py`: `--dt`, `--cells`, and a
    provenance stamp in the summary JSON.
  - `tools/revalidation/reval_e270cdc.sh` (new): the replay driver.
  - `tests/test_stepper_staleness.py` (new, 11 tests).
- **Why:** to flag runs touched by a fixed stepper by query rather than by reading harnesses, and to
  run the full-replay re-validation. See [[REVALIDATION_E270CDC_RESULTS]].
- **Output-changing?** **No.** With no `--dt`, run_probe uses the historical `DT`, and the harness
  loops and keys are unchanged. The only addition to their output is the `provenance` / `dt` /
  `dt_ratio` keys in the summary JSON.
- **Verified:**
  - 11/11 staleness tests pass.
  - Results index: 102 ETDRK4 runs flagged stale, 0 KG/TG runs flagged.
  - The full suite shows the same failing-test set as before.
  - `--dt` smoke test: `er_fin` agrees to 1.4e-6 at dt/2.
  - Parity: 1.7e-12 with real L, 2.1e-12 with complex L.

---

## CL-002 — Stepper order gates (Phase A)
- **Commit:** `0419ff9` · branch `phase-a/stepper-order-gates`
- **Files:**
  - `tests/test_stepper_order_jax.py` (new)
  - `tools/mutation_probe.py` (6 `stepper` mutations; `TESTS` is now a list; new `--tests` option)
  - `.github/workflows/physics-identities.yml` (also triggers on `solver/**`; runs the order gates)
- **Why:** to give accuracy a guard, since identity tests cannot see it. See [[STEPPER_ORDER_GATES]].
- **Output-changing?** No. No solver or harness code was modified.
- **Verified:**
  - 10/10 new tests pass in WSL `~/jax_irer` on CPU (about 40 s).
  - Mutation probe: 16/16 caught. With identities only, all 6 stepper mutations survive.
  - First execution of the JAX ETDRK4 fix: order 4.0–4.08.

---

## CL-001 — ETDRK4 coefficient and stage-c fix
- **Commit:** `e270cdc` · branch `fix/etdrk4-integrator-order`
- **Files:**
  - `solver/etdrk4_coeffs.py` (new)
  - `solver/core.py` (coefficient construction now calls `etdrk4_coefficients`)
  - `solver/kernels.py` (`compute_kt_stage_c(E2, a_k, Q, N_b, N_n)`)
  - `jax_scout/physics.py` (full-circle contour, complex coefficients cast to `cd`, stage c uses `n_n`, `KT_CONTOUR_M` 64 → 128)
  - `tools/conservative_geometry_campaign.py`
  - `tools/conservative_stepper_contract_audit.py`
  - `tests/test_etdrk4_order.py` (new)
- **Why:** two bugs made ETDRK4 roughly order 0.6. See [[ETDRK4_INTEGRATOR_BUGS_2026-10]].
- **Output-changing?** **Yes**, for every run on an ETDRK4 path: Phase C dissipative, C1 dispersive
  and the C2 conservative NLS branch.
  - At the production dt=0.005: ψ error after T=2 goes from 8.5% to 0.27% (N=48).
  - The coefficients are now complex128 rather than float64, which doubles their memory: 4 extra
    N³ complex arrays, about 32 MB each at N=128.
- **Not affected:** `jax_scout/phase_d_c3_wave.py` (KG split-step) and every TG harness.
- **Verified:**
  - `tests/test_etdrk4_order.py`: 5 tests pass. The CuPy step test fails at order 2.01 when the old
    stage c is put back.
  - GPU convergence table in the bug record.
  - The full test suite shows the same 25 failures before and after; all of them predate this change.
- **Not yet verified:** the JAX edit has not been run (no jax in `.venv`). Covered by Phase B3.
- **Results affected:** a\*, C2.7, C2 norm drift, C1. Status: STALE_PENDING_REVALIDATION.

---

## Associated docs
- [[ETDRK4_INTEGRATOR_BUGS_2026-10]] · [[IMPLEMENTATION_PLAN_2026-10]] · [[BASELINE_AUDIT_NUMERICAL]]

## Branches
- [[Branch - Validation - Index]]
