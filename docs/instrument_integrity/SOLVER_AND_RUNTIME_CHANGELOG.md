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
| 2026-10-04 | CL-007 | MCP research tools (11, fenced writes, no launch); local spec editor; `prune_empty` | no | phase-e |
| 2026-10-04 | CL-006 | Spec layer: `irer_specs`, `jax_scout/registry.py`, `tools/run_spec.py`, `specs/`, index `specs` table | no (new path; harness-equivalent to <1e-10) | `a3cb757` |
| 2026-10-04 | CL-005 | Component-hash provenance; harness manifests + registry; content-level staleness | no (stamp + inert manifests) | `333f17f` |
| 2026-10-04 | CL-004 | Telemetry stream (`TelemetryWriter`), live HUD invariants, TG-B1S/B2-midplane wiring | no (observer only) | `2185d9f` |
| 2026-10-04 | CL-003 | Provenance `steppers`; stale-run view; `--dt`/`--cells` for the a\* harnesses; re-validation driver | no (default paths byte-identical) | `d3d87fd` |
| 2026-10-04 | CL-002 | Stepper order + MMS gates, stepper mutations, CI extension | no (tests/tooling) | `0419ff9` |
| 2026-10-02 | CL-001 | ETDRK4: complex-safe contour coefficients + stage-c `N_n` | **YES** | `e270cdc` |

---

## CL-007 — MCP research tools and spec editor (Phase E2–E3)
- **Branch:** `phase-e/specs-mcp-ui`
- **Files:**
  - `mcp_server/research_tools.py` (new); `mcp_server/server.py` (11 `research_*` tools registered)
  - `jax_scout/registry.py` (`export()`, `__main__`); `docs/registry/components.json` (generated)
  - `irer_specs.prune_empty`
  - `tools/serve_spec_ui.py` (new); `ui/spec_editor/index.html` (new)
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
