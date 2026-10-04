---
tags: [plan, infra, validation]
date: 2026-10-04
branch: Branch - Validation - Index
status: running
---

# Implementation plan — October 2026

*Approved 2026-10-04. Progress is tracked in the Phase status table below and in [[SOLVER_AND_RUNTIME_CHANGELOG]].*

## Phase status

| phase | status | record |
|---|---|---|
| A order gates | **complete** 2026-10-04 | [[STEPPER_ORDER_GATES]] |
| B stale flags + re-validation | B1–B3 complete; **B4 replay running** (launched 2026-10-04 13:58) | [[REVALIDATION_E270CDC_RESULTS]] |
| C telemetry | **complete** (feb/C2.7 wiring deferred until the replay ends) | [[TELEMETRY_STREAM]] |
| D provenance + manifest | **complete** | [[PROVENANCE_AND_HARNESS_REGISTRY]] |
| E specs / MCP / UI | pending | |
| F basin mapping | pending | |


## Context
On 2026-10-02 we found two ETDRK4 bugs (a contour `real()` that is invalid for complex L, and stage c
using `N_a` instead of `N_n`). Together they made the Phase C / C1 / C2 integrator about order 0.6.
Both bugs passed every identity test and the CuPy↔JAX parity check. The fix and an order test are
committed (`e270cdc`, branch `fix/etdrk4-integrator-order`), along with the process plan
`docs/PROCESS_PLAN_2026-10.md` (`7112065`).

This plan schedules what is left of that process plan:
1. order gates for every stepper
2. a stepper field, stale flags and a **full-replay** re-validation of a\* and C2.7
3. a telemetry stream
4. import-hash provenance and the harness manifest
5. the spec layer, then the MCP tools, then the UI
6. basin mapping

Decisions already made: the MCP tools live in the repo's `mcp_server/`, the UI is a standalone local
web page, and the re-validation is a full overnight-plus replay.

Two principles carry over from earlier work:
- **Descriptive, never prescriptive.** `orchestrator/` died because it gated experimentation.
- **The HUD has no buttons.** Nothing new may launch a run or block one.

Each phase is a separate branch and PR, and each closes with a results or record doc in the usual
format (What changed / Issues raised / Associated docs / Branches).

---

## Phase A — Stepper order gates (P1) · ~1 day · no GPU needed

There are three steppers:

| stepper | code | expected order |
|---|---|---|
| ETDRK4 | `solver/core.py`, `jax_scout/physics.py:step` | 4 |
| C3 KG Strang split | `jax_scout/phase_d_c3_wave.py:kg_evolve` | 2 |
| TG RK4 | `jax_scout/gravity_TG_B1S_state_load_feedback_gpu.py:rk4_step` | 4 |

**1. New `tests/test_stepper_order_jax.py`**
- Uses dt ladders on tiny grids (N=12–16, short T) and follows the existing design rule in
  `tests/test_physics_identities.py`: **exercise the harness code path; never re-implement it.**
- Order tests:
  - ETDRK4 dissipative and `kinetic_mode="conservative"`: threshold > 3.5.
  - KG Strang: threshold > 1.9, plus an energy drift that scales as dt².
  - TG RK4: threshold > 3.7.
- Manufactured-solution (MMS) tests:
  - Each test wraps the stepper's nonlinear term with a stage-aware source term S(t), where
    S = ∂ₜψ* − Lψ* − N(ψ*) for a chosen smooth ψ*(x,t).
  - Stage times: ETDRK4 c = 0, ½, ½, 1; RK4 likewise; Strang uses the kick times.
  - Assertion: the error against ψ* converges at the expected order.
- These wrappers live in the test file only. No production API change.
- The existing `tests/test_etdrk4_order.py` (CuPy + numpy) stays as the CuPy-side gate.

**2. CI** — `.github/workflows/physics-identities.yml`
- Add `solver/**` and `tests/test_stepper_order_jax.py` to the `paths` filters.
- Add the new file to the pytest line. CPU JAX x64 is already set up there.

**3. `tools/mutation_probe.py`** — add a `stepper` bug class (existing `MUTATIONS` tuple format):

| mutation | expected result |
|---|---|
| `etdrk4_stage_c_uses_Na` | caught |
| `etdrk4_contour_half_real` | caught |
| `kg_strang_kick_asymmetric` (full kick, no half-kicks) | caught (drops to order 1) |
| `tg_rk4_stage_weight` | caught |

Target: zero survivors in the `stepper` class. Record the result in an H3 addendum.

**Verify:**
- WSL: `JAX_PLATFORMS=cpu ~/jax_irer/bin/python -m pytest tests/test_stepper_order_jax.py tests/test_physics_identities.py`
- `.venv`: `pytest tests/test_etdrk4_order.py`
- WSL: `tools/mutation_probe.py --only stepper` → 4/4 caught.

---

## Phase B — Stepper field, stale flags, full-replay re-validation (P3) · ~1 day engineering + ~20 h GPU

### B1. Record the stepper
- In `jax_scout/provenance.py:stamp()`, add `stepper`: a list inferred from `sys.modules`.
  - `solver.core` or `jax_scout.physics` → `ETDRK4`
  - `jax_scout.phase_d_c3_wave` → `KG-strang`
  - the TG module → `TG-RK4`
- Mirror it in `flat_stamp()` as `stepper`.
- For runs that predate the field, `tools/build_results_index.py` infers the stepper from a static
  **AST import scan** of each harness file (transitive within the repo, no execution). Record
  `stepper_source = recorded | inferred`.

### B2. Fixes registry and stale view
- New `docs/registry/COMPONENT_FIXES.json`, hand-maintained, one entry per instrument fix:
  `{id, component:"ETDRK4", fix_commit:"e270cdc", affects_steppers:["ETDRK4"], doc, summary}`.
- `build_results_index.py` additions:
  - a `stepper` column on `runs`
  - a `component_fixes` table
  - a `v_stale_runs` view: runs whose stepper is affected, whose `code_epoch` is not a descendant of
    `fix_commit` (via `git merge-base --is-ancestor`, cached), and that have no
    `edges(rel='revalidated_by')`. Status is `STALE_PENDING_REVALIDATION`.
- **Descriptive only.** It is a view, not a gate.
- Backfill: one `COMPONENT_FIXES` entry for e270cdc. Optionally add the three earlier ledger bugs
  (C2.6, C2.8b, C3) with their fix commits, so the view also covers history.

### B3. Pre-replay gates (must pass before any GPU hours)
1. WSL: the Phase A tests plus `tests/test_etdrk4_order.py` under JAX. This is the first execution
   of the JAX-mirror edit.
2. `tools/solver_parity_check.py`: CuPy↔JAX parity on the fixed code. This is a sanity check, not
   independence evidence.
3. Add a `--dt` flag (default = current module constant, so behaviour is unchanged) to
   `jax_scout/feb_gain_ladder_longt.py` and `jax_scout/feb_astar_confirm.py`, threaded through
   `core_saturation_search`. Today dt comes from `afield_current_coupled.dt`.

### B4. Full replay
Run IDs are suffixed `_REVAL_e270cdc`. Use WSL `~/jax_irer` on the GTX 1080, detached and
checkpointed, with the original arguments read from the old `summary.json`.

| # | run | original | est. |
|---|---|---|---|
| 1 | `feb_gain_ladder_longt.py` N=96, T=72000, K=6, seed 20260619, 6 gains | `FEB_GAIN_LADDER_LONGT_T72000_20260701_175708` | ~6–7 h |
| 2 | `feb_astar_confirm.py`, all 7 cells (T144k ×1.15 / ×1.125, seeds 620/621, bracket ×1.16/×1.175/×1.20) | `FEB_ASTAR_CONFIRM_20260702_003055` | ~9.6 h |
| 3 | **New:** ×1.15 cell at dt/2, T=72000 — proves the verdict is dt-converged | — | ~2.5 h |
| 4 | `phase_d_c2_7_rederivation.py` (R0 CFL re-baseline, R2, R3 N=96 boosts) | `C27_REDERIVE` | ~0.8 h |
| 5 | C1 spot-check: re-run the `c1_longT_confirm` setting | `c1_longT_confirm.json` | ~1 h |

**Decision rules, fixed before the runs** (this is the "derive before you measure" step):

| result | unchanged if … | otherwise |
|---|---|---|
| a\* | the late-slope zero crossing stays in (×1.15, ×1.16), the ×1.15 slope satisfies \|slope\| < 0.001 per 1k at T=144k, all 3 seeds keep slope ≈ 0, and the dt/2 cell agrees in sign and to within 0.0005 | `A_STAR_SHIFTED`: report the new bracket |
| C2.7 | v/2Dk ∈ [0.999, 1.001] for n = 1, 2, and mass ≥ 0.999 | `C2_7_SHIFTED` |
| C2 norm drift | Re-measure. Expected to shrink, since Codex's "timestep-sensitive" note fits the bug. | — |

**Close-out:**
- New `docs/REVALIDATION_E270CDC_RESULTS.md`.
- Add `revalidated_by` edges.
- Update `IRER_MASTER_HYPOTHESIS_CATALOG.md`: §10 instrument-integrity ledger (bugs #4 and #5), plus
  the a\* and C2.7 rows.
- Update the evidence package (`03`/`07`), the memory files, and the §3 table in
  `PROCESS_PLAN_2026-10.md`.

**Verify:**
- `build_results_index.py` lists the pre-fix ETDRK4 runs in `v_stale_runs` and **no** KG/TG runs.
- After close-out, the a\* and C2.7 lineage rows drop out of the view.

---

## Phase C — Telemetry stream (P6, part 1) · ~1 day · scoped in SESSION_SYNTHESIS §5

**1. `jax_scout/snapshots.py:SnapshotWriter`**
- Add `telemetry(t, **scalars)`. It appends one JSON line to `<run>/telemetry.jsonl` through the
  existing bounded queue and writer thread.
- It never blocks; drops are counted in `n_dropped`.
- Lines are flushed whole, so a torn last line is tolerated by readers.

**2. Invariant declarations**
- A harness passes `invariants={name: tolerance}`, e.g. `charge_rel_drift: 1e-10`,
  `ledger_residual_abs: …`. These are written once as `telemetry_meta.json`.
- The values are the scalars the harnesses already compute (`ledger_residual_abs`,
  `profile_overlap`, charge, energy, `boundary_flux_proxy_max`). Nothing new is computed.

**3. `tools/hud_monitor.py`**
- Tail `telemetry.jsonl` and plot the invariant residuals live beside the fields, with the declared
  tolerance as a line.
- Breaches are flagged in the rendered output only. Still no control path.

**4. Wiring**
- Wire the ~6 harnesses that produce catalogued results: TG B1S/B2 (`gravity_TG_B1S_*_gpu.py`,
  `gravity_TG_B2_*`), `feb_gain_ladder_longt.py`, `feb_astar_confirm.py`,
  `phase_d_c2_7_rederivation.py`, `core_saturation_search.py`.
- **Ideally done before the B4 replay starts**, so the 20 h of runs are watchable. If that slips,
  B4 runs as-is.

**Verify:**
- Extend `tests/test_snapshots.py`: telemetry never blocks under a full queue, a reader tolerates a
  partial line, and line counts match samples when nothing is dropped.
- Watch a 5-minute TG run live in `hud_monitor.py`.

---

## Phase D — Import-hash provenance + harness manifest (P5, part 1) · ~1–2 days

**1. Component hashes in `stamp()`**
- For every module in `sys.modules` whose `__file__` is inside the repo, record the git **blob
  hash**, computed in-process as `sha1("blob <len>\0" + bytes)`, with no subprocess.
- Diff against `git ls-files -s` (one call, cached) to get `dirty_components`.
- New keys: `component_hashes: {relpath: blob}` and `dirty_components: [...]`.
- This makes a run reproducible to the exact files it imported, and lets B2's stale view match on
  *file content* rather than commit ancestry. Upgrade the view to prefer hashes when present.

**2. Harness manifest**
- A module-level `HARNESS = {...}` dict:
  - `id`
  - `branch`
  - `status` (`ACTIVE` | `SUPERSEDED` | `RETIRED` | `PROPOSED`)
  - `superseded_by`
  - `invariants` (shared with Phase C)
  - `produces` (run-id prefixes)
- It is read with an **AST parse, never by import**, so a broken harness still registers.

**3. Registry generator**
- New `tools/build_harness_registry.py` writes `docs/HARNESS_REGISTRY.md` plus a `harnesses` table
  in the results index, joined to runs via provenance `harness`.
- Pattern: `tools/build_doc_lineage.py`.

**4. CI check** (new workflow step)
- A harness that writes to `sweep_runs/` must declare `HARNESS`.
- This checks the declaration exists. It does not gate running.
- Start by adding manifests to the ~6 active harnesses. Everything else is backfilled as
  `status: UNREVIEWED` by the generator rather than edited by hand.

**Verify:**
- Unit tests for blob hashing: it must match `git hash-object`.
- A dirty-file test.
- The registry builds, and every run in the index resolves to a harness row or to `UNKNOWN`.

---

## Phase E — Spec layer → MCP → UI (P5 part 2, P6 part 2, P7) · ~1–2 weeks

### E1. Spec layer
- **Package `irer_specs/`** (pydantic v2, already a dependency through `orchestrator.contracts`):
  - Models: `ExperimentSpec`, `SubstrateRef`, `ProtocolSpec`, `ObserverRef`, `Prediction`,
    `Requirement`.
  - Schemas are exported as JSON Schema to `schemas/*.schema.json`.
- **Component registry `jax_scout/registry.py`**
  - Decorators `@substrate(name, version, params_schema)`, `@observer(...)`, `@protocol(...)`.
  - Initial entries wrap existing code rather than rewriting it:
    - Substrates: `physics.build_operators` + `step` (ETDRK4 dissipative/conservative),
      `build_kg` + `kg_evolve`, TG `rk4_step`.
    - Observers: `phase_d_c3_wave.invariants`, `centroid_x`, `momentum_x`, `stability_metrics`.
    - Protocols: `gaussian`, `qball_petviashvili`, `multiseed_ic`.
- **Spec fields**
  - `id`
  - `substrate {name, version, params}`
  - `protocol {ic, grid {N, L}, dt, T, chunk, stop}`
  - `observers[]`
  - `sweep` (grid or list)
  - `prediction {statement, quantities, expected, tolerance}`, required before launch (P4)
  - `requires[{spec_id, verdict}]`, which **warns, never blocks**
  - `budget_h`
- **Executor `tools/run_spec.py <spec.json>`**
  1. Validate the spec.
  2. Print a warning for each unmet `requires`.
  3. Resolve components.
  4. Run, with telemetry and the Phase D stamp.
  5. Write the run directory: `spec.json`, `summary.json`, `telemetry.jsonl`, and `verdict.json`
     set to `PENDING_REVIEW`.
  6. Auto-score `prediction` against the outputs into `prediction_check`.
- **Index:** a `specs` table, plus `edges` of `instantiates` (spec→run) and `requires` (spec→spec).
  The existing long-format `run_params` / `run_metrics` absorb everything else.
- **Pilot:** re-express C2.7 R3 (~48 min) and one TG-B1S run as specs. **Acceptance:** spec-run
  outputs match the harness outputs to round-off.

### E2. MCP tools (repo `mcp_server/`)
- New module `mcp_server/research_tools.py`, registered in `mcp_server/server.py`. The existing
  legacy-ledger tools are left as they are.
- **Read-only:**
  - `list_runs(substrate, branch, status, stepper)`, `get_run(run_id)`
  - `tail_telemetry(run_id, n)`, `list_stale_runs()`
  - `list_components()`, `get_schema(name)`, `list_specs()`, `get_spec(id)`
  - `harness_registry()`
- **Write-limited:**
  - `propose_spec(spec_json)` validates against the schema and writes to `specs/proposed/` only.
    **No launch tool.**
  - `draft_reading(run_id, text)` writes machine-drafted paired-reading blocks, marked as such.
- Reuse the path whitelist and guards from `mcp_server/config.py` and `mcp_server/guards.py`
  (`McpConfig.is_path_allowed`).
- DeepInfra and OCR agents run through the Librarian's existing `providers.py` / `ocr.py` as clients
  of this MCP. Only connection configuration needs documenting; no Librarian code changes.
- **Verify:** tests in the style of `tests/test_mcp_read_tools.py` over a synthetic index; a
  path-escape test; and a check that `propose_spec` rejects an invalid spec and never writes outside
  `specs/proposed/`.

### E3. Standalone spec UI
- `ui/spec_editor/index.html`: one static page that loads react-jsonschema-form from a CDN. No build
  step.
- `tools/serve_spec_ui.py` (stdlib `http.server`, bound to 127.0.0.1) serves it, with three
  endpoints: GET schemas, GET specs, POST spec (validate, then write to `specs/drafts/`).
- **No launch button**, following the HUD rule. The page shows the `run_spec.py` command to copy.
  It also lists proposed specs from agents for review and promotion (`proposed/` → `approved/`).
- **Verify:** a round-trip test (schema → form → POST → file validates), and a manual edit of the
  C2.7 pilot spec.

---

## Phase F — Basin mapping (P8) · ~2–3 weeks · requires B (fixed, re-validated solver) and E1 (specs)

### F1. Forward ensembles
- A spec with `sweep` over parameter point × IC family (`multiseed_ic` K-counts, gaussian,
  Q-ball) × seed.
- Observer = a final-state descriptor vector: mass, n_nodes, late er-slope, spectral peaks,
  permutation-invariant node-geometry moments, and phase winding.

### F2. Clustering
- New `tools/basin_cluster.py`: standardise the descriptors, run HDBSCAN (or sklearn DBSCAN), and
  produce a per-parameter-point basin table with basin sizes.
- Verification target: at a\* it must recover the known IC-dependent node counts (n_fin 4 at seed
  619 vs 6 at 620/621) as distinct basins.

### F3. Continuation
- New `jax_scout/continuation.py`.
- The core equation is the time-T map residual F(ψ, a) = Φ_T(ψ) − ψ, with phase conditions for the
  ω0 rotation and translations. A period unknown is added for breathing (periodic-orbit) states.
- Solver: matrix-free Newton–Krylov (`jax.jvp` + `jax.scipy.sparse.linalg.gmres`), with
  pseudo-arclength continuation in `param_a`. BifurcationKit.jl is the reference design.
- Pilot: N=32 branch through a\*. **Acceptance:** the branch passes through the B4-validated a\*,
  and the Floquet multipliers change stability at the measured bracket.

**Verify:**
- An ensemble at 3 parameter points reproduces known results.
- The continuation residual stays < 1e-8 along the branch.

---

## Optional Phase G — independent implementation (P2)
An OpenCL + `pyvkfft` Phase C ETDRK4 on the 5500XT at N ≤ 32, written from
`docs/IRER_MATH_REFERENCE.md`, not from the code. Its parity with CuPy is the first truly
independent cross-check. Schedule it whenever there is appetite.

---

## Sequencing and dependencies
```
A (gates) ──► B1/B2 (stale view) ──► B3 gates ──► B4 replay (overnight×2) ──► B close-out
                     C (telemetry) ── ideally before B4 ┘
B ──► D (hashes/manifest) ──► E1 (specs) ──► E2 (MCP) ──► E3 (UI)
B + E1 ──► F (basins)
```

## Critical files
- **Modify:**
  - `jax_scout/provenance.py`, `tools/build_results_index.py`
  - `jax_scout/snapshots.py`, `tools/hud_monitor.py`
  - `.github/workflows/physics-identities.yml`, `tools/mutation_probe.py`
  - `jax_scout/feb_gain_ladder_longt.py`, `jax_scout/feb_astar_confirm.py` (the `--dt` flag)
  - `mcp_server/server.py`
- **New:**
  - `tests/test_stepper_order_jax.py`
  - `docs/registry/COMPONENT_FIXES.json`
  - `tools/build_harness_registry.py`
  - `irer_specs/`, `schemas/`, `jax_scout/registry.py`, `tools/run_spec.py`
  - `mcp_server/research_tools.py`
  - `ui/spec_editor/`, `tools/serve_spec_ui.py`
  - `tools/basin_cluster.py`, `jax_scout/continuation.py`
- **Reuse, don't duplicate:**
  - `provenance.stamp` / `write_json`
  - `SnapshotWriter`'s queue and atomic writes
  - `build_results_index`'s long-format tables and `edges`
  - `McpConfig` path whitelist
  - the `mutation_probe` safety-restore pattern
  - `phase_d_c3_wave.invariants`

## End-to-end verification (after E)
1. Write the C2.7 R3 spec in the UI.
2. `tools/run_spec.py` runs it with the fixed solver; `hud_monitor.py` shows live invariants.
3. The run is indexed with stepper, component hashes and a `prediction_check`.
4. The MCP `list_runs` and `tail_telemetry` tools return it.
5. `list_stale_runs` no longer lists C2.7.
6. CI is green on the order gates; the mutation probe reports 0 stepper survivors.

## Associated docs
- [[PROCESS_PLAN_2026-10]] · [[ETDRK4_INTEGRATOR_BUGS_2026-10]] · [[SOLVER_AND_RUNTIME_CHANGELOG]]

## Branches
- [[Branch - Validation - Index]]
