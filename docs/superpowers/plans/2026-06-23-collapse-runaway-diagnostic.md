# Collapse/Runaway Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a post-run, replay-backed diagnostic pass that distinguishes collapse-like runaway, fragmenting blowup, delocalized growth, spin-down decay, and stable outcomes for selected Phase C candidates without modifying the PDE or solver physics path.

**Architecture:** Add one new analysis-only runner under `jax_scout/` that replays saved candidates with the existing solver path, captures sparse snapshots, computes derived geometry/spectral diagnostics after the run, and writes per-candidate JSON plus a run-level CSV summary. Keep plotting/reporting separate from capture so the diagnostic logic can be tested without launching heavy runs.

**Tech Stack:** Python, NumPy, JAX replay through existing `jax_scout.physics` path, existing `jax_scout.transfer_diag` geometry helpers, `unittest`, Markdown docs.

---

### Task 1: Add failing unit tests for diagnostic helpers

**Files:**
- Create: `F:/quantule_mapper/tests/test_core_saturation_collapse_diag.py`
- Modify: `F:/quantule_mapper/tests/test_core_saturation_replay.py` (only if shared fixtures are useful)
- Reference: `F:/quantule_mapper/jax_scout/core_saturation_replay.py`

- [ ] **Step 1: Write failing tests for labeling and metric helpers**

Add tests that define small synthetic metric traces and assert:
- `COLLAPSE_LIKE_RUNAWAY` requires rising peak density, shrinking radius, rising compactness, geometry spike, no split, and no high-k artifact flag.
- `FRAGMENTING_BLOWUP` wins when node count rises before failure.
- `DELOCALIZED_GROWTH` wins when energy grows without strong localization.
- `SPIN_DOWN_DECAY` and `SATURATED_BOUND_STATE` map from saved class-compatible conditions.

- [ ] **Step 2: Run the new test file to verify it fails**

Run:

```powershell
python -m unittest tests.test_core_saturation_collapse_diag -v
```

Expected: FAIL because the diagnostic module does not exist yet.

- [ ] **Step 3: Add a failing test for candidate-group selection from CSV rows**

Cover the required groups:
- high-target K=1 blowups
- high-target K=6 TRUEs
- low-target K=1 TRUE/SPIN_DOWN
- baseline K=1/K=6 rows

- [ ] **Step 4: Run the targeted test again**

Run:

```powershell
python -m unittest tests.test_core_saturation_collapse_diag -v
```

Expected: FAIL on missing functions or wrong imports.

- [ ] **Step 5: Commit**

Do not commit yet; batch with implementation once the code is green.

### Task 2: Implement the analysis-only diagnostic runner

**Files:**
- Create: `F:/quantule_mapper/jax_scout/core_saturation_collapse_diag.py`
- Modify: `F:/quantule_mapper/jax_scout/core_saturation_replay.py` (only if a shared replay/capture helper avoids duplication)
- Reference: `F:/quantule_mapper/jax_scout/core_saturation_search.py`
- Reference: `F:/quantule_mapper/jax_scout/transfer_diag.py`

- [ ] **Step 1: Implement pure helper functions first**

Implement helpers for:
- loading candidate rows and selecting required groups
- sparse replay capture using the existing solver path
- computing per-snapshot metrics: `rho_peak`, node count, participation ratio, core radius, mass-inside-radius, compactness proxy, `omega_sq` minimum, `|grad log Omega|`, curvature proxy, high-k spectral fraction
- deriving summary flags like `split_before_blowup` and `time_to_blowup`
- assigning cautious diagnostic labels

- [ ] **Step 2: Implement the CLI wrapper**

Support a direct analysis command with explicit inputs, for example:

```powershell
python jax_scout/core_saturation_collapse_diag.py --outdir <dir>
```

The script should internally analyze:
- `sweep_runs/CORE_SAT_HUNT_20260623_123527`
- `sweep_runs/CORE_SAT_HUNT_20260623_120758`
- `sweep_runs/CORE_SAT_HUNT_20260623_113318`
- `ref_feb56dc7`

- [ ] **Step 3: Save machine-readable outputs**

Write:
- `collapse_runaway_diagnostics.csv`
- per-candidate `diagnostic_summary.json`
- optional `frames.npz` only for analyzed candidates when needed

- [ ] **Step 4: Run unit tests**

Run:

```powershell
python -m unittest tests.test_core_saturation_collapse_diag -v
python -m unittest tests.test_core_saturation_replay -v
```

Expected: PASS

- [ ] **Step 5: Commit**

Do not commit yet; wait until the docs and report pass are also complete.

### Task 3: Run the diagnostic pass and write the science report

**Files:**
- Create or modify: `F:/quantule_mapper/docs/PHASE_C_COLLAPSE_RUNAWAY_DIAGNOSTIC.md`
- Write generated data under a dedicated analysis directory in `F:/quantule_mapper/sweep_runs/`

- [ ] **Step 1: Execute the diagnostic runner on the required candidate groups**

Run the new CLI against the finished normalization pilot runs and the feb control.

- [ ] **Step 2: Inspect the generated CSV and per-candidate summaries**

Confirm the output contains:
- `run_dir`
- `idx`
- `K`
- `ic_norm`
- `target_initial_mass`
- `class`
- `time_to_blowup`
- `er_final_or_last`
- `rho_peak_max`
- `core_radius_min`
- `compactness_max`
- `omega2_min_min`
- `grad_log_omega_max`
- `curvature_proxy_max`
- `node_count_mid`
- `node_count_last`
- `split_before_blowup`
- `high_k_fraction_max`
- `diagnostic_label`

- [ ] **Step 3: Write the Markdown report**

Separate:
- evidence
- comparison by candidate group
- cautious inference
- caveats
- next experiment options

- [ ] **Step 4: Verify the report and outputs exist**

Run:

```powershell
Get-Item docs/PHASE_C_COLLAPSE_RUNAWAY_DIAGNOSTIC.md
Get-Item <analysis_outdir>/collapse_runaway_diagnostics.csv
```

Expected: both files exist.

- [ ] **Step 5: Commit**

Leave commit choice for after review; do not include bulk generated artifacts unless intentionally selected.
