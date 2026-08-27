# Gravity Runtime Architecture Hardening Plan

Author: Codex  
Timestamp: 2026-07-15  
Scope: implementation architecture proposal for the gravity-like mirror sector. This is not a theory verdict and does not modify protected production systems.

## Problem Statement

The gravity-like sector has advanced faster than the runtime architecture. The current evidence chain is valuable, but it is spread across many standalone scripts. That was appropriate while the science was exploratory. It is now becoming a risk because the latest temporal-geometric simulations combine multiple substrates, feedback fields, diagnostics and long detached GPU runs.

The immediate runtime lesson from D4 is specific:

> Full field evolution can be on GPU while checkpoint diagnostics silently move large arrays back to CPU. That is not a CPU simulation, but it can still stall coupled substrate runs and hide performance failure until late.

The architecture needs to preserve the frozen evidence scripts while extracting a safer shared runtime layer.

## Goals

1. Preserve all historical mirror scripts and run artifacts.
2. Centralize shared operators, substrates, source families, diagnostics, run manifests and GPU guards.
3. Prevent accidental CPU fallback or full-field host transfer during simulation loops.
4. Make long GPU campaigns restartable, row-addressable and auditable.
5. Keep scientific model changes separate from runtime refactors.
6. Make Claude/Jake review easier by replacing script archaeology with typed manifests and stable module boundaries.

## Non-Goals

- Do not modify production geometry, Hunter, protected validation plans, GPU launch infrastructure or CPU/GPU fallback policy.
- Do not rewrite historical artifacts.
- Do not consolidate by changing equations.
- Do not promote any gravity, geodesic, photon, temporal-lapse or production-readiness claim.

## Proposed Package Shape

The lowest-risk route is to add a mirror-only shared package while leaving existing scripts callable.

Suggested location:

```text
jax_scout/gravity_runtime/
```

Suggested modules:

```text
jax_scout/gravity_runtime/
  __init__.py
  config.py
  devices.py
  artifacts.py
  manifests.py
  checkpoints.py
  row_runner.py
  profiling.py
  arrays.py
  substrates/
    kg.py
    nls.py
    spatial_effective_medium.py
  operators/
    spectral.py
    divergence_form.py
    radial.py
  sources/
    state_load.py
    phase_relaxation.py
    phase_locking.py
    threshold.py
  feedback/
    temporal_geometric.py
  diagnostics/
    modal.py
    orbital.py
    energy_ledger.py
    force_contract.py
    source_semantics.py
    com.py
  experiments/
    registry.py
```

Existing standalone scripts should become thin experiment drivers that import this package. Historical scripts can remain as archived evidence endpoints until their behavior is reproduced by the shared runtime.

## Migration Stages

### A0 Evidence Freeze

Create a machine-readable registry of existing gravity-like scripts and artifacts:

- script path;
- stage name;
- frozen model assumptions;
- run directories;
- supported labels;
- superseded labels;
- protected files not touched;
- known caveats.

This prevents refactors from blurring which result came from which code path.

### A1 Runtime Guard Extraction

Extract and reuse:

- GPU preflight;
- x64 assertion;
- JAX/JAXLIB/version capture;
- git state capture;
- command-line capture;
- artifact hashing;
- run marker writing.

Every full field evolution runner should call the same guard before compiling or evolving arrays.

### A2 Device-Resident Diagnostics

Move all live loop diagnostics that touch full fields into JAX:

- modal diagnostics;
- orbital distance;
- phase alignment;
- COM diagnostics;
- force-contract diagnostics;
- energy ledger;
- boundary flux;
- source fields.

Allowed CPU conversions during field evolution:

- scalar diagnostics after `block_until_ready()`;
- small fixed metadata;
- sparse/eigen checks explicitly marked non-field-evolution.

Disallowed during field evolution:

```text
np.asarray(full_state[...])
np.asarray(phi)
np.roll(full field)
CPU-side full-field FFT
CPU-side full-field orbital/profile diagnostics
```

When full snapshots are required, write them at sparse preregistered intervals and mark them as snapshots, not live loop diagnostics.

### A3 Row-Oriented Runner

Long campaigns should be row-addressable:

- one row has one config hash;
- each row writes `ROW_<id>_STARTED.json`;
- each row writes `ROW_<id>_COMPLETE.json` or `ROW_<id>_FAILED.json`;
- rows append to machine-readable summary tables;
- interrupted campaigns can resume missing rows without rerunning completed evidence.

The D4 rows runner is a partial example of this pattern.

### A4 Config-First Experiments

Define experiment matrices in JSON/YAML-like Python dicts and write them before execution:

```text
model_spec.json
preregistered_matrix.json
preregistered_gates.json
```

The driver should execute the declared matrix, not construct hidden cases inside the loop.

### A5 Compatibility Wrappers

For existing standalone scripts:

- keep the CLI stable;
- move shared logic into runtime modules;
- retain old script names as wrappers;
- write a note in each wrapper identifying the shared module version used.

This avoids breaking existing handoffs while reducing duplicated solver code.

### A6 Validation Tests

Add mirror-only tests:

- GPU preflight test, skipped unless a GPU environment is explicitly available.
- Operator contract tests.
- Force identity tests for spatial effective medium.
- Energy/charge conservation smoke tests for KG.
- Source semantics null tests.
- Static code audit for forbidden full-field host transfers inside loops.

The static audit is important. It should flag patterns such as:

```text
np.asarray(state[0])
np.asarray(full_state[0])
np.asarray(off_state[0])
```

inside functions with names like `run_*`, `evolve_*`, `loop_*` or `*_decomposition`.

## GPU Runtime Rules

### Hard Requirement For Field Evolution

Every full 3D field evolution must run under:

```python
import jax

assert jax.default_backend() == "gpu"
assert any(d.platform == "gpu" for d in jax.devices())
```

Saved metadata must include:

- backend;
- devices;
- selected device;
- JAX/JAXLIB versions;
- x64 state;
- command line;
- git state;
- config hash.

### Synchronization

Timings must synchronize:

```python
jax.tree_util.tree_map(lambda x: x.block_until_ready(), output)
```

or equivalent scalar synchronization.

### Host Transfer Policy

Allowed:

- scalar diagnostics;
- CSV/JSON rows;
- low-frequency snapshots clearly declared in the matrix;
- CPU postprocessing after evolution completes.

Not allowed:

- repeated full-field transfers at checkpoint cadence;
- CPU-side diagnostics that could be JAX reductions;
- accidental fallback to NumPy arrays in the live evolution path.

## Performance Hardening

### Compile And Shape Discipline

Keep shapes static inside JIT boundaries:

- grid size;
- number of integration steps per chunk;
- field tuple structure;
- diagnostic output keys.

Avoid dynamic retracing by changing config fields inside a single row.

### Scalar Streaming

Write scalar diagnostics continuously, but keep full fields on device:

- modal phase;
- modal amplitude;
- modal leakage;
- energy/charge;
- COM;
- profile overlap;
- force residual;
- boundary flux.

### Snapshot Discipline

Full fields should be saved only when the matrix declares:

- exact snapshot cadence;
- fields to save;
- compression;
- purpose.

For long TG runs, snapshot cadence should be far lower than scalar diagnostic cadence.

### Detached Execution

Standard detached launch contract:

- one command file saved in the run directory;
- tmux/session or PID recorded;
- stdout/stderr path recorded;
- no manual monitoring required;
- final marker always written when the script exits normally.

Expected markers:

```text
RUN_COMPLETE.json
RUN_STOPPED_AT_<stage>.json
RUN_FAILED.json
```

## Evidence Governance

Every result row should include:

- run id;
- stage;
- source family;
- enabled/disabled source families;
- model spec hash;
- matrix hash;
- grid;
- box;
- timestep;
- physical duration;
- sample cadence;
- selected GPU;
- git commit;
- script path;
- imported reference run if applicable.

Every documentation result should state whether it is:

```text
EVIDENCE
INFERENCE
SPECULATION
REJECTED INTERPRETATION
OPEN QUESTION
```

## Proposed Immediate Follow-Ups

### 1. Finish Current D4/D5 Evidence Before Refactoring Core Models

Do not centralize the TG-B1S equations before the current D4/D5 closure is reviewed. Refactoring a frozen model mid-validation would muddy the evidence chain.

### 2. Add A GPU Transfer Audit

Create a small static audit tool:

```text
tools/gravity_runtime_audit.py
```

It should scan `jax_scout/gravity_*.py` for high-risk patterns:

- `np.asarray(` near `state`, `phi`, `full_state`, `off_state`;
- `np.roll(` on field arrays inside run loops;
- missing `preflight`;
- missing `jax_enable_x64`;
- missing run markers.

The first mode should report only. No automatic rewriting.

### 3. Extract Device And Artifact Utilities

First safe extraction candidates:

- preflight/environment capture;
- git state capture;
- artifact hashing;
- row marker writing;
- CSV/JSON writers;
- config hash.

These do not change equations.

### 4. Extract Device-Resident Diagnostics

Second extraction candidates:

- `orbital_metrics_jax`;
- modal diagnostics;
- energy ledger;
- boundary flux.

These should be tested against existing CSV outputs on short rows before becoming default.

### 5. Build One Canonical Runner For Long GPU Rows

Create a generic row runner with:

- preregistered matrix input;
- resume missing rows;
- row-level markers;
- scalar streaming;
- final summary.

Then adapt D4/D5-like campaigns to it.

## Review Boundary For Claude

Claude should treat this plan as an implementation-hardening proposal, not as a science claim. The key science decision still depends on D4/D5:

- If D4/D5 pass, the frozen state-load model may support an orbitally bounded modal-frequency shift.
- If D4 or D5 fails, the correct result remains unresolved, numerical, structural, or basin-limited.

Architecture work should not change that outcome.

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `25020bf` (2026-07-22) — *dual substrate*
**Revised since:** 3 commit(s), most recently `9517d9f` (2026-08-26)

**Harness code changed since it was written:** 4 commit(s) to `jax_scout/`.
  - `06edae5` 2026-08-27 — Fix run provenance: 30% -> 86% of runs now dated
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Also referenced by (same date or earlier):** [[RUN_QUEUE]]

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
