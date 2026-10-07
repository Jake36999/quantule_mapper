---
tags: [record, infra, specs, mcp, ui]
date: 2026-10-04
branch: Branch - Validation - Index
status: complete
---

# Experiments as data: specs, MCP tools and the spec editor (Phase E)

This is the build record for Phase E of [[IMPLEMENTATION_PLAN_2026-10]]. An experiment is now a small
validated JSON record, not a new script. One executor runs any spec. Agents can read everything and
*propose* specs, but cannot run them. A local web page edits specs with a form generated from the
schema.

**The test that matters:** the a\* probe expressed as a spec reproduces the harness it replaces
(`core_saturation_search.run_probe`) to **< 1e-10** in er(t).

> [!info] Sources
> - E1: `irer_specs/__init__.py`, `schemas/experiment_spec.schema.json`, `jax_scout/registry.py`,
>   `tools/run_spec.py`, `specs/{drafts,proposed,approved}/`, `tests/test_spec_layer.py`
> - E2: `mcp_server/research_tools.py`, `mcp_server/server.py` (11 `research_*` tools),
>   `docs/registry/components.json`, `tests/test_mcp_research_tools.py`
> - E3: `web/spec_editor/index.html`, `tools/serve_spec_ui.py`, `tests/test_spec_ui.py`

## How it fits together
```
specs/drafts ──(UI: save)──► specs/drafts ──(UI: approve)──► specs/approved ──(human: run_spec.py)──► sweep_runs/<ID>_<stamp>
                         agent (MCP propose_spec) ──► specs/proposed ──(UI: approve)──┘
run dir: spec.json · summary.json (provenance + final observers + prediction_check) · telemetry.jsonl · verdict.json = PENDING_REVIEW
```

## E1 — the spec layer

**Spec fields**
- `substrate`: name, version and params
- `protocol`: IC, grid, dt, physical T, sampling interval, stop rule
- `observers`
- optional `sweep` (grid or zip over dotted paths)
- `invariants`: declared telemetry tolerances
- **`prediction`**: required and written before launch; this is P4, "derive before you measure"
- `requires`: edges to other specs' verdicts. These are warnings only.

**Components** are named and versioned in `jax_scout/registry.py`. Every one wraps existing code
that the order gates already cover:
- Substrates: `etdrk4-sncgl` (`physics.step`), `kg-strang` (`kg_evolve`), `tg-rk4` (`b1s.rk4_step`).
- ICs: the harnesses' own builders (`multiseed`, `nls_soliton`, `qball`), plus `gaussian` and `load_npz`.
- Observers: the harnesses' own measurements (`centroid_x`, `invariants`, `detect_nodes`, TG
  `diagnostics`).

**Dependency-free.** The JAX environment has no pydantic or jsonschema, so `irer_specs` carries the
schema as a dict and implements exactly the JSON Schema keywords it uses. The JSON file is generated
from that dict, and a test keeps the two in sync.

**The executor** (`tools/run_spec.py`):
- streams every observer value to telemetry;
- stamps provenance (commit, steppers, component hashes);
- writes `verdict.json = PENDING_REVIEW`;
- scores the pre-registered prediction into `prediction_check`, which is explicitly **not** a verdict.

**Two pilots** live in `specs/approved/`:
- `astar-probe-pilot` — the basis of the equivalence test.
- `c27-r3-soliton-transport-n1` — C2.7 R3 with its prediction written from Galilean invariance:
  v = 2Dk = 1.25664 at relative tolerance 1e-3. It is queued to run on the GPU after the B4 replay.

## E2 — MCP tools (repo `mcp_server/`)
There are 11 new tools named `research_*`, kept separate from the legacy ledger tools.

**Read-only:**
- `list_runs` (filters: substrate, branch, stepper, harness, stale_only)
- `get_run`
- `tail_telemetry`, with breaches
- `list_stale_runs`
- `list_components`. It falls back to the generated `components.json` when JAX is absent, which is
  the case in `.venv`.
- `get_schema`, `list_specs`, `get_spec`
- `harness_registry`

**Write-limited:**
- `propose_spec` validates the spec and writes **only** to `specs/proposed/`. It opens the file in
  mode `'x'`, so it never overwrites; the id must be schema-valid, so it cannot escape the folder;
  and every call is audited.
- `draft_reading` appends **MACHINE-DRAFTED** readings to `docs/runs/_machine_drafts/<run_id>.md`
  and never touches the human paired-reading blocks.

**There is no launch tool.** A test asserts that none exists.

**Agents (DeepInfra via the Librarian)** connect to this server as MCP clients using the Librarian's
existing `providers.py` / `ocr.py`. No Librarian code changes are needed. Suggested split, from
[[PROCESS_PLAN_2026-10]] P6:
- Large models: `list_runs`, `get_run` and `tail_telemetry` for numbers; `draft_reading` for the
  reading backlog; `propose_spec` for follow-ups.
- Vision/OCR models: rendered `telemetry*.png` and snapshot renders, for morphology triage only,
  always paired with the numbers.

## E3 — spec editor
`python tools/serve_spec_ui.py` → http://127.0.0.1:8765/ (bound to localhost; standard library only).
- One static page: react-jsonschema-form loaded from a CDN, with a raw-JSON tab as a fallback when
  offline.
- It lists specs by folder, edits them, validates them (schema plus registered component names),
  saves drafts, and approves drafts or agent proposals.
- **No run button.** An approved spec shows the `run_spec.py` command to copy.

> [!warning] Bug found in the browser test
> react-jsonschema-form sends empty defaults for untouched optional objects (`sweep: {axes: {}}`),
> which then failed validation. `irer_specs.prune_empty` now drops empty optional containers on the
> server before validation and saving, so every client is covered. Regression test:
> `test_form_defaults_are_pruned_before_validation`.

## What this changes

| status | item |
|---|---|
| **Unchanged** | Every existing harness and its output. Specs are a parallel path. |
| **Newly possible** | Running an experiment without writing a script; agents reading runs and proposing specs through a fenced channel; editing specs through a form generated from the schema. |

## Issues raised

| # | issue | status |
|---|---|---|
| 1 | The `mcp` package is not installed in `.venv`, so the server cannot start (the 3 legacy MCP tests fail for the same reason) | OPEN — `.venv/Scripts/pip install mcp` (needs your OK) |
| 2 | The legacy MCP tools `run_smoke_simulation` / `validate_artifact` can launch GPU/CPU work, which contradicts the no-launch rule | OPEN — recommend retiring them, or fencing them behind a flag |
| 3 | The C2.7 pilot spec has not run yet (the GPU is busy with B4) | OPEN — run after the replay and compare with the R3 replay |
| 4 | The form library loads from a CDN | NOTED — the JSON tab works offline |

## Associated docs
- [[IMPLEMENTATION_PLAN_2026-10]] · [[PROCESS_PLAN_2026-10]] · [[PROVENANCE_AND_HARNESS_REGISTRY]] · [[TELEMETRY_STREAM]] · [[SOLVER_AND_RUNTIME_CHANGELOG]]

## Branches
- [[Branch - Validation - Index]]
