# MCP Tools Specification — IRER Simulation Stack

**Version:** MCP-v1.0  
**Date:** 2026-06-18  
**Purpose:** Define the MCP server interface that allows Claude (or any MCP-capable agent) to safely monitor, assess, and report simulation results, and to stage and launch runs within the approved output hierarchy.

---

## 1. Design Principles

1. **Read-first.** All read-only tools are safe to call at any time without user confirmation.
2. **Write tools require explicit manifest input.** No tool may silently mutate configs, overwrite artifacts, or choose its own output path.
3. **Incompatible variants cannot be ranked together.** Any tool returning ranked results must enforce the solver-contract compatibility gate (Rule C-1 through C-5 in OUTPUT_HIERARCHY.md).
4. **GPU runs are gated by a dry-run check.** `run_simulation_manifest` requires a prior call to `stage_simulation_manifest` which returns a staged manifest the user/agent must review before execution.
5. **Provenance is immutable.** MCP tools may never delete or overwrite an existing provenance file.
6. **All write operations stamp the audit log.** Every tool that modifies state appends to `runtime_logs/run_lifecycle_audit.jsonl`.

---

## 2. Proposed Server Structure

```
mcp/
├── __init__.py
├── server.py               # FastMCP entrypoint
├── tools/
│   ├── __init__.py
│   ├── read_tools.py       # get_run_status, query_ledger, read_audit_log, read_provenance,
│   │                       # list_artifacts, inspect_hdf5_schema
│   ├── analysis_tools.py   # compare_runs, get_pareto_front, audit_data_contract,
│   │                       # summarise_generation
│   └── run_tools.py        # run_smoke_simulation, stage_simulation_manifest,
│                           # run_simulation_manifest, validate_artifact
├── guards.py               # Compatibility checks, path whitelisting, manifest validation
├── schemas.py              # Pydantic models for all tool inputs/outputs
└── config.py               # Root directory, DB paths, allowed hunt names
```

---

## 3. Tool Specifications

### 3.1 `get_run_status`

**Type:** Read  
**Purpose:** Return the current lifecycle status of a specific run by config_hash (and optionally seed).

**Input schema:**
```json
{
  "config_hash": "string (required) — full or 12-char prefix",
  "seed": "integer (optional)",
  "hunt_name": "string (optional) — filter by hunt"
}
```

**Output schema:**
```json
{
  "config_hash": "string",
  "seed": "integer | null",
  "hunt_name": "string | null",
  "generation": "integer | null",
  "status": "SUCCESS | FAIL | PENDING_VALIDATION | NOT_FOUND",
  "fitness": "float | null",
  "solver_contract_version": "string | null",
  "variant_label": "string | null",
  "refinement_status": "VALIDATED_PROVISIONAL | REFINEMENT_STABLE | PHYSICS_UNCERTIFIED | null",
  "artifact_path": "string | null",
  "provenance_path": "string | null",
  "utc_timestamp": "string | null"
}
```

**Implementation note:** Queries `simulation_ledger.db` runs + metrics tables. Does not open HDF5.

---

### 3.2 `query_ledger`

**Type:** Read  
**Purpose:** Execute a filtered query against the simulation ledger and return tabular results.

**Input schema:**
```json
{
  "hunt_name": "string (optional)",
  "generation_min": "integer (optional)",
  "generation_max": "integer (optional)",
  "solver_contract_version": "string (optional) — MUST be provided to compare across runs",
  "variant_label": "string (optional)",
  "status": "SUCCESS | FAIL | any (optional)",
  "sse_max": "float (optional) — upper bound on log_prime_sse",
  "limit": "integer (optional, default 50, max 500)",
  "order_by": "log_prime_sse | generation | timestamp (optional, default log_prime_sse)"
}
```

**Output schema:**
```json
{
  "rows": [
    {
      "config_hash": "string",
      "seed": "integer | null",
      "generation": "integer",
      "status": "string",
      "log_prime_sse": "float | null",
      "solver_contract_version": "string | null",
      "variant_label": "string | null",
      "refinement_status": "string | null",
      "origin": "string"
    }
  ],
  "total_matched": "integer",
  "compatibility_warning": "string | null — set if query spans incompatible contracts/variants"
}
```

**Safety:** If the query would return results spanning multiple `solver_contract_version` values, `compatibility_warning` is set and the results are annotated with their contract version. They are NOT silently mixed.

---

### 3.3 `read_audit_log`

**Type:** Read  
**Purpose:** Return recent audit log entries, optionally filtered by stage, config_hash, or time window.

**Input schema:**
```json
{
  "stage": "string (optional) — h5_write | result_ingest | hunter_persist | ...",
  "config_hash": "string (optional)",
  "hunt_name": "string (optional)",
  "since_utc": "string (optional) — ISO-8601",
  "limit": "integer (optional, default 100, max 1000)"
}
```

**Output schema:**
```json
{
  "events": [
    {
      "timestamp": "string",
      "stage": "string",
      "config_hash": "string | null",
      "generation": "integer | null",
      "job_id": "string | null",
      "details": "object"
    }
  ],
  "total_lines_scanned": "integer"
}
```

---

### 3.4 `read_provenance`

**Type:** Read  
**Purpose:** Return the full structured provenance report for a given run.

**Input schema:**
```json
{
  "config_hash": "string (required)",
  "seed": "integer (optional)",
  "provenance_path": "string (optional) — override path lookup"
}
```

**Output schema:**
```json
{
  "found": "boolean",
  "provenance_path": "string | null",
  "schema_version": "string | null",
  "solver_contract_version": "string | null",
  "variant_label": "string | null",
  "spectral_fidelity": {
    "log_prime_sse": "float",
    "bragg_peaks_detected": "integer",
    "bragg_prime_sse": "float | null"
  },
  "falsifiability": "object | null",
  "full_payload": "object — complete provenance JSON"
}
```

---

### 3.5 `list_artifacts`

**Type:** Read  
**Purpose:** List HDF5 artifacts under the output hierarchy, with metadata from the `/identity` group.

**Input schema:**
```json
{
  "hunt_name": "string (optional)",
  "generation": "integer (optional)",
  "solver_contract_version": "string (optional)",
  "variant_label": "string (optional)",
  "include_legacy": "boolean (optional, default false)",
  "limit": "integer (optional, default 100)"
}
```

**Output schema:**
```json
{
  "artifacts": [
    {
      "path": "string",
      "size_bytes": "integer",
      "has_identity_group": "boolean",
      "hunt_name": "string | null",
      "generation": "integer | null",
      "config_hash": "string | null",
      "seed": "integer | null",
      "solver_contract_version": "string | null",
      "variant_label": "string | null",
      "utc_start": "string | null",
      "is_legacy": "boolean"
    }
  ],
  "total_found": "integer"
}
```

**Implementation note:** Opens each HDF5 with `h5py` in read mode; reads only `/identity` group if present, otherwise marks `is_legacy: true`. Does not load field data.

---

### 3.6 `inspect_hdf5_schema`

**Type:** Read  
**Purpose:** Return the structure and dataset shapes/dtypes of an HDF5 artifact without loading field data.

**Input schema:**
```json
{
  "artifact_path": "string (required) — absolute path to .h5 file"
}
```

**Output schema:**
```json
{
  "path": "string",
  "identity": "object | null — contents of /identity group",
  "datasets": [
    {
      "name": "string — dataset path e.g. /psi_final",
      "shape": "array of integers",
      "dtype": "string",
      "compression": "string | null",
      "size_bytes": "integer"
    }
  ],
  "solver_contract_json": "object | null — parsed from /solver_contract dataset",
  "sentinel_code": "float | null",
  "sentinel_reason": "string | null",
  "is_fail_artifact": "boolean"
}
```

---

### 3.7 `run_smoke_simulation`

**Type:** Write (GPU)  
**Purpose:** Launch a single low-cost smoke test (small grid, few steps) to verify the solver runs correctly without consuming production compute.

**Safety gates:**
- Hard-coded `N_grid ≤ 32`, `T_steps ≤ 100`
- Output path is always under `runs/_smoke/`
- Stamps `variant_label: SMOKE` in identity group
- Returns artifact path but does NOT write to `simulation_ledger.db`
- Appends `smoke_run` event to audit log

**Input schema:**
```json
{
  "params": "object — physics params dict (required)",
  "seed": "integer (optional, default 0)",
  "N_grid": "integer (optional, default 16, max 32)",
  "T_steps": "integer (optional, default 50, max 100)"
}
```

**Output schema:**
```json
{
  "status": "SUCCESS | FAIL",
  "artifact_path": "string | null",
  "sentinel_code": "float | null",
  "sentinel_reason": "string | null",
  "wall_time_seconds": "float",
  "final_energy": "float | null",
  "solver_contract_version": "string"
}
```

---

### 3.8 `stage_simulation_manifest`

**Type:** Write (no GPU)  
**Purpose:** Validate and stage a simulation manifest. Returns a staged manifest the user must review before `run_simulation_manifest` is called. This is the required first step before any production run.

**Input schema:**
```json
{
  "params": "object (required) — physics params dict",
  "hunt_name": "string (required)",
  "generation": "integer (required)",
  "seed": "integer (optional, default 0)",
  "N_grid": "integer (optional)",
  "T_steps": "integer (optional)",
  "dt": "float (optional)",
  "variant_label": "string (optional, default inferred from params)"
}
```

**Output schema:**
```json
{
  "staged": "boolean",
  "staged_manifest_path": "string | null",
  "config_hash": "string",
  "variant_label": "string",
  "affect_topology": "string",
  "affect_strength": "float",
  "solver_contract_version": "string",
  "expected_output_path": "string",
  "compatibility_warnings": "array of strings",
  "validation_errors": "array of strings",
  "review_required": "boolean — always true",
  "message": "string — human-readable summary for review"
}
```

**Validation checks performed:**
- `param_rho_vac` ≥ 0.05 (DEGENERATE_GEOMETRY warning if below)
- `N_grid` is power of 2
- solver_contract_version derived from params (checks γ_A)
- Output path does not already exist (no silent overwrite)
- CFL check: `param_c_affect * dt * k_max ≤ 1.0` (for A integrator)

---

### 3.9 `run_simulation_manifest`

**Type:** Write (GPU)  
**Purpose:** Execute a previously staged manifest. Requires `staged_manifest_path` from `stage_simulation_manifest`. This is the only tool that launches a real GPU simulation.

**Safety gates:**
- `staged_manifest_path` must exist and be no older than 30 minutes
- Output path must match the path in the staged manifest (cannot be redirected)
- Appends `dispatch` event to audit log before launching
- On completion, appends `h5_write` event to audit log

**Input schema:**
```json
{
  "staged_manifest_path": "string (required)",
  "confirm": "boolean (required, must be true) — explicit confirmation this is intentional"
}
```

**Output schema:**
```json
{
  "status": "SUCCESS | FAIL | PENDING_VALIDATION",
  "artifact_path": "string | null",
  "config_hash": "string",
  "sentinel_code": "float | null",
  "wall_time_seconds": "float",
  "audit_event_id": "string"
}
```

---

### 3.10 `validate_artifact`

**Type:** Write (CPU)  
**Purpose:** Run the validation pipeline on an existing HDF5 artifact and write provenance.

**Safety gates:**
- Does not overwrite an existing provenance if `force=false`
- Checks solver_contract_version compatibility before running
- Appends `validation_write` event to audit log

**Input schema:**
```json
{
  "artifact_path": "string (required)",
  "params_path": "string (required) — path to params.json",
  "output_dir": "string (optional, default inferred from artifact path)",
  "force": "boolean (optional, default false)"
}
```

**Output schema:**
```json
{
  "status": "PASSED | FAILED | SKIPPED",
  "provenance_path": "string | null",
  "log_prime_sse": "float | null",
  "solver_contract_version": "string | null",
  "compatibility_ok": "boolean",
  "validation_schema_version": "string",
  "early_rejected": "boolean"
}
```

---

### 3.11 `compare_runs`

**Type:** Read  
**Purpose:** Side-by-side comparison of two or more runs. Enforces compatibility gate before comparing metrics.

**Input schema:**
```json
{
  "config_hashes": "array of strings (required, 2–10 entries)",
  "seeds": "array of integers (optional — one per hash)",
  "fields": "array of strings (optional) — subset of metric fields to compare"
}
```

**Output schema:**
```json
{
  "compatible": "boolean — false if variants/contracts differ",
  "compatibility_note": "string | null",
  "comparison": [
    {
      "config_hash": "string",
      "seed": "integer | null",
      "solver_contract_version": "string",
      "variant_label": "string",
      "log_prime_sse": "float | null",
      "bragg_peaks_detected": "integer | null",
      "refinement_status": "string | null",
      "affect_topology": "string | null",
      "affect_strength": "float | null",
      "N_grid": "integer | null",
      "generation": "integer | null"
    }
  ],
  "winner": "string | null — config_hash with lowest log_prime_sse (only set if compatible=true)"
}
```

---

### 3.12 `get_pareto_front`

**Type:** Read  
**Purpose:** Return the current Pareto-optimal front for a hunt.

**Input schema:**
```json
{
  "hunt_name": "string (required)",
  "solver_contract_version": "string (required) — must be explicit",
  "variant_label": "string (required) — must be explicit",
  "generation": "integer (optional) — if omitted, returns current front",
  "sse_max": "float (optional)"
}
```

**Output schema:**
```json
{
  "hunt_name": "string",
  "solver_contract_version": "string",
  "variant_label": "string",
  "generation": "integer | null",
  "front": [
    {
      "config_hash": "string",
      "seed": "integer | null",
      "log_prime_sse": "float",
      "refinement_status": "string | null",
      "params": "object — key physics params"
    }
  ],
  "champion": "object | null — entry with lowest log_prime_sse on REFINEMENT_STABLE runs"
}
```

---

### 3.13 `audit_data_contract`

**Type:** Read  
**Purpose:** Check a run (or a sample of recent runs) for data contract compliance: HDF5 identity group present, provenance naming unique, ledger discriminator columns populated, solver contract consistent.

**Input schema:**
```json
{
  "config_hash": "string (optional) — single run check",
  "hunt_name": "string (optional) — scan all runs in hunt",
  "sample_size": "integer (optional, default 20)"
}
```

**Output schema:**
```json
{
  "total_checked": "integer",
  "compliant": "integer",
  "non_compliant": "integer",
  "issues": [
    {
      "config_hash": "string",
      "artifact_path": "string | null",
      "issues": [
        "missing /identity group",
        "provenance filename collision",
        "solver_contract_version mismatch between HDF5 and ledger",
        "param_rho_vac below degenerate threshold",
        "A_dot_final in spectral space (label misleading)"
      ]
    }
  ],
  "summary": "string"
}
```

---

### 3.14 `summarise_generation`

**Type:** Read  
**Purpose:** Produce a structured summary of one generation: run counts, SSE distribution, sentinel breakdown, new champions, Pareto progress.

**Input schema:**
```json
{
  "hunt_name": "string (required)",
  "generation": "integer (required)",
  "solver_contract_version": "string (required)",
  "variant_label": "string (required)"
}
```

**Output schema:**
```json
{
  "hunt_name": "string",
  "generation": "integer",
  "solver_contract_version": "string",
  "variant_label": "string",
  "total_runs": "integer",
  "succeeded": "integer",
  "failed": "integer",
  "sentinel_breakdown": {
    "math_explosion": "integer",
    "physics_drift": "integer",
    "geometry_sanity": "integer"
  },
  "sse_stats": {
    "min": "float",
    "median": "float",
    "p25": "float",
    "p75": "float",
    "golden_count": "integer (SSE < 1.0)",
    "silver_count": "integer (SSE < 3.0)"
  },
  "new_champion": "object | null",
  "pareto_front_size": "integer",
  "degenerate_geometry_count": "integer (rho_vac < 0.05)"
}
```

---

## 4. Safety Rules for GPU Jobs

1. **Smoke before production.** Always call `run_smoke_simulation` before a new param configuration runs at full scale.

2. **Stage and review.** `stage_simulation_manifest` must be called and its output reviewed before `run_simulation_manifest`. The staged manifest includes `compatibility_warnings` and `validation_errors` that must be clear.

3. **No path override.** `run_simulation_manifest` uses the `expected_output_path` from the staged manifest. It cannot be redirected by a separate argument.

4. **Explicit confirmation.** `run_simulation_manifest` requires `confirm: true` in the input. An agent must not set this autonomously without user review of the staged manifest summary.

5. **Degenerate geometry warning.** If `param_rho_vac < 0.05`, `stage_simulation_manifest` emits a warning and tags the expected output as `DEGENERATE_GEOMETRY`. The run may still proceed but cannot enter the main leaderboard.

6. **CFL guard.** If `param_c_affect * dt * k_cut > 1.0`, staging fails with a validation error. The A integrator is explicit symplectic-Euler and will become unstable above this threshold.

7. **No overwrite of existing artifacts.** If the expected output path already contains a completed artifact, staging fails unless the user explicitly provides `overwrite: true` with a stated reason.

8. **Audit trail on failure.** If `run_simulation_manifest` fails or is interrupted, the MCP server writes a `dispatch_failure` event to the audit log before returning.

---

## 5. Example Workflow: Smoke Run → Validation → Provenance → Ledger → Report

```
Step 1: Smoke test (safety check)
─────────────────────────────────
Agent calls: run_smoke_simulation({
  params: {param_D: 1.5, param_eta: 0.3, param_rho_vac: 0.8, param_a_coupling: 2.0, param_s: -0.1, param_f: 0.0},
  seed: 42, N_grid: 16, T_steps: 50
})
→ Returns: {status: "SUCCESS", sentinel_code: null, wall_time_seconds: 12.3}

Step 2: Inspect smoke artifact
──────────────────────────────
Agent calls: inspect_hdf5_schema({artifact_path: "runs/_smoke/..."})
→ Returns: all expected datasets present, solver_contract correct

Step 3: Stage production manifest
──────────────────────────────────
Agent calls: stage_simulation_manifest({
  params: {param_D: 1.5, ...},
  hunt_name: "IRER_HUNT_001",
  generation: 3, seed: 42,
  N_grid: 128, T_steps: 1200, dt: 0.005
})
→ Returns: {
    staged_manifest_path: "...",
    solver_contract_version: "IRER-SNCGL-LOCAL-RHO-ETDRK4-v1",
    expected_output_path: "runs/IRER_HUNT_001/2026-06-18/IRER-SNCGL.../LOCAL-RHO/gen_0003/a3f8c2e91b4d_42/artifact.h5",
    compatibility_warnings: [],
    review_required: true,
    message: "Ready to launch N=128 run. No compatibility issues."
  }

[USER/AGENT REVIEWS staged manifest]

Step 4: Launch production run
──────────────────────────────
Agent calls: run_simulation_manifest({
  staged_manifest_path: "...", confirm: true
})
→ Returns: {status: "PENDING_VALIDATION", artifact_path: "...", wall_time_seconds: 847}

Step 5: Validate
─────────────────
Agent calls: validate_artifact({
  artifact_path: "runs/.../artifact.h5",
  params_path: "runs/.../params.json"
})
→ Returns: {status: "PASSED", log_prime_sse: 0.73, provenance_path: "runs/.../provenance.json"}

Step 6: Read provenance
────────────────────────
Agent calls: read_provenance({config_hash: "a3f8c2e91b4d", seed: 42})
→ Returns: full provenance including spectral_fidelity, solver_contract, falsifiability scores

Step 7: Check ledger
─────────────────────
Agent calls: get_run_status({config_hash: "a3f8c2e91b4d", seed: 42})
→ Returns: {status: "SUCCESS", fitness: 0.73, refinement_status: "VALIDATED_PROVISIONAL"}

Step 8: Generation summary
───────────────────────────
Agent calls: summarise_generation({
  hunt_name: "IRER_HUNT_001", generation: 3,
  solver_contract_version: "IRER-SNCGL-LOCAL-RHO-ETDRK4-v1",
  variant_label: "LOCAL-RHO"
})
→ Returns: {total_runs: 64, golden_count: 3, silver_count: 12, new_champion: {...}}

Step 9: Audit data contract
────────────────────────────
Agent calls: audit_data_contract({hunt_name: "IRER_HUNT_001", sample_size: 10})
→ Returns: {total_checked: 10, compliant: 10, non_compliant: 0}
```

---

## 6. Minimum Viable Implementation Order

To get read-only monitoring running immediately (no GPU dependency):

1. `get_run_status` — queries existing SQLite
2. `query_ledger` — queries existing SQLite
3. `read_audit_log` — reads existing JSONL
4. `inspect_hdf5_schema` — reads existing HDF5 with h5py
5. `list_artifacts` — filesystem walk + h5py
6. `read_provenance` — reads existing JSON files
7. `summarise_generation` — aggregates from SQLite
8. `audit_data_contract` — cross-checks HDF5 + SQLite + JSON

These 8 tools together give Claude full visibility into the simulation state without writing anything.

Write tools (`run_smoke_simulation`, `stage_simulation_manifest`, `run_simulation_manifest`, `validate_artifact`) should be implemented after the output hierarchy migration is complete (OUTPUT_HIERARCHY.md Phase 1-2).

---

## 7. Implementation status — MCP-v1.0 (2026-06-18)

**All 8 read-only tools are IMPLEMENTED and tested** (`tests/test_mcp_read_tools.py`, 15 tests, GPU-independent):

| Tool | Status | Notes |
|---|---|---|
| `get_run_status` | DONE | ledger join (runs+metrics), locates artifact/provenance |
| `query_ledger` | DONE | sets `compatibility_warning` when results span variants/contracts |
| `read_audit_log` | DONE | JSONL filter by stage/config_hash/hunt/time |
| `read_provenance` | DONE | collision-free filename aware (seed/run_id) |
| `list_artifacts` | DONE | reads `/identity` only; flags legacy artifacts |
| `inspect_hdf5_schema` | DONE | shapes/dtypes/compression + sentinel + contract |
| `summarise_generation` | DONE | SSE stats, sentinel breakdown, champion, degenerate count |
| `audit_data_contract` | DONE | cross-checks HDF5 `/identity` vs ledger, provenance presence |

**Package layout (deviation noted):** implemented under `mcp_server/` — **not** `mcp/` as sketched in §2 — because a top-level `mcp` package shadows the `mcp` SDK (`from mcp.server.fastmcp import FastMCP`). Layering: `config.py` (paths + read whitelist) → `data_access.py` (pure, SDK-free, testable) → `server.py` (thin FastMCP entrypoint, `python -m mcp_server.server`). The 8 tools register correctly with FastMCP.

**Safety properties already enforced:** read-only (DBs opened `mode=ro`); path inputs checked against a project-root whitelist; ranking queries refuse to silently mix incompatible variants (`compatibility_warning`). Write/GPU tools remain unimplemented per §6.
