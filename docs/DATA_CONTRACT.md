# Data Contract — IRER Simulation Stack

**Version:** DC-v1.0  
**Date:** 2026-06-18  
**Scope:** Authoritative specification of what every module writes, what every module reads, and the minimum discriminator fields required to prevent confusing incompatible runs.

---

## 1. Current Contract (as-is)

### 1.1 HDF5 Artifact — solver/run.py

#### SUCCESS path datasets

| Dataset | Type | Shape | Notes |
|---|---|---|---|
| `psi_final` | complex128 | (N,N,N) | Real-space final field |
| `omega_sq_final` | float64 | (N,N,N) | Conformal factor Ω² at t_final |
| `A_final` | float64 | (N,N,N) | Affect field in **real space** |
| `A_dot_final` | complex128 | (N,N,N) | Affect field velocity in **spectral space** ⚠️ (label is misleading — not real-space) |
| `N_a_stage` | complex128 | (N,N,N) | ETDRK4 stage-A nonlinear term (optional) |
| `N_b_stage` | complex128 | (N,N,N) | ETDRK4 stage-B nonlinear term (optional) |
| `N_c_stage` | complex128 | (N,N,N) | ETDRK4 stage-C nonlinear term (optional) |
| `telemetry/step` | int64 | (T,) | Step indices at telemetry cadence |
| `telemetry/energy` | float64 | (T,) | ∫ρ dV at each telemetry step |
| `telemetry/energy_sparkline` | float64 | (100,) | Resampled energy trajectory |
| `telemetry/C_invariant` | float64 | (T,) | ∫ρ² dV (second Casimir) |
| `extended_telemetry/step_count` | int64 | (1,) | Final step index |
| `extended_telemetry/sim_time` | float64 | (1,) | Wall time in seconds |
| `extended_telemetry/dt` | float64 | (1,) | Time step |
| `extended_telemetry/grid_shape` | int32 | (1,) | N_grid |
| `extended_telemetry/params_hash` | S128 | (1,) | config_hash bytes |
| `solver_contract` | S512 | (1,) | JSON string (see §1.1.1) |

#### FAIL path datasets

| Dataset | Type | Notes |
|---|---|---|
| `psi_final` | complex128 | Snapshot at failure |
| `sentinel_code` | float64 | 1002=math_explosion, 1003=physics_drift, 1004=geometry_sanity |
| `sentinel_reason` | S64 | Human-readable reason |
| `telemetry/step` | int64 | Partial history |
| `telemetry/energy` | float64 | Partial history |
| `telemetry/C_invariant` | float64 | Partial history |

#### 1.1.1 solver_contract JSON (embedded in S512 dataset)

```json
{
  "solver_contract_version": "IRER-SNCGL-LOCAL-RHO-ETDRK4-v1",
  "geometry_source": "local_stage_rho",
  "auxiliary_geometry": false,
  "topology_cap_in_simulation": false,
  "linear_operator": "-D*k^2 - eta + i*rho_vac"
}
```

### 1.2 Ledger — simulation_ledger.db (orchestrator/schema_utils.py)

#### `runs` table

| Column | Type | Notes |
|---|---|---|
| config_hash | TEXT PK | SHA-256 of params JSON |
| generation | INTEGER | |
| status | TEXT | SUCCESS / FAIL |
| fitness | REAL | log_prime_sse (lower = better) |
| origin | TEXT | NATURAL / CROSSOVER / MUTATION |
| parent_1 | TEXT | |
| parent_2 | TEXT | |
| staged_path | TEXT | |
| staged_at | TEXT | |
| staged_config_hash | TEXT | |
| timestamp | DATETIME | SQLite default CURRENT_TIMESTAMP |

#### `parameters` table

| Column | Type |
|---|---|
| config_hash | TEXT PK |
| param_D, param_eta, param_rho_vac, param_a_coupling, param_splash_coupling, param_splash_fraction | REAL |

#### `metrics` table

| Column | Type | Notes |
|---|---|---|
| config_hash | TEXT PK | |
| log_prime_sse | REAL | Primary fitness |
| bragg_peaks_detected | INTEGER | |
| bragg_prime_sse | REAL | |
| collapse_event_count | INTEGER | |
| pcs | REAL | Phase Coherence Score |
| refinement_status | TEXT | VALIDATED_PROVISIONAL / REFINEMENT_STABLE / PHYSICS_UNCERTIFIED |
| solver_contract_version | TEXT | Extracted from provenance |
| ... (20+ further metric columns) | | |

### 1.3 Provenance JSON — validation_pipeline.py (ProvenanceAssembler)

Filename: `provenance_reports/provenance_{config_hash}.json`

Key top-level sections: `spectral_fidelity`, `aletheia_metrics`, `solver_contract`, `topology`, `lom_telemetry`, `falsifiability`, `empirical_bridge`, `tensor_validation`, `statistical_validation`.

### 1.4 Audit JSONL — orchestrator/diagnostics/runtime_audit.py

File: `runtime_logs/run_lifecycle_audit.jsonl`

Each line:
```json
{
  "timestamp": "2026-06-18T12:00:00+00:00",
  "stage": "h5_write | result_ingest | hunter_persist | ...",
  "config_hash": "...",
  "generation": 5,
  "job_id": "...",
  "details": {}
}
```

---

## 2. Current Contract Gaps

The following gaps mean that future agents or tools **cannot reliably identify what variant produced a given artifact** without reading deeply into embedded JSON:

| Gap | Location | Severity | Detail |
|---|---|---|---|
| No hunt_name in HDF5 | solver/run.py | HIGH | Cannot link artifact to session without ledger |
| No UTC timestamp in HDF5 | solver/run.py | HIGH | Audit ordering requires reading filesystem mtime |
| No seed in HDF5 | solver/run.py | HIGH | Two runs of same params with different seeds produce same filename |
| No git commit hash anywhere | all | HIGH | Cannot pin artifact to code version |
| No variant_label in HDF5 or ledger | all | HIGH | LOCAL-RHO vs CAUSAL-AFFECT vs ADDITIVE-POT indistinguishable without reading solver_contract JSON string |
| No affect_topology or affect_strength in ledger | schema_utils.py | HIGH | A-coupling variants will get ranked alongside baseline without gate |
| solver_contract is a JSON blob in an S512 dataset | solver/run.py | MEDIUM | Not readable by standard HDF5 attribute browsers; requires manual parsing |
| A_dot_final is in spectral space | solver/run.py | MEDIUM | Dataset name suggests real space; callers may misinterpret as real-space velocity |
| Provenance filename is provenance_{config_hash}.json | validation_pipeline.py | HIGH | Re-running the same params overwrites the previous provenance; multi-seed runs cannot be distinguished |
| runs table PK is config_hash | schema_utils.py | HIGH | Multi-seed runs (same params, different seed) collapse to one ledger row |
| No N_grid / dt / T_steps in ledger | schema_utils.py | MEDIUM | Grid upgrade comparisons require reading provenance JSON |
| No GPU/backend label in artifact or ledger | all | LOW | Cannot flag mixed CPU/GPU result sets |
| param_rho_vac=0 is in search bounds | burn_in_config.json | HIGH | ρ_vac=0 → Ω²=0 everywhere → geometry degenerate (only saved by conformal floor); also sets ω₀=0 (no vacuum oscillation); whole runs with ρ_vac≈0 are testing a different physics |

---

## 2.1 Implementation status — DC-v1.0 (2026-06-18)

The following gaps are now **CLOSED** in code (all verified by GPU-independent tests):

| Gap | Status | Where | Test |
|---|---|---|---|
| No `/identity` group in HDF5 | **CLOSED** | `solver/run.py` writes `/identity` on SUCCESS + FAIL via `orchestrator/run_identity.write_identity_group`; built in `worker_cupy.py` | `test_data_contract.py::TestIdentityGroup`, `test_run_identity.py::TestIdentityGroupRoundTrip` |
| `A_dot_final` mislabelled | **CLOSED** | renamed → `A_dot_k_final` (write `solver/run.py`; read `validation_pipeline.py` accepts both, legacy alias kept) | `TestADotFinalLabel`, `test_phase_v13_causal_field.py` |
| `runs` PK = config_hash (multi-seed collision) | **CLOSED** | composite PK `(config_hash, seed)` for fresh DBs; opt-in `migrate_runs_to_composite_pk()` for existing DBs | `test_ledger_identity.py` (9 tests) |
| No discriminator columns in ledger | **CLOSED** | `runs` gains seed/run_id/hunt_name/utc_start/solver_contract_version/variant_label/affect_topology/affect_strength/git_commit/n_grid/dt/t_steps/gpu_backend/artifact_hash/provenance_hash; populated by `result_processor._upsert_run_row` from the artifact `/identity` | `TestLedgerSchemaDiscriminators`, `test_ledger_identity.py` |
| No `param_affect_coupling` in parameters | **CLOSED** | added (+ `param_affect_topology`) to `parameters`; written by `result_processor` | `test_ledger_identity.py::test_parameters_has_affect_coupling` |
| Provenance filename collision | **CLOSED** | single source `run_identity.provenance_path_for_artifact` reads `/identity` (seed+run_id+utc); used by writer `validation_pipeline` and readers `result_processor` / `run_validation`; legacy `provenance_{hash}.json` fallback for old artifacts | `TestProvenanceNaming`, `test_run_identity.py::TestProvenanceNaming` |
| Compatibility gate undefined | **PARTIAL** | `run_identity.are_rankable` / `compatibility_key` implemented + tested; **not yet wired** into the orchestrator ranking/champion path | `test_run_identity.py::TestCompatibilityGate` |
| k=0 A-field secular runaway | **CLOSED** | `solver/core.py:update_field_of_affect` projects DC source + pins A/A_dot zero modes | `test_run_identity.py::TestK0Runaway` |

Still **OPEN** (require a deliberate operational step or the A-coupling phase):
- `param_rho_vac` lower bound still `0.0` in `burn_in_config.json` (degenerate geometry) — set `≥ 0.05` and/or split into `param_omega0` + `param_rho_vac` (see IRER_MATH_SANITY_CHECK §7.2).
- Existing live ledgers keep config_hash-only PK until `migrate_runs_to_composite_pk()` is run against them.
- Compatibility gate not yet enforced in `result_processor.process_result` / champion selection — `are_rankable` exists but is not called yet.
- `artifact_hash`/`provenance_hash` columns exist and `artifact_hash` is populated; `provenance_hash` wiring pending.

## 3. Proposed Final Contract

### 3.1 Required Discriminator Fields

Every artifact, ledger row, provenance file, and audit event MUST carry this complete identity tuple:

```
(hunt_name, run_id, utc_timestamp, generation, config_hash, seed,
 solver_contract_version, validation_contract_version, variant_label,
 affect_topology, affect_strength, git_commit, N_grid, dt, T_steps,
 gpu_backend, artifact_hash)
```

**Definitions:**

| Field | Type | Source | Notes |
|---|---|---|---|
| `hunt_name` | str | orchestrator config | e.g. "BURN_IN_STRESS_TEST_001" |
| `run_id` | str | JobManifest.job_id | UUID4 prefix |
| `utc_timestamp` | str | ISO-8601 UTC | Time simulation started |
| `generation` | int | JobManifest.generation | |
| `config_hash` | str | SHA-256 of params | Existing |
| `seed` | int | JobManifest.seed | Must be in HDF5 and ledger |
| `solver_contract_version` | str | orchestrator/contracts.py | e.g. "IRER-SNCGL-LOCAL-RHO-ETDRK4-v1" |
| `validation_contract_version` | str | validation_pipeline.py SCHEMA_VERSION | e.g. "SFP-v3.2-ARCS" |
| `variant_label` | str | derived from solver_contract | "LOCAL-RHO" / "CAUSAL-AFFECT" / "ADDITIVE-POT" |
| `affect_topology` | str | param lookup | "none" / "vacuum_ref" / "additive_potential" |
| `affect_strength` | float | param_affect_coupling | 0.0 = baseline |
| `git_commit` | str | git rev-parse HEAD at launch | |
| `N_grid` | int | simulation config | |
| `dt` | float | simulation config | |
| `T_steps` | int | simulation config | |
| `gpu_backend` | str | runtime detection | "cupy" / "numpy" |
| `artifact_hash` | str | SHA-256 of HDF5 file | Written to provenance after artifact is closed |
| `provenance_hash` | str | SHA-256 of provenance JSON | Written to ledger |

### 3.2 Proposed HDF5 Schema Additions

Add an `/identity` group to every HDF5 artifact at write time (solver/run.py):

```
/identity/
  hunt_name         (S256)
  run_id            (S64)
  utc_start         (S64)    ISO-8601
  generation        (int64)
  config_hash       (S128)
  seed              (int64)
  solver_contract_version  (S128)
  variant_label     (S64)
  affect_topology   (S64)
  affect_strength   (float64)
  git_commit        (S64)
  N_grid            (int32)
  dt                (float64)
  T_steps           (int32)
  gpu_backend       (S32)
```

Rename `A_dot_final` → `A_dot_k_final` to make the spectral-space nature explicit.

### 3.3 Proposed Ledger Schema Additions

Add columns to `runs`:
```sql
ALTER TABLE runs ADD COLUMN run_id TEXT;
ALTER TABLE runs ADD COLUMN hunt_name TEXT;
ALTER TABLE runs ADD COLUMN utc_start TEXT;
ALTER TABLE runs ADD COLUMN seed INTEGER;
ALTER TABLE runs ADD COLUMN solver_contract_version TEXT;
ALTER TABLE runs ADD COLUMN variant_label TEXT;
ALTER TABLE runs ADD COLUMN affect_topology TEXT;
ALTER TABLE runs ADD COLUMN affect_strength REAL;
ALTER TABLE runs ADD COLUMN git_commit TEXT;
ALTER TABLE runs ADD COLUMN N_grid INTEGER;
ALTER TABLE runs ADD COLUMN dt REAL;
ALTER TABLE runs ADD COLUMN T_steps INTEGER;
ALTER TABLE runs ADD COLUMN gpu_backend TEXT;
ALTER TABLE runs ADD COLUMN artifact_hash TEXT;
ALTER TABLE runs ADD COLUMN provenance_hash TEXT;
```

Change `runs` PRIMARY KEY from `config_hash` to `(config_hash, seed)` to support multi-seed runs. This is a breaking migration.

Add `param_affect_coupling` to `parameters` table.

### 3.4 Proposed Provenance Naming

Replace `provenance_{config_hash}.json` with:

```
provenance_{config_hash}_{seed}_{utc_date}_{run_id[:8]}.json
```

This makes each provenance file unique to a specific run, not just a parameter set.

### 3.5 Compatibility Gate

Before two runs can be ranked together, result_processor.py MUST check:

1. `solver_contract_version` is identical
2. `variant_label` is identical
3. `affect_topology` is identical
4. `N_grid` is identical (different grids test different spectral modes)

Any result that does not pass this check must be tagged `PHYSICS_UNCERTIFIED` regardless of SSE value.

---

## 4. Schema Map — Who Writes, Who Reads Each Field

| Field | Written by | Read by | Notes |
|---|---|---|---|
| `psi_final` | solver/run.py | validation_pipeline (ArtifactLoader) | Primary field |
| `omega_sq_final` | solver/run.py | validation_pipeline (ContractEnforcer) | Geometry snapshot |
| `A_final` | solver/run.py | not currently read by pipeline | Passive; will be read when A-coupling implemented |
| `A_dot_k_final` | solver/run.py | not currently read | Spectral velocity |
| `telemetry/*` | solver/run.py | validation_pipeline (LOMTelemetryEngine) | |
| `solver_contract` JSON | solver/run.py | result_processor._validate_result() | Extracted from provenance payload |
| `log_prime_sse` | quantulemapper_real (via pipeline) | result_processor.py, schema_utils ledger | THE primary fitness signal |
| `provenance JSON` | validation_pipeline (ProvenanceAssembler) | result_processor._validate_result() | Source of truth for SSE ingest |
| `runs` table | result_processor._write_worker_result_to_ledger() | orchestrator_engine (champion logic), Hunter | |
| `metrics` table | result_processor._write_worker_result_to_ledger() | Hunter, pareto logic | |
| `parameters` table | result_processor._write_worker_result_to_ledger() | Hunter crossover/mutation | |
| `audit JSONL` | runtime_audit.log_lifecycle_event() | external monitoring / MCP tools | Append-only |
| `config_hash` | orchestrator/contracts.py (JobManifest.from_params) | everywhere | SHA-256 of params JSON |

---

## 5. Migration Path

### Phase 1 (immediate — no GPU run required)
- Add `/identity` group write to solver/run.py
- Rename `A_dot_final` → `A_dot_k_final` in solver/run.py  
- Update test_phase_v13_causal_field.py to assert `A_dot_k_final`
- Add hunt_name, seed, solver_contract_version, variant_label, affect_topology to audit JSONL `details`

### Phase 2 (before next hunt)
- Add ledger columns (runs, parameters) for all discriminator fields
- Fix provenance naming collision
- Add compatibility gate to result_processor.process_result()

### Phase 3 (before A-coupling runs)
- Add param_affect_coupling to parameters table
- Bump solver contract to IRER-SNCGL-CAUSAL-AFFECT-ETDRK4-v1 when γ_A ≠ 0
- Add variant_label to all write paths
- Add artifact_hash computation to validation_pipeline

### Backwards compatibility
All existing artifacts (pre-DC-v1.0) are tagged `LEGACY` in the ledger via the `refinement_status` = `PHYSICS_UNCERTIFIED` pathway. They cannot be ranked against DC-v1.0 artifacts without explicit override.
