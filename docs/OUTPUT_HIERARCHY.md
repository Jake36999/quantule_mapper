# Output Hierarchy — IRER Simulation Stack

**Version:** OH-v1.0  
**Date:** 2026-06-18  
**Companion:** docs/DATA_CONTRACT.md

---

## 1. Proposed Directory Hierarchy

```
{project_root}/
│
├── runs/                                 # Primary artifact store (NEW — replaces scattered simulation_data/)
│   └── {hunt_name}/                      # e.g. BURN_IN_STRESS_TEST_001
│       └── {YYYY-MM-DD}/                 # UTC date of hunt start
│           └── {solver_contract}/        # e.g. IRER-SNCGL-LOCAL-RHO-ETDRK4-v1
│               └── {variant_label}/      # LOCAL-RHO / CAUSAL-AFFECT / ADDITIVE-POT
│                   └── gen_{NNNN}/       # Zero-padded generation, e.g. gen_0003
│                       └── {hash12}_{seed}/   # First 12 chars of config_hash + seed
│                           ├── artifact.h5        # HDF5 simulation output
│                           ├── params.json        # Full parameter dict at launch
│                           └── provenance.json    # Validation provenance (post-pipeline)
│
├── pareto_snapshots/                     # Pareto-front snapshots per generation
│   └── {hunt_name}/
│       └── gen_{NNNN}_pareto.json
│
├── archive_runs/                         # GC-eligible artifacts moved here before deletion
│   └── {hunt_name}/
│       └── gen_{NNNN}/
│           └── {hash12}_{seed}.h5.gz    # Compressed on bleed
│
├── provenance_reports/                   # LEGACY path — kept for backwards compat
│   └── provenance_{config_hash}.json    # OLD naming (collision-prone, deprecated)
│
├── simulation_ledger.db                  # SQLite — authoritative ledger
│
├── queue.db                              # SQLite — job queue (separate DB)
│
└── runtime_logs/
    ├── run_lifecycle_audit.jsonl         # Append-only structured event log
    ├── orchestrator_{date}.log           # Rotating plaintext logs
    └── worker_{worker_id}_{date}.log
```

---

## 2. Naming Conventions

### Artifact directory name

```
{first_12_of_config_hash}_{seed}
```

Example: `a3f8c2e91b4d_42`

This makes each directory unique per (params × seed) pair. Two runs of the same params with different seeds get different directories.

### config_hash

SHA-256 of `json.dumps(params, sort_keys=True)`. Computed by `orchestrator/contracts.py:JobManifest.from_params`.

### Provenance filename (new)

```
provenance_{config_hash[:12]}_{seed}_{YYYYMMDD_HHMMSS}_{run_id[:8]}.json
```

Example: `provenance_a3f8c2e91b4d_42_20260618_143022_3f9a1b2c.json`

### Solver contract version → variant_label mapping

| solver_contract_version | variant_label | affect_topology |
|---|---|---|
| IRER-SNCGL-LOCAL-RHO-ETDRK4-v1 | LOCAL-RHO | none |
| IRER-SNCGL-CAUSAL-AFFECT-ETDRK4-v1 | CAUSAL-AFFECT | vacuum_ref |
| IRER-SNCGL-ADDITIVE-POT-ETDRK4-v1 | ADDITIVE-POT | additive_potential |

---

## 3. HDF5 Internal Layout

### Top-level groups and datasets

```
artifact.h5
│
├── /identity/                    # Run discriminator block (NEW)
│   ├── hunt_name                 S256
│   ├── run_id                    S64
│   ├── utc_start                 S64    ISO-8601
│   ├── generation                int64
│   ├── config_hash               S128
│   ├── seed                      int64
│   ├── solver_contract_version   S128
│   ├── variant_label             S64
│   ├── affect_topology           S64
│   ├── affect_strength           float64
│   ├── git_commit                S64
│   ├── N_grid                    int32
│   ├── dt                        float64
│   ├── T_steps                   int32
│   └── gpu_backend               S32
│
├── /psi_final                    complex128  (N,N,N)  gzip-4
├── /omega_sq_final               float64     (N,N,N)  gzip-4
├── /A_final                      float64     (N,N,N)  gzip-4   real-space
├── /A_dot_k_final                complex128  (N,N,N)  gzip-4   SPECTRAL-space (⚠ renamed)
│
├── /telemetry/
│   ├── step                      int64       (T,)     gzip-4
│   ├── energy                    float64     (T,)     gzip-4
│   ├── energy_sparkline          float64     (100,)   gzip-4
│   └── C_invariant               float64     (T,)     gzip-4
│
├── /extended_telemetry/
│   ├── step_count                int64       (1,)
│   ├── sim_time                  float64     (1,)     wall-clock seconds
│   ├── dt                        float64     (1,)
│   ├── grid_shape                int32       (1,)
│   └── params_hash               S128        (1,)
│
├── /solver_contract              S512        (1,)     JSON string (keep for compat)
│
│   [FAIL path only:]
├── /sentinel_code                float64     (1,)     1002/1003/1004
└── /sentinel_reason              S64         (1,)
```

---

## 4. Compatibility Rules

### Rule C-1: Solver contract must match before ranking

Runs with different `solver_contract_version` values MUST NOT be ranked on the same fitness leaderboard. The result_processor must set `_refinement_status = "PHYSICS_UNCERTIFIED"` if the contract does not match `orchestrator/contracts.SOLVER_CONTRACT_VERSION`.

### Rule C-2: Variant label isolates run pools

The Pareto archive and champion selection must operate per `variant_label`. A LOCAL-RHO champion is not comparable to a CAUSAL-AFFECT candidate.

### Rule C-3: Grid size must match for spectral comparison

`log_prime_sse` depends on k-space resolution. Runs with `N_grid=64` and `N_grid=128` are measuring at different spectral scales. They cannot be ranked together. The compatibility gate checks `N_grid` equality.

### Rule C-4: Legacy artifacts are permanently isolated

Any artifact written before DC-v1.0 (i.e., lacking the `/identity` group) must be tagged `LEGACY` and `PHYSICS_UNCERTIFIED`. They may be inspected but must not enter a live leaderboard.

### Rule C-5: param_rho_vac=0 runs flag as degenerate geometry

Any run where `param_rho_vac < 1e-6` should be tagged `DEGENERATE_GEOMETRY` in the metrics table, because Ω² = (ρ_vac/ρ)^α → 0 everywhere (geometry collapses to the conformal floor). These runs test a degenerate limit, not the full SNCGL PDE.

---

## 5. Module → Output Mapping

| Module | Output | Location |
|---|---|---|
| solver/run.py | HDF5 artifact | runs/{hunt}/{date}/{contract}/{variant}/gen_NNNN/{hash}_{seed}/artifact.h5 |
| solver/run.py | Lifecycle event | runtime_logs/run_lifecycle_audit.jsonl |
| validation_pipeline.py | Provenance JSON | runs/.../provenance.json (new) OR provenance_reports/ (legacy) |
| validation_pipeline.py | Lifecycle event | runtime_logs/run_lifecycle_audit.jsonl |
| result_processor.py | Ledger rows | simulation_ledger.db (runs, parameters, metrics) |
| result_processor.py | Lifecycle event | runtime_logs/run_lifecycle_audit.jsonl |
| orchestrator_engine.py | Pareto snapshot | pareto_snapshots/{hunt}/gen_NNNN_pareto.json |
| artifact_gc.py | Compressed archive | archive_runs/{hunt}/gen_NNNN/{hash}_{seed}.h5.gz |
| quantulemapper_real.py | log_prime_sse | Embedded in provenance JSON (spectral_fidelity.log_prime_sse) |

---

## 6. Transition from Current Paths

The current stack uses `simulation_data/` as `data_dir` (from burn_in_config.json). This path is passed through `JobManifest.session_dir` and `generation_dir`.

**Transition plan:**
1. Add `artifact_root` key to burn_in_config.json pointing to `runs/`
2. In solver/run.py, build the full hierarchical path from identity fields rather than accepting a flat `output_path`
3. Keep `output_path` as a fallback for backwards compat — if it lacks a `/identity/` write, write it flat as before
4. Update orchestrator_engine.py to build paths using the new hierarchy builder (to be added to `orchestrator/path_utils.py`)

### Backwards-compatible path builder

```python
def build_artifact_path(
    root: str,
    hunt_name: str,
    utc_date: str,        # YYYY-MM-DD
    solver_contract: str,
    variant_label: str,
    generation: int,
    config_hash: str,
    seed: int,
) -> str:
    return os.path.join(
        root,
        hunt_name,
        utc_date,
        solver_contract,
        variant_label,
        f"gen_{generation:04d}",
        f"{config_hash[:12]}_{seed}",
        "artifact.h5",
    )
```
