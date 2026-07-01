---
name: dc-v1-hardening-state
description: Where the Quantule Mapper data-contract (DC-v1.0) hardening stands and what is intentionally deferred
metadata: 
  node_type: memory
  type: project
  originSessionId: 0e9a0bad-bd6b-44f4-aea3-dc2feb389751
---

DC-v1.0 data-contract hardening landed 2026-06-18. **Closed in code + tested** (all GPU-independent): HDF5 `/identity` group (solver/run.py via `orchestrator/run_identity.py`), `A_dot_final`→`A_dot_k_final` rename (reader keeps legacy alias), ledger composite PK `(config_hash, seed)` + discriminator columns + `migrate_runs_to_composite_pk()`, `param_affect_coupling`, collision-free provenance via `run_identity.provenance_path_for_artifact`, and the k=0 affect-field secular-runaway gate in `solver/core.py:update_field_of_affect` (proven by `tests/test_run_identity.py::TestK0Runaway`). Full status table lives in `docs/DATA_CONTRACT.md §2.1`; math resolutions in `docs/IRER_MATH_SANITY_CHECK.md §7`.

**Phase 2 landed 2026-06-18 (also tested, GPU-independent):**
- Compatibility gate WIRED: `result_processor._evaluate_compatibility_gate` (uses `run_identity.are_rankable`) marks incompatible runs `champion_eligible=0` + `PHYSICS_UNCERTIFIED`; `orchestrator_engine` promotion chain has a `_champion_eligible` guard. New `champion_eligible` ledger column. Tests: `test_compatibility_gate.py`.
- `param_rho_vac` SPLIT into `param_omega0` (oscillator, `core.py` L_k `i*omega0`) + `param_rho_vac` (geometry). Back-compat: omega0 defaults to rho_vac. Canonical defaults (1.0) in `contracts.py`. burn_in rho_vac bound 0.0→0.05; param_omega0 bounds added. Tests: `test_data_contract.py::TestRhoVacOmega0Split`.
- MCP read-only tools BUILT: `mcp_server/` package (NOT `mcp/` — shadows SDK), 8 tools in `data_access.py` + FastMCP `server.py`. Tests: `test_mcp_read_tools.py` (15).

**Phase 3 landed 2026-06-18 (write/GPU MCP tools + runbook, all gates tested):**
- `orchestrator/path_utils.py` — pure output-hierarchy path builder (`build_artifact_path`, staging/smoke paths) per OUTPUT_HIERARCHY.md.
- `mcp_server/guards.py` — power-of-2, degenerate-geometry (rho_vac<0.05), CFL (`c_affect*dt*k_cut<=1`), overwrite, smoke caps.
- `mcp_server/write_tools.py` + 4 server tools: `stage_simulation_manifest` (no GPU, fully tested), `run_simulation_manifest`/`run_smoke_simulation`/`validate_artifact` (GPU/CPU exec injected via `launcher`/`runner` so gates are tested here). Server now registers **12 tools** (8 read + 4 write). Tests: `test_mcp_write_tools.py` (25). Stage→review→run: output path comes from the staged manifest (cannot be redirected); confirm=true + <30min freshness required.
- `docs/GPU_RUNBOOK.md` — exact commands for ledger migration + γ_A=0 byte regression (expects psi/omega/energy max|Δ|=0; A-field WILL differ due to k=0 gate — that's correct, not a regression) + smoke + MCP launch.

Full session suite: 138 tests pass (test_data_contract, test_run_identity, test_ledger_identity, test_compatibility_gate, test_mcp_read_tools, test_mcp_write_tools, test_phase_v13).

**Still deferred:**
- GPU box: execute GPU_RUNBOOK steps 1–2 (migration + regression) — cannot run here (no CuPy).
- A-coupling itself (vacuum_ref modulation) — gate cleared, still unimplemented.
- `provenance_hash` ledger column exists but not yet populated; write tools' default launcher/runner unrun (GPU/heavy-dep only).

**Why this matters:** ordering is hardening → GPU γ_A=0 regression → A-coupling → falsification ladder, per two reviewer agents. See [[no-cupy-dev-box]].

Pre-existing failures NOT caused by this work: `test_validation_derived_metrics.py::test_validation_gate_topological_override_keeps_tda_path` (TDA override feature unimplemented; fails on HEAD too), `test_e2e_integration.py` (FastAPI/Starlette version mismatch at collection), `test_no_subprocess_bypass.py` (targets missing `adaptive_hunt_orchestrator.py`).
