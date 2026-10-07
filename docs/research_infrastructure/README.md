---
tags: [index, infra, plan]
date: 2026-10-04
branch: Branch - Validation - Index
status: running
---

# Research infrastructure (October 2026 onward)

This folder holds the plans and build records for the experiment platform:
- order gates
- stale-flagging
- telemetry
- provenance
- specs
- MCP
- UI
- basin mapping

It is kept apart from earlier infrastructure documents such as [[INFRASTRUCTURE_AND_ORCHESTRATION_ASSESSMENT]],
[[SESSION_SYNTHESIS_2026-09-17]] and [[INTEGRATED_PLAN_2026-09]], which remain the record of what came before.

| doc | kind | status |
|---|---|---|
| [[PROCESS_PLAN_2026-10]] | process plan (why) | complete |
| [[IMPLEMENTATION_PLAN_2026-10]] | phased implementation plan (what/how) | running |
| [[TELEMETRY_STREAM]] | Phase C build record | complete |
| [[PROVENANCE_AND_HARNESS_REGISTRY]] | Phase D build record | complete |
| [[HARNESS_REGISTRY]] | generated registry of every run-writing script | generated |
| [[EXPERIMENT_SPECS_MCP_UI]] | Phase E build record | complete |
| [[BASIN_MAPPING]] | Phase F build record | complete |
| [[RUN_VIEWER]] | run gallery + viewer + recording + render queue | complete |
| [[BATCHED_RUNS]] | vmapped sweeps + where GPU time goes | complete |
| [[SCREEN_AND_VERIFY]] | fp32 screen → cluster → fp64 verify of boundaries and outliers | complete |

Phase records are added here as each phase closes. A change to a solver or the runtime is recorded in
[[SOLVER_AND_RUNTIME_CHANGELOG]] (`docs/instrument_integrity/`).

## Branches
- [[Branch - Validation - Index]]
