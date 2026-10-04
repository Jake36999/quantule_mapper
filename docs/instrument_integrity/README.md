---
tags: [index, validation, instrument-integrity]
date: 2026-10-04
branch: Branch - Validation - Index
status: running
---

# Instrument integrity (October 2026 onward)

This folder holds the solver and stepper correctness work that began on 2026-10-02. It is kept apart
from the earlier docs so that work done before and after the ETDRK4 fix cannot be confused.

Earlier integrity material stays where it was:
- [[BASELINE_AUDIT_NUMERICAL]]
- the instrument-integrity ledger in [[IRER_MASTER_HYPOTHESIS_CATALOG]] §10
- `docs/evidence_package/07_*`

| doc | kind | status |
|---|---|---|
| [[SOLVER_AND_RUNTIME_CHANGELOG]] | changelog — **every** solver/runtime change | running |
| [[ETDRK4_INTEGRATOR_BUGS_2026-10]] | bug report + fix record | complete |
| [[SEARCH_STACK_AUDIT_2026-10]] | audit of the Hunter / search stack | complete |
| [[STEPPER_ORDER_GATES]] | record of the Phase A order and manufactured-solution gates | — |
| [[REVALIDATION_E270CDC_RESULTS]] | full-replay re-validation of a\* and C2.7 | — |

`evidence/` holds the scripts and raw outputs behind each record, one subfolder per investigation.

The plans these docs serve are in `docs/research_infrastructure/`.

## Branches
- [[Branch - Validation - Index]]
