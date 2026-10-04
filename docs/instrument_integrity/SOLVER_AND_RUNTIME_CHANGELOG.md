---
tags: [record, validation, changelog]
date: 2026-10-04
branch: Branch - Validation - Index
status: running
---

# Solver and runtime changelog

This is the single record of **every change to the solvers, steppers, provenance or run-time
plumbing**, starting October 2026. The newest entry goes at the top.

Each entry states:
- what changed, with commit and files
- why it changed
- whether it changes numerical output
- how the change was verified
- which earlier results it affects

**Rule:** a commit that touches `solver/`, a stepper in `jax_scout/`, `jax_scout/provenance.py`,
`jax_scout/snapshots.py` or a runtime tool is not done until it has an entry here.

**Output-changing?** means the change alters the numbers a run produces. "No" means it changes only
bookkeeping, telemetry or tests.

---

## Index

| date | id | summary | output-changing? | commit |
|---|---|---|---|---|
| 2026-10-02 | CL-001 | ETDRK4: complex-safe contour coefficients + stage-c `N_n` | **YES** | `e270cdc` |

---

## CL-001 — ETDRK4 coefficient and stage-c fix
- **Commit:** `e270cdc` · branch `fix/etdrk4-integrator-order`
- **Files:**
  - `solver/etdrk4_coeffs.py` (new)
  - `solver/core.py` (coefficient construction now calls `etdrk4_coefficients`)
  - `solver/kernels.py` (`compute_kt_stage_c(E2, a_k, Q, N_b, N_n)`)
  - `jax_scout/physics.py` (full-circle contour, complex coefficients cast to `cd`, stage c uses `n_n`, `KT_CONTOUR_M` 64 → 128)
  - `tools/conservative_geometry_campaign.py`
  - `tools/conservative_stepper_contract_audit.py`
  - `tests/test_etdrk4_order.py` (new)
- **Why:** two bugs made ETDRK4 roughly order 0.6. See [[ETDRK4_INTEGRATOR_BUGS_2026-10]].
- **Output-changing?** **Yes**, for every run on an ETDRK4 path: Phase C dissipative, C1 dispersive
  and the C2 conservative NLS branch.
  - At the production dt=0.005: ψ error after T=2 goes from 8.5% to 0.27% (N=48).
  - The coefficients are now complex128 rather than float64, which doubles their memory: 4 extra
    N³ complex arrays, about 32 MB each at N=128.
- **Not affected:** `jax_scout/phase_d_c3_wave.py` (KG split-step) and every TG harness.
- **Verified:**
  - `tests/test_etdrk4_order.py`: 5 tests pass. The CuPy step test fails at order 2.01 when the old
    stage c is put back.
  - GPU convergence table in the bug record.
  - The full test suite shows the same 25 failures before and after; all of them predate this change.
- **Not yet verified:** the JAX edit has not been run (no jax in `.venv`). Covered by Phase B3.
- **Results affected:** a\*, C2.7, C2 norm drift, C1. Status: STALE_PENDING_REVALIDATION.

---

## Associated docs
- [[ETDRK4_INTEGRATOR_BUGS_2026-10]] · [[IMPLEMENTATION_PLAN_2026-10]] · [[BASELINE_AUDIT_NUMERICAL]]

## Branches
- [[Branch - Validation - Index]]
