# Baseline Audit — Rollup (Stage 1, COMPLETE)

The baseline audit establishes *what the current Quantule Mapper stack is, what it demonstrates, and how robust
that is* — descriptively, treating the implementation as the reference system (no redesign). Five sub-audits:

| # | sub-audit | doc |
|---|---|---|
| 1.1 + 1.5 | Physics & Scope | `BASELINE_AUDIT_PHYSICS.md` |
| 1.2 | Numerical | `BASELINE_AUDIT_NUMERICAL.md` |
| 1.3 | Validation | `BASELINE_AUDIT_VALIDATION.md` |
| 1.4 | Architecture | `BASELINE_AUDIT_ARCHITECTURE.md` |
| 0 | Evidence inventory | `EVIDENCE_INVENTORY.md` |

## Top-line findings

1. **What the stack is** — a **local, real (dissipative) cubic-quintic-septic Ginzburg–Landau field on a
   density-sourced scalar conformal geometry**, first-order in time (ETDRK4). The "Non-Local" and "Complex" of
   "S-NCGL" are *not* realized as non-local coupling or complex kinetics in the validated substrate. **CuPy
   production and the jax_scout mirror share the same local RHS** (parity confirmed at code level).

2. **What it legitimately demonstrates** (via the *discriminating* `css.classify` v3 gate + a\*-arc late-slope):
   reproducible long-time standing bound attractors governed by a gain/loss balance (a\* ≈ ×1.15, ±~0.5%), a
   parameter-controlled basin, a participating (stabilizing) density-sourced geometry, and a local
   accretion/nucleation response. **Scope = stability sector.**

3. **What it does not / cannot** — no inertial or relational mobility, no transport (scope boundary of the real
   operator); and the theory-side hypotheses **prime-harmonic resonance, TDA topology, tensor routing, Payan
   spin** are `FALSIFIED/NULL` as active mechanisms in this substrate.

4. **Numerics** — the integrator (ETDRK4, spectral, FP64, dealiased) is sound. The one artefact that *did*
   masquerade as physics — short-window "saturation" over-reporting bound states — was **caught and corrected**
   (late-slope→0 criterion, longer-T + N128 confirms) and is now a standing validation rule.

5. **Validation** — the closed claims rest on `css.classify` **alone**; the production `validation_pipeline.py`
   was never applied to Phase C output, and its headline metrics (prime-SSE, TDA) are `NON-DISCRIMINATING`. Honest
   attribution: the result stands on the metrics that discriminate; the null metrics are correctly outside the
   evidence chain.

6. **Architecture** — the load-bearing backbone is production-grade (RHS-parity solver, mature orchestrator /
   contracts / ledger / tests). The debt is **accumulation and disconnection** (dead experiment scripts; the
   hunter and the production-validation pipeline disconnected from the validated jax_scout gate), not
   core-correctness.

## Consolidated residuals → Stage 2 hardening backlog (recorded, not actioned)
- **Parity:** produce a one-shot jax↔CuPy bit-parity artifact (RHS confirmed; linear operator + output
  un-verified).
- **Long-time numerics:** quantify FP64 accumulation at the longest T (a dt-convergence / higher-precision
  control run); currently argued-physical, not measured.
- **Gate calibration:** the v3 thresholds are single-exemplar (feb) — broaden the calibration set.
- **Validation reconciliation:** the production pipeline vs the operative gate; the HDF5↔npz adapter gap.
- **Hunter:** re-aim the objective onto the validated stability criterion; reconnect or scope its role.
- **Cleanup:** archive/prune the 25 dead `afield/payan/bridge/corridor/transfer` scripts; the two-solver
  duplication.
- **Tests:** add solver-parity and mobility-script coverage.
- **Config/diagnostics:** `param_rho_vac` default mismatch (0 vs 1.0), permissive `collapse_threshold` (1e10),
  `collapse_dynamics` 2-term heuristic.
- **Evidence:** off-box archive of the ~22 load-bearing runs + key states (Git stays lightweight by choice).

## Status & next
**Stage 1 is complete. No physics was changed; nothing was redesigned.** Per `BASELINE_REVIEW_PLAN.md` the next
step is the **Provenance Kinetic Audit** (`PHASE_D_FORMALISM_GAP_REVIEW.md`) — a reading of the Declaration
appendices to settle whether IRER intends a dissipative or a conservative/dispersive substrate. That gates
whether any Stage-3 capability expansion is theory-warranted, and it is a *reading* task, not a run. Stage 2
(hardening) draws its backlog from the residuals above.
