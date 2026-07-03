# Baseline Audit — Architecture (Stage 1.4)

**Descriptive only.** Component inventory + maturity classification: `PRODUCTION` · `EXPERIMENTAL` ·
`TECH-DEBT` · `DEAD/LEGACY`. Dispositions are `→ Stage 2` pointers, not actions.

## Method-parity resolution (the open question from 1.1/1.2)
The central integrity question was: does the CuPy **production** solver differ from the jax_scout **mirror** (the
substrate all Phase C ran on)? **Code-level answer: the nonlinear physics is identical.**
- `solver/kernels.py:calculate_nonlinear_rhs(psi, rho, lap_cov, lap_flat, D, a, s, f)` ≡ jax_scout
  `physics._nonlinear_rhs` (same signature; local cubic-quintic-septic).
- Both alias the production "splash" params to the local coefficients: `s←param_splash_coupling`,
  `f←param_splash_fraction` (`solver/core.py:71-72`, `physics.py:358-359`). **Neither is non-local.**
- The non-local **Field of Affect** *is* computed in production (`solver/core.py:update_field_of_affect`, called
  `solver/run.py:78`) but is **not dynamically coupled** into the baseline ψ-evolution
  (`IRER_MATH_REFERENCE.md:342`); in the mirror it is the optional, default-off `a_vec`. So **both baselines are
  local** and consistent.

`RESIDUAL:` the **linear operator** `L_k` and a **bit-level output comparison** were not re-verified line-by-line
here; RHS parity + shared ETDRK4 derivation strongly indicate faithfulness, but a one-shot bit-parity artifact is
still a `→ Stage 2` item. Net: the "jax_scout is equivalence-proven to CuPy" claim is now **code-parity confirmed
on the RHS**, downgraded-caveat rather than unproven.

## Components

### Solvers — TWO stacks (physics-parity confirmed)
| component | files | status | note |
|---|---|---|---|
| CuPy production solver | `solver/{core,kernels,run}.py`, `worker_cupy.py`, `worker_daemon.py` | `PRODUCTION` | the "real" engine; not runnable on the dev box (no CuPy); computes uncoupled Field-of-Affect |
| jax_scout FP64 mirror | `jax_scout/physics.py`, `core_saturation_search.py` | `PRODUCTION` (research) | **the substrate every Phase C claim ran on**; RHS-faithful to production |
| `TECH-DEBT` | — | — | two solver codebases to keep in sync = maintenance burden; parity is asserted, only RHS-verified |

### Hunter
`aste_hunter.py` (NSGA-II + SGN + SBD + ASMT, SQLite ledger). `PRODUCTION`-grade code, **but**: (a) objective
mis-aligned (optimises `log_prime_sse` — non-discriminating, see validation audit); (b) its ledger
(`simulation_ledger.db`) is **not on this box** (lives on the production path); (c) **decoupled from the validated
workflow** — Phase C used direct `feb_*` runners, not the hunter. `→ Stage 2:` re-aim objective + reconnect or
scope its role.

### Orchestration / contracts / provenance — the backbone
`orchestrator/` (18: engine, service, `result_processor`, `contracts`, `job_manifest`, `path_utils`,
`diagnostics/`), `backlog_orchestrator.py`, `trigger_api.py`, `app.py`; DC-v1.0 **data contract**; `config_hash`
deterministic identity; SQLite ledger + `provenance_reports`. `PRODUCTION` — the most mature layer; governance
scanner reports 16/16 contract-compliant (`test-bench-adaptation`).

### Validation stack
`validation_pipeline.py` (11-stage CuPy), `quantulemapper_real.py` (prime-SSE), `tda_profiler.py`. `PRODUCTION`
code but **disconnected from the actual gate** (runs on HDF5, never on Phase C `.npz`; core metrics
non-discriminating — see validation audit). The operative gate is `css.classify` in the jax_scout path.

### Tests — reasonably mature
50 `test_*.py` (+ fixtures/mocks): `test_data_contract`, `test_hunter*`, `test_ledger_identity`,
`test_compatibility_gate`, `test_core_saturation_{search,replay,robustness,collapse_diag}`,
`test_e2e_integration`, `test_boundary_continuity`, `test_legacy_artifact_not_promoted`. `PRODUCTION` for the
contract/hunter/ledger/classifier layers. `Gaps:` no jax↔CuPy solver-parity test; the mobility scripts
(`feb_kick_inertia`, `feb_adiabatic_drag`) and the drag physics-variant have no unit tests.

### jax_scout experiment scripts — accumulation
~6 core/shared + **13 recent `feb_*`** (the Phase C arc; some load-bearing, all one-off runners) + **25
`afield_*`/`payan_*`/`bridge_*`/`corridor_*`/`transfer_*`** = the A-field-routing / Payan / Stage-B-tensor /
bridge-hunt era. The 25 are `DEAD/LEGACY` (their hypotheses were falsified — routing `NO_SUPPORT`, Payan
`NO_SIGNAL`, tensor `NO_SUPPORT`). `TECH-DEBT:` ~38 one-off experiment scripts live alongside the 6 core modules;
the dead set should be archived/pruned (`→ Stage 2`).

### Supporting packages
`plugins/` (19), `quantule_viz/` (14), `tools/` (14), `run/` (7), `metrics/` (6), `mcp_server/` (6),
`gravity/unified_omega.py` (Ω² factor), `compiled_knowledge_base/` (5). Mixed `PRODUCTION`/`EXPERIMENTAL`; not
audited in depth here (not load-bearing for the closed claims) — flagged for a lighter pass if needed.

## Reproducibility & environment
- **Deterministic:** `config_hash` identity, frozen geometry `e8d6a78ea`, resumable runs (`--out` skips done
  keys), SQLite ledger + provenance. `PRODUCTION`-adequate.
- **Three-environment split:** Windows dev (`py_compile` only — no cupy/jax), WSL GPU (jax_scout runs),
  CuPy production GPU box. Documented; a real constraint on what can be verified where.
- `RESIDUAL:` results not versioned (by design — Git is lightweight; `EVIDENCE_INVENTORY.md` is the manifest);
  jax↔CuPy bit-parity un-artifacted.

## Classification summary
| status | components |
|---|---|
| `PRODUCTION` | CuPy solver, jax_scout core, orchestrator + contracts + ledger, test suite, hunter (code), validation pipeline (code) |
| `EXPERIMENTAL` | the 13 recent `feb_*` runners (research one-offs, some load-bearing) |
| `TECH-DEBT` | two-solver duplication; ~38 accumulated experiment scripts; hunter objective/workflow disconnect; validation-path disconnect; no solver-parity / mobility-script tests |
| `DEAD/LEGACY` | 25 `afield_*`/`payan_*`/`bridge_*`/`corridor_*`/`transfer_*` (falsified hypotheses); BSSN engine; Stage-B tensor path; `CORE_SAT_HUNT` discovery era |

**Load-bearing backbone is production-grade** (solver parity confirmed on RHS, mature orchestrator/contracts/
tests). The debt is **accumulation and disconnection** (dead experiment scripts; hunter and production-validation
disconnected from the validated jax_scout gate), not core-correctness. All fixes are `→ Stage 2`.
