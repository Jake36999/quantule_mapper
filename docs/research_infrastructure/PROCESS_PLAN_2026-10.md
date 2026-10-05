---
tags: [plan, process, verification, infrastructure]
date: 2026-10-04
branch: Main branch
status: complete
---

# Process Plan — October 2026

**Status: superseded by the scheduled [[IMPLEMENTATION_PLAN_2026-10]] (2026-10-04).** This document
keeps the reasoning; the implementation plan has the schedule.

This plan adds steps to the process. It does not replace [[INTEGRATED_PLAN_2026-09]] or
[[SESSION_SYNTHESIS_2026-09-17]] §5–8; it builds on them. It came out of an audit of the
search/optimisation stack and the ETDRK4 solver on 2026-10-02/03.

---

## 1. What triggered this: two integrator bugs that every existing check passed

Commit `e270cdc` (branch `fix/etdrk4-integrator-order`) fixes two ETDRK4 bugs that sat in the CuPy
solver, the JAX mirror and two tools:

1. **Contour coefficients.** The Kassam–Trefethen coefficients took `real()` of an
   upper-half-circle mean. That shortcut only works when `L` is real, and every `L_k` here is complex.
2. **Stage c.** It used `2·N_b − N_a` instead of the Cox–Matthews `2·N_b − N_n`.

| dt (N=48, feb, T=2) | before | after |
|---|---|---|
| 0.01 | 12.7% | 1.7% |
| **0.005** (production) | **8.5%** | **0.27%** |
| 0.0025 | 5.5% | 0.025% |

The observed order went from about 0.6 to about 3.5–4. Evidence: `docs/instrument_integrity/evidence/etdrk4_2026-10/`; full record [[ETDRK4_INTEGRATOR_BUGS_2026-10]].

**Scope.** Affected: everything that ran through ETDRK4 — Phase C, the C1 dispersive runs and the
C2 NLS conservative branch. **Not affected:** C3 KG and the whole TG/gravity stack, which uses the
`phase_d_c3_wave` split-step integrator.

**Why it survived.** [[BASELINE_AUDIT_NUMERICAL]] §1 marked the integrator `CONFIRMED`. CuPy↔JAX
parity agreed to 1.7e-12 because both were copies of the same code. The identity tests check what
the equations conserve, so they cannot see an integrator that solves the right equation
inaccurately. A dt-halving test catches both bugs in seconds (`tests/test_etdrk4_order.py` does
this now).

---

## 2. Steps to add

### P1 — Verification gates for every stepper
- Every stepper (ETDRK4, the C3 split-step, TG) gets a **dt-halving order test** and a
  **manufactured-solution test**.
- These go beside the identity checks, not instead of them.
- Add "wrong stage-c argument" and "half-circle contour + `real()`" to the
  [[gravity_maturity/H3_MUTATION_PROBE_RESULTS|H3]] mutation catalog.

### P2 — An independence rule for parity evidence
- Parity counts as independent evidence **only when the implementations share no code**.
- The natural lane for this is the AMD 5500XT: an OpenCL + `pyvkfft` implementation at small N,
  written from the equations rather than ported from `physics.py`.
- ROCm does not support that card, so OpenCL/Vulkan is the realistic route.

### P3 — Automatic impact listing when a component is fixed
- Add a `stepper` (and later `component_hashes`, see P5) field to the provenance stamp.
- When a component is fixed, every run that used it inside the affected commit range is marked
  `STALE_PENDING_REVALIDATION`.
- This turns "which results does this bug touch?" into a query instead of an afternoon of reading.
- **First use:** the re-validation list in §3.

### P4 — "Derive before you measure" as a spec field
- [[SESSION_SYNTHESIS_2026-09-17]] §7 found that the gap was sequencing, not rigour.
- Make it structural: an experiment spec carries a `prediction` block, filled in **before launch**,
  stating what the equations predict and any parameter-free consequences.
- The verdict is then scored against that prediction.

### P5 — Experiments as data, without a dependency-tracing nightmare
- **Hash only what was imported.** At launch, take `sys.modules`, keep the files from this repo,
  and record each one's `git hash-object`. This is exact, automatic and cheap, and needs no
  syntax-tree logging. It plays the role of a lockfile for code.
- **Three stable seams.** Specs compose these by name:
  - **Substrates:** stepper + operator. There are about three: Phase C ETDRK4, NLS, KG/TG.
  - **Observers:** pure functions from state to scalars, each with a name and version.
  - **Protocols:** initial condition, schedule and stop rule.
- **The spec** is a small validated JSON/YAML record: substrate, operator terms, IC, parameters or
  sweep, observers, stop rule, `prediction` (P4) and `requires:` edges.
- **The run record** is the spec hash + commit + diff hash + component hashes, with
  content-addressed artifacts and a verdict.
- **Storage:** the SQLite ledger can hold it in four tables — specs, runs, artifacts, verdicts.
- **Descriptive, never prescriptive.** `orchestrator/` died because it gated experimentation
  ([[SESSION_SYNTHESIS_2026-09-17]] §6). A `requires:` edge raises a **warning** on an ad-hoc run.
  It never blocks one.
- This sits directly on the harness-manifest proposal in §6 of the synthesis. The manifest describes
  harnesses; the spec describes experiments.

### P6 — Telemetry stream first, agent oversight second
- The `telemetry.jsonl` channel ([[SESSION_SYNTHESIS_2026-09-17]] §5) is the prerequisite for
  agents overseeing runs: they read invariant residuals far more reliably than plots.
- Suggested MCP tool surface (built on the Librarian's existing services):
  - **Read-only:** list runs, tail telemetry, read a verdict, query run summaries.
  - **Write-limited:** an agent may write a *spec file*. It may not launch a run; launches need
    approval.
- Agent roles:
  - **Large DeepInfra models:** query Parquet/SQLite summaries rather than raw dumps; draft first
    passes of the 102 unfilled paired-reading blocks, clearly labelled as machine-drafted; propose
    specs.
  - **Vision/OCR models:** qualitative triage of morphology only, always paired with a numeric
    check.

### P7 — The UI comes from the schemas
- If every substrate, observer and protocol publishes a JSON Schema, a schema-driven form library
  (e.g. react-jsonschema-form) generates the interface automatically. That gives MATLAB-like
  convenience without hand-building adaptive interfaces.
- The same schemas are what the MCP write tools validate against.

### P8 — Basin mapping as its own workflow
- **Stage 1 — forward ensembles:** many ICs and seeds per parameter point; cluster the final
  states on state descriptors.
- **Stage 2 — continuation:** follow each branch with pseudo-arclength continuation, using
  matrix-free Newton–Krylov on the time-T map (JAX autodiff makes this feasible).
  BifurcationKit.jl is the reference design.
- This replaces the GA / inverse-GP search, which blends distinct basins. It needs the fixed solver
  first, because continuation amplifies time-stepping error.

---

## 3. Re-validation list (from §1)

| result | path | what to re-check |
|---|---|---|
| a\* location (×1.15 gain, `param_a ≈ 0.55`), bracket ±0.5% | Phase C ETDRK4 | Re-bracket with the fixed solver; check dt-robustness. |
| C2.7 NLS solitons, `v = 2Dk` (0.9999) | C2 conservative ETDRK4 | Re-run the fit; coefficient error reached 95% at high k on this branch. |
| C2 "quasi-conservative" norm drift | C2 conservative ETDRK4 | Re-measure; Codex's "ETDRK4 norm loss was timestep-sensitive" fits this bug. |
| C1 dispersive-channel results | `physics.py`, `D_imag ≠ 0` | Spot-check headline numbers. |
| JAX mirror edit | `jax_scout/physics.py` | Run `tests/test_etdrk4_order.py` and the parity check in WSL (`~/jax_irer`). |

Not on the list: C3 KG, TG-A/B1S/B2, the H2 box ladder, S1–S3. None of them use ETDRK4.

---

## 4. Suggested order (when the project comes back off the shelf)

1. ~~Commit the solver fix with order gates~~ — **done**, `e270cdc`.
2. P1: order + manufactured-solution gates for the C3 split-step and TG steppers.
3. P3: `stepper` provenance field + stale flags, then work the §3 list.
4. P6: telemetry stream (already scoped as small in the synthesis).
5. P5: import-hash provenance + harness manifest, then the spec layer.
6. P7 + MCP surface (P6).
7. P8: basin mapping.
8. P2: independent OpenCL implementation, whenever there is appetite.

The standing test from [[INTEGRATED_PLAN_2026-09]] still applies to every item: **does this advance
validation, or add machinery?** P1–P3 are validation. P5–P7 are machinery and should stay small.

---

## What changed as a result

- **Code / model changes:** `e270cdc` — ETDRK4 coefficient and stage-c fixes, plus
  `tests/test_etdrk4_order.py`.
- **Verdicts changed:** none yet. §3 lists the results that need re-validation.
- **What was done next, and why:** planning only; the project is partly shelved.

## Issues raised

| # | issue | status | resolved by |
|---|---|---|---|
| 1 | ETDRK4 coefficients invalid for complex `L` | **CLOSED 2026-10-04** | `e270cdc` |
| 2 | ETDRK4 stage c uses `N_a` instead of `N_n` | **CLOSED 2026-10-04** | `e270cdc` |
| 3 | Pre-fix ETDRK4 results not re-validated | OPEN | §3 |
| 4 | JAX mirror edit untested (no jax in `.venv`) | OPEN | §3, last row |
| 5 | Parity checks are not independent evidence | OPEN | P2 |
| 6 | No stepper order gates for C3 / TG | OPEN | P1 |

## Associated docs

- [[INTEGRATED_PLAN_2026-09]] · [[SESSION_SYNTHESIS_2026-09-17]] · [[BASELINE_AUDIT_NUMERICAL]]
- [[gravity_maturity/H3_MUTATION_PROBE_RESULTS]] · [[IRER_MASTER_HYPOTHESIS_CATALOG]]

## Branches

- [[Main branch]]
