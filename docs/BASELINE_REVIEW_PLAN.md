# Quantule Mapper — Baseline Review & Hardening Plan (plan of record)

**Premise:** Phase C is closed as a scoped success (see `PHASE_C_SCOPE_RECONCILIATION.md`). We now establish a
reproducible, scientifically defensible **baseline of what exists**, before any redesign.

## Governing rule (non-negotiable)

> **Do not redesign while auditing.** Audit documents describe only *what exists and what it demonstrates*.
> Any "we should change / add / replace X" goes to a **parking lot** (Stage 2/3 docs), never inline in an audit.
> An audit that starts saying "I would replace this…" has stopped being an audit.

**Evidence discipline (applied in every sub-audit):** tag each item with a status, and cite the artifact
(file:line, run id, or ledger row) that supports it. Taxonomies:
- Physics/theory: `CONFIRMED` · `INFERRED` · `ABSENT` · `PLACEHOLDER` · `FALSIFIED`
- Validation metrics: `DISCRIMINATING` · `NON-DISCRIMINATING` · `UNTESTED`
- Code/architecture: `PRODUCTION` · `EXPERIMENTAL` · `TECH-DEBT` · `DEAD/LEGACY`

---

## Stage 0 — Evidence inventory  → `EVIDENCE_INVENTORY.md`
*(added; the accumulated data is the audit's evidence base and is now on-disk-only after the repo cleanup)*
- Catalogue the load-bearing results: a\* gain-ladder + `FEB_ASTAR_CONFIRM`, kick/inertia, adiabatic-drag
  morphology ladders, feb-basin / joint-basin, N128 resolution, the SQLite ledgers, key saved `psi_fin` states.
- For each: run id, script, params, what claim it supports, where it lives on disk (not in git).
- Produce an **evidence manifest** (hashes + paths) so every audit assertion is traceable to data, and so the
  GB-scale results survive outside version control.

## Stage 1 — Baseline audit  → `BASELINE_AUDIT.md` (five sub-audits)

**1.1 Physics** — formally document the *implemented* operator: `L_k = −D·k² − η + iω₀` (real diffusive; ω₀=0),
cubic-quintic-septic real gain, non-local "splash", conformal `Ω²(ρ)` covariant Laplacian, ETDRK4. Map each
IRER concept to `CONFIRMED/INFERRED/ABSENT/PLACEHOLDER/FALSIFIED` (e.g. geometry↔density coupling = CONFIRMED;
prime-harmonic resonance = FALSIFIED/null; Payan spin, tensor routing, BSSN = ABSENT/FALSIFIED). Builds on
`PHASE_C_SCOPE_RECONCILIATION.md` and `IRER_MATH_REFERENCE.md`.

**1.2 Numerical** — ETDRK4 (Kassam–Trefethen contour), dealiasing (0.5 vs 2/3), dt/CFL, FP64 geometry path,
boundary/periodicity, conservation diagnostics, convergence (N48/96/128). **Central question: could any
numerical artefact masquerade as physics?** Codify the already-learned lessons: the T6000→T24000→T72000
*validation-window artefact* (short windows over-reported bound states); the 2-term-vs-3-term collapse-balance
approximation and permissive `collapse_threshold` flagged in `IRER_MATH_REFERENCE.md`.

**1.3 Validation** — inventory every metric. Anchor: the `css.classify` **v3 gate** is the promotion criterion
(note its single-exemplar calibration caveat). Mark prime-SSE and TDA/Betti `NON-DISCRIMINATING` (null across
stable/failed). **Document the two-path split** (`PHASE_C_VALIDATION_STACK_AUDIT.md`): jax-scout gates on
`css.classify`; the richer `validation_pipeline.py` (prime-SSE/TDA/tensor/MC) is the CuPy stack on HDF5 and is
*not* applied to jax-scout `.npz`. Per metric: hypothesis tested / discriminates? / reproducible? / null test? /
keep–demote–remove.

**1.4 Architecture** — Hunter (`aste_hunter.py`: NSGA-II, **objective mis-aligned** — optimises prime-SSE, a
signature the substrate can't generate), Worker (CuPy production + jax_scout FP64 mirror; frozen geometry
`e8d6a78ea`; equivalence claim — verify against `PHASE_C_METHOD_PARITY_AUDIT.md`), Validator (two paths),
Orchestrator + SQLite ledger + provenance + `config_hash` determinism, DC-v1.0 data contract + governance
scanner. Classify `PRODUCTION/EXPERIMENTAL/TECH-DEBT/DEAD`; list the legacy/falsified branches (BSSN, Stage-B
tensor routing, Payan coupling, the afield_* routing line).

**1.5 Scope** — a single canonical page: *what the simulation can legitimately claim* vs *what is explicitly
out of scope*, promoted from `PHASE_C_SCOPE_RECONCILIATION.md` to the permanent project boundary.

## Bridge — Provenance kinetic audit  → `PHASE_D_FORMALISM_GAP_REVIEW.md`
*(a reading task, prerequisite to Stage 3 — not a simulation)*
Search the Declaration appendices for whether IRER pins down first- vs second-order time dynamics, real-diffusive
vs complex-dispersive kinetics, and whether Payan/FMIA/forces imply transport. Outcome ∈ {theory is dissipative
(mobility null is a theory result) · theory is conservative (baseline is stability-sector only) · under-specified
(Phase D must choose and justify)}. This **gates** whether any capability expansion is theory-warranted.

## Stage 2 — Baseline hardening  → `BASELINE_HARDENING_PLAN.md`
*(defensibility only — no new PDE terms, operators, or IRER assumptions)*
Candidates (to be prioritised from Stage 1 findings): reconcile or cleanly delineate the two validation paths;
formalise + test the jax↔CuPy frozen-geometry equivalence; codify the "validation window long enough" gate as a
standing rule; demote null diagnostics (prime-SSE/TDA) to explicitly-flagged exploratory; align the hunter's
search objective with the *validated* physics (retire prime-SSE as the fitness core) — flagged as a tool change
requiring its own re-validation, not a physics change; evidence manifest + reproducibility runbook; test coverage
for the classifier gates; remove duplicate metrics; GPU/profiling. **No mobility work here.**

## Stage 3 — Capability expansion RFCs  → `CAPABILITY_EXPANSION_RFC.md` (one per capability)
*(only after Stages 1–2 + the provenance audit)*
Each proposed capability (candidates: complex/dispersive kinetics `i·D_disp·k²`; prime-forcing term; second-order
time; conservative core for spin; A-field transport) documented as a **hypothesis**, with: which IRER hypothesis
requires it · exact PDE change · numerical/solver/GPU consequence · new testable observable + failure modes ·
**risk to the validated baseline** (does it invalidate prior results? can it coexist? is it default-off /
contract-stamped, like the drag variant?). No implementation until an RFC is accepted.

## Stage 4 — Roadmap  → `ROADMAP_V2.md`
Three gated tracks: **A** baseline / scientific-software / reproducibility (start now); **B** experimental physics
/ new operators (gated on Stages 1–2 + provenance audit); **C** infrastructure / GPU scaling / hunter /
distributed execution. Prioritised, with explicit entry gates per track.

---

**Suggested execution order:** Stage 0 → 1.1/1.5 (largely ready now) → 1.2/1.3/1.4 → provenance bridge → Stage 2
→ (Stage 3 RFCs / Stage 4 roadmap). Nothing in Stages 0–2 changes physics; Stage 3 only *proposes* it.
