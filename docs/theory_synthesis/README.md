# IRER / Quantule Mapper — Theory Synthesis Bundle

A five-report consolidation that connects the **IRER theory** (author: Jake McIntosh) → its **numerical
implementation** in Quantule Mapper → the **simulation evidence** → the **mathematics/identities** now inside the
model. It is the layer above the raw per-experiment docs: it translates IRER concepts into standard language, states
what is active/falsified/paused in code, and prepares the project for external literature and data comparison.

**Framing (binding on all five reports):** this is a *numerical-theory consolidation*, not a proof that IRER is
physically true. No matter claim, no gravity claim, no emergent-physics proof. Every result is a measured observable
with a conservation/parity gate. Conceptual authorship is Jake McIntosh's; AI is a formalization/implementation/review
tool. Nulls, falsifications, and retractions are first-class results.

## The five reports
| # | report | purpose | read it for |
|---|---|---|---|
| 1 | `IRER_THEORY_AND_CONCEPT_TRANSLATION.md` | translate IRER concepts (AIS, OIW, RD, PAS, FMIA, Payan, manifold, RFD, gravity) into standard mathematical/physical language + implementation status + search keywords | "what does this IRER term mean in known physics, and is it in the code?" |
| 2 | `IRER_NUMERICAL_MODEL_AND_MATHS_TRACEBACK.md` | theory → equation → code (module::fn) → observable → result, per sector | "which equation, which script, which metric, which verdict?" |
| 3 | `IRER_MATHEMATICAL_PROOFS_AND_IDENTITIES.md` | identities & proof obligations (passed *and* failed), numerical-theory framing | "what is proven about the model — and which failed identities found the bugs?" |
| 4 | `QUANTULE_MAPPER_ARCHITECTURE_AND_DESIGN_PATTERNS.md` | why the monolith was retired for sector harnesses + separated search/validation/interpretation | "why is it built this way — and what should agents NOT re-merge?" |
| 5 | `IRER_SIMULATION_RESULTS_AND_THEORY_STATUS.md` | public-facing synthesis: confirmed / falsified / retracted / paused / speculative + external-search candidates | "where does the theory stand after simulation contact?" |

## How this relates to the other canonical documents
- **`docs/_Declaration of Intellectual Provenance v9.txt`** — the source of the theory's concepts and authorship
  framing. This bundle copies its disciplined structure but adds implementation + evidence; it never redefines the
  author's concepts, only offers formal translations for testing.
- **`docs/IRER_MASTER_HYPOTHESIS_CATALOG.md`** — the verdict backbone (every tested hypothesis, CONFIRMED/FALSIFIED/
  NULL/RETRACTED/OPEN/PAUSED). Report 5 is its narrative synthesis; Reports 1–3 cite its rows.
- **`docs/PHASE_D_CLOSEOUT_CONSOLIDATION.md`** and the per-experiment docs (`PHASE_C_*`, `PHASE_D_C1..C3_*`,
  `PHASE_D_C2_6_GEOMETRY_OFF_BUG_REPORT.md`, `IRER_GRAVITY_RUNG_A_D_RESULTS.md`, …) — the numbers, telemetry, and
  caveats behind each catalog row.

## Suggested reading order
- **New agent / context reset:** README → Report 5 (status) → Report 1 (translation) → Report 4 (architecture,
  so you don't re-merge search+physics or re-open settled baselines).
- **Verifying a claim:** Report 2 (find the equation+code) → Report 3 (the identity that gates it) → the cited
  per-experiment doc.
- **Preparing external comparison:** Report 1 (search keywords) + Report 5 §8 (analogue/data candidates).

## Status of this bundle & the Codex-audit handoff
This is **Claude's first-pass synthesis** (interpretive, context-heavy). It is grounded in the catalog, the
per-experiment docs, and code read directly this session; items where the exact function/module name was uncertain
are marked `NEEDS_REVIEW` in Report 2. The appropriate next step is a **Codex traceback audit** (execution/verification,
not interpretation): verify every cited function/module path, confirm each equation↔code mapping, list stale docs or
missing scripts, and produce a correction table + a `QUANTULE_MAPPER_MATH_IMPLEMENTATION_REGISTRY.md`. Codex should
run **no new science** and should **not** re-author the conceptual interpretation.

## What this bundle is for (the author's four reasons, satisfied)
1. **Translation** — IRER concepts mapped to searchable model families (nonlinear Schrödinger / Ginzburg–Landau
   solitons, Klein–Gordon Q-balls, soliton phase-force laws, conformal/analogue-gravity scalars).
2. **Context reset** — future agents start from the mature project state instead of re-opening settled baselines
   (e.g. "does prime-SSE predict stability?" — no; "can conservative nodes move?" — yes, corrected).
3. **Active/falsified/open split** — the theory becomes a tested-hypothesis map, not a flat list of ideas.
4. **Maths sanity check** — each active script is traced to its actual equation, observable, and validation gate (a
   Phase-A-style check for the modern multi-script engine).
