# Quantule Mapper — Architecture & Design-Pattern Rationale

**Report 4 of 5.** Why the project is built the way it is now — specifically, why it moved from a single monolithic
"adaptive engine" toward a set of sector-specific physics harnesses with separated search, validation, and
interpretation layers. This exists so future contributors (human or agent) do not "improve" the system by folding
everything back into one adaptive hunter, and do not re-litigate baselines that are already settled.

## 0. The shift in one line
> **Old model:** a monolithic adaptive engine that hunts for "interesting" outcomes.
> **New model:** sector-specific, well-posed physics harnesses + invariant/parity gates + a hypothesis catalog +
> reproducible smoke modes + a Hunter that *searches* but does not *interpret*.

## 1. Layer separation (and why each boundary exists)
| layer | responsibility | why separate |
|---|---|---|
| **Solver / physics** (`jax_scout/physics.py`, `phase_d_*`, `phase_d_c3_wave.py`) | integrate a named PDE; nothing else | a solver that also scores/steers can silently bias results; the hot loop must be pure dynamics |
| **Observables / diagnostics** (`metrics/*`, momentum-density fns) | read-only measurement from saved/live fields | measurement must not perturb dynamics; observables can be swapped/hardened (peak-track → momentum-density) without touching physics |
| **Validation / provenance** (`validation_pipeline.py`, `orchestrator/`) | conservation/parity gates, gravity/collapse telemetry, config hashes | a result is only trustworthy with its gate + provenance; keeping this outside the solver means gates can be added without recompiling physics |
| **Search / Hunter** (`aste_hunter.py`) | explore parameter space toward a *validated* objective | search must optimize a *pre-validated* observable, never define truth; a hunter that owns interpretation optimizes for its own artifacts |
| **Interpretation / docs** (`docs/*`, this bundle, the catalog) | assign verdicts, preserve nulls/retractions | the scientific claim lives in reviewed docs, not in a score a hunter emitted |

## 2. Why the monolithic adaptive engine was retired
- **It conflated dynamics, scoring, and search.** When one loop evolves *and* decides what is interesting, a
  measurement artifact (e.g. a fragile peak-tracker, a mis-set geometry flag) becomes an optimization target. Three
  of this project's bugs (C2.6 geometry-off, C2.8b tracker, C3 boost-IC) are exactly the class of error a monolith
  would have *amplified* by hunting toward them.
- **It optimized a proxy (prime-SSE) that turned out to be null.** The stability sector's biggest early lesson —
  prime-SSE never predicted stability (0/60) — was only cleanly establishable once the objective was decoupled and
  re-aimed at a validated stability metric (H7 re-discovery).
- **It resisted per-sector reasoning.** The transport question needed *different substrates* (dissipative GL,
  conservative NLS, geometry-on, true-flat, KG) with substrate-specific invariants. A monolith wants one code path;
  the physics wanted five clearly-separated ones sharing only the nonlinearity and spectral core.

## 3. Design patterns now in force (with rationale)
1. **Mirror-first (jax_scout) scouting.** Exploratory physics runs on the JAX mirror; production CuPy is changed only
   after the mirror validates. *Why:* fast iteration without risking the production/reference solver; the frozen
   Phase C operator stays byte-identical (parity re-checked on every kinetic-term addition: C1, C2, geom_fac all
   default to bit-exact baseline).
2. **CuPy/JAX parity gates.** The two engines share the identical operator; parity is proven (rel-L2 1.7e-12).
   *Why:* results must be engine-independent; a divergence is a bug signal, not "just FP".
3. **Validation metrics out of the hot loop.** Conservation/flux/adjoint checks are computed in a separate pass.
   *Why:* they must not alter dynamics, and they must be improvable independently (the momentum-density observable
   replaced peak-tracking with no solver change).
4. **Explicit geometry-on/off gating.** `param_geom_off` / `geom_fac` make "flat vs geometry" an explicit, verified
   switch. *Why:* the C2.6 bug was precisely an *implicit*, unverified "off" (`a_coupling=0`) that wasn't off.
   Every geometry claim now runs geometry-on **and** true-null.
5. **Provenance everywhere.** Runs carry command, seeds, git commit, dt/N, and conservation telemetry; harnesses
   checkpoint per-case and save the object field (crash-resilient, resumable via `--object-npy`). *Why:*
   reproducibility, and recovery when a background run dies at a session boundary.
6. **Correct-boost-IC + robust-observable + conservation-telemetry as a standing gate.** Every transport/interaction
   claim must clear all three. *Why:* it is the direct antidote to the three bugs; a null that contradicts a symmetry
   identity is treated as an instrument fault until proven physical.
7. **Short official smoke modes** (`--quick` on C2.7/C2.9). *Why:* replication (and Codex) must not depend on long
   N=96 holds or time out; a bounded smoke reproduces the qualitative result in minutes.
8. **Output hygiene.** Regenerable render/diagnostic trees (`quantule_viz/outputs/`) are gitignored; unique handover
   Markdown is promoted to `docs/`. *Why:* keep the repo a source-of-truth, not a 180 MB artifact dump, without
   losing unique analysis.
9. **Nulls and retractions are first-class.** The catalog and result docs preserve falsified and retracted
   hypotheses. *Why:* they are the map's negative space; deleting them invites re-derivation of wrong conclusions.

## 4. Role boundaries for agents (explicit, to prevent regression)
- **Hunter:** searches parameter space toward a *validated* objective (e.g. the stability metric). It does not decide
  what a result *means*.
- **Codex (secondary execution/audit assistant):** runs long/synchronous campaigns, replicates finalized harnesses,
  and audits equations/tracebacks/paths. It does **not** design the scientific measurement or own frontier
  interpretation (per the author's directive; e.g. the two-node observable was designed by Claude before any
  delegation, and rung-B design + gravity interpretation are retained by Claude).
- **Claude (frontier interpretation):** designs measurements, assigns verdicts, writes theory synthesis, and gates
  claims against identities.
- **Author (Jake McIntosh):** owns the theory, the conceptual authority, and the scope decisions (which sector, which
  hypothesis, when to pause).

## 5. Why this matters for the next phase
The project is no longer "a simulation script." It is a **controlled numerical test bench** with separated physics,
search, validation, telemetry, and interpretation. That separation is what let three instrument bugs be found and
corrected, what makes the results reproducible and engine-independent, and what will let external comparison proceed
(Report 1's translations + Report 5's status) without an agent first "re-verifying the basics" or re-merging the
search and physics layers.

## Cross-references
Concepts: Report 1. Equation↔code: Report 2. Identities/proofs (incl. the bugs these patterns caught): Report 3.
Status: Report 5. Catalog: `docs/IRER_MASTER_HYPOTHESIS_CATALOG.md`. Codex handoffs & audits:
`docs/PHASE_D_C2_6_CODEX_AUDIT_HANDOVER.md`, `docs/GRAVITY_LADDER_CODEX_HANDOFF.md`,
`docs/codex_conservative_c2_campaign_archive/`.
