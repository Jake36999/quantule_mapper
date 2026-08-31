# TG Consolidation & Validation Framework

Author: Claude (primary), 2026-07-16. Companion to `TG_SEMANTIC_BASELINE_AND_POSTULATE_AUDIT.md`.
Purpose: (1) lock down the proven parts, (2) define the phase plan (robustness → characterization → basins →
foundations) with its validation metrics, (3) set the organization pattern for the loose scripts/docs and the
statistical run corpus. This is the *start and the pattern* — not a completed reorganization.

---

## 1. Lockdown: protected-proven components

**Rule (mirrors the production-side protection):** files below are FROZEN-PROVEN. No edits without a queue row +
review; new science goes in NEW files that import them. Integrity is enforced by `TG_PROTECTED_REGISTRY.json`
(SHA-256 of each file at freeze time) — any change is detectable, and CI/audit can diff against it.

| tier | component | why protected |
|---|---|---|
| T0 | `jax_scout/phase_d_c3_wave.py` | validated Q-ball machinery (whole C3 + TG chain rests on it) |
| T0 | `jax_scout/gravity_TG_B1S_state_load_feedback_gpu.py` | the frozen A-hill scaffold + the shared dynamics/helpers every TG stage imports; `S0_AUTHORITATIVE` lives here |
| T1 | `gravity_TG_B1S_{backreaction_robustness, long_time, drift_decomposition, D4_rows, D4_D5_closure}_gpu.py` | the D-series evidence chain endpoints |
| T1 | `gravity_TG_B1S_frequency_contract.py`, `gravity_TG_B1S_box_dependence.py` | banked mechanism contracts (FC-1, box-dependence) |
| T1 | `gravity_TG_B2_{two_node_awell, static_force, dynamical_force, definitive_force, overnight_dynamical}.py` | the TG-B2 evidence chain incl. the definitive instrument |
| T0 | conventions: `S0 = 135.6862187684289`; single-node B1S source normalization; A-well = `exp(+ε_G G)` on the TG-B2 branch; Δφ=0 pair placement | semantic commitments — changing any of these forks a new hypothesis branch by definition |

**Runtime schedule/policy** (the "particular runtime schedule" this maturity level needs):
- **Lanes:** Colab A100 = campaigns/characterization (measured ~8× the 1080); local 1080 = scouts, validation,
  single overnight campaigns; CPU = contracts/post-processing.
- **Scheduler:** `docs/RUN_QUEUE.md` remains the single dispatch surface; every run gets a row before launch.
- **Refactor ordering:** no consolidation of the frozen modules into the planned `gravity_runtime/` package until
  the characterization phase closes (Codex's hardening-plan rule: never refactor a frozen model mid-validation).

## 2. Phase plan and validation metrics

Phases run in order; each has preregistered metrics and a documented exit. Colab-heavy items marked ◆.

### Phase R — Robustness (is the TG-B2 result solid?)
| test | metric | pass condition |
|---|---|---|
| R1 grid/dt/box convergence ◆ | `<F_R_well>` under N∈{64,96}, dt/2, L∈{16,20} | sign unchanged; magnitude drift < preregistered tol (suggest 20%) |
| R2 ε_G scaling | `<F_R_well>(ε_G)` for ε_G∈{0.03,0.06,0.12} | linear through origin (R²>0.99); antisymmetry holds |
| R3 normalization consistency | static predictor rerun with single-node norm | static↔dynamic ratio stable across seps |
| R4 dealiasing (ALIAS-1) | one masked row | Δω + `<F_R>` unchanged within tol |
| R5 placement/phase robustness | Δφ∈{0, π}, placement offsets | attraction direction invariant (magnitude may vary) |

### Phase C — Characterization (what IS the force?)
| test | metric |
|---|---|
| C1 F(sep) map + screening fit ◆ | `<F_R>(sep)` for sep∈{2.5…8}; fitted range vs coupled-mode prediction (~0.8) |
| C2 mass/amplitude scaling ◆ | `<F_R>` vs node charge/amplitude (ω-family sweep) — does the force scale with node "mass"? (the gravity-relevant scaling) |
| C3 T/G profile contract (FC-2) | dynamical steady T/G vs analytic screened solve |
| C4 ledger completion (LEDGER-1) | booked non-variational work channels → residual at solver floor |
| C5 midplane stress-flux cross-check | independent force observable agrees with body force |

### Phase B — Stability basins (what other configurations work?)
| test | metric |
|---|---|
| B1 coupling-space sweep ◆ | (κ, ω_T, ω_G, ε_G, γ_T, γ_G) grid → attractor classification per the theory taxonomy (STABLE_NONRADIATING … RUNAWAY_COLLAPSE); map the bounded-attraction basin; verify the κ²<ω_T²ω_G² boundary empirically |
| B2 two-node bound states ◆ | can the loop *bind* a pair (loop force vs bare force balance)? capture/escape boundary |
| B3 multi-node scouts | 3-node configurations; superposition/additivity of the mediated force |

### Phase F — Foundations completion (from the postulate audit; separate hypothesis branches)
| item | content |
|---|---|
| F1 R_coh source | operationalize the coherence-transition resolution source `[−d_t K_phase]_+`; TG-S-style semantics gate first |
| F2 N_t wiring | dual-coefficient model (`∂_tφ = N_tΠ` + A_s) as a NEW branch; never assume N_t=A_s |
| F3 χ_out channel | named radiation/relief diagnostic + attractor-class taxonomy runs |
| F4 VAR-1 / LR-1 | variational action; long-range ω_G→0 fork (design-first, already queued) |

### Statistical cross-analysis (enabled by the corpus)
The corpus is ~33 TG run dirs + C-series + Gravity-D. Start: a machine-readable **run index**
(`docs/gravity_maturity/TG_RUN_INDEX.csv`) with one row per run: `run_dir, stage, script, config_hash, N, L, dt, T,
seps, key_observable, value, verdict, gates, superseded_by`. Populate incrementally (each new run appends; backfill
the TG corpus first). This is what turns the pile of runs into a dataset: cross-run regressions (e.g. Δω and `<F_R>`
vs parameters), consistency checks between stages, and later basin maps all read from it.

## 3. Organization pattern (the start — apply going forward, backfill opportunistically)

- **One index:** `docs/gravity_maturity/` gets a short `INDEX.md` mapping stage → scripts → runs → results doc →
  verdict (the human-readable twin of the run index CSV). Every new doc/script adds its line at creation time.
- **Naming:** scripts `gravity_TG_<stage>_<topic>.py` (existing pattern, now official); results docs
  `TG_<stage>_<topic>_RESULTS.md`; every run dir gets `ROW/RUN` markers + `summary.json` (already standard);
  superseded artifacts get `SUPERSEDED.json` (already standard).
- **Verdict discipline:** machine labels in `summary.json`; the *endorsed* verdict lives only in the results doc
  after primary review (auto-labels are never authoritative — standing rule).
- **Doc layering (unchanged):** catalog → theory synthesis → evidence package → gravity_maturity working docs;
  promotions out of gravity_maturity go through the master catalog only after phase exits.

## 4. Immediate ordering

1. **Freeze now:** write `TG_PROTECTED_REGISTRY.json` (done alongside this doc) — the lockdown is active from today.
2. **Phase R first** (R1+R2 are one Colab capsule ◆; R3 is CPU; R4/R5 cheap) — robustness before any new physics.
3. **Then Phase C**, then **Phase B**. Phase F items are parallel *design* work (no runs) until selected.
4. The run queue carries one row per phase item; the index files grow as rows complete.

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `25020bf` (2026-07-22) — *dual substrate*
**Revised since:** 7 commit(s), most recently `3877db6` (2026-08-31)

**Harness code changed since it was written:** 5 commit(s) to `jax_scout/`.
  - `06edae5` 2026-08-27 — Fix run provenance: 30% -> 86% of runs now dated
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine
  - `25020bf` 2026-07-22 — dual substrate

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Also referenced by (same date or earlier):** [[RUN_QUEUE]], [[gravity_maturity/CONCEPT_IMPLEMENTATION_AND_GAP_ANALYSIS_CODEX_CONTINUATION_20260717]], [[gravity_maturity/TG_RECOVERED_CONCEPTS_INTEGRATION_AND_REPRIORITIZATION]]

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
