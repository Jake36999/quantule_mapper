# Empirical Comparison Branch (QUARANTINED)

**Status:** BANKED / complete-for-now (Jake, 2026-07-12). Two qualitative `CLOSE ANALOGUE` overlays reviewed and
banked (EMP-V1-MM, EMP-V5-NG). The branch remains active and re-openable; one follow-up is parked (see below). No
further digitization queued.
**Match-level ceiling for everything in this directory:** `CLOSE ANALOGUE` (never `EXACT`, never `SAME FAMILY`).
**Consumed by the run/validation pathway or the Hunter:** **NO — and never automatically.**

---

## Banked results (complete-for-now)

| id | source | metric | verdict | review |
|---|---|---|---|---|
| EMP-V1-MM | Mitschke & Mollenauer 1987 | V1 sign law | weak-interaction branch consistent with π/2 sign law (attraction clear; anti-phase at noise floor); no quantitative λ | `EMP-V1-MM_CLAUDE_REVIEW.md` |
| EMP-V5-NG | Nguyen et al. 2014 | V5 collision | anti-phase transmission direction matches C3, **mechanistically** (interference-node protection); coarse (no off-phase/speed) | `EMP-V5-NG_CLAUDE_REVIEW.md` |

Both are qualitative external corroborations of the *structure* of our results — they do not establish physical
correspondence and change no verdict. Neither can go quantitative without data that likely does not exist.

## Parked follow-up (not queued; re-open only if the data surfaces)

- **EMP-V5-NG-VEL — velocity-resolved collision outcomes.** The only realistic path to externally testing the C3
  **speed dependence** (anti-phase transmits only below ~0.5c). Would require collision-velocity-resolved survival
  data — check Nguyen 2014 supplementary movies or a follow-up/related matter-wave-soliton-collision paper. If such
  a dataset is found, extend the V5 schema with `speed_value`/`speed_units` (the comparator already ingests speed
  and reports a `phase_and_speed` coverage scope) and digitize per the standard flow. Until then: **parked, not a
  task.** The anti-phase *narrowness* (off-phase captures) is a separate gap that experiments essentially never
  produce — treated as a known limit, not a follow-up.

---

## What this is

A **separate, opt-in branch** of the External Legibility & Quantitative Validation phase for **Level-3 empirical
comparison**: comparing Quantule Mapper behaviour against **digitized experimental data** from published papers
(e.g. soliton-interaction force curves, phase-dependent BEC collision outcomes).

It is deliberately **quarantined** from:
- the internal-analytic validation harness (`tools/external_validation/`, metrics V1–V7),
- the method-sanity external run (`external_data/nlse_reduced_v2/`),
- the production solver, the Hunter objective, and the production validation pipeline.

This is *not* a rung of the main architecture. It is a side channel whose outputs are **context only** until a
human explicitly promotes them.

## Why a separate branch (the concern this addresses)

Digitized paper data is fundamentally lower-confidence than our own runs: it carries extraction uncertainty, is
selected from figures the original authors chose to publish, and describes different physical systems under an
*analogue* mapping. If it leaked into the Hunter's search conditions or the production validation gates, it could
silently bias what the model is optimized toward or how it is judged — a subtle, hard-to-notice error. So it is
kept physically and procedurally apart.

## Hard isolation rules (binding)

1. **No file in this directory is referenced by** `tools/external_validation/validation_config.yaml`, the
   `external_data.py` manifest, any Hunter config/objective, or the production validation pipeline. Verified at
   creation: the harness reads only explicit hardcoded `sweep_runs/` paths and a hardcoded manifest list — nothing
   here is auto-discovered.
2. **The Hunter MUST NOT search for, optimize toward, or gate on any condition, target, or dataset defined here**
   unless a human makes a deliberate, reviewed change that explicitly wires it in. Absence of such a wiring is the
   default and correct state.
3. **Digitized empirical data never promotes a verdict, never becomes a pass/fail gate, and never changes IRER
   framing or the maturity timeline.** A match is a *faithfulness-context* observation, not evidence of physical
   correspondence, matter, gravity, or unification. (Preservation rules apply — see
   `../IRER_PRESERVATION_RULES_FOR_EXTERNAL_VALIDATION.md`.)
4. **Match-level is capped at `CLOSE ANALOGUE`.** Digitized data may never be labelled `EXACT` or `SAME FAMILY`,
   regardless of how well it fits.
5. **Every dataset requires a completed provenance record** (`DIGITIZATION_PROVENANCE_TEMPLATE.md`) before any
   comparison is run against it. No provenance → not usable.
6. **All results are `PROVISIONAL_UNTIL_CLAUDE_REVIEW`** and must state the digitization uncertainty.

## What lives here vs. off-git

- **Here (git, small):** this README, provenance records, digitized comparison *tables* (small CSV/JSON), and
  comparison reports.
- **Off-git (E: drive, like other raw external data):** raw downloaded papers, figure images, and any large
  extracted arrays. See `datasets/README.md`.

## How it connects back (opt-in only)

If/when a dataset is digitized and provenance-recorded, a Level-3 comparison may be run against the **existing V1
(force law) or V5 (collision phase diagram) results** — but only via an explicit, separately-invoked comparison,
clearly labelled Level-3 / `CLOSE ANALOGUE`, and reviewed by Claude before any conclusion is drawn. It does **not**
re-enter the `run_external_validation.py --metric all` pathway.

## Directory layout

```
empirical_comparison_branch/
  README.md                          # this charter
  TARGETS.md                         # candidate datasets + status
  DATA_SCHEMA.md                     # V1 CSV schema (comparator + digitization share)
  EMP_V5_NG_DATA_SCHEMA.md           # V5 collision phase×outcome CSV schema
  DIGITIZATION_PROVENANCE_TEMPLATE.md# one filled copy required per dataset
  CODEX_DIGITIZATION_HANDOFF.md      # bounded task: EMP-V1-MM (done)
  CODEX_DIGITIZATION_HANDOFF_V5_NG.md# bounded task: EMP-V5-NG (Nguyen collisions)
  EMP-V1-MM_CLAUDE_REVIEW.md         # V1 review verdict
  datasets/README.md                 # where raw data lands (off-git) + naming

tools/external_validation/empirical_comparison/   # the quarantined comparator (opt-in entry point)
  run_empirical_comparison.py        # --self-test | --metric {v1,v5} --data ... --provenance ...
  README.md
```

## Digitization decision — RESOLVED

Jake approved digitized paper figures (2026-07-11), with provenance recorded and match-level capped at
`CLOSE ANALOGUE`. The branch is active. Next step is Codex figure digitization per
`CODEX_DIGITIZATION_HANDOFF.md`; the comparator (`tools/external_validation/empirical_comparison/`) is built and
self-tested and consumes datasets that match `DATA_SCHEMA.md`.
