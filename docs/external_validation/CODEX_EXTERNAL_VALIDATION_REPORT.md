# Codex External Validation Report

Final decision: `SOURCE_REGISTRY_UPDATED_WITH_SECOND_TIER_CANDIDATES`

Status: provisional until Claude review.

This report summarizes a documentation-only source audit and first-pass metric-preparation pass for the "External Legibility and Quantitative Validation" phase described in `docs/FUTURE_WORK_AND_EXTERNAL_VALIDATION_PLAN.md`.

No simulations were run. No solver physics, Hunter objectives, validation gates, production defaults, or project verdicts were changed. No claim is made that IRER proves matter, gravity, unification, or external physical correspondence.

## 2026-07-14 Gravity D Spatial Effective-Medium Package

A separate third-party rerun package has been added for the standalone Gravity D spatial effective-medium mechanism:

- `docs/external_validation/GRAVITY_D_EFFECTIVE_MEDIUM_EXTERNAL_VALIDATION_PACKAGE.md`
- `docs/external_validation/artifact_bundles/gravity_d_effective_medium_characterization.zip`
- `docs/external_validation/artifact_bundles/gravity_d_effective_medium_characterization_zip_hash.csv`
- `sweep_runs/GRAVITY_D_DYNAMICS_CHARACTERIZATION_GPU_20260713_221727/`

This package is limited to the characterized spatial operator:

```text
i d_t psi = -D div(N(x) grad psi)
d<P>/dt = -D integral grad(N) |grad psi|^2 dV
F_cg ~= -D K_grad grad N(R)
```

It includes fixed configs, GPU environment versions, run manifests, primary CSVs, artifact hashes, the exact force
identity, the reduced-model feature library, holdout partition, sign-error analysis, and negative tests. It must not
be cited as gravity, geodesic, temporal-lapse, equivalence-principle, IRER-source, or production validation.

## Files Produced

- `docs/external_validation/EXTERNAL_SOURCE_REGISTRY.md`
- `docs/external_validation/VALIDATION_METRIC_CANDIDATES.md`
- `docs/external_validation/DATASET_CANDIDATES.md`
- `docs/external_validation/CODEX_EXTERNAL_VALIDATION_REPORT.md`

## Source Registry Summary

The registry currently contains:

| Category | Count | Notes |
|---|---:|---|
| Verified primary/review sources | 23 | Includes Gordon, Karpman-Solov'ev, Coleman, Friedberg-Lee-Sirlin, Lee-Pang, analogue gravity, Madelung, Takabayasi, CGLE/dissipative soliton sources, and dataset papers. |
| Secondary / metadata-only placeholders | 3 | NLS Galilean and VK criterion entries need stronger publication/textbook sources before publication-facing use. |
| Explicit TO_VERIFY relevance item | 1 | Salih 2026 exists, but relevance/weight is not assessed and must be reviewed by Claude. |
| Dataset candidates | 8+ | Most are future candidates; NIST dark-soliton dataset is open but not a direct bright-collision/Q-ball metric dataset. |

## Analytic Formula Coverage

| Topic | Status | Extracted comparison hook | Match level |
|---|---|---|---|
| Karpman-Solov'ev / Gordon two-soliton force law | Source-backed, constants need normalization | `q_ddot = -C exp(-lambda q) cos(Delta_phi)`; phase crossover near `pi/2` | SAME FAMILY |
| NLS Galilean boost | Formula prepared; primary source should be hardened | `v = 2Dk` from `omega = D k^2` / Galilean invariance | EXACT EQUATION |
| Cubic-quintic / cubic-quintic-septic collision/capture | Source-backed as family, no universal closed-form capture law | outcome grid, mass/current retention, radiation fraction | CLOSE ANALOGUE |
| Complex Ginzburg-Landau dissipative solitons | Source-backed as family | gain/loss-balanced localized states; CGLE/CQ-CGLE forms | SAME FAMILY |
| Klein-Gordon / Q-ball / VK stability | Source-backed, VK primary source should be hardened | `Phi=phi(r) exp(i omega t)`, `Q`, `E/Q`, `dQ/domega` | SAME FAMILY |
| Analogue-gravity acoustic metric | Source-backed review formula | acoustic metric density/sound-speed dependence; polytropic exponent mapping | CLOSE ANALOGUE |
| Madelung / Bohm / quantum-hydrodynamic stress | Source-backed | continuity, Bohm quantum potential, stress/pressure tensor candidates | CLOSE ANALOGUE / SPECULATIVE |

## Salih 2026 Verification

The "Salih 2026" citation was found and verified as an existing Frontiers in Physics article:

- Mahgoub A. Salih, "A scalar-field model with information-theoretic interpretation: solitonic matter, gauge symmetry breaking, and dark matter candidates", Frontiers in Physics 14, 2026.
- DOI: `10.3389/fphy.2026.1806936`
- Registry status: `VERIFIED_SOURCE / TO_VERIFY_RELEVANCE`
- Match level: `SPECULATIVE`

This source must not be used as a foundation for IRER framing until Claude reviews its relevance, quality, and relationship to established Q-ball literature.

## Dataset Readiness Summary

| Dataset bucket | Best candidates found | Current readiness |
|---|---|---|
| Optical soliton molecule collisions | DFT soliton-molecule collision papers | FUTURE_CANDIDATE; raw data not confirmed. |
| BEC bright soliton phase-dependent collisions | Controlled-relative-phase bright matter-wave collision papers | FUTURE_CANDIDATE; likely paper-digitization or author request. |
| Dissipative soliton datasets | Fiber-laser real-time dissipative soliton papers | FUTURE_CANDIDATE; raw sequences not confirmed. |
| Q-ball / oscillon numerical benchmarks | Q-ball dynamics and oscillon/Q-ball papers | FUTURE_CANDIDATE; raw numerical data/code not located. |
| Analogue-gravity BEC/acoustic metric experiments | Analogue-gravity review trail | FUTURE_CANDIDATE; requires selecting a concrete experiment. |
| Open soliton image dataset | NIST dark solitons in BECs | YES_NOW for image/morphology tasks only; not a direct bright-collision or Q-ball comparison. |

## Second-Tier Candidates Added

| Candidate | Verification status | Match level | Readiness | Supports |
|---|---|---|---|---|
| Mitschke & Mollenauer, "Experimental observation of interaction forces between solitons in optical fibers" | `VERIFIED_METADATA_ONLY`; full observable extraction still needed | SAME FAMILY | FUTURE_CANDIDATE / V1 priority source | V1 phase-force law experimental companion |
| Nguyen et al., "Collisions of matter-wave solitons" | `VERIFIED_SOURCE` | CLOSE ANALOGUE | FUTURE_CANDIDATE unless raw/tabulated data are found | V5 phase-dependent collision behavior |
| JOSS `NLSE` package | `VERIFIED_SOURCE` | EXACT EQUATION only for reduced flat-NLS tests | YES_NOW for numerical-method sanity only | V2 reduced-NLS identity checks; not empirical validation |
| Fontaine et al., "Observation of the Bogoliubov Dispersion in a Fluid of Light" | `VERIFIED_SOURCE` | CLOSE ANALOGUE | FUTURE_CANDIDATE | V7-style analogue-fluid density/dispersion context |
| Hulet Lab matter-wave soliton programme | `VERIFIED_SOURCE` as a source trail | CLOSE ANALOGUE / SOURCE_TRAIL | FUTURE_CANDIDATE | BEC bright-soliton dynamics source trail; avoid duplicate use where Nguyen/Rice entries already cover the data |

Lower-priority notes added:

- Optical dark soliton datasets are contrast classes, not direct bright-soliton benchmarks.
- Photon-fluid nonlocality experiments are future analogue-hydrodynamic candidates.
- Q-ball/oscillon repositories remain an active search target; no machine-readable benchmark repository was verified in this update.

## Gaps and Required Follow-Up

1. Replace the secondary NLS Galilean source with a primary textbook or canonical NLS reference before publication-facing use.
2. Add a primary VK citation or a modern rigorous stability reference for the exact `dQ/domega` convention.
3. Decide whether paper-digitized collision curves are acceptable for V1/V5, or whether raw datasets are required.
4. Treat all Salih 2026 uses as `TO_VERIFY_RELEVANCE` until Claude reviews it.
5. Do not run external quantitative fits until Claude finalizes which metrics and data sources are allowed.

## Guardrail Attestation

- No IRER reinterpretation.
- No verdict changes.
- No cautious-language framing changes.
- No new science simulations.
- No solver physics changes.
- No Hunter, validation, config, or production-default changes.
- No physical-correspondence claim.
- No matter, gravity, or unification claim.

## Recommended Claude Review Questions

1. Which V1 constants, if any, should be treated as comparison targets for nonintegrable cubic-quintic-septic solitons?
2. Should the V1/V5 external comparison allow digitized paper figures, or only machine-readable/source-provided data?
3. Is Salih 2026 worth retaining as a speculative/contextual source, or should it be excluded from the foundation entirely?
4. Which analogue-gravity experiment should be chosen if V7 moves beyond formula-level comparison?
5. Which source should become the authoritative publication-facing reference for NLS Galilean invariance and VK stability?

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `25020bf` (2026-07-22) — *dual substrate*
**Revised since:** 4 commit(s), most recently `60093e5` (2026-08-27)

**Harness code changed since it was written:** 4 commit(s) to `jax_scout/`.
  - `06edae5` 2026-08-27 — Fix run provenance: 30% -> 86% of runs now dated
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
