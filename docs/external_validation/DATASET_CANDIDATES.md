# Dataset Candidates

Status: provisional Codex dataset discovery for Claude review. These are candidates for future comparison only. No dataset entry implies that IRER matches an experiment or external numerical result.

Availability labels: `OPEN_DATA`, `PAPER_DATA_ONLY`, `PAYWALLED`, `REQUEST_AUTHORS`, `UNKNOWN`.

Quantitative-readiness labels:
- `YES_NOW`: machine-readable data or sufficiently tabulated numeric data are available now.
- `FUTURE_CANDIDATE`: likely needs digitization, author request, measurement-design work, or a more precise model mapping.

| ID | Category | Source / dataset | Availability | Format | Comparison quantity | Units / scaling needed | Can support quantitative comparison now? | Future-only? | Caveats |
|---|---|---|---|---|---|---|---|---|---|
| DATA-BEC-001 | BEC soliton images | NIST Dark solitons in BECs dataset | OPEN_DATA | Image dataset with labels; NIST repository page | Soliton morphology, classifier validation, density-image processing | Imaging units, trap geometry, dark-soliton vs bright-soliton mapping | YES_NOW for image/morphology tasks only | No for C2 bright collision metrics | Dark solitons are density depletions, not project bright node/soliton collisions. Useful as data-engineering analogue, not direct physics benchmark. |
| DATA-BEC-002 | BEC dark-soliton paper | Dark solitons in Bose-Einstein condensates: a dataset for many-body physics research | OPEN_DATA | Paper plus linked dataset | Labeled solitonic excitations in BEC images | Same as DATA-BEC-001 | YES_NOW for image tasks | Future candidate for dynamics | Strong open dataset, but not phase-dependent bright-soliton collision telemetry. |
| DATA-BEC-003 | BEC bright-soliton collisions | Realizing bright-matter-wave-soliton collisions with controlled relative phase | PAPER_DATA_ONLY | Journal paper; figures/tables likely require extraction | Phase-dependent stability/bounce/pass at low velocity | Trap units, scattering length, 1D/3D reduction, velocity normalization | No | Yes | Strong conceptual candidate for phase-dependent soliton collision, but raw data were not located in this pass. |
| DATA-BEC-004 | BEC matter-wave collisions | Collisions of matter-wave solitons | PAPER_DATA_ONLY | PDF/paper | Relative-phase-dependent collision outcomes; real-time imaging | Experimental length/time units and phase preparation need mapping | No | Yes | Candidate for qualitative phase-law comparison; quantitative use likely requires digitization or author data. |
| DATA-OPT-001 | Optical soliton molecule dynamics | Analysis of dispersive Fourier transform dataset using dynamic mode decomposition | UNKNOWN | Paper; DFT dataset referenced | Molecule vibration modes, spectral-temporal trajectories | Optical cavity round trips, spectral-to-time conversion, pulse separation scaling | No | Yes | Dataset exists in the paper context, but open raw data were not confirmed. |
| DATA-OPT-002 | Optical soliton molecule collisions | Real-Time Access to Collisions between a Two-Soliton Molecule and a Singlet | PAPER_DATA_ONLY | Article figures; possible supplementary material not verified | Collision scenario, bond exchange, quasi-elastic event labels | Fiber-laser propagation and DFT scaling | No | Yes | Good phenomenology candidate; not an exact equation match. Raw data availability not confirmed. |
| DATA-OPT-003 | Dissipative optical soliton molecules | Real-time observation of dissipative optical soliton molecular motions | PAPER_DATA_ONLY | Article/arXiv; DFT measurements | Internal motion, phase drift, vibration categories | Fiber cavity round-trip units and spectral mapping | No | Yes | Useful for dissipative soliton molecule analogy; raw numerical time series not confirmed. |
| DATA-OPT-004 | Dissipative soliton turn-on | Real-time measurements of dissipative solitons in a mode-locked fiber laser | PAPER_DATA_ONLY | arXiv/paper; real-time spectral and temporal measurements | Formation dynamics, collision/break-up before stable mode locking | Optical time-lens/DFT scaling | No | Yes | Good external-data lead for Phase C-style dissipative attractors, but quantitative comparison needs raw sequences. |
| DATA-OPT-005 | Microresonator-filtered fiber laser | Dissipative soliton generation and real-time dynamics in microresonator-filtered fiber lasers | PAPER_DATA_ONLY | Paper/arXiv | Soliton formation and interaction dynamics, Newton's-cradle-like events | Microcomb/fiber-cavity parameters | No | Yes | Future candidate for dissipative interaction patterns, not direct equation match. |
| DATA-QB-001 | Q-ball / oscillon numerics | Dynamics of nontopological solitons: Q balls | PAPER_DATA_ONLY | Journal paper; numerical simulations | Q-ball stability and interactions | Model potential, charge normalization, dimensionality | No | Yes | Good benchmark-source candidate; raw numerical data not located. |
| DATA-QB-002 | Q-ball / oscillon numerics | Oscillons and bubbles in Q-ball dynamics | PAPER_DATA_ONLY | arXiv/paper | Q-ball/anti-Q-ball collision outcomes and intermediate states | Potential and dimension mapping required | No | Yes | Future candidate for C3 collision/capture language; not ready for direct metric computation. |
| DATA-QB-003 | Q-ball / oscillon theory | Oscillons from Q-balls in generalized models | PAPER_DATA_ONLY | arXiv/paper | Relation between oscillon and Q-ball sectors | Model differs; no project-unit mapping yet | No | Yes | Use as literature source more than dataset unless code/data are found. |
| DATA-AG-001 | Analogue gravity BEC/acoustic | Analogue Gravity review corpus | PAPER_DATA_ONLY | Review papers; experiment references | Acoustic metric density/sound-speed dependence | Requires selecting a concrete experiment and EOS | No | Yes | Good source path for formula extraction; not a dataset by itself. |
| DATA-AG-002 | Analogue gravity BEC/acoustic | Acoustic analogue / BEC horizon experiments from analogue-gravity literature | UNKNOWN | Experiment papers; raw data not located | Metric response, propagation/horizon observables | Fluid density, velocity field, sound speed, EOS | No | Yes | Needs a narrowed target experiment before dataset comparison. |
| DATA-CGLE-001 | Dissipative soliton simulation | CGLE dissipative soliton papers and reviews | PAPER_DATA_ONLY | Figures/tables, sometimes simulation parameter sets | Existence windows, pulse classes, exploding/creeping soliton behavior | CGLE coefficient normalization and propagation variable | No | Yes | Good literature benchmark; quantitative comparison needs source code/data or digitization. |
| DATA-OPT-006 | Optical soliton-force experiment | Experimental observation of interaction forces between solitons in optical fibers | PAPER_DATA_ONLY | Optics Letters article; input/output separation observables | V1 experimental companion: attractive/repulsive soliton-force sign and separation trend | Fiber propagation distance, pulse separation, amplitude, and phase conventions | No | Yes | Highest-priority missing V1 citation; raw data not verified. Use after full observable extraction. |
| DATA-BEC-005 | BEC bright-soliton collision | Collisions of matter-wave solitons | PAPER_DATA_ONLY | arXiv/Nature Physics paper; Rice repository metadata | Phase-dependent collision behavior, trajectory jump, collapse threshold context | BEC trap, scattering length, atom number, velocity, and phase conventions | No | Yes | Useful for phase-dependent collision behavior, not exact C2/C3 equation matching. |
| DATA-NUM-001 | External reduced-NLS solver | NLSE: A Python package to solve the nonlinear Schrodinger equation | OPEN_DATA | Published open-source numerical package | Reduced flat-NLS boost/propagation sanity checks | Must intentionally reduce project equation to flat NLS; package units/API need mapping | YES_NOW for method sanity only | No for empirical validation | Numerical-method comparator only; do not treat as external physical validation. |
| DATA-AG-003 | Photon-fluid dispersion experiment | Observation of the Bogoliubov Dispersion in a Fluid of Light | PAPER_DATA_ONLY | PRL/arXiv paper; group-velocity/dispersion measurements | Future density-dependent dispersion or sound-speed scaling comparison | Photon-fluid density, nonlinear index, paraxial time mapping | No | Yes | V7-adjacent analogue-fluid candidate, not a current gravity result. |
| DATA-BEC-006 | BEC bright-soliton source trail | Hulet Lab matter-wave soliton programme | UNKNOWN | Research group source trail | Links to BEC bright soliton dynamics and collision papers | Source-trail only; use linked papers/datasets for quantitative work | No | Yes | Do not duplicate Nguyen/Rice entries unless a distinct dataset is found. |
| DATA-OPT-007 | Optical dark soliton contrast class | Optical dark soliton datasets/experiments | PAPER_DATA_ONLY | Papers or image datasets depending on source | Contrast class for dark/notch soliton morphology | Dark soliton density depletion differs from bright localized soliton | No | Yes | Lower priority; contrast class only, not a direct bright-soliton benchmark. |
| DATA-AG-004 | Photon-fluid nonlocality experiments | Experimental characterization of nonlocal photon fluids | PAPER_DATA_ONLY | Optica article; photon-fluid measurements | Future analogue-hydrodynamic comparison | Nonlocal optical response and photon-fluid scaling | No | Yes | Lower-priority V7-adjacent candidate. |
| DATA-QB-004 | Q-ball/oscillon repositories | Machine-readable Q-ball or oscillon benchmark code/data | UNKNOWN | Search target; no repository verified | Q(omega), collision outcomes, oscillon/Q-ball dynamics | Depends on model/potential and dimensionality | No | Yes | Active search target. Keep unverified until a real repository is located. |

## Immediate Readiness Summary

| Bucket | Best current candidate | Readiness | Comment |
|---|---|---|---|
| Optical soliton molecule collisions | DATA-OPT-002 | FUTURE_CANDIDATE | Strong phenomenology; raw data not verified. |
| BEC bright soliton phase-dependent collisions | DATA-BEC-003 / DATA-BEC-004 | FUTURE_CANDIDATE | Good paper candidates; likely digitization or author request needed. |
| BEC bright soliton phase-dependent collisions, second tier | DATA-BEC-005 / DATA-BEC-006 | FUTURE_CANDIDATE | Nguyen/Hulet is a strong source trail; no raw/tabulated data verified. |
| Dissipative soliton datasets | DATA-OPT-004 / DATA-CGLE-001 | FUTURE_CANDIDATE | Good qualitative match; raw sequence access unclear. |
| Q-ball / oscillon numerical benchmarks | DATA-QB-001 / DATA-QB-002 | FUTURE_CANDIDATE | Good benchmark direction; raw data/code not located. |
| Q-ball / oscillon repositories | DATA-QB-004 | FUTURE_CANDIDATE | Active search target only; no repository verified. |
| Analogue-gravity BEC/acoustic metric experiments | DATA-AG-001 / DATA-AG-002 | FUTURE_CANDIDATE | Need a specific experiment and EOS mapping. |
| Photon-fluid analogue dispersion | DATA-AG-003 / DATA-AG-004 | FUTURE_CANDIDATE | V7-adjacent analogue-fluid context; no current gravity result. |
| Public open soliton dataset | DATA-BEC-001 | YES_NOW for image/morphology only | Useful open data, but not a direct bright-collision or Q-ball metric. |
| External reduced-NLS numerical method | DATA-NUM-001 | YES_NOW for method sanity only | Not empirical validation and not a full IRER solver comparison. |

## Recommended Next Dataset Actions

1. For V1/V5, decide whether paper-digitized phase/collision curves are acceptable, or whether only author-provided raw data should count.
2. For BEC bright soliton comparisons, prioritize phase-dependent collision papers before the NIST dark-soliton image dataset.
3. For Q-ball comparisons, search specifically for repositories attached to Q-ball/oscillon papers before attempting any digitization.
4. For analogue gravity, choose one concrete acoustic/BEC experiment before recording dataset-level readiness.
5. Treat optical dark soliton datasets as contrast classes only, unless Claude explicitly asks for a dark-soliton morphology comparison.
6. Use external NLS packages only for reduced-equation numerical sanity checks, never as empirical validation.

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `25020bf` (2026-07-22) — *dual substrate*
**Revised since:** 10 commit(s), most recently `e42b5bb` (2026-09-11)

**Harness code changed since it was written:** 6 commit(s) to `jax_scout/`.
  - `caf61af` 2026-09-10 — Build HUD items 1+2: snapshot writer and universal field renderer
  - `06edae5` 2026-08-27 — Fix run provenance: 30% -> 86% of runs now dated
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine
  - *…and 1 more.*

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Also referenced by (same date or earlier):** [[FUTURE_WORK_AND_EXTERNAL_VALIDATION_PLAN]], [[external_validation/CODEX_EXTERNAL_VALIDATION_REPORT]], [[external_validation/EXTERNAL_VALIDATION_BEHAVIOURAL_COMPARISON_REPORT]], [[external_validation/VALIDATION_METRIC_CANDIDATES]]

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
