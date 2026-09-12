# Future Work & External Validation Plan

**Phase name:** External Legibility and Quantitative Validation
**Status:** planning document — sits *above* the evidence package; tells Claude/Codex what comes next.
**Not:** a "proof phase", not a "unification phase", not a matter/gravity claim.

This document is written after the three-layer documentation foundation was completed:

```
catalog          (docs/IRER_MASTER_HYPOTHESIS_CATALOG.md)   = what was tested
theory synthesis (docs/theory_synthesis/)                    = what it means
evidence package (docs/evidence_package/)                    = where the artifacts live
```

and after an independent external-formalism comparison ("Agnostic Formalism Comparison for the IRER /
Quantule Mapper") mapped the implemented core onto standard literature. The purpose of the next phase is to
convert that qualitative "this looks like known physics" result into **quantitative faithfulness metrics**
measured against documented analytic laws, and to line up the external sources/datasets a later comparison
would use.

---

## 0. Framing (binding)

**Overlap with standard models is expected and is not a weakness.** A stated project goal is to translate
relationships *between* already-documented nonlinear field models into one conceptual language (IRER). When the
Quantule Mapper reproduces Karpman–Solov'ev soliton forces, Vakhitov–Kolokolov stability, Galilean/Lorentz
transport, and conformal analogue-geometry, that is evidence the **instrument is correct**, and it makes the
model **legible and testable**. It is not evidence of novel physics, and this plan does not treat it as such.

The distinctive research target is the *assembly and interpretation* — in particular the closed-loop algebraic
self-metrization (field sources its own conformal metric `Ω²(ρ)`, propagates via Laplace–Beltrami) — which is
**novel-but-delicate** (non-canonical / quasi-conservative, per the contract review) and whose gravity-like
payoff is **unproven and paused**. Keep that precise.

**Match-level vocabulary** (used throughout; never blurred):
`exact equation` ⊃ `same family` ⊃ `close analogue` ⊃ `speculative interpretation`.

**Authorship:** IRER's concepts and the research direction are Jake McIntosh's. AI roles are
formalization, implementation, measurement design, and review.

---

## 1. Current project status

| sector | status |
|---|---|
| Phase C — stability | **CLOSED** — a\* attractor confirmed; mobility + structural predictors NULL |
| Phase D dissipative (D.1–D.6) | **CLOSED** — nodes couple + merge, never transport |
| Phase D conservative NLS (C2) | **POSITIVE** — v=2Dk transport; two-body π/2 force + capture (after the C2.6 fix) |
| Phase D wave-kinetic KG (C3) | **POSITIVE** — VK-stable Q-ball; inertial transport; mapped collision phase diagram |
| geometry contract | **REVISED** — quasi-conservative (self-adjoint w.r.t. Ω³); C2′ = design-only fix |
| gravity-density | **PAUSED** — channel live, Ω²(ρ) is a saturation cliff, documented re-entry gate |
| Hunter / objective | **RE-AIMED** — stability objective; prime-SSE retired |

**One-line:** two sectors closed, transport positive across two conservative substrates (NLS + KG), gravity
scoped-and-paused, three instrument bugs caught and corrected — **no matter claim, no gravity claim, no
physical-proof claim.**

---

## 2. Strategic goal for the next phase

Move from **internal correctness** (does the code do what the equations say?) to **external legibility** (do the
measured results match the documented analytic laws those equations are known to obey, quantitatively?).

Three tracks, in dependency order:

1. **Internal quantitative validation** (§3) — Claude, using *existing run data* or cheap reruns. Compare
   Quantule Mapper outputs against known analytic laws. This is the core new work and the highest value.
2. **Optional bounded science runs** (§4) — new data; ranked by value/cost; approval-gated.
3. **External literature/dataset comparison** (§5) — Codex, *after* metrics are specified: gather sources,
   formulas, datasets, and compute first-pass comparisons. No reinterpretation of IRER, no verdict changes.

---

## 3. Internal quantitative validation metrics

Each metric below is a **fit of existing/near-existing data to a documented analytic form**, with an explicit
success and falsification condition. These are faithfulness tests of the simulator against known physics — not
claims of new physics. Run folders are the local (gitignored) `sweep_runs/` names recorded in the evidence
package manifests.

### V1 — Karpman–Solov'ev / Gordon phase-force fit *(highest value)*

**Claim tested:** the measured two-body interaction is not merely "π/2-crossover-shaped" but *quantitatively*
follows the standard soliton perturbation-theory law.

**Analytic target** (Karpman–Solov'ev 1981; Gordon 1983, two identical NLS solitons, half-separation `q`,
relative phase `Δφ = 2Ψ`):

```
q̈  = −C · e^(−λq) · cos(Δφ)      with, for canonical NLS solitons, C = 4A³ and λ = 2A
Ψ̈  = −C · e^(−λq) · sin(Δφ)
```

**Data:** existing C2.9 static-pair runs (`sweep_runs/…C29…` `track_*.npz`, separation-vs-time per Δφ) and the
C3 two-Q-ball static runs (`sweep_runs/C3_TWOQBALL_FULL/` `pair_*.npz`). Both already sweep Δφ.

**Method:** from each static pair, extract centre-of-mass separation `2q(t)` via the momentum-density observable
(already validated), estimate the initial relative acceleration `q̈` at fixed initial separations, and fit
`q̈(q, Δφ)` to the form above.

**Metrics reported:**
- fitted decay rate `λ` vs the canonical `2A` (from the measured amplitude);
- fitted prefactor `C` vs `4A³`;
- phase sign accuracy (does `sign(q̈)` follow `−cos(Δφ)` at every tested Δφ?);
- π/2 crossover error (Δφ at which fitted `q̈ = 0`, vs π/2);
- residual RMS by phase;
- **C2 vs C3 comparison** — do the first-order (NLS) and second-order (KG) substrates fit the *same* functional
  law (cross-substrate universality, now quantitative)?

**Success:** sign law holds at all Δφ; crossover within a few % of π/2; residuals small; C2 and C3 share the
exponential-cos structure. Yields the target sentence: *"The measured Quantule Mapper two-body phase force
follows the standard soliton-interaction sign law and fits an exponential-overlap model within [X]% residual
across tested phases/separations, in both the NLS and KG substrates."*

**Falsification:** sign law breaks at some Δφ, or the separation dependence is not exponential, or C2 and C3
fit incompatible forms → the measured interaction is *not* the canonical KS/Gordon force and the "same law"
claim must be narrowed.

*Note:* our nonlinearity is cubic-quintic-septic (non-integrable), so **exact** agreement with the pure-cubic
KS/Gordon constants is not expected — the honest target is "same functional family, sign law exact, constants
close" (a `same-family`/`close-analogue` result, not `exact-equation`). State it that way.

### V2 — Galilean transport line (C2)

**Claim tested:** corrected C2 soliton velocity obeys `v = 2Dk`.
**Data:** corrected-substrate transport runs (`sweep_runs/C27_REDERIVE/`, `param_geom_off=True`), swept carrier `k`.
**Method:** fit `v_measured` (momentum-density) vs `k`.
**Metrics:** fitted slope vs `2D`; intercept (should be ~0); mass retention per point; R².
**Success:** slope = 2D within ~1%, intercept ≈ 0. **Falsification:** slope off, or nonzero intercept, or mass loss correlated with k.
*(This is an `exact-equation` target — Galilean invariance is exact for the NLS; deviation = numerics.)*

### V3 — KG / Q-ball VK branch

**Claim tested:** the C3 Q-ball sits on a VK-stable branch (`dQ/dω < 0`).
**Data:** `sweep_runs/C3_EXACT_VK2/` `Q(ω)` points (Q ≈ 59→49 over the scanned window; `dQ/dω = −849`).
**Method:** fit `Q(ω)`, report slope sign and magnitude; where a standard Q-ball existence-window formula is
available (§5), overlay it.
**Metrics:** `dQ/dω` sign + value; monotonicity; branch coverage; (if a literature `E/Q < m₀` threshold is
extracted) the absolute-stability fraction of the window.
**Success:** `dQ/dω < 0` across the window (VK-stable). **Falsification:** sign flip or non-monotone `Q(ω)`.
*(`same-family` target — VK criterion is the standard tool; our object is a standard complex-scalar Q-ball.)*

### V4 — KG boost fidelity line

**Claim tested:** measured C3 centroid speed tracks the intended Lorentz-boost speed; the ~10% looseness is
measurement (short window + Q-ball breathing), not a physical transport defect.
**Data:** `sweep_runs/C3_WAVE_BOOSTFIX2/`, `C3_EXACT_VK2/` (exact-contraction boost).
**Method:** fit `v_measured` vs intended `v`; separate systematic (slope ≠ 1) from noise (breathing jitter); test
whether exact Lorentz contraction changed the slope (it did **not** — that was FALSIFIED as the limiter).
**Metrics:** slope, residual vs breathing period, window-length dependence.
**Success:** slope → 1 as box/T grow (looseness shrinks with window). **Falsification:** fixed slope < 1
independent of window → a real transport deficit, not measurement. *(Feeds RUN-4.)*

### V5 — C3 collision phase-diagram metric

**Claim tested:** the capture/transmission boundary is a clean function of `(Δφ, v/c)` — capture generic;
transmission only near exact anti-phase below ~0.5c.
**Data:** `sweep_runs/C3_COLLISION_LADDER_FULL/`, `C3_ANTIPHASE_LADDER/`, `C3_PHASE_{pi2,3pi4,7pi8}/` (`collide_*.npz`).
**Method:** tabulate outcome + radiation fraction + E/Q drift + elasticity per `(Δφ, v/c)` cell; render the grid.
**Metrics:** boundary location; radiation fraction vs v; E/Q drift (conservation fidelity, ≤5.6e-6 seen);
elasticity per cell; classifier-ambiguity flags.
**Success:** reproduces the documented diagram with machine-clean conservation. **Falsification:** conservation
drift grows, or the boundary is classifier-artifact-sensitive.

### V6 — Old-vs-corrected C2 overlay *(the maturity figure)*

**Claim tested:** the pre-C2.6 "pinning / drag μ≈0.036" was the `D_eff = D/151` instrument artifact; the
corrected substrate transports at `v = 2Dk`.
**Data:** existing `sweep_runs/C24_LOCAL_N96/`, `C23_N96/` (old) vs `C27_REDERIVE/` (corrected); C2.6 gate/audit
outputs for the `D/151` one-step identity (`D_eff` ratio 0.006511 ≈ 1/151).
**Method:** single overlay plot (old μ≈0.036 creep vs corrected 2Dk translation) + one identity table.
**Metrics:** the two transport curves; the `D/151` number old-vs-corrected.
**Success:** the overlay makes the artifact→correction transition legible in one figure. *(This is mostly
ENR-6/ENR-7 — mechanical, from existing summaries.)*

### V7 — Ω² conformal-law diagnosis *(gravity re-entry design constraint, NOT a gravity result)*

**Claim tested / insight:** the vacuum cliff is a *physically-diagnosable* property of the conformal exponent,
not just a numerical accident.

The comparison report gives, for a polytropic acoustic metric `p = kρ^γ`, the conformal factor
`Ω² ∝ ρ^((3−γ)/2)`, i.e. the IRER exponent maps as:

```
Ω²(ρ) = (ρ_vac/ρ)^a   ⇒   a = (γ − 3)/2      (equivalently  γ = 2a + 3)
```

For the **physical BEC** case `γ = 2` this gives `a = −1/2`: Ω² *vanishes* as ρ→0. The IRER production value
`a ≈ 2.31` implies `γ ≈ 7.6` **and the opposite sign** — Ω² *diverges* as ρ→0. That divergence at the near-zero
simulation vacuum is exactly the saturation cliff the gravity ladder hit
(`GRAVITY_LADDER_GEOMETRY_DECISION.md`). So the standard acoustic-metric mapping *explains the cliff*: our
conformal law sits outside the graded, physically-motivated regime.

**Method:** tabulate `a → γ` for the production and candidate exponents; overlay the IRER `Ω²(ρ)` against the
BEC-analogue `ρ^((3−γ)/2)` curve; mark the vacuum-divergence region.
**Deliverable:** a re-entry *design constraint* — "for a graded, analogue-gravity-consistent well, target
`a < 0` (or an equation of state with `γ < 3`) on a filled `ρ_vac` background", feeding RUN-5. Treat strictly
as `close-analogue` design guidance; **not** a gravity result.

---

## 4. Optional bounded science runs (new data — approval-gated)

Ranked by scientific value ÷ cost. These are **future science**, not documentation fixes. Cross-referenced to
`docs/evidence_package/EVIDENCE_GAPS.md` (RUN-2..5).

| id | hypothesis | expected outcome | success metric | falsification | telemetry | who | cost |
|---|---|---|---|---|---|---|---|
| **RUN-2** — C2 anti-phase collision (NLS analogue of the C3 diagram) | if cross-substrate universality extends to collisions, the first-order NLS substrate should also show a narrow anti-phase node-protected transmission window | PASS_THROUGH near Δφ=π at low v; capture elsewhere | transmission at Δφ=π / capture at Δφ<π, matching C3's phase×speed structure | no anti-phase channel in NLS (transmission phase-independent, or none) → universality is KG-specific | ∫ρ conservation, momentum-density velocity, radiation fraction, classifier | **Claude** (measurement design) | ~10–20 min |
| **RUN-3** — C3 captured-remnant long-time fate | a captured Q-ball pair either relaxes to a long-lived "Q-ball molecule" or slowly radiates/decays | bound oscillating remnant, slow Q drift | Q, E drift over long T; remnant amplitude/width stability | remnant disperses → capture was transient | E, U(1) charge, radial profile time series | **Claude** | ~15 min |
| **RUN-4** — C3 continuum/bigger-box velocity fidelity | the ~10% boost looseness is window-limited; larger box + longer T tightens `v_frac→1` | v_frac → 1 as box/T grow | slope→1, residual↓ vs V4 baseline | fixed slope < 1 → real deficit | boost line, breathing period, mass retention | **Claude** | ~20–30 min |
| **C2′** — canonical divergence-form geometry | adding the metric-variation term `D·Ω′(ρ)\|∇ψ\|²ψ` makes geometry-on C2 exactly conservative | machine-exact conservation with geometry on | ∫ρ / energy drift → ~1e-13 | still quasi-conservative → the covariant term is not the whole story | full conservation telemetry, parity vs geom-off | **Claude** (RFC-first; `docs/PHASE_D_C2PRIME_CANONICAL_GEOMETRY_RFC.md`) | design + run |
| **RUN-5** — stable-overdense-load on ρ_vac background | a persistent core with ρ>ρ_vac on a filled ρ_vac background yields a *graded* Ω² well (re-entry condition) | overdense load survives; Ω²(ρ) graded, no vacuum cliff | load persistence + monotone graded Ω²(r) | load dissolves on filled bg (as before) → re-entry still blocked | load mass/persistence, radial Ω², T_info shear | **Claude** (design); Codex may replicate once defined | research sub-project |

### TG-A/TG-B - temporal-geometric feedback formalization and reduced scout

**Codex addendum, 2026-07-14. Conceptual authorship remains Jake McIntosh.** This branch is motivated by
`docs/theory_synthesis/IRER_TEMPORAL_GEOMETRIC_FEEDBACK_LOOP.md`. It is not a production re-entry and does not change
the current Gravity D or G1 labels.

**Claim to test:** a phase-locked OIW/node can enter a bounded temporal-geometric relaxation cycle:

```text
Psi -> R_res[Psi] -> T or N_t -> G or A_s -> modified Psi evolution -> outgoing perturbation
```

**TG-A design gate:** before simulation, specify the resolution source, choose either conservative exchange or
explicit dissipative accounting, derive the temporal response PDE, geometric response PDE, positive coefficient
maps, energy/action ledger, null controls, flat-state stability and frequency diagnostics. Do not start with a full
3D double-field solver.

**TG-B reduced scout:** after TG-A review, run a zero-dimensional exchange model first, then implement a 1D radial or
2D axisymmetric KG-first scout with a later NLS comparison. Sweep phase relation, carrier, and separation for two
OIW/node structures; measure whether `R_res` peaks before `T`, `T` before `G`, and `G` before outgoing flux.

**Primary metrics:** event frequency, temporal-field frequency, geometric-field frequency, outgoing flux frequency,
cross-correlation lags, wavelet/coherence spectra, energy or action accounting, and whether an FMIA-like
low-resistance channel forms without being hand-coded.

**Success:** causal phase-ordered sequence survives source-off, temporal-off, geometric-off, feedback-off,
phase-scrambled, grid, timestep, and box controls. **Falsification:** pulses are absent, track numerical cadence,
require a hand-written emission trigger, or disappear under matched controls.

**G1b boundary:** TG-A/TG-B does not replace objective local-clock calibration. A positive feedback result can
characterize a temporal-geometric field mechanism, but it cannot establish objective time dilation while the clock
instrument remains uncalibrated.

**Interpretation boundary:** this can support a temporal-geometric feedback-loop scout, not gravity, photons,
geodesics, universal free fall, or production readiness. Photon-like labels require a separately defined outgoing
field observable and quantization test.

**No Section-4 run is required** for the current results to stand. RUN-2 is the most scientifically informative
and cheapest — the natural first pick when Claude returns to active science.

---

## 5. External literature / dataset comparison programme (Codex, after §3 metrics exist)

Codex **collects comparison formulas and datasets; it does not reinterpret IRER or change verdicts.** Every
entry carries a match-level label (`exact` / `same-family` / `close-analogue` / `speculative`) and a source
reliability label. Foundational correspondences below are Claude-vouched; recent/specific citations from the
comparison report (e.g. the "Salih 2026" info-theoretic Q-ball paper) must be marked **to-verify** — deep-
research agents do fabricate plausible citations, so source existence is confirmed before any lean is placed.

**Analytic laws to extract (with canonical references):**
- NLS / Gross–Pitaevskii bright/dark solitons; Galilean boost `v=2Dk`.
- Cubic-quintic / cubic-quintic-septic solitons; flat-top solitons; non-integrable collision/capture.
- Complex Ginzburg–Landau dissipative solitons (gain/loss-balanced attractors).
- Karpman–Solov'ev (1981) / Gordon (1983) two-soliton equations of motion → **V1 target law**.
- Klein–Gordon / Q-balls / non-topological solitons; Coleman (1985); Friedberg–Lee–Sirlin; Vakhitov–Kolokolov
  criterion and `E/Q < m₀` absolute-stability threshold → **V3 target**.
- Analogue-gravity acoustic metrics (Unruh; Visser 1998); polytropic `a = (γ−3)/2` → **V7 target**.
- Madelung / Bohm quantum hydrodynamic stress tensor (Madelung 1927; Takabayasi 1952) → T_info analogue.

**Datasets to locate (open/published; comparison quantity noted):**
- Optical soliton-molecule collision spectra (dispersive Fourier transform, real-time) → collision phase
  behaviour, capture/fusion.
- BEC bright-soliton collision experiments → phase-dependent bounce/pass, velocity–momentum relation.
- Dissipative-soliton (fiber-laser / cavity) datasets → existence range, pinning.
- Q-ball / oscillon numerical benchmark data → `Q(ω)`, VK sign.
- Analogue-gravity BEC/acoustic experiments → metric response to density, propagation.

**Codex deliverables:**
```
docs/external_validation/EXTERNAL_SOURCE_REGISTRY.md      (sources, existence-verified, reliability)
docs/external_validation/VALIDATION_METRIC_CANDIDATES.md  (extracted formulas, units/scaling caveats, match-level)
docs/external_validation/DATASET_CANDIDATES.md            (datasets, availability, comparison quantity)
```
Each row: source verified? · formula/dataset extracted · units/scaling requirements · reliability · match-level.

---

## 6. Agent role split (governance)

| | **Claude** | **Codex** |
|---|---|---|
| scientific comparison design | ✅ owns | ❌ |
| metric choice / measurement design | ✅ owns | ❌ |
| interpretation, cautious language, verdict/status updates | ✅ owns | ❌ (no verdict changes) |
| running §3 fits, §4 measurement runs | ✅ | replicate only, once defined |
| documentation enrichment (ENR-1..7) | review | ✅ executes |
| source audit / formula extraction / dataset gathering | review | ✅ executes |
| first-pass external metric computation | review | ✅ (reports uncertainty; flagged provisional) |

**No Codex output changes a verdict without Claude review.** Codex external metrics are *initial/provisional*
until Claude reviews the data and comparison.

For feedback-loop formalization specifically: Jake owns the conceptual hypothesis and accepts model choices. Codex
may propose and implement candidate mathematics, controls and instrumentation. Any scientific verdict remains
provisional until reviewed.

---

## 7. Near-term execution order

1. **Codex:** documentation enrichment (`EVIDENCE_GAPS.md` ENR-1..7) — fill manifest paths, checksums, link
   integrity, contact sheets, render canonical figures (V5/V6 grids, VK line, radial cliff) from existing `.npz`.
2. **Codex:** external equation/source audit → `docs/external_validation/` registry (§5), match-level + verify.
3. **Claude:** **V1 Karpman–Solov'ev / Gordon quantitative comparison** (existing C2.9 + C3 pair data) — the
   flagship, cheap, converts the strongest "looks like known physics" result into a quantitative faithfulness
   metric. Then V2–V7 as data allows.
4. **Codex:** external dataset/source collection (§5 deliverables).
5. **Claude:** review Codex's external metrics + data; fold verified comparisons into theory status.
6. **Optional science:** RUN-2 (C2 anti-phase collision) as the first new-data pick.

*(Steps 1–2 and 4 run in parallel on Codex while Claude does step 3 on the frontier; step 5 is the join.)*

---

## 8. What this phase will and will not have established

**Will (if metrics pass):** that the Quantule Mapper's implemented mechanisms reproduce the documented analytic
laws they are built from — quantitatively, in the field's own language, with match-levels stated honestly. This
makes the work *legible to outsiders* without overclaiming.

**Will not:** prove IRER is a unified theory, prove matter or gravity emerges, or claim experimental
correspondence. External datasets are comparison *candidates*; agreement, where found, is a faithfulness check
of a known-physics simulator, not a discovery. The one genuinely distinctive element (closed-loop algebraic
self-metrization) remains novel-but-delicate and, on the gravity side, paused behind the V7/RUN-5 re-entry gate.

---

*Cross-refs:* `docs/IRER_MASTER_HYPOTHESIS_CATALOG.md`, `docs/theory_synthesis/`,
`docs/evidence_package/` (esp. `EVIDENCE_GAPS.md`, `08_external_comparison_candidates.md`),
`docs/PHASE_D_C2_9_TWONODE_ROBUST_RESULTS.md`, `docs/PHASE_D_C3_*`,
`docs/GRAVITY_LADDER_GEOMETRY_DECISION.md`, `docs/PHASE_D_C2PRIME_CANONICAL_GEOMETRY_RFC.md`, and the external
"Agnostic Formalism Comparison for the IRER / Quantule Mapper" report.

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `a81b494` (2026-07-10) — *Future Work & External Validation Plan: quantitative-faithfulness roadmap*
**Revised since:** 12 commit(s), most recently `bd93089` (2026-09-12)

**Harness code changed since it was written:** 6 commit(s) to `jax_scout/`.
  - `caf61af` 2026-09-10 — Build HUD items 1+2: snapshot writer and universal field renderer
  - `06edae5` 2026-08-27 — Fix run provenance: 30% -> 86% of runs now dated
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine
  - *…and 1 more.*

**Later documents that cite this one** — the downstream consequences:

- [[ACTION_PLAN_2026-08]] &middot; `2026-08-25`
- [[INTEGRATED_PLAN_2026-09]] &middot; `2026-09-12`
- [[Main branch]] &middot; `2026-08-25`
- [[SESSION_SYNTHESIS_2026-08]] &middot; `2026-08-25`
- [[SYSTEM_PRESSURE_TEST_AND_METHOD_REVIEW]] &middot; `2026-08-25`
- [[external_validation/CODEX_EXTERNAL_VALIDATION_REPORT]] &middot; `2026-07-22`

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
