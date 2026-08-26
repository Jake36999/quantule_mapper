# RFC: The IRER Geometry-Density Gravity-Like Target (Design & Search Ladder)

**Status: DESIGN / FRAMING ONLY. No run, no solver change, no claim.** This document corrects how the IRER
gravity-like mechanism should be *framed and searched for*, distinguishes it from the relational dynamics Phase D
already measured, and lays out a controlled search ladder. Cautious language is mandatory (§7). This is **not yet an
emergent-gravity claim.**

## 0. The core framing (preserve verbatim)
> **IRER gravity should be searched for as dense sequential geometry-mediated interaction around coherent mass-like
> field loads, not as simple two-node phase attraction.**

Do **not** describe the target as ordinary "mass attracts mass," a Newtonian force law, or a two-node phase-force.
The candidate mechanism is:

> A coherent mass-like field load perturbs the shared geometric field. Nearby structures respond locally; their
> response changes the field seen by neighbours; the effect propagates through **sequential local resolutions**.
> Around a large coherent load, the **density/rate of these geometry-mediated interactions** increases. Apparent
> attraction would be a coarse-grained **path bias toward lower mutual realization cost**, not a primitive force.

## 1. Grounding in the IRER theory (verified against `docs/_Declaration of Intellectual Provenance v9.txt`)
The framing is not ad hoc — it is the direct operational reading of the author's stated mechanisms:
- **Time as Chronology of Resolution (§8):** time is "the ordered progression of field resolutions (collapse events,
  governed by Resolution Field Dynamics)." → the effect must **propagate sequentially through local resolutions**,
  not appear as an instantaneous global field. (Supports the propagation-delay test, E.)
- **Resolution Field Dynamics / Informational Collapse Duality (§9):** collapse is "a continuous field phenomenon
  driven by informational gradients"; regions of high PAS/RD "create 'tension' or 'potential wells'... influencing
  the likelihood, direction, and nature of subsequent collapse events for itself and neighbouring configurations...
  it effectively helps shape the 'terrain' that other informational states navigate." → a **coherent load reshapes
  the resolution terrain** that neighbours traverse. This is the mechanism, stated in the theory's own words.
- **Fields of Minimal Informational Action / Axis of Least Effort / Coherent Manifold Channels:** motion follows
  paths of least mutual realization cost. → apparent attraction is a **coarse-grained path bias**, not a primitive
  attractive force.
- **Gradient-Derived Informational Forces:** forces are emergent from informational gradients (∇F/∇S), never
  primitive. → acceleration is a *late, coarse-grained* observable, downstream of the field/tensor response.
- **Non-instantaneous OIW propagation & hysteresis (Observer-Resolution Loop, §~feedback delay):** "due to the
  non-instantaneous propagation of OIW interactions and the time taken for PAS field restructuring... exhibits
  hysteresis." → the response has **measurable local delay**.
- **Informational Manifold Topology:** the substrate geometry the loads perturb.

## 2. Grounding in the codebase (the substrate + telemetry already exist)
- **`gravity/unified_omega.py`** — the shared geometric field: `Ω² = (ρ_vac/ρ)^a` (soft-clipped). Density ρ=|ψ|²
  sources the conformal metric Ω². A coherent load (high, stable ρ) *is* a persistent perturbation of Ω². **This is
  the geometry-response field to probe.** (Note the C2.6 finding: the soft-clip reshapes Ω²(ρ) vs the nominal law —
  the response is real but its exact form is a contract item; the gravity ladder must run with a *characterized*
  geometry, geometry-on vs a true geometry-null.)
- **`metrics/tensor_validation.py`** — `construct_T_info(ρ, φ)` builds the informational stress-energy tensor
  T_ij = κ(∂_iφ)(∂_jφ) − δ_ij·L; `tensor_symmetry_test` (|T_ij−T_ji|); `perfect_fluid_reduction_test` (mean
  off-diagonal **shear**). → the **tensor-response / alignment** observable.
- **`metrics/collapse_dynamics.py`** — `compute_correlation_length` (ξ via FFT autocorrelation),
  `compute_nonlinear_balance`, `compute_fractal_dimension_boxcount`. → **interaction-density / structural** proxies.
- **`validation_pipeline.py`** — `LOMTelemetryEngine` extracts collapse events + a **spatial gravity map**
  (`spatial_gravity_omega_sq` = Ω² at collapse-event sites, `gravity_timeline.csv`); `TensorValidationEngine` runs
  the T_info symmetry/shear checks. → the **gravity-map + tensor telemetry** the ladder reports on.
These are read-only diagnostics; the ladder uses them, it does not add active source terms without a separate RFC.

## 3. Three interaction categories — keep them separate
The single most important discipline: **do not let a phase-force be mislabelled as gravity.**
| # | category | signature | example / status | is it gravity? |
|---|---|---|---|---|
| 1 | **Phase-force / coherence** | force depends on *relative phase*; vanishes under phase randomization | C2.9 two-soliton law (crossover Δφ=π/2) — measured | **No** — baseline nonlinear-wave dynamics |
| 2 | **Charge-like / EM-like** | depends on phase *gradient / winding / current / chirality* (Payan alignment) | not yet isolated | **No** unless separately isolated |
| 3 | **Geometry-density / gravity-like** | a large *coherent load* creates a persistent **tensor/geometric response**; **neutral / phase-averaged** probes see altered path availability / capture | **untested** | **Candidate** — the actual target |
The gravity-like effect must show up **first** in interaction-density / tensor-response / current-redirection — and
must **survive phase averaging** better than ordinary two-node phase locking. Acceleration is the *last* thing to
look at, not the first.

## 4. The IRER geometry-density gravity ladder (with controls)
Each rung is geometry-on vs a **true geometry-null** (via `param_geom_off`, the C2.6-verified flat switch), with
phase-shuffled controls. Read-only telemetry: Ω²-field, T_info (symmetry + shear), correlation length ξ, LOM gravity
map, per-event interaction density.

**A. Large coherent-load baseline.** Build a large stable node/cluster (a Phase C standing attractor, or a stack of
C2.7 solitons).
- Does it create a **persistent Ω²/tensor response field** (not just its own density bump)?
- Does response **density/tensor magnitude fall with distance** (a measurable profile, not asserted 1/r)?
- Does **tensor alignment persist** in time (T_info off-diagonal / principal-axis orientation stable)?

**B. Neutral probe near load.** Introduce a small **phase-neutral or phase-averaged** probe structure at a distance.
- Does the probe's **path bend or capture differently geometry-on vs geometry-off**?
- Does **current/tensor alignment change *before* the trajectory does** (response precedes motion)?

**C. Phase-shuffled / phase-averaged controls.** Randomize the relative phase between load and probe (ensemble).
- If the apparent attraction **disappears under phase randomization**, classify it as **category-1 phase-force /
  coherence — NOT gravity-like.** (This is the primary false-positive guard.)

**D. Tensor-null / geometry-null controls.** Same density layout, geometry response removed/neutralized
(`param_geom_off`), or T_info coupling ablated.
- If the effect **survives geometry-null**, it is **not** the geometry-density mechanism (it's density-overlap /
  phase / boundary). Only a geometry-on-minus-geometry-off *difference* counts.

**E. Propagation-delay test.** Perturb the large load (small displacement / amplitude pulse) and watch the Ω² /
tensor / current response at increasing radius.
- Does the response **propagate outward sequentially with a measurable local delay** (consistent with Chronology of
  Resolution / non-instantaneous OIW), rather than updating as an instant global field?

## 5. Success criteria (candidate-support, not proof)
- Geometry-on **differs measurably** from geometry-off (rung D difference non-zero).
- A large coherent load produces a **persistent** tensor/geometric response (rung A).
- Probe **path bias correlates with tensor/current alignment *before* motion** (rung B: response precedes trajectory).
- The effect **survives phase averaging better** than ordinary two-node phase locking (rung C: not category-1).
- Response **propagates with a measurable local delay** (rung E).
- **Null controls weaken or remove** the effect (D, C behave as controls should).
Meeting these supports the **candidate mechanism**; it is *not* an emergent-gravity proof.

## 6. Falsification criteria (any one ⇒ not the geometry-density mechanism)
- Geometry-on and geometry-off behave the same.
- Phase shuffling removes the entire effect (⇒ it was category-1 phase-force).
- Tensor/current alignment does **not** precede motion.
- Large-load density does **not** change interaction-density observables.
- Apparent attraction is fully explained by ordinary NLS phase-force behaviour.

## 7. Language contract (mandatory)
**Use:** "gravity-like", "geometry-density interaction", "sequential tensor-mediated response", "candidate
mechanism", "not yet an emergent-gravity claim", "path bias toward lower mutual realization cost".
**Never use:** "gravity proven", "IRER proves gravity", "mass attracts mass", "Newtonian law reproduced", "matter
proven".

## 8. Scope & sequencing
Design-only. Prerequisites before any rung runs: (i) a **characterized geometry** (the C2.6 soft-clip contract —
know the actual Ω²(ρ) being used, geometry-on vs true-null); (ii) a **large coherent load** that is genuinely stable
(Phase C attractor or stacked C2.7 solitons); (iii) read-only telemetry wired (Ω², T_info, ξ, LOM map) with
geometry-null and phase-shuffle controls built in from the first pilot. The ladder is **interaction-density /
tensor-response first** by construction; acceleration is measured last and only after the tensor/response signal is
established. No active gravity source term is added without a separate, gated RFC.

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `c2f320e` (2026-07-09) — *Phase D closeout consolidation + IRER geometry-density gravity-target RFC*
**Revised since:** 2 commit(s), most recently `3eb93af` (2026-08-26)

**Harness code changed since it was written:** 15 commit(s) to `jax_scout/`.
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine
  - `25020bf` 2026-07-22 — dual substrate
  - `9a839c4` 2026-07-10 — Phase D C3 collision ladder: add relative-phase (--dphi) + BOUNCE clas
  - *…and 10 more.*

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Also referenced by (same date or earlier):** [[IRER_GRAVITY_RUNG_A_D_RESULTS]], [[PHASE_D_CLOSEOUT_CONSOLIDATION]]

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
