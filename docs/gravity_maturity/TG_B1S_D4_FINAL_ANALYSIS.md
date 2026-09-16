# TG-B1S D4 — Final Comprehensive Analysis (primary review)

Author: Claude (primary reviewer), 2026-07-15.
Consolidates: Codex's first-pass D4 banking (`TG_B1S_D4_RESULTS.md`, `TG_B1S_D4_SUMMARY.json`,
`TG_B1S_D4_DOCUMENTATION_INPUTS.md`) + Claude's frequency contract (`TG_B1S_FREQUENCY_CONTRACT_RESULTS.md`) and
box-dependence discriminator (`TG_B1S_BOX_DEPENDENCE_RESULTS.md`). This is the authoritative synthesis of what the
D4 campaign established and what remains to close it.

Run: `sweep_runs/TG_B1S_D4_ROWS_GPU_20260715_091558` (WSL/JAX CUDA, x64, git `9fc7785`), marker `D4_RUN_COMPLETE.json`.

## 1. Formal verdict (unchanged from Codex's banking)

```text
TG_STATE_LOAD_D4_NUMERICAL_VALIDATION_FAILED
```

D4 did **not** close. 5 of 6 rows passed; the `larger_box` row missed the preregistered frequency-scale gate.
D5 stays **blocked**; no promotion to `TG_STATE_LOAD_ORBITALLY_BOUNDED_FREQUENCY_SHIFT_SUPPORTED` or
`TG_STATE_LOAD_BOUNDED_FEEDBACK_SUPPORTED`. Current formal status remains `TG_STATE_LOAD_LONG_TIME_DRIFT_UNRESOLVED`.
No tolerance relaxation is justified by this result.

## 2. Row-by-row

| row | N | L | dx | result | fitted Δω∞ | rel. freq. diff (gate ≤0.60) | early/late |
|---|---|---|---|---|---:|---:|---:|
| baseline_50P | 48 | 10 | 0.208 | PASS | −2.1627e-6 | 0.0055 | 0.301 |
| output_cadence_50P | 48 | 10 | 0.208 | PASS | −2.1612e-6 | 0.0048 | 0.307 |
| dt_half_50P | 48 | 10 | 0.208 | PASS | −2.1627e-6 | 0.0055 | 0.301 |
| grid_refined_50P | 56 | 10 | 0.179 | PASS | −2.1627e-6 | 0.0055 | 0.301 |
| **larger_box_50P** | **56** | **12** | 0.214 | **FAIL** | **−8.524e-7** | **0.6037** | 0.0014 |
| absorber_wider_50P | 48 | 10 | 0.208 | PASS | −2.1627e-6 | 0.0055 | 0.301 |

Reference Δω∞ = −2.150936882840006e-06 (D3 100P). The failed row's other diagnostics stayed clean: profile overlap
0.99999999990, modal leakage 3.38e-5, ledger residual 1.09e-5, boundary flux 6.3e-8, T/G `APPROACH_FIXED_PROFILE`,
no structural drift. So this is a **frequency-scale/comparability** failure, not runaway, boundary contamination,
structural drift, or solver instability.

## 3. What the five passing rows established (positively)

At the baseline geometry (L=10), the state-load modal frequency shift Δω ≈ −2.16e-6 is **robust** to:
timestep halving (identical to ~1e-11), output cadence, grid refinement (N=48→56), and a wider absorber. That is a
genuine and non-trivial numerical-stability result — the shift is not a timestep, sampling, resolution, or absorber
artifact at L=10. Codex's campaign design and preregistered gates are sound.

**Mechanism, additionally explained (Claude FC-1).** The frequency contract established the shift's sign and scale
analytically: the D3 `delta_omega_infty` convention (slope of θ_full−θ_off, θ≈−ωt) means the physical shift is
**+2.15e-6 (node oscillates faster)**; both analytic legs predict positive, and the measured value is **bracketed**
by the unrelaxed fixed-profile (+5.40e-7) and fully-relaxed fixed-Q (+4.69e-6) limits. The audit's earlier
"sign puzzle" was a reporting-convention artifact, not a physics discrepancy. So the shift the passing rows
validate is **mechanistically understood**, not merely numerically converged.

## 4. Anatomy of the `larger_box` failure

**Isolating variable = L, not resolution.** From the D4 driver's `selected_cases`: `grid_refined` (N=56, L=10)
passed cleanly, so the N=56 grid itself is fine. The *only* variable unique to the failing row is the box
enlargement L=10→12 (which also moved the absorber shell r=3.5→4.2 and nudged dx 0.208→0.214). The shift dropped
to 39% of the reference.

**Box-dependence discriminator (Claude, `TG_B1S_BOX_DEPENDENCE_RESULTS.md`).** Recomputing the FC-1 analytic shift
across five geometries (including two resolution controls) gives two clean eliminations:

- **NOT resolution/dx.** `larger_box` (dx=0.214) and `larger_box_fine` (N=68, dx=0.176) give *identical* analytic
  shifts (dx_effect = 0.000); baseline and `baseline_fine` likewise. Fully grid-converged.
- **NOT the local quasi-static mechanism.** The local screened-stiffening picture predicts the shift should *rise
  ~32%* in the bigger box (Petviashvili Q-ball at L=12 has slightly more gradient-energy overlap; ∫(A−1)|∇φ|²:
  1.81e-4→2.40e-4). The measurement *fell 61%*. **Analytic and measured move in opposite directions.**

**Correction (Codex catch, verified by Claude in code).** An earlier draft of this section listed "fixed-reference
comparability" as the top candidate, implying the row's shift was measured against a foreign-geometry baseline.
That is wrong. Verified in `jax_scout/gravity_TG_B1S_D4_rows_gpu.py` (`d4_row`, lines 129–132): each D4 row builds
its own per-geometry config, solves the Q-ball at that geometry, and evolves **both** the full-loop and
feedback-off states there. `delta_omega` is the slope of `(θ_full−θ_full[0]) − (θ_off−θ_off[0])` — a genuine
**same-geometry** L=12 full-minus-off shift. The fixed L=10 reference enters **only** the gate's relative-difference
test (line 138–139), not the measured shift. So the L=12 shift really is ~39% of the L=10 shift; this is a
**real, same-geometry, box-dependent measurement**, not a full-vs-off comparison artifact.

Therefore the measured −61% drop is genuine and same-geometry, is not contained in the modelled quasi-static local
mechanism (which predicts +32%), and must originate in what the **static** estimate omits. Corrected candidate
ranking:

1. **Absorber-position coupling (leading).** The absorber shell moved r=3.5→4.2 (0.35·L). The static analytic solve
   neglects the absorber entirely; the dynamical run does not. `absorber_wider` tested absorber *width* at L=10
   (passed) but not absorber *position*, which only changed in `larger_box`. A repositioned absorber changes the
   T/G tail boundary and thus the dynamical steady A(x) the node actually sees.
2. **T/G transient dynamics over 50P.** The quasi-static estimate uses the fully-settled screened response; the
   50-period dynamical shift integrates the ring-in and any slow T/G evolution, which can differ at L=12.
3. **Modal-projection / fit-window or a subtly changed stationary branch at L=12.** The Petviashvili solve is
   mildly box-sensitive (it drove the analytic +32%); the dynamical modal fit could weight this differently.

Separately, a **gate-design question** remains (not an artifact claim): the gate requires the L=12 shift to match a
*fixed* L=10 reference, i.e. it implicitly demands box-invariance. If the shift is legitimately (mildly)
box-dependent, that requirement — not the measurement — may be what needs revisiting. But the measured drop is far
larger than any mild dependence the local mechanism predicts, so a row-level cause (candidates 1–3) must be found
first.

## 5. Recommended closing test (for the CX run-data discrepancy review)

Since the full-minus-off subtraction is already same-geometry, the closing work is **row-level diagnostics on the
existing D4 artifacts** (Codex's lane, CPU, no new field evolution) to find *which* row-level quantity drives the
real L=12 scale change, testing the corrected candidates in order:

1. **Absorber-position isolation (decisive).** From existing artifacts, compare the larger_box row's late-time
   `T_peak`/`G_peak`/`G_node` and T/G approach slopes against baseline; if the dynamical steady A(x) at the node is
   materially weaker at L=12, absorber repositioning is the cause. If artifacts are insufficient, queue **one
   preregistered row** that varies absorber *position* at fixed L=10 (or pins the absorber to the node at L=12) —
   not a tolerance change.
2. **Per-geometry Q-ball branch + T/G + phase-fit comparison** (Codex's diagnostic list): charge, energy, node
   width, core amplitude, own-reference overlap, gradient energy, S_state integral/distribution; T/G peaks and
   late-time values; δθ(t) early/late slopes, fit residual, instantaneous-frequency variance.
3. **Transient/fit-window** (TRANS-1): confirm whether the 50P window over-weights the L=12 ring-in.

Outcomes:
- A row-level cause is identified (e.g. absorber-position) → characterize it, decide whether box-invariance is a
  fair D4 requirement, and only then re-run/extend the gate before D5.
- No row-level cause explains it → the shift is genuinely box-sensitive; box-invariance becomes an explicit,
  documented requirement (or explicit non-requirement) of any bounded-shift claim, decided before D5.

Either way: **no tolerance relaxation, no parameter tuning, no post-hoc gate change to force a pass** — the
resolution is a mechanism/observable question, not a threshold question. Claude verifies the CX closure before any
D5 launch.

## 6. Secondary observations

- **Transient window (TRANS-1).** early/late slope difference is 0.30 at 50P but 0.058 at the 100P run — the early
  transient decays and the shift is cleaner over longer windows. D5/longer rows should preregister a transient
  discard window; the 50P window is the least comfortable place to measure "constant" frequency.
- **Screening (SCREEN-1).** G-tail length ~1.4–1.7 (source-size-inflated); naive c_G/ω_G = 0.65, light coupled-mode
  range ≈ 0.82 → short-range Yukawa mediation confirmed. This is *why* a purely local effect should be nearly
  box-insensitive — consistent with the analytic Δω being flat across L and inconsistent with the measured drop
  being a local-mechanism property.

## 7. Status and boundaries

- Formal: `TG_STATE_LOAD_D4_NUMERICAL_VALIDATION_FAILED`; `TG_STATE_LOAD_LONG_TIME_DRIFT_UNRESOLVED` retained.
- D5 blocked; no bounded-feedback promotion.
- Even a fully-passed D4/D5 would support only a numerically converged, orbitally bounded modal-frequency shift in
  the tested phenomenological state-load T/G model — **not** gravity, time dilation, objective chronology, geodesic
  motion, universal free fall, photon emission, IRER validation, or production readiness.
- No production geometry, Hunter, solver, or master-theory verdict was touched.

## 8. Immediate next actions

1. **CX (run-data lane, CPU on existing artifacts):** the §5 absorber-position isolation + per-geometry Q-ball/T-G/
   phase-fit comparison → identify the row-level quantity driving the real L=12 scale change.
2. **If a row-level cause is found (e.g. absorber-position):** characterize it, decide whether box-invariance is a
   fair D4 requirement, and only then extend/re-run the gate before D5. If artifacts are insufficient, queue **one
   preregistered** row (e.g. absorber-position at fixed L) — never a post-hoc tolerance change.
3. **Claude (primary review):** verify the CX closure before any D5 launch or promotion.

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `25020bf` (2026-07-22) — *dual substrate*
**Revised since:** 12 commit(s), most recently `59b7ec3` (2026-09-12)

**Harness code changed since it was written:** 5 commit(s) to `jax_scout/`.
  - `caf61af` 2026-09-10 — Build HUD items 1+2: snapshot writer and universal field renderer
  - `06edae5` 2026-08-27 — Fix run provenance: 30% -> 86% of runs now dated
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Also referenced by (same date or earlier):** [[RUN_QUEUE]], [[gravity_maturity/CLAUDE_HANDOFF_20260715_TG_B1S_AND_RUNTIME]], [[gravity_maturity/CONCEPT_IMPLEMENTATION_AND_GAP_ANALYSIS]], [[gravity_maturity/CONCEPT_IMPLEMENTATION_AND_GAP_ANALYSIS_CODEX_CONTINUATION_20260717]], [[gravity_maturity/TG_B1S_BOX_DEPENDENCE_RESULTS]]

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
