# Claude Handoff: TG-B1S State-Load Feedback And Runtime Architecture

Author: Codex  
Timestamp: 2026-07-15  
Scope: live gravity-maturity handoff, not a master-theory verdict.

This document summarizes the recent TG-B1S state-load feedback work and the runtime issue discovered during D4 closure. It is intended to let Claude resume review without reconstructing the full Codex session.

## Current Scientific Status

The current frozen state-load model has not been redesigned during this sequence.

Preserved model constraints:

- Source family: `S_state` only.
- Fixed global normalization: `S0 = 135.6862187684289`.
- Disabled sources: `R_relax`, `L_lock`, `P_threshold`.
- Frozen T/G equations, damping, coupling signs, coefficient maps, Q-ball initialization and absorber formulation.
- No gravity, geodesic, universal-free-fall, photon, objective-time or IRER-validation claim is made.

Current formal status before D4/D5 closure:

```text
TG_STATE_LOAD_LONG_TIME_DRIFT_UNRESOLVED
```

Current preserved labels:

```text
TG_NODE_STATE_LOAD_SOURCE_SUPPORTED
TG_PHASE_TENSION_RELAXATION_SOURCE_SUPPORTED
TG_STATE_LOAD_FEEDFORWARD_CHAIN_SUPPORTED
TG_STATE_LOAD_GEOMETRIC_BACKREACTION_DETECTED
TG_STATE_LOAD_BACKREACTION_ROBUST
TG_STATE_LOAD_BACKREACTION_ROBUST_BUT_WEAK
TG_STATE_LOAD_CONTINUOUS_SLOW_DRIFT
```

The active D4 validation campaign is intended to determine whether the more precise label below becomes defensible:

```text
TG_STATE_LOAD_ORBITALLY_BOUNDED_FREQUENCY_SHIFT_SUPPORTED
```

That label is not yet promoted at the time of this handoff.

## Evidence Chain

### TG-S Source Semantics

Run directory:

```text
sweep_runs/TG_SOURCE_SEMANTICS_GPU_20260714_133513
```

Summary document:

```text
docs/gravity_maturity/TG_S_SUMMARY.json
```

Result:

- The stationary Q-ball baseline gate passed.
- `S_state` classified as `NODE_STATE_LOAD` and passed its semantics gate.
- `R_relax` classified as `PHASE_TENSION_RELAXATION` and passed semantics, but it remains disabled in the state-load branch.
- `L_lock` did not cleanly separate locking from null/breathing/unlocking controls.
- `P_threshold` remained comparison-only and cannot promote feedback by itself.

Key baseline/source facts:

- Stationary `S_state` integral: `135.6862187684289`.
- `S_state` was stable and nonzero on the stationary node.
- `S_state` was invariant under global phase and translation controls.

### TG-B1S State-Load Feed-Forward And Short Backreaction

Run directory:

```text
sweep_runs/TG_B1S_STATE_LOAD_GPU_20260714_140157
```

Summary document:

```text
docs/gravity_maturity/TG_B1S_SUMMARY.json
```

Result:

- A validated stationary Q-ball supplied the state load.
- Source-off and temporal-off removed downstream response fields.
- Geometric-off retained `T` while removing `G`.
- Global phase rotation did not alter the response.
- Translating the node translated the response.
- The source used one fixed global normalization, not case-wise normalization.

Key metrics:

- Q-ball residual: `1.1854848438084138e-09`.
- `|dE/E|`: `5.470133127026032e-14`.
- `|dQ/Q|`: `6.768139891839495e-14`.
- Profile overlap: `0.999999999999997`.
- `T_peak`: `0.00225097903078486`.
- `G_peak`: `0.0005703573852635597`.
- Full-loop minus feedback-off core energy: `-1.5745503789688087e-09`.
- Full-loop minus feedback-off node frequency: `4.778801965255042e-08`.

Labels supported at that stage:

```text
TG_STATE_LOAD_FEEDFORWARD_CHAIN_SUPPORTED
TG_STATE_LOAD_GEOMETRIC_BACKREACTION_DETECTED
```

### TG-B1S-R Backreaction Robustness

Run directory:

```text
sweep_runs/TG_B1S_BACKREACTION_ROBUSTNESS_GPU_20260714_142319
```

Summary document:

```text
docs/gravity_maturity/TG_B1S_R_SUMMARY.json
```

Result:

- The short backreaction exceeded the numerical floor.
- The effect vanished at zero feedback coupling.
- The lambda ladder was smooth over `lambda_fb in {0, 0.25, 0.5, 1, 1.5, 2}`.
- The detected effect was robust but weak.
- Bounded long-time feedback was not promoted.

Key metrics:

- Best SNR observable: `delta_core_amp_final`.
- Best SNR: `131.65175529252588`.
- `delta_modal_frequency` SNR: `5.520083225871324`.
- 10-period attractor class: `STABLE_SHIFTED_NODE`.
- 10-period modal phase drift: `-9.114930590925496e-05`.
- 10-period profile overlap: `0.9999999981230413`.

Labels supported:

```text
TG_STATE_LOAD_BACKREACTION_ROBUST
TG_STATE_LOAD_BACKREACTION_ROBUST_BUT_WEAK
```

### TG-B1S-D Drift Decomposition

Run directory:

```text
sweep_runs/TG_B1S_DRIFT_DECOMPOSITION_GPU_20260714_184736
```

Summary document:

```text
docs/gravity_maturity/TG_B1S_D_SUMMARY.json
```

Result:

- The prior 50-period `CONTINUOUS_SLOW_DRIFT` classification was decomposed into phase and shape channels.
- The 100-period primary run indicated a stable frequency-shift pattern, but formal promotion remained blocked because D4 numerical validation and D5 basin stability had not yet closed.

Primary 100-period metrics:

- Measured primary classification: `STABLE_FREQUENCY_SHIFT`.
- Fitted frequency shift: `-2.150936882840006e-06`.
- Final phase drift: `-0.0013329967322306402`.
- Final phase-aligned orbital distance: `4.9140079664143094e-05`.
- Profile overlap: `0.9999999987895385`.
- `delta_A_core`, `delta_width`, `delta_E_core`, orbital distance and modal leakage: `BOUNDED_OSCILLATORY`.
- `delta_Q`: below numerical floor.
- No phase-aligned structural channel showed secular growth in the primary run.

Formal status retained:

```text
TG_STATE_LOAD_LONG_TIME_DRIFT_UNRESOLVED
```

## D4 Closure Status — COMPLETE = FAILED (updated 2026-07-15, Claude primary review)

**Superseding note:** the section below was written mid-run (3/6 rows). D4 has since finished and did **not** close.
Formal label `TG_STATE_LOAD_D4_NUMERICAL_VALIDATION_FAILED`: 5/6 rows PASS, `D4_larger_box_50P` failed the
frequency-scale gate (rel-diff 0.6037 > 0.60). D5 remains blocked. The comprehensive synthesis — including the
isolating-variable finding (failure is box-size L, not resolution), the box-dependence discriminator (the local
mechanism predicts the *opposite* sign of change, so the drop is comparability/absorber/dynamical, not physical
box-fragility), and the recommended same-geometry-baseline closing test — is in:

```text
docs/gravity_maturity/TG_B1S_D4_FINAL_ANALYSIS.md
```

Supporting: `TG_B1S_D4_RESULTS.md` (Codex first pass), `TG_B1S_FREQUENCY_CONTRACT_RESULTS.md`,
`TG_B1S_BOX_DEPENDENCE_RESULTS.md`. The mid-run detail below is retained as history.

Active run directory:

```text
sweep_runs/TG_B1S_D4_ROWS_GPU_20260715_091558
```

Launch context:

- WSL/JAX GPU environment.
- Script: `jax_scout/gravity_TG_B1S_D4_rows_gpu.py`.
- Shared diagnostic helper patched in `jax_scout/gravity_TG_B1S_drift_decomposition_gpu.py`.
- Baseline row imported from `sweep_runs/TG_B1S_D4_D5_CLOSURE_GPU_20260715_003411`.

Rows complete at the time of this handoff:

```text
D4_baseline_50P          PASS
D4_output_cadence_50P   PASS
D4_dt_half_50P          PASS
```

Rows still pending at the last status check:

```text
D4_grid_refined_50P
D4_larger_box_50P
D4_absorber_wider_50P
```

Completed frequency checks:

| row | fitted delta omega |
| --- | ---: |
| reference 100P | `-2.150936882840006e-06` |
| D4 baseline 50P | `-2.162695983559886e-06` |
| D4 output cadence 50P | `-2.1611703237875384e-06` |
| D4 dt/2 50P | `-2.162695983550959e-06` |

The completed rows preserve the sign and magnitude of the reference frequency shift. This is encouraging but not yet sufficient for promotion.

## Runtime Issue Found During D4

The D4 rows initially appeared GPU-light because the evolution was GPU-backed, but the live orbital/profile diagnostics were pulling full 3D fields back to NumPy at every checkpoint:

```text
np.asarray(full_state[0])
np.asarray(off_state[0])
```

That was not a CPU simulation, but it was a CPU-host transfer of the full field inside the checkpoint loop. For these coupled substrate runs, that overhead is large enough to stall or distort runtime.

Patch applied:

- Added `orbital_metrics_jax(...)` to `jax_scout/gravity_TG_B1S_drift_decomposition_gpu.py`.
- The live runner now computes phase alignment, COM deltas, modal overlap, profile distances and reference overlap as JAX reductions on the selected GPU.
- The host receives scalar diagnostics only.
- The old NumPy `orbital_metrics(...)` helper remains for offline/reference use.

Important nuance:

- The GPU live metric conservatively reports relative COM displacement but does not perform a CPU-side roll/search in the checkpoint loop.
- The D4 rows compare same-origin full/off states, so the phase-plus-translation distance is effectively the phase-aligned orbital distance for live closure purposes.
- The translated D5 basin row should use a translated modal reference and same-origin full/off comparison; it should not require host-side roll/search.

## What Claude Should Review First

1. Confirm whether the D4 rows run completed and inspect:

```text
sweep_runs/TG_B1S_D4_ROWS_GPU_20260715_091558/numerical_validation.csv
sweep_runs/TG_B1S_D4_ROWS_GPU_20260715_091558/asymptotic_frequency.csv
sweep_runs/TG_B1S_D4_ROWS_GPU_20260715_091558/structural_trends.csv
sweep_runs/TG_B1S_D4_ROWS_GPU_20260715_091558/attractor_classification.csv
```

2. If D4 passes all rows, run or review D5 basin-orbital stability before any promotion.

3. If D4 fails, use the smallest accurate label:

```text
TG_STATE_LOAD_D4_NUMERICAL_VALIDATION_FAILED
TG_STATE_LOAD_DRIFT_NUMERICALLY_INDUCED
TG_STATE_LOAD_STRUCTURAL_SLOW_DRIFT_CONFIRMED
TG_STATE_LOAD_LONG_TIME_DRIFT_UNRESOLVED
```

4. If D4 and D5 both pass, the precise promotion target is:

```text
TG_STATE_LOAD_ORBITALLY_BOUNDED_FREQUENCY_SHIFT_SUPPORTED
```

The broader label below should be secondary and used only if the complete phi/T/G boundedness, energy ledger, boundary and basin gates all pass:

```text
TG_STATE_LOAD_BOUNDED_FEEDBACK_SUPPORTED
```

## Scientific Boundary

Even a passing D4/D5 result would support only this bounded claim:

> A persistent node-state source, coupled through the tested T/G response model, produces a numerically converged and orbitally bounded shift in the node's modal frequency.

It would not establish:

- emergent gravity;
- gravitational time dilation;
- objective chronology;
- geodesic motion;
- universal free fall;
- photon emission;
- IRER validation.

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

**Also referenced by (same date or earlier):** [[RUN_QUEUE]]

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
