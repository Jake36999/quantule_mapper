# TG-B1S D4 Numerical Validation Results

Author: Codex  
Timestamp: 2026-07-15  
Run directory: `sweep_runs/TG_B1S_D4_ROWS_GPU_20260715_091558`

## Verdict

```text
TG_STATE_LOAD_D4_NUMERICAL_VALIDATION_FAILED
```

D4 completed, but it did not pass the preregistered numerical-validation gate. Five of six rows passed. The larger-box row failed the frequency-scale tolerance.

This blocks D5 and blocks any promotion to:

```text
TG_STATE_LOAD_ORBITALLY_BOUNDED_FREQUENCY_SHIFT_SUPPORTED
TG_STATE_LOAD_BOUNDED_FEEDBACK_SUPPORTED
```

## GPU Evidence

The run saved GPU preflight in:

```text
sweep_runs/TG_B1S_D4_ROWS_GPU_20260715_091558/gpu_preflight.json
```

Recorded environment:

- backend: `gpu`
- device: `cuda:0`
- JAX: `0.10.2`
- jaxlib: `0.10.2`
- x64 enabled: `true`
- WSL2 Linux platform
- git commit: `9fc7785df19d24f4be29e5d90bad59ed993da87b`

## Completed Rows

| row | result | fitted `delta_omega_infty` | relative frequency difference | early/late slope diff | note |
| --- | --- | ---: | ---: | ---: | --- |
| `D4_baseline_50P` | PASS | `-2.162695983559886e-06` | `0.005466966889494886` | `0.300788079666371` | imported baseline row |
| `D4_output_cadence_50P` | PASS | `-2.1611703237875384e-06` | `0.00475766677728843` | `0.30653728059865926` | cadence check |
| `D4_dt_half_50P` | PASS | `-2.162695983550959e-06` | `0.005466966885344657` | `0.30078807962155246` | timestep-halved check |
| `D4_grid_refined_50P` | PASS | `-2.1626959888870264e-06` | `0.00546696936615545` | `0.30078807168936356` | refined grid check |
| `D4_larger_box_50P` | FAIL | `-8.524302969263542e-07` | `0.6036934864398061` | `0.0014225347910343498` | failed `frequency_scale` gate |
| `D4_absorber_wider_50P` | PASS | `-2.162694528549176e-06` | `0.005466290435098957` | `0.30079046380316765` | absorber check |

Reference `delta_omega_infty`:

```text
-2.150936882840006e-06
```

Preregistered relative frequency tolerance:

```text
<= 0.60
```

The larger-box row returned:

```text
relative_frequency_difference = 0.6036934864398061
```

which is just beyond the preregistered tolerance. It must be treated as a failed gate, not rounded into a pass.

## What Passed

The following rows preserve the frequency-shift sign and approximate scale:

```text
D4_baseline_50P
D4_output_cadence_50P
D4_dt_half_50P
D4_grid_refined_50P
D4_absorber_wider_50P
```

For those rows:

- no structural channel was reported as secular;
- phase-aligned orbital distances remained below gate;
- profile overlap remained high;
- modal leakage remained below gate;
- ledger residual remained below gate;
- boundary flux proxy remained below gate;
- T/G fields classified as `APPROACH_FIXED_PROFILE`.

## What Failed

The row:

```text
D4_larger_box_50P
```

failed only the frequency-scale validation. Other diagnostics in that row remained clean:

- early/late relative slope difference: `0.0014225347910343498`
- final phase-translation-aligned distance: `1.3480312131603541e-05`
- max phase-translation-aligned distance: `3.3807384577917285e-05`
- profile overlap: `0.9999999999041367`
- modal leakage max: `3.380153409364178e-05`
- ledger residual: `1.090697469063052e-05`
- boundary flux proxy max: `6.329027677419215e-08`
- T/G class: `APPROACH_FIXED_PROFILE`

This pattern suggests a frequency-scale discrepancy rather than loss of boundedness, boundary contamination, structural drift or solver instability.

## Interpretation

Evidence:

```text
The frequency-shift sign and scale are stable under timestep halving,
output-cadence change, grid refinement and absorber widening, but not under
the larger-box row as configured.
```

Inference:

```text
D4 does not close. The current frozen-model frequency-shift claim remains
numerically promising but formally unresolved under the preregistered gates.
```

Open question:

```text
Does the larger-box discrepancy reflect a physical box/normalization effect,
a changed Q-ball/profile branch, a diagnostic/reference mismatch, or a
legitimate falsification of box robustness?
```

No parameter tuning or tolerance relaxation is justified by this result.

## Next Step

Before D5:

```text
Review the D4_larger_box_50P discrepancy.
```

Claude's FC-box / SCREEN-1 discriminator subsequently showed that the analytic local screened-stiffening mechanism predicts the L=12 shift should increase, not decrease. Codex also verified that the D4 row already computed the larger-box shift from same-geometry full-loop and feedback-off trajectories. The remaining review is therefore not a simple missing subtraction; it is a comparability/dynamic-measurement audit of why the L=12 same-geometry full-minus-off shift is smaller than the L=10 reference.

Recommended review should be analysis-first:

- compare the Q-ball solve/profile between baseline and larger-box row;
- compare norm, charge, profile width and `S_state` integral;
- check whether the larger-box row changed the effective stationary branch;
- compare fixed physical support and local grid resolution;
- check whether the frequency observable was referenced against a non-comparable baseline;
- do not rerun with altered parameters until the discrepancy is explained or explicitly accepted as failure.

## Boundaries

This result does not support:

- bounded feedback promotion;
- gravity;
- objective chronology;
- geodesic motion;
- universal free fall;
- photon emission;
- IRER validation;
- production readiness.

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `25020bf` (2026-07-22) — *dual substrate*
**Revised since:** 5 commit(s), most recently `09f47d8` (2026-08-27)

**Harness code changed since it was written:** 4 commit(s) to `jax_scout/`.
  - `06edae5` 2026-08-27 — Fix run provenance: 30% -> 86% of runs now dated
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Also referenced by (same date or earlier):** [[RUN_QUEUE]], [[gravity_maturity/CLAUDE_HANDOFF_20260715_TG_B1S_AND_RUNTIME]], [[gravity_maturity/TG_B1S_D4_FINAL_ANALYSIS]]

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
