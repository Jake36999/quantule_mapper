# Gravity D Spatial Effective-Medium Dynamics Characterization

Status: `D_EFFECTIVE_MEDIUM_DYNAMICS_CHARACTERIZED`

This report documents the bounded GPU characterization of the confirmed Gravity D spatial effective-medium mechanism. It does not claim Newtonian gravity, relativistic geodesics, temporal lapse validation, universal free fall, IRER source confirmation, or production readiness.

## Operator And Exact Force Contract

The characterized mirror operator is:

```text
i d_t psi = -D div(N_B(x) grad psi)
```

with bounded coefficient profiles such as:

```text
N_B(x) = 1 / (1 + beta S_B(x))
```

Throughout this report, `N_B(x)` denotes a spatial coefficient, bounded kinetic coefficient, or effective-medium field. It is not a temporal lapse in the tested operator.

The exact momentum response used as the primary diagnostic is:

```text
d<P>/dt = -D integral grad(N_B) |grad psi|^2 dV
```

This identity is the reference model. All coarse-grained descriptions are lower-order approximations to this integrated operator law.

## Evidence Chain

Accepted labels entering this round:

- `D_SPATIAL_EFFECTIVE_MEDIUM_ATTRACTION_GPU_CONVERGED`
- `D_SPATIAL_EFFECTIVE_MEDIUM_ROBUSTNESS_FIRST_PASS_CLOSED`

Characterization run:

- `sweep_runs/GRAVITY_D_DYNAMICS_CHARACTERIZATION_GPU_20260713_221727/`
- GPU backend: `gpu`
- selected device: `cuda:0`
- JAX/JAXLIB: `0.10.2`
- x64: enabled
- git commit recorded by run: `9fc7785df19d24f4be29e5d90bad59ed993da87b`

Primary evidence files:

- `gpu_preflight.json`
- `environment_versions.json`
- `preregistered_matrix.json`
- `run_manifest.csv`
- `instantaneous_force_atlas.csv`
- `trajectory_metrics.csv`
- `model_comparison.csv`
- `coarse_grained_coefficients.csv`
- `curve_collapse_metrics.csv`
- `holdout_validation.csv`
- `numerical_validation.csv`
- `falsification_results.csv`
- `source_profile_metadata.csv`
- `artifact_hashes.csv`

## GPU Reproduction, Convergence, And Robustness Closure

The refined baseline reproduced the accepted inward force:

```text
N=96, L=30, dt=0.001, T=4
source=Gaussian rho_B^2, source width=1.5, N_min=0.5
probe r0=4, probe sigma=1, carrier k=0
initial radial force = -6.211596169843487e-03
```

The matched flat control gave zero analytic force and stable global/periodic COM diagnostics. The earlier raw flat drift of about `-1.3069e-3` was traced to Claude's fixed slab-window COM diagnostic, not to physical force, finite-box wrap, or coefficient mismatch.

Representative numerical validation:

| case | configs | force spread | classification changed | max norm error | max energy error |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 3 | 1.95e-15 | 0 | 4.44e-16 | 3.33e-16 |
| strong | 3 | 2.28e-15 | 0 | 2.22e-16 | 2.22e-16 |
| compact bump | 3 | 7.38e-12 | 0 | 4.44e-16 | 7.33e-15 |

The robustness closure established that genuinely distinct source shapes retain the directional response: Gaussian, super-Gaussian, compact bump, and two-lobe profiles all produced inward force when matched by `N_min` and integrated coefficient deficit.

## Force Atlas And Dynamical Regimes

The characterization atlas covered source families, strength ladder, distance ladder, probe-width ladder, carrier perturbations, shell symmetry cases, and two-lobe symmetry cases.

Trajectory classifications included:

- `FREE_LIKE` far-field near-null and two-lobe midpoint cases.
- `WEAK_DEFLECTION` weak-strength and shell-exterior cases.
- `STRONG_DEFLECTION` baseline, strong, near-source, carrier, and two-lobe off-axis cases.
- `TURNING_POINT` shell-interior standard and long cases.

The two-lobe midpoint stayed free-like through `T=12`, while the off-axis two-lobe case gave strong inward response. This supports the interpretation that the force tracks coefficient-gradient geometry rather than grid origin or source label.

## Coarse-Grained Model

The best reduced model was the gradient-energy-weighted coefficient-gradient approximation:

```text
F_cg ~= -D K_grad grad N(R)
```

In the fitted scalar form from `model_comparison.csv`:

```text
F ~= c (-K_grad grad_N_at_COM)
c = 0.2800836375056448
normalized RMSE = 0.09307384958591862
R2 = 0.9913372585232578
radial sign accuracy = 0.9533678756476683
```

This supports `D_GRADIENT_ENERGY_WEIGHTED_FORCE_SUPPORTED`, but the exact integrated force remains the authoritative model:

```text
F = -D integral grad(N) |grad psi|^2 dV
```

Reduced-model sign-error examination:

- `reduced_model_sign_errors.csv`
- `REDUCED_MODEL_SIGN_ERROR_ANALYSIS.md`

There were 6 sign errors out of 193 non-flat atlas rows. They were all small-magnitude far-tail or near-null cases, concentrated in compact-width super-Gaussian far-field rows plus one shell far-exterior row and one very weak Gaussian far-tail row. This preserves `D_REDUCED_MODEL_PARTIAL`; the local COM approximation is useful, not universal.

## Model Comparisons

| model | normalized RMSE | R2 | sign accuracy | interpretation |
| --- | ---: | ---: | ---: | --- |
| `M0_Kgrad_gradN_reduced` | 0.0931 | 0.9913 | 0.9534 | best coarse-grained model |
| `M2_log_gradient` | 0.6392 | 0.5914 | 0.9534 | weaker empirical predictor |
| `M1_local_gradient` | 0.6958 | 0.5158 | 0.9534 | weaker empirical predictor |
| `M4_source_tail` | 0.7137 | 0.4907 | 0.9637 | source-tail correlation, not sufficient |
| `M5_newtonian` | 0.9897 | 0.0206 | 0.9482 | poor inverse-square benchmark |
| `M3_ray` | 1.0528 | -0.1085 | 0.1088 | poor global point-ray description |

The position-dependent-mass/ray picture may still be a limited high-carrier, compact-packet analogy, but the global data are better described as a finite-width wave-medium force weighted by gradient energy.

## Falsification Results

| test | result |
| --- | --- |
| probe internal structure | structure-dependent, force spread about 54.6% |
| shell interior | `LOCAL_MEDIUM_SHELL_RESPONSE`, interior max force about 3.09e-02 |
| compact-source exterior decay | far exterior force about 1.66e-13 |
| constant-N source-free background | zero force |
| weak-field superposition | quantified but not promoted |

Rejected for this spatial model:

- Newtonian exterior vacuum field.
- Newtonian shell-theorem behaviour.
- Universal free fall.
- Global point-ray dynamics.
- Relativistic geodesic validation.

## Bounded Labels

Supported:

- `D_EFFECTIVE_MEDIUM_DYNAMICS_CHARACTERIZED`
- `D_GRADIENT_ENERGY_WEIGHTED_FORCE_SUPPORTED`
- `D_FINITE_WIDTH_WAVE_FORCE_CONFIRMED`
- `D_NEWTONIAN_COMPARISON_REJECTED`
- `D_NO_UNIVERSAL_FREE_FALL_LIMIT_FOUND`
- `D_REDUCED_MODEL_PARTIAL`

Not assigned:

- `GRAVITY_CONFIRMED`
- `GEODESIC_CONFIRMED`
- `IRER_GRAVITY_VALIDATED`
- `EQUIVALENCE_PRINCIPLE_CONFIRMED`
- `PRODUCTION_READY`

## Interpretation

Evidence: the exact operator-force identity closes, the GPU baseline and controls reproduce, convergence and robustness checks are stable, and the force atlas is best captured by a gradient-energy-weighted medium-force law.

Inference: the phenomenon is a robust finite-width wave response in a bounded spatial coefficient field.

Speculation: the mechanism may be useful as an analogue-medium or position-dependent-kinetic-coefficient model, but a temporal lapse or nonlocal IRER source-field branch would be a separate hypothesis.

Rejected interpretations: the characterized spatial operator is not Newtonian gravity, not a relativistic geodesic result, not universal free fall, and not an IRER gravity source.

Proposed next action: if the project wants a gravity-facing branch, test a separate temporal-metric or nonlocal source-field implementation with the same force-contract discipline and paired null controls.

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `25020bf` (2026-07-22) — *dual substrate*
**Revised since:** 8 commit(s), most recently `bc5b54c` (2026-08-31)

**Harness code changed since it was written:** 5 commit(s) to `jax_scout/`.
  - `06edae5` 2026-08-27 — Fix run provenance: 30% -> 86% of runs now dated
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine
  - `25020bf` 2026-07-22 — dual substrate

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Also referenced by (same date or earlier):** [[GRAVITY_AUDIT_D_PROBE_RESULTS]], [[IRER_GRAVITY_REENTRY_AUDIT_AND_VALIDATION_PLAN]], [[RUN_QUEUE]], [[gravity_maturity/TG_TWO_NODE_SIGN_CHAIN_DESIGN_CONTRACT]]

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
