---
run_id: "GRAVITY_D_DYNAMICS_CHARACTERIZATION_GPU_20260713_221727"
date: 2026-07-13
family: "Gravity-D"
sector: "gravity"
verdict: "D_REDUCED_MODEL_PARTIAL"
complete: false
source: derived
n_csv: 17
n_plots: 5
tags: [run, gravity, Gravity_D]
---
# GRAVITY_D_DYNAMICS_CHARACTERIZATION_GPU_20260713_221727

*Spatial-geometry gravity mirror (non-Newtonian)* &middot; **Gravity-D** &middot; `2026-07-13`

> [!abstract] Verdict
> `D_REDUCED_MODEL_PARTIAL`

> [!info] Derived note - no `summary.json`
> This run predates or bypasses the `summary.json` convention. Fields below are reconstructed from: `TECHNICAL_HANDOFF.md`, `REDUCED_MODEL_SIGN_ERROR_ANALYSIS.md`, `OPEN_QUESTIONS.md`, `DOCUMENTATION_INPUTS.md`, `preregistered_matrix.json`.
> The verdict shown is `RUN_COMPLETE.json.status` where present, otherwise the first verdict-shaped token found in the run's own report - **treat it as indicative and confirm against the source document.**

## From `TECHNICAL_HANDOFF.md`

This run characterizes the confirmed spatial effective-medium operator only.


- GPU preflight, environment versions, matrix, CSV tables, plots, and hashes are in this run directory.
- The exact operator-force model remains the reference observable.
- Global and periodic COM are primary; slab COM is diagnostic only.


- Use the bounded labels in CHARACTERIZATION_SUMMARY.json.
- Treat source, width, and carrier dependencies as characterization of a finite-width wave-medium force.


- Do not describe this as gravity, temporal lapse, geodesic motion, universal free fall, or IRER confirmation.

## From `REDUCED_MODEL_SIGN_ERROR_ANALYSIS.md`

Model examined: `F_cg = c * (-K_grad * grad_N_at_COM)`, with `c = 0.2800836375056448` from `model_comparison.csv`.

Sign-error count: 6 out of 193 non-flat atlas rows.

All sign errors are small-magnitude far-tail or near-null cases. They are concentrated in compact-width super-Gaussian far-field rows, plus one shell far-exterior row and one strong Gaussian far-tail row. This supports the interpretation that the coarse-grained local-COM approximation is excellent in resolved-gradient regions but can miss sign in weak tail regions where the integrated exact force is tiny and finite-width weighting samples nonlocal gradients.

| run_id | source | N_min | r | sigma | exact Fr | predicted Fr |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| E1_core_supergaussian4_r4p0_w0p25_n0p5 | supergaussian4 | 0.5 | 6.52561530883978 | 0.407850956802486 | 2.43705519701282E-05 | -1.16407703107332E-05 |

## From `OPEN_QUESTIONS.md`

- Whether a temporal-metric implementation produces compatible or distinct trajectories.
- Whether a physically derived IRER environment can generate the bounded coefficient field.
- Whether higher-resolution compact-packet ray limits improve M3 agreement.
- Whether long-duration bound/capture candidates survive stricter boundary and convergence tests.

## From `DOCUMENTATION_INPUTS.md`

- `D_EFFECTIVE_MEDIUM_DYNAMICS_CHARACTERIZED`
- `D_FINITE_WIDTH_WAVE_FORCE_CONFIRMED`
- `D_GRADIENT_ENERGY_WEIGHTED_FORCE_SUPPORTED`
- `D_NEWTONIAN_COMPARISON_REJECTED`
- `D_NO_UNIVERSAL_FREE_FALL_LIMIT_FOUND`
- `D_REDUCED_MODEL_PARTIAL`


- `instantaneous_force_atlas.csv`
- `trajectory_metrics.csv`
- `model_comparison.csv`
- `holdout_validation.csv`
- `falsification_results.csv`
- `numerical_validation.csv`


- This is a spatial effective-medium result only.
- Probe-width and internal-structure dependence must not be relabeled as universal free fall.
- Newtonian and geodesic-like models are benchmarks, not promoted interpretations.


The confirmed divergence-form spatial operator is best interpreted as a finite-width wave-medium interaction whose force is governed by the gradient-energy-weighted coefficient gradient.

## Plots

*Generated from this run's CSVs by `tools/build_run_catalogue.py`. Regenerable — delete and rebuild to refresh.*

![[_plots/GRAVITY_D_DYNAMICS_CHARACTERIZATION_GPU_20260713_221727/coarse_grained_coefficients.png]]

*coarse grained coefficients*

![[_plots/GRAVITY_D_DYNAMICS_CHARACTERIZATION_GPU_20260713_221727/curve_collapse_metrics.png]]

*curve collapse metrics*

![[_plots/GRAVITY_D_DYNAMICS_CHARACTERIZATION_GPU_20260713_221727/documentation_round_hashes.png]]

*documentation round hashes*

![[_plots/GRAVITY_D_DYNAMICS_CHARACTERIZATION_GPU_20260713_221727/falsification_raw.png]]

*falsification raw*

![[_plots/GRAVITY_D_DYNAMICS_CHARACTERIZATION_GPU_20260713_221727/instantaneous_force_atlas.png]]

*instantaneous force atlas*

## Figures (produced by the run itself)

*Copied from the run directory so Obsidian can display them. These are the run's own rendered output, not regenerated here.*

![[_figures/GRAVITY_D_DYNAMICS_CHARACTERIZATION_GPU_20260713_221727/plots__effective_potential_by_probe_family.svg]]

*`plots/effective_potential_by_probe_family.svg`*

![[_figures/GRAVITY_D_DYNAMICS_CHARACTERIZATION_GPU_20260713_221727/plots__force_over_Kgrad_vs_gradN.svg]]

*`plots/force_over_Kgrad_vs_gradN.svg`*

![[_figures/GRAVITY_D_DYNAMICS_CHARACTERIZATION_GPU_20260713_221727/plots__force_vs_distance_by_source.svg]]

*`plots/force_vs_distance_by_source.svg`*

![[_figures/GRAVITY_D_DYNAMICS_CHARACTERIZATION_GPU_20260713_221727/plots__force_vs_probe_width.svg]]

*`plots/force_vs_probe_width.svg`*

![[_figures/GRAVITY_D_DYNAMICS_CHARACTERIZATION_GPU_20260713_221727/plots__force_vs_strength.svg]]

*`plots/force_vs_strength.svg`*

![[_figures/GRAVITY_D_DYNAMICS_CHARACTERIZATION_GPU_20260713_221727/plots__holdout_prediction_error.svg]]

*`plots/holdout_prediction_error.svg`*

![[_figures/GRAVITY_D_DYNAMICS_CHARACTERIZATION_GPU_20260713_221727/plots__moment_expansion_error_vs_width.svg]]

*`plots/moment_expansion_error_vs_width.svg`*

![[_figures/GRAVITY_D_DYNAMICS_CHARACTERIZATION_GPU_20260713_221727/plots__numerical_convergence.svg]]

*`plots/numerical_convergence.svg`*

![[_figures/GRAVITY_D_DYNAMICS_CHARACTERIZATION_GPU_20260713_221727/plots__ray_vs_wave_trajectories.svg]]

*`plots/ray_vs_wave_trajectories.svg`*

![[_figures/GRAVITY_D_DYNAMICS_CHARACTERIZATION_GPU_20260713_221727/plots__regime_map_distance_vs_carrier.svg]]

*`plots/regime_map_distance_vs_carrier.svg`*

![[_figures/GRAVITY_D_DYNAMICS_CHARACTERIZATION_GPU_20260713_221727/plots__regime_map_strength_vs_width.svg]]

*`plots/regime_map_strength_vs_width.svg`*

![[_figures/GRAVITY_D_DYNAMICS_CHARACTERIZATION_GPU_20260713_221727/plots__shell_interior_force.svg]]

*`plots/shell_interior_force.svg`*

![[_figures/GRAVITY_D_DYNAMICS_CHARACTERIZATION_GPU_20260713_221727/plots__two_source_superposition_error.svg]]

*`plots/two_source_superposition_error.svg`*

## Artifacts

- **Run directory** (gitignored, local only): `sweep_runs/GRAVITY_D_DYNAMICS_CHARACTERIZATION_GPU_20260713_221727/`
- **CSV data** (17): `artifact_hashes.csv`, `coarse_grained_coefficients.csv`, `curve_collapse_metrics.csv`, `documentation_round_hashes.csv`, `falsification_raw.csv`, `falsification_results.csv`, `holdout_validation.csv`, `instantaneous_force_atlas.csv`, `model_comparison.csv`, `numerical_validation.csv` ...

## Previous experiments

*Auto-derived: the preceding runs in the same family (`Gravity-D`). Correct by hand if the lineage is wrong — edits here survive rebuilds only if you move the link below the Review notes marker.*

- [[GRAVITY_C_CLOCK_20260713_001146]] &middot; `2026-07-13`
- [[GRAVITY_C3_BACKREACT_20260713_104631]] &middot; `2026-07-13`
- [[GRAVITY_C2_20260713_104416]] &middot; `2026-07-13`

## Associated docs

- [[Branch - Gravity - Index]]
- [[EXPERIMENT_TRACKER]]
- [[GRAVITY_D_EFFECTIVE_MEDIUM_DYNAMICS_CHARACTERIZATION]]
- [[evidence_package/00_project_timeline/PROJECT_MATURITY_TIMELINE]]
- [[external_validation/CODEX_EXTERNAL_VALIDATION_REPORT]]
- [[external_validation/GRAVITY_D_EFFECTIVE_MEDIUM_EXTERNAL_VALIDATION_PACKAGE]]
- [[external_validation/artifact_bundles/gravity_d_effective_medium_characterization/README]]
- [[external_validation/artifact_bundles/gravity_d_effective_medium_characterization/docs/CODEX_EXTERNAL_VALIDATION_REPORT]]

## Next experiment

- [[GRAVITY_D_DYNAMICS_CHARACTERIZATION_GPU_E0_SMOKE_20260713_220905]] &middot; `2026-07-13`

## Branches

- [[Main branch]] &larr; via [[Branch - Gravity - Index|gravity sector index]]
- Sector: `gravity` &middot; family: `Gravity-D`

---

## Review notes

*(Preserved across catalogue rebuilds - everything above this line is regenerated.)*

> [!important] Paired reading - fill your block before reading the other one.
> A disagreement here is the point, not a problem. Record the resolution; do not
> overwrite the disagreement.

### Reading - Claude

*(filled 2026-09-17, plan item C2. Read from the run's CSV tables and `REDUCED_MODEL_SIGN_ERROR_ANALYSIS.md`
rather than primarily from the plots -- the substance here is tabular. Flagged so the reading is not
taken as a visual review it is not.)*

- **Observation:** The run's own sign analysis reports **6 sign errors in 193 non-flat atlas rows**
  for the coarse-grained reduced model `F_cg = c * (-K_grad * grad_N_at_COM)`. They are not
  scattered: every one is a **far-tail or near-null** case, concentrated in compact-width
  super-Gaussian far-field rows plus one shell far-exterior and one Gaussian far-tail row. The exact
  forces involved are tiny (2.4e-05 down to 1.5e-09) and in several rows the predicted magnitude is
  *smaller* than the exact one by one to two orders (e.g. shell far-exterior: exact 7.61e-07,
  predicted -2.27e-08).

- **Reading:** This is a **locality failure, not a sign ambiguity**, and today's
  [[../gravity_maturity/S2_V7_ACOUSTIC_METRIC_RESULTS|S2]] result says why in structural terms.

  S2 showed the exact body force is the **geodesic** of the reconstructed acoustic metric, with
  acceleration `-(3/4) grad A / A` -- a *pointwise* law. The reduced model evaluates exactly that
  kind of local gradient, but **at the centre of mass**, while the exact force is an *integral over
  the extended body* weighted by `|grad phi|^2`. Wherever the body is wide compared with the scale
  over which the gradient varies, sampling the gradient at one point is not the same as integrating
  it -- and in the far tail the integrand changes sign across the body, so the COM sample can land
  on the wrong side.

  That predicts exactly the observed pattern: failures confined to far-field and weak-gradient rows,
  worst for **compact-width super-Gaussian** sources, which have the sharpest profile edges and so
  the most gradient variation across the body.

  It also means the reduced model is not rescuable by fixing a coefficient. The correct local law
  would need the weighted centroid of `|grad phi|^2`, not the COM of the density -- and
  [[../gravity_maturity/TG_P1A_ENERGY_OBSERVABLE|P1-a]] already established those differ, since the
  gradient term is only **4.5%** of the node's energy and its fraction varies 29% non-monotonically
  across the mass axis.

  The verdict `D_REDUCED_MODEL_PARTIAL` and the instruction "do not promote the local coarse-grained
  law as universal" are both correct and should stand.

- **Confidence:** High that the failures are a locality/weighting artefact rather than a sign
  ambiguity in the physics -- the pattern (far-tail only, sharp-edged sources worst) is what that
  explanation predicts, and it was recorded before S2 gave the reason. Medium on the specific claim
  that a `|grad phi|^2`-weighted centroid would fix it; that is a prediction, not a measurement.

- **What would change my mind:** Sign errors appearing in **resolved-gradient, near-field** rows,
  which locality cannot explain. Or a demonstration that re-deriving `F_cg` about the
  `|grad phi|^2`-weighted centroid leaves the 6 failures intact -- that would mean the discrepancy
  is not about where the gradient is sampled.

> [!note] Cheap follow-up this suggests
> Recompute the 6 failing rows with the gradient sampled at the `|grad phi|^2`-weighted centroid
> instead of the density COM. The atlas already exists, so this is a re-analysis with no new
> simulation -- the same shape as [[../gravity_maturity/TG_P1A_ENERGY_OBSERVABLE|P1-a]] and the
> [[../gravity_maturity/P1B_FROZEN_REFERENCE_REANALYSIS|P1-b reanalysis]].

### Reading - Jake

- **Observation:**
- **Reading:**
- **Confidence:**
- **What would change my mind:**

### Comparison

**Agree on:**

**Disagree on:**

**Resolution:**

