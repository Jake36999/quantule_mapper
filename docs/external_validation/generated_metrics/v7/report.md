# V7 Omega Exponent Diagnosis

Status: provisional until Claude review.

- Metric ID: `v7`
- Status: `RAN`
- Match-level candidate: `CLOSE ANALOGUE`
- Validation depth: `INTERNAL_DATA_BEHAVIOUR_CHECK`
- Behavioural result: `PARTIAL`
- Formula target: `Omega^2 = (rho_vac / rho)^a, gamma = 2a + 3; design-constraint diagnosis only`
- No new simulation: `True`

## Fit Parameters

```json
{
  "a_effective": 2.9259806516213103,
  "cap_fraction": 0.3090277777777778,
  "fit_available": true,
  "floor_fraction": 0.003472222222222222,
  "gamma_effective": 8.851961303242621,
  "graded_fraction": 0.6875,
  "log_intercept": 6.66156140796944,
  "n_fit": 198
}
```

## Fit Quality

```json
{
  "bec_reference_a": -0.5,
  "bec_reference_gamma": 2.0,
  "cap_fraction": 0.3090277777777778,
  "cliff_detected": true,
  "design_constraint_only": true,
  "effective_gamma": 8.851961303242621,
  "effective_steeper_than_nominal": true,
  "fitted_effective_exponent": 2.9259806516213103,
  "floor_fraction": 0.003472222222222222,
  "graded_fraction": 0.6875,
  "implied_gamma": 7.6196,
  "n_radial_points": 288,
  "nominal_a": 2.3098,
  "nominal_gamma": 7.6196,
  "production_a": 2.3098,
  "rows": 3,
  "same_sign_as_bec_analogue": false,
  "unsaturated_points": 198
}
```

## Rows

| source | profile | omega_min | omega_max | cap_like_fraction |
| --- | --- | --- | --- | --- |
| sweep_runs/GRAVITY_AD_bg0/summary.json | radial_omega_only | 711.498 | 935906.42 | 0.7083333333333334 |
| sweep_runs/GRAVITY_DESAT_PILOT/desat_pilot.json | production_desat_summary |  |  |  |
| sweep_runs/PHASE_D_CODEX_REPRODUCTION_20260709_233433/gravity_geometry/radial_profiles.csv |  |  |  |  |

## Warnings

- None

Guardrail: this report is a first-pass metric preparation artifact, not a verdict change or a physical-correspondence claim.
