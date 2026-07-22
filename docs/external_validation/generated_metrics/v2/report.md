# V2 Galilean Transport

Status: provisional until Claude review.

- Metric ID: `v2`
- Status: `RAN`
- Match-level candidate: `EXACT EQUATION`
- Validation depth: `INTERNAL_DATA_BEHAVIOUR_CHECK`
- Behavioural result: `PASS`
- Formula target: `v = 2Dk for corrected flat-NLS C2 transport`
- No new simulation: `True`

## Fit Parameters

```json
{
  "expected_slope_2D": 2.0,
  "intercept": -9.986275745059968e-08,
  "r2": 1.0,
  "slope": 1.9997819749017656,
  "slope_relative_error": 0.00010901254911721558
}
```

## Fit Quality

```json
{
  "D_used": 1.0,
  "expected_slope_2D": 2.0,
  "intercept": -9.986275745059968e-08,
  "mass_retention_min": 0.9996329889054258,
  "mean_mass_retention": 0.999754131719139,
  "n_boosts": 2,
  "n_points": 2,
  "r_squared": 1.0,
  "slope": 1.9997819749017656,
  "slope_percent_error": 0.010901254911721558,
  "source_of_D": "project C2.7 convention or explicit summary field if present"
}
```

## Rows

| source | n | k | v_measured | v_expected_2Dk | v_frac | mass_ret |
| --- | --- | --- | --- | --- | --- | --- |
| sweep_runs/C27_REDERIVE/r3_n96.json | 1 | 0.6283185307179586 | 1.256499972363777 | 1.2566370614359172 | 0.999890907982625 | 0.999875274532852 |
| sweep_runs/C27_REDERIVE/r3_n96.json | 2 | 1.2566370614359172 | 2.5130000445903122 | 2.5132741228718345 | 0.999890947716754 | 0.9996329889054258 |

## Warnings

- sweep_runs/C27_REDERIVE/r3_n96.json lacks explicit D; using project C2.7 convention D=1.0.

Guardrail: this report is a first-pass metric preparation artifact, not a verdict change or a physical-correspondence claim.
