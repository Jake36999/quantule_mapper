# V3 VK Branch

Status: provisional until Claude review.

- Metric ID: `v3`
- Status: `RAN`
- Match-level candidate: `SAME FAMILY`
- Validation depth: `INTERNAL_DATA_BEHAVIOUR_CHECK`
- Behavioural result: `PASS`
- Formula target: `dQ/domega < 0 under the stated C3 Q-ball branch convention`
- No new simulation: `True`

## Fit Parameters

```json
{
  "dQ_domega_fit": -849.138404192748,
  "dQ_domega_reported": -849.1384041927649,
  "intercept": 869.8962280490451
}
```

## Fit Quality

```json
{
  "Q_max": 59.04310622445989,
  "Q_min": 48.911713893680094,
  "dQ_domega_fit": -849.138404192748,
  "monotone_decreasing": true,
  "monotonic_decreasing": true,
  "n_points": 4,
  "omega_max": 0.968,
  "omega_min": 0.956,
  "sign": "negative",
  "sign_pass": true
}
```

## Rows

| source | omega | mu | Q | residual |
| --- | --- | --- | --- | --- |
| sweep_runs/C3_EXACT_VK2/summary.json | 0.956 | 0.08606400000000003 | 59.04310622445989 | 5.219224004929012e-09 |
| sweep_runs/C3_EXACT_VK2/summary.json | 0.96 | 0.07840000000000003 | 53.85843595985838 | 1.5029628492782802e-09 |
| sweep_runs/C3_EXACT_VK2/summary.json | 0.964 | 0.0707040000000001 | 50.28707678448786 | 1.1854843981743674e-09 |
| sweep_runs/C3_EXACT_VK2/summary.json | 0.968 | 0.06297600000000003 | 48.911713893680094 | 6.629093127649136e-11 |

## Warnings

- None

Guardrail: this report is a first-pass metric preparation artifact, not a verdict change or a physical-correspondence claim.
