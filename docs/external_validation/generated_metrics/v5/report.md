# V5 C3 Collision Phase Grid

Status: provisional until Claude review.

- Metric ID: `v5`
- Status: `RAN`
- Match-level candidate: `CLOSE ANALOGUE`
- Validation depth: `INTERNAL_DATA_BEHAVIOUR_CHECK`
- Behavioural result: `PASS`
- Formula target: `phase x speed collision outcome grid with conservation/radiation telemetry where available`
- No new simulation: `True`

## Fit Parameters

```json
{
  "phase_count": 5,
  "speed_count": 5
}
```

## Fit Quality

```json
{
  "anti_phase_transmission_cells": 3,
  "max_charge_drift": 1.489717348076215e-06,
  "max_dE_rel": 6.588342536583086e-06,
  "max_dQ_rel": 1.489717348076215e-06,
  "max_energy_drift": 6.588342536583086e-06,
  "missing_phase_speed_cells": 9,
  "n_cells": 16,
  "n_missing_cells": 9,
  "non_antiphase_capture_fraction": 1.0,
  "outcome_counts": {
    "CAPTURE": 13,
    "PASS_THROUGH": 3
  },
  "outcomes": [
    "CAPTURE",
    "PASS_THROUGH"
  ],
  "phase_values": [
    "0.00",
    "1.57",
    "2.36",
    "2.75",
    "3.14"
  ],
  "rows": 16,
  "speed_values": [
    "0.15",
    "0.3",
    "0.45",
    "0.6",
    "0.75"
  ]
}
```

## Rows

| source | phase | vfrac | speed | outcome | reason | radiation_fraction | mass_retention | dE_rel_max | dQ_rel_max | elasticity | identity_ambiguous |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sweep_runs/C3_COLLISION_LADDER_FULL/summary.json | 0.00 | 0.45 | 0.246465 | CAPTURE | cores merged & bound (sep_min=0.11, sep_end=0.23) | 0.5961 | 0.9852 | 2.2809128470644162e-07 | 1.847009639918737e-08 | nan | True |
| sweep_runs/C3_COLLISION_LADDER_FULL/summary.json | 0.00 | 0.6 | 0.32861999999999997 | CAPTURE | cores merged & bound (sep_min=0.02, sep_end=0.32) | 0.7864 | 0.987 | 4.262417139433944e-07 | 4.5814494729775894e-08 | nan | True |
| sweep_runs/C3_COLLISION_LADDER_FULL/summary.json | 0.00 | 0.75 | 0.410775 | CAPTURE | cores merged & bound (sep_min=0.03, sep_end=0.28) | 0.9101 | 0.9952 | 5.606232512315461e-06 | 9.153927477078283e-07 | nan | True |
| sweep_runs/C3_ANTIPHASE_LADDER/summary.json | 3.14 | 0.15 | 0.08215499999999999 | PASS_THROUGH | two coherent cores overlapped then re-separated (sep_min=5.71->end 6.88, vout_rel=+0.084 vs 2v_in=0.164, rad=0.02) | 0.0211 | 1.0004 | 1.0333643588934839e-07 | 8.75381032416308e-09 | 0.5132684743424276 | True |
| sweep_runs/C3_ANTIPHASE_LADDER/summary.json | 3.14 | 0.3 | 0.16430999999999998 | PASS_THROUGH | two coherent cores overlapped then re-separated (sep_min=0.51->end 9.45, vout_rel=+0.298 vs 2v_in=0.329, rad=0.07) | 0.0695 | 0.9993 | 2.1822251945453044e-07 | 1.6200408832542434e-08 | 0.9061722018537308 | True |
| sweep_runs/C3_ANTIPHASE_LADDER/summary.json | 3.14 | 0.45 | 0.246465 | PASS_THROUGH | two coherent cores overlapped then re-separated (sep_min=0.01->end 8.54, vout_rel=+0.247 vs 2v_in=0.493, rad=0.08) | 0.076 | 0.996 | 1.1938013400270684e-07 | 1.0798126201380554e-08 | 0.5019384203147882 | True |
| sweep_runs/C3_ANTIPHASE_LADDER/summary.json | 3.14 | 0.6 | 0.32861999999999997 | CAPTURE | overlapped & stayed bound (sep_min=0.05, sep_end=5.61, rad=0.05) | 0.0464 | 0.9923 | 6.988338293879637e-07 | 1.2098745018793675e-07 | 0.10580388940635384 | True |
| sweep_runs/C3_ANTIPHASE_LADDER/summary.json | 3.14 | 0.75 | 0.410775 | CAPTURE | overlapped & stayed bound (sep_min=0.00, sep_end=0.06, rad=0.10) | 0.1017 | 0.9919 | 6.588342536583086e-06 | 1.489717348076215e-06 | 0.07981898798039112 | True |
| sweep_runs/C3_PHASE_pi2/summary.json | 1.57 | 0.3 | 0.16430999999999998 | CAPTURE | overlapped & stayed bound (sep_min=0.00, sep_end=0.00, rad=0.42) | 0.4195 | 0.9964 | 2.4242653753661074e-07 | 1.1072379527060577e-08 | nan | True |
| sweep_runs/C3_PHASE_pi2/summary.json | 1.57 | 0.45 | 0.246465 | CAPTURE | overlapped & stayed bound (sep_min=0.00, sep_end=0.00, rad=0.52) | 0.5225 | 0.992 | 1.607984040945959e-07 | 1.1305107503362175e-08 | nan | True |
| sweep_runs/C3_PHASE_pi2/summary.json | 1.57 | 0.6 | 0.32861999999999997 | CAPTURE | overlapped & stayed bound (sep_min=0.00, sep_end=0.00, rad=0.54) | 0.5435 | 0.9895 | 3.325793951206514e-07 | 7.528112318920935e-08 | nan | True |
| sweep_runs/C3_PHASE_3pi4/summary.json | 2.36 | 0.3 | 0.16430999999999998 | CAPTURE | overlapped & stayed bound (sep_min=0.00, sep_end=0.00, rad=0.46) | 0.4643 | 0.9987 | 2.3097685335907856e-07 | 1.2258808738630512e-08 | nan | True |
| sweep_runs/C3_PHASE_3pi4/summary.json | 2.36 | 0.45 | 0.246465 | CAPTURE | overlapped & stayed bound (sep_min=0.00, sep_end=0.00, rad=0.51) | 0.5096 | 0.9952 | 1.4749414615432348e-07 | 1.5572533953600204e-08 | nan | True |
| sweep_runs/C3_PHASE_3pi4/summary.json | 2.36 | 0.6 | 0.32861999999999997 | CAPTURE | overlapped & stayed bound (sep_min=0.00, sep_end=0.00, rad=0.52) | 0.5204 | 0.9915 | 5.138926346494816e-07 | 9.493676627885944e-08 | nan | True |
| sweep_runs/C3_PHASE_7pi8/summary.json | 2.75 | 0.3 | 0.16430999999999998 | CAPTURE | overlapped & stayed bound (sep_min=0.00, sep_end=0.00, rad=0.50) | 0.4967 | 0.9992 | 2.219509298310475e-07 | 1.505272495018784e-08 | nan | True |
| sweep_runs/C3_PHASE_7pi8/summary.json | 2.75 | 0.45 | 0.246465 | CAPTURE | overlapped & stayed bound (sep_min=0.00, sep_end=0.00, rad=0.52) | 0.5192 | 0.9958 | 1.242056736471816e-07 | 1.0433594888343855e-08 | nan | True |

## Warnings

- None

Guardrail: this report is a first-pass metric preparation artifact, not a verdict change or a physical-correspondence claim.
