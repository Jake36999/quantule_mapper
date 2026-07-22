# TG-A Action And Source Audit Results

Timestamp: 2026-07-14.
Run directory: `F:/quantule_mapper/sweep_runs/TG_A_CONTRACTS_20260714_114135`.
Status: `TG_A_ACTION_AND_SOURCE_AUDIT_CLOSED`.

## Model Class

Selected: explicitly budgeted dissipative response model with reciprocal T/G exchange.

Energy ledger:

```text
dE_TG/dt = integral alpha_T R_res V_T dV - integral gamma_T V_T^2 dV - integral gamma_G V_G^2 dV - boundary_absorption + numerical_residual
```

The T/G quadratic energy is positive in the tested regime because `|kappa_TG| < omega_T omega_G` with margin `0.7125`.

## Resolution Sources

- Primary continuous source: `R_coh = [-d_t K_phase]_+`.
- Secondary threshold source: `R_thr = sigmoid((P-P_c)/delta_P)[d_t P]_+`.
- Neither source uses outgoing flux, T/G pulse timing, or node outcome as input.

## Source Nulls

| case | family | integrated source | peak source | pass |
| --- | --- | ---: | ---: | --- |
| flat_field | R_coh | 0.000000e+00 | 0.000000e+00 | True |
| static_node | R_coh | 0.000000e+00 | 0.000000e+00 | True |
| coherence_transition | R_coh | 7.362403e-01 | 2.495838e-01 | True |
| phase_scrambled_density_matched | R_coh | 0.000000e+00 | 0.000000e+00 | True |
| threshold_transition | R_thr | 4.825245e-01 | 1.039396e-01 | True |
| source_removed | R_coh | 0.000000e+00 | 0.000000e+00 | True |

## Linear Modes

| mode | omega^2 | omega | stable |
| --- | ---: | ---: | --- |
| 0 | 5.957825e-01 | 7.718695e-01 | True |
| 1 | 1.689217e+00 | 1.299699e+00 | True |

## Gate Decision

TG-A permits TG-B0 if the status is `TG_A_ACTION_AND_SOURCE_AUDIT_CLOSED`.