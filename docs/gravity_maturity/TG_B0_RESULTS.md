# TG-B0 Reduced Feedback Exchange Results

Timestamp: 2026-07-14.
Run directory: `F:/quantule_mapper/sweep_runs/TG_B0_EXCHANGE_20260714_114141`.
Status: `TG_B0_LIMIT_CYCLE_SUPPORTED`.

## Intervention Causality

| test | pass |
| --- | --- |
| source_off_removes_T | True |
| temporal_off_removes_T_and_G | True |
| geometric_off_retains_T_removes_G | True |

## Representative Rows

| run | source | attractor | max T | max G | lag R->T | lag T->G | freq T class | freq G class |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| coherence_k0.55_open | coherence | DAMPED_OSCILLATION | 6.859683e-02 | 8.039854e-02 | 7.000000e-01 | 1.350000e+00 | DRIVEN_OR_NONLINEAR_MODE | DRIVEN_OR_NONLINEAR_MODE |
| coherence_closed_feedback | coherence | BOUNDED_LIMIT_CYCLE | 6.873621e-02 | 8.079793e-02 | 7.000000e-01 | 1.350000e+00 | DRIVEN_OR_NONLINEAR_MODE | DRIVEN_OR_NONLINEAR_MODE |
| threshold_k0.55_open | threshold | DAMPED_OSCILLATION | 1.957653e-02 | 2.301340e-02 | 1.050000e+00 | 1.400000e+00 | DRIVEN_OR_NONLINEAR_MODE | DRIVEN_OR_NONLINEAR_MODE |

## Interpretation

TG-B0 is a reduced exchange gate. It can support bounded response and intervention causality, but it cannot prove spatial radiation or node stabilization.