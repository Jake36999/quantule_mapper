# Triangle Spacing Screen N96 T6000

Standalone diagnostic CuPy screen using `solver.core.ETDRK4Solver` with externally constructed aligned-phase triangle ICs.

This is Phase C dissipative geometry testing only. These results do not apply to the conservative or moving substrate.

- Spacings: `[0.28, 0.36, 0.45, 0.53]`
- Fixed params hash family: FEB Phase C dissipative parameters

## Results

| spacing | finite | final nodes | rho max | raw energy | spacing drift | mass CV | peak CV | late raw-energy trend |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 0.28 | True | 1 | 0.936381 | 17172.6 |  | 0 | 0 | 0.0124492 |
| 0.36 | True | 3 | 0.933234 | 13653.8 | -0.000645462 | 0.000471416 | 0.000525883 | -0.00635845 |
| 0.45 | True | 3 | 0.931427 | 13511 | 1.82427e-05 | 0.00103363 | 0.000822577 | -0.00463814 |
| 0.53 | True | 3 | 0.931629 | 13510.5 | -9.98603e-06 | 0.000345201 | 8.6819e-05 | -0.00434824 |

## Stage 1 Promotion Review

Stage 1 passed: all four N=96/T=6000 diagnostic cases completed finite on the CuPy path. The `0.28` merge-control collapsed to one detected node and is rejected for overnight follow-up.

Promote at most the following spacing(s) to an N=96/T=12000 follow-up, still as diagnostic-only priors:
- `0.45`: final nodes `3`, spacing drift `1.82427e-05`, mass CV `0.00103363`, peak CV `0.000822577`, late raw-energy trend `-0.00463814`, late rho-max trend `-0.0106829`.
- `0.36`: final nodes `3`, spacing drift `-0.000645462`, mass CV `0.000471416`, peak CV `0.000525883`, late raw-energy trend `-0.00635845`, late rho-max trend `-0.00358191`.

Hold `0.53` as a boundary/control candidate rather than a primary overnight pick: it retained three nodes with excellent balance metrics, but the final pairwise distances were less clean (`0.5301`, `0.5301`, `0.4700`) than the more symmetric `0.45` and `0.36` cases.

Measured wallclock was about 440-446 seconds per N=96/T=6000 case. A same-resolution N=96/T=12000 follow-up is estimated at about 15 minutes per spacing, or about 30-35 minutes for the two recommended spacings including overhead.

## Caveats

- Node counts are diagnostic detector outputs from final rho, not validation-gate certifications.
- Screen artifacts are standalone diagnostics and do not alter production defaults or configs.
- No long GIFs were generated; each case has a compact final rho PNG.
