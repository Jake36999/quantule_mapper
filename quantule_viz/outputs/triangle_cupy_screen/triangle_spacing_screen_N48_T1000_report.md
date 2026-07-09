# Triangle Spacing Screen N48 T1000

Standalone diagnostic CuPy screen using `solver.core.ETDRK4Solver` with externally constructed aligned-phase triangle ICs.

This is Phase C dissipative geometry testing only. These results do not apply to the conservative or moving substrate.

- Spacings: `[0.28, 0.32, 0.36, 0.4, 0.45, 0.49, 0.53]`
- Fixed params hash family: FEB Phase C dissipative parameters

## Results

| spacing | finite | final nodes | rho max | raw energy | spacing drift | mass CV | peak CV | late raw-energy trend |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 0.28 | True | 1 | 0.954043 | 1720 |  | 0 | 0 | 0.1699 |
| 0.32 | True | 3 | 0.953553 | 1535.46 | 0.000739959 | 0.00811821 | 0.00261224 | 0.139513 |
| 0.36 | True | 3 | 0.954658 | 1484.91 | -0.000434945 | 0.00459478 | 0.00126215 | 0.137711 |
| 0.40 | True | 3 | 0.953273 | 1470.29 | -4.24232e-05 | 0.00835881 | 0.0027024 | 0.13844 |
| 0.45 | True | 3 | 0.9556 | 1466.13 | -0.000282441 | 0.00174628 | 0.000826828 | 0.139355 |
| 0.49 | True | 3 | 0.95523 | 1465.53 | -0.00143842 | 0.00360009 | 0.00034501 | 0.139627 |
| 0.53 | True | 3 | 0.955208 | 1465.38 | -0.000297658 | 0.00307878 | 0.00144918 | 0.139545 |

## Recommendation

Promote at most the following spacing(s) to N=96/T=4000, still as diagnostic-only priors:
- `0.36`: final nodes `3`, spacing drift `-0.000434945`, mass CV `0.00459478`, late raw-energy trend `0.137711`.
- `0.45`: final nodes `3`, spacing drift `-0.000282441`, mass CV `0.00174628`, late raw-energy trend `0.139355`.
- `0.53`: final nodes `3`, spacing drift `-0.000297658`, mass CV `0.00307878`, late raw-energy trend `0.139545`.

## Caveats

- Node counts are diagnostic detector outputs from final rho, not validation-gate certifications.
- Screen artifacts are standalone diagnostics and do not alter production defaults or configs.
- No long GIFs were generated; each case has a compact final rho PNG.