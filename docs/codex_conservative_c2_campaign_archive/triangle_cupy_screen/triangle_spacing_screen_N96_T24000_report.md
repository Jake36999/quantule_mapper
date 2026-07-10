# Triangle Spacing Screen N96 T24000

Standalone diagnostic CuPy screen using `solver.core.ETDRK4Solver` with externally constructed aligned-phase triangle ICs.

This is Phase C dissipative geometry testing only. These results do not apply to the conservative or moving substrate.

- Spacings: `[0.45, 0.36]`
- Fixed params hash family: FEB Phase C dissipative parameters

## Results

| spacing | finite | final nodes | rho max | raw energy | spacing drift | mass CV | peak CV | late raw-energy trend |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 0.45 | True | 3 | 0.953786 | 9928.96 | -7.52129e-05 | 0.00158205 | 0.00648586 | -0.195549 |
| 0.36 | True | 3 | 0.955671 | 10057.2 | -0.000701722 | 0.00043333 | 0.000565094 | -0.193091 |

## Confirmation Review

The N=96/T=24000 confirmation pass completed finite for both selected spacings. Both retained three detected nodes and remained equilateral-like by the diagnostic geometry detector.

Recommended overnight priority, still as Phase C dissipative geometry priors only:
- `0.45`: cleaner spacing drift (`-7.52129e-05`) and very symmetric final pairwise distances (`0.449971`, `0.449971`, `0.449985`), with mass CV `0.00158205` and peak CV `0.00648586`.
- `0.36`: lower mass/peak imbalance (`0.00043333` / `0.000565094`) but larger spacing drift (`-0.000701722`), with final pairwise distances (`0.358321`, `0.358321`, `0.358637`).

Both cases show strongly negative late raw-energy trends over this diagnostic window. Late rho-max trend is positive for both (`0.0161853` for `0.45`, `0.0231188` for `0.36`), so the overnight pass should be framed as a longer diagnostic hold/trend check rather than a stability claim.

Measured wallclock was about 1775-1781 seconds per N=96/T=24000 case. A same-resolution N=96/T=48000 follow-up is estimated at about 59-60 minutes per spacing, and N=96/T=96000 at about 2 hours per spacing.

## Caveats

- Node counts are diagnostic detector outputs from final rho, not validation-gate certifications.
- Screen artifacts are standalone diagnostics and do not alter production defaults or configs.
- No long GIFs were generated; each case has a compact final rho PNG.
