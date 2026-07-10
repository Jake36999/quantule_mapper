# Conservative C2 RHS Flux Source Isolation

Final decision: `RHS_NONLINEAR_ALGEBRAIC_FLUX_CONFIRMED`
Wallclock seconds: `9.215`

## Maxima

- max physical/spectral full flux diff: `9.3357632304051e-19`
- max reconstruction roundtrip diff: `2.579316968343015e-16`
- max linear flux: `1.6596912409609063e-16`
- max nonlinear flux: `0.00038857810155681715`
- max full flux: `0.00038857810155689526`

## Samples

| t | nonlinear flux | linear flux | full flux | spectral diff | roundtrip diff | status |
|---:|---:|---:|---:|---:|---:|---|
| 0.0 | 3.1204902409264404e-16 | -2.9239342827579024e-17 | 2.138945629209073e-16 | 3.295292472999693e-21 | 5.378550339604393e-17 | numerical_zero |
| 0.25 | -0.00019287586936156295 | -7.134861826352037e-17 | -0.0001928758693616686 | 6.7407851557105e-19 | 2.1109416298292771e-16 | fail |
| 0.5 | -0.00033359289237938145 | -3.65064025021203e-17 | -0.00033359289237948786 | 5.185568537232997e-19 | 5.789639601072594e-17 | fail |
| 0.75 | -0.00038857810155681715 | -1.991441995154047e-17 | -0.00038857810155689526 | 6.74186092109443e-19 | 2.579316968343015e-16 | fail |
| 1.0 | -0.0003493345907089313 | -1.6596912409609063e-16 | -0.00034933459070899905 | 9.3357632304051e-19 | 1.9770426615273173e-16 | fail |

No stability claim is made. This is diagnostic-only and did not run geometry campaigns or N64 replay.