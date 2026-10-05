# NLSE Reduced V2 Comparison Report

Status: provisional until Claude review.

Final decision: `NLSE_REDUCED_V2_COMPARISON_READY`

## Scope

This is a reduced-equation numerical-method sanity check only. It is not empirical validation, not a full IRER comparison, and not a physical-correspondence claim.

## External Package Inspection

- Source root: `E:\quantule_mapper_external_data\DATA-NUM-001_nlse_package\extracted\source`
- Package version from setup.py: `2.3.0`
- Requirements: `matplotlib==3.8.4, numba==0.59.1, numpy==1.26.4, pyFFTW==0.13.1, pyopencl==2024.2.1, pyvkfft==2024.1.2.post0, scipy==1.13.1, setuptools==68.1.2, tqdm==4.66.1`
- License file first line: `MIT License`
- Run mode: `import-from-unpacked-source with local child-process compatibility shims`

## Project C2 V2 Reference

- Slope: `1.9997819749017656`
- Expected slope 2D: `2.0`
- Slope percent error: `0.010901254911721558`
- Minimum mass retention: `0.9996329889054258`

## External NLSE Reduced Run

- Slope: `2.0000000000003184`
- Expected slope 2D: `2.0`
- Slope percent error: `1.5920598173124745e-11`
- Intercept: `-1.755426704015259e-09`
- R^2: `1.0`

## Side-by-Side Rows

| k | project v | external v | expected v | project mass ret | external mass ret |
|---:|---:|---:|---:|---:|---:|
| 0.6283185307179586 | 1.256499972363777 | 1.2566370596806908 | 1.2566370614359172 | 0.999875274532852 | 1.0000000000000668 |
| 1.2566370614359172 | 2.5130000445903122 | 2.5132741211168086 | 2.5132741228718345 | 0.9996329889054258 | 1.0000000000000513 |

## Caveats

- External NLSE package was run from archived source with local CPU compatibility shims because pyfftw/numba/tqdm were not installed in the active venv; CuPy was shadow-disabled inside the child process to keep the package on its CPU path without pyvkfft.
- This is a reduced-equation method sanity check only, not empirical validation and not a full IRER comparison.

## Required Statements

- External NLSE only; no Quantule Mapper simulations run.
- No production physics changed.
- No verdicts changed.
- No physical-correspondence claims added.
- Result is a reduced-equation method sanity check only.
- Provisional until Claude review.
