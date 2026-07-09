# C2 Nonlinear RHS Source Map

## conservative_wrapper_factor

- Source: `tools/conservative_geometry_campaign.py:281`
- Symbolic description: `N_k^C2 = i * N_k^baseline`
- Input representation: `spectral psi_k`
- Output representation: `spectral RHS`
- Category: `Hamiltonian/conservative wrapper factor`
- Expected norm-neutrality: Only if the baseline real-space operator is real self-adjoint in the active state/discretization.

## geometry_covariant_correction

- Source: `solver/kernels.py:33`
- Symbolic description: `i * D * (Delta_cov - Delta_flat) psi`
- Input representation: `physical fields derived from spectral psi_k`
- Output representation: `spectral RHS after FFT/dealias`
- Category: `derivative/covariant/conformal geometry-dependent`
- Expected norm-neutrality: Requires the effective discrete geometry operator to be self-adjoint after the state-dependent geometry construction.

## cubic_density_a

- Source: `solver/kernels.py:31`
- Symbolic description: `i * a * psi * rho`
- Input representation: `physical psi/rho`
- Output representation: `spectral RHS after FFT/dealias`
- Category: `local multiplicative density-dependent`
- Expected norm-neutrality: Pointwise phase-only if a and rho are real.

## quintic_density_s

- Source: `solver/kernels.py:31`
- Symbolic description: `i * s * psi * rho^2`
- Input representation: `physical psi/rho`
- Output representation: `spectral RHS after FFT/dealias`
- Category: `local multiplicative density-dependent`
- Expected norm-neutrality: Pointwise phase-only if s and rho are real.

## septic_density_f

- Source: `solver/kernels.py:31`
- Symbolic description: `i * f * psi * rho^3`
- Input representation: `physical psi/rho`
- Output representation: `spectral RHS after FFT/dealias`
- Category: `local multiplicative density-dependent`
- Expected norm-neutrality: Pointwise phase-only if f and rho are real.

## reference_contract

- Source: `jax_scout/physics.py:311`
- Symbolic description: `kinetic_mode='conservative' sets L_k=-i*D*k^2 and kfac=1j`
- Input representation: `spectral psi_k`
- Output representation: `spectral RHS`
- Category: `C2 reference contract`
- Expected norm-neutrality: Docs/comments describe near-conservation, but term decomposition is needed for generic evolved states.
