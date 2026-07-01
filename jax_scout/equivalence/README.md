# FP64 Equivalence Proof — JAX port vs CuPy backbone

Proves that `jax_scout/physics.py` (the JAX ETDRK4 port) reproduces the production
CuPy backbone (`solver/core.py` + `solver/kernels.py` + `gravity/unified_omega.py`)
to FP64 round-off, before the JAX port is used as a scout for `vmap` sweeps / autodiff.

## Result (N=32³, 40 ETDRK4 steps)

```
worst rel_L2(psi_k) = 5.78e-16   (threshold 1e-8)   ->   PASS
```

That is **machine epsilon for FP64** (~2.2e-16). The initial `fft·dealias` state is
bit-identical (both back ends call cuFFT); after 40 steps — ≈960 FFTs plus the full
geometry / covariant-Laplacian / cubic-quintic-septic pipeline — divergence is still
~6e-16. The two engines output the same numbers to the last bit double precision allows.

Bit-identity is *not* the bar (different reduction/fusion order across cuFFT-via-CuPy and
cuFFT-via-XLA makes it impossible); agreement at FP64 round-off that does not grow is.

## What is compared

The raw spectral-space integrator only: `psi_k0 = fft(psi0)·dealias`, then `step()`
applied N times. Run-loop extras (phase centering every 50 steps, the decoupled causal
"field of affect", telemetry) are deliberately excluded so this isolates the physics math.
The causal A-field is decoupled from `psi` evolution in the current solver, so it does not
affect the `psi` trajectory.

## Reproduce

The two engines live in different runtimes (CuPy = native Windows; JAX-GPU = WSL2), so the
harness shares one IC on disk and compares trajectory dumps.

```powershell
# 1. shared initial condition + config           (native, numpy)
.\.venv\Scripts\python.exe jax_scout\equivalence\make_ic.py

# 2. CuPy reference trajectory                    (native .venv, cupy)
.\.venv\Scripts\python.exe jax_scout\equivalence\dump_cupy.py

# 3. JAX port trajectory                          (WSL2 jax[cuda12] venv, x64)
wsl -d Ubuntu -- bash -lc '. ~/jax_irer/bin/activate && python /mnt/f/quantule_mapper/jax_scout/equivalence/dump_jax.py'

# 4. compare + verdict                            (native, numpy)
.\.venv\Scripts\python.exe jax_scout\equivalence\compare.py
```

Artifacts (`psi0.npy`, `cfg.json`, `cupy_traj.npz`, `jax_traj.npz`) land in `artifacts/`.

## Notes / next

- FP64 parity REQUIRES `jax.config.update("jax_enable_x64", True)` before any array is
  created (set in `dump_jax.py`). Without it JAX silently downcasts complex128 → complex64.
- Parameter-name fidelity that matters: geometry uses **`param_a_coupling`** (conformal
  exponent) while the nonlinearity uses **`param_a`** (cubic coeff) — different knobs.
- `physics.py` is dtype-parametric: the same code runs the FP32 scout
  (`real_dtype=float32, complex_dtype=complex64`). Re-run this harness in FP32 to quantify
  the scout's accuracy envelope before trusting it for exploration.
- Next: refactor `build_operators` to a fully traced (jnp) form for `vmap` over a parameter
  batch, then benchmark batched throughput vs the CuPy one-process-per-config loop.
