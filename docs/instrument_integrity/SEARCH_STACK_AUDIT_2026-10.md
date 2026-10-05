---
tags: [record, validation, search-stack]
date: 2026-10-02
branch: Branch - Validation - Index
status: complete
---

# Search-stack audit — October 2026

This audit covers the Hunter and its satellite scripts, done on 2026-10-02. It is a code review
backed by small numerical tests; it makes no physics claim.

The integrator section (§0) was written before the second ETDRK4 bug was found. The full record of
both bugs is [[ETDRK4_INTEGRATOR_BUGS_2026-10]].

> [!info] Sources
> Test scripts: `docs/instrument_integrity/evidence/etdrk4_2026-10/t1_etdrk4_coeffs.py`,
> `t1b_order.py`, `t2_optimizer.py`. `t2` needs `quantulemapper_real.py` importable, so run it from
> the repo root.

Scripts reviewed: `aste_hunter.py`, `quantulemapper_real.py`, `predator_sweep.py`, `fss_scaling_analyzer.py`,
`tda_profiler.py`, `worker_cupy.py` → `solver/core.py`. Tests in this folder use NumPy only, with no GPU.

## 0. Solver: ETDRK4 coefficients are wrong when L is complex (`t1_etdrk4_coeffs.py`, `t1b_order.py`)
`solver/core.py:110-123` and `jax_scout/physics.py:330-353` build Q, f1, f2, f3 from a mean over the
**upper half** of the contour and then take `real()`. That is the Kassam–Trefethen shortcut, and it is only
valid when L is real. Here `L_k = -D k² - η + i ω0`, which is complex. The conservative NLS branch
(`-i D k²`) is complex too. *(Corrected 2026-10-03: the C3 KG and TG stack use a split-step
integrator and are not affected; an earlier draft said "NLS/KG".)*
- Coefficient error is 1.5e-3 to 7e-3 relative at the Phase C baseline, and up to 95% for f1 on NLS-like L.
- Measured order: error 1.03e-2 → 5.1e-3 → 2.5e-3 → 1.3e-3 as dt halves. That is **1st order**.
  The fixed scheme gives 4e-12 → 3e-15, which is 4th order.
- CuPy and JAX share the same code, so CuPy↔JAX parity checks cannot catch this.
  `BASELINE_AUDIT_NUMERICAL.md §1` marks it CONFIRMED.
- Fix: use the full circle (2M points) and keep the complex values (`self.Q = dt*Q_acc/(2M)` etc., no real()).
  Then store the coefficients as complex128.
- Impact: the error shrinks as dt→0, so qualitative results probably survive. Quantitative rates
  (drift slopes, decay rates, the v=2Dk fit, Ω² cliff location) carry an O(dt·ω) bias. They should be
  re-checked with a dt-halving test after the fix.

## 1. Hunter "prime" mode (NSGA-II + SGN + SBD + ASMT)
- **Target-shuffle null is a no-op.** `calculate_bipartite_sse` sorts the targets, which undoes the
  permutation. The null equalled the main metric in 1000/1000 cases, so the "falsifiability_gap" term from
  it is always 0.
- **Crossover destroys diversity.** The operator is a 70% per-gene 50/50 blend plus ~1% multiplicative
  mutation. On a flat landscape (no selection at all) the mean per-parameter spread falls 1.13 → 0.11
  in 30 generations. A parameter at 0 can only move by ±0.0015.
- **SGN steps are tiny.** The step is capped below 0.01 in parameter ranges of width 4 and 10, so this is
  effectively a no-op.
- **ASMT is an inverse GP** (spectral features → parameters). That mapping is many-to-one, so the GP
  averages distinct basins.
- **`dominates()` is not Pareto.** Any fitness>0 run beats any fitness=0 run regardless of objectives.
- **Fitness is a hand-weighted sum.** It mixes 1/SSE (unbounded) with penalties of very different
  scales, so whichever term is largest dominates the selection.

## 2. Hunter "stability" mode (the one that produced H7.3)
This is a plain elitist GA with Gaussian mutation on 3 axes, inside a box built around the already-known
a*. That is sound and simple. Re-finding a* inside that box is a check that the pipeline works, not
independent discovery.

## 3. Satellites
- `fss_scaling_analyzer.py` **cannot ever produce a candidate.**
  - ℓ is constant across all rows, so the 10-term design matrix has rank 6. curve_fit returns
    variances around 1e9–1e14, confidence comes out at about 3e-7, and the < 0.65 gate always raises.
  - The query also orders by `fitness ASC`, which picks the *worst* runs.
  - It hard-codes D, η and ρ_vac. It is not finite-size scaling.
- `predator_sweep.py`: `load_target_params` omits `param_a`, so every predator child runs with
  `param_a = 0` (the solver default). That is the one axis later validated as sensitive.
- `tda_profiler.py`: reasonable. The 0.5 persistence cut sits just above lattice-square noise (≈0.41).

## Status
No fixes were applied to the search stack. The Hunter's prime mode is retired, and stability mode is
sound but narrow. Basin finding moves to ensemble clustering plus continuation:
[[IMPLEMENTATION_PLAN_2026-10]] Phase F.

## Associated docs
- [[ETDRK4_INTEGRATOR_BUGS_2026-10]] · [[SOLVER_AND_RUNTIME_CHANGELOG]] · [[PROCESS_PLAN_2026-10]]

## Branches
- [[Branch - Validation - Index]]
