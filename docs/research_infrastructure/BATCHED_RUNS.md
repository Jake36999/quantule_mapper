---
tags: [record, infra, performance]
date: 2026-10-07
branch: Branch - Validation - Index
status: complete
---

# Batched (vmapped) spec runs, and where the time actually goes

The goal was more configurations per GPU-hour, so the basin method can cover more of the parameter
space. **Batching works, and it is exact, but its gain on this GPU is modest.** Profiling found two
bigger levers: one fixed here (provenance overhead), and one measured below (numerical precision).

> [!info] Sources
> - `jax_scout/physics.py`: `operator_args`, `ops_from_args` (`build_operators` now uses them;
>   output byte-identical)
> - `jax_scout/registry.py`: `batch_key`, `BatchedETDRK4`, `max_batch`
> - `tools/run_spec.py`: `_setup_member` / `_run_members` / `_finalize`, `run_batch`,
>   `plan_batches`, `--no-batch`, `--batch-size`
> - `jax_scout/provenance.py`: `_git_state()`, which caches git once per process
> - Tests: `test_batched_sweep_matches_point_by_point`, `test_batch_planning_separates_incompatible_points`
> - Benchmarks: `F:\Maths_exploration\batch\` (16-point `param_a` sweeps of the a\* probe spec,
>   T=5, GTX 1080)

## What batching is
`tools/run_spec.py` now groups the points of a sweep that can share one compiled call: same substrate
(`etdrk4-sncgl`), grid, dt, schedule and static operator arguments. It then advances them together
with `jax.vmap` over (parameters, state).
- **What may vary inside a batch:** the eight `physics.BATCHABLE_PARAMS` (D, eta, rho_vac, omega0, a,
  s, f, a_coupling) and the initial condition (e.g. the seed).
- **What is per-member code, exactly as in a single run:** initial conditions, observers, telemetry,
  recording and summaries.
- **What forces a batch of one:** a different `kinetic_mode`, `D_imag`, `dt`, substrate or schedule,
  because these select different code paths.
- **Batch size** follows a GPU memory budget (`registry.max_batch`: about 17 members at N=96, more at
  smaller N).

**Equivalence (tested):** a 3 × 2 `param_a` × seed sweep, batched against one point at a time, agrees
on every observer to within **1e-10**.

The parameter resolution in `physics.build_operators` was split into `operator_args` +
`ops_from_args` so both paths read parameters identically. Every `Ops` array was compared before and
after on both the dissipative and conservative branches: 60/60 are **byte-identical**.

## Measured: where the time goes (16-point sweep, 1,000 steps each)

**First profile (N=16):** 70 s of wall time, of which **60 s was `provenance.stamp()`**. It ran 3–4
`git` calls per stamped file, at about 0.6 s each from WSL on `/mnt/f`, and each member stamps two
files. Physics stepping was about 5 s. **Fix:** commit, status and branch are read once per process.
A run cannot change its own code.

**After the fix:**

| grid | batched | one at a time | gain | per member-step |
|---|---|---|---|---|
| N=16 | 22.7 s | 30.4 s | **1.34×** | dominated by fixed costs |
| N=32 | 42.8 s | 52.2 s | **1.22×** | ≈ 1.9 ms |
| N=48 | 135.0 s | 136.0 s | **none** | ≈ 7.5 ms |

**Reading.** From N=48 up, one member already saturates the GTX 1080, so running members side by side
only queues the same arithmetic. Batching helps only where a member leaves the GPU idle, which means
small screening grids. It stays on by default because it is never slower and is exact.

## Precision (fp64 vs fp32) — the remaining lever
The GTX 1080 runs float64 at 1/32 of its float32 rate, and every run here is complex128. Measured on
the a\* probe (FEB, ×1.15, K=6, seed 619) over the full replay length, T=360 (72,000 steps), with
`docs/instrument_integrity/evidence/batching/fp32_bench.py` (output alongside it):

| grid | fp64 | fp32 | speed-up | max drift of er(t), fp32 vs fp64 | final \|ψ\|² field, rel. L2 |
|---|---|---|---|---|---|
| N=48 | 6.29 ms/step | 1.70 ms/step | **3.7×** | 1.2e-3 | 3.5e-3 |
| N=96 | 60.0 ms/step | 13.2 ms/step | **4.5×** | 3.4e-3 | 4.7e-3 |

**What fp32 is good enough for.**
- **Yes: screening.** Telling which basin a point lands in, its node count and whether it grows or
  decays. The basins differ by tens of percent in shape descriptors, and fp32 drifts by about 0.5%.
- **No: knife-edge measurements.** The a\* bracket is ±0.5% in gain, and its late slopes differ by
  about 1e-3 per 1k steps, so 0.3% drift in er is the same size as the signal. Bracketing, continuation
  and anything quoted in the catalog stay fp64.

**Built: the precision option.** Set `"precision": "fp32"` in a spec's `protocol`, make it a sweep axis
(`protocol.precision`), or pass `tools/run_spec.py --precision fp32` to override every point.
- Only `etdrk4-sncgl` offers fp32. KG and TG specs that ask for it are rejected.
- fp32 and fp64 points never share a vmapped batch.
- `summary.json` records `precision`, and the gallery marks these runs **fp32 screen** (or
  **mixed screen**), so a screening run is never mistaken for a result.

**Suggested workflow** (the screen→verify helper is not built yet):
1. Screen in fp32, batched, at N=32–48.
2. Cluster the end states.
3. Re-run cluster boundaries and outliers in fp64 at N=96.

Against plain fp64 at N=96, that is roughly 4.5× from precision. Small-grid batching adds about 1.2×
and dropping to N=48 about 10× more, for screening only.

## Issues raised

| # | issue | status |
|---|---|---|
| 1 | Provenance stamping cost 85% of a small sweep | RESOLVED — git state cached per process |
| 2 | Batching gives no gain at N ≥ 48 on the GTX 1080 | NOTED — compute-bound |
| 3 | Batching covers ETDRK4 only (KG/TG run one at a time) | OPEN — same pattern if needed |
| 4 | fp32 is 3.7–4.5× faster, but drifts about 0.1–0.5% | RESOLVED as opt-in `protocol.precision` (CL-013); screening only |

## Associated docs
- [[BASIN_MAPPING]] · [[RUN_VIEWER]] · [[SOLVER_AND_RUNTIME_CHANGELOG]] · [[PROVENANCE_AND_HARNESS_REGISTRY]]

## Branches
- [[Branch - Validation - Index]]
