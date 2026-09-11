# Gravity D Effective-Medium Replication - Codex

Date: 2026-07-13

Branch: `codex/mirror-effective-medium-replication`

## Scope

This report supersedes the earlier CPU-first note. The CPU work is retained only as an independent short-time reference and analytic force-contract cross-check.

The primary replication now follows Claude's D workflow:

- Source script being replicated: `jax_scout/gravity_D_neutral_probe.py`
- Claude target artifact: `sweep_runs/GRAVITY_D_PROBE_20260713_134638/summary.json`
- GPU mirror used for full evolution: `jax_scout/gravity_D_neutral_probe_gpu.py`

The GPU mirror preserves Claude's source, probe, divergence-form spatial operator, windowed COM convention, and snapshot cadence. It does not modify Claude's original script or any production solver. It adds diagnostics needed for hardening: momentum, norm, analytic operator force, RHS momentum derivative, one-step finite-difference force, coefficient extrema, and baseline-subtracted trajectories.

The tested operator remains the spatial effective-medium operator

```text
H = -D div(N grad),  D = 0.3
```

This is not a temporal lapse implementation and is not a gravity verdict.

## GPU Proof

All full field-evolution runs in this report were executed through the established WSL/JAX environment:

```text
wsl.exe -d Ubuntu -- bash -lc 'cd /mnt/f/quantule_mapper && . ~/jax_irer/bin/activate && ...'
```

Saved preflight:

- `sweep_runs/GRAVITY_D_GPU_CODEX_20260713/gpu_preflight_check_env.txt`
- Per-run JSON preflights under each GPU run directory, for example `sweep_runs/GRAVITY_D_GPU_CODEX_20260713/claude_T4/gpu_preflight.json`

Preflight output:

```text
jax            : 0.10.2
backend        : gpu
devices        : [CudaDevice(id=0)]
x64 enabled    : True
compile+run    : OK  (5 steps @ 16^3, dtype=complex128)
final finite   : True

ENV CHECK PASS
```

The GPU mirror also asserts `jax.default_backend() == "gpu"` before running any suite.

## Exact Claude-Workflow Replication

Claude's saved successful run uses `N=64`, `L=30`, `dt=0.002`, `beta=1`, and records samples through `t=3.996`. This corresponds to a `T=4.0` run in the script's snapshot cadence. The `T=3` handoff wording is therefore treated as secondary to Claude's actual artifact.

Command:

```text
wsl.exe -d Ubuntu -- bash -lc 'cd /mnt/f/quantule_mapper && . ~/jax_irer/bin/activate && XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python jax_scout/gravity_D_neutral_probe_gpu.py --suite claude --T 4.0 --out sweep_runs/GRAVITY_D_GPU_CODEX_20260713/claude_T4'
```

Comparison to Claude artifact:

| run | Claude drift_x | GPU drift_x | max COM sample diff | norm/mass |
|---|---:|---:|---:|---:|
| main | -1.2519725509046342e-02 | -1.2519725509048563e-02 | 1.33e-15 | 1.000000000000 |
| flat null | -1.6511389298479173e-04 | -1.6511389298523582e-04 | 1.33e-15 | 1.000000000000 |
| neg beta | +9.731206169621753e-03 | +9.731206169620421e-03 | 1.33e-15 | 1.000000000000 |
| wide probe | -6.309313261692928e-03 | -6.309313261692484e-03 | 8.88e-16 | 1.000000000000 |

This reproduces Claude's full D result essentially exactly while running under the verified JAX GPU backend.

## Operator Force Contract

For the implemented operator,

```text
d<P_x>/dt = -D int (d_x N) |grad psi|^2 dV
```

The GPU mirror checks analytic force, RHS-derived momentum derivative, and one-step finite difference at every saved sample.

Claude-suite force diagnostics:

| run | initial analytic force | max RHS relative residual | max FD relative residual |
|---|---:|---:|---:|
| main | -3.458820300e-02 | 1.739e-06 | 7.189e-04 |
| flat null | 0.000000000e+00 | 0.000e+00 | 0.000e+00 |
| neg beta | +2.079653852e-02 | 3.516e-06 | 6.953e-04 |
| wide probe | -7.520812084e-02 | 2.421e-07 | 1.861e-04 |

The exact operator-force identity closes across the trajectory to the expected numerical tolerance. The finite-difference residual is larger because it is a forward one-step estimate over a finite `dt`, but it remains small and improves under timestep refinement.

## Hardening Controls

Command:

```text
wsl.exe -d Ubuntu -- bash -lc 'cd /mnt/f/quantule_mapper && . ~/jax_irer/bin/activate && XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python jax_scout/gravity_D_neutral_probe_gpu.py --suite hardening --T 4.0 --out sweep_runs/GRAVITY_D_GPU_CODEX_20260713/hardening_T4'
```

| run | drift_x | initial force | force residual | norm |
|---|---:|---:|---:|---:|
| nonsingular reversed hill, `N=1+beta*Shat` | +1.778285106e-02 | +3.870449091e-02 | 3.532e-08 | 1.000000000000 |
| larger box, `N=64,L=40` | -1.564040242e-02 | -3.459023049e-02 | 9.973e-04 | 1.000000000000 |

The nonsingular reversed-gradient control reverses both force and COM drift without using a reciprocal denominator. The larger-box signal remains inward, though `N=64,L=40` is spatially coarser and has the expected larger force residual.

## Convergence

Command:

```text
wsl.exe -d Ubuntu -- bash -lc 'cd /mnt/f/quantule_mapper && . ~/jax_irer/bin/activate && XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python jax_scout/gravity_D_neutral_probe_gpu.py --suite convergence --T 4.0 --out sweep_runs/GRAVITY_D_GPU_CODEX_20260713/convergence_T4'
```

| run | dx | dt | drift_x | momentum change | initial force | max RHS rel | max FD rel | norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| N64 L30 | 0.46875 | 0.002 | -1.251972551e-02 | -1.800520944e-01 | -3.458820300e-02 | 1.739e-06 | 7.189e-04 | 1.000000000000 |
| N96 L30 | 0.31250 | 0.001 | -1.384620412e-02 | -1.800521411e-01 | -3.458820486e-02 | 5.946e-10 | 3.596e-04 | 1.000000000000 |
| N128 L40 | 0.31250 | 0.001 | -1.384620413e-02 | -1.800521411e-01 | -3.458820486e-02 | 6.128e-10 | 3.596e-04 | 1.000000000000 |

Convergence assessment:

- The initial force converges to `-3.458820486e-02`.
- Momentum change at fixed time agrees between the refined runs.
- The fixed-resolution larger-box comparison `(L,N)=(30,96),(40,128)` agrees in final drift to about `1e-11`.
- The coarse `N=64,L=30` run has the correct sign and force but lower final displacement magnitude, so final displacement alone should not be overinterpreted.

This supports convergence of the spatial effective-medium response under refinement and fixed-`dx` box comparison.

## Characterization

Command:

```text
wsl.exe -d Ubuntu -- bash -lc 'cd /mnt/f/quantule_mapper && . ~/jax_irer/bin/activate && XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python jax_scout/gravity_D_neutral_probe_gpu.py --suite characterization --T 1.0 --out sweep_runs/GRAVITY_D_GPU_CODEX_20260713/characterization_T1'
```

All characterization cases use `N=96,L=30,dt=0.001`, normalized probe norm `1.0`, and report initial acceleration as `force/norm`.

Distance sweep:

| probe x | drift_x over T=1 | initial acceleration |
|---:|---:|---:|
| 2.5 | -1.879330e-03 | -3.562084e-02 |
| 3.0 | -2.152249e-03 | -2.404455e-02 |
| 4.0 | -9.526488e-04 | -6.211596e-03 |
| 5.0 | -1.464613e-04 | -7.289333e-04 |
| 6.0 | -9.754201e-06 | -4.061306e-05 |
| 8.0 | -5.217697e-09 | -1.619511e-08 |

The response is inward and graded with distance, decaying rapidly in the far region. It is not reported as a `1/r^2` law.

Width sweep, fixed center `x=4`, fixed norm:

| probe sigma | drift_x over T=1 | initial acceleration |
|---:|---:|---:|
| 0.6 | -1.119247e-03 | -8.692734e-03 |
| 0.8 | -1.064502e-03 | -7.197240e-03 |
| 1.0 | -9.526488e-04 | -6.211596e-03 |
| 1.3 | -7.078528e-04 | -4.742124e-03 |
| 1.6 | -4.503300e-04 | -3.297463e-03 |
| 2.0 | -2.000393e-04 | -1.783377e-03 |

Width dependence is strong. This establishes directional robustness, not universality. The behavior is consistent with the spatial-operator force law's dependence on `|grad psi|^2`.

## Raw Artifacts

- GPU preflight: `sweep_runs/GRAVITY_D_GPU_CODEX_20260713/gpu_preflight_check_env.txt`
- GPU smoke: `sweep_runs/GRAVITY_D_GPU_CODEX_20260713/smoke/`
- Claude replication: `sweep_runs/GRAVITY_D_GPU_CODEX_20260713/claude_T4/`
- Hardening controls: `sweep_runs/GRAVITY_D_GPU_CODEX_20260713/hardening_T4/`
- Convergence: `sweep_runs/GRAVITY_D_GPU_CODEX_20260713/convergence_T4/`
- Characterization: `sweep_runs/GRAVITY_D_GPU_CODEX_20260713/characterization_T1/`
- CPU reference retained: `sweep_runs/GRAVITY_D_EFFECTIVE_MEDIUM_CODEX_20260713/`

Each GPU run directory contains `gpu_preflight.json`, per-case `.json`, per-case `_samples.npz`, `summary.csv`, and `summary.json`.

## Protected Scope

No production geometry, Hunter, launch infrastructure, protected validation configuration, master theory verdict, or production gravity ladder file was intentionally modified for this GPU replication.

New/updated mirror files from this Codex task:

- `jax_scout/gravity_D_neutral_probe_gpu.py`
- `jax_scout/gravity_D_effective_medium_replicate.py` (CPU reference from the earlier attempt)
- `docs/GRAVITY_D_EFFECTIVE_MEDIUM_CODEX_REPLICATION.md`

The worktree already contains unrelated dirty files from prior/user work. Those were not reverted or edited as part of this replication.

## Verdict

Permitted verdict:

```text
D_SPATIAL_EFFECTIVE_MEDIUM_ATTRACTION_GPU_CONVERGED
```

Meaning:

- Claude's full mirror result is reproduced in the verified GPU environment.
- The flat null is small and baseline-subtracted.
- The coefficient-gradient reversal reverses the force and motion.
- A nonsingular reversed-gradient control also reverses the response.
- The operator-force identity closes throughout the trajectory.
- Norm is conserved to displayed precision.
- Refinement and fixed-`dx` box comparison support convergence of the spatial effective-medium response.
- Distance and width characterization show graded response and width dependence.

Not claimed:

- gravity;
- temporal-lapse validation;
- geodesic attraction;
- equivalence-principle universality;
- IRER confirmation;
- production readiness.

The supported conclusion is:

```text
A bounded, environment-defined variable spatial coefficient produces sustained, sign-reversible inward acceleration of a neutral probe under the implemented divergence-form spatial operator.
```

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `25020bf` (2026-07-22) — *dual substrate*
**Revised since:** 9 commit(s), most recently `caf61af` (2026-09-10)

**Harness code changed since it was written:** 5 commit(s) to `jax_scout/`.
  - `caf61af` 2026-09-10 — Build HUD items 1+2: snapshot writer and universal field renderer
  - `06edae5` 2026-08-27 — Fix run provenance: 30% -> 86% of runs now dated
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Also referenced by (same date or earlier):** [[GRAVITY_AUDIT_D_PROBE_RESULTS]], [[GRAVITY_D_EFFECTIVE_MEDIUM_ROBUSTNESS_FIRST_PASS]], [[IRER_GRAVITY_REENTRY_AUDIT_AND_VALIDATION_PLAN]]

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
