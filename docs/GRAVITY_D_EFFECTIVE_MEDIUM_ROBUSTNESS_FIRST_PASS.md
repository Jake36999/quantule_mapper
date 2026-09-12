# Gravity D Spatial Effective-Medium Robustness - First Pass

Date: 2026-07-13

Branch: `codex/mirror-effective-medium-replication`

Accepted starting verdict:

```text
D_SPATIAL_EFFECTIVE_MEDIUM_ATTRACTION_GPU_CONVERGED
```

This robustness pass tests only the confirmed spatial effective-medium mechanism:

```text
i d_t psi = -D div(N_B grad psi)
N_B = 1 / (1 + beta S_B)
```

It does not test temporal-lapse gravity, geodesic motion, equivalence-principle universality, IRER source semantics, or production readiness.

## GPU Execution

All completed field-evolution runs in the accepted robustness artifact used the established WSL/JAX GPU environment:

```text
wsl.exe -d Ubuntu -- bash -lc "cd /mnt/f/quantule_mapper && . ~/jax_irer/bin/activate && XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python jax_scout/gravity_D_robustness_gpu.py --out sweep_runs/GRAVITY_D_ROBUSTNESS_GPU_20260713_195713"
```

Preflight is saved at:

```text
sweep_runs/GRAVITY_D_ROBUSTNESS_GPU_20260713_195713/gpu_preflight.json
```

The driver asserts `jax.default_backend() == "gpu"` before building the preregistered matrix.

## Driver And Artifacts

Driver:

```text
jax_scout/gravity_D_robustness_gpu.py
```

Accepted run directory:

```text
sweep_runs/GRAVITY_D_ROBUSTNESS_GPU_20260713_195713/
```

Primary artifacts:

- `preregistered_matrix.json`
- `robustness_results.csv`
- `robustness_results.json`
- `robustness_summary.json`
- `source_matching.csv`
- per-run `*.json`
- per-run `*_trajectory.npz`
- plots:
  - `radial_force_by_source.svg`
  - `radial_force_by_Nmin.svg`
  - `radial_force_by_width.svg`
  - `orientation_radial_force.svg`

Aborted/failed setup runs retained:

- `GRAVITY_D_ROBUSTNESS_GPU_20260713_194053`: stopped after R0 because the first driver used a spherical COM window, which did not reproduce Claude's accepted slab-window diagnostic.
- `GRAVITY_D_ROBUSTNESS_GPU_20260713_194643`: stopped after detecting a zero-force residual reporting bug for flat nulls.

Both issues were diagnostic/reporting issues in the new robustness driver, not production solver changes.

## R0 - Frozen Reference

Accepted refined reference drift:

```text
-1.38462041e-02
```

R0 reproduced the reference:

| run | raw radial drift | net radial drift | radial force | norm ratio |
|---|---:|---:|---:|---:|
| reference rho2, N96 L30 dt0.001 T4 | -1.3887459638e-02 | -1.2580599664e-02 | -6.2115961698e-03 | 1.000000000000 |
| flat null | -1.3068599738e-03 | -1.3068599713e-03 | 0.0 | 1.000000000000 |

The small raw-drift difference from the accepted table is within tolerance and comes from comparing the full `T=4.0` final COM rather than the previous saved sample at `t=3.996`.

## R1 - Source Profiles

Sources were matched to the baseline by `N_min=0.5`, background `N=1`, and integrated coefficient deficit.

| source | matched width | integrated deficit | deficit second moment | radial force | net drift |
|---|---:|---:|---:|---:|---:|
| rho2 | 1.500000 | 14.3794898963 | 3.8251467120 | -6.211596e-03 | -9.600395e-04 |
| rho | 1.060660 | 14.3794898963 | 3.8251467120 | -6.211596e-03 | -9.600395e-04 |
| supergaussian4 | 1.790127 | 14.3794898963 | 2.6614784474 | -4.108963e-03 | -8.961703e-04 |
| flat null | 1.500000 | 0.0 | n/a | 0.0 | -2.495781e-12 |

All matched smooth bounded wells produced inward radial acceleration. Magnitudes differ, especially for the super-Gaussian; this is reported as source-shape dependence, not failure.

## R2 - Strength Ladder

| target N_min | beta | radial force | net drift |
|---:|---:|---:|---:|
| 1.00 | 0.000000 | 0.000000e+00 | 0.000000e+00 |
| 0.90 | 0.111111 | -7.614483e-04 | -1.227306e-04 |
| 0.80 | 0.250000 | -1.684378e-03 | -2.694331e-04 |
| 0.65 | 0.538462 | -3.509620e-03 | -5.533980e-04 |
| 0.50 | 1.000000 | -6.211596e-03 | -9.600395e-04 |

The response vanishes at flat coefficient and increases smoothly inward with deformation strength. No abrupt numerical transition appeared.

## R3 - Rotational And Translational Checks

| run | radial force | transverse force | net drift |
|---|---:|---:|---:|
| x-axis | -6.211596e-03 | 4.24e-18 | -9.600395e-04 |
| y-axis | -6.211596e-03 | 3.48e-18 | -9.600395e-04 |
| z-axis | -6.211596e-03 | 4.70e-19 | -9.600395e-04 |
| diagonal | -6.211596e-03 | 5.59e-18 | -9.600395e-04 |
| translated source/probe pair | -6.260583e-03 | 8.55e-18 | -9.672701e-04 |

Rotations preserve radial direction and magnitude to displayed precision. The translated pair changes radial force by about 0.8%, which is small for a non-grid-symmetric offset and remains inward.

## R4 - Probe Amplitude

| amplitude | raw radial force | force / amplitude^2 | net drift |
|---:|---:|---:|---:|
| 0.5 | -8.647051e-03 | -3.458820e-02 | -9.600395e-04 |
| 1.0 | -3.458820e-02 | -3.458820e-02 | -9.600395e-04 |
| 2.0 | -1.383528e-01 | -3.458820e-02 | -9.600395e-04 |

The raw force scales as amplitude squared, while normalized acceleration and COM drift are amplitude-independent, as expected for the linear wave equation and normalized COM diagnostic.

## R5 - Probe Width

Total norm was held fixed.

| probe sigma | radial force | gradient energy | net drift |
|---:|---:|---:|---:|
| 0.50 | -1.015501e-02 | 6.000000 | -1.110616e-03 |
| 0.75 | -7.484576e-03 | 2.666667 | -1.092131e-03 |
| 1.00 | -6.211596e-03 | 1.500000 | -9.600395e-04 |
| 1.25 | -4.991974e-03 | 0.960000 | -7.580046e-04 |
| 1.60 | -3.297463e-03 | 0.585937 | -4.538936e-04 |

Width dependence remains strong and smooth. This supports directional robustness, not universality. The trend is consistent with the exact operator-force law involving `|grad psi|^2`.

## R6 - Carrier Perturbation

Each carrier run has a matched beta=0 free-propagation control.

| carrier kx | raw drift | free drift | medium-induced net drift | radial force |
|---:|---:|---:|---:|---:|
| -0.2 | -1.204252e-01 | -1.200000e-01 | -4.251724e-04 | -6.327364e-03 |
| 0.0 | -9.600395e-04 | -2.495781e-12 | -9.600395e-04 | -6.211596e-03 |
| +0.2 | +1.185853e-01 | +1.200000e-01 | -1.414654e-03 | -6.327364e-03 |

Raw final COM is dominated by carrier motion for `kx=+-0.2`, but the medium-induced force and free-subtracted drift remain inward.

## Force Contract And Norm

All completed accepted runs have:

- inward radial force for every positive deformation;
- zero radial force for flat nulls;
- force-contract residual below `2e-4`;
- norm ratio within `1e-8` of unity.

The largest force-contract residual is from the narrowest width case and remains small.

## Verdict

```text
D_SPATIAL_EFFECTIVE_MEDIUM_ROBUSTNESS_FIRST_PASS
```

Meaning:

The converged spatial effective-medium attraction remains directionally stable under the tested source, strength, orientation, translation, amplitude, width, and carrier perturbations.

Not claimed:

- universal free fall;
- temporal-lapse gravity;
- geodesic validation;
- inverse-square law;
- IRER gravity source;
- production readiness.

## Protected-File Diff Check

No production geometry, Hunter, protected validation/configuration file, GPU launcher, CPU/GPU fallback behavior, master theory verdict, or production gravity ladder was intentionally modified.

New standalone files from this phase:

- `jax_scout/gravity_D_robustness_gpu.py`
- `docs/GRAVITY_D_EFFECTIVE_MEDIUM_ROBUSTNESS_FIRST_PASS.md`

Previously added standalone mirror/reference files on this branch:

- `jax_scout/gravity_D_neutral_probe_gpu.py`
- `jax_scout/gravity_D_effective_medium_replicate.py`
- `docs/GRAVITY_D_EFFECTIVE_MEDIUM_CODEX_REPLICATION.md`

Command requested:

```text
git diff --name-only 9fc7785df19d24f4be29e5d90bad59ed993da87b...HEAD
```

Output:

```text

```

Because these artifacts are currently uncommitted, the triple-dot committed diff is empty. Scoped working-tree status shows only approved standalone mirror/report files for this Codex task.

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `25020bf` (2026-07-22) — *dual substrate*
**Revised since:** 11 commit(s), most recently `bd93089` (2026-09-12)

**Harness code changed since it was written:** 6 commit(s) to `jax_scout/`.
  - `caf61af` 2026-09-10 — Build HUD items 1+2: snapshot writer and universal field renderer
  - `06edae5` 2026-08-27 — Fix run provenance: 30% -> 86% of runs now dated
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine
  - *…and 1 more.*

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Also referenced by (same date or earlier):** [[GRAVITY_AUDIT_D_PROBE_RESULTS]], [[IRER_GRAVITY_REENTRY_AUDIT_AND_VALIDATION_PLAN]]

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
