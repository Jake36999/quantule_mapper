# Gravity D Effective-Medium Robustness Closure

Verdict: `D_SPATIAL_EFFECTIVE_MEDIUM_ROBUSTNESS_FIRST_PASS_CLOSED`

This is a targeted closure pass for the already accepted spatial effective-medium result. It does not test temporal-lapse gravity, geodesic motion, equivalence-principle behavior, IRER source confirmation, or production readiness.

Run directory:

`sweep_runs/GRAVITY_D_ROBUSTNESS_CLOSURE_GPU_20260713_204559/`

Command:

```powershell
wsl.exe -d Ubuntu -- bash -lc "cd /mnt/f/quantule_mapper && . ~/jax_irer/bin/activate && XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python jax_scout/gravity_D_robustness_closure_gpu.py --out sweep_runs/GRAVITY_D_ROBUSTNESS_CLOSURE_GPU_20260713_204559"
```

GPU preflight was saved in `gpu_preflight.json`: backend `gpu`, selected device `cuda:0`, JAX `0.10.2`, jaxlib `0.10.2`, x64 enabled, WSL2 Linux, git commit `9fc7785df19d24f4be29e5d90bad59ed993da87b`.

## R7 COM-Null Diagnostic

The raw flat drift of `-1.3069e-3` is reproduced only by Claude's slab-window COM diagnostic. It is absent in the periodic/circular COM and absent to numerical precision in the global Cartesian COM under fixed-dx box refinement. The null has zero analytic radial force and conserved norm.

| run | grid | L | dx | global raw drift | slab raw drift | periodic raw drift | radial force |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R7_flat_L30_N96 | 96 | 30 | 0.3125 | -5.2807e-08 | -1.3068599738e-03 | 0 | 0 |
| R7_flat_L40_N128 | 128 | 40 | 0.3125 | 0 | -1.3068599738e-03 | 0 | 0 |
| R7_flat_L50_N160 | 160 | 50 | 0.3125 | 8.8818e-16 | -1.3068599738e-03 | -1.3323e-15 | 0 |

Interpretation: the `-1.3069e-3` flat drift is a slab-window measurement artifact from free probe spreading across a fixed window, not a finite-box wrap, force, or coefficient effect. The suitable diagnostic limit is global or periodic COM, where the raw flat drift shrinks to numerical zero. The large relative force residual reported for the `L=50` null is denominator noise at zero analytic force; the absolute residual is `2.62e-17`.

Paired subtraction is now formalized for positive-coefficient runs as:

`delta_x_net = delta_x_medium - delta_x_free`

with the free control matched exactly in grid, box, dt, T, probe center, probe width, carrier, and diagnostic.

## R8 Source Shapes

The old width-rescaled `rho_B` versus `rho_B^2` comparison is not used as a separate shape claim here. The closure pass compares genuinely distinct smooth positive profiles matched to the baseline by `N_min = 0.5` and integrated coefficient deficit.

| source family | width | deficit | second moment | initial radial force | global net drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| gaussian | 1.5 | 14.3794898963 | 3.8251467120 | -6.211596e-03 | -9.600395e-04 |
| supergaussian4 | 1.7901271814 | 14.3794898963 | 2.6614784474 | -4.108963e-03 | -8.961710e-04 |
| compact_bump | 2.6496571772 | 14.3794898963 | 2.5676249096 | -3.788022e-03 | -8.833849e-04 |
| two_lobe | 1.1998239372 | 14.3794898963 | 4.3365624420 | -1.927700e-03 | -4.082333e-04 |

All tested non-flat source families retain inward radial force. Differences in magnitude are profile-dependent characterization, not a ranking of physical source quality. Force-contract absolute residuals remain at or below `8.33e-14` for R8 positives.

## R9 Translation Refinement

Joint source/probe offsets were applied as fractions of the grid cell while preserving the source-probe separation. The radial force remains inward for every offset. Relative spread decreases with refinement:

| grid | force range | relative spread |
| ---: | ---: | ---: |
| 64 | -6.445875e-03 to -6.211596e-03 | 3.6941% |
| 96 | -6.314753e-03 to -6.211596e-03 | 1.6455% |

The remaining translation sensitivity is bounded as a discretization effect in this targeted pass. R9 positive-run force-contract absolute residuals are at most `6.27e-10` on the coarse grid and return to `5.04e-17` on the refined grid.

## R10 Normalization

`closure_results.csv` now separates:

- `probe_norm`
- `raw_force`
- `force_per_norm`
- `raw_momentum`
- `momentum_per_norm`
- `absolute_force_residual`
- `relative_force_residual`

The same table includes `source_family`, `source_shape`, `deficit_second_moment`, and paired net drift fields for global, slab-window, and periodic COM diagnostics.

## Evidence Files

Primary files in the run directory:

- `gpu_preflight.json`
- `preregistered_matrix.json`
- `closure_results.csv`
- `closure_results.json`
- `source_matching.csv`
- `source_matching.json`
- `closure_summary.json`
- per-run `*.json` metrics
- per-run `*_trajectory.npz` trajectory snapshots
- `artifact_hashes.csv`
- `git_status_short.txt`
- `git_diff_name_only.txt`
- `git_diff_cached_name_only.txt`

The artifact inventory provides SHA-256 hashes for all run evidence files. The saved git status and diff listings show the broader dirty worktree directly; this closure pass only adds standalone mirror code, this bounded report, and run artifacts. No production geometry, Hunter code, GPU launcher, CPU fallback policy, protected validation file, master theory verdict, or production gravity ladder was modified by this closure pass.

## Closure

The closure criteria are met:

- the R0/R7 free-COM drift is explained as a slab-window COM artifact and controlled by global/periodic diagnostics;
- positive medium runs use exactly matched flat controls for net drift;
- Gaussian, super-Gaussian, compact-bump, and two-lobe profiles all retain inward force under matched `N_min` and deficit;
- translation sensitivity decreases with refinement and remains bounded;
- force and momentum normalization fields are explicit in the machine-readable table;
- artifact hashes and git state are saved with the evidence package.

Bounded conclusion: `D_SPATIAL_EFFECTIVE_MEDIUM_ROBUSTNESS_FIRST_PASS_CLOSED`.

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `25020bf` (2026-07-22) — *dual substrate*
**Revised since:** 10 commit(s), most recently `e42b5bb` (2026-09-11)

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
