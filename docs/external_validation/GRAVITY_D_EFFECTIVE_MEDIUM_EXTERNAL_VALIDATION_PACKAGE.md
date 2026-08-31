# Gravity D Effective-Medium External Validation Package

Status: third-party rerun package for the standalone spatial effective-medium mechanism.

This package supports external reproduction of the characterized operator:

```text
i d_t psi = -D div(N(x) grad psi)
d<P>/dt = -D integral grad(N) |grad psi|^2 dV
F_cg ~= -D K_grad grad N(R)
```

It does not assert Newtonian gravity, relativistic geodesics, universal free fall, IRER source semantics, or production readiness.

## Primary Run Artifacts

Core characterization run:

```text
sweep_runs/GRAVITY_D_DYNAMICS_CHARACTERIZATION_GPU_20260713_221727/
```

Portable artifact bundle for third-party transfer:

```text
docs/external_validation/artifact_bundles/gravity_d_effective_medium_characterization.zip
docs/external_validation/artifact_bundles/gravity_d_effective_medium_characterization_zip_hash.csv
```

The bundle contains the primary CSV/JSON/MD/SVG evidence files, the bounded documentation set, the standalone characterization driver, and `BUNDLE_MANIFEST.csv` with SHA-256 hashes for every bundled file. The local `sweep_runs` path remains the internal source of record; the zip is the portable handoff artifact until a public repository commit or external artifact host is assigned.

Required files:

- `gpu_preflight.json`
- `environment_versions.json`
- `git_state_before.txt`
- `preregistered_matrix.json`
- `run_manifest.csv`
- `instantaneous_force_atlas.csv`
- `trajectory_metrics.csv`
- `regime_classification.csv`
- `model_comparison.csv`
- `coarse_grained_coefficients.csv`
- `curve_collapse_metrics.csv`
- `holdout_validation.csv`
- `numerical_validation.csv`
- `falsification_results.csv`
- `source_profile_metadata.csv`
- `reduced_model_sign_errors.csv`
- `REDUCED_MODEL_SIGN_ERROR_ANALYSIS.md`
- `artifact_hashes.csv`
- `standalone_file_hashes.csv`
- `TECHNICAL_HANDOFF.md`
- `CHARACTERIZATION_SUMMARY.json`
- `OPEN_QUESTIONS.md`
- `DOCUMENTATION_INPUTS.md`

Earlier evidence layers:

```text
sweep_runs/GRAVITY_D_GPU_CODEX_20260713/
sweep_runs/GRAVITY_D_ROBUSTNESS_GPU_20260713_195713/
sweep_runs/GRAVITY_D_ROBUSTNESS_CLOSURE_GPU_20260713_204559/
```

## Fixed Reference Configuration

Baseline:

```text
grid N = 96
box L = 30
dt = 0.001
T = 4
D = 0.3
source = Gaussian rho_B^2
source width = 1.5
target N_min = 0.5
probe r0 = 4
probe sigma = 1
carrier k = 0
coefficient N = 1 / (1 + beta S)
beta = 1/N_min - 1
```

Reference result:

```text
initial radial force = -6.211596169843487e-03
matched flat force = 0
global/periodic COM flat null = numerical zero
```

## Environment

Recorded by `gpu_preflight.json` and `environment_versions.json`:

```text
WSL2 Ubuntu
python = 3.12.3
JAX = 0.10.2
jaxlib = 0.10.2
backend = gpu
device = cuda:0
x64 = enabled
git commit = 9fc7785df19d24f4be29e5d90bad59ed993da87b
```

Exact command is stored in `gpu_preflight.json`.

## Exact Force Derivation Target

External validators should check the divergence-form operator identity:

```text
H = -D div(N grad)
d<P>/dt = -D integral grad(N) |grad psi|^2 dV
```

The exact integrated force, RHS momentum derivative, and finite-difference momentum derivative are recorded in the run tables. The exact integrated law is the primary model; the coarse-grained law is an approximation.

## Reduced-Model Feature Library

The preregistered interpretable feature library was:

- `-grad_N_at_COM`
- `-grad_lnN_at_COM`
- `-K_grad * grad_N_at_COM`
- `-source_grad_at_COM`
- `-1/r^2`
- ray feature `-D |k|^2 grad_N_at_COM`

Best reduced predictor:

```text
F ~= c (-K_grad grad_N_at_COM)
c = 0.2800836375056448
normalized RMSE = 0.09307384958591862
R2 = 0.9913372585232578
radial sign accuracy = 0.9533678756476683
```

Sign-error analysis:

```text
6 sign errors / 193 non-flat atlas rows
all small-magnitude far-tail or near-null cases
```

## Holdout Partition

Training/holdout strategy:

- train on Gaussian and super-Gaussian selected strengths and widths;
- hold out compact bump, two-lobe, shell, strong `N_min=0.35`, far distances, and unseen widths;
- reverse source-family holdout for a second check.

See:

- `holdout_validation.csv`
- `model_comparison.csv`
- `coarse_grained_coefficients.csv`

## Negative And Falsification Tests

Negative tests included:

- flat coefficient null;
- constant-`N` source-free backgrounds;
- probe internal-structure dependence;
- shell interior map;
- compact-source exterior decay;
- Newtonian `1/r^2` benchmark;
- ray Hamiltonian benchmark;
- two-lobe midpoint symmetry case.

Rejected for this spatial model:

- Newtonian exterior field;
- shell-theorem behaviour;
- universal free fall;
- global point-ray description.

## Rerun Instructions

Run from WSL2 with the established JAX GPU venv:

```bash
cd /mnt/f/quantule_mapper
. ~/jax_irer/bin/activate
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
python jax_scout/gravity_D_dynamics_characterization_gpu.py \
  --stages all \
  --long-runs enabled \
  --out sweep_runs/GRAVITY_D_DYNAMICS_CHARACTERIZATION_GPU_RERUN
```

The run must fail if `jax.default_backend() != "gpu"` or no GPU device is available.

## Bounded External Claim

The externally testable claim is:

```text
Bounded spatial coefficient gradients in the divergence-form operator generate a reproducible finite-width wave force whose exact law is gradient-energy weighted.
```

Do not externalize this as a gravity, geodesic, temporal-lapse, equivalence-principle, or IRER-source validation claim.

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `25020bf` (2026-07-22) — *dual substrate*
**Revised since:** 6 commit(s), most recently `f2b0527` (2026-08-27)

**Harness code changed since it was written:** 5 commit(s) to `jax_scout/`.
  - `06edae5` 2026-08-27 — Fix run provenance: 30% -> 86% of runs now dated
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine
  - `25020bf` 2026-07-22 — dual substrate

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Also referenced by (same date or earlier):** [[IRER_GRAVITY_REENTRY_AUDIT_AND_VALIDATION_PLAN]], [[external_validation/CODEX_EXTERNAL_VALIDATION_REPORT]]

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
