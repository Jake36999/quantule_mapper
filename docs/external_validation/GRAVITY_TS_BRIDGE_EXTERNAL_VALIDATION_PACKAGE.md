# Gravity TS Bridge External Validation Package

Status: third-party rerun package for the standalone temporal-spatial bridge audit.

This package supports external reproduction of three bounded claims:

```text
1. Temporal clock throttling can be reproduced as a separate N_t observable.
2. Spatial effective-medium force can be reproduced as a separate A_s operator response.
3. The tested relational mutual source weakens toward no temporal field at the constructed-source level in the raw weak-probe limit.
```

It does not assert gravity, geodesics, equivalence-principle behaviour, universal free fall, IRER gravity validation, or production readiness.

## Portable Bundle

```text
docs/external_validation/artifact_bundles/gravity_ts_bridge_audit_20260714.zip
docs/external_validation/artifact_bundles/gravity_ts_bridge_audit_20260714_zip_hash.csv
```

The bundle contains:

- bridge decomposition run artifacts;
- high-resolution bridge refinement artifacts;
- static-lapse KG temporal-equation scout artifacts;
- Claude C.1-C.3 provenance rerun artifacts;
- standalone driver scripts;
- bridge technical report;
- this external-validation note;
- `BUNDLE_MANIFEST.csv` with SHA-256 hashes for every bundled file.

## Primary Run Artifacts

```text
sweep_runs/GRAVITY_TS_BRIDGE_GPU_20260714_065422/
sweep_runs/GRAVITY_TS_BRIDGE_REFINEMENT_GPU_20260714_085110/
sweep_runs/GRAVITY_TS_TEMPORAL_KG_GPU_20260714_084143/
sweep_runs/GRAVITY_C_SERIES_PROVENANCE_20260714_092539/
```

## Environment

The GPU runs record:

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

Each GPU run has:

- `gpu_preflight.json`
- `environment_versions.json`
- `git_state_before.txt`
- `artifact_hashes.csv`

The C-series provenance rerun has:

- `environment_versions.json`
- `command_manifest.csv`
- captured stdout/stderr logs;
- copied C.1/C.2/C.3 summary JSON files;
- `artifact_hashes.csv`.

## Rerun Commands

Run from WSL2 with the established JAX environment:

```bash
cd /mnt/f/quantule_mapper
. ~/jax_irer/bin/activate
```

Bridge decomposition:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
python jax_scout/gravity_TS_temporal_spatial_bridge_gpu.py \
  --out sweep_runs/GRAVITY_TS_BRIDGE_GPU_RERUN \
  --arms all \
  --source objective,relational \
  --durations short,standard \
  --stop-on-preflight-fail true
```

Bridge refinement:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
python jax_scout/gravity_TS_bridge_refinement_gpu.py \
  --out sweep_runs/GRAVITY_TS_BRIDGE_REFINEMENT_GPU_RERUN \
  --N 128 \
  --L 40 \
  --dt 0.0005 \
  --T 1.0 \
  --sample-dt 0.05 \
  --source objective,relational \
  --arms all
```

Temporal KG scout:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
python jax_scout/gravity_TS_temporal_kg_audit_gpu.py \
  --out sweep_runs/GRAVITY_TS_TEMPORAL_KG_GPU_RERUN \
  --N 64 \
  --L 30 \
  --dt 0.001 \
  --T 4 \
  --sample-dt 0.01 \
  --mass 12 \
  --probe-width 2.0
```

C-series provenance rerun:

```bash
python jax_scout/gravity_C_series_provenance_wrapper.py \
  --out sweep_runs/GRAVITY_C_SERIES_PROVENANCE_RERUN \
  --phi-iso sweep_runs/PHASE_D_C2_8_TWONODE_CODEX_20260709_123611/phi_iso.npy \
  --L 20
```

## Acceptance Checks

External validators should verify:

- GPU preflight says `backend=gpu` and `device=cuda:0` or another GPU device.
- Flat bridge arms have zero force and zero clock shift.
- Temporal-only bridge arms have clock shift and zero spatial force.
- Spatial-only bridge arms have nonzero inward force and no clock shift.
- Temporal-spatial bridge arms preserve the spatial force without double-counting.
- Objective weak-probe constructed-lapse response is amplitude-invariant at matched position.
- Objective evolved KG packet shows a measured near/flat and near/far frequency difference.
- The measured KG packet frequency does not yet match the constructed weighted lapse; this calibration gap must be reported.
- Relational weak-probe constructed field weakens strongly with raw probe amplitude.
- Relational evolved KG packet rates are not yet a clean monotonic weak-probe clock result.
- Static-lapse KG energy drift stays near `1e-10` for the scout runs.
- C.1-C.3 summaries reproduce `C1_FROZEN_PASS`, `C2_SOURCE_MAP_DONE`, and `C3_BACKREACTION_STABLE`.
- C.1 is interpreted as shared frozen-lapse clock consistency, not source-layer objective-lapse emergence.
- C.3's original `freq_monotone_with_lam=false` field is a reporting-condition bug; the frequency rows increase from the frozen value to the self-relieved value.

## Bounded External Claim

The externally testable bridge claim is:

```text
The current mirror evidence separates temporal clock throttling from the spatial effective-medium force. A static temporal coefficient inside a KG equation produces a reproducible near/far oscillation-frequency difference, while the tested raw relational mutual source collapses toward no temporal field in the weak-probe constructed-source limit.
```

Do not externalize this as gravity, geodesic validation, equivalence-principle confirmation, or production readiness.

Current recommended labels:

```text
TEMPORAL_THROTTLING_C_SERIES_REPRODUCED
SPATIAL_MEDIUM_FORCE_REPRODUCED
TEMPORAL_SPATIAL_DECOMPOSITION_PASSED
TEMPORAL_KG_LAPSE_EQUATION_IMPLEMENTED
OBJECTIVE_TEMPORAL_KG_DIFFERENTIAL_RESPONSE_DETECTED
RELATIONAL_WEAK_PROBE_FIELD_COLLAPSE_REPRODUCED
TEMPORAL_KG_CLOCK_RATE_CALIBRATION_OPEN
TEMPORAL_SPATIAL_METRIC_UNIFICATION_OPEN
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

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
