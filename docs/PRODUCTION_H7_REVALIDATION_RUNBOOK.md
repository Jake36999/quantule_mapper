# Production H7 Re-Validation Runbook (A5) — local CuPy (`.venv`)

**Status:** *prepared, not yet run.* The harness + evaluator are built and unit-tested
([`tools/production_h7_revalidation.py`](../tools/production_h7_revalidation.py),
`tests/test_production_h7_revalidation.py` 6 pass, no cupy). **All steps run locally on this PC** — the worker +
validate steps use the repo **`.venv`** (`cupy 14.0.1`, GTX 1080); there is **no separate machine**. Prefix CuPy
commands with `.venv/Scripts/python.exe` (NOT the PATH system python). This is the last A-track gate before the
production Hunter re-aim can be called *validated*. (A1/H4 parity has already PASSED here — `PARITY_WITHIN_TOL`.)

**Question (production analog of the jax_scout re-discovery PASS):** with production `stability_metrics` now
flowing worker → HDF5 → `validation_pipeline` → provenance → Hunter (A3/A4/A4b), does the re-aimed objective
re-find **a\*≈×1.15** as the certifiable long-time attractor on the CuPy path, rank the matched controls below it,
and **refuse to certify a short-window (T=12000) run**? No physics/geometry/gate/kinetic change — a fixed replay.

## Prerequisites
- **A3/A4/A4b done** (this repo): worker emits `/stability_metrics`; `validation_pipeline` carries it into the
  provenance report; `aste_hunter` resolves the folded provenance filename. 
- **Recommended first: A1/H4 CuPy bit-parity** (`tools/solver_parity_check.py run --backend cupy` + `compare` vs
  `parity/jax_ref.npz`) so the production solver is confirmed to match the jax_scout mirror the a\* result came from.

## Cell set (a\* + matched controls + a window-artifact probe)
feb params are frozen (`= core_saturation_search.FEB`); only `param_a` varies. `N=96, dt=0.005, L=10, seed=20260619`.

| cell | role | param_a | T | expected |
|---|---|---|---|---|
| `astar_longT` | a\* target | 0.55223 (feb×1.15) | 36000 | **certifiable, top-ranked** |
| `decayer_longT` | matched control | 0.50421 (feb×1.05) | 36000 | below a\* (slow decay) |
| `grower_longT` | matched control | 0.60025 (feb×1.25) | 36000 | below a\* (grows / band-penalised) |
| `astar_shortT` | window-artifact probe | 0.55223 (feb×1.15) | 12000 | **NOT certifiable** (window gate) |

## Steps (all local; PY=.venv python for the CuPy steps)
```bash
PY=".venv/Scripts/python.exe"   # the repo venv with cupy 14.0.1; NOT the PATH system python

# 0. write the worker configs (numpy only)
python tools/production_h7_revalidation.py build-configs --out a5_configs
#    -> a5_configs/<cell>.params.json  (worker_cupy --params shape) + cells_index.json

# 1. run each cell through the production worker in .venv (writes HDF5 with /stability_metrics)
for cell in astar_longT decayer_longT grower_longT astar_shortT; do
  CUDA_VISIBLE_DEVICES=0 "$PY" worker_cupy.py --params a5_configs/$cell.params.json --output a5_artifacts/$cell.h5
done

# 2. validate each artifact in .venv -> provenance_reports/provenance_{config_hash}[_folded].json
#    (validation_pipeline carries /stability_metrics into the report; A4)
for cell in astar_longT decayer_longT grower_longT astar_shortT; do
  "$PY" validation_pipeline.py a5_artifacts/$cell.h5
done

# 3. score the provenance reports and print the verdict (numpy only)
python tools/production_h7_revalidation.py evaluate --provenance-dir provenance_reports --json-out a5_verdict.json
```

## Verdict (PASS criteria — all must hold)
- `astar_longT` is **certifiable** (`T≥24000`, late-slope≈0, er in band);
- `astar_longT` score **≥** both controls (top of the long-T set);
- `decayer_longT` and `grower_longT` score **below** a\*;
- `astar_shortT` is **not certifiable** — the window gate refuses to promote a short-window run.

→ `PRODUCTION_H7_REVALIDATION_PASS`. Any miss → `REVIEW` with the failed checks named. **FAIL triggers** (mirroring
the jax_scout spec): a control out-ranks a\*, or a short-window run certifies, or a\* is not certifiable.

## Caveats (read before interpreting)
- **Cross-IC re-validation (important).** The production worker's initial condition is a **single Gaussian packet +
  noise** (`solver/run.py:initialize_psi`), *not* the jax_scout multiseed-blob IC the a\* basin was mapped with. So
  this is a genuine **cross-IC** test of a\*. If a\* fails to reach the balance under the production IC, that is a
  *finding* (IC-sensitivity, or a real limit), **not** a bug to patch — do **not** modify the solver IC to force a
  pass; investigate and report. The objective's er is self-normalised, so its *shape* metrics are IC-agnostic; only
  whether the production IC lands in the a\* basin is at issue.
- **The gate still certifies.** `css.classify` / the production validation gate remains the promotion certifier;
  this harness scores the re-aimed *objective's* re-find, it does not replace the gate.
- **Numerics** are set for jax_scout parity (`N=96, dt=0.005, L=10, T=36000≥MIN_STABILITY_T`); the a\* long-T
  result stands at T=36000/72000/144000.

## On success
Only after this PASSes (locally in `.venv`) may the status advance from `HUNTER_REAIM_OBJECTIVE_VALIDATED_ON_JAX_SCOUT` to a
production-validated re-aim. Until then: `HUNTER_PRODUCTION_DEPLOYMENT_PENDING`.
