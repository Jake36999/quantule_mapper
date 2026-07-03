# Validation Path Reconciliation (H6)

Resolves the two-path disconnect found in the validation audit. **No code change here** — this establishes the
canonical labelling and a decision on whether to bridge the paths later.

## The two paths
| | Path 1 — **active Phase C gate** | Path 2 — **production pipeline** |
|---|---|---|
| entry | `core_saturation_search.classify` (`css.classify`, v3) | `validation_pipeline.py` (11-stage) |
| runtime | jax_scout FP64 (WSL GPU) | CuPy production (`.venv` on this PC) |
| input | `.npz` / in-memory `er(t)`, `psi_fin` | **HDF5** (`psi_final`/`config_hash`) |
| what it does | energy-stability verdict (+ a\*-arc late-slope, mobility metrics) | spectral prime-SSE, TDA, falsifiability, empirical-bridge, tensor, Monte-Carlo |
| status for Phase C | **the only gate every closed claim passed** | **never applied** to Phase C `.npz` output |

## Findings (from the audit, restated as policy)
1. **`css.classify` is the authoritative Phase C gate.** Every SUPPORTED claim rests on it (+ late-slope /
   mobility metrics). This is the promotion criterion.
2. **`validation_pipeline.py` is a separate CuPy/HDF5 production path** and was **not** run on the Phase C
   artifacts. Its two adapter-free engines (`prime_log_sse`, TDA) *were* run post-hoc and returned **null**
   (0/60; ~0 topology) — i.e. **non-discriminating** for stability in this substrate.
3. Therefore the "richer" pipeline **did not** validate the closed claims, and must not be cited as if it did.

## Should we build the `.npz → HDF5` adapter? (decision)
**Not now — low priority.** An adapter would let the full production pipeline ingest jax_scout output, but its
headline metrics (prime-SSE, TDA) are *non-discriminating* for the current dissipative substrate, so it would add
no stability-discrimination. Build the adapter only if/when a **Phase D** capability (e.g. a transport/spectral
sector) makes those metrics discriminating again — at which point they become relevant. Recorded as a
**conditional Stage-3 item**, gated on a capability that revives spectral/topological structure.

## Labelling policy for all future runs
To prevent the two paths from being conflated again:
- **GATE metrics** — `css.classify` verdict, `late_drift`, late-window slope, mobility metrics — are the
  **certifying** metrics. Report them as the verdict.
- **POST-HOC / EXPLORATORY diagnostics** — `prime_log_sse`, TDA/Betti, empirical-bridge (JSA/C4), tensor,
  Monte-Carlo — must be labelled **exploratory / non-certifying** wherever reported, and never used to promote a
  state. (The empirical-bridge "JSA/SPDC" naming over-claims — `IRER_MATH_REFERENCE.md:276` — flag as analogy.)
- A run's `summary.json` should keep GATE fields and any exploratory fields in clearly separated sections.

## Net
Path 1 (`css.classify`) = the validated gate. Path 2 (`validation_pipeline.py`) = a production/exploratory stack,
currently disconnected and non-discriminating for Phase C. Keep them **explicitly separate and clearly labelled**;
defer any adapter/bridge until a capability makes the Path-2 metrics meaningful. The `aste_hunter` objective
mismatch (it optimises the non-discriminating prime-SSE) is a related item held under **H7** (hunter re-aim), not
actioned here.
