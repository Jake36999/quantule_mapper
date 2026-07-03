# Baseline Audit — Validation (Stage 1.3)

**Descriptive only.** Inventory of every validation metric: what hypothesis it tests, whether it *discriminates*
stable↔unstable, is reproducible, has a null test. Status: `DISCRIMINATING` · `NON-DISCRIMINATING` · `UNTESTED`.
The keep/demote/remove call is recorded as a **→ Stage 2 disposition (proposed)**, not an audit-time action (per
the no-redesign rule). Reference: `PHASE_C_VALIDATION_STACK_AUDIT.md`, `core_saturation_search.py`,
`validation_pipeline.py`.

## Headline — two disconnected validation paths
1. **jax-scout `css.classify` (v3 energy-stability gate)** — the **only** gate every Phase C claim actually
   passed through.
2. **`validation_pipeline.py`** — the CuPy *production* stack (11-stage: spectral prime-SSE, TDA, falsifiability,
   empirical-bridge, tensor, Monte-Carlo). Runs on **HDF5**; **never applied to the jax-scout `.npz` Phase C
   outputs**. Its two core engines (prime-SSE, TDA) *were* run post-hoc on a numpy `rho` and returned **null**.

**⇒ The closed Phase C claims rest on `css.classify` alone.** The "richer" production validation did not
validate them; where it *was* run post-hoc, its headline metrics do not discriminate stability.

## A. Active path — `css.classify` v3 gate + a\*-arc metrics (the promotion criterion)

| metric | hypothesis it tests | discriminates? | reproducible / null | → Stage 2 disposition |
|---|---|---|---|---|
| `er_fin / er_max / er_min` (energy ratios) | is the state bounded/localized? | `DISCRIMINATING` | yes / window-artefact history is its stress test | KEEP |
| `late_drift` (norm. late-half fractional change) | growth vs decay vs steady | `DISCRIMINATING` | yes | KEEP |
| `floor_ratio`, `bounded_breathing` exception (v3) | bounded breather vs floorward decay/grower | `DISCRIMINATING` | yes | KEEP (single-exemplar-calibrated — see caveat) |
| **`late_slope_10/50pct_per1k`** (a\*-arc) | long-time stationarity (slope→0) | `DISCRIMINATING` (the *sharpest* stability criterion) | yes | **KEEP / promote to primary** |
| `n_fin`, core-density | fragmentation / delocalization | `DISCRIMINATING` | yes | KEEP |
| mobility: μ=v/k, circular-COM, node-centroid, origin/well mass, drag taxonomy | inertial / relational mobility | `DISCRIMINATING` (origin-mass check prevented a COM false-positive) | yes | KEEP |

**Caveats on the active gate:** (a) calibrated on a **single exemplar** (feb) — `LATE_DRIFT_MAX=0.15`,
breathing thresholds tuned to one stable trace; (b) its window-sensitivity was a real artefact (v2 over-rejected
breathing; v2/v3 both over-reported at short T) — corrected by the late-slope→0 refinement. It is discriminating
and reproducible, but its thresholds are provisional (one calibration point).

## B. Production path — `validation_pipeline.py` (not applied to Phase C)

| metric / engine | hypothesis it tests | discriminates? | reproducible / null | → Stage 2 disposition |
|---|---|---|---|---|
| `prime_log_sse` (spectral prime SSE) | prime-harmonic resonance = stability | `NON-DISCRIMINATING` (0/60 null; 999 for all; flat across TRUE/FAIL) | yes; has phase-scramble & target-shuffle nulls | **DEMOTE to exploratory** (it is also the `aste_hunter` fitness — mismatch, §C) |
| TDA / Betti (persistent homology taxonomy) | topological structure = stability | `NON-DISCRIMINATING` (~0 persistent topology, flat) | yes | **DEMOTE to exploratory** |
| Falsifiability (phase ablation) | is structure phase-coherent? | `UNTESTED` on Phase C artifacts | — | assess or retire |
| EmpiricalBridge (JSA / C4, "SPDC") | match to quantum-optics data | `UNTESTED`; **name over-claims** (FFT≠two-photon JSA, `IRER_MATH_REFERENCE.md:276`) | — | relabel as analogy / quarantine |
| TensorValidation (symmetry / shear) | anisotropic-metric routing | `UNTESTED` here; the related Stage-B routing was `NO_SUPPORT` | — | assess vs the Stage-B null |
| StatisticalValidation (Monte-Carlo p) | significance of a match | `UNTESTED` on Phase C | — | assess |

**Adapter gap:** the orchestrated pipeline requires **HDF5** (`psi_final`/`config_hash`); Phase C emits
`.npz`/`.json`, so the full pipeline never ingested Phase C output. Only `prime_log_sse` + TDA run adapter-free
on a numpy `rho`.

## C. The `aste_hunter` objective mismatch (validation-level)
The NSGA-II hunter minimises `log_prime_sse` (+ harmonic/PCS objectives) — i.e. it optimises for the metric that
this audit marks **`NON-DISCRIMINATING`** for stability. Its directed search (SGN/ASMT) also steers
`param_a_coupling`/`param_splash_coupling`, not the stability-critical `param_a`. So the search objective and the
*validated* stability criterion (`css.classify` / late-slope) are **disconnected**. (Architecture audit 1.4 owns
the hunter code; this notes the *validation* consequence: the hunter is not scored on what actually predicts
stability.) `→ Stage 2 disposition:` align the hunter objective with the validated criterion (flagged as a tool
change needing its own re-validation, not a physics change).

## D. What validated the closed claims (honest attribution)
Every SUPPORTED Phase C claim was gated by `css.classify` (v3) + the a\*-arc late-slope/mobility metrics — all
`DISCRIMINATING`, `reproducible`, and stress-tested by the window-artefact correction. The `NON-DISCRIMINATING`
(prime/TDA) and `UNTESTED` (falsifiability/bridge/tensor/MC) metrics contributed **nothing** to the supported
claims and, where run, returned null. This is consistent and honest: the stability result stands on the metrics
that discriminate; the null/untested metrics are correctly outside the evidence chain.

## Summary
- **KEEP (discriminating, load-bearing):** `css.classify` v3 energy-stability gate + a\*-arc late-slope +
  mobility metrics.
- **DEMOTE to explicitly-flagged exploratory (non-discriminating):** prime-SSE, TDA/Betti.
- **ASSESS / RELABEL / QUARANTINE (untested here):** falsifiability, empirical-bridge (over-named), tensor, MC.
- **RECONCILE (→ Stage 2):** the two-path disconnect (production pipeline vs the actual gate), the HDF5↔npz
  adapter gap, the hunter-objective mismatch, and the single-exemplar calibration of the gate thresholds.
All dispositions are Stage-2 items — this audit changes nothing.
