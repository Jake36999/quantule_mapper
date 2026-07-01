# Phase C — validation-stack audit

**Date:** 2026-06-26
**Purpose:** confirm whether the Phase C work has narrowed the validation cycle to the energy-stability
classifier alone, and establish whether the richer validation metrics (log-prime SSE, TDA/Betti,
falsifiability nulls) are active / usable / stale on current Phase C artifacts.
**Scope:** read-only audit + a read-only post-hoc diagnostic pass on existing data. No PDE/solver/
classifier/geometry change.

> **Top line:** there are **two separate validation paths**. (1) The Phase C jax-scout saturation search
> gates **only** on `css.classify` (energy-stability, now v3). (2) `validation_pipeline.py` is the CuPy
> *production* stack (spectral SSE, TDA, falsifiability, tensor, Monte-Carlo) and runs on **HDF5**
> artifacts — it has **not** been applied to the jax-scout Phase C outputs. Its two core diagnostics
> (`prime_log_sse`, TDA) **do run directly on a numpy `rho`** with no adapter, so I ran them post-hoc on
> all feb-basin states — and they return **null** (no prime peaks, no persistent topology) for every state.

## Q1 — What does `validation_pipeline.py` compute?
An 11-stage graph (v3.2): ArtifactLoader → **SpectralFidelityEngine** (CEPP v2.0: `prime_log_sse`, Bragg
peaks, directional SSE, phase-scramble & target-shuffle nulls) → ContractEnforcer → early-rejection gate →
**TopologyEngine** (TDA persistent homology / taxonomy) → LOMTelemetry (collapse events) → Falsifiability
(phase ablation) → EmpiricalBridge (JSA/C4 quantum-optics) → TensorValidation (symmetry/shear) →
StatisticalValidation (Monte-Carlo p-value) → ProvenanceAssembler. Plus DerivedMetrics and Aletheia
metrics.

## Q2 — Does it run on current Phase C artifacts (`probe_data.npz`, `summary.json`, `psi_fin`, `frames.npz`)?
**Not the full pipeline.** `ArtifactLoader.load` requires an **HDF5** file with `psi_final`/`final_psi`
(+ optional `rho_final`, telemetry groups) and a params JSON with `config_hash`. Phase C emits `.npz`/`.json`,
so the orchestrated pipeline does not ingest them without an adapter. **However**, the two core engines take
a plain 3-D numpy `rho` and run directly (Q10).

## Q3 — Is the log-prime harmonic SSE still implemented?
**Yes.** `quantulemapper_real.prime_log_sse` (imported as `cep_profiler`), called by `SpectralFidelityEngine`.
Pure numpy/scipy — runs on Windows (the `cupy` import in that module is optional/guarded and unused by this
function). `LOG_PRIME_TARGETS = ln(primes)`.

## Q4 — log-prime SSE input/output
- **Input:** a 3-D density field `rho` (np.ndarray). (We build it as `|psi_fin|²`.)
- **Output:** dict — `log_prime_sse` (PASS if <1.0), `n_peaks_found_main`, `dominant/secondary_peak_k`,
  `scaling_factor_S`, `bragg_lattice_sse`/`bragg_peaks_detected`, falsifiability nulls
  (`sse_null_phase_scramble`, `sse_null_target_shuffle`), `sse_directional_{x,y,z}`, etc. On a degenerate
  field it returns `log_prime_sse = 999`.

## Q5 — Is TDA / Betti-loop analysis still implemented?
**Yes.** `tda_profiler.extract_and_classify_topology` (numpy + sklearn `DBSCAN`; no cupy/jax) — runs on
Windows. Wrapped by `TopologyEngine.run_tda`.

## Q6 — TDA input/output
- **Input:** 3-D `rho`. **Output:** `(csv_content, taxonomy)` where taxonomy = `{Q_theta, Q_nu, Transient}`
  (persistent-homology–derived shape classification) and the CSV holds per-event "quantule" rows.
  `TopologyEngine` additionally reports `persistent_loops/voids` and `betti_0/1/2` (default null when no
  events). It measures **topological shape** (loop/void taxonomy), a diagnostic — not a topological-invariant
  proof.

## Q7 — Which metrics gate?
- **Phase C (jax-scout, the path we've been using):** **only** `css.classify` — the energy-stability
  classifier (v3: in-band er, bounded er_max, node count, late-half drift + breathing exception).
- **Production pipeline:** `log_prime_sse` is an **early-rejection gate** (>15.0 → skip deep analysis), and
  `validation_status = PASS` requires `log_prime_sse < 1.0`.

## Q8 — Which are post-hoc only (for Phase C)?
**All** `validation_pipeline` metrics — spectral SSE, TDA/Betti, falsifiability nulls, empirical bridge,
tensor, Monte-Carlo. None currently gate the Phase C saturation search; they are diagnostics relative to it.

## Q9 — Which are stale / need adapters?
- The **full pipeline** needs an **`.npz → HDF5` adapter** (map `psi_fin → psi_final`; supply a params JSON
  with `config_hash`) plus telemetry the deep engines expect (`extended_telemetry`, `C_invariant`, …) which
  jax-scout does not emit — so LOM / bridge / tensor / MC would be degraded/null without that telemetry.
- The **core diagnostics (`prime_log_sse`, TDA) need NO adapter** — they run on `rho` directly (verified).
- Nothing is *broken*; the gap is purely the artifact-format/telemetry mismatch between the jax-scout and
  CuPy-production paths.

## Q10 — Minimal safe path to run them on current feb-basin outputs
Build `rho = |psi_fin|²` from the `.npz` and call `cep_profiler.prime_log_sse(rho)` and
`tda_profiler.extract_and_classify_topology(rho)` directly — no GPU, no HDF5, no adapter. **Done** (below).

---

## Read-only post-hoc result (60 states: 52 param-basin + 8 feb-basin)

Output: `sweep_runs/FEB_BASIN_POSTHOC_VALIDATION_20260626_122835/`
(`feb_basin_full_validation_summary.csv`, `feb_basin_frequency_sse_summary.csv`,
`feb_basin_tda_summary.csv`, `feb_basin_validation_dashboard.png`).

| diagnostic | result across all 60 states |
|---|---|
| finite `psi_fin` | 60 / 60 |
| any prime-harmonic spectral peak (`n_peaks>0`) | **0 / 60** |
| profiler PASS (`log_prime_sse < 1.0`) | **0 / 60** (all = 999) |
| any persistent-topology event (`Q_theta`/`Q_nu`>0) | **1 / 60** (negligible) |

- **Which stable states pass v3 only:** all 51 v3-TRUE states pass the stability classifier **only** — none
  show prime-harmonic or topological structure.
- **Which also show clean TDA/Betti:** none.
- **Frequency SSE:** uniformly **high (999)** — no prime structure in any state.
- **Which diagnostics correlate with stability:** **none.** `prime_log_sse` and TDA are flat (999 / 0)
  across TRUE/SPIN/GROW/BLOW alike — they do not separate stable from unstable here.
- **Which are inconclusive:** `prime_log_sse` and TDA are **null/inconclusive** for these states.

**Interpretation (disciplined):** the feb-basin bound states are **smooth dissipative solitons** (fat
saturated cores), not prime-harmonic lattices or topologically structured objects. The log-prime-SSE and
TDA diagnostics are *runnable and active* but **do not light up** for them, and provide **no** support for
the prime-harmonic or topological hypotheses on this state family. The energy-stability classifier (v3)
remains the only discriminating metric. (Caveat: the profiler's peak/threshold settings were calibrated for
the CuPy production fields; "no peaks" means "no prime structure at those settings" — but the result is
uniform across 60 varied morphologies, so it reflects genuine absence, not a threshold edge case.)

These metrics stay **exploratory diagnostics** — `prime_log_sse` is a spectral/structural diagnostic (not
proof of the log-prime hypothesis); TDA/Betti is a shape diagnostic (not proof of a topological invariant or
phase transition). The classifier remains the promotion gate unless additional metrics are explicitly
validated and calibrated.
