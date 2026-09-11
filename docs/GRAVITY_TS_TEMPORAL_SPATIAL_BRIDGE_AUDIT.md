# Gravity TS Temporal-Spatial Bridge Audit

Status: standalone mirror audit. Production gravity remains closed.

This bridge audit keeps Claude's C-series temporal throttling branch separate from the Gravity D spatial effective-medium branch:

```text
temporal chronology factor: N_t(x)
spatial kinetic coefficient: A_s(x)
```

It does not claim gravity, geodesic motion, equivalence-principle behaviour, IRER gravity validation, or production readiness.

## Evidence

Primary bridge decomposition run:

```text
sweep_runs/GRAVITY_TS_BRIDGE_GPU_20260714_065422/
```

GPU evidence:

```text
backend = gpu
device = cuda:0
JAX = 0.10.2
jaxlib = 0.10.2
x64 = true
```

Four-arm result:

| source | duration | flat force | flat clock shift | temporal-only clock shift | temporal-only force | spatial force | combined force delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| objective | short | 0 | 0 | 4.127887e-03 | 0 | 6.211596e-03 | 0 |
| objective | standard | 0 | 0 | 9.752609e-03 | 0 | 6.211596e-03 | 0 |
| relational | short | 0 | 0 | 1.764074e-02 | 0 | 4.308223e-03 | 0 |
| relational | standard | 0 | 0 | 1.179074e-02 | 0 | 4.308223e-03 | 0 |

Force-contract closure at standard duration:

| run | radial force | RHS residual | finite-difference residual |
| --- | ---: | ---: | ---: |
| objective spatial-only | -6.211596e-03 | 3.570735e-17 | 5.708866e-10 |
| objective temporal-spatial | -6.211596e-03 | 3.570735e-17 | 5.708866e-10 |
| relational spatial-only | -4.308223e-03 | 4.859252e-17 | 7.112475e-10 |
| relational temporal-spatial | -4.308223e-03 | 4.859252e-17 | 7.112475e-10 |

Weak-probe check:

| source | amplitude 1.0 clock shift | amplitude 0.1 clock shift | interpretation |
| --- | ---: | ---: | --- |
| objective | 4.127887e-03 | 4.127887e-03 | amplitude-invariant at matched position |
| relational | 7.775720e-02 | 1.046068e-03 | weak raw probe loses most temporal field |

Original bridge-run labels, superseded/tightened by post-review G0:

- `TEMPORAL_THROTTLING_REPRODUCED`
- `SPATIAL_MEDIUM_FORCE_REPRODUCED`
- `TEMPORAL_SPATIAL_DECOMPOSITION_PASSED`
- `OBJECTIVE_WEAK_PROBE_LAPSE_FIELD_LIMIT_REPRODUCED`
- `RELATIONAL_WEAK_PROBE_FIELD_COLLAPSE_REPRODUCED`

## Numerical Refinement

Refinement run:

```text
sweep_runs/GRAVITY_TS_BRIDGE_REFINEMENT_GPU_20260714_085110/
```

Configuration:

```text
N = 128
L = 40
dt = 0.0005
T = 1.0
```

Refined arm comparison:

| source | flat force | temporal-only clock shift | temporal-only force | spatial force | combined force delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| objective | 0 | 4.486118e-03 | 0 | 6.211596e-03 | 0 |
| relational | 0 | 1.676835e-02 | 0 | 4.308223e-03 | 0 |

Refined force-contract residuals remained small:

```text
objective spatial RHS residual = 8.921823e-17
objective spatial FD residual  = 1.427184e-10
relational spatial RHS residual = 5.687470e-17
relational spatial FD residual  = 1.778174e-10
```

## True Temporal-Equation Scout

Temporal KG run:

```text
sweep_runs/GRAVITY_TS_TEMPORAL_KG_GPU_20260714_084143/
```

This run evolved a static-lapse Klein-Gordon mirror equation:

```text
phi_t = N_t Pi
Pi_t  = div(N_t grad phi) - N_t m^2 phi
```

This is the first bridge audit that evolves a temporal lapse equation rather than only measuring a clock factor. It is still a standalone mirror scout, not a validated production metric.

Primary diagnostics:

| run | weighted expected rate | measured packet zero-crossing rate | energy relative error |
| --- | ---: | ---: | ---: |
| flat near | 1.000000 | 1.001733 | 1.671889e-10 |
| objective near | 0.878139 | 0.986507 | 1.073196e-10 |
| objective far | 0.986458 | 0.999437 | 1.588856e-10 |
| relational amp 1.0 | 0.911673 | 0.999032 | 1.233109e-10 |
| relational amp 0.1 | 0.998782 | 0.999933 | 1.659769e-10 |

The exact weighted-lapse diagnostics show objective near/far redshift structure and amplitude invariance. The packet zero-crossing rate is much less sensitive because the finite KG packet includes mass/gradient dispersion. It should be treated as a clock-readout caveat, not as a falsification of the static-lapse equation.

KG bounded labels after review tightening:

- `TEMPORAL_KG_LAPSE_EQUATION_IMPLEMENTED`
- `OBJECTIVE_TEMPORAL_KG_DIFFERENTIAL_RESPONSE_DETECTED`
- `RELATIONAL_WEAK_PROBE_FIELD_COLLAPSE_REPRODUCED`
- `TEMPORAL_KG_CLOCK_RATE_CALIBRATION_OPEN`

Review caveat:

```text
The constructed weighted lapse is not yet validated as the measured clock rate.
```

For the objective near packet:

```text
weighted expected rate = 0.878139
measured zero-crossing rate = 0.986507
difference = 0.108368
```

The evolved KG equation therefore detects a temporal response, but the current finite-packet zero-crossing observable is not calibrated to `N_t` or to the weighted lapse. The measured near/far and near/flat differences are the current non-tautological temporal evidence:

```text
flat near measured rate - objective near measured rate = 0.01523
objective far measured rate - objective near measured rate = 0.01293
```

The relational weak-probe collapse is currently established at the constructed field level. The measured KG rates for relational amplitudes are not monotonic, so relational temporal KG dynamics remain inconclusive.

## C-Series Provenance Rerun

C-series provenance run:

```text
sweep_runs/GRAVITY_C_SERIES_PROVENANCE_20260714_092539/
```

The wrapper reran Claude's original C.1-C.3 scripts without changing their implementations and captured exact commands, stdout, summaries, environment, git state, and hashes.

Results:

- C.1: `C1_FROZEN_PASS` as historical script verdict; current interpretation is `SHARED_FROZEN_LAPSE_CLOCK_CONSISTENCY`
- C.2: `C2_SOURCE_MAP_DONE`
- C.3: `C3_BACKREACTION_STABLE`

Important C.2 caution:

```text
own-N width sweep rate spread = 0.7279185915093409
converging_toward_point_value = false
```

Important C.3 result:

```text
lam=0.0 freq/omega = 0.8000117106452346
lam=1.0 freq/omega = 0.8889019007169273
dt refinement = stable
```

The original C.3 summary field `freq_monotone_with_lam = false` is a reporting-condition bug: the script checks for decreasing frequency. The rows increase monotonically from the frozen value toward the self-relieved value. This is annotated in the provenance wrapper for future reruns and should not be treated as a physics failure.

## Inference

The bridge audit supports a decomposition, not a unification:

```text
N_t controls temporal clock/lapse observables.
A_s controls spatial effective-medium force observables.
```

The objective source behaves like a reproducible supplied temporal field in weak-probe amplitude checks. The relational mutual source remains probe-dependent and weakens toward no temporal field for raw weak probes, so it is currently unsuitable as a universal gravity source.

This statement is bounded to the tested raw mutual-source form. It is not a proof that every possible relational or nonlocal source is unsuitable for gravity.

The spatial force remains the already-characterized Gravity D mechanism:

```text
d<P>/dt = -D integral grad(A_s) |grad psi|^2 dV
```

## Rejected Interpretations

This audit does not establish:

- gravity confirmed;
- geodesic motion;
- equivalence principle;
- universal free fall;
- Newtonian exterior field;
- IRER gravity source;
- production geometry.

The restricted `N_t = A_s` common-field rows are only a separate hypothesis check. They are not a metric-unification claim.

## Proposed Next Action

The smallest next experiment is a temporal clock calibration audit:

1. Compute the static-lapse KG eigenfrequency benchmark:

```text
omega^2 u = -N_t div(N_t grad u) + N_t^2 m^2 u
```

2. Compare evolved packet frequencies against those eigenfrequencies rather than against the constructed weighted lapse alone.
3. Build a localized clock mode or high-mass/WKB clock whose measured frequency can be calibrated.
4. Refine the KG branch at `N=64,L=30,dt=0.001`, `N=96,L=30,dt=0.0005`, and `N=128,L=40,dt=0.0005`.
5. Only then implement a combined metric branch with independently specified `N_t(x)` and spatial factor `a_s(x)`.

For a metric of the form:

```text
ds^2 = -N_t(x)^2 dt^2 + a_s(x)^2 dx^2
```

one candidate first-order KG system is:

```text
phi_t = (N_t / a_s^3) Pi
Pi_t  = div(N_t a_s grad phi) - N_t a_s^3 m^2 phi
```

The mapping between the validated Gravity D coefficient `A_s` and any metric spatial factor `a_s` is a new hypothesis and must not be assumed.

## Current Verdict

```text
Temporal dynamics detected.
Spatial dynamics confirmed.
Temporal-spatial decomposition passed.
Full temporal clock calibration and metric unification remain open.
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
