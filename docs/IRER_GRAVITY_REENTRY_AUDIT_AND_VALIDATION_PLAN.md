# IRER Gravity Re-Entry Audit And Validation Plan

Status: design-only. Production gravity remains closed.

This file is the current re-entry ledger after the Gravity D standalone spatial effective-medium characterization. It replaces the older mixed-encoding draft, which contained stale conclusions from intermediate audits. Nothing here changes the production solver, Hunter, validation gates, production geometry, GPU launch infrastructure, or CPU/GPU fallback policy.

## Current Bottom Line

The Gravity D standalone mirror has characterized a real spatial effective-medium mechanism:

```text
i d_t psi = -D div(N(x) grad psi)
d<P>/dt = -D integral grad(N) |grad psi|^2 dV
F_cg ~= -D K_grad grad N(R)
```

Here `N(x)` is a spatial coefficient, bounded kinetic coefficient, or effective-medium field. It is not a temporal lapse in this tested model.

Banked labels:

- `D_SPATIAL_EFFECTIVE_MEDIUM_ATTRACTION_GPU_CONVERGED`
- `D_SPATIAL_EFFECTIVE_MEDIUM_ROBUSTNESS_FIRST_PASS_CLOSED`
- `D_EFFECTIVE_MEDIUM_DYNAMICS_CHARACTERIZED`
- `D_GRADIENT_ENERGY_WEIGHTED_FORCE_SUPPORTED`
- `D_FINITE_WIDTH_WAVE_FORCE_CONFIRMED`
- `D_NEWTONIAN_COMPARISON_REJECTED`
- `D_NO_UNIVERSAL_FREE_FALL_LIMIT_FOUND`
- `D_REDUCED_MODEL_PARTIAL`

This does not reopen production gravity. It does not validate temporal-lapse gravity, geodesic motion, universal free fall, an inverse-square vacuum field, an IRER gravity source, or production readiness.

## Current Audit Ledger

### Audit A - Explicit Interaction Diagnostics

Current status:

- Explicit A/B interaction intensity is validated as a lawful interaction diagnostic within an explicit subsystem/environment model.
- It is deterministic, decomposition-controlled within that A/B setup, and predictive for some interaction observables.
- Later audits showed it is an overlap-family quantity in the tested settings.
- It is not a validated universal gravity source.
- It is not a validated self-sourcing single-field gravity observable.
- It must not be described as closing the source problem for gravity.

Current interpretation:

```text
I_int = useful explicit-subsystem interaction diagnostic
I_int != universal gravity source
I_int != temporal lapse source by itself
```

### Audit B/C - Throttling Lapse And Clock Path

Current status:

- The temporal throttling branch was implemented and tested, not merely sketched.
- C.1 used two different clock mechanisms driven by a shared frozen `N_A|E`; it showed shared supplied-lapse response consistency.
- C.2 let clocks generate their own relational `N_A|E`; it showed source/probe dependence, overlap-family degeneracy in tested clock observables, and weak-probe non-universality.
- C.3 implemented dynamic feedback `psi_A -> I_A|E -> N_A|E -> d_t psi_A`; the backreaction run was stable and timestep-convergent.
- Increasing backreaction partially self-relieved the throttling loop, shifting the fractional clock frequency from the frozen value near `0.80` toward about `0.889`.
- This is computationally real temporal-throttling evidence, but not yet a successful gravity source.
- Its GR-divergent probe dependence is currently a warning sign unless it survives matched normalization, compact-probe and source-exposure controls.

Current interpretation:

```text
temporal throttling branch = implemented and stable in mirror tests
mutual I_A|E source = cautionary / partial for gravity
objective temporal-lapse emergence = open / not established
```

### Audit D - Spatial Effective-Medium Branch

Current status:

- A bounded environment-defined spatial coefficient creates a robust finite-width wave force.
- The exact force law is gradient-energy weighted:

```text
F = -D integral grad(N) |grad psi|^2 dV
```

- The useful leading coarse-grained approximation is:

```text
F_cg ~= -D K_grad grad N(R)
```

- The reduced model is partial, not exact: six sign errors out of 193 non-flat atlas rows, all far-tail or near-null cases.
- The mechanism is probe-structure dependent and therefore not universal free fall.
- The mechanism is short-ranged for compact coefficient profiles and does not create a Newtonian exterior field.
- Smooth shell tests show local-medium shell response, not Newtonian shell-theorem behaviour.
- A global point-ray/geodesic-like model fails as a description of the characterized atlas.

Current interpretation:

```text
Gravity D spatial branch = characterized effective-medium wave force
Gravity D spatial branch != temporal gravity
Gravity D spatial branch != Newtonian gravity
Gravity D spatial branch != geodesic validation
Gravity D spatial branch != IRER source validation
```

## Supported, Rejected, And Open

Supported for the standalone spatial model:

- Bounded spatial coefficient gradients generate a reproducible finite-width wave force.
- The exact operator-force identity closes.
- The response is robust across source shapes, strengths, orientations, translations, amplitudes, carriers, and representative numerical refinements.
- The dominant coarse-grained predictor is gradient-energy weighted: `K_grad grad N`.

Rejected for the standalone spatial model:

- Newtonian inverse-square exterior field.
- Newtonian shell-theorem behaviour.
- Universal free fall.
- Global point-ray description.
- Relativistic geodesic validation.
- Temporal-lapse gravity.
- IRER gravity-source confirmation.

Open and separate hypotheses:

- Temporal-spatial bridge between `N_t(x)` chronology and `A_s(x)` spatial wave response.
- Metric-consistent temporal lapse.
- Nonlocal environment field.
- IRER-derived source semantics.
- A full metric with separately specified temporal and spatial components.
- Production re-entry after a validated source and metric are independently specified.

## Temporal-Geometric Feedback Loop - Codex Addendum 2026-07-14

Codex added a formalization note at:

```text
docs/theory_synthesis/IRER_TEMPORAL_GEOMETRIC_FEEDBACK_LOOP.md
```

This note preserves Jake McIntosh's conceptual authorship and consolidates a loop already distributed across the
original IRER concepts:

```text
OIW phase locking
-> RD/PAS rise
-> resolution activity
-> chronology / temporal load
-> geometric or manifold response
-> modified OIW propagation
-> outgoing perturbation or relaxation channel
```

This is more ambitious than the current temporal-spatial bridge. The bridge audit decomposes supplied or constructed
`N_t` and `A_s`. The feedback-loop hypothesis requires both temporal and geometric fields to be dynamically updated
from the OIW/resolution state and to feed back into later OIW evolution.

Current status:

- The full `Psi -> R_res -> T -> G -> Psi` loop is **not implemented**.
- C-series temporal throttling is a tested limb of the loop.
- Gravity D spatial effective-medium force is a tested limb of the loop.
- The temporal-spatial bridge is a decomposition, not dynamic mutual generation.
- G1 shows temporal clock calibration is still open after the first local-clock instrument failed.

The next valid branch is design-first and is deliberately named outside the gravity-maturity `G2` sequence:

```text
TG-A_FEEDBACK_ACTION_AND_SOURCE_AUDIT
TG-B_REDUCED_FEEDBACK_SCOUT
```

This does not replace `G1b` objective local-clock calibration. A feedback-loop result may characterize a
temporal-geometric field mechanism, but it cannot establish objective time dilation while the clock instrument
remains uncalibrated.

The scout should start in 1D radial or 2D axisymmetric form, preferably KG-first, and test whether resolution-source
activity produces an ordered sequence:

```text
R_res peak -> T pulse -> G pulse -> outgoing flux -> reduced local load
```

No photon, gravity, geodesic, universal-free-fall, or production claim follows from adding this hypothesis. A photon
analogue would require a separately defined outgoing-field observable, reproducible emission spectrum, and energy or
action accounting.

## Required Temporal-Spatial Bridge

The temporal C-series and spatial Gravity D branch are currently separate:

| branch | implemented object | tested interpretation |
| --- | --- | --- |
| Claude C-series | `N_A|E = 1/(1 + beta I_A|E)` in clock evolution | relational chronology throttling |
| Gravity D | `A_s(x)` inside `-D div(A_s grad psi)` | spatial effective-medium force |

The next audit must not assume that one field belongs in both slots. It must separately test `N_t(x)` and `A_s(x)` before attempting a common-field or metric-consistent model.

Required four-arm decomposition:

| arm | temporal component | spatial component | purpose |
| --- | --- | --- | --- |
| `FLAT` | `N_t = 1` | `A_s = 1` | removes clock shift and medium force |
| `TEMPORAL_ONLY` | `N_t(x) != 1` | `A_s = 1` | tests objective or relational chronology changes |
| `SPATIAL_ONLY` | `N_t = 1` | `A_s(x) != 1` | reproduces the characterized finite-width medium force |
| `TEMPORAL_SPATIAL` | `N_t(x) != 1` | `A_s(x) != 1` | tests whether both observables coexist without double-counting |

Keep source ladders separate:

- Objective environment source: `S_E(x)` generates a probe-independent `N_t(x)` and/or `A_s(x)`.
- Relational IRER source: `I_A|E(x)` generates probe-dependent `N_A|E` and tests whether chronology remains relational after matched controls.

Only after those arms pass should a restricted common-field hypothesis be tested:

```text
N_t(x) = A_s(x)
```

That common-field case is a separate hypothesis, not the default bridge.

### Bridge Audit Execution - 2026-07-14

Codex executed the first standalone GPU bridge audit in:

```text
sweep_runs/GRAVITY_TS_BRIDGE_GPU_20260714_065422/
```

The run used the verified WSL/JAX GPU mirror environment (`backend=gpu`, `device=cuda:0`, x64 enabled). It did not modify production geometry, Hunter, launch infrastructure, CPU/GPU fallback policy, protected validation files, or production verdicts.

Bounded labels recorded by the bridge audit:

- `TEMPORAL_THROTTLING_REPRODUCED`
- `SPATIAL_MEDIUM_FORCE_REPRODUCED`
- `TEMPORAL_SPATIAL_DECOMPOSITION_PASSED`
- `OBJECTIVE_WEAK_PROBE_LAPSE_FIELD_LIMIT_REPRODUCED`
- `RELATIONAL_WEAK_PROBE_FIELD_COLLAPSE_REPRODUCED`

Post-review tightened temporal KG labels:

- `TEMPORAL_KG_LAPSE_EQUATION_IMPLEMENTED`
- `OBJECTIVE_TEMPORAL_KG_DIFFERENTIAL_RESPONSE_DETECTED`
- `TEMPORAL_KG_CLOCK_RATE_CALIBRATION_OPEN`

The audit result should be read narrowly:

- The temporal-only arm reproduced clock-rate differences with `A_s=1`.
- The spatial-only arm reproduced the medium force with `N_t=1`.
- The flat arm removed both observables.
- The combined arm retained both observables without forcing `N_t=A_s`.
- The objective source showed amplitude-invariant constructed weak-probe clock-field response at matched position.
- The static-lapse KG scout showed a measured near/far temporal-frequency differential, but the measured packet clock rate did not match the constructed weighted lapse magnitude.
- The relational source weakened toward no temporal field in the raw weak-probe constructed-field limit; the evolved KG relational rates were not monotonic, so relational temporal KG dynamics remain inconclusive.

The restricted `COMMON_FIELD` rows are recorded only as a separate hypothesis check. They do not promote a common metric field or production-ready gravity interpretation.

## Re-Entry Requirements

Production gravity remains closed until a separate branch satisfies all of the following:

1. Source semantics are specified independently of the desired gravity result.
2. The temporal component, spatial component, or full metric choice is explicit.
3. The branch distinguishes supplied-coefficient response from emergent source-to-coefficient generation.
4. Flat, source-free, reversed-gradient, and matched-null controls pass.
5. Any claimed clock effect is tested as a temporal coupling, not inferred from a spatial kinetic coefficient.
6. Any claimed force effect is compared against the exact operator-force identity for the implemented operator.
7. Universal-gravity claims require probe-structure independence; the characterized spatial branch fails that criterion.
8. Newtonian claims require exterior field and shell benchmarks; the characterized spatial branch fails those criteria.

## Next Valid Branches

The two clean follow-up branches are:

- Temporal-metric branch: implement a genuine lapse or full metric and test clock-rate/trajectory consequences separately from the spatial coefficient mechanism.
- Nonlocal source-field branch: derive a bounded coefficient field from a nonlocal environment/source rule and test whether it produces long-range exterior structure without tuning to the target force.

Both are separate hypotheses. Neither inherits a gravity verdict from the characterized spatial effective-medium result.

## Documentation Links

- `docs/GRAVITY_D_EFFECTIVE_MEDIUM_CODEX_REPLICATION.md`
- `docs/GRAVITY_D_EFFECTIVE_MEDIUM_ROBUSTNESS_FIRST_PASS.md`
- `docs/GRAVITY_D_EFFECTIVE_MEDIUM_ROBUSTNESS_CLOSURE.md`
- `docs/GRAVITY_D_EFFECTIVE_MEDIUM_DYNAMICS_CHARACTERIZATION.md`
- `docs/external_validation/GRAVITY_D_EFFECTIVE_MEDIUM_EXTERNAL_VALIDATION_PACKAGE.md`

## Guardrail

Use the following current project statement:

```text
Quantule Mapper's Gravity D standalone mirror identifies a reproducible finite-width wave force generated by gradients of a bounded spatial kinetic coefficient. The exact momentum response is the coefficient gradient weighted over the probe's gradient-energy density. A leading coarse-grained approximation explains most measured force variation, with residual errors concentrated in near-null and multi-gradient regimes. The mechanism is short-ranged, probe-structure-dependent and non-Newtonian. It does not establish temporal gravity, geodesic motion, universal free fall or an IRER gravity source.
```

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `25020bf` (2026-07-22) — *dual substrate*
**Revised since:** 2 commit(s), most recently `3eb93af` (2026-08-26)

**Harness code changed since it was written:** 3 commit(s) to `jax_scout/`.
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
