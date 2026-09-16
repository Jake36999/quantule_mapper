# TG-B1S Equation Audit Addendum

Author: Codex, recording Claude review findings  
Timestamp: 2026-07-15  
Scope: documentation addendum for the frozen TG-B1S state-load feedback model. This does not change equations, labels, scripts or production systems.

## Purpose

Claude reviewed the TG-B1S implementation after the D4 validation campaign was already running. This addendum records the review findings that should accompany any later promotion decision.

The current formal status remains:

```text
TG_STATE_LOAD_LONG_TIME_DRIFT_UNRESOLVED
```

The promotion target remains blocked until D4 and D5 close:

```text
TG_STATE_LOAD_ORBITALLY_BOUNDED_FREQUENCY_SHIFT_SUPPORTED
```

No gravity, geodesic, objective-time, universal-free-fall, photon, production-readiness or IRER-validation claim is made.

## Implemented Model Summary

The frozen TG-B1S model evolves:

```text
state = (phi, pi, T, V_T, G, V_G)
```

with the chain:

```text
S_state -> T -> G -> A(G) -> delta phi
```

Implemented equations:

```text
phi_t = pi

pi_t = c^2 div(A grad phi)
       - m^2 phi
       + (a rho + s rho^2 + f rho^3) phi

A = exp(-epsilon_G G)

T_t = V_T

V_T,t = c_T^2 laplacian(T)
        - omega_T^2 T
        - gamma_T V_T
        + alpha_T S_state
        - kappa_TG G
        - absorber V_T

G_t = V_G

V_G,t = c_G^2 laplacian(G)
        - omega_G^2 G
        - gamma_G V_G
        - kappa_TG T
        - absorber V_G
```

The source is:

```text
S_state = [0.5 max(E_dens, 0) / e_ref
           + 0.5 |q_dens| / q_ref] / S0
```

with frozen global normalization:

```text
S0 = 135.6862187684289
```

Disabled in this branch:

```text
R_relax
L_lock
P_threshold
```

## Review Finding 1: Non-Variational Ledger Semantics

Claude's review classifies the implemented TG-B1S model as a phenomenological, non-variational feedback model.

The T/G subsystem has a stable symmetric coupling structure, but the full phi/T/G chain is not derived from one action with all reciprocal terms included.

Missing reciprocal channels in the current frozen model include:

- a reciprocal phi force from the `S_state -> T` channel, such as a term proportional to `T delta S_state / delta phi*`;
- a direct source of `G` by the phi gradient-energy channel, such as a term proportional to `A |grad phi|^2`.

Therefore the current energy ledger is an accounting diagnostic, not an exact Noether energy contract for the whole coupled system.

Current interpretation:

```text
ledger closure passing at weak coupling supports numerical boundedness,
but does not prove an exact conserved total energy for the model.
```

This is acceptable for the present bounded phenomenological audit, but any future claim of a mature feedback law should either:

1. derive a variational phi/T/G model; or
2. explicitly define a dissipative/accounting ledger that includes all non-reciprocal work terms.

## Review Finding 2: Sign Chain Places The Node On An A-Hill

The implemented sign chain is:

```text
S_state >= 0
T > 0 near node
G ~= -kappa_TG T / omega_G^2 < 0
A = exp(-epsilon_G G) > 1 near node
```

So the frozen state-load loop stiffens the spatial kinetic coefficient around the node.

This is not an error for the current claim, which is a self/backreaction and modal-frequency-shift audit. It is, however, a first-class design gate before any two-node or attraction experiment.

Reason:

```text
The Gravity D force characterization found wave packets respond to gradients
of the spatial coefficient through a gradient-energy-weighted force.
```

An `A`-hill around one node would not automatically create attraction for another node. Before any two-node TG experiment, the project must explicitly decide and document the intended sign convention for:

- `T -> G`;
- `G -> A`;
- whether node load should produce an `A`-hill or an `A`-well;
- whether attraction, repulsion or self-frequency shift is the intended observable.

Current standing gate:

```text
TG_TWO_NODE_SIGN_CHAIN_DECISION_REQUIRED
```

This is a design-decision gate, not a negative result for TG-B1S.

## Review Finding 3: Frequency Shift Needs An Analytic Contract

The D3/D4 evidence indicates a numerically stable negative modal-frequency shift:

```text
delta_omega ~= -2.15e-6
```

Claude notes that a naive fixed-profile perturbative estimate from the direct `A` stiffening channel gives the correct order of magnitude but the opposite sign. That suggests the measured shift may be dominated by profile/charge relaxation rather than the direct stiffening term.

Current interpretation:

```text
The frequency shift is numerically converged in completed rows,
but not yet analytically explained.
```

Recommended next audit:

```text
TG_B1S_FREQUENCY_CONTRACT_CHECK
```

Suggested contract:

1. Freeze `A(x)` or `G(x)` from representative TG-B1S runs.
2. Solve the modified stationary Q-ball/eigenfrequency problem using the A-weighted divergence operator.
3. Match the conserved charge or another preregistered invariant.
4. Predict the modal frequency shift.
5. Compare the predicted sign and magnitude with the measured `delta_omega ~= -2.15e-6`.

Promotion impact:

- D4/D5 can establish numerical orbital boundedness.
- The frequency-contract check would explain the mechanism of the shift.
- A promotion to `TG_STATE_LOAD_ORBITALLY_BOUNDED_FREQUENCY_SHIFT_SUPPORTED` is stronger if this contract closes, but D4/D5 remain the formal gates defined for the current campaign.

### FC-1 Correction: Sign Puzzle Resolved

Author: Codex, recording Claude FC-1 result  
Timestamp: 2026-07-15  
Reference:

```text
docs/gravity_maturity/TG_B1S_FREQUENCY_CONTRACT_RESULTS.md
sweep_runs/TG_B1S_FREQ_CONTRACT_20260715_181825
```

Claude subsequently ran the recommended frequency-contract check. The preregistered formal verdict was:

```text
TG_B1S_FREQUENCY_CONTRACT_SIGN_MATCH_ONLY
```

The important correction is that the apparent sign puzzle was a reporting-convention artifact. The D3/D4 observable `delta_omega_infty` is the slope of a modal phase difference, and the modal phase convention is approximately:

```text
theta ~= -omega t
```

Therefore:

```text
delta_omega_physical = -delta_omega_infty
```

The measured D3 value:

```text
delta_omega_infty ~= -2.150936882840006e-06
```

corresponds to a positive physical frequency shift:

```text
omega_full - omega_off ~= +2.150936882840006e-06
```

This matches the direction expected from geometric stiffening of the node by an `A`-hill. The earlier statement that the measured sign opposed the naive stiffening estimate is superseded.

The magnitude is bracketed by two analytic limits:

| analytic leg | predicted physical delta omega | ratio to measured |
| --- | ---: | ---: |
| fixed profile / no relaxation | `+5.398e-07` | `0.25` |
| fixed-Q envelope / relaxed adiabatic limit | `+4.690e-06` | `2.18` |
| measured physical shift | `+2.150936882840006e-06` | `1.0` |

Current bounded interpretation:

```text
The measured modal-frequency shift is consistent in sign and scale with
quasi-static geometric stiffening plus partial profile relaxation.
```

This improves the mechanism story but does not replace D4/D5 as the formal promotion gates.

## Minor Numerical Note: RK4 And Dealiasing

TG-B1S uses an RK4 path for the KG evolution. That is acceptable because the stationary baseline was re-gated and the D4 dt/2 row matched baseline closely.

However, this path does not currently use the same dealiasing mask associated with earlier frozen C3 spectral evolution. The smooth Q-ball, modest amplitude and grid-refined D4 row mitigate this in practice, but the deviation should be documented.

Current note:

```text
No current evidence suggests aliasing drives the TG-B1S frequency shift,
but future high-amplitude or sharper-node TG runs should include an explicit
dealiasing audit or a restored dealiasing contract.
```

## Review Endorsements

The review found no implementation error that invalidates D3/D4.

Confirmed strengths:

- `S_state` is the only active source in the frozen branch.
- `S0` is hardcoded as a global normalization, not recalibrated per case.
- Validated C3 machinery is reused for Q-ball construction and invariants.
- Later TG-B1S stages import one shared dynamics module, avoiding equation drift.
- Positive coefficient map `A = exp(-epsilon_G G)` remains nonsingular.
- T/G mass matrix is stable for the tested parameters.
- Label discipline remains conservative.

## Required Before Any Broader Use

Before a two-node TG experiment:

- close or explicitly bracket the sign-chain decision;
- decide whether `A` should represent an attractive well, a repulsive hill or only a self-impedance field;
- define the observable before running.

Before a mature feedback-law claim:

- either derive a variational phi/T/G model or explicitly book the non-variational work terms;
- perform the frequency-contract check;
- finish D4/D5 closure.

Before any external narrative:

- report the model as a bounded, phenomenological state-load temporal-geometric feedback audit;
- preserve the distinction between numerical frequency shift, analytic explanation and gravity interpretation.

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `25020bf` (2026-07-22) — *dual substrate*
**Revised since:** 12 commit(s), most recently `59b7ec3` (2026-09-12)

**Harness code changed since it was written:** 5 commit(s) to `jax_scout/`.
  - `caf61af` 2026-09-10 — Build HUD items 1+2: snapshot writer and universal field renderer
  - `06edae5` 2026-08-27 — Fix run provenance: 30% -> 86% of runs now dated
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Also referenced by (same date or earlier):** [[RUN_QUEUE]], [[gravity_maturity/TG_B1S_PROPOSED_CHECKS_CATALOG]], [[gravity_maturity/TG_TWO_NODE_SIGN_CHAIN_DESIGN_CONTRACT]]

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
