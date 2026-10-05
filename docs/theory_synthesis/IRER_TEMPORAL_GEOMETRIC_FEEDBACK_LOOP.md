# IRER Temporal-Geometric Feedback Loop

Codex formalization addendum.
Timestamp: 2026-07-14.
Conceptual authorship: Jake McIntosh.
Codex role: documentation synthesis, notation, computational boundary-setting, and proposed test design.

This document is an additive bridge note. It does not overwrite the original provenance document, Claude's theory
synthesis, the Gravity D reports, the bridge audit, or any production verdict. It collects the already-present
IRER concepts that point toward a closed temporal-geometric feedback loop and separates them from what has actually
been implemented in Quantule Mapper.

## One-Sentence Status

The theory contains the conceptual ingredients for a loop in which OIW phase-locking creates resolution activity,
resolution activity creates temporal load, temporal load drives geometric strain, and geometric strain feeds back on
future OIW propagation; the current computational work has tested several limbs of that loop, but not the full
bidirectionally coupled system.

## Source Trace

Primary original source:

- `docs/_Declaration of Intellectual Provenance v9.txt` is read-only for this pass. It is the authorship and
  original-concept source, not an implementation document.

Relevant original concepts found there:

| source location | concept | relevance to the loop |
| --- | --- | --- |
| lines 219-238 | AIS, OIW, Informational Resonance, RD | OIWs are phase-bearing substrate waves; RD measures coherent overlap and acts as a dynamic selector. |
| lines 249-254 | PAS | PAS is both produced by local field conditions and reshapes the later RFD landscape; this is the original feedback seed. |
| lines 256-306 | Quantules and angular deficits | a node/Quantule is a semi-stable incomplete closure whose mismatch generates compensating rotational response. |
| lines 314-329 | time and RFD | time is chronology of resolution; resolution events arise from PAS/RD thresholds and field gradients. |
| lines 390-427 | Observer-Resolution Loop and Chrono-Coherence | repeated resolution loops sculpt temporal domains, hysteresis, phase slips, and temporal torsion. |
| lines 469-507 | Payan states and gradient-derived forces | Payan transitions, misalignment cost, force propagation, and manifold deformation appear as core mechanisms. |
| lines 516-580 | FMIA and manifold channels | stable evolution follows low-resistance or minimal-action pathways, with path memory and hysteresis. |
| lines 778-836 | early splash model | collapse-event redistribution perturbs the local environment and changes cascade spectra; this is an ancestor mechanism, not a modern two-field solver. |

Relevant Claude/Codex synthesis and computational records:

| document | current statement |
| --- | --- |
| `docs/theory_synthesis/IRER_THEORY_AND_CONCEPT_TRANSLATION.md` | OIW, RD, PAS, RFD, FMIA, time-as-chronology, and conformal geometry are mapped into mathematical analogues; chrono-coherence and observer-loop layers are not fully implemented. |
| `docs/GRAVITY_AUDIT_C_CLOCK_RESULTS.md` | C.1/C.2/C.3 implemented temporal throttling, including stable self-relieving backreaction, but did not evolve an independent geometric field. |
| `docs/GRAVITY_D_EFFECTIVE_MEDIUM_DYNAMICS_CHARACTERIZATION.md` | Gravity D confirmed a spatial effective-medium wave force with exact law `d<P>/dt = -D int grad(A_s)|grad psi|^2 dV`; it is not a temporal-lapse result. |
| `docs/GRAVITY_TS_TEMPORAL_SPATIAL_BRIDGE_AUDIT.md` | the bridge audit decomposes `N_t` and `A_s`; it does not unify them as mutually generated fields. |
| `docs/gravity_maturity/G1_RESULTS.md` | the first independent KG clock calibration attempt failed as a local-clock instrument; objective temporal differential response remains detected, but clock-law calibration is open. |

## Working Hypothesis

Name:

```text
TEMPORAL_GEOMETRIC_RESOLUTION_FEEDBACK_LOOP
```

Plain statement:

```text
OIW overlap and phase locking raise RD/PAS. When a node begins to form, completed-resolution activity creates
local chronology load. The chronology load drives a geometric/manifold response. The geometric response changes
the allowed OIW propagation and decay channels. Stable nodes persist by entering a bounded relaxation cycle rather
than simply collapsing, exploding, or decaying chaotically.
```

Operational chain:

```text
Psi_OIW
  -> phase coherence / RD
  -> PAS / resolution activity R_res
  -> temporal response T or N_t
  -> geometric response G or A_s
  -> modified Psi evolution
  -> outgoing perturbation or radiation channel chi_out
  -> reduced local tension and continued node evolution
```

Important interpretation:

The phrase "the system chooses a stable decay route" should be implemented as attractor selection, not agency. A
simulation should contain competing dynamical channels and then measure which attractor basin the coupled fields
enter under conservation, coherence, boundary, and resolution constraints.

## Field Dictionary

The following symbols are proposed for future tests. They are not asserted as final IRER law.

| symbol | meaning | current implementation status |
| --- | --- | --- |
| `Psi(x,t)` | OIW/node substrate field; may be NLS-like or KG-like depending on test substrate | active in C2/C3 and Gravity D mirror work |
| `rho = |Psi|^2` | resonance-density proxy / field intensity | active |
| `R_res[Psi]` | resolution activity source: continuous RD/PAS load or thresholded resolution events | not implemented as a dedicated modern field |
| `T(x,t)` | temporal-resolution strain or chronology-load field | partially represented by clock factors/lapse in C-series and TS bridge |
| `N_t(T)` | positive chronology factor, for example `N_t = 1/(1 + T)` or `exp(-T)` | supplied/static or relational in mirror tests; not dynamically sourced by `G` |
| `G(x,t)` | geometric/manifold strain field | partially represented by `Omega^2(rho)` and Gravity D `A_s`; not dynamically sourced by `T` |
| `A_s(G)` | positive spatial kinetic/effective-medium coefficient | active as supplied/frozen coefficient in Gravity D |
| `chi_out` | outgoing propagating disturbance/radiation channel | observed only indirectly as radiation/halo in collisions; not a named feedback field |

## Candidate PDE Scaffold

This scaffold is a testable notation layer, not a claim that IRER has been fully reduced to these equations.
It is also not yet a ready-to-code model. TG-A must first choose the exchange class, energy budget, and resolution
source, because otherwise the signs and derivative couplings could build the desired pulse ordering into the
simulation.

### OIW / node evolution

For an NLS-like scout:

```text
i d_t Psi =
  -D div(A_s(G) grad Psi)
  + V_T(T) Psi
  + U'_Psi(|Psi|^2) Psi
  + C_feedback[T,G,Psi].
```

For a KG-like scout:

```text
d_t phi = N_t(T) Pi
d_t Pi  = div(A_s(G) grad phi) - N_t(T) (m^2 phi + U'_phi(phi)) + C_feedback[T,G,phi].
```

The KG form is preferred for the first serious radiative scout because it naturally supports second-order
propagation, field momentum, localized modes, and outgoing radiation diagnostics.

### Resolution source

Candidate continuous source:

```text
R_cont[Psi] = W_phase(Psi) * S_RD(rho, grad rho, rho_c)
```

where `W_phase` measures local phase-locking/coherence and `S_RD` is a smooth increasing function of RD above a
background scale.

Candidate threshold source:

```text
R_thr[Psi] = H_smooth(PAS[Psi] - PAS_c) * max(d_t PAS[Psi], 0).
```

These two sources must be run separately. A positive result from a threshold rule should not be promoted unless a
continuous-source or action-derived version reproduces the effect.

### Temporal response field

Second-order response:

```text
d_tt T + gamma_T d_t T - c_T^2 Lap T + omega_T^2 T + lambda_T T^3
  = alpha_T R_res[Psi] + alpha_TG d_t G.
```

Relaxation scout:

```text
tau_T d_t T = -T + alpha_T K_T * R_res[Psi].
```

Here `K_T *` denotes a local or bounded nonlocal smoothing kernel. The second-order form is better for pulse and
frequency tests; the relaxation form is cheaper and useful as a null/simple-limit comparison.

### Geometric response field

Second-order response:

```text
d_tt G + gamma_G d_t G - c_G^2 Lap G + omega_G^2 G + lambda_G G^3
  = alpha_G d_t T + alpha_GR R_res[Psi].
```

Relaxation scout:

```text
tau_G d_t G = -G + alpha_G K_G * d_t T + alpha_GR K_GR * R_res[Psi].
```

This is the minimal mathematical representation of the user's description that "one field's actions become a
reflection on the other." It must be validated by coupling-off controls before any interpretation is promoted.

### Exchange-class decision

TG-A must choose one of two explicit model classes before implementation.

Conservative exchange model:

```text
L = L_Psi + L_T + L_G + L_int
```

with `L_int` deriving all cross-coupling signs. Example potential coupling:

```text
L_int = -kappa_TG T G
```

Example derivative or gyroscopic coupling:

```text
L_int = (eta / 2) (T d_t G - G d_t T).
```

In this class, the relative signs are not free tuning parameters; they are fixed by the action or Hamiltonian.

Explicitly dissipative model:

```text
E_initial = E_Psi + E_T + E_G + E_outgoing + E_dissipated + epsilon_numerical.
```

This class may include damping, relaxation and irreversible export, but every loss term must be recorded in the
accounting ledger. Mixing conservative and dissipative terms without this ledger is disallowed for the first scout.

### Positive coefficients

Use positive maps so temporal and spatial coefficients cannot cross zero:

```text
N_t(T) = exp(-T)              or              N_t(T) = 1 / (1 + T), T > -1
A_s(G) = exp(-G)              or              A_s(G) = 1 / (1 + G), G > -1
```

Do not assume `N_t = A_s`. That is a separate common-field hypothesis.

## Resolution Source Problem

The unresolved central object is:

```text
R_res[Psi].
```

A density source such as `R_res = |Psi|^2` only says field presence creates temporal load. A threshold rule such as
`H(PAS - PAS_c)` risks inserting resolution events by hand. Neither automatically represents completion of a
resolution cycle.

The first scout should preregister at least two source families.

Continuous coherence-transition source:

```text
R_coh = [-d_t K_phase]_+
```

where `K_phase` measures local phase mismatch or phase-gradient cost. This asks whether temporal load is generated
when a region becomes more phase-locked.

Smooth threshold source:

```text
R_thr = sigmoid((P - P_c) / delta_P) [d_t P]_+
```

where `P` is a separately defined PAS proxy. This asks whether threshold-crossing resolution activity generates the
loop. A positive threshold-source result should not be promoted unless the continuous source, or an action-derived
equivalent, shows the same qualitative cycle.

## Relation To Verified Dynamics

### Already explored / supported pieces

- OIW/RD field proxies exist as complex fields and `rho = |psi|^2`.
- Conservative NLS and KG substrates support coherent transport and phase-dependent interaction/collision structure.
- The C-series showed computationally real temporal throttling, including stable self-relieving backreaction in the
  relational clock toy model.
- Gravity D confirmed a spatial medium force under a bounded spatial coefficient:

```text
d<P>/dt = -D int grad(A_s) |grad psi|^2 dV.
```

- The temporal-spatial bridge audit showed that `N_t` and `A_s` can be decomposed into separate arms and measured
  without conflating temporal and spatial effects.

### Not yet explored / not yet formalized

- A dynamic field equation for `T` sourced by modern resolution activity `R_res[Psi]`.
- A dynamic field equation for `G` sourced by `T`, by `d_t T`, or by resolution activity.
- A closed `Psi -> R_res -> T -> G -> Psi` loop.
- A named outgoing radiation channel produced without an explicit "emit now" rule.
- FMIA bridges/wires emerging as low-resistance decay routes under dynamic geometric stress.
- A calibrated perturbation frequency or phase-locking sequence across event, temporal, geometric, and outgoing channels.
- The hypothesis that photon-like radiation is a minimal geometric relaxation mode.
- The hypothesis that an emission pattern is an inverse or complementary image of the source/node geometry.

## Photon / Radiation Hypotheses

The original documentation contains support for light as an OIW-like propagating coherence phenomenon and for
Payan/force propagators, but it does not yet explicitly establish:

- photon as the minimal possible geometric relaxation mode;
- photon emission as necessary for stable node persistence;
- photon map as inverse image of an atom or node.

Proposed open labels:

```text
PHOTON_AS_MINIMAL_GEOMETRIC_RELAXATION_MODE
EMISSION_PATTERN_AS_SOURCE_GEOMETRY_PROJECTION
```

The second label is the testable rendering of "inverse image." It requires a precise observable. Candidate meanings:

- spatial anticorrelation between source mode and outgoing field;
- complementary nodal structure;
- Fourier-dual structure;
- Green-function response to a transition density;
- radiation pattern determined by a source-mode difference;
- geometric complement of a bound node mode.

These are different hypotheses and must not be collapsed into one claim.

## Why The Node Does Not Simply Become A Radioactive Core

The live theoretical conflict can be rendered without agency:

```text
If resolution activity accumulates without a relief channel, the node should tend toward runaway collapse,
fragmentation, or chaotic leakage. If temporal throttling and geometric response create an accessible relaxation
attractor, the node can persist by periodically or quasi-periodically exporting accumulated load through outgoing
perturbations.
```

Candidate attractor classes:

```text
STABLE_NONRADIATING_NODE
PERIODICALLY_RADIATING_NODE
QUASIPERIODICALLY_RADIATING_NODE
SINGLE_BURST_RELAXATION
CONTINUOUS_LEAKAGE
FRAGMENTATION
RUNAWAY_COLLAPSE
CHAOTIC_DECAY
NUMERICALLY_UNRESOLVED
```

The simulation should not contain a hard-coded rule saying "choose the stable route." The route is supported only
if the attractor emerges under fixed equations and survives controls.

## Perturbation Frequency Observables

Separate the frequency question into observable channels:

| observable | definition |
| --- | --- |
| event frequency | rate of threshold/resolution events in `R_res` |
| node frequency | internal phase or breathing frequency of the localized node |
| temporal frequency | dominant frequency of `T(t)` or `N_t(t)` near the node |
| geometric frequency | dominant frequency of `G(t)` or `A_s(t)` near and outside the node |
| emission repetition rate | inverse mean interval between outgoing flux pulses |
| outgoing carrier frequency | spectral peak inside each emitted perturbation packet |
| exchange frequency | spectral peak in inferred power transfers `P_Psi_to_T`, `P_T_to_G`, `P_G_to_Psi` |

A strong positive signature is not merely a spectral peak. It is a causal phase-ordered sequence:

```text
R_res peak -> T pulse -> G pulse -> outgoing flux pulse -> reduced local load
```

with stable phase lags under grid, timestep, box, and phase-scrambled controls.

Distinguish inserted frequencies from emergent frequencies:

```text
inserted: omega_T, omega_G
emergent: nonlinear limit-cycle frequency, beat frequency, node transition frequency,
          pulse repetition frequency, outgoing carrier frequency
```

A firing frequency is scientifically interesting only if it is not trivially equal to an inserted oscillator
frequency, changes predictably under parameter sweeps, converges numerically, appears across nearby initial
conditions, and follows an independently derived mode or transition relation.

## Proposed Test Method: TG-A/TG-B

### TG-A - formalization gate

Before simulation:

1. Choose the implementation substrate: KG first, NLS comparison second.
2. Choose either a conservative action/Hamiltonian model or an explicitly dissipative accounting model.
3. Pick one continuous `R_res` and one thresholded `R_res`; preregister both.
4. Specify whether `T` and `G` are second-order wave fields or relaxation fields.
5. Derive an energy/action budget or an explicit dissipative accounting ledger.
6. Define positive maps `N_t(T)` and `A_s(G)`.
7. Derive the outgoing KG/OIW flux from the chosen action.
8. Define null limits:
   - `alpha_T = 0`
   - `alpha_G = 0`
   - `C_feedback = 0`
   - source-off
   - flat `T=0`, flat `G=0`
9. Linearize about the flat state and verify the coupled `T/G` system is stable before adding a node.
10. Identify the natural normal-mode frequencies of `T` and `G`, so inserted frequencies are not mistaken for
    emergent firing frequencies.
11. Define frequency diagnostics before any run.

### TG-B0 - zero-dimensional exchange test

Before a spatial PDE, test a reduced node-amplitude model:

```text
d_t a = f(a,T,G)
d_tt T + gamma_T d_t T + omega_T^2 T = alpha_T R(a) + C_T(T,G)
d_tt G + gamma_G d_t G + omega_G^2 G = alpha_G R(a) + C_G(T,G)
```

Purpose:

- identify stable fixed points;
- identify Hopf bifurcations and limit cycles;
- reject runaway coupling signs cheaply;
- estimate damping and coupling scales before radiation tests.

### TG-B1 - reduced spatial scout

Do not start with a full 3D double-field solver. Start with a 1D radial or 2D axisymmetric scout:

```text
phi or Psi node, preferably KG first
T temporal-response field
G geometric-response field
outgoing-flux diagnostic derived from phi/Psi
```

Do not initially add a separate `chi_out` radiation field. The KG field already supports outgoing propagating
radiation. Measure it through the derived radial energy flux, schematically:

```text
J_r ~ -phi_t partial_r phi
```

with the exact expression derived from the chosen action. A fourth radiation field should be introduced only if the
theory requires a physically distinct channel and its coupling has an explicit energy-transfer law.

Initial experiments:

- two OIW/node structures with controlled phase difference `Delta phi`;
- carrier/separation sweep around the already-known anti-phase and capture/pass-through regimes;
- source-off and phase-scrambled controls;
- temporal-coupling-off control;
- geometric-coupling-off control;
- feedback-off control;
- timestep, grid, and box refinements.

Primary measurements:

- phase-locking time;
- node-formation or resolution-source onset time;
- `T` pulse time and amplitude;
- `G` pulse time and amplitude;
- outgoing flux time, amplitude, carrier frequency, and repetition rate;
- cross-correlation lags `R_res -> T`, `T -> G`, `G -> J_out`;
- wavelet/coherence spectra;
- total energy/action budget and boundary flux;
- whether a lower-tension decay path or bridge/channel forms.

Success criteria:

- source-off removes both temporal and geometric pulses;
- temporal-off removes `T` and downstream `G`;
- geometric-off retains `T` but removes `G` and any geometry-mediated feedback;
- feedback-off retains emitted response but removes its effect on node evolution;
- causal ordering survives refinement;
- frequencies do not lock to timestep, grid, box size, or diagnostic cadence;
- radiation emerges without a hand-written emission trigger;
- energy/action accounting closes, or all dissipation is explicitly quantified.

Negative outcomes to preserve:

- no stable phase order;
- no reproducible perturbation frequency;
- pulse timing tracks numerical cadence;
- emitted radiation appears only because a threshold rule directly injects it;
- full loop disappears when source and controls are matched;
- KG and NLS substrates disagree in a way that cannot be characterized.

### TG-B2 - second-substrate comparison

Only after a stable KG regime is found or rejected, repeat the same coarse-grained source and feedback logic with the
NLS substrate. The question is whether both substrates show the same attractor class and causal ordering, not whether
their waveforms are identical.

## Guardrails

- Do not call this gravity, geodesic motion, universal free fall, or a photon result.
- Do not infer photon structure until `chi_out` or outgoing flux has a precise observable definition.
- Do not use one shared field `N` for temporal and spatial effects unless testing the restricted common-field
  hypothesis.
- Do not tune emission thresholds after seeing the spectra.
- Do not promote a "stable decay route" unless it emerges from the dynamics without being hand-coded.
- Keep production geometry, Hunter, production verdicts, protected validation files, launch infrastructure, and
  CPU/GPU fallback policy untouched.

## Recommended Next Action

Create a design-only TG-A note and, after review, a small standalone scout:

```text
jax_scout/gravity_TG_B_temporal_geometric_feedback_scout.py
```

The first implementation should be intentionally small and falsifiable. The target result is not "photon found" or
"gravity found." The target result is:

```text
A reproducible causal sequence in which OIW/node resolution activity drives a temporal-response pulse, the temporal
pulse drives a geometric perturbation, and the geometric perturbation either emits or modulates an outgoing wave with
a measurable frequency and phase lag.
```
