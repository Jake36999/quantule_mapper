# Phase C — Mobility Endpoint

**Status markers:**
`PHASE_C_NUMERICAL_ATTRACTOR_RESULT_COMPLETE` ·
`STANDING_DISSIPATIVE_ATTRACTORS_SUPPORTED` ·
`MOBILE_MATTER_LIKE_TRANSPORT_NOT_SUPPORTED_IN_CURRENT_SUBSTRATE`

**Wording discipline:** site-pinning / relaxational resistance / local accretion — **not** inertia, **not**
matter-motion, **not** particle/topology/matter proof.

## Endpoint statement

> The current dissipative S-NCGL / conformal-geometry model supports robust standing bound attractors, but not
> mobile matter-like excitations under the tested probes. Across 4-node and 6-node stable morphologies, static
> gain wells produced no coherent relocation; responses were no coupling, local accretion, or new-node
> nucleation. Combined with the phase-kick null and the operator audit, this supports the conclusion that the
> current substrate has no inertial or advective transport channel. Matter-like transport would require a
> deliberate model extension, not another search within the current baseline.

## The three mobility probes

### 1. Galilean phase-kick (inertial mobility) — NOT SUPPORTED
`ψ → ψ·exp(i k x)`, k=2π n/L, n=0–3, on the confirmed a×1.15 state (`FEB_KICK_INERTIA_20260702_122013`).
Velocity kick-independent at the control noise floor (v_x ≈ +0.0002 for all n); displacement ~0.001 box;
mobility μ = v/k ≈ 0; nodes 4→4, mass/peak retained. `GALILEAN_KICK_NO_TRANSPORT_CONFIRMED`.

### 2. Operator audit — the inertial null is STRUCTURAL
Linear propagator `L_k = −D·k² − η + i·ω₀` (`physics.py:289`) with ω₀=0 at feb → purely real; every mode damps
(`exp(−Dk²dt)`). Kinetic RHS `D·∇²` with D real (diffusion, not Schrödinger `i·D·∇²`). No dispersive/advective
channel ⇒ no configuration can coast. `DISSIPATIVE_SUBSTRATE_NO_INERTIAL_CHANNEL`. See
`docs/PHASE_C_KICK_INERTIA_AND_OPERATOR_FINDING.md`.

### 3. Static gain-well (relational mobility) — NOT SUPPORTED, generalized across morphologies
Default-off `drag_field` variant (baseline bit-identical). Static well, V0 ladder 0.075→0.40 (16×), w=1.0,
offset=1.8, on three stable a×1.15 morphologies:

| state | morphology | verdict |
|---|---|---|
| seed619 | 4-node | `ACCRETION_ONLY_NO_RELOCATION` |
| seed620 | 6-node | `STATIC_WELL_NO_COUPLING` |
| seed621 | 6-node | `ACCRETION_THEN_NUCLEATION` (new 7th node at V0=0.4) |

Existing structure never relocates in any case (origin never depletes, node centroids never shift coherently);
response is always local (none → accretion → nucleation). Density-COM shifts are mass-weighted artifacts of
local growth, not migration. `STATIC_WELL_ACCRETION_ONLY_GENERALIZED_ACROSS_MORPHOLOGIES`. See
`docs/PHASE_C_ADIABATIC_DRAG_*_RESULTS.md`. **Moving-well test not run** (static relocation gate never passed).

## Config vs physics

- **Inertial null:** physics / universal (operator-proven, config-independent).
- **Relocation null:** now spans different stable morphologies (4-node + two 6-node) over a 16× well range ⇒
  best read as a **substrate/physics property of the current dissipative model**, not a single-config accident.
  (Not a claim that *all conceivable* configurations are immobile — a stronger/advective drive or a different
  formalism was not tested; those are Phase D.)

## What this closes for the current solver

| # | claim | status |
|---|---|---|
| 1 | Stable attractor existence | SUPPORTED |
| 2 | Long-time stationary/breathing core (a\*) | SUPPORTED |
| 3 | Parameter-controlled gain/loss basin | SUPPORTED |
| 4 | Inertial phase-kick mobility | NOT SUPPORTED |
| 5 | Static relational mobility (weak/moderate gain wells) | NOT SUPPORTED |
| 6 | Local accretion / nucleation response to gain preference | SUPPORTED |

## Optional Phase D — a mobility-capable formalism (future, NOT patched into Phase C)

Transport would require a deliberate extension giving the substrate a conservative/advective channel:
- imaginary / dispersive kinetic term (complex-Ginzburg-Landau `(1+ib)∇²`, or Schrödinger `i·D·∇²`);
- explicit advective / current-coupled transport (the A-field, reintroduced in a stable basin);
- moving source / pump field;
- second-order (wave-like) time dynamics.

Each is a formalism decision to be scoped as its own phase, with the same default-off / contract-stamped /
falsifier-first discipline used here.
