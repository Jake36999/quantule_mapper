# RFC: C3 — Second-Order (Wave) Kinetic Substrate: the Inertia Candidate

**Status: DESIGN ONLY.** Mirror-first prototype if accepted; separate module (does not touch `physics.py` Ops or any
Phase C path); bounded runs only. This is candidate C3 from `docs/PHASE_D_KINETIC_OPERATOR_RFC.md`, promoted to the
front of the queue by the C2.4 closure (order agreed 2026-07-09: C3 → coefficient scout → C4 → C2′).

## 1. Why C3, and why now
The conservative arc closed negative with a *mechanism*, not just a null: in first-order Schrödinger dynamics
(`iψ_t = …`), momentum lives in the **phase field** (P = ∫ρ∇φ), and continuity permits a static density to host
steady through-flow (∇·(ρ∇φ) = 0) — C2.4's fountain. A standing structure is never *obliged* to move unless it is a
true soliton (and for feb/a\* none exists, C2.3).

A **second-order-in-time** substrate removes that escape route structurally. With independent momentum field
π = ψ_t, the momentum density is the *rate of change of the configuration itself*: a structure holding momentum IS a
structure whose configuration is moving. There is no phase channel for momentum to stream through a static density.
C3 therefore tests the sharpest remaining hypothesis:

> Do IRER-family structures move when the substrate gives the field genuine configurational inertia?

## 2. The equation
Complex nonlinear Klein–Gordon (NKG) with the IRER local nonlinearity:

```
ψ_tt = c²∇²ψ − m²ψ + g(ρ)ψ ,      ρ = |ψ|²,  g(ρ) = a·ρ + s·ρ² + f·ρ³
```

- **c** — substrate wave speed (new parameter; sets the causal cone). Default c=1 in box units.
- **m** — mass gap (new parameter). m>0 is required for localized standing objects (sets the ω<m binding window).
- g(ρ) reuses the IRER cubic-quintic-septic family — coefficient choices coordinated with the C2.5 family scout
  (Lane B), since the same existence structure (saturation) governs both.
- Geometry coupling deliberately **excluded** from the prototype (flat metric); added later only if the flat
  substrate transports (same discipline as C2: one new mechanism at a time).

### Exact invariants (continuum)
- **Energy** E = ∫ |ψ_t|² + c²|∇ψ|² + m²ρ − G(ρ),  G′=g — time translation.
- **Charge** Q = Im ∫ ψ\*ψ_t — U(1). NOTE: the conserved U(1) quantity is *charge*, not mass ∫ρ. Reports must track
  both and never conflate Q-conservation with ∫ρ-conservation (∫ρ genuinely breathes in NKG).
- **Momentum** P = −Re ∫ ψ_t\*∇ψ — spatial translation. This is the transport-relevant one: it is built from the
  *time derivative of the configuration*, which is the structural difference from C2.

## 3. The native objects: Q-balls / oscillons
NKG with attractive saturating g is the classic **Q-ball** setting: stationary states ψ = φ(r)e^{−iωt} with
0 < ω < m, existence governed by Coleman's condition (min over φ of [m² − 2G(φ²)/φ²] < ω²). For the IRER-family g
with a>0 (attractive cubic) and saturation from s,f — plausible; the C2.5 scout's g_max analysis carries over
(binding window requires g to beat m² at feasible amplitude).
- **Search method:** the C2.3 Petviashvili machinery applies nearly verbatim — the stationary equation is
  `(m² − ω²)φ = c²∇²φ + g(φ²)φ`, i.e. the C2 stationary problem with μ → (m²−ω²)/1 and D → c². Scan ω instead of μ.
- **Transport test:** unlike C2, theory *guarantees* a true Q-ball moves — NKG is Lorentz covariant, so boosted
  solutions exist at all |v|<c. The empirical questions are (i) do stable Q-balls exist for IRER-family g, and
  (ii) does an **initial-velocity kick** (π₀ = −v·∇ψ₀ — a configurational kick, no phase trick) actually set the
  found object in motion with density+momentum co-moving. That co-motion is the falsifiable transport observable.

## 4. Discretization + stepper (prototype)
- State (ψ_k, π_k) pseudo-spectral. The **linear part is solved exactly per mode**: each k evolves as a harmonic
  oscillator with ω_k² = c²k² + m² (rotation matrix — no CFL from the linear part, no ETDRK4 machinery needed).
- Nonlinearity via Strang splitting: half-kick π += (dt/2)·g(ρ)ψ (real, local) → exact linear rotation dt → half-kick.
  Symplectic-ish, time-reversible; energy telemetry measures the splitting error. dt criterion from the nonlinear
  timescale only (dt ≪ 1/g_max), not from k_max.
- Dealias as in C2 (2/3 mask, applied symmetrically).
- New module `jax_scout/phase_d_c3_wave.py`; no shared Ops mutation; imports only grids/diagnostics.

## 5. Gates (all before any campaign)
1. Linear parity: g=0 evolution matches the analytic mode rotation to machine precision.
2. Invariant telemetry: E and Q drift = splitting-error-only (dt² scaling demonstrated on a dt-ladder).
3. Q-ball existence scan (Petviashvili-in-ω) — same classification discipline as C2.5 (TRUE_BRANCH / UNIFORM /
   NO_CONVERGENCE), N=48.
4. Stability: evolve found φ for T≳ several internal periods; reject breathers-that-disperse.
5. **Transport gate:** velocity kick π₀ = −v·∇ψ₀ with v = 0.1c, 0.25c; measure centroid AND core velocity AND P(t);
   success = density advects at ≈v with P coupled to density motion, surviving dt/N refinement. Failure = C2-style
   decoupling (would be a major surprise and itself decisive).
6. Null control: unkicked object stays put (breather-wander baseline, per the C2.3 peak-tracker lesson).

## 6. Success / failure criteria (verbatim targets)
- SUCCESS: `C3_INERTIAL_TRANSPORT_CONFIRMED` — localized object, density+momentum co-moving at the imparted
  velocity, dt/N-robust. This would be Phase D's first genuine matter-like motion and reframes the whole program.
- PARTIAL: objects exist but only oscillons (finite lifetime) — report lifetimes; transport measured within life.
- FAILURE: `C3_NO_STABLE_OBJECTS` for IRER-family g (existence window empty) — pushes to C4/non-local or coefficient
  co-design with Lane B.

## 7. Relation to IRER / scope guardrails
C3 is a *different dynamical postulate* for the informational substrate (wave-like propagation of resonance modes —
V13 lineage), not an extension of Phase C. No production/CuPy changes; no gate/classifier changes; no matter claims —
"transport" here means the measured co-motion observable in §5.5. Runs bounded and checkpointed; ~2–4h GPU for the
full gate battery at N=48–96 once the prototype exists (~a day of engineering).
