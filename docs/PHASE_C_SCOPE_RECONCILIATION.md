# Phase C — Scope Reconciliation

## Why Phase C was a stability-sector test, not a full matter-sector test

The mobility null (phase-kick + static-well) is best read as a **scope boundary of the current stack**, not a
failure. The current stack was progressively narrowed from a full BSSN-style spacetime engine toward a
tractable dissipative S-NCGL / conformal-geometry feedback engine, and was scoped to test **node formation and
density-derived geometric feedback**, not matter transport. Within that intended scope, Phase C is a strong
positive result.

## The scope is documented, not just recalled

- **Designed research question** ([IRER_MATH_REFERENCE.md:321](IRER_MATH_REFERENCE.md)): *"Do the nonlinear,
  geometry-coupled OIW dynamics spontaneously organise into spectral structures?"* — a question about **stable
  spectral organization**, not motion/transport.
- **Designed role of the geometry** ([IRER_MATH_REFERENCE.md:59](IRER_MATH_REFERENCE.md)): the conformal factor
  deforms in response to density to *"create gradient forces that drive the field toward **equilibrium**
  configurations"* — i.e. **stabilization**, not transport.
- **The reference itself frames the limits as scope** ([IRER_MATH_REFERENCE.md:346](IRER_MATH_REFERENCE.md)):
  *"…not correctness bugs that invalidate results — they are scope limitations that constrain what the
  simulation can find. The mathematics being evolved is sound."*
- **Design-history pivot (author recollection, consistent with the code):** an earlier BSSN full-relativity
  engine was stepped back (compute bottleneck + not viable as the engine) to the conformal Ω²(ρ) approach —
  a deliberate narrowing from *full spacetime/matter dynamics* to *density-field + conformal-geometry feedback*.
- **Kinetic clarification:** the field ψ is complex (it carries phase), but the implemented kinetic operator is
  real-diffusive (`L_k = −D·k² − η + iω₀`, ω₀=0 — a real `D·∇²`, not `i·D·∇²`). So the substrate is dissipative
  in its kinetics despite the "Complex GL" name; the "complex" refers to the field, not the diffusion
  coefficient. Whether a complex/dispersive kinetic term was *intended* is a theory-audit question (below), not
  settled here.

## Reasonable scope of the current stack

**In scope — and validated by Phase C (SUPPORTED):**
- collapse / relaxation dynamics;
- bounded standing node / attractor formation;
- gain/loss basin mapping and its boundary (a\* ≈ ×1.15, ±~0.5%);
- density → conformal-geometry **self-referential coupling as a stabilization mechanism** (the geometry
  participates in the dynamics and drives toward equilibrium configurations);
- local accretion / nucleation response to a gain preference.

**Out of scope — structurally excluded by the real-diffusive operator (NOT SUPPORTED, and not expected):**
- inertial / Galilean motion; conserved momentum;
- advective transport; coherent relocation;
- protected quantized spin / circulation;
- force-mediated interaction; geodesic bending / lensing of structures;
- mobile matter-like excitations.

**On the "does emergent gravity affect the field" scope question:** partially answered, honestly split — the
density-sourced geometry **does** shape the field toward stable equilibria (in-scope, achieved), but it does
**not** yet act as a gravitational *force* that routes/bends structures (Stage-B anisotropic-metric routing =
`NO_SUPPORT`), because force requires a transport channel the dissipative substrate lacks. So "geometry as a
stabilizing feedback" ✓; "geometry as an analogue-gravity force" not demonstrated.

## Net statement

> The Quantule Mapper Phase C solver is a **stability-sector implementation** of IRER — a dissipative S-NCGL /
> conformal-geometry attractor + geometric-feedback engine. Within that scope the result is strong: reproducible,
> seed/resolution/morphology-checked long-time standing attractors governed by gain/loss balance, with a
> participating density-sourced geometry. The mobility tests then correctly identify the **transport/matter
> sector** as out of scope for the current baseline operator. We were asking the stability-sector engine to
> answer a matter-sector question; the null is the honest scope boundary.

`PHASE_C_STABILITY_SECTOR_VALIDATED_AS_NUMERICAL_MODEL` ·
`PHASE_C_MATTER_TRANSPORT_OUT_OF_SCOPE_FOR_CURRENT_BASELINE` ·
`PHASE_D_FORMALISM_REQUIRED_FOR_MOBILITY_SECTOR`

## On the "Technical Synthesis and Theoretical Refinement" document (2026-07-02)

That multi-tab synthesis **aligns qualitatively** with the independently-derived Phase C findings on the
dissipative/conservative split (real- vs complex-GL kinetics; geometry inert without transport; Payan/spin needs
a conservative core; `aste_hunter` mis-aimed at a prime signature the substrate can't generate; "faithful
dissipative half"). Those points are consistent with our evidence and are retained.

**However, it is a scope-reconciliation / forward-aspiration document, not primary empirical evidence, and the
following claims in it are NOT established by Phase C and are deliberately NOT adopted into this record** (several
contradict our actual findings):
- "best-run SSE ≈ 0.00087 / 0.996 correlation with SPDC data" — Phase C found the prime/spectral signature
  **null** (0/60) and snapshot/spatial spectra non-discriminating; this benchmark is unverified here;
- "PPN γ = 1 derivation" — asserted, not verified in this campaign;
- "z ≈ −0.162 dynamic exponent / w < −1 phantom dark-energy prediction" — no scaling runs; aspirational;
- "k ≈ ln 2 ≈ 0.693 as the risky prediction / milestone" — the prime hypothesis is a **null** in Phase C;
- "80% ready for collaboration", SPDC triangulation, semiconductor-fabrication specifics — forward aspiration,
  outside Phase C scope;
- the various Phase-D implementation mandates (Madelung/CGL, RK4/second-order, discrete energy-momentum, contract
  guardrails) — reasonable **RFC material for a future phase**, not Phase C results.

Use that document for the qualitative scope split only; keep the empirical record to what Phase C actually
measured.

## Next (optional, deliberate — not another simulation)

A **theory/provenance kinetic audit** would resolve whether the dissipative substrate is *faithful to the theory*
(→ the mobility null is a theory-level result) or a *stability-sector-only implementation* (→ Phase D should add
the conservative/transport sector). Search the Declaration appendices for whether IRER pins down: first- vs
second-order time dynamics; real-diffusive vs complex-dispersive kinetics; whether Payan/FMIA/forces imply
transport. Preliminary read (from the math reference + provenance intro): the operator choice appears **under-
specified** in the theory and the current stack was scoped to the stability/geometry sector — but this should be
confirmed by the appendix audit before Phase D commits to a kinetic form. Output → `PHASE_D_FORMALISM_GAP_REVIEW.md`.
