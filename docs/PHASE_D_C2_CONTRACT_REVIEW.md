# Phase D / C2 — Conservative-Branch Contract Review (answer to the Codex escalation)

**Date:** 2026-07-08. **Question under review** (from `CONSERVATIVE_C2_HANDOVER_SUMMARY.md` / the Codex
weighted-invariant audit, `quantule_viz/outputs/conservative_geometry_campaign/`):

> Should Conservative C2 `kinetic_mode='conservative'` conserve total sum(|psi|^2) for the full nonlinear
> geometry-corrected RHS, or only for the linear dispersive substrate / special symmetric states?

**Answer: neither ordinary norm nor any fixed Ω-weighted norm is conserved by the C2 branch as implemented, and
this is the mathematically expected behaviour of the operator it contains — not a solver bug. The correct label is
`linear-conservative / nonlinear quasi-conservative (geometry-exchange)`. If true conservation is wanted, the fix is
a different operator (canonical/divergence form, "C2′"), not a different norm.** The Codex audit data confirm this
diagnosis in detail (below), including a smoking-gun near-pass that the audit itself surfaced.

## 1. What the geometry correction actually is

The implemented correction (mirror `jax_scout/physics.py::_cov_laplacian`, same algebra in the CuPy path) is

```
lap_cov = ( lap_flat + (D_spatial − 2) · (∇Ω·∇ψ)/Ω ) / Ω²          (D_spatial = 3)
```

which at d=3 is precisely the **Laplace–Beltrami operator** of the conformal metric `g_ij = Ω²δ_ij`:
`Δ_LB ψ = Ω⁻²(∇²ψ + (d−2)(∇lnΩ)·∇ψ)`. The C2 branch therefore evolves
`i ψ_t = −D·Δ_LB(Ω(ρ)) ψ − g(ρ)ψ` — a Schrödinger flow with a *live, density-sourced* metric.

Three standard facts about this operator explain every audit result at once:

1. **Δ_LB is self-adjoint with respect to the √g = Ω³ measure, not the flat one.** So the ordinary-inner-product
   adjoint test *must* fail (measured mismatch 7.8e-2) while the flat Laplacian passes (4e-16) — both observed.
2. **The audit's own smoking gun:** under the `sqrt_g_weighted_norm` (Ω³) inner product the frozen conformal
   Laplacian's mismatch drops to **2.15e-4 — a ~360× improvement and the only near-pass in the whole table**
   (`weighted_operator_adjoint_results.csv` line: `sqrt_g_weighted_norm, frozen_conformal_laplacian_proxy,
   2.154e-4`). That is the fingerprint of the Δ_LB identification. The residual 2.15e-4 is a *discretization*
   defect: the code evaluates Δ_LB in **non-divergence form** ((lap + correction)/Ω²), which breaks exact discrete
   self-adjointness even under the correct weight. A divergence-form spectral discretization
   `Ω⁻³·P†(Ω·Pψ)` (P = spectral gradient) is exactly discretely self-adjoint under the Ω³ weight.
3. **No fixed weighted norm can be conserved anyway, because Ω = Ω(ρ(t)) is live.** Frozen-Ω self-adjointness
   would conserve ∫|ψ|²Ω³ only for a *frozen* metric; with the measure itself evolving, no fixed Ω-power norm is
   an invariant. Hence `NO_WEIGHTED_INVARIANT_FOUND` is the *expected* outcome, not a dead end — the audit closed
   the "wrong norm" escape route correctly.

**The deeper structural point:** a canonical Hamiltonian flow `i ψ_t = δH/δψ*` with *any* real functional H
conserves ordinary ∫|ψ|² exactly (U(1)/Noether), no matter how complicated the geometry. The implemented term is a
Δ_LB transplant, **not** the functional derivative of the natural geometric energy: for
`H_kin = D∫ Ω(ρ)|∇ψ|² d³x` (= D∫ g^{ij}∂_iψ*∂_jψ √g for conformal g at d=3), the canonical flow is

```
i ψ_t = −D ∇·(Ω(ρ)∇ψ)  +  D·Ω′(ρ)·|∇ψ|²·ψ  +  V′(ρ)ψ
```

— divergence form **plus a metric-variation term** `D·Ω′(ρ)|∇ψ|²ψ` that the implementation omits. Because the
implemented RHS is not of canonical form, mass conservation is genuinely broken algebraically — matching the
trajectory-flux measurements (~−3.9e-4 fractional at t=0.75, zero at t=0 for symmetric states, dominated by
`geometry_covariant_correction`).

## 2. Contract verdict (the decision the audit asked for)

- **As implemented, C2 is intentionally-quasi-conservative in the nonlinear geometry sector.** Relabel per the
  audit's branch 2: **"linear-conservative / nonlinear quasi-conservative geometry-exchange."** Ordinary-norm drift
  in geometry-ON conservative runs is *physics of the implemented operator*, to be telemetered, not treated as
  solver error. All existing C2.x reports already describe the branch as quasi-conservative; the label is now
  grounded in an identified mechanism rather than an observation.
- **The geometry-OFF limit (`param_a_coupling = 0`) is algebraically clean.** `lap_cov − lap_flat ≡ 0` exactly, and
  the local density terms are pointwise phase-like (the term-flux decomposition measured them at numerical noise).
  So the pure-NLS branch conserves ∫|ψ|² algebraically; its only losses are stepper error + boost protocol.
- **If true conservation is wanted, build "C2′ canonical geometry" as a design-first RFC** (not a patch): the
  canonical flow above, discretized in divergence form (`P†ΩP` exactly self-adjoint) with the real, local
  metric-variation term (norm-neutral pointwise). C2′ would conserve ordinary norm **exactly algebraically** and
  discretely up to stepper error only — and it is the *physically motivated* variant, being the actual
  Euler–Lagrange flow of the geometric energy. Scope: RFC + mirror prototype + parity/adjoint/flux gates before any
  campaign; default-off like every C2 flag.

## 3. Reconciliation with the jax-mirror C2.2/C2.3 results (no contradiction; complementary)

- C2.2 found geometry-ON ≈ geometry-OFF kick-loss and "loss dominantly numerical, converging away as dt→0." That is
  consistent: at the tested dt (1e-3…2.5e-4) the **ETDRK4 stepper defect dominates** — the Codex contract audit
  measured it directly (one-step defect ∝ dt², −18.7% at T=4/dt=1e-3, halving per dt-halving; same scaling our
  dt-ladder saw). The *algebraic* geometry flux (~1e-4 fractional per unit time) is one to two orders below that at
  those dt, invisible under the stepper error.
- The Codex RK4 diagnostic (one-step defect ~1.5e5× smaller) stripped the stepper error away and **exposed** the
  small algebraic flux underneath, then localized it. So: jax mirror = transport phenomenology + dt-scaling;
  CuPy/Codex = operator-algebra localization. Together: **C2.2's extrapolated ~0.03 residual decomposes into
  (mostly) remaining stepper/boost-protocol error + (small, geometry-ON only) the identified algebraic flux.**
- The C2.3 chain currently running is **geometry-OFF pure NLS — unaffected by the blocker** (the non-adjoint
  operator never acts). It may continue; its momentum/peak diagnostics address an independent question (the
  velocity anomaly / winding picture).

## 4. Policy (agrees with and sharpens the Codex recommendation)

1. **Geometry-ON long conservative campaigns stay bounded** (short horizons, with ordinary-norm + flux +
   adjoint telemetry) until the relabel is accepted into the docs/gates or C2′ exists. N64/T-long remains blocked.
2. **Geometry-OFF conservative runs are unblocked** — the blocker term is identically absent.
3. **ETDRK4 is deprecated for invariant-sensitive conservative diagnostics** (use RK4-class steppers with flux
   telemetry there); ETDRK4 remains fine for transport phenomenology *with dt-ladders*, which is how C2.2/C2.3 use it.
4. **No solver/production changes from this review.** C2′ is RFC-first, mirror-first, default-off if pursued.

## 5. Cross-references

- Codex chain: `tools/conservative_stepper_contract_audit.py` (`ETDRK4_ERROR_CONFIRMED`) →
  `tools/conservative_rk4_stepper_diagnostic.py` (`RK4_GEOMETRY_REPLAY_PROMISING`) →
  `tools/run_rk4_integrity_batch.py` (`RK4_RHS_FLUX_FAIL`) → `tools/run_rhs_flux_source_isolation.py`
  (`RHS_NONLINEAR_ALGEBRAIC_FLUX_CONFIRMED`) → `tools/run_rhs_term_flux_decomposition.py`
  (`DISCRETE_OPERATOR_ADJOINT_FAILURE`) → `tools/run_weighted_invariant_audit.py` (`NO_WEIGHTED_INVARIANT_FOUND`).
- jax mirror: `docs/PHASE_D_C2_RESULTS.md`, `docs/PHASE_D_C2_NATIVE_SOLITON_RESULTS.md`,
  `docs/PHASE_D_C2_2_LOSS_SOURCE_RESULTS.md`, C2.3 (in flight).
- Guardrails: no IRER-proof claims; no conservative-stable-node claims; norm drift never hidden by normalization.
