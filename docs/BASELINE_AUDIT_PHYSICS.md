# Baseline Audit — Physics & Scope (Stage 1.1 + 1.5)

**Descriptive only.** This documents *what the current stack implements and demonstrates* — not what it should
become. No redesign. Every claim cites an artifact (file:line, run id, or doc). Concept status tags:
`CONFIRMED` · `INFERRED` · `PLACEHOLDER` · `FALSIFIED/NULL` · `ABSENT/OUT-OF-SCOPE`.

Reference substrate = the **jax_scout FP64 mirror** (`jax_scout/physics.py`), which every Phase C validation ran
on (frozen geometry `e8d6a78ea`; claimed CuPy-equivalent — see `PHASE_C_METHOD_PARITY_AUDIT.md`). Params:
`SWEEP_PARAM_ORDER = [param_D, param_eta, param_rho_vac, param_omega0, param_a_coupling, param_s, param_f,
param_a]` (`physics.py:374`).

## 1. The implemented operator (as evolved)

Complex scalar field ψ on a periodic 3-torus (N³, L=10, dt=0.005), **first-order in time**, integrated by
ETDRK4 (Kassam–Trefethen 64-point contour, `physics.py:_construct_ops`):

```
∂_t ψ = L·ψ + N(ψ)
L_k    = −D·k² − η + i·ω₀                                   (physics.py:289)   [linear, k-space]
N(ψ)   = D·(Δ_g − Δ_flat)·ψ  +  a·ψρ + s·ψρ² + f·ψρ³        (physics.py:174)   [ρ = |ψ|²]
```

| term | code | role | status |
|---|---|---|---|
| `−D·k²` (D real) | `physics.py:289` | **real diffusion** (not `i·D·k²` dispersion) → dissipative kinetics | CONFIRMED |
| `−η` | `physics.py:289` | uniform linear loss | CONFIRMED |
| `+i·ω₀` | `physics.py:289`; feb `ω₀=0` | uniform phase rotation (k-independent → no transport); **off** | PLACEHOLDER |
| `a·ψρ` | `physics.py:176` | **cubic gain** (feb a=0.48>0); the critical stability knob (a\*≈×1.15) | CONFIRMED |
| `s·ψρ²`, `f·ψρ³` | `physics.py:176` | quintic / septic (feb s=0.013, f=−0.49) higher-order saturation | CONFIRMED |
| `D·(Δ_g − Δ_flat)` | `physics.py:177,217` | **conformal-geometry correction** (isolates ρ-curvature from flat Laplacian in L) | CONFIRMED |
| `Ω²(ρ)=(ρ_vac/ρ)^α`, α=`param_a_coupling` | `IRER_MATH_REFERENCE.md:251`; `gravity/unified_omega.py` | density-sourced conformal factor; Ω²<1 where ρ>ρ_vac | CONFIRMED |

**Net kinetic character:** with ω₀=0 the linear propagator `L_k = −D·k² − η` is **purely real** → every mode
damps (`E=exp(L_k·dt)`); the substrate is a **real (dissipative) Ginzburg–Landau reaction–diffusion** with a
complex field and a density-sourced conformal geometry. ("Complex" in "S-NCGL" = the complex *field* ψ, not
complex kinetic coefficients.) See `PHASE_C_KICK_INERTIA_AND_OPERATOR_FINDING.md`.

**Non-locality — PLACEHOLDER (key finding).** The "Non-Local" of "Sourced Non-Local Complex GL" is **not
implemented as non-local** in this mirror. `build_operators` aliases the production splash parameters onto the
**local** higher-order coefficients: `param_s ← param_splash_coupling`, `param_f ← param_splash_fraction`
(`physics.py:358-359`). There is **no convolution / FFT-nonlocal term** in `jax_scout/physics.py`. So the
validated substrate is a **purely local** cubic-quintic-septic model. (Whether CuPy production carries a true
non-local term is a separate method-parity/architecture-audit question, not resolvable from this mirror.)

**Optional couplings — ABSENT from the Phase C baseline (all default-off).** `n_op`/`step` accept `rho_vac_eff`
(A-modulated vacuum), `a_vec` (current-coupled A-field, gamma_A), `q_tensor` (anisotropic metric, Stage B), and
`drag_field` (adiabatic well) — **all `None` in every Phase C run** ("All None = scalar local geometry, gamma_A=0,
exact baseline", `physics.py:192`). So the baseline = **scalar, local, isotropic** conformal geometry only.

## 2. CONFIRMED concepts (implemented and empirically validated)
| IRER concept | realization | evidence |
|---|---|---|
| density → conformal-geometry feedback | Ω²(ρ), covariant Laplacian participates in evolution | `physics.py:217`; whole basin arc |
| standing bound attractors | long-time-stationary localized states | `FEB_ASTAR_CONFIRM` (a×1.15 flat to T=144k) |
| gain/loss stability basin | param-controlled; a\* ≈ ×1.15 knife-edge (±~0.5%) | `FEB_GAIN_LADDER`, `FEB_JOINT_BASIN` |
| local accretion / nucleation response | gain well grows local mass / new node, not relocation | `FEB_ADIABATIC_DRAG_V0LADDER*` |
| complex field carrying phase; first-order relaxational dynamics | ψ∈ℂ, ETDRK4 | `physics.py` |

## 3. INFERRED (implemented, but the IRER-ontology mapping is interpretive, not proven)
- "Informational Indifference → gradient forces toward equilibrium" — the conformal factor does drive to
  equilibrium (`IRER_MATH_REFERENCE.md:59`), but that ontological label on the Ω² mechanism is interpretation.
- ψ as "Ontological Informational Wave amplitude", ρ as "Resonance Density", Ω² as "Informational Manifold" —
  naming/interpretation over the concrete math (`IRER_MATH_REFERENCE.md:205-207`).

## 4. FALSIFIED / NULL (tested; no support as an active mechanism in this substrate)
| concept | result | evidence |
|---|---|---|
| prime-harmonic resonance as the stability mechanism | 0/60 prime peaks; stability is gain/loss balance | `FEB_BASIN_POSTHOC_VALIDATION` |
| TDA/Betti topology as stability discriminator | ~0 persistent topology, flat across TRUE/FAIL | `FEB_BASIN_POSTHOC_VALIDATION` |
| anisotropic-metric / FMIA tensor **routing** | `NO_SUPPORT` (Stage B) | `stage-b-tensor-geometry-no-support` |
| Payan / phase-winding as stability predictor | `NO_SIGNAL` (gap −0.08) | `payan-diagnostic-defined` |
| inertial (Galilean) mobility | μ≈0, structurally absent (real operator) | `FEB_KICK_INERTIA` + operator audit |
| static relational relocation | accretion/nucleation, never migration (3 morphologies) | `FEB_ADIABATIC_DRAG_*` |
| mobile matter-like transport | not supported in current substrate | `PHASE_C_MOBILITY_ENDPOINT.md` |

## 5. ABSENT / OUT-OF-SCOPE (not in the implemented substrate)
- **Conservative / dispersive transport sector** — no imaginary kinetic term (`i·D·k²`); D is real → absent.
- **Protected quantized spin / Payan circulation** — a dissipative gradient-flow can't protect topological
  charge; rotational "nodes" are transient (see `hifi-continuation-vortex-real`).
- **Non-local splash** (true convolution) — absent (aliased to local s/f; §1).
- **A-field current coupling / anisotropic metric** — present only as default-off options, absent from all
  validated results.
- **Full matter formation; BSSN / full spacetime dynamics** — out of scope; BSSN engine was stepped back to the
  conformal Ω² approach earlier in the project.

## 6. Scope boundary (Stage 1.5 — canonical)

> The current Quantule Mapper baseline is a **stability-sector / standing-attractor / geometric-feedback
> simulator**: a *local, real (dissipative) cubic-quintic-septic Ginzburg–Landau field on a density-sourced
> scalar conformal geometry*, integrated first-order in time. It legitimately demonstrates: collapse/relaxation,
> reproducible long-time standing bound attractors governed by gain/loss balance, a parameter-controlled basin,
> and local accretion/nucleation response, with a participating (stabilizing) density-sourced geometry.
>
> It is **not** a matter-sector / transport simulator: it has no dispersive/advective/conservative channel, no
> protected spin, and no non-local coupling in the validated substrate. Inertial and relational mobility nulls
> are **scope boundaries of this operator**, not failures. Any of those capabilities requires a deliberate,
> RFC-scoped formalism extension (Stage 3), not a patch.

**Frozen for downstream audits:** the operator (§1) and scope boundary (§6) are the reference for the numerical,
validation, and architecture audits. Open method-parity question passed to the architecture audit: does CuPy
production implement a true non-local splash and/or any imaginary kinetic term that the jax_scout mirror omits?
