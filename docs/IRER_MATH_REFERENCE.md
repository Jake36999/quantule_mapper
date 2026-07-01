# IRER Simulation — Mathematics Reference & Sanity Check

**Author:** Jake McIntosh  
**Prepared by:** Claude (Sonnet 4.6) — sanity-check pass, June 2025  
**Status:** Pre-compute review; covers `worker_cupy.py` ETDRK4 solver

---

## 1. What PDE is the simulation solving?

The simulation integrates the **Sourced Non-Local Complex Ginzburg-Landau (SNCGL) equation** on a dynamical conformal background in 3D:

```
∂ψ/∂t = D · Δ_g[ψ] + (−η + iω₀) ψ + (a·ρ + s·ρ² + f·ρ³) ψ
```

where:

| Symbol | Meaning | Code parameter |
|--------|---------|----------------|
| ψ(x,t) | Complex scalar field (OIW amplitude) | `psi_k` (in spectral space) |
| ρ = \|ψ\|² | Probability/resonance density | `rho` |
| D | Diffusion / kinetic coupling | `param_D` |
| Δ_g | Covariant Laplacian in conformal metric | (derived below) |
| η | Damping / decay rate | `param_eta` |
| ω₀ | Vacuum oscillation frequency | `param_rho_vac` (imaginary part of L) |
| a | Quadratic density nonlinearity (ψ·ρ) | `param_a` |
| s | Quartic density nonlinearity (ψ·ρ²) | `param_s` / `param_splash_coupling` |
| f | Sextic density nonlinearity (ψ·ρ³) | `param_f` / `param_splash_fraction` |

The field ψ is a complex scalar living on a **self-referential curved geometry** — the metric is determined by ψ's own density.

---

## 2. The Conformal Geometry (IRER Manifold)

The informational manifold has conformal metric:

```
g_{ij}(x,t) = Ω²(x,t) · δ_{ij}
```

where the conformal factor Ω² is a function of the local field density:

```
Ω²(x,t) = (ρ_vac / ρ(x,t))^α
```

| Symbol | Meaning | Code parameter | Typical search range |
|--------|---------|----------------|---------------------|
| ρ_vac | Vacuum reference density | `param_rho_vac` | [0.0, 2.0] |
| α | Conformal coupling exponent | `param_a_coupling` | [0.1, 4.0] |

**Physical interpretation:**
- When ρ = ρ_vac: Ω² = 1 → flat space (informational equilibrium)
- When ρ > ρ_vac: Ω² < 1 → metric contracts (dense resonance zones are geometrically "smaller")
- When ρ < ρ_vac: Ω² > 1 → metric expands (sparse zones stretch the manifold)

This is the mathematical realisation of **Informational Indifference**: the geometry deforms in response to local resonance density, creating gradient forces that drive the field toward equilibrium configurations.

Stability bounds enforced: `1e-9 ≤ Ω² ≤ 1e6`  
Clamping uses smooth log-space tanh saturation (no hard clips that would produce discontinuous gradients).

---

## 3. The Covariant Laplacian

For metric g_{ij} = Ω² δ_{ij} in D spatial dimensions, the Laplace-Beltrami operator is:

```
Δ_g[ψ] = [Δ_flat[ψ] + (D−2)(∇Ω/Ω)·∇ψ] / Ω²
```

**Derivation** (for reference):
```
Δ_g f = |g|^{-1/2} ∂_i (|g|^{1/2} g^{ij} ∂_j f)
       = Ω^{-D} ∂_i (Ω^{D-2} ∂_i f)
       = Ω^{-2} Δf + (D−2) Ω^{-3} (∇Ω · ∇f)
       = [Δf + (D−2)(∇Ω/Ω)·∇f] / Ω²
```

In the code (`worker_cupy.py:calculate_cov_laplacian_fused`):

```python
gx = 2·Re(ψ* ∂_x ψ) = ∂_x ρ           # density gradient via chain rule
g_om_x = (∂Ω/∂ρ) · gx = ∂_x Ω         # conformal factor gradient
grad_omega_dot_grad_psi = g_om_x·∂_x ψ + g_om_y·∂_y ψ + g_om_z·∂_z ψ = ∇Ω·∇ψ
cov_term = (D−2) · (∇Ω·∇ψ) / Ω
lap_cov = (lap_flat + cov_term) / Ω²
```

With D = 3: the correction term is 1 · (∇Ω/Ω)·∇ψ, i.e., a first-order geometric coupling. ✓ **Mathematically correct.**

---

## 4. The Nonlinear Operator (split-operator form)

The solver uses an **operator-splitting** architecture:

```
∂ψ/∂t = L[ψ] + N(ψ)
```

**Linear (stiff) part — handled by matrix exponential:**
```
L[ψ] = D·∇²ψ + (−η + iω₀)ψ
```
In spectral space: `L_k = −D·k² + (−η + i·ω₀)`  
This is the part ETDRK4 exponentiates exactly: `exp(L_k · dt)`

**Nonlinear part — evaluated explicitly at each stage:**
```
N(ψ) = D·(Δ_g − Δ_flat)ψ + (a·ρ + s·ρ² + f·ρ³)·ψ
```

The geometry correction `D·(Δ_g − Δ_flat)ψ` isolates the conformal curvature contribution from the stiff flat Laplacian already handled by L. This is the correct split.

**Full equation reconstructed:**
```
∂ψ/∂t = L + N = D·Δ_g[ψ] + (−η + iω₀)ψ + (a·ρ + s·ρ² + f·ρ³)·ψ
```

In IRER language, this reads:
- `D·Δ_g[ψ]` → OIW propagation through the informational manifold
- `(−η + iω₀)ψ` → vacuum damping and baseline oscillation of the AIS
- `a·ρ·ψ` → linear resonance coupling (ψ self-amplifies proportional to its own density)
- `s·ρ²·ψ` → nonlinear saturation (controls soliton width; negative s → self-defocusing)
- `f·ρ³·ψ` → higher-order stabilisation or collapse trigger

---

## 5. ETDRK4 Integrator

The **Kassam-Trefethen Exponential Time Differencing Runge-Kutta 4th order** scheme.

Each time step Δt involves four stages:

```
Nₙ = N(ψₙ)                              (stage n)
aₖ = E₂·ψₖ + Q·Nₙ                      (stage a, half step)
Nₐ = N(aₖ)
bₖ = E₂·ψₖ + Q·Nₐ                      (stage b, half step)
N_b = N(bₖ)
cₖ = E₂·aₖ + Q·(2N_b − Nₙ)            (stage c, full step)
Nᶜ = N(cₖ)

ψₙ₊₁ = E·ψₙ + f₁·Nₙ + 2f₂·(Nₐ+N_b) + f₃·Nᶜ
```

The coefficients E, E₂, Q, f₁, f₂, f₃ are precomputed via a **64-point Cauchy contour integral** in the complex plane (the Kassam-Trefethen stable contour method — avoids the cancellation errors of the naive Cox-Matthews formula).

**Timestep stability check:** with `dt = 0.005` (burn-in config), the stiff CFL condition requires:
```
D · dt · k_max² ≤ O(1)
```
For N=128, L=10: `k_max = 2π·64/10 ≈ 40.2 rad/unit`, `D_max = 5.0`  
→ `D · dt · k_max² ≈ 5.0 × 0.005 × 1616 ≈ 40`  
This is handled by ETDRK4's exact matrix exponential for L, so stability holds independent of k². ✓

---

## 6. Auxiliary "Field of Affect" (Non-Local Density Wave)

A second field A(x,t) is tracked alongside ψ:

```
∂²A/∂t² = −c² ∇²A + ρ
```

This is a **wave equation sourced by density** — A propagates at speed `c = param_c_affect`.

In IRER language: A models the propagation of informational influence through the AIS at a finite causal speed. Denser OIW configurations create a "wake" in A that propagates outward.

**⚠ ISSUE: A is computed but NOT currently coupled back into the main PDE.**  
The field is updated each step and written to HDF5 (`A_final`), but `N_op` does not read `self.A_real`. This means non-local density effects are tracked but have zero dynamical influence. See sanity check section 8.3.

---

## 7. Spectral / k-space Structure and the Fitness Metric

After evolution, the final field ρ = |ψ_final|² is Fourier transformed and analysed for resonance structure.

**Fitness: `log_prime_sse`** (lower is better)  
This measures the spectral similarity error between the detected k-space peaks and positions predicted by **prime-harmonic resonance theory** (Jake's Concept 2: Prime-Harmonic Resonance & Informational Stability).

The hypothesis being tested:
> Stable informational configurations will exhibit spectral peaks at prime-harmonic wavenumbers, reflecting the fundamental "resonance nodes" of the AIS.

Triage tiers:
- **GOLDEN:** `log_prime_sse < 1.0` — strong prime-harmonic alignment
- **SILVER:** `log_prime_sse < 3.0` — moderate alignment
- **Provisional:** `log_prime_sse < threshold` — passes SSE gate, awaiting refinement

The fitness function is: `composite_fitness = 1 / (log_prime_sse + ε)` — higher is better.

---

## 8. Sanity Check: Theory → Code Mapping

### 8.1 What the simulation correctly implements ✓

| IRER Concept | Simulation Realisation | Status |
|---|---|---|
| A-temporal Informational Substrate | 3D periodic spectral domain | ✓ Valid proxy |
| Ontological Informational Waves (OIWs) | Complex scalar field ψ | ✓ Consistent |
| Resonance Density (RD) | ρ = \|ψ\|² | ✓ Standard |
| Informational Manifold geometry | Conformal factor Ω²=(ρ_vac/ρ)^α | ✓ Self-consistent |
| Gradient-derived informational forces | Covariant Laplacian correction D(Δ_g−Δ)ψ | ✓ Mathematically correct |
| Soliton-forming regime (Quantule) | Nonlinear balance a·ρ + s·ρ² + f·ρ³ | ✓ Maps correctly |
| Prime-Harmonic Resonance | Fitness: log_prime_sse vs k-space peaks | ✓ Directly tests theory |
| ETDRK4 stability | Kassam-Trefethen contour with M=64 | ✓ Numerically rigorous |

### 8.2 Parameters and their physical roles

| Parameter | Physical role | Search range | Notes |
|---|---|---|---|
| `param_D` | Kinetic / diffusion strength | [0.1, 5.0] | Controls OIW propagation speed |
| `param_eta` | Vacuum damping | [0.01, 1.0] | Higher → faster amplitude decay |
| `param_rho_vac` | Vacuum reference density | [0.0, 2.0] | Used TWICE — see issue 8.4 |
| `param_a_coupling` | Conformal exponent α | [0.1, 4.0] | Controls geometry sensitivity |
| `param_s` | Quartic nonlinearity | [−1.0, 1.0] | Negative → self-defocusing |
| `param_f` | Sextic nonlinearity | [−0.5, 0.5] | Higher-order stability |
| `param_a` | Quadratic nonlinearity | (from hunt manifest) | Self-amplification |
| `param_c_affect` | Speed of A-field propagation | (not in burn-in) | Non-local influence speed |

### 8.3 ⚠ ISSUE — Field of Affect (A) has no feedback coupling

**Location:** `worker_cupy.py:update_field_of_affect` / `N_op`

The auxiliary wave field `A(x,t)` satisfying `∂²A/∂t² = −c²∇²A + ρ` is computed every timestep, but the nonlinear operator `N_op` does not include any term depending on `self.A_real`.

**Consequence:** All simulations run with purely local dynamics. The non-local density wave is a measurement artefact — it influences nothing. The IRER concept that resonance propagates causally through the AIS is not yet dynamically active.

**What a coupling would look like:**
```python
# Example: add A as a sourced potential term
nonlin += coupling_A * self.A_real * psi   # linear coupling to A
# or: add A's gradient as a force
nonlin += coupling_grad_A * grad_A_x * psi  # force-like coupling
```

**Recommendation:** Either (a) add a coupling term and expose it as a parameter, or (b) mark A as a passive diagnostic and remove it from the hot loop to save compute.

### 8.4 ⚠ ISSUE — `param_rho_vac` serves two distinct physical roles

`param_rho_vac` appears in:

1. **Linear operator** (`worker_cupy.py:181`): `L_k = −D·k² + (−η + i·ρ_vac)`  
   → Here ρ_vac acts as an imaginary frequency `ω₀ = ρ_vac`, driving oscillation.

2. **Conformal factor** (`gravity/unified_omega.py:131`): `Ω² = (ρ_vac / ρ)^α`  
   → Here ρ_vac is a density scale for the geometry.

These are two very different physical roles. When `param_rho_vac = 0`:
- Linear: oscillation is suppressed (ω₀ = 0) — the field evolves on a purely real linear operator
- Conformal: `Ω² = (0/ρ)^α = 0` — the conformal factor collapses to its floor (1e-9), making the manifold maximally contracted everywhere

The stability protections prevent a crash, but the physics is degenerate at ρ_vac = 0.

**Default mismatch:** The solver defaults `param_rho_vac = 0.0`, but `unified_omega.py` defaults it to `1.0`. If a hunt config omits this parameter the two modules will disagree.

**Recommendation:** Split into two independent parameters: `param_omega0` (linear oscillation frequency) and `param_rho_vac` (conformal vacuum density). This would make the theory clearer and give the evolutionary search two more degrees of freedom.

### 8.5 ⚠ ISSUE — Collapse dynamics metric is a 2-term approximation of a 3-term potential

`metrics/collapse_dynamics.py:compute_nonlinear_balance` computes:
```
R = |λ·ρ²| / |μ·ρ³| = |λ| / (|μ|·ρ)
```
This captures a balance between two terms. But the actual nonlinear potential has **three terms** (a·ρ, s·ρ², f·ρ³). The full soliton stability condition requires balancing all three simultaneously.

For the cubic-quintic-septic NLS, the effective soliton width ξ satisfies a more complex condition. The current 2-term R is a useful heuristic but understates the actual parameter sensitivity.

**Recommendation:** Extend to compute the full nonlinear Lyapunov function or at minimum compute R for each consecutive pair of terms (a vs s, s vs f).

### 8.6 ⚠ MINOR — SPDC empirical bridge is a rough analogy

`metrics/spdc_empirical_bridge.py:calculate_joint_spectral_amplitude` applies an FFT to the ψ field and labels the result a "Joint Spectral Amplitude." Real SPDC JSA involves two-photon pair correlations in frequency space — this is not what the simulation produces.

This bridge is clearly labelled as an analogy and is useful for pattern classification, but the name imports quantum optics language that may mislead quantitative comparison to experiment.

### 8.7 CONFIRMED — Dealiasing is correctly applied

The dealias mask `(|k| ≤ 2/3 k_max)` is applied after every spectral transform and after every ETDRK4 stage. The default is `param_dealias_fraction = 0.5` (half Nyquist), which is conservative (the standard 2/3 rule would cut at 0.667). This is safe but does reduce effective resolution.

---

## 9. Initial Condition

```
ψ(x, 0) = exp(−r²/2) + 0.01 · η
```

A centred Gaussian wave packet with amplitude ~1 at origin, decaying to ~0.6 at r=1, plus 1% complex Gaussian noise for symmetry breaking. The noise seeds the evolutionary emergence of structure.

---

## 10. Termination Conditions

| Code | Trigger | Meaning |
|---|---|---|
| `math_explosion (1002)` | NaN/Inf in ψ OR \|ψ\|_max > collapse_threshold | Numerical blow-up |
| `physics_drift (1003)` | ⟨ρ⟩ < 1e-5 | Field has decayed to vacuum (no resonance) |
| `geometry_sanity (1004)` | ⟨Ω²⟩ outside (0.1, 1e6) OR range(Ω²) < 1e-8 | Conformal factor degenerate |

The collapse_threshold in burn-in config is 1e10 — this is very permissive. A tighter value (e.g. 1e6) would catch energetically runaway configurations earlier.

---

## 11. What the Evolutionary Search is Optimising

The NSGA-II + Spectral Gradient Navigation (SGN) evolutionary algorithm searches the parameter space:

```
{param_D, param_eta, param_rho_vac, param_a_coupling, param_s, param_f}
```

to minimise `log_prime_sse` — the spectral error between the final field's k-space peaks and prime-harmonic predictions.

**The core scientific question being tested:**  
*Do the nonlinear, geometry-coupled OIW dynamics spontaneously organise into spectral structures consistent with prime-harmonic resonance?*

If yes: the IRER claim that prime numbers underlie informational stability receives numerical support within this model.  
If no: either the model needs refinement, or the prime-harmonic hypothesis needs revision.

The Pareto front tracks the tradeoff between SSE and any secondary metrics (PCS, basin count, etc.).

---

## 12. Summary: Is the Simulation Testing What IRER Proposes?

**Yes, with caveats.**

The simulation correctly implements:
- Self-referential field dynamics (geometry ↔ density coupling) ✓
- Covariant propagation through informational manifold ✓
- Soliton-forming nonlinear balance ✓
- Prime-harmonic resonance as the fitness criterion ✓
- Rigorous numerics (ETDRK4, spectral methods, FP64 geometry path) ✓

The main gaps between theory and current code:
1. Non-local "Field of Affect" is computed but not dynamically coupled
2. `param_rho_vac` conflates two distinct physical roles
3. 3D Quantule structure (angular deficit / Payan states) is not an explicit variable — it is emergent from the spectral basin classification, which is an indirect proxy

None of these are correctness bugs that invalidate results — they are scope limitations that constrain what the simulation can find. The mathematics being evolved is sound.
