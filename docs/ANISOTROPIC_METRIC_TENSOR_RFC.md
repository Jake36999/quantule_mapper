# RFC — Anisotropic Metric Tensor (resonance-weighted g_ij), staged

**Status:** DRAFT design contract. NOT implemented in the solver. Stage 0 of a staged path
(RFC → passive diagnostic → minimal proxy → tiny validation). JAX scout only; production
`gravity/unified_omega.py` untouched. **Date:** 2026-06-21.

## 1. Why (theory + measurement converge)

- **Measurement:** the γ_A=0 universe couples geometry to a SCALAR (Ω²(ρ)) → isotropic →
  globally-coupled web (global_mode_fraction ≈ 0.89). Even the current-coupled A-field localizes
  on bridges but rarely restructures the coupling (1/3 web→wires), because the geometry it feeds
  into is still scalar/isotropic — the manifold has no directional degrees of freedom.
- **Theory (the framework's own prescription):** the ASTE engineering brief states the scalar
  Ω²(ρ) coupling is *Nordström-style scalar gravity* and must be upgraded to source geometry from
  the **Informational Stress-Energy Tensor T_μν** (Madelung form ψ=√ρ e^{iφ}), with anisotropy
  from phase energy `ρ ∂_iφ ∂_jφ` and density gradients, tanh-stabilized before conformal scaling.
  Provenance Concept 22 specifies a "resonance-weighted metric tensor g_ij(RD,PAS)", not a scalar.
- **Conclusion:** the missing directional degree of freedom is the spatial metric's anisotropy.

## 2. Minimal anisotropic extension

Keep the scalar conformal factor Ω² as the base; add a bounded, traceless anisotropy:
```
g_ij = Ω² ( δ_ij + λ Q_ij )            Q_ij traceless, ||Q|| bounded, λ ∈ [0, λ_max]
```
λ = 0 reproduces the current scalar conformal geometry EXACTLY (the equivalence gate).

## 3. What sources the anisotropy Q_ij (candidates, tested passively in Stage 1)

Bounded, traceless (subtract δ_ij·tr/3), normalized:
- **phase-current direction:** `Q^J_ij = J_i J_j/(|J|²+ε) − δ_ij/3`,  J = ρ∇φ = Im(ψ*∇ψ)
- **stress-energy (phase):** `T_ij = ρ ∂_iφ ∂_jφ = J_i J_j/ρ` (deviatoric part)
- **density-gradient:** `Q^ρ_ij = ∂_i√ρ ∂_j√ρ` (deviatoric, normalized)
- **A-field direction:** `Q^A_ij = A_i A_j/(|A|²+ε) − δ_ij/3` (current-coupled branch)
- **hybrid:** J/A-aligned anisotropy where J and A agree.

The theory-preferred source is the stress-energy/phase-current tensor (direction = energy flow),
which is the directional information the scalar Ω² discards.

## 4. Reduction, guards, validation

- **Reduction:** λ=0 → g_ij = Ω² δ_ij (current model), bit-identical (equivalence test required).
- **Guards:** Q traceless + clipped (||Q|| ≤ Q_max) so eigenvalues of (δ+λQ) stay positive
  (no metric singularity / signature flip); tanh-stabilize the source trace before scaling
  (per ASTE brief); cap anisotropy strength λ_max; reject runaway / bridge-saturation /
  space-filling exactly as the scalar branches do; k=0 / DC handled; contract-stamped
  (`IRER-SNCGL-ANISO-METRIC-ETDRK4-v1`), default-off (λ=0), segregated from λ=0 rankings.
- **Validation (Stage 3):** on best A-localized bridge configs + the 1 partial-web→wires case
  (gen18) + ≥2 scalar web controls + 1 no-bridge control, measure: energy bounded@1600,
  curvature bounded, bridge non-saturated, A localization, anisotropy localization,
  global_mode_fraction DROP, pairwise_fraction RISE, phase-current localization, bridge/void
  separation under phase kick, resolved (non-boundary-pinned) response, seed robustness.
  Classes: `ANISOTROPIC_GEOMETRY_PROMISING` / `ANISOTROPIC_STRUCTURE_ONLY` /
  `ANISOTROPIC_DISTORTION_REJECT` / `CONTRACT_FAIL`.

## 5. Stage gate

Implement the Stage-2 minimal proxy (anisotropic diffusion `∂_i(D_eff,ij ∂_j ψ)`,
`D_eff,ij = D(δ_ij + λ Q_ij)`, λ=0 equivalence preserved) ONLY if the Stage-1 PASSIVE diagnostic
shows that a Q_ij derived from J_info / stress-energy / A is **localized on the bridge, aligned
with the bridge axis, and distinguishes the one web→wires case (gen18) from the failures (gen29
anti-shift, gen20 no-shift)**. Immediate question: *does a passive anisotropic tensor explain the
difference between global-web response and the partial web→wires case?* If no, simple J/A tensors
are insufficient and Payan-state/phase-alignment coupling becomes the next RFC.
