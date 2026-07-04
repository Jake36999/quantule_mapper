# Phase D.2 — Informational Stress-Tensor Bridge, First-Pass Results

**Classification: `STRESS_TENSOR_FIRST_PASS_INCONCLUSIVE`** (not a pure density proxy, but no clean coupling/bridge
signal in the feb/a\* family — a proper test needs the D.3 node library + metric refinements). Read-only diagnostic
(`jax_scout/info_stress_tensor.py`); no solver/gate/physics/production change.

## What ran
`T_ij = ρ·∂_iφ ∂_jφ + ∂_i√ρ ∂_j√ρ` (κ=η=1) on four saved Phase C fields, reusing `transfer_diag`
(`detect_nodes`, `geometry_fields`=Ω²/R/J, `corridor_pair_metrics`). Metrics: ‖T‖, shear/deviatoric fraction,
eigenvalue anisotropy (top-2% voxels), density-proxy rank-correlation, and per-node-pair **axial** vs perpendicular
projected stress on the inter-node segment alongside the corridor conductance / path-align / J-flux.

| config | nodes | ‖T‖ mean | shear frac | aniso(top2%) | ‖T‖~ρ rankcorr | axial-frac~conductance r |
|---|---|---|---|---|---|---|
| a\* (4-node) | 4 | 0.284 | 0.82 | 1.00 | 0.55 | −0.79 |
| a\* seed620 (6-node) | 6 | 0.272 | 0.82 | 1.00 | 0.51 | +0.50 |
| a\* seed621 (6-node) | 6 | 0.297 | 0.82 | 1.00 | 0.52 | −0.73 |
| grower a1.20 | 4 | 0.349 | 0.82 | 1.00 | 0.57 | −0.53 |

## Reading (honest)
- **Not a pure density proxy** (the `DENSITY_PROXY_ONLY` fail is *not* triggered): ‖T‖ rank-correlates with ρ only
  ~0.5, so ~half its spatial structure is directional/other — the tensor carries information beyond density.
- **But the structural metrics don't discriminate** stable a\* from the grower: shear-fraction (0.82) and top-2%
  anisotropy (1.00) are **identical** across all four. That is largely **by construction** — the phase-current term
  `κ·J⊗J/ρ` is rank-1, so the highest-stress voxels are trivially maximally anisotropic and shear-dominated. The
  metric is reading the construction, not the physics.
- **No consistent bridge signal**: the axial-stress-fraction ↔ corridor-conductance correlation flips sign across
  configs (−0.79 / +0.50 / −0.73 / −0.53) with only 6–15 node-pairs each — noise, not a robust "stress sees the
  bridges" relation. This echoes the earlier geometry-routing nulls ([[routing-null-promising-for-payan]],
  [[stage-b-tensor-geometry-no-support]]): a J-based current diagnostic already routed null, and a stress tensor
  built from J inherits that.

## Why it's INCONCLUSIVE, not a firm negative
The first pass is **under-powered and incomplete**, in exactly the way that matches the project's own hunch that
*node spacing/formation and a diverse node population* are the missing ingredients:
1. **Too little contrast.** All four are feb/a\* family (4–6 nodes, few well-separated pairs). There are no
   deliberately *bridged* vs *isolated* configs, no spacing sweep — so "does stress predict coupling?" can't be
   answered here. **This needs D.3 (node-library expansion).**
2. **Metrics unfinished.** Not yet computed: the **∇·T force-density** (the most physically-meaningful coupling
   observable), the **density-strain term isolated** from the rank-1 current term (the non-rank-1 part may carry the
   real coupling info), and per-pair **phase-difference / spacing** regressions. These are the next refinements.

## Conclusion → feeds the roadmap
The stress tensor is **worth keeping as a candidate bridge observable** (it is not a density proxy), but it **cannot
be validated on the feb/a\* family alone.** The first pass therefore *reprioritises the roadmap*: **D.3 (Phase C
node-library expansion — diverse spacing/node-count/bridging via targeted Hunter searches) is the enabling
prerequisite for a proper D.2 test.** Re-run this diagnostic (with ∇·T + term-separation + a bridged-vs-isolated
contrast) on that library before any decision to elevate the stress tensor beyond a diagnostic. **It remains
read-only; it is not, and will not be, an active source term on this evidence.**
