# γ_A = 0 baseline (negative control) for the A-field prototype

Branch: `IRER_A_FIELD_GEOMETRIC_FEEDBACK_v1_PROTOTYPE`. This is the control the γ_A>0
prototype must beat. All measured with validated instruments (corrected `prime_log_sse`
crash fix; node_count/spectral_modes validated; SDG diagnostic `IRER-SDG-DIAG-v1`).

## Setup
Corrected ETDRK4 SNCGL solver, **local geometry only**: `Ω² = (ρ_vac/ρ)^α`, γ_A = 0
(A-field computed but **decoupled** from ψ). JAX scout @48³, gain bounds, multiseed IC
(6 same-phase Gaussian seeds), 800 steps; top stable_multinode candidates deep-dived with
intact + ablation + isolated-baseline.

## Findings (the failure mode to overturn)
1. **Multiseed nodes are INDEPENDENT condensates**, not a coupled structure. Each planted
   node self-focuses and persists on its own.
2. **Isolated single seed persists** about as well as in the cluster (energy retained,
   1–2 nodes) — removing the others does not weaken a node.
3. **Ablation just removes one node**; the remaining nodes are unchanged (n→n−1, energy flat).
4. **No genuine mutual support**: deep-dive mutual_support = 1/8 (and that 1 is a blow-up
   artifact, ablation→NaN), PROMOTE_TO_CUPY = 0/8.
5. **Geometry response insufficient**: 5/8 runaway/saturated curvature (curv_max 6–139),
   the rest borderline (curv_max 0.17–62 vs validated coherent reference ≈0.03). Geometry
   does not coherently follow node structure with bounded curvature.
6. High final phase coherence (~0.95) is **inherited from the same-phase IC**, not emergent.

## Why
With γ_A = 0 the only inter-node coupling is the **local** conformal map Ω²(ρ) — pointwise,
no long-range channel. So nodes cannot support each other; they coexist independently.

## What would overturn it (γ_A>0 promotion criterion)
The A-field must change THIS exact failure mode: full cluster persists, **isolated nodes
decay/weaken vs the cluster**, **ablation disrupts the remaining nodes**, phase-scrambling
weakens support if phase-locking matters, A/Ω²/curvature follow the nodes coherently, fields
bounded (no NaN/blow-up/clipping/curvature runaway). Anything less = the prototype fails and
finite-speed A feedback (in this topology/range) does not produce mutual support.
