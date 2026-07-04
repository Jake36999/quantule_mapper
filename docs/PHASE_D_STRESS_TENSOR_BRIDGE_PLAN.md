# Phase D.2 — Informational Stress-Tensor Bridge (READ-ONLY diagnostic) — Plan

**Rationale.** C1 (`PHASE_D_C1_RESULTS.md`) showed the missing Phase-D layer is **not** "kinetic term = motion" — it
is the **measurement of directional interaction, stress, shear, bridge tension, and coupling** between stable nodes.
A scalar density ρ tells us *where* nodes are; it does not carry the **directional** structure of how they strain
and couple. The **informational stress tensor** is the candidate bridge observable — the coarse-graining variable
between "stable field nodes" and "larger communicating structures" (D.4 coupling laws, D.6 zoom-out).

**Discipline (the whole point of doing this first).** This is a **read-only measurement layer**. It is **NOT** a
force/source term. It does not touch the solver, the frozen Phase C operator, the gate, or provenance-as-certifier.
It is implemented in `jax_scout` / analysis and run on **already-saved** Phase C (and C1) fields. No production/CuPy
change. **The stress tensor is not used as an active term until it proves predictive as a diagnostic.**

## Source form (design ancestor — the "Causal Field of Affect / V13" blueprint, Tab 4)
That doc proposes:
```
T_info_{μν} = κ · ρ · ∂_μφ ∂_νφ  +  η · ∂_μ√ρ ∂_ν√ρ  −  g_{μν} · L_FMIA
```
We adopt the **form** (density-weighted phase-gradient stress + density-gradient "quantum-pressure" strain + a trace
term), and **reject** the doc's older claims (prime-log/SPDC validation targets, BSSN/Hamiltonian framing, Ω
clamping, production rollout, active coupling, `worker_cupy` refactor) — those conflict with the hardened baseline.

## The diagnostic tensor (this project, read-only)
Field ψ (complex), ρ=|ψ|², φ=arg(ψ). At each grid point build the symmetric 3×3:
```
T_ij = κ · ρ · (∂_iφ)(∂_jφ)              # phase-gradient / current stress (directional flow of |ψ|² info)
     + η · (∂_i√ρ)(∂_j√ρ)                # density-gradient strain (quantum-pressure-like)
     [ − δ_ij · L_eff ]                  # optional scalar trace term, ONLY if well-defined; else omit (report both)
```
- Start `κ=η=1` (report sensitivity). φ-gradients computed **phase-safe** (∇φ via `Im(∇ψ / ψ)` with an ρ-floor to
  avoid vacuum phase noise), matching `transfer_diag`'s `J_info = ρ∇φ` convention.
- Optional geometry term (deferred): weight by Ω²(ρ) from `unified_omega` if the metric-coupled variant is wanted —
  flagged, not required for the first pass.
- `L_eff` (trace) is under-specified in the theory (FMIA Lagrangian); the first pass runs **trace-free** and also
  with a simple `L_eff = ½(κ ρ|∇φ|² + η|∇√ρ|²)` variant, and reports whether the trace changes the conclusions.

## Derived metrics (the "zoom-out" variables)
Per point / per node / per node-pair:
- **magnitude** `‖T‖` (Frobenius), and the **isotropic vs deviatoric** split (pressure vs shear).
- **anisotropy**: eigenvalues λ₁≥λ₂≥λ₃ and eigenvectors of T; anisotropy ratio (λ₁−λ₃)/(λ₁+λ₂+λ₃); **principal
  stress direction** (eigenvector of λ₁).
- **shear**: off-diagonal magnitude / the deviatoric norm.
- **divergence** `∇·T` (a force-density-like observable; its magnitude + direction).
- **node-integrated stress**: integrate T (and ‖T‖, ∇·T) over each detected node region (`transfer_diag.detect_nodes`).
- **bridge-projected stress** between node **pairs**: T projected **along** the node-pair axis (tension/compression)
  vs **perpendicular** (shear) — evaluated on the inter-node segment/corridor.
- **stress flux across the inter-node corridor** cross-section (∮ T·n̂).
- **correlations** (the actual test): do the above track `J_info=ρ∇φ`, current/vorticity, node **spacing**,
  bridge/corridor conductance (`transfer_diag`), and **instability** (C1 runs) — **better than density alone**?

## Reuse (don't reinvent)
`jax_scout/transfer_diag.py` already provides `detect_nodes`, node centroids, corridor/bridge metrics, `J_info=ρ∇φ`,
and conductance. The stress tensor is the **field-level generalisation** of that graph-level bridge diagnostic
(FMIA transfer): `transfer_diag` = graph-level interaction; `T_info` = field-level directional stress; D.4/D.6 = the
node-level laws derived from them. Node detection + bridges come from `transfer_diag`; T_info adds the directionality.

## Test set (existing saved fields — no new sims required for the first pass)
1. **a\*** stationary attractor (`FEB_GAIN_LADDER…/a1.15_…probe.npz`) — the pinned isolated reference.
2. **multi-node stable configs** — node-count / per-blob families from the feb-basin work (K3–K8), and the
   seed620/621 6-node a\* states.
3. **bridge / FMIA candidates** — any corridor/bridge-hunt states, if available.
4. **C1 unstable-dispersion runs** — as a **contrast** (does ‖T‖/∇·T spike where a\* destabilises?).

## Success / failure classification (pre-registered)
- **`STRESS_BRIDGE_PROMISING`** — tensor metrics **distinguish** pinned-isolated nodes from bridge-forming/coupled
  ones; stress projections **correlate** (above a null) with measured phase/current/energy transfer; anisotropy
  predicts node spacing / bridge direction / instability / coupling **better than density alone**.
- **`STRESS_TENSOR_DENSITY_PROXY_ONLY`** — the tensor metrics collapse to a monotone function of ρ (no extra info).
- **`STRESS_TENSOR_NO_SIGNAL`** — no predictive relation to coupling / bridge formation / instability.
- **`STRESS_TENSOR_NUMERICAL_ARTIFACT`** — large values only track numerical blow-up (e.g. the C1 fragmenters),
  not physical coupling.

## Outputs
`jax_scout/info_stress_tensor.py` (read-only extraction + metrics; numpy, runs on saved fields) +
`docs/PHASE_D_STRESS_TENSOR_BRIDGE_RESULTS.md` (the classification + evidence). **Only if `PROMISING`** does D.4/D.5
use it to guide coupling-law extraction or a transport branch. **Never** promote it to an active source term on the
strength of this diagnostic alone.
