# Phase D / C2.4 — Local Boost (Zero Winding): Pinning Is Physical; the Flow-Through Mechanism

**Classification: `C2_LOCAL_BOOST_STILL_PINNED_PHYSICAL`.** A phase ramp with the Galilean-correct gradient applied
*locally across the core* — with the total ring winding exactly zero (∮∇φ = 1e-17, verified) — produces the **same
~0.5%-of-Galilean creep as the winding boost**. The C2.3 ring-winding hypothesis is refuted as the *cause* (it was a
correct description of where momentum sits, but topology is not why the core doesn't move). The quasi-soliton is
**physically pinned**: imposed momentum becomes steady current flowing *through* the stationary structure, not
translation of it. Pure NLS geometry-off, N=96, dt=2.5e-4, T=6, settled object loaded from C2.3 (no re-settle).

## Result table
| k_loc | v_cent (r²) | v/2Dk_loc | μ = v/k | mass ret | core frac (start→end) | P: start→end |
|---|---|---|---|---|---|---|
| 0 (control) | +0.0000 (0.96) | — | — | 0.876 | 0.860→0.876 | ~0 |
| 0.314 | +0.0093 (1.00) | **0.0054** | 0.0296 | 0.873 | 0.860→0.874 | 3892→3090 |
| 0.628 | +0.0185 (1.00) | **0.0054** | 0.0295 | 0.862 | 0.860→0.871 | 7782→6072 |

## What the numbers pin down
1. **Protocol-independence ⇒ physical pinning.** Local-boost mobility μ=0.0295 vs winding-boost μ≈0.035 — and the
   ratio (0.84) matches the imparted-momentum ratio (P0_local/P0_winding = 7782/9620 = 0.81). In both protocols the
   centroid velocity is **≈0.7% of the Ehrenfest velocity 2D·P/M**, exactly ∝ total imparted momentum. Same weak
   drag channel; topology irrelevant.
2. **The bump-on-condensate explanation is dead (honest retraction of an interim reading).** The core-fraction
   diagnostic shows the core holds **86% of the mass** throughout; the background (~14%) cannot absorb the missing
   momentum as a heavy reservoir.
3. **The actual mechanism: flow-through.** Momentum is ~conserved on kick timescales (78–79% retained at T=6, the
   decay being the known numerical sink), the mass-weighted phase gradient stays large (mean local velocity ~2.2 at
   k=0.628), yet the density is static. By continuity (∂ρ/∂t = −∇·J, J = 2Dρ∇φ), a static density with nonzero
   through-core current requires ∇·(ρ∇φ) ≈ 0: the system relaxes to a **steady superflow streaming through the
   stationary density profile** — a fountain, not a projectile. Only ~0.7% of the imposed momentum ends up in
   density-advecting form.
4. **Why this is allowed:** Galilean invariance forces a *true soliton* to translate under exactly this phase
   structure — but C2.3 proved **no true stationary soliton exists** in this substrate (no Petviashvili fixed point;
   saturation caps the branch). The quasi-soliton is a metastable self-trapped standing structure, not an eigenstate
   family with a moving member; nothing obliges it to advect, and empirically it does not.
5. **Local boosts are gentler:** kick-loss at k=0.628 is 0.014 vs 0.020 for the winding boost at the same k and
   window — but the transport outcome is identical.

## Phase D consequence: the conservative transport arc closes NEGATIVE for this substrate
- Trivial co-motion excepted (Galilean covariance moves structure *and* background together — no relational motion),
  **structure cannot be put in relative motion with respect to the background**: imposed relative momentum becomes
  flow-through plus an ~0.03·k drag creep. Combined with the dissipative sector (gain/loss-pinned, D.1–D.6):
  **neither IRER substrate tested so far supports ballistic matter-like transport of its native structures.**
  Dissipative: pinned by gain/loss balance. Conservative (as-implemented + pure NLS): no true soliton exists, and
  the metastable structure decouples from momentum.
- The measured mobilities stand as real, reproducible *drag coefficients* (μ≈0.03–0.04 conservative, μ≈0.001
  dissipative), not transport channels.
- Stage-4 two-node in this substrate is moot for transport (nothing moves relationally); it could only study
  flow-through interactions.

## What remains open (the honest branch points)
1. **C2′ canonical geometry** (`docs/PHASE_D_C2PRIME_CANONICAL_GEOMETRY_RFC.md`): a *different* substrate — whether
   its exact-conservative geometry coupling admits true solitons (and hence transport) is unknown; would need its
   own soliton hunt after the gates.
2. **C3 (second-order/wave kinetic term)** from the Phase D kinetic-operator RFC: the natural "matter has inertia"
   candidate — hyperbolic dynamics carries momentum in the field configuration itself rather than the phase, so the
   flow-through decoupling mechanism does not apply. Untested.
3. **C4 (non-local kinetic term)**: untested.
4. Different nonlinearity coefficients (the no-soliton result is specific to the feb/a\* cubic-quintic-septic values;
   a coefficient family where a true soliton branch exists below saturation would reopen the conservative question).

## Provenance + guardrails
`jax_scout/phase_d_c2_4_local_boost.py`; run `sweep_runs/C24_LOCAL_N96` (+smoke `/tmp/c24_smoke`). Object:
`sweep_runs/C23_N96_k2/object_psi.npy` (settled, deterministic). Zero-winding construction verified to 1e-17;
core-fraction diagnostic r<2.5. Mirror-only; geometry-off (contract blocker absent); Phase C untouched; no clipping;
no matter claims; the interim bump-on-condensate speculation is explicitly retracted above.
