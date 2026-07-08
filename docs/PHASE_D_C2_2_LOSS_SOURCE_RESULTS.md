# Phase D / C2.2 — Loss-Source Isolation for the Native Soliton's Ballistic Transport

**Headline: the kick-associated "radiation" that made C2.1's transport look lossy is NOT geometry-induced and is
DOMINANTLY a numerical / boost-protocol artifact — it converges toward zero as the timestep is refined. The native
conservative soliton's ballistic mobility (v∝k) is real and resolution-robust; the transport gets progressively
CLEANER with better numerics.** This upgrades the C2.1 reading: the conservative substrate supports genuine ballistic
soliton transport that is increasingly clean in the continuum limit, with at most a small (≤~0.03 at n=2, N=96-bounded)
unresolved residual.

**Classification (two-level):**
- **Primary: `C2_BOOST_PROTOCOL_RADIATION`** — the kick-associated loss is a discretization artifact of the
  instantaneous plane-wave boost at finite dt; it falls monotonically as dt→0 (n=2 loss 0.205 → 0.132 → 0.090 across
  dt = 1e-3, 5e-4, 2.5e-4), ~85% removed by dt-refinement alone.
- **Trend / sub-verdict: `C2_PURE_NLS_CLEAN_TRANSPORT_SUPPORTED` (in the continuum limit)** — with geometry off (pure
  NLS, Galilean-invariant) the mobility is robust (v/k → ~0.038, r²=1.00) and the loss extrapolates to ~0.03, an upper
  bound since N is held fixed. Clean coherent transport is recoverable in the refined limit, not yet demonstrated at a
  single fixed resolution.
- **Explicitly NOT `C2_GEOMETRY_BREAKS_CLEAN_TRANSPORT`** (the C2.1 hypothesis) — refuted below.

## Method
Mirror only; conservative substrate (`kinetic_mode="conservative"`) reached via `build_operators`; Phase C default
(dissipative) untouched; no clipping; no matter claims. Native solitons from C2.1: A=1.0 σ=0.15 and A=0.5 σ=0.15.
- **Geometry-off switch (exact):** `param_a_coupling = 0` makes `omega_sq ≡ 1` and `d_omega_d_rho ≡ 0`, so the
  covariant Laplacian collapses to the flat Laplacian and the geometric correction `D_diff·(lap_cov − lap_flat)`
  vanishes identically → a **pure cubic-quintic-septic NLS, which is Galilean-invariant**. If geometry breaks clean
  transport, geometry-off must remove the kick-associated loss.
- **Boost / loss metrics:** Galilean boost `ψ·exp(ikx)`, k=2πn/L. Kick-associated loss = (n=0 control mass) − (boosted
  mass) over a matched physical boost window — isolates the kick-specific loss from the resting quasi-conservative
  bleed. Mobility μ = dv/dk from the k-ladder; v/k per kick; r² of the linear position fit.
Harness `jax_scout/phase_d_c2_2_loss_source.py` (N-aware grid/centroid so the resolution test works).

## Result 1 — geometry is NOT the loss source (N=96, dt=1e-3, kicks n=0..3)
| candidate | geom | n=1 loss | n=2 loss | n=3 loss | loss/k² | μ |
|---|---|---|---|---|---|---|
| A=1.0 σ=0.15 | ON  | 0.054 | 0.203 | 0.409 | 0.117 | 0.048 |
| A=1.0 σ=0.15 | OFF | 0.051 | 0.205 | 0.412 | 0.118 | 0.062 |
| A=0.5 σ=0.15 | ON  | 0.041 | 0.193 | 0.469 | 0.130 | 0.008 |
| A=0.5 σ=0.15 | OFF | 0.039 | 0.187 | 0.462 | 0.128 | 0.031 |
**Kick-associated loss is identical geometry-on vs geometry-off** (0.054↔0.051, 0.203↔0.205, 0.409↔0.412). A pure NLS
is Galilean-invariant yet radiates the *same* amount → **the C2.1 "Ω²(ρ) breaks Galilean invariance" hypothesis is
refuted.** What geometry *does* do is act as a **mild mobility drag** (μ larger with geometry off — A=0.5: 0.008→0.031)
— it slightly suppresses motion, it does not source the radiation. Loss scales ∝k² in all four cases (n=3 confirms).

## Result 2 — the loss is dt-numerical: it converges away (N=96, geom off / pure NLS, A=1.0)
| dt | n=0 control mass | n=1 kick-loss | n=2 kick-loss | loss/k² | v/k (n=2) |
|---|---|---|---|---|---|
| 1e-3   | 0.793 | 0.051 | 0.205 | 0.118 | 0.057 |
| 5e-4   | 0.822 | 0.028 | 0.132 | 0.083 | 0.045 |
| 2.5e-4 | 0.876 | 0.020 | 0.090 | 0.057 | 0.039 |
- **Monotone convergence toward zero.** n=2 kick-loss 0.205 → 0.132 → 0.090 (each dt-halving reduces it; successive
  decrements 0.073, 0.042 → ratio 0.58 → effective order p≈0.8, sub-linear, consistent with a dealiasing/aliasing
  interaction of the moving structure rather than smooth 4th-order time error).
- **Aitken/Richardson extrapolation → continuum (dt→0) n=2 loss ≈ 0.03** (down from 0.205 at dt=1e-3: **~85% of the
  "loss" was dt-discretization error**). This ~0.03 is an **upper bound** on any physical loss — it still holds N=96
  fixed, so spatial refinement would only reduce it further.
- **The resting bleed is also dominantly numerical:** the n=0 control retains 0.793 → 0.822 → 0.876 as dt→0.
- **Mobility is real and gets CLEANER with dt:** v/k → ~0.038 with n=1/n=2 values converging (0.037/0.039 at
  dt=2.5e-4) → genuine v∝k. The dt=1e-3 apparent super-linearity (v/k 0.049→0.057) and inflated μ=0.062 were themselves
  numerical; the converged mobility is μ≈0.038.

## Result 3 — spatial (N) convergence: BLOCKED by CFL (honest caveat)
`N=128, dt=1e-3` went **non-finite in ~2 min (COLLAPSE)** — at higher N the dispersion `|D·k²|` is stiffer and dt=1e-3
is above the stable window. A clean N-convergence needs a *paired* smaller dt (≈2.5e-4) and was not run here (cost +
collapse risk; the dt-ladder already gives a decisive, unconfounded answer). So the spatial-discretization component of
the residual is untested — noted as the top follow-up, and a reason the ~0.03 dt-residual is an upper bound.

## Interpretation
The C2.1 "lossy ballistic transport" was **substantially a numerical artifact** of (i) an instantaneous plane-wave
boost applied to a relaxed-Gaussian (not the exact NLS soliton) and (ii) finite-dt evolution of a moving localized
structure against the dealias mask. As dt→0 the loss converges away and the ballistic mobility sharpens. Geometry
plays no role in the radiation (it is a mild drag on speed). So the honest upgraded picture:
- **Ballistic transport is genuine and resolution-robust** (v∝k, μ≈0.038, r²=1.00) — the first real motion in Phase D
  stands and strengthens.
- **The lossiness is dominantly numerical / boost-protocol**, not intrinsic geometry-breaking; clean coherent
  transport is recoverable in the refined continuum limit (≤~0.03 residual at N=96, likely smaller with N-refinement).

## Comparison across the Phase D arc
| | dissipative (Phase C / D.1–D.6) | conservative C2.1 (dt=1e-3) | conservative C2.2 (dt→0) |
|---|---|---|---|
| motion | pinned (μ≈0.001) | ballistic, "lossy" (μ≈0.05) | ballistic, increasingly clean (μ≈0.038) |
| kick-loss n=2 | n/a (no motion) | ~0.20 | ~0.09 → extrapolates ~0.03 |
| loss cause | — | *thought* geometry | shown: dt/boost-protocol numerical |

## Addendum (2026-07-08): the algebraic component has since been localized
A parallel Codex diagnostic campaign on the CuPy side (see `docs/PHASE_D_C2_CONTRACT_REVIEW.md` and
`quantule_viz/outputs/conservative_geometry_campaign/`) independently reproduced the dt-scaling (ETDRK4 one-step
defect ∝ dt²) and, using an RK4 diagnostic stepper ~1.5e5× less lossy per step, exposed and localized a small
REAL algebraic norm flux (~1e-4 fractional per unit time) to the `geometry_covariant_correction` term: the
implemented `lap_cov` is the Laplace–Beltrami operator of the live conformal metric, self-adjoint w.r.t. the Ω³
measure (audit near-pass 2.15e-4) but not the flat one, and not canonical for a live Ω(ρ). This refines this
report's conclusion: geom-ON ≈ geom-OFF here because the ETDRK4 stepper error dominates at the tested dt; the dt→0
residual decomposes into remaining stepper/boost error + (geometry-ON only) that identified algebraic flux. The
pure-NLS (geometry-off) branch is algebraically norm-conserving — strengthening the clean-transport trend.

## Follow-ups (not done)
1. **Paired N-dt convergence** (N=128 @ dt≈2.5e-4, N=160 @ smaller dt) to test spatial convergence and drive the
   residual toward its true continuum value — the one loose end.
2. **Exact NLS soliton profile** (imaginary-time / Petviashvili ground state) instead of a relaxed Gaussian, so the
   boost acts on a true eigenstate — removes the boosted-non-eigenstate shelf.
3. **Adiabatic / co-moving boost** as an independent check that the residual is not acceleration-induced.
4. Only after a clean single-resolution mover: **two-node interaction** (C2.1 Stage 4, still gated off).

## Guardrails honoured
Mirror only; geometry-off via the existing `param_a_coupling` (no new code path); Phase C dissipative default
byte-identical and untouched; no clipping/caps added to force conservation; convergence + CFL collapse documented
honestly; **no matter/transport over-claims** — the loss is called numerical only on the strength of the dt-convergence
trend, the residual is reported as a bounded open item, and mobility is asserted from v/k + r² across resolutions.
