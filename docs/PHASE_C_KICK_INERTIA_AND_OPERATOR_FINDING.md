# Phase C — Kick / Inertia Test and the Operator Finding

**Date:** 2026-07-02 · run `FEB_KICK_INERTIA_20260702_122013` · script `jax_scout/feb_kick_inertia.py`
· N96 / L=10 / dt=0.005 · geometry frozen `e8d6a78ea` · target = confirmed a\* bound state (param_a ×1.15 ≈ 0.552,
seed 619, 4-node), boosted from its gain-ladder `psi_fin`.

## Accepted conclusion

The a×1.15 bound state is a **robust standing dissipative soliton / attractor**. It does **not** show Galilean
inertial transport under a phase-ramp kick. That negative is **structurally expected**: the current solver has a
real, diffusive kinetic operator at ω₀=0, so phase-gradient momentum is **damped rather than advected**. The
kick test therefore does **not** falsify IRER-style *relational* matter-likeness; it falsifies only
**Newtonian / coasting inertia in this dissipative substrate**. The honest observation is that the imposed
phase-gradient perturbation was dissipated while the attractor remained pinned and intact.

Formal verdicts: `GALILEAN_KICK_NO_TRANSPORT_CONFIRMED`, `DISSIPATIVE_SUBSTRATE_NO_INERTIAL_CHANNEL`.

## 1. Kick test setup

A Galilean phase-ramp boost `ψ → ψ · exp(i k x)` (a momentum kick) was applied to the settled a×1.15 field.
`k = 2π·n/L` is quantised to keep the periodic box seam-free, so n=0,1,2,3 are the only clean on-lattice
boosts (n=0 = no-kick control for intrinsic drift/breathing). Each kicked state was evolved T=10,000 steps,
with the center-of-density sampled every 100 steps via a circular (periodic-safe) centroid, unwrapped, then
linear-fit for velocity `v_x`. Coherence tracked as mass, peak, node-count, and transverse COM.

## 2. Results

| kick n | k | v_x | displacement (boxes) | mass ratio | peak ratio | nodes |
|---:|---:|---:|---:|---:|---:|---:|
| 0 (control) | 0.000 | +0.0002 | ~0.000 | 0.997 | 1.001 | 4→4 |
| 1 | 0.628 | +0.0002 | ~0.000 | 0.994 | 0.997 | 4→4 |
| 2 | 1.257 | +0.0002 | ~0.000 | 0.984 | 0.989 | 4→4 |
| 3 | 1.885 | +0.0003 | ~0.000 | 0.968 | 0.980 | 4→4 |

Mobility μ = v/k ≈ 0 (velocity is kick-independent and sits at the n=0 control noise floor). The reported
"effective mass M/μ ≈ 4.6×10⁸" is the degenerate μ→0 (immovable) limit, **not** a finite particle mass.

## 3. Interpretation of the run

The velocity is identical to the no-kick control at every kick strength — displacement is ~0.001 of a box
over the whole run regardless of kick. The structure stays intact: **nodes 4→4, transverse COM ≈ 0, mass and
peak preserved.** The kick *is* felt — mass sheds slightly more with larger kicks (0.997→0.968) — so the
imposed phase gradient is being **dissipated, not advected**. No fragmentation and no decay products were
observed.

## 4. Operator audit — why the null is structural, not a soliton property

The evolution operator was inspected directly (`jax_scout/physics.py`):

- **Linear propagator** (`physics.py:289`):
  ```
  L_k = −D_diff · k²  −  η  +  i·ω₀
  ```
  At the tested bound-state parameters **ω₀ = 0**, so `L_k = −D·k² − η` is **purely real**. Every Fourier mode
  simply damps (`E = exp(L_k·dt) = exp[(−D k² − η)·dt]`, a real decay factor). There is no oscillatory /
  advective phase evolution.
- **Kinetic RHS term** (`physics.py:174, 222`): `D_diff · (lap_cov − lap_flat)` with `D_diff` **real** — a real
  `D·∇²` (diffusion), not a Schrödinger `i·D·∇²` (dispersion).
- The solver itself documents the distinction (`physics.py:232–235`): a real `D·∇²` "just smooths/dissipates";
  only an **imaginary** `i·D` would be "metric-like (Schrödinger, |ψ|²-conserving), redirects flow."

**Therefore the model at these parameters is a real (dissipative) Ginzburg–Landau reaction–diffusion system with
no dispersive / inertial term.** A phase-ramp boost `exp(i k x)` has **no advective channel by construction** —
its k-modes are exponentially damped, not carried. The pinning is a property of the *substrate*, not of the
a×1.15 attractor specifically: nothing in this model can coast.

## 5. What this does and does not say

**Does say:**
- Newtonian / ballistic inertia is **not defined** in this dissipative version of the model; the kick test was
  probing a structurally-absent quantity.
- The a×1.15 attractor absorbs and dissipates an imposed phase gradient while remaining coherent and pinned.

**Does not say:**
- It does **not** show the structure "failing" to be matter-like, "shattering", or decaying — the data show a
  preserved 4-node structure retaining most of its mass and peak.
- It does **not** falsify relational / field-bound matter-likeness (the IRER framing of stabilized geometric
  perturbations responding to their field environment). That question requires a different probe.

## 6. The next (relational) mobility test

For a dissipative attractor the right mobility question is not "can it coast after an impulse?" but **"can it
track a slowly-moving energetic/geometric preference?"** — i.e. **adiabatic drag**: introduce a weak localized
"preference" (a movable gain/loss or parameter well), (1) hold it static and test whether the attractor biases
toward it, then (2) move it slowly and measure tracking, lag, deformation and a slip threshold. This is a
**default-off physics variant** (V₀=0 reproduces baseline exactly), kept segregated from Phase C basin results,
and labelled as **relational mobility / adiabatic tracking** — never inertial-matter language. Design in
`docs/PHASE_C_ADIABATIC_DRAG_DESIGN.md` (pending review before the moving-well run).

## Provenance

Run `FEB_KICK_INERTIA_20260702_122013`; kicks n=0–3, T=10,000, dt=0.005, N=96, L=10; target field
`sweep_runs/FEB_GAIN_LADDER_LONGT_T72000_20260701_175708/a1.15_ladder_T72000_probe.npz`; geometry frozen
`e8d6a78ea`; no solver changes (read-only boost of a saved state).
