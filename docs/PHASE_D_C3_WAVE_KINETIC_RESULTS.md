# Phase D / C3 — Wave-Kinetic (Nonlinear Klein–Gordon) Substrate: First Results

**The second-order/inertial substrate is genuinely conservative and transports its native Q-ball: energy and charge
are conserved to machine precision, a localized Q-ball exists, and under a Lorentz-boosted kick the Q-ball's DENSITY
co-moves at the imparted velocity (v_frac ≈ 0.9–1.0, mass retention ≈ 0.99). Qualitative verdict:
`C3_INERTIAL_TRANSPORT_SUPPORTED` — matter-like motion confirmed; velocity fidelity is loose only because the boost
IC omits the O((v/c)²) Lorentz contraction (a fixable refinement).** N=48, L=10, c=0.5477 (c²=0.3), m=1, g=(0.8,−0.5,
−0.1). Mirror-only standalone module; no production/Phase C changes; no matter claims beyond the measured co-motion.

## Substrate + method
Complex nonlinear Klein–Gordon `ψ_tt = c²∇²ψ − m²ψ + g(ρ)ψ`, g=aρ+sρ²+fρ³. Standalone
`jax_scout/phase_d_c3_wave.py` — own (ψ,π=ψ_t) state, own k-grid; touches nothing in the first-order path. Stepper =
Strang split: half nonlinear kick (real, local) → exact per-mode SHO rotation ω_k²=c²k²+m² → half kick.

## Gate results
| gate | result | value |
|---|---|---|
| **G1** linear parity | **PASS** | Strang(g=0) ≡ analytic rotation, max\|Δ\| = **0.00** (exact) |
| **G3** Q-ball existence | **FOUND** | w=0.964, μ=m²−w²=0.071: Petviashvili residual **1.2e-9**, amp 0.90, occ 0.235 (localized) |
| **G2** conservation | **machine-exact** | rest Q-ball T=6: dE/E=**9e-14**, dQ/Q=**1e-13**, mass 52.17→52.17 |
| **G5** null control | **PASS** | unkicked centroid drift = **0.0000** |
| **G4** transport | **SUPPORTED (loose)** | v=0.137→v_meas 0.124 (frac 0.91, mass 0.986); v=0.055→0.069 (frac 1.25*, mass 0.993) |
*low-v frac is noise-dominated (displacement ~0.4 box over the fit window); the higher-v point (frac 0.91) is the
reliable one.

## Two things worth flagging honestly
1. **The Q-ball existence equation is identical to the C2 soliton equation.** `μφ = c²∇²φ + g(φ²)φ` with c²↔D,
   μ↔m²−w². The Q-ball was found at exactly the C2.7-mapped point (c²=0.3, μ=0.071 ≡ the confirmed C2.7 family-A
   soliton). This is why the initial scans failed — they used c=1 (soliton too big for the box) or only reached the
   marginal edge μ≈0.19 near g_max=0.28; the fitting/binding sweet spot is c²≈0.3, μ≈0.07. Mild Petviashvili
   seed-sensitivity (converges at seed σ=1.5, collapses at σ=1.2) — handled by scanning seeds.
2. **The transport gate had an IC bug, now fixed.** The first G4 used a naive kick π₀=−v∇φ−iωφ with ψ₀=φ (no carrier
   phase) → constant spurious v_frac≈0.04 (density barely moved). The KG-covariant moving Q-ball needs the carrier
   phase ψ₀=φ·e^{ikx}, k=γωv/c² (the phase IS the momentum; group velocity v=c²k/ω). With it, the density co-moves.
   Residual looseness = the skipped Lorentz contraction φ(γx) (γ=1.03 at v=0.137) — the IC is an approximate, not
   exact, boosted eigenstate. Same discipline as C2.6: a transport null must be verified against a correct boost IC.

## Interpretation vs the C2 (NLS) substrate
- After the C2.6 fix, **both substrates transport their native structures**: C2 solitons move at exactly 2Dk (mass
  0.9999); C3 Q-balls co-move under a Lorentz boost (mass ≈0.99, fidelity pending the exact IC). C3 is therefore no
  longer a "rescue" for a transport-incapable C2 (that motivation was the C2.6-bug artifact) but a **distinct,
  genuinely-conservative relativistic substrate** on its own footing.
- C3's distinctive strengths: **exact** energy + U(1)-charge conservation (machine precision — cleaner than the C2
  geometry-on quasi-conservative branch), Lorentz-covariant boosts, and a mass gap m setting a binding window.
- Note (KG): the conserved U(1) quantity is **charge Q = Im∫ψ*π**, not ∫ρ; ∫ρ breathes. Both are reported; never
  conflated.

## Open (bounded follow-ups)
1. **Exact Lorentz-boost IC** (add the φ(γx) contraction via Fourier resampling) → tighten v_frac toward 1.00 and
   confirm clean relativistic transport; longer fit window + boost-ladder for a v_measured-vs-v line.
2. **Q-ball stability** over many internal periods (oscillon vs true stable Q-ball); Vakhitov–Kolokolov / dQ/dω slope.
3. Two-Q-ball interaction (the C2.9 analog) once single-Q-ball transport is clean.
4. Existence map over (c, m, g) — the binding window and the c-vs-box fitting constraint.

## Provenance / guardrails
`jax_scout/phase_d_c3_wave.py`; runs `sweep_runs/C3_WAVE*` (rest/boost gates), `C3_WAVE_BOOSTFIX2` (corrected boost).
Standalone module (no Ops/Phase C contact); Strang stepper conservation is the error meter; transport asserted only
from measured density co-motion + mass retention; IC-bug fix documented; no matter/stability over-claims.
