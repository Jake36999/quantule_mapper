# 07 · Instrument-Integrity Ledger

Three measurement bugs were found and corrected. This folder frames them as a **methodological achievement** — the
model is credible *because* it is self-correcting, not despite the bugs. Each was caught the same way: a result
contradicted a known identity/symmetry, and that contradiction was chased rather than shipped.

## The three bugs
| bug | symptom (what it looked like) | root cause | how it was caught | fix | manifest |
|---|---|---|---|---|---|
| **C2.6 geometry-off** | "conservative pinning / flow-through", universal drag μ≈0.036 | log-tanh soft-clip squashes Ω²=1→~151 → `a_coupling=0` gave D_eff=D/151 (99.3% of dispersion cancelled) | a **linear packet failed the Galilean identity v=2Dk** (moved at ~1/151) | `Ops.geom_fac` (default bit-exact) + `param_geom_off`; Codex RK4-re-audited | `EV-C26-BUG` |
| **C2.8b tracker** | two-node elasticity e=3.21 | projected-density peak-tracking breaks when cores overlap/breathe/merge | **e>1 violates energy conservation** | momentum-density observable v=2D·∫Im(ψ*∂ψ)/∫ρ (C2.9) | `EV-C28B-BUG` |
| **C3 boost-IC** | Q-ball density barely moves (v_frac≈0.04) | naive kick π=−v∇φ−iωφ lacks the carrier phase that *is* the momentum | **v_frac constant, independent of v** (a tell) | ψ₀=φ(γx)·e^{ikx}, k=γωv/c² | `EV-C3-BUG` |

## The retractions they triggered (preserved, not deleted)
- C2.6 → C2.1b, C2.2b, C2.3b, C2.4 ("conservative pinning" family); "arc closed negative".
- C2.8b → the C2.8 "n=4 survives" / any C2.8b elasticity number.
- C3 boost-IC → the first-pass "density barely moves" reading.
- Later refinement (not a bug, but a correction): "the repulsive channel transmits" → narrowed to the exact
  anti-phase node (off-phase 3π/4, 7π/8 capture).

## The standing rule that came out of it
> Every transport/interaction claim is gated on **(1) a correct boost initial condition, (2) a robust (non-peak-
> tracking) observable, and (3) conservation telemetry.** A null that contradicts a symmetry identity is treated as an
> instrument fault until proven physical.

This rule is encoded in the architecture (Report 4 §3.6) and is why the strongest results (v=2Dk, dE/E~1e-13, VK
slope, the collision phase diagram) are trustworthy.

## Docs
`docs/PHASE_D_C2_6_GEOMETRY_OFF_BUG_REPORT.md`, `PHASE_D_C2_8_TWONODE_RESULTS.md` (C2.8b),
`PHASE_D_C3_WAVE_KINETIC_RESULTS.md` (boost-IC); proof ledger Report 3 §1–§3; catalog §10.
