# IRER Mathematical & Numerical Proof Ledger

**Report 3 of 5.** Identities, proof obligations, and numerical-theory sanity checks for Quantule Mapper. **Framing:
these prove what the numerical model is doing, not what nature is doing.** A "numerical identity CONFIRMED" means the
discrete integrator reproduces an analytic identity to stated precision — evidence the code is faithful, not evidence
IRER is physically true. Both *passed* and *failed* identities are recorded: the failed ones are how the project
found its three instrument bugs, and the standing methodological rule (contradicted symmetry ⇒ instrument fault until
proven physical) came from them.

Notation: ψ complex field, ρ=|ψ|², D diffusion/dispersion coefficient, k wavenumber, ω internal frequency, c wave
speed (KG), m mass gap (KG), Ω²(ρ) conformal factor. dV = (L/N)³.

---

## §1. Galilean transport identity (C2 NLS) — CONFIRMED (and it exposed a bug)
**Claim.** For the NLS `i ψ_t = −D∇²ψ + N(ρ)ψ` (N real, translation- and Galilean-invariant), boosting a stationary
soliton by ψ→ψ·e^{ikx} translates it at the group velocity **v = 2Dk**, shape-preserving.
**Sketch.** The Galilean transform ψ'(x,t)=ψ(x−vt,t)·e^{i(kx−ωt)} solves the same NLS with v=2Dk, ω=Dk² (the
nonlinearity depends only on ρ=|ψ|², invariant under the phase). So a boosted soliton is an exact moving solution.
**Numerical result (fixed substrate, C2.7).** v/2Dk = **0.9999**, mass retention **0.9999** at N=96. `CONFIRMED` as a
numerical identity in the corrected benchmark substrate.
**Role as a bug detector.** A *linear* packet must also obey v=2Dk. On the pre-fix "geometry-off" path it moved at
~1/151 of the prediction → a **contradiction of a known identity** → instrument fault (see §2).

## §2. Geometry-off contract — FAILED (old) → FIXED (D_eff = D/151 bug)
**Obligation.** `param_geom_off=True` (or nominally `a_coupling=0`) must yield a *genuinely flat* substrate:
the covariant correction `D·(lap_cov − lap_flat)` must vanish identically.
**Fault (C2.6).** The conformal soft-clip `_soft_clip_log_with_derivative(Ω², 1e-9, 1e6, β=3)` is a log-space tanh
*squash*, not a boundary clamp. Its window centre is log-mean(1e-9,1e6) ≈ e^{−3.45}; for Ω²=1, z=β(ln1−centre)/half
gives soft(1) = exp(centre + half·tanh z) ≈ **151**. So at `a_coupling=0`, Ω²≈151, `lap_cov ≈ lap_flat/151`, and the
correction cancels ~99.3% of the kinetic term → the substrate silently integrates **i ψ_t = −(D/151)∇²ψ − g(ρ)ψ**.
**Consequence.** The universal "drag" μ ≈ 0.036 = **2·D_eff = 2·(D/151)** exactly (the Galilean velocity of the
broken substrate). Every "conservative pinning / flow-through" verdict (C2.1b, C2.2b, C2.3b, C2.4) was an artifact.
**Fix + audit.** `Ops.geom_fac` multiplier (default 1.0, bitwise-identical to baseline; `param_geom_off`→0.0 = true
flat). Verified: one-step phase exact to 7 digits; linear packet at 2Dk; true soliton at 2Dk (mass 0.9999).
Independently re-audited by Codex (ETDRK4 ≡ RK4; D_eff ratio 0.006511 ≈ 1/151). `FIXED`.

## §3. Momentum-density velocity observable — DERIVED (supersedes peak-tracking)
**Claim.** A localized structure's centre-of-mass velocity is **v = 2D·(∫Im ψ*∂ₓψ)/(∫ρ)**.
**Proof.** For `i ψ_t = −D∇²ψ + N(ρ)ψ` with N real: ρ_t = 2Re(ψ*ψ_t) = 2Re(ψ*·i(D∇²ψ − Nψ)) = −2D·Im(ψ*∇²ψ) =
−∇·J with **J = 2D·Im(ψ*∇ψ)** (the N-term is pure-imaginary ⇒ zero contribution). Then
d⟨x⟩/dt = ∫x ρ_t dV = −∫x ∇·J dV = ∫J_x dV = 2D·∫Im(ψ*∂ₓψ) dV; dividing by M=∫ρ gives v. For a boost e^{ikx},
∂ₓ→ik ⇒ ∫Im(ψ*ikψ)=k∫ρ ⇒ v=2Dk, consistent with §1. `DERIVED`.
**Why it supersedes peak-tracking.** It is an integral over a window, robust when cores overlap, breathe, or merge —
where projected-density peak-finding failed (unphysical elasticity e=3.21; false-repel static readings, C2.8b).

## §4. Static two-body phase-force law — CONFIRMED (crossover at Δφ=π/2)
**Claim.** Two identical coherent structures with relative phase Δφ, at separation d, feel an interaction whose sign
∝ cos(Δφ): **attractive for Δφ<π/2, repulsive for Δφ>π/2, crossover at π/2.**
**Origin.** The classic overlapping-soliton interaction: the leading force between weakly-overlapping tails carries a
cos(Δφ) factor from the interference of the two fields in the overlap region (in-phase = constructive = attractive;
anti-phase = destructive = repulsive). This is standard for NLS/GL solitons and Q-balls.
**Numerical result.** C2.9 (NLS) and C3 (KG) both give attract for Δφ∈{0,π/4}, neutral at π/2, repel for {3π/4,π} —
**crossover at π/2 in both substrates** (cross-substrate universality). `CONFIRMED`.
**Important limit.** This static/pairwise law does **not** govern head-on collisions (Report 5 / C3 phase diagram):
statically-repulsive phases still capture dynamically; only the exact-anti-phase interference node survives.

## §5. Klein–Gordon conservation identities — CONFIRMED (machine-exact)
**Claim.** For `ψ_tt = c²∇²ψ − m²ψ + g(ρ)ψ`, the Noether invariants are conserved:
- Energy **E = ∫[|ψ_t|² + c²|∇ψ|² + m²ρ − G(ρ)]**, G′=g (time-translation).
- U(1) charge **Q = Im∫ψ*ψ_t** (global phase). NB the conserved charge is **Q, not ∫ρ** — ∫ρ *breathes* in KG.
- Momentum **P = −Re∫ψ_t*∇ψ** (spatial translation).
**Numerical result.** Rest Q-ball, T=6: dE/E = **9e-14**, dQ/Q = **1e-13** (Strang split-step). `CONFIRMED`.
**Consequence.** KG is the *cleaner* collision platform: capture and transmission results carry dE/E ≤ 6.6e-6 even
through violent relativistic overlap, so outcomes are physics, not numerical drift.

## §6. Vakhitov–Kolokolov stability criterion — CONFIRMED (C3 Q-ball on the stable branch)
**Claim.** A soliton/Q-ball branch is linearly stable iff **dQ/dω < 0** (VK). For a rest Q-ball, Q(ω)=ω·∫|φ|².
**Numerical result.** Per-ω Petviashvili scan: Q(ω) = 59.0→53.9→50.3→48.9 over ω=0.956→0.968, monotone decreasing ⇒
**dQ/dω = −849 < 0** ⇒ the C3 Q-ball is on the **VK-stable branch** (genuine stable soliton, not a transient
oscillon). `CONFIRMED`.

## §7. Laplace–Beltrami adjointness — FAILED-BY-DESIGN (C2 geometry is quasi-conservative)
**Obligation (if C2 geometry-on is to conserve ∫ρ).** The geometry-corrected operator must be self-adjoint under the
ordinary (flat) inner product.
**Finding (Codex adjointness audit + contract review).** The implemented `lap_cov = (lap_flat + (d−2)(∇Ω·∇ψ)/Ω)/Ω²`
is exactly the **Laplace–Beltrami operator of the conformal metric g=Ω²δ** (d=3). Δ_LB is self-adjoint w.r.t. the
**√g = Ω³** measure, *not* the flat one → the flat-inner-product adjoint test fails (mismatch 1.09e-1); the √g-weighted
test near-passes (2.15e-4, the residual being a non-divergence-form discretization). Moreover, for a *live* Ω(ρ) no
*fixed* Ω-power norm is conserved (`NO_WEIGHTED_INVARIANT_FOUND`), and the implemented term is not canonical (it omits
the metric-variation term D·Ω′(ρ)|∇ψ|²ψ of the true δH/δψ* flow).
**Verdict.** C2 geometry-on is **linear-conservative / nonlinear quasi-conservative (geometry-exchange)** — *not* a
bug, a property of the operator. Exact conservation would require the **C2′ canonical (divergence-form + metric-
variation)** branch, which is `DESIGN-ONLY` (RFC). `FAILED-BY-DESIGN` (correctly characterized).

## §8. The Ω² gravity cliff — PROVEN (why de-saturation does not unblock the ladder)
**Claim.** In the low-background regime, the conformal law Ω² = (ρ_vac/ρ)^a produces a **saturation cliff** at the
load boundary, not a graded gravitational potential — and softening the soft-clip cannot fix it.
**Argument.** A coherent load has core ρ_core ~ O(1) and simulation vacuum ρ_vac,sim → 0 (≈1e-12). Then across the
core boundary Ω² = (ρ_vac/ρ)^a spans (ρ_vac/1)^a to (ρ_vac/1e-12)^a — with a=2.31 that is ~27 orders of magnitude.
The soft-clip merely *caps* this at 1e6; removing/loosening the cap (β=0.75, wider window) exposes the true
divergence → the spatial cliff got **250× steeper** (max radial jump 54.6→13390), not graded. The cliff is the
`(ρ_vac/ρ)^a` divergence at the near-zero vacuum, **not** the cap. `PROVEN`.
**Corollary.** C2′ does not help either — it changes the conservation *form* but uses the same Ω²(ρ) factor.
A graded well requires vacuum at ρ≈ρ_vac + a load *denser* than ρ_vac + stable on that background — an unmet regime.
Hence the gravity ladder is `PAUSED`.

---

## Summary ledger
| # | identity / obligation | status |
|---|---|---|
| §1 | Galilean transport v=2Dk (NLS) | CONFIRMED (numerical, corrected substrate) |
| §2 | geometry-off flat contract | FAILED (old a_coupling=0, D/151) → FIXED (geom_fac) |
| §3 | momentum-density velocity v=2D·∫Im(ψ*∂ψ)/∫ρ | DERIVED (supersedes peak-tracking) |
| §4 | static phase-force crossover at π/2 (NLS & KG) | CONFIRMED |
| §5 | KG energy + U(1) charge conservation | CONFIRMED (dE/E~1e-13) |
| §6 | VK stability dQ/dω<0 (C3 Q-ball) | CONFIRMED (−849) |
| §7 | C2 geometry flat-adjointness | FAILED-BY-DESIGN (quasi-conservative; C2′ = the fix, design-only) |
| §8 | Ω² gravity cliff / de-saturation | PROVEN (cliff is vacuum-ρ divergence; ladder paused) |

## Open mathematical gaps
- **C2′ canonical geometry** derivation → implementation (would upgrade §7 to exact conservation).
- **A graded-geometry regime** for a gravity well (§8) — a stable-overdense-load-on-ρ_vac-background construction.
- **Q-ball velocity fidelity** (§1/§5 in KG): the ~10% looseness is measurement (breathing), not the IC; a
  continuum-limit v-vs-k line (bigger box, longer T) would tighten it to an identity.
- **Full convergence documentation** (dt/N ladders) is complete for C2.2/C2.7 but partial elsewhere.

## Cross-references
Equation↔code: Report 2. Concepts: Report 1. Design rationale (why these gates are separated from the solver hot
loop): Report 4. Detailed derivations & data: `docs/PHASE_D_C2_6_GEOMETRY_OFF_BUG_REPORT.md`,
`docs/PHASE_D_C2_2_LOSS_SOURCE_RESULTS.md`, `docs/PHASE_D_C2_CONTRACT_REVIEW.md`,
`docs/PHASE_D_C3_WAVE_KINETIC_RESULTS.md`, `docs/GRAVITY_LADDER_GEOMETRY_DECISION.md`, `docs/IRER_MATH_REFERENCE.md`.
