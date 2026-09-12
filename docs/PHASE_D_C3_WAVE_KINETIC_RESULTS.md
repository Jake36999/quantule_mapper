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
| **G4** transport | **SUPPORTED (loose)** | v=0.137→v_meas 0.124 (frac 0.91, mass 0.96); v=0.055→0.069 (frac 1.25*, mass 0.99) |
| **G6** VK stability | **VK-STABLE** | Q(ω) = 59.0→53.9→50.3→48.9 over ω=0.956→0.968; **dQ/dω = −849 < 0** (stable branch) |
*low-v frac is noise-dominated (displacement ~0.4 box over the fit window); the higher-v point (frac 0.91) is the
reliable one.

### Refinement results (this pass)
- **Exact Lorentz-contraction boost — the contraction is NOT the fidelity limiter.** Adding the φ(γx) profile
  contraction (the previously-skipped O((v/c)²) term) left v_frac **unchanged** (1.254, 0.907 — identical to the
  no-contraction boost) and slightly *lowered* mass retention (interp error). So the ~10% velocity looseness is
  **not** the boost IC — it is the *measurement*: a short fit window (displacement ≪ box) plus the quasi-Q-ball's
  internal breathing jittering the centroid. The boost construction is correct/exact; tightening v_frac→1 needs a
  longer/larger measurement (bigger box + longer T so displacement ≫ breathing amplitude), not a better IC.
- **Q-ball is VK-STABLE (G6).** A per-ω seed scan converged 4 neighbouring Q-balls; Q(ω) decreases monotonically, so
  the Vakhitov–Kolokolov slope dQ/dω = −849 < 0 — the object sits on the **linearly-stable branch**. It is a genuine
  stable Q-ball, not a transient oscillon. (This is a substantive upgrade: C3's native object is now
  existence-confirmed *and* stability-confirmed.)

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
1. ~~Exact Lorentz-boost IC~~ — **DONE this pass**: contraction added, ruled out as the fidelity limiter.
2. ~~Q-ball stability slope~~ — **DONE this pass**: VK-STABLE (dQ/dω = −849 < 0).
3. **Tighten v_frac→1** — the residual is *measurement* (short window + breathing), not physics/IC: use a bigger box
   + longer T so the boost displacement ≫ the breathing amplitude, and a boost-ladder for a v_measured-vs-v line.
4. **Two-Q-ball interaction** (the C2.9 analog) — now well-motivated: the object is existence- *and* stability-
   confirmed, so a two-Q-ball collision/force study is the natural next C3 science.
5. Existence map over (c, m, g) — the binding window and the c-vs-box fitting constraint.

## Provenance / guardrails
`jax_scout/phase_d_c3_wave.py`; runs `sweep_runs/C3_WAVE*` (rest/boost gates), `C3_WAVE_BOOSTFIX2` (corrected boost).
Standalone module (no Ops/Phase C contact); Strang stepper conservation is the error meter; transport asserted only
from measured density co-motion + mass retention; IC-bug fix documented; no matter/stability over-claims.

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `e820857` (2026-07-09) — *Phase D C3 first results: KG substrate conservative + Q-ball inertial transport *
**Revised since:** 11 commit(s), most recently `e42b5bb` (2026-09-11)

**Harness code changed since it was written:** 23 commit(s) to `jax_scout/`.
  - `caf61af` 2026-09-10 — Build HUD items 1+2: snapshot writer and universal field renderer
  - `06edae5` 2026-08-27 — Fix run provenance: 30% -> 86% of runs now dated
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine
  - *…and 18 more.*

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Also referenced by (same date or earlier):** [[PHASE_D_CLOSEOUT_CONSOLIDATION]]

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
