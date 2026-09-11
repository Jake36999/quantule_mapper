# Phase D / C3 — Collision Ladder: Results (in-phase + anti-phase channels)

**Result: C3 Q-ball collisions form a relational phase diagram, but CAPTURE is the generic outcome across almost the
entire (relative-phase × speed) plane. Transmission is a NARROW feature confined to Δφ ≈ π (exact anti-phase) and
v ≲ 0.45–0.6c — enabled by the destructive-interference node the two anti-phase fields force at the collision
midplane, which forbids merging. Even Δφ=3π/4 and 7π/8 (statically repulsive, close to anti-phase) still CAPTURE; the
static attract→repel force (crossover Δφ=π/2, from C2.9/C3) does NOT map onto the collision capture/transmit boundary.**
So the substrate supports transmission, but only through the anti-phase node channel; a head-on collision otherwise
binds, its kinetic energy overwhelming the static repulsion. Energy and U(1) charge conserved to machine precision
throughout (dE_rel ≤ 6.6e-6) — capture and transmission results alike are trustworthy. C3 collision results only —
**not** a general theory claim. VK-stable branch (a=0.8, s=−0.5, f=−0.1, c²=0.3, w=0.964), head-on, separation 10,
L=20, N=80, dt=0.001.

## Collision phase diagram (outcome vs relative phase × impact speed)
| v/c | Δφ=0 | Δφ=π/2 | Δφ=3π/4 | Δφ=7π/8 | Δφ=π |
|---|---|---|---|---|---|
| 0.15 | CAPTURE | — | — | — | **PASS_THROUGH** (bounce, elast 51%) |
| 0.30 | CAPTURE | CAPTURE | CAPTURE | CAPTURE | **PASS_THROUGH** (elast **91%**, rad 0.07) |
| 0.45 | CAPTURE | CAPTURE | CAPTURE | CAPTURE | **PASS_THROUGH** (elast 50%, rad 0.08) |
| 0.60 | CAPTURE | CAPTURE | CAPTURE | — | CAPTURE |
| 0.75 | CAPTURE | — | — | — | CAPTURE |
Off-anti-phase captures are *violent* (radiation 0.42–0.54); the Δφ=π transmissions are *clean* (radiation 0.02–0.08).
**The only transmission cells are Δφ=π at v ≤ 0.45c.** The transmission band in Δφ is narrow (7π/8 already captures) —
it is a node feature at exact anti-phase, not a broad repulsive region.

## In-phase (attractive) channel — capture-dominated at all speeds

## The ladder
| v/c | outcome | sep_min | sep_end | radiation frac | mass ret | dE_rel max | dQ_rel max |
|---|---|---|---|---|---|---|---|
| 0.15 | CAPTURE | — | (merged) | — | — | — | — |
| 0.30 | CAPTURE | — | (merged) | — | — | — | — |
| **0.45** | **CAPTURE** | 0.11 | 0.23 | 0.60 | 0.985 | 2.3e-7 | 1.8e-8 |
| **0.60** | **CAPTURE** | 0.02 | 0.32 | 0.79 | 0.987 | 4.3e-7 | 4.6e-8 |
| **0.75** | **CAPTURE** | 0.03 | 0.28 | 0.91 | 0.995 | 5.6e-6 | 9.2e-7 |
(0.15c/0.30c from `docs/PHASE_D_C3_TWOQBALL_RESULTS.md`.) In every case the cores reach sep_min≈0 (full overlap) and
end at sep≈0.2–0.3 ≪ the 6-unit re-separation threshold — they **merge and stay bound**. No re-separation at any
speed ⇒ the identity-ambiguity flag (pass-through vs rebound) is moot: nothing transmits.

## What changes with speed: radiation, not the outcome
- **Radiation increases monotonically with v/c: 0.60 → 0.79 → 0.91.** At higher impact speed the collision is more
  violent and sheds more density into a radiation halo around the merged remnant (mass outside the core windows).
  The *outcome* stays CAPTURE, but the captured object is increasingly a small dense remnant embedded in a large
  radiation shell — capture trending toward disruption, not a clean binding.
- **Total |ψ|² mass is conserved** (0.985–0.995; the small deficit is dealiasing), and **E, Q are conserved to
  machine precision** (dE_rel ≤ 5.6e-6, dQ_rel ≤ 9.2e-7 even in the most violent 0.75c overlap). The Strang stepper
  holds; the result is not a numerical artifact.

## Anti-phase (repulsive) channel — transmission below a speed threshold
| v/c | outcome | sep_min | sep_end | vout/2v_in (elasticity) | radiation | dE_rel max |
|---|---|---|---|---|---|---|
| 0.15 | PASS_THROUGH (gentle bounce) | 5.71 | 6.88 | 0.51 | 0.02 | 1.0e-7 |
| 0.30 | **PASS_THROUGH** | 0.51 | 9.45 | **0.91** | 0.07 | 2.2e-7 |
| 0.45 | PASS_THROUGH | 0.01 | 8.54 | 0.50 | 0.08 | 1.2e-7 |
| 0.60 | **CAPTURE** | 0.05 | 5.61 | — | 0.05 | 7.0e-7 |
| 0.75 | **CAPTURE** | 0.00 | 0.06 | — | 0.10 | 6.6e-6 |
- **Transmission is real and cleanest at moderate speed.** At 0.30c the cores fully approach, re-separate to 9.45,
  and emerge with **91% of the incoming closing speed** and only 7% radiation — a near-elastic pass-through. At 0.45c
  they overlap completely (sep_min 0.01) and still re-separate (8.54) but keep only ~50% of the speed (more
  inelastic). At 0.15c it is a gentle repulsive bounce (they barely reach sep 5.7, zero radiation, re-separate).
- **The transmission window closes at high speed.** At 0.60c and 0.75c anti-phase collisions **CAPTURE** — the
  overlap is violent enough that the phase distinction (destructive interference that normally repels) is scrambled,
  and the pair merges like the in-phase case. So the repulsive protection against merging holds only below
  v ≈ 0.5–0.6c.
- **Identity ambiguity:** for symmetric anti-phase, transmission vs reflection is again undefinable; "PASS_THROUGH"
  here means *the two coherent cores re-separate* (whether they passed through or bounced is a labelling convention).
  The physically robust statement is transmission-or-bounce with the measured elasticity.

## Interpretation (bounded)
**Capture is the generic collision outcome; transmission is a narrow anti-phase node feature.** Three precise
statements from the phase diagram:
1. **Almost everywhere the cores CAPTURE** — Δφ ∈ {0, π/2, 3π/4, 7π/8} all merge at every tested speed, and Δφ=π
   captures too above ~0.5–0.6c. A head-on collision generically binds: the incoming kinetic energy overwhelms the
   static repulsion.
2. **Transmission occurs only at Δφ ≈ π and v ≲ 0.45c.** At exact anti-phase the two fields destructively interfere
   to a forced zero (a node) at the collision midplane; you cannot merge through an enforced zero, so the pair
   bounces/transmits (cleanest, ~91% elastic, at 0.30c). Detuning the phase even to 7π/8 spoils the node and the pair
   captures — so the transmission band is *narrow* in Δφ (a node feature, not a repulsive region) and bounded in v
   (at high speed the violent overlap scrambles the node and it captures).
3. **The static force law does NOT govern collisions.** The static attract/repel crossover is at Δφ=π/2 (C2.9/C3),
   but statically-repulsive collisions (3π/4, 7π/8) still capture. Static pairwise force and dynamic collision
   outcome are different physics; only the exact-anti-phase node survives into the collision.
Coherent IRER-family structures are therefore predominantly *binding* under head-on collision, with a narrow,
node-protected transmission channel at exact anti-phase — a relational phase diagram, not a single "sticky" or
"elastic" law.

## Honest caveats
- **Radiation metric** uses fixed W_WIN=3 windows around the tracked centroids; as a merged object broadens, some of
  its own mass is counted as "radiation," so the fraction is an upper-ish estimate. Outcomes (re-separation vs merge)
  are robust regardless.
- **Identity ambiguity** (transmission vs reflection for identical cores) is flagged throughout; the robust
  observables are re-separation, outgoing elasticity, and radiation — not a literal pass/rebound label.
- **Elasticity** is estimated from the outgoing separation slope over the clean-separated frames; the near-elastic
  0.30c anti-phase point (91%) is the cleanest, the 0.45c (50%) is more inelastic.
- **Extremes:** the in-phase 0.75c capture (rad 0.91) is borderline CAPTURE/DISRUPT (bound remnant + heavy radiation);
  a genuinely clean high-speed pass-through would need v→c (causal-limit / CFL territory), untested.

## Open (bounded follow-ups, none started)
1. **Off-phase (Δφ=π/2, 3π/4) collisions** — map the transmit→capture boundary in the full phase×speed plane.
2. **Asymmetric-velocity collisions** + the momentum-based elasticity observable (C2.9 hardening) to quantify the
   captured-vs-radiated energy budget cleanly.
3. **Long-time fate** of a captured remnant (stable oscillating "Q-ball molecule" vs slow decay) — well-posed given
   the machine-clean conservation.

## Provenance / guardrails
`jax_scout/phase_d_c3_collision_ladder.py`; run `sweep_runs/C3_COLLISION_LADDER_FULL` (+ smoke `C3_LADDER_SMOKE`).
Telemetry: total E + U(1) charge conservation, |ψ|² mass retention, windowed centroid separation, sep_min +
re-separation test, radiation-leakage estimate, identity-ambiguity flag. Standalone C3 module; no production/Phase C
change; no gravity work; **C3 higher-speed collision results only, not a general theory claim.**

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `e53c31c` (2026-07-10) — *Phase D C3 higher-speed collision ladder: capture-dominated up to 0.75c (no pass*
**Revised since:** 11 commit(s), most recently `caf61af` (2026-09-10)

**Harness code changed since it was written:** 6 commit(s) to `jax_scout/`.
  - `caf61af` 2026-09-10 — Build HUD items 1+2: snapshot writer and universal field renderer
  - `06edae5` 2026-08-27 — Fix run provenance: 30% -> 86% of runs now dated
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine
  - *…and 1 more.*

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
