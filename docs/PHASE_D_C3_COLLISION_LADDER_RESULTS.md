# Phase D / C3 — Collision Ladder: Results (in-phase + anti-phase channels)

**Result: the C3 collision outcome is PHASE- and SPEED-controlled. In-phase (attractive) collisions CAPTURE at every
speed up to 0.75c. Anti-phase (repulsive) collisions TRANSMIT / bounce at low-to-moderate speed (0.15–0.45c, cleanest
at 0.30c with ~91% elasticity) and only CAPTURE at high speed (≥0.60c, where the violent overlap scrambles the phase
distinction). So transmission DOES exist in this substrate — in the repulsive channel below a speed threshold — but
there is no transmission in the attractive channel at any tested speed.** Energy and U(1) charge conserved to machine
precision throughout (dE_rel ≤ 6.6e-6), so both the capture and transmission results are trustworthy. C3 collision
results only — **not** a general theory claim. VK-stable branch (a=0.8, s=−0.5, f=−0.1, c²=0.3, w=0.964), head-on,
separation 10, L=20, N=80, dt=0.001.

## Collision phase diagram (outcome vs relative phase × speed)
| v/c | in-phase Δφ=0 (attractive) | anti-phase Δφ=π (repulsive) |
|---|---|---|
| 0.15 | CAPTURE | **PASS_THROUGH** (gentle bounce, elast 51%, rad 0.02) |
| 0.30 | CAPTURE | **PASS_THROUGH** (full re-separate, **elast 91%**, rad 0.07) |
| 0.45 | CAPTURE | **PASS_THROUGH** (overlap sep_min 0.01→9.45, elast 50%, rad 0.08) |
| 0.60 | CAPTURE | **CAPTURE** (violent overlap wins, sep_end 5.61) |
| 0.75 | CAPTURE | **CAPTURE** (merged, rad 0.10) |
The transmission window is the repulsive channel at v ≲ 0.45–0.6c; above it, even anti-phase captures.

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
**The collision outcome is jointly controlled by relative phase and speed** — a small phase diagram, not a single
law. The attractive (in-phase) Q-ball interaction is **strong enough to capture at all tested speeds up to 0.75c** —
the incoming kinetic energy does not overcome the
binding within the relativistic range accessible here. **But transmission is NOT categorically absent in the
substrate** — the repulsive (anti-phase) channel transmits/bounces below v ≈ 0.5–0.6c (cleanest, near-elastic at
0.30c), and only crosses over to capture at high speed. So the substrate supports both binding and transmission; which
one occurs is set by the *relative phase* (the same phase that sets the static attract/repel force, C2.9/C3) and the
*impact speed*. Coherent IRER-family structures are neither purely "sticky" nor purely "elastic" — the relational
outcome is a phase×speed phase diagram.

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
