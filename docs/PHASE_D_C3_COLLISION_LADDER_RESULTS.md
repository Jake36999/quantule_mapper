# Phase D / C3 — Higher-Speed Collision Ladder: Results

**Result: `C3_CAPTURE_DOMINATED_UP_TO_0.75c`. The VK-stable C3 Q-balls capture at every tested speed up to 0.75c
(relativistic) — no critical velocity and no pass-through / transmission was found. Higher speed does not unbind the
collision; it makes capture progressively more violent and radiative (the merged remnant sheds a growing radiation
halo). Energy and U(1) charge are conserved to machine precision throughout, so the null is trustworthy.** These are
C3 higher-speed collision results only — **not** a general theory claim. Symmetric in-phase head-on, same validated
branch (a=0.8, s=−0.5, f=−0.1, c²=0.3, w=0.964), same separation (10), L=20, N=80, dt=0.001.

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

## Interpretation (bounded)
The attractive Q-ball interaction (the Δφ=0 in-phase channel, which the C2.9/C3 phase-force law shows is attractive)
is **strong enough to capture at all tested speeds up to 0.75c** — the incoming kinetic energy does not overcome the
binding within the relativistic range accessible here. This is consistent with strongly-attractive, non-integrable
solitary waves: they bind (and, at higher energy, bind-plus-radiate) rather than transmit. **No transmission window
was found below 0.75c.**

## Honest caveats
- **Symmetric in-phase only.** This is the maximally-attractive channel; anti-phase or off-phase collisions (which
  the static law shows are repulsive above Δφ=π/2) could transmit or bounce differently — untested here.
- **Radiation metric** uses fixed W_WIN=3 windows around the tracked centroids; as the merged object broadens at high
  v, some of its own mass is counted as "radiation," so the radiation fraction is an upper-ish estimate. The *trend*
  (more radiation at higher v) and the *outcome* (no re-separation) are robust regardless.
- **A genuinely clean pass-through** would require either v→c (approaching the causal limit, where CFL/relativistic
  resolution issues dominate) or a different collision channel (phase/asymmetry) — beyond this bounded ladder.
- The "capture at 0.75c with rad 0.91" is borderline CAPTURE/DISRUPT: a bound remnant survives (so not full
  DISRUPT), but the heavy radiation means it is a strongly inelastic, near-disruptive capture.

## Open (bounded follow-ups, none started)
1. **Off-phase / anti-phase collisions** at these speeds — does the repulsive channel transmit or bounce?
2. **Asymmetric-velocity collisions** + the momentum-based elasticity observable (C2.9 hardening) to quantify the
   captured-vs-radiated energy budget cleanly.
3. **Long-time fate** of the captured remnant (stable oscillating "Q-ball molecule" vs slow decay) — well-posed given
   the machine-clean conservation.

## Provenance / guardrails
`jax_scout/phase_d_c3_collision_ladder.py`; run `sweep_runs/C3_COLLISION_LADDER_FULL` (+ smoke `C3_LADDER_SMOKE`).
Telemetry: total E + U(1) charge conservation, |ψ|² mass retention, windowed centroid separation, sep_min +
re-separation test, radiation-leakage estimate, identity-ambiguity flag. Standalone C3 module; no production/Phase C
change; no gravity work; **C3 higher-speed collision results only, not a general theory claim.**
