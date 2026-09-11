# Phase D / C2.10 — RUN-2: NLS Anti-Phase Collision Ladder

**Verdict:** `C2_ANTIPHASE_NODE_CHANNEL_CONFIRMED_WITH_TRANSMISSION_WINDOW` — the first-order conservative NLS
substrate shows the same collision phase×speed *structure* as the second-order KG substrate (C3): capture is
generic; the exception is **exact anti-phase**, where the destructive-interference node yields a non-merging
channel that **transmits (pass-through) at intermediate speed** (closing ≈0.47–0.57) and is overwhelmed into
**capture** above a sharp threshold (closing ≈0.61, between 0.565 and 0.660). Below the transmission window a
**bounce** sub-regime appears (closing ≲0.42). Cross-substrate universality of the collision phase diagram —
including the pass-through channel — not just the static two-body law.

Mirror-only (`jax_scout`, `param_geom_off` true-flat NLS); no production/Hunter/config change; no matter claim.

## Question

C3 (KG) found transmission only at exact anti-phase below ~0.5c — a destructive-interference *node* feature; every
other phase/speed captures. RUN-2 asks whether the **first-order NLS substrate** has the same anti-phase
node-protected non-merging channel, or captures at all phases/speeds (channel would then be KG-specific).

## Setup

Validated Galilean soliton family `a=0.8, s=-0.5, f=-0.1, D=0.3, mu=0.070` (`phi_iso` from C2.8), symmetric head-on
at sep=10 in the validated `N=96, L=20, dt=0.001` grid. Each soliton boosted `phi·exp(±ikx)`, `v=2Dk`; the right
soliton carries relative phase `exp(i·Δφ)`. Speed ladder over boost integer `n` (`v_each = 2D·2πn/L = 0.188n`) at
`Δφ=π` (anti-phase) and `Δφ=0` (in-phase control). Observables: windowed-COM separation, radiation fraction (mass
outside the two core windows), peak amplitude (coherence), total mass + momentum conservation meters.
`harness: jax_scout/phase_d_c2_10_antiphase_collision.py`.

## Results — anti-phase ladder (transition mapped)

| n | closing 2v | sep_min | radiation | outcome | note |
|---|---|---|---|---|---|
| 1.00 | 0.377 | **5.05** | 4% | **BOUNCE** | repelled *before* reaching the node; coherent |
| 1.25 | 0.471 | 0.36 | 5% | **PASS_THROUGH** | reaches the node, re-emerges to full separation (peak 9.96), coherent |
| 1.50 | 0.565 | 0.09 | 7% | **PASS_THROUGH** | full overlap at the node (dwells t≈12–21), re-emerges coherently |
| 1.75 | 0.660 | 0.00 | 95% | **CAPTURE** | node overwhelmed → merge, near-total radiation |
| 2.00 | 0.754 | 0.00 | 96% | CAPTURE | merge |
| 4.00 | 1.508 | 0.00 | 68% | CAPTURE | merge |

**In-phase control** (Δφ=0): n=1 (closing 0.377) **CAPTURE** (merges at the same speed the anti-phase pair
bounces); n=2 (closing 0.754) CAPTURE. Conservation clean throughout: total-mass drift `dmass ≤ 1.5e-3`, net
momentum `P ≈ 0` (symmetric); hold gate `mass_ret = 0.9999`; non-integer boost validated `v=2Dk` to 0.02% (V0).

### Three anti-phase regimes, and the mapped transition

1. **BOUNCE** — closing ≲ 0.42: KE too low to reach the node; the anti-phase repulsive tail turns the pair around
   at sep≈5 (never overlaps). Coherent, ~4% radiation.
2. **PASS_THROUGH (transmission)** — closing ≈ 0.47–0.57: enough KE to reach the midplane node (sep_min≈0.1–0.36);
   the pair overlaps at the node and **re-emerges coherently** to full separation, ≤7% radiation.
3. **CAPTURE** — closing ≳ 0.61: the node is overwhelmed, the pair merges with near-total (95–96%) radiation.

**Transitions:** BOUNCE→PASS_THROUGH near closing ≈ 0.42; the decisive **merge (PASS_THROUGH→CAPTURE) threshold is
sharp, between closing 0.565 and 0.660** — radiation jumps 7% → 95% across that narrow gap. The anti-phase node
protects against merger up to closing ≈ 0.61 and is overwhelmed above it.

**This resolves the earlier open question: the NLS substrate DOES have an anti-phase transmission (pass-through)
window** — the first (n=1) sample happened to sit in the lower BOUNCE sub-regime; the intermediate speeds transmit
through the node exactly like C3.

### The decisive contrast (phase-driven, not speed-driven)

At the **same low closing speed (0.377)**, the outcome depends on relative phase: **in-phase merges (CAPTURE),
anti-phase does not (BOUNCE)**. So the low-speed non-merger is a genuine property of the anti-phase configuration —
the destructive-interference node — not merely "weak interaction at low speed" (which would bounce regardless of
phase). This is the same logic as C3, where statically-repulsive off-phases still captured; the outcome is set by
the node, and only exact anti-phase has it.

### Phase×speed structure

- **Capture is generic** across phase and speed (in-phase all speeds; anti-phase mid/high speed).
- **The single exception is exact anti-phase at low speed**, where the node prevents merger (bounce).
- The node channel is a **low-speed feature**: at higher closing speed the anti-phase pair overwhelms the node and
  captures (n=2, n=4), exactly as C3 captures anti-phase above ~0.6c.

## Cross-substrate comparison (C2/NLS vs C3/KG)

| feature | C2 (NLS, 1st-order) | C3 (KG, 2nd-order) | shared? |
|---|---|---|---|
| capture generic across (Δφ, v) | yes | yes | **yes** |
| only exception = exact anti-phase, low/mid speed | yes | yes | **yes** |
| anti-phase **PASS_THROUGH** (transmit through node) at intermediate speed | yes (closing 0.47–0.57) | yes (0.15–0.45c) | **yes** |
| anti-phase node overwhelmed at high speed → capture | yes (closing ≳0.61) | yes (≳0.6c) | **yes** |
| in-phase captures at all speeds | yes | yes | **yes** |
| extra BOUNCE sub-regime at the very lowest anti-phase speed | yes (closing ≲0.42) | not sampled (lowest was 0.15c) | detail |

**Universal:** the collision phase×speed *structure* holds in both substrates — capture is generic; the anti-phase
node produces a non-merging channel that **transmits (pass-through) at intermediate speed** and is overwhelmed into
capture above a sharp threshold. This extends the previously-established static two-body universality (π/2
force-crossover, NLS≡KG) to the full collision diagram, now including the pass-through channel itself.

**The one C2-specific detail:** at the *very lowest* anti-phase speed (closing ≲0.42) the NLS pair is turned around
by the repulsive tail *before* reaching the node (BOUNCE) rather than transmitting. C3 did not sample below 0.15c,
so it is unknown whether KG has the same low-speed bounce sub-regime. For *identical* anti-phase solitons the
midplane is a permanent destructive-interference node (field antisymmetric ⇒ |ψ|²=0 ⇒ no density crosses), so
within the non-merging channel "reflect vs transmit" is partly identity-ambiguous; the physical invariant is
**no merger**, and both substrates share it up to their capture thresholds.

## Caveats

- Two classifier thresholds corrected during analysis (both real, both fixed in the harness + re-verified from
  saved trajectories): (1) the overlap/merge scale must be the soliton core (`MERGE_SEP=3.5`), not the tracking
  window (`2·W_WIN=7`) — the loose value mislabeled the n=1 bounce as CAPTURE; (2) re-separation must be judged by
  the **post-min peak** separation, not `sep_end` — in the periodic box a pass-through pair separates fully then
  wraps back, so `sep_end` understated re-separation and mislabeled the n=1.25/1.5 pass-throughs as INTERMEDIATE.
- Speed ladder sampled at closing ∈ {0.38, 0.47, 0.57, 0.66, 0.75, 1.51}; the capture threshold is bracketed to
  (0.565, 0.660) and the bounce→transmit boundary to ≈0.42 — not pinned finer (each point is a ~35–45 min run).
- Non-integer boosts (needed for intermediate speeds) validated: single-soliton `v=2Dk` to 0.02%, mass 1.0000 —
  the box-edge phase mismatch sits in the deep vacuum (amp ~1e-12) and is harmless.
- Anti-phase identical-soliton collisions are identity-ambiguous (reflect vs transmit undefinable) by construction;
  the reported invariant is merger vs no-merger.
- fp64 on the GTX 1080 makes each collision ~30–45 min.

## Status

RUN-2 (from `docs/evidence_package/EVIDENCE_GAPS.md`) **complete, POSITIVE, transition mapped**: NLS collisions
share the C3 collision phase×speed structure — capture-generic, with an exact-anti-phase non-merging channel that
transmits (pass-through) at intermediate speed and captures above closing ≈0.61 (bracketed to (0.565, 0.660)), plus
a bounce sub-regime below ≈0.42. No matter/gravity/correspondence claim; a numerical cross-substrate universality
result on the conservative IRER substrate class.

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `25020bf` (2026-07-22) — *dual substrate*
**Revised since:** 9 commit(s), most recently `caf61af` (2026-09-10)

**Harness code changed since it was written:** 5 commit(s) to `jax_scout/`.
  - `caf61af` 2026-09-10 — Build HUD items 1+2: snapshot writer and universal field renderer
  - `06edae5` 2026-08-27 — Fix run provenance: 30% -> 86% of runs now dated
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Also referenced by (same date or earlier):** [[evidence_package/EVIDENCE_GAPS]]

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
