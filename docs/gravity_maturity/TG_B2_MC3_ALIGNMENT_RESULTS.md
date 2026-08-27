# TG-B2 MC-3 Alignment Sweep — Does the Loop-Mediated Force Track Relative Phase?

Author: Claude (primary), 2026-07-18. Local CPU run `jax_scout/gravity_TG_B2_alignment_sweep.py` (imports the frozen
static-force machinery unchanged). Run `sweep_runs/TG_B2_ALIGNMENT_20260718_000535`. Tests recovered-concept **MC-3**
(relative phase Δφ = the Payan/alignment variable). Complementary to Codex's parallel Colab cooled-secular-binding run.

## Result (quasi-static)

The A-well **loop-mediated** two-node force is **strongly alignment-modulated** (sep=3.0, sweep Δφ ∈ [0, π]):

| Δφ | `F_R_well` | reading |
|---:|---:|---|
| 0 (in-phase) | −2.34e-5 | attract (strongest) |
| π/4 | −2.26e-5 | attract |
| π/2 | −1.92e-5 | attract |
| 3π/4 | −0.72e-5 | attract (weak) |
| 7π/8 | −0.09e-5 | ~null |
| π (anti-phase) | **+0.15e-5** | **repel** |

Monotonic weakening with Δφ; **sign crossover near anti-phase (Δφ ≈ 0.9–1.0 π)**; `F_R_hill = −F_R_well` throughout
(sign-flip control intact). Force range/mean = 1.67 — the alignment dependence is large, not a small correction.

## Closer look (separation sweep + functional form, 2026-07-18)

Running the sweep across sep ∈ {2.5, 3.0, 3.5, 4.0, 5.0} refines the picture: the loop force is well-described by a
**two-component decomposition**

```
F_R_well(Δφ)  ≈  a·cos(Δφ)  +  b      (a<0, b<0;  R² 0.88 → 0.99, improving with separation)
```

| sep | crossover (Δφ/π) | a (cos / phase term) | b (DC offset) | −b/a |
|---:|---:|---:|---:|---:|
| 2.5 | 0.950 | −1.25e-5 | −1.60e-5 | −1.29 |
| 3.0 | 0.921 | −1.18e-5 | −1.46e-5 | −1.24 |
| 3.5 | 0.878 | −1.18e-5 | −1.33e-5 | −1.13 |
| 4.0 | 0.854 | −1.20e-5 | −1.19e-5 | −0.99 |
| 5.0 | 0.793 | −0.96e-5 | −0.76e-5 | −0.79 |

Two readings fall out:
1. **A DC, alignment-independent attraction `b`** — present even at Δφ=π/2 where cos=0 (the force is still attractive
   there). This is the "monopole" mediated attraction: the alignment-*independent* part.
2. **A phase-modulated term `a·cos(Δφ)`** — attractive in-phase, repulsive anti-phase: the alignment-*dependent*
   (MC-3 / Payan) part. `a` is roughly separation-independent; `b` **falls with separation faster than `a`**.
3. **Consequence:** the crossover is *not* a fixed anti-phase feature — it **moves monotonically from ~anti-phase
   (0.95π at sep 2.5) toward π/2 (0.79π at sep 5.0)** as the DC part weakens relative to the phase part, at
   cos(Δφ_c) = −b/a. In the far field the loop force would approach a pure π/2 phase law (b→0); in the near field the
   monopole attraction dominates and it stays attractive until near anti-phase.

**Speculative flag (to test, not claimed):** a mediated force that separates into a *phase-independent monopole
attraction* + a *phase-dependent cos term* is suggestively parallel to IRER's own split (gravity from RD gradients;
EM from chiral/phase oscillations). This is a quasi-static curve fit — the parallel is a hypothesis-to-test, not an
identification.

## What it says — and what it does NOT (labeling discipline)

**Candidate reading (not a confirmation):** the mediated loop force is not alignment-independent — it carries its own
**alignment→coupling law**, consistent with MC-3's claim that alignment governs coupling. This is a *potential*
predicted mode worth testing further, not a confirmed identification.

**Two honest qualifications:**
1. **The crossover is near anti-phase (~π), NOT at π/2.** The bare KS/Gordon two-body force crosses at π/2; the *loop*
   force crosses near π. So this is a **distinct law from the bare force** — MC-3's "alignment governs coupling" is
   supported in spirit, but the loop force is **not** the π/2 law and should not be equated with it. (The repulsion
   onset sits exactly where the anti-phase destructive-interference node lives — a possible cross-link to MC-4, the
   anti-phase pass-through node; flagged, not claimed.)
2. **Quasi-static.** This is the static screened-response body force; the definitive dynamical force differed from
   static by ~2.3× in magnitude at Δφ=0. So the *trend* (alignment-modulated, crossover near anti-phase) is the
   candidate result; a dynamical F_R(Δφ) sweep would be needed to confirm it survives the dynamics.

**Mechanism note:** the A-well *depth* is nearly Δφ-independent (`A_well_min` ≈ 0.99994–0.99996 across the sweep); the
alignment dependence comes from the **phase structure reshaping the gradient-energy overlap and the S_state
distribution** (`S_integral` 0.47→0.65→0.52 across Δφ), not from changing well depth. (The `corr(F, grad_align) =
−0.94` is not independent evidence — the ∇φ-alignment readout is itself monotonic in Δφ; the real content is that F
is monotonic in Δφ with an anti-phase crossover.)

## Falsifiable prediction it generates (why the reframe earns its keep)

The two-node A-well body force should **weaken as Δφ increases and cross to repulsion near anti-phase** — testable
**dynamically** (a Δφ sweep of the definitive body-force run). If the dynamical force does *not* show this trend, the
alignment-modulation reframe fails here. This is the test that keeps MC-3 honest rather than a relabel.

## Dynamical test (2026-07-18) — the quasi-static decomposition does NOT survive the dynamics

`gravity_TG_B2_dynamical_alignment.py` on the 1080 (`TG_B2_DYN_ALIGN_20260718_004440`), body-force `<F_R>` at sep=3.0
across Δφ. Result (3/5 rows at first review, run completing):

| Δφ | quasi-static F_R | **dynamical `<F_R>`** | agree? |
|---:|---:|---:|---|
| 0 | −2.34e-5 (attract) | **−5.45e-5 (attract)** | ✓ (dyn ~2.3× static, as before) |
| π/2 | −1.92e-5 (attract) | **+2.50e-5 (REPEL)** | ✗ **opposite sign** |
| 3π/4 | −0.72e-5 (attract) | −0.16e-5 (attract) | ~ (weak, antisym degrading) |

**The dynamical force is non-monotonic and flips sign at π/2**, contradicting the clean quasi-static `a·cos+b`
prediction (which stays attractive through π/2). Δφ=π/2 is exactly where the pair carries **maximal phase-driven
relative momentum** (`~sin Δφ`), so the nodes are in strong relative motion and the force is measured on a
dynamically-evolving (not static-in-place) configuration — the same momentum-current regime that contaminated the
earliest COM scouts.

**Full sweep (completed, 11.05h):** Δφ=0 −5.45e-5 (attract) · π/2 **+2.50e-5 (repel)** · 3π/4 −1.6e-6 (noise, std≫mean)
· 7π/8 +3e-8 (noise, antisym 6.5e-4 broken) · π −1.2e-7 (noise). **The run's auto-verdict
`..._CONFIRMS_QUASISTATIC_...` is a FALSE POSITIVE and is OVERRIDDEN** — its "weakens + both signs" logic was fooled
by the force collapsing into noise at high Δφ. The actual pattern is *attract at 0 → strong repel at π/2 → noise
beyond*, which is nothing like the quasi-static monotonic `a·cos+b`.

**Conclusion:** the `a·cos(Δφ)+b` decomposition (incl. the DC/monopole + phase-cos reading and the gravity+EM
parallel) is a **quasi-static artifact that does not carry over to the dynamics.** The deeper reason: for Δφ≠0 the
phase gradient drives **real relative motion** of the pair (∝sin Δφ, maximal at π/2), so `<F_R>` is measured on a
*moving* pair, not a static one — the "alignment" reading is really **motion-confounding**. **Only Δφ=0 (zero relative
momentum) gives a clean force.** So the alignment *dependence* cannot be cleanly isolated dynamically this way, and
**MC-3 as a clean loop-force alignment law is NOT supported dynamically.** Converges with the Codex secular-binding
result (`CX_GPU_FOLLOWUP_SPRINT_REVIEW_20260718.md` #1): in-phase attraction is robust; two-node dynamics beyond
in-phase are motion-confounded / unresolved.

## Boundaries

Quasi-static fit was a candidate reframe; the dynamical test down-weights it. Mirror-only, frozen modules, near-field;
no gravity/UFF/IRER claim.

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `25020bf` (2026-07-22) — *dual substrate*
**Revised since:** 3 commit(s), most recently `9517d9f` (2026-08-26)

**Harness code changed since it was written:** 4 commit(s) to `jax_scout/`.
  - `06edae5` 2026-08-27 — Fix run provenance: 30% -> 86% of runs now dated
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Also referenced by (same date or earlier):** [[RUN_QUEUE]], [[gravity_maturity/CX_GPU_FOLLOWUP_SPRINT_REVIEW_20260718]]

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
