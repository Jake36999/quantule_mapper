# Phase D / C2.6 — Geometry-Off Bug: Root Cause, Fix, and What It Invalidates

**Headline: every "pure NLS / geometry-off" conservative run in C2.2–C2.5 silently ran with geometry ON in a
squashed form that cancelled ~99.3% of the kinetic term (effective dispersion D_eff = D/151). All conservative
TRANSPORT verdicts from that arc are instrument artifacts. After the fix, a true soliton translates at exactly the
Galilean velocity with mass retention 0.9999 — the first clean coherent transport in the program.**

## 1. Discovery chain (one line each)
C2.5 Tier-1b found true stationary solitons that still "pinned" → P2b showed even the winding boost (an exact moving
solution) crept at 0.7% with momentum conserved → dense peak-tracking showed the density *never moves at all* → a
**linear wave packet also failed to translate** (solver-level, zero theory ambiguity) → k-arrays and ETDRK4 `E`
verified exact, but one `physics.step` applied 1/158 of the correct linear phase → `n_op` emitted an
amplitude-independent `+i·D·k²·ψ` term → `lap_cov ≈ 0` → **ω² ≈ 151 with `a_coupling = 0`.**

## 2. Root cause
`_soft_clip_log_with_derivative` (mirroring production `unified_omega`) is **not a boundary clamp — it is a global
log-space tanh squash with slope β=3 at the window centre**, and the window [1e-9, 1e6] is asymmetric (centre at
ω² = 0.032, not 1). Composition with the second squash in `_fused_process_omega` maps the "flat" ω²=1:

```
omega_sq = 1  --(beta=3 squash)-->  337  --(fused squash)-->  151.5
```

So with `param_a_coupling = 0` (intended: flat geometry), `lap_cov = lap_flat/151.5 ≈ 0` and the geometry-correction
term `D·(lap_cov − lap_flat)` cancels 99.34% of the dispersion. The simulated substrate was
`iψ_t = −(D/151)∇²ψ − g(ρ)ψ`, not the pure NLS.

**Quantitative closure of every anomaly:**
| observation (bugged runs) | prediction with D_eff = D/151 |
|---|---|
| feb "drag mobility" μ = 0.034–0.037 (C2.2/2.3/2.4) | 2·D_eff = 2·2.7329/151 = **0.0362** |
| C2.5 winding boost v = 0.0087 (D=1.0, k=0.628) | 2·D_eff·k = **0.0083** |
| geom-ON ≈ geom-OFF kick-loss (C2.2) | both ran geometry-squashed — same substrate |
| "no stationary soliton" (C2.3) | the Petviashvili hunt solved the FULL-D equation; the evolution ran D/151 — mismatched equations |
| one-step phase −2.49e-6 vs −3.95e-4 | ratio 1/158 ≈ 1/151 (+ quadrature detail) |

**The objects were never pinned — they were moving at the exact Galilean velocity of the bugged substrate.** The
"flow-through pinning mechanism" (C2.4) and "pinned even isolated" (C2.5 P2b) interpretations are retracted.

## 3. The fix (mirror-only; commit this)
- `Ops.geom_fac` (default **1.0 — bitwise-exact**: `D·(1.0·x) ≡ D·x` in IEEE): multiplier on the covariant
  correction inside `_nonlinear_rhs`. New param `param_geom_off=True` → `geom_fac=0.0` → **true flat geometry**.
- `a_coupling=0` alone remains NOT flat (documented on the field); the harness ops-builders
  (`phase_d_c2_2_loss_source._ops`, `phase_d_c2_5_family_scout._ops_family`) now set `param_geom_off` for their
  geometry-off paths (C2.3/C2.4 inherit).
- **Gates passed:** C1 parity re-run → `C1_PARITY_PASS` (dissipative default byte-for-byte, max|Δ|=0.0 over 50
  steps); one-step phase exact to 7 digits; linear packet translates at 2Dk; **boosted true soliton (a=0.8, s=−0.2,
  f=0, D=1.0, μ=0.2, Petviashvili residual 2.6e-13) translates at exactly 2Dk with mass_ret 0.9999** —
  `CLEAN_GALILEAN_SOLITON_TRANSPORT_CONFIRMED` on the fixed substrate (pedestal present and harmless).

## 4. Blast radius (honest accounting)
**Invalidated (instrument artifacts — all "transport" readings on the conservative branch):**
- C2.1 Stage-3 boost mobilities; C2.2 kick-loss *attribution* (the dt-convergence data itself is fine, but it
  characterised the bugged substrate); C2.3 velocity anomaly + ring-winding explanation + drag μ≈0.037;
  C2.4 local-boost "pinning is physical" + flow-through mechanism; C2.5 P2 "PINNED" verdicts + P2b.
**Still valid:**
- Phase C dissipative results (production-consistent then and now; the squash is part of the de facto geometry
  definition production has always used — parity intact).
- C2.5 P0/P1 existence machinery and results *as statements about the stationary equation* (Petviashvili solves its
  own operator, unaffected by the stepper); the C2.3 finding that feb/a\* has no branch under the FULL-D stationary
  equation stands, but its relevance to the bugged evolution was nil — and the feb/a\* existence question must be
  re-posed on the fixed substrate.
- The Codex adjointness/contract findings (they audited the code as-is; their conclusions are about the operator
  algebra and remain correct).
**Flagged for contract review (production-side, NOT hot-patched):** the soft-clip squash redefines Ω²(ρ) everywhere
— the documented law "(ρ_vac/ρ)^a_coupling with soft clipping at extremes" is actually a ~β-amplified, offset
version. Phase C results are self-consistent, but the *stated* geometry law and the *implemented* one differ. This
belongs with the C2 contract review / Codex audit lane (`docs/PHASE_D_C2_CONTRACT_REVIEW.md` addendum territory).

## 5. Next (the re-derivation campaign, pending go-ahead)
1. Re-run the C2.5 family scout (P0–P2) on the true substrate — the existence map AND transport gates, now solving
   matched equations. feb/a\* re-scouted with correct effective D.
2. Re-pose C2.1–C2.4's questions on the fixed substrate only where they still matter (most collapse into: "true
   solitons exist and transport cleanly — map the family space and then do two-node").
3. Two-node interaction (old Stage 4) becomes meaningful the moment a moving-soliton family is confirmed at N=96.
4. CFL re-baselining: the fixed substrate has REAL dispersion (the old "dt=1e-3 stable at N=96" was measured with
   D_eff≈0 — stability windows must be re-established before any long campaign).

## 6. Credit where due
The bug surfaced because the C2.5 verdict (true solitons that pin) contradicted a mathematical identity (Galilean
covariance of the discrete spectral NLS). Chasing that contradiction — rather than writing the convenient
"stronger null" — is what exposed the instrument. The same discipline (null controls, exact-solution probes,
independent integrators) should gate every future transport claim.

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `0886dd0` (2026-07-09) — *Phase D C2.6: geometry-off bug found+fixed -- all conservative transport verdict*
**Revised since:** 6 commit(s), most recently `f2b0527` (2026-08-27)

**Harness code changed since it was written:** 22 commit(s) to `jax_scout/`.
  - `06edae5` 2026-08-27 — Fix run provenance: 30% -> 86% of runs now dated
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine
  - `25020bf` 2026-07-22 — dual substrate
  - *…and 17 more.*

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Also referenced by (same date or earlier):** [[PHASE_D_C2_6_CODEX_AUDIT_HANDOVER]], [[PHASE_D_CLOSEOUT_CONSOLIDATION]]

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
