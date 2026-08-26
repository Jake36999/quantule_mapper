# TG-B2 Two-Node Method Assessment — Why the Evolution Is Violent, and the Plan to Settle the Dynamical Sign

Author: Claude (primary), 2026-07-16. Companion to `TG_B2_TWO_NODE_FORCE_RESULTS.md`. Purpose: assess *why* the
two-node evolution is so violent that it obscures the weak loop force, answer the diagnostic questions, and design a
clean experiment (an overnight long run) to settle the dynamical sign. For Codex's secondary review. No model change.

## 1. The problem in one line

The quasi-static calc says the A-well attracts (weak, ~2e-5). The dynamical run cannot confirm it because the
two-node state **breathes violently** and the weak loop force oscillates with the breathing instead of accumulating
a secular inward impulse (`<J_well> = +4.35e-5 ± 1.08e-4`, consistent with zero).

## 2. Root cause — it is the bare φ initial condition, NOT the geometry

**Decisive control:** the feedback-**off** arm (A=1, T/G present but *not coupled back* to φ — i.e. **zero geometry
acting on φ**) breathes identically to the full-loop arms:

| quantity (feedback-OFF, A=1) | value |
|---|---|
| amp = max\|φ\| swing | **0.904 → 1.416 (56.5%)** |
| half-space momentum `P_R` swing | **−7.74 → +5.26** |
| massR (KG, non-conserved) swing | 76.9 → 83.7 |
| breathing period | ~8.0 (vs single-node internal 2π/ω = 6.52) |
| loop's contribution on top (`J`) | ~2e-4 (≈ 3e-5 of the breathing) |

So the violence is entirely in the **bare two-node φ field**. Superposing two individually-stationary Q-balls at
overlapping separation (3.5, comparable to node size ~1–2, same phase) is **not** a solution — the constructive
interference in the overlap creates excess density, and the pair launches into a large-amplitude breather (56%).
The geometric feedback loop is a small perturbation riding on this; it did not cause the violence.

## 3. The diagnostic questions, answered

**Q: Is it the geometric fields' initial reactions with each other, then averaging out?**
**A: No — ruled out.** With the geometry switched off entirely, the pair breathes the same. The T/G fields are a
weak perturbation, not the driver. (There *is* a secondary T/G ring-up transient — see below — but it is sub-dominant.)

**Q: Is there a missing factor in the implementation?**
**A: Not in the geometry sector.** The missing factor is a **valid quasi-stationary two-node initial state.** The
implementation faithfully evolves whatever IC it is given; the IC we gave it (a raw superposition) is a violently
excited state, so the evolution is violent. This is a well-known soliton-superposition artifact, not a code bug.
Two *secondary* factors worth noting but not the primary cause:
- **T/G ring-up:** the S_state source switches on at t=0 (nodes present, T=G=0), impulsively exciting the T/G damped
  oscillators (this is the same underdamped ring-up FC-1 saw as the single-node `T_peak` overshoot, Q~15). Its
  damping time ~1/γ ≈ 12–16 is **comparable to the run length T=12**, so over one window it has *not* settled. Small
  vs the bare breathing, but it means A is still ringing.
- **Non-variational coupling (equation-audit finding 1):** the model is not derived from one action, so total energy
  is only approximately booked. This is *not* the violence driver (off breathes without any loop), but it means we
  cannot use exact energy conservation as a cleanliness check on long runs; `charge Q` (exactly conserved) and the
  `P_total` symmetry (conserved to 2e-14 here) are the trustworthy invariants to monitor instead.

**Q: Why did the quasi-static calc mislead?**
**A:** It assumed (a) a frozen stationary φ and (b) a fully-settled steady A. Dynamically **both fail**: φ breathes at
56%, and A never settles (T/G ring-up ~ window). When the medium response is not adiabatically fast versus the node
dynamics, the quasi-static force law does not apply — and the sign can wash out or flip.

## 4. What this implies for settling the dynamical sign

To see the weak secular loop force, we must remove the two things burying it: **the bare breathing** and **the
unsettled A**. Three levers, best used together:

- **A. Quiet the initial state.** Relax/cool the two-node configuration toward quasi-stationary before measuring —
  imaginary-time / gradient flow on φ (optionally with a soft separation constraint so they don't merge). A quiet
  pair lets a weak secular force produce a clean monotone drift.
- **B. Larger separation.** Less tail overlap → far less breathing (breathing should fall faster with separation
  than the screened force does), improving signal-to-breathing even though both weaken.
- **C. Long run + time-average.** Evolve for `T ≫` both the breathing period (~8) and the T/G damping time (~13) —
  e.g. `T ~ 200–400` (many periods, several damping times) — let the ring-up damp, then **time-average the loop
  impulse over the settled oscillation** to extract the secular DC force. Monitor charge Q and P_total as cleanliness
  gates.
- **(Diagnostic) Adiabaticity check.** Directly compare the T/G response time to the node breathing time; if
  non-adiabatic (as it currently appears), record that the quasi-static law is inapplicable and the dynamical average
  is the authoritative measure.
- **(Optional) Soft geometry turn-on.** Ramp the feedback coupling 0→full over a turn-on time to avoid impulsively
  kicking the T/G fields (reduces the ring-up transient).

## 5. Proposed overnight run (design, up to ~10 h, automated, review-in-the-morning)

Goal: settle the dynamical sign by extracting the **secular, time-averaged** loop force from a quiet, long-averaged
two-node evolution. Draft spec (to be finalized after Codex's review):

- **IC:** cooled/relaxed two-node state (lever A) at a **sweep of separations** (lever B), e.g. sep ∈ {3, 4, 5, 6}.
- **Arms per separation:** off, A-well, A-hill (sign control).
- **Duration:** long per arm (lever C), `T` several × the T/G damping time; time-average `J` over the settled window
  (discard the first ~1–2 damping times as transient).
- **Observable:** field-momentum impulse `J = P_R(full) − P_R(off)`, time-averaged; **primary verdict = sign of the
  secular average**, with the A-well/A-hill antisymmetry as the control.
- **Gates:** charge Q conserved; P_total ≈ 0; nodes remain distinct (no merger) — else that separation is discarded.
- **Automation:** row-addressable (one row per separation×arm), `ROW_*_STARTED/COMPLETE/FAILED.json` markers,
  scalar streaming (no full-field host transfer — the runtime rule), a final `RUN_COMPLETE.json` + `summary.json`
  with the per-separation secular `<J>` and verdict, plus a stop-rule (halt a row if Q drifts or a node merges).
- **Lane:** local GPU overnight, or the Colab A100 fast-lane if the sweep is heavy (decide at prep time).

**Honest expectation to preregister:** either (i) a quiet pair shows a clean secular A-well **attraction** (confirms
the quasi-static result dynamically), (ii) a clean **repulsion** (the quasi-static sign does not survive — a real
result), or (iii) a **null within noise** (the loop force is too weak / too non-adiabatic to produce net motion at
these parameters). All three are reportable; we do not tune to attraction.

## 6. Status

- Two-node inter-node attraction is **quasi-static only, dynamically unconfirmed** (`TG_B2_TWO_NODE_FORCE_RESULTS.md`).
- Violence root cause: **bare non-stationary two-node IC** (evidence: feedback-off breathes 56%), not the geometry.
- Next: Codex secondary review → finalize + build the automated overnight long-average run (§5) → review in the morning.
- Boundaries unchanged: mirror-only; frozen TG-B1S and production untouched; no gravity/UFF/IRER claim.

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `25020bf` (2026-07-22) — *dual substrate*
**Revised since:** 2 commit(s), most recently `3eb93af` (2026-08-26)

**Harness code changed since it was written:** 3 commit(s) to `jax_scout/`.
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Also referenced by (same date or earlier):** [[RUN_QUEUE]], [[gravity_maturity/CONCEPT_IMPLEMENTATION_AND_GAP_ANALYSIS_CODEX_CONTINUATION_20260717]]

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
