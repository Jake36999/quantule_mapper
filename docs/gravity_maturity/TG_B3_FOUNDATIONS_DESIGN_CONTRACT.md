# TG-B3 Foundations Design Contract — Closing the R_res and N_t Gaps (+ χ_out)

Author: Claude (primary), 2026-07-16. Design-only; no runs, no frozen-module edits (registry-protected imports only).
Closes the Phase-F gaps from `TG_SEMANTIC_BASELINE_AND_POSTULATE_AUDIT.md`, grounded in the ORIGINAL theorizing
(`D:\memory_bank\bank 1\2025_May.txt`) and Jake's hint (2026-07-16): *"as OIWs interact and collapse, their action
requires action over time which results in the initial interaction with the geometric field; that has a ripple
effect as the energy released is propagated across both substrates but primarily active in the geometric field."*

## 0. Theory grounding (checked against the source transcripts)

| design element | original-theory support |
|---|---|
| Collapse/resolution is a **continuous, deterministic, rate-like process** (not an event flag) | "collapse not as a single all-or-nothing flip, but as a **continuous transition** governed by a field equation" — Resolution Field Dynamics, 2025_May.txt:21254 |
| **Temporal load sourced by the rate of coherence gain** | "Time — emergent direction of **increasing resonance coherence**" (master doc 2.8); "time irreversibility = **coherence ratchet**" (5.6) |
| **Geometry sourced by resonance-release dynamics; relief = manifold ripple** | "released resonance energy unbinds localized fields, **pushing the surrounding informational manifold outward**" (:19645); "collapse cascade = decoherence **ripple through manifold**" (v2 3.9); "informational curvature linked to **resonance release dynamics**" (v2 3.10) |
| Release propagates across both substrates, **primarily geometric** | Jake's hint — becomes a *testable prediction* (§4) |

## 1. The key implementation shortcut (F1)

**The theory-faithful resolution-rate source already exists and passed its semantics gate.** TG-S built and validated
`R_relax` — classified `PHASE_TENSION_RELAXATION`, i.e. it "robustly detects mismatch reduction": the local **rate of
phase-tension decrease** = the **rate of coherence gain** = the loop doc's own continuous candidate
`R_coh = [−∂_t K_phase]_+`. It passed TG-S semantics (quiet on nulls, active on genuine locking, distinguishes
relaxation from disorder, numerically convergent) and was then *disabled* in the state-load branch. **F1 = re-enable
it as the T-source on a new branch** — the machinery and its gates are already banked; it needs re-gating on the new
branch, not invention.

## 2. The TG-B3 model (new branch; frozen TG-B1S/B2 untouched)

State `(φ, π, T, V_T, G, V_G)`, KG scout form from the loop doc — now with **both** coefficients wired:

```text
φ_t = N_t(T) · π                                        [F2: temporal lapse ON the substrate]
π_t = c² ∇·(A_s(G) ∇φ) − N_t(T)·(m²φ − g(ρ)φ)
T:   V_T,t = c_T²∇²T − ω_T²T − γ_T V_T + α_T·R_lock[φ]  − κG − absorb·V_T
G:   V_G,t = c_G²∇²G − ω_G²G − γ_G V_G + α_G·R_rel[φ]   − κT − absorb·V_G
N_t = exp(−β_T·T̂)        A_s = exp(+ε_G·G)   (positive maps; N_t ≠ A_s by construction — separate fields)
```

**Two rate sources, two jobs** (the theory's split, per §0):

```text
R_lock[φ] = [−∂_t K_phase]_+      K_phase = ρ·|∇θ|² (phase-tension density; the validated TG-S R_relax observable)
            → sources T: "locking-in costs time" (coherence-ratchet = chronology load)

R_rel[φ]  = [−∂_t e_bind]_+       e_bind = local bound/coherent energy density (preregistered definition)
            → sources G: "released energy pushes the manifold" (the hint's 'initial interaction with the
              geometric field'; curvature from release dynamics)
```

Comparison arm: `S_state` (the current state load) runs as a third source family, connecting TG-B3 to the validated
baseline and letting us measure what the rate sources *change*.

## 3. The polarity contract (SIGN-1 lesson applied BEFORE building)

Signs are a theory commitment, fixed before any run, with the flip-control mandatory in every experiment:

```text
locking/dense node → R_lock > 0 → T > 0 (chronology load) → N_t < 1  (node region ticks SLOWER — throttling)
release/geometric response      → A_s < 1 at the node      (A-well — the confirmed attraction polarity)
```

The κ and α_G signs must be derived to satisfy this contract given both sources (note the current B2 chain gets
G<0 via −κT; adding +α_G·R_rel pushes G positive — the resolution of this tension is a *derivation task* in B3-0,
not a tuning knob). If the derivation cannot satisfy both commitments simultaneously, that is a reportable theory
finding, not a reason to tune.

## 4. χ_out — the named relief channel (F3) and the emission experiment

- **Named observables:** outgoing shell flux for each substrate at preregistered radii —
  `chi_out_G(r,t)`, `chi_out_T(r,t)`, `chi_out_phi(r,t)` — plus cumulative exported energy per channel. The absorber
  stays a sponge; χ_out is measured *inside* it.
- **The emission experiment (B3-2):** drive a genuine "reconfiguration of dense OIW patterns" — a **two-Q-ball
  capture/merger** from the C3 collision map's capture regime — with the loop on. Measure the emitted ripple:
  amplitude, propagation speed (must match c_T/c_G — a built-in dispersion check), waveform, and the
  **cross-substrate energy split**.
- **Preregistered prediction from the hint:** the released energy propagates across both substrates but
  **primarily in G**: `E_out(G) > E_out(φ_radiation)`. Falsifiable; if the split goes the other way, that is the
  result.
- **Attractor classification** per the theory taxonomy (STABLE_NONRADIATING … RUNAWAY_COLLAPSE) on the post-merger
  remnant — which simultaneously addresses the node-formation half-gap at the "consolidation with relief" level
  (a formation-from-wavepackets scout remains a later, separate item).

## 5. Ledger (the variational half-gap, now load-bearing)

`N_t` multiplying the substrate dynamics makes honest energy bookkeeping mandatory, not optional. B3 must choose the
exchange class explicitly (loop doc rule): either derive the conservative action `L = L_φ + L_T + L_G + L_int`
(preferred — also fixes reciprocal terms and the §3 signs from first principles = VAR-1 payoff), or run the explicit
dissipative ledger with **every** loss/work term booked. Mixing without the ledger is disallowed.

## 6. Staged plan (semantics-first, per TG-S precedent; each stage gated)

| stage | content | gate |
|---|---|---|
| B3-0 | polarity-contract derivation + exchange-class decision (design doc) | signs derived, not tuned; Jake reviews |
| B3-S | semantics re-gate of `R_lock` and `R_rel` on the new branch (reuse TG-S event suite + TG-S5 derivative audit — rate sources are ∂_t-based, so dt/cadence-noise audits are mandatory) | TG-S-style classification passes |
| B3-1 | single-node closed loop with N_t + A_s: baseline stationarity, bounded backreaction, **extended frequency contract** (FC-1 machinery now has TWO predicted contributions: geometric stiffening + temporal throttling) | baseline + ledger + contract |
| B3-2 | the merger/emission χ_out experiment (§4) | dispersion check + preregistered split prediction |
| B3-3 | two-node force under the full dual-coefficient loop (does the temporal limb add attraction?) | body-force observable; antisymmetry + null controls |

Numerical cautions carried in from the audits: `[·]_+` rectifiers get a preregistered smooth (softplus) variant;
`N_t·π` changes the effective CFL — own dt audit; no full-field host transfers in loops; all new files import the
frozen registry modules.

## 7. Boundaries

Design contract only. New hypothesis branch — results will be statements about the **rate-sourced, dual-coefficient
loop**, reported against the state-load baseline. Mirror-only; production and frozen branches untouched; no gravity /
UFF / IRER-validation claim regardless of outcome; all outcomes (including "rate sources break the boundedness")
are reportable results.

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `25020bf` (2026-07-22) — *dual substrate*
**Revised since:** 1 commit(s), most recently `616fe31` (2026-08-25)

**Harness code changed since it was written:** 3 commit(s) to `jax_scout/`.
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Also referenced by (same date or earlier):** [[RUN_QUEUE]], [[gravity_maturity/CONCEPT_IMPLEMENTATION_AND_GAP_ANALYSIS_CODEX_CONTINUATION_20260717]], [[gravity_maturity/TG_RECOVERED_CONCEPTS_INTEGRATION_AND_REPRIORITIZATION]]

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
