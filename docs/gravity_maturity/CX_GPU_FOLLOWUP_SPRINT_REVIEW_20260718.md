# Review: Codex GPU Follow-up Sprint (4 recovered-concept tests)

Author: Claude (primary review), 2026-07-18. Reviews `colab_jobs/results/cx_gpu_followup_tests/` — a four-test Colab
sprint covering recovered-concept candidates from the gap analysis. Verdict on the sprint: **honest, disciplined work
— every test reports a bounded verdict (unresolved / inconclusive / detected / admissible), no over-claiming.** Two
land as advances, two land as honest "not yet," and all four are correctly scoped. Details and my cross-checks below.

## 1. Cooled two-node secular force — `TG_B2_COOLED_PAIR_SECULAR_SIGN_UNRESOLVED`

Ran the definitive body-force driver at cool_T ∈ {0, 40}, seps {3,4}. Findings (from `cooled_pair_metrics.csv`,
`secular_force_assessment.json`):
- **Body force: robustly ATTRACT** at all rows (−5.45e-5 sep3, −4.85e-5 sep4; cooled −5.35e-5/−4.70e-5), sign-flip
  control holds. Re-confirms the hardened TG-B2 result.
- **Separation trajectory: CLOSING** (loop-differential) at all 4 rows — consistent with attraction.
- **Momentum-impulse proxy: DISAGREES** — INWARD at cool_T=0 but flips OUTWARD_OR_NULL at cool_T=40, while body
  force and separation stay attractive/closing.
- **Cooling (T=40) did not materially quiet the pair** (off breathing peak-to-peak ~unchanged/slightly up).

**My read (slightly less conservative than the auto-verdict, and I think fairer):** 2 of 3 observables — the *clean*
body force and the separation-closing — agree on attraction; only the **momentum-impulse proxy** (the known
breathing/absorber-contaminated observable from the whole two-node saga) dissents, and it flips with cooling. So this
is not a symmetric "unresolved"; it is **body-force + separation lean attractive, momentum proxy unreliable.** Codex's
conservative `SECULAR_SIGN_UNRESOLVED` is defensible and correctly honours the "body-force ≠ secular binding"
guardrail — but the weight of evidence tilts attractive, and the blocker is specifically the contaminated momentum
proxy, not a genuine sign conflict in the clean observables. **Secular *binding* (does a pair actually bind) remains
formally open; a cleaner secular observable — or a genuinely quiet IC — is the missing piece, not more body-force.**

## 2. Gravity-D load-capacity / yield map (MC-1) — `GRAVITY_D_LOAD_CAPACITY_MONOTONE_OR_INCONCLUSIVE`

The MC-1 test: is the production saturation cliff a critical-RD **yield point** (Concept-20 "elastic yield")? The
pilot matrix came back **monotone or inconclusive — no clear yield-point onset found.** So **MC-1's reframe is NOT
confirmed** by this pilot. This *vindicates the labeling-discipline caution*: "the cliff is a predicted yield point"
was a candidate, and the first test to look for the predicted onset did not find it. MC-1 stays a candidate reframe,
now with one inconclusive result against a clean confirmation. Next: a wider/finer matrix, or accept that the cliff
may not be the yield-point observable. (Pilot only — not a falsification, but not support either.)

## 3. Clock-migration characterization (GAP-2 / G1) — `TG_CLOCK_MIGRATION_RESPONSE_DETECTED`

Tests whether the G1 "clock migrates toward the source" is a real field response or just an instrument fault. Result:
**5/5 near-clocks migrate toward the source; flat control null (max projection 7.7e-11).** So the migration is a
**reproducible, directional field response**, not noise — the clock falls toward the load, a gravitational-infall-
shaped signal. **This is a genuine advance for the temporal sector:** it reframes the G1 "calibration failure" as the
clock *being* a probe of the lapse gradient (responding to load), exactly the gap-analysis conjecture. **Discipline:**
a *detected response*, NOT a calibrated clock law and NOT time dilation — but it converts a dead-end failure into a
live, characterizable signal.

## 4. Rate-source semantics bridge (GAP-1 / B3-S) — `TG_RATE_SOURCE_ADMISSIBLE_FOR_DESIGN_REVIEW`

Re-gates `R_relax` (phase-tension relaxation = rate-of-coherence-gain) through the TG-S semantics suite on the path to
B3. Result: **R_relax admissible** (tg_s returncode 0). So the **resolution-RATE source (GAP-1 / R_coh) has a
validated source path for B3** — the foundations plan's shortcut (re-enable the TG-S-validated R_relax as the rate
source) is confirmed viable. Positive for B3-S; the source can proceed to design-review/feedback wiring.

## Cross-link to the parallel local run (my dynamical alignment sweep)

My overnight 1080 dynamical alignment sweep (`TG_B2_DYN_ALIGN_20260718_004440`, 3/5 Δφ rows at review time)
independently reinforces test #1's lesson: the body force at **Δφ=0 attracts (−5.45e-5)** but at **Δφ=π/2 flips to
REPEL (+2.50e-5)**, non-monotonically — so the clean quasi-static `a·cos(Δφ)+b` decomposition **does NOT survive the
dynamics** at intermediate phases (π/2 has maximal phase-driven relative motion, contaminating the force). Both runs
converge on: **in-phase attraction is robust; the two-node dynamics *beyond* in-phase (alignment law, secular
binding) are messy and unresolved.** See `TG_B2_MC3_ALIGNMENT_RESULTS.md` §dynamical.

## Sprint scorecard

| test | verdict | reading |
|---|---|---|
| cooled secular force | SECULAR_SIGN_UNRESOLVED | body-force+separation lean attractive; momentum proxy dissents → secular *binding* still open |
| yield map (MC-1) | MONOTONE_OR_INCONCLUSIVE | MC-1 reframe not confirmed (vindicates the caution) |
| clock migration (GAP-2) | RESPONSE_DETECTED | **advance** — G1 "failure" is a real load-seeking clock response |
| rate-source bridge (GAP-1) | ADMISSIBLE | **advance** — R_relax validated as the B3 rate source |

Boundaries unchanged: mirror-only; frozen modules; no gravity/UFF/IRER/time-dilation claim. All four are candidate/
diagnostic results, not confirmations.
