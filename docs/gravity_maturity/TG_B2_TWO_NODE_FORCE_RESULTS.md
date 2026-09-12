# TG-B2 Two-Node Inter-Node Force — Results (A-well vs A-hill)

Author: Claude (primary), 2026-07-15. Tests the SIGN-1 ruling: the theory-faithful **A-well** sign
(`A = exp(+ε_G G)`, dense node → lower A → attraction) vs the frozen TG-B1S **A-hill** (`exp(−ε_G G)`, control).
New labelled branch; frozen TG-B1S untouched. Mirror-only; production closed.

## Headline (FINAL, 2026-07-16 — supersedes the earlier "NOT DYNAMICALLY CONFIRMED")

```text
TG_B2_DEFINITIVE_AWELL_ATTRACTION_CONFIRMED     (dynamical, body-force observable, A100 run)
```

The theory-faithful **A-well loop produces a real, secular, dynamical inter-node ATTRACTION** in the mirror; the
frozen A-hill produces repulsion (sign control). The earlier "not dynamically confirmed" status is superseded: the
obstruction was **contaminated proxy observables** (half-space momentum; masked-COM), not the physics. The
definitive measurement — the verified Gravity-D body force evaluated on the live fields and time-averaged over the
settled window (`sweep_runs`-equivalent in `colab_jobs/Colab_runs/cl_tg_b2_definitive_force_20260716_113033`) —
gives `<F_R_well> = −5.45e-5 ± 0.80e-5` (sep 3.0) and `−4.85e-5 ± 0.84e-5` (sep 4.0): **negative (attractive) at
100% of settled samples**, perfectly antisymmetric under the A-sign flip (well+hill ≈ 1e-9), off/A=1 null exact,
all gates passing, consistent across separations, and consistent with the static prediction in sign, order of
magnitude (~2×, normalization caveat), and falloff shape (dyn ratio 0.89 vs static 0.92). Full detail in
§Definitive below. History of the earlier retractions preserved below unchanged.

## Instrument 1 (COM evolution) — FAILED, RETRACTED

`jax_scout/gravity_TG_B2_two_node_awell.py` evolved two Q-balls (full-loop vs feedback-off) and read the loop force
from the node-separation COM curvature, `F_TG = a_full − a_off`. It is **not trustworthy** and its labels are
retracted:

- At **Δφ=π/2**: `F_TG_well = −7e-9` (well "attracts"). But the trajectory showed **mass sloshing** across the x=0
  split (massR/massL 50/50 → 63/39) — Δφ=π/2 maximizes the momentum *current* (`~sin Δφ`), contaminating the COM
  observable.
- At **Δφ=0**: `F_TG_well = +2.3e-7` (well "repels") — the **opposite sign**, dominated by an in-phase tail-interference
  transient (`v0 = +7.5e-3`).

Two phase choices → opposite signs → the COM-curvature observable does not measure a physical central force. Runs
`TG_B2_TWO_NODE_AWELL_20260715_232649` (π/2) and `..._233906` (0) are kept as the negative instrument record.

## Instrument 2 (static analytic force) — CLEAN

`jax_scout/gravity_TG_B2_static_force.py` (CPU). Same transient-free approach as FC-1: build the two in-phase
Q-balls, solve the **static screened T/G** response (exact Fourier, as FC-1/box-dependence), form A, and apply the
**verified Gravity-D momentum law** (force-contract residual 3.6e-17) restricted to the right node:

```text
F_R,x = − c² ∫_{x>0} (∂_x A) |∇φ|² dV        F_R,x < 0 ⇒ node pushed toward centre ⇒ ATTRACTION
```

Run `sweep_runs/TG_B2_STATIC_FORCE_20260715_235154`:

| separation | F_R_well | reading | F_R_hill | bare A=1 |
|---:|---:|---|---:|---:|
| 2.5 | −2.58e-5 | ATTRACT | +2.58e-5 | 0.0 |
| 3.0 | −2.34e-5 | ATTRACT | +2.34e-5 | 0.0 |
| 3.5 | −2.21e-5 | ATTRACT | +2.21e-5 | 0.0 |
| 4.0 | −2.15e-5 | ATTRACT | +2.15e-5 | 0.0 |
| 5.0 | −1.63e-5 | ATTRACT | +1.63e-5 | 0.0 |

**Three self-checks pass:**
1. **Bare control:** `A=1` → force = 0.0 exactly (no spurious force from the setup/observable).
2. **Sign control:** flipping A-well→A-hill reverses the force, near-perfectly antisymmetric (linear in ε_G ⇒ it is
   genuinely the A-coupling, not noise).
3. **Falloff:** |F_R_well| decreases monotonically with separation (2.58e-5 → 1.63e-5).

## Interpretation (bounded)

The theory-faithful A-well sign produces **inter-node attraction** in the mirror — realizing, for the first time on
the corrected-sign branch, the direction IRER's throttling logic (dense → slower → lower lapse → attraction) and
Jake's original intuition predicted. The frozen TG-B1S A-hill gives the opposite (repulsion), consistent with its
being the *anti*-throttling scaffold. This is the first genuinely **gravity-relevant** positive of the TG line.

## Honest caveats (sizing the claim)

- **Quasi-static.** The T/G response is the steady screened solve (absorber neglected), and F_R is the *instantaneous*
  force on the initial configuration — not a full dynamical trajectory. A clean **dynamical** confirmation (nodes
  actually converging, measured with a proper field-momentum observable rather than the broken COM-curvature) is the
  key next hardening step.
- **Effective-medium body force.** F_R uses the Gravity-D body-force law (verified as the *total* force for a single
  probe). Applied to the pair (each node a probe in the other's A-field) the sign is robust; the fully self-consistent
  two-node force could differ in magnitude.
- **Weak & near-field.** Force ~2e-5 (ε_G=0.06); separations 2.5–5.0 are comparable to the node size, so this is the
  near-field regime and the gentle falloff should not be read as a clean Yukawa law.
- **Short-range, NOT gravity.** Screened mediation → not 1/r², not universal free fall, not IRER-validated. Long-range
  is the separate LR-1 fork. Mirror-only; frozen TG-B1S and production untouched.
- **Source-normalization mismatch between the two instruments (Codex catch).** The static script recomputes the
  S_state energy/charge reference maxima (`e_ref`, `q_ref`) from the *two-node* field per separation, whereas the
  dynamical run (and the frozen TG-B1S model) uses the *single-node* B1S reference normalization. `S0` is frozen in
  both. This does **not** change the sign (normalization scales S_state → T → G → (A−1) → force linearly, no flip),
  but it means the static magnitude/falloff is **not a strict quantitative predictor** of the dynamical run. The
  overnight dynamical run uses the single-node B1S normalization consistently; a corrected static predictor should
  too if a strict static↔dynamic comparison is wanted.

## Dynamical test (instrument 3) — does NOT confirm the quasi-static attraction

`jax_scout/gravity_TG_B2_dynamical_force.py` (GPU). Uses the **field-momentum** observable (the fix for the COM
contamination): `p_x = −2 Re(π* ∂_x φ)`, half-space momentum `P_R = ∫_{x>0} p_x dV`, loop impulse
`J = P_R(full) − P_R(off)`, at Δφ=0. Run `sweep_runs/TG_B2_DYNAMICAL_FORCE_20260716_001134` (N=64, L=16, sep=3.5,
T=12).

**Instrument is sound:** momentum conserved to `P_tot ≤ 2.3e-14`; `J_well` and `J_hill` near-perfectly antisymmetric
(`<J_well>+<J_hill> = 5e-8`, so it *is* the linear A-coupling, not noise); starts from rest (`P_R(0)~2e-15`).

**But the physics does not confirm attraction:**
- The in-phase two-node superposition is a **violently breathing** excited state: `P_R,off` swings from −7.74 to
  +5.26 (amplitude ~13). It is not a quasi-static pair.
- The loop impulse **oscillates with the breathing**: `J_well ∈ [−1.76e-4, +2.65e-4]`, positive 64% of samples.
- **Time-averaged `<J_well> = +4.35e-5 ± 1.08e-4`** — i.e. *weakly outward* and statistically consistent with zero
  (|mean|/std = 0.40; with correlated samples, not significant). This is the **opposite** of the static-predicted
  inward attraction, and certainly not a confirmation of it.
- The single-time snapshot `J_well(T) = −1.76e-4` that produced the run's auto-label
  `TG_B2_DYNAMICAL_AWELL_ATTRACTION_CONFIRMED` is just one point of an oscillating quantity — **retracted.**

**Reading:** the quasi-static calc assumes A adiabatically tracks the node into a steady well; dynamically, the T/G
response time (ω_T=1.25, ω_G=0.85 → periods ~5–7) is *not* fast compared to the node breathing, so the adiabatic
well never cleanly forms and the loop force oscillates in sign rather than accumulating a secular inward impulse.
Much of the breathing is also a crude-initial-condition artifact (superposing two non-stationary Q-balls at
overlapping separation).

**Net:** A-well quasi-static attraction is real *as a quasi-static statement* but **does not survive a proper
dynamical test in this configuration.** No confirmed inter-node attraction.

## Definitive dynamical measurement (instrument 4) — CONFIRMS attraction

`jax_scout/gravity_TG_B2_definitive_force.py`, run on the **Colab A100 fast lane** (Codex capsule,
`colab_jobs/Colab_runs/cl_tg_b2_definitive_force_20260716_113033`; archive hash verified, 0 integrity mismatches,
x64/float64, 45 min). Primary observable = the **dynamical Gravity-D body force** on the live fields,
`F_R(t) = −c²∫_{x>0}∂_xA|∇φ|²dV`, time-averaged over the settled window (discard first 40%); off (A=1) is an exact
built-in null; A-well/A-hill antisymmetry is the control. CPU-prevalidated (off-null exact; antisymmetry 2e-16).

| sep | `<F_R_well>` | std | frac. negative | `<F_R_hill>` | off null | gates |
|---:|---:|---:|---:|---:|---:|---|
| 3.0 | **−5.450e-5** | 7.95e-6 | **100%** | +5.450e-5 | 0.0 | PASS |
| 4.0 | **−4.853e-5** | 8.38e-6 | **100%** | +4.852e-5 | 0.0 | PASS |

The force is not an average over sign-flipping oscillation — it is **steadily attractive**, modulated ±15% by the
breathing, at every settled sample, at both separations.

**Cross-analysis (all instruments):**

| observable | sep 3.0 | sep 4.0 | direction |
|---|---:|---:|---|
| dynamical body force (definitive) | −5.45e-5 | −4.85e-5 | **ATTRACT** |
| sep-differential (overnight proxy) | −2.41e-5 | −2.41e-5 | ATTRACT |
| static quasi-static prediction | −2.34e-5* | −2.15e-5* | ATTRACT (*two-node-norm caveat) |
| momentum-slope J (overnight proxy) | +4.5e-6 | +9.1e-6 | *uninterpretable* |

Three independent observables agree on attraction at both separations. The lone discordant proxy — the half-space
momentum slope — is demonstrably not a direction meter: its own calibration reference (the bare pair's secular
momentum drift) **flips sign between separations** (+1.65e-3 at sep 3.0, −3.54e-3 at sep 4.0) while nothing physical
flips; it is dominated by breathing-radiation/absorber momentum bookkeeping. Overnight run auto-verdict
(`...SIGN_INCONSISTENT_OR_NULL...`) correctly reflects the *proxies'* unreliability, not the physics. Falloff-shape
consistency: dynamic sep4/sep3 ratio 0.890 vs static 0.919. Magnitude: dynamic ≈ 2.3× static — attributable to the
static script's two-node normalization (Codex catch) plus dynamical A-φ correlation; same sign, same order.

**Bounded interpretation:** the corrected-sign (A-well, throttling-consistent) temporal-geometric loop produces a
weak, short-range, steadily attractive inter-node force in the dynamical mirror, sign-controlled and null-checked —
realizing dynamically the direction IRER's throttling logic predicted. **Not** gravity: near-field, screened
(no 1/r²), one resolution/box, effective-medium body-force observable (a midplane stress-flux measurement remains
the fully independent cross-check), no UFF, no IRER validation. Frozen TG-B1S and production untouched.

**Hardening next (not auto-launched):** N/dt/box convergence; ε_G linear scaling; extended F(q)+screening fit at
larger separations; midplane-flux independent cross-check.

## Superseded history — the earlier "better-conditioned dynamical test" plan (kept for the record)

1. **Better-conditioned two-node state:** relax/cool the pair toward a quasi-stationary configuration (imaginary-time
   or a bound two-soliton ansatz) to suppress the breathing artifact before measuring the force.
2. **Long-time averaging over many breathing periods** with proper (decorrelated) statistics to extract any secular
   DC loop force underneath the oscillation.
3. **Adiabaticity check:** compare the T/G response time to the node breathing time; if non-adiabatic (as it appears),
   the quasi-static force law does not apply and a fully dynamical treatment is mandatory.
4. Only if a secular dynamical attraction is established: ε_G scaling, F_R(q) map, and (separately) the long-range
   LR-1 question.

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `25020bf` (2026-07-22) — *dual substrate*
**Revised since:** 11 commit(s), most recently `bd93089` (2026-09-12)

**Harness code changed since it was written:** 6 commit(s) to `jax_scout/`.
  - `caf61af` 2026-09-10 — Build HUD items 1+2: snapshot writer and universal field renderer
  - `06edae5` 2026-08-27 — Fix run provenance: 30% -> 86% of runs now dated
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine
  - *…and 1 more.*

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Also referenced by (same date or earlier):** [[RUN_QUEUE]], [[gravity_maturity/CONCEPT_IMPLEMENTATION_AND_GAP_ANALYSIS_CODEX_CONTINUATION_20260717]], [[gravity_maturity/TG_B2_METHOD_ASSESSMENT_AND_DYNAMICAL_PLAN]], [[gravity_maturity/TG_B2_OVERNIGHT_RESULTS]]

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
