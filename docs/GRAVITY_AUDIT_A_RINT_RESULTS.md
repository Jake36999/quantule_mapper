# Gravity Audit A — Interaction-Intensity Diagnostic (Stage A + A.1 hardening)

**Verdict (narrowed, per reviewer):** `EXPLICIT_SUBSYSTEM_INTERACTION_INTENSITY_VALIDATED` +
`RESOLUTION_LOAD_CANDIDATE_PROMISING`. "Source" is deliberately **avoided** — that would imply a connection to
completed-resolution throughput, which is undetermined. What is established: for an **explicit target–environment
(A/B) split**, `I_int` is deterministic, grid-convergent, decomposition-invariant (exact invariances),
density-independent, and predictive of an independent later observable — but only on a distance-only sweep so far.
The decomposition problem is resolved **only within the explicit-subsystem model** — this supports probe–environment
experiments, **not** single-field self-gravity. **Audit A.2 (below) now closes the source audit:** the coupling is
lawful (momentum-conserving, symmetric), `I_int` is the best relational predictor (marginally over overlap), and the
force is a directional gradient distinct from the intensity. (Label held conservative — it has run a notch strong
3× in this thread.)

## Audit A.2 — coupling faithfulness + predictor generality + force∝∇ (`A2_PASS`)

`jax_scout/gravity_A2_predictor_coupling.py`:
- **P1 coupling faithfulness (PASS):** λ=0 → no spurious self-force (|P_A|=6e-9); λ=1 → **exact equal-and-opposite
  momentum transfer** (P_A=−236.2, P_B=+236.2, |P_A+P_B|=1.2e-8); A↔B exchange exact; global-phase covariance exact.
  The coupling `κ|ψ_B|²ψ_A` (= probe responding to the load's density) is the natural gravity coupling.
- **P2 predictor generality (PASS, marginal):** over 24 configs (sep×phase×amplitude), `I_int` is the best predictor
  of the independent |P_A(T)| — Spearman **0.942** (held-out 0.952) vs overlap 0.916, separation 0.769, cross-energy
  0.013. Honest caveat: `I_int` is an **overlap-family** quantity (weighted `∫ρ_B²ρ_A`), so it *edges out* plain
  overlap rather than dominating — it wins on held-out phases but is not in a different class.
- **P3 force∝∇ (PASS — now quantitative, A.2b):** the force is directional (symmetric env → ~0 despite I_int>0;
  reflected env → sign reverses). The apparent "100× magnitude mismatch" was a **normalization bug** — `momentum_x`
  returned a bare grid sum, `integ` includes `dV=(L/N)³` (1/dV=110.6). With a dV-consistent momentum, the NLS
  **Ehrenfest law `dP_A/dt = −κ∫ρ_A∇ρ_B` holds quantitatively** (F_FD vs F_analytic rel-err 2e-5 under dt refine).
  **Correction:** the force uses **`∇ρ_B`** (the coupling-potential gradient), **not `∇I_int`** — the reflection null
  passes for many gradient observables and does not select `∇I_int`.

- **A.2b second target family (PASS):** for a Gaussian target (different morphology), `I_int` (Spearman 0.990) still
  beats overlap (0.963) and separation (0.898) — the predictor advantage is not profile-specific.

**Scope split (reviewer, adopted):**
- **A-P (probe–environment interaction): CLOSED** as `EXPLICIT_PROBE_ENVIRONMENT_INTERACTION_LOAD_VALIDATED`.
- **A-S (single-field self-sourcing): OPEN** — no decomposition-free local source established.
- The harness coupling `κρ_B ψ_A` is **density coupling** used as a controlled harness — **not** "the natural IRER
  gravity coupling"; `I_int` is the load *diagnostic* computed from it. Whether the lapse should be sourced by
  `I_int` vs plain density/overlap is the **open question Audit B's overlap-control tests.**

**Mechanism decision (Jake, 2026-07-12): H-G− (throttling).** Reason: a higher node interaction field means a larger
(exponentially growing) interaction "lightcone", so resolution cost per event rises and completed throughput falls —
`R_res = K/(1+β·I_int)`, `β>0` (finite local resolution capacity). Recorded as the chosen **hypothesis** (its
consequences are what Audit C would later test); the alternative H-G+ (interaction accelerates chronology) is
rejected by Jake's mechanism.

**Firmly NOT claimed:** that `I_int` *equals* the completed-resolution rate `R_res`. `I_int` is interaction
*intensity*; the predictive test shows the **force ∝ ∇I_int** (not `I_int` itself), so force/clock effects run
through the *gradient* of the rate and the still-undetermined map `R_res = G(I_int, ψ, E)` (Audit B). No
metric/clock/force/gravity claim.

Read-only field-algebra diagnostic (`jax_scout/gravity_A_rint_diagnostic.py`); no production/solver/Hunter/config
change; gravity sector PAUSED.

## What was measured (and what it is / isn't)

On the true-flat NLS substrate the linear `D∇²` is additive and cancels in `F_full − F_self`, so the interaction
residual is exactly the nonlinear cross-term:
```
ΔF_int = g(ρ_full)ψ_full − g(ρ_L)ψ_L − g(ρ_R)ψ_R ,   g(ρ)=aρ+sρ²+fρ³
I_int   = ∫ |ΔF_int|² dV          # interaction INTENSITY / power, units (field/time)²  — NOT a rate
γ_int   = ‖ΔF_int‖ / ‖ψ‖          # a rate-scale candidate, units 1/time
```
**Crucial distinction (do not conflate):** `I_int` is interaction *intensity*, not a rate, and *not* the completed
resolution rate. The theory's causal chain is three-layered and only the first link is measured here:
```
I_int  (measured)  →  R_res = G(I_int, ψ, E)  (completed-resolution rate — UNDETERMINED)  →  N = R_res/R_ref  (lapse)  →  dτ = N·dt
```
Audit A validated a candidate `I_int`. It has **not** determined `G`, and therefore says nothing yet about the
clock direction. (The clock question "slower or faster?" is answered by `G`, not by picking a sign: if `R_res`
literally means completed resolutions per reference time, higher `R_res` = *faster* clock; gravitational *slowing*
from stronger interaction requires either `G` decreasing or `I_int` acting as an interaction *load*.)

## Results (N=96, L=20; validated C2 solitons)

| test | result | status |
|---|---|---|
| **A1 distance/isolation** | `I_int` → 3.9e-10 at max separation, rises monotonically on approach (sep 4 → 2.33, sep 2 → 392) | ✅ implementation check |
| **A2 excludes self-activity** | self-activity 3.78 vs `I_int` 5.2e-4 at distance (frac 1.4e-4): a breathing *isolated* node has `|∂_tψ|²>0` but `I_int≈0` | ✅ |
| **A3a isolated-density** | isolated soliton `I_int = exactly 0` at any amplitude (peak ρ 0.98/2.2/3.9 → 0/0/0) — density alone cannot *produce* `I_int` | ✅ (but partly by construction) |
| **A4 determinism/convergence** | bit-identical on repeat; grid 96→128 (spectral placement) rel-err 3e-11 | ✅ |
| **A5 naming** | `I_int`=2.33 (intensity) vs `γ_int`=0.137 (rate); **local field saved** (a clock/metric needs the field, not a global norm) | recorded |
| **A6 decomposition robustness** | `I_int` ordering (grows on approach) preserved under **both** spectral and spatial-mask decompositions | ✅ not representation-dependent |
| **A7 fixed-total-mass phase** | at held ∫ρ, `I_int` still varies 0.48/0.17/1.00 across Δφ (spread 0.83) → carries info beyond gross density | partial (see caveat) |

*(Periodic-box: separation folds as `min(sep, L−sep)`; the true isolated limit is sep=10.)*

## Honest limitations (reviewer, accepted)

- **Some passes are partly by construction.** A3a (isolated → 0) follows because `N(ψ_L+0)−N(ψ_L)−N(0)=0`; large
  separation → 0 because localized fields stop overlapping. These are valid *implementation* checks, not proof that
  the residual is the physical resolution rate.
- **Intensity, not rate** (see above). Naming corrected.
- **Density control is only partial.** A3a shows density can't *produce* `I_int`; A7 holds *total* mass but not the
  *pointwise* density profile. A fully matched-**pointwise**-density, different-coupling control is still owed — it
  is the decisive density-vs-relation discriminator.
- **Decomposition is unique only for prepared pairs / early interaction.** During violent overlap/capture (identical
  solitons) the `ψ_L,ψ_R` split is ambiguous; A6 shows robustness of *ordering* for prepared pairs, not a unique
  local rate through a collision.

## A.1 completion (done — the decisive controls)

**Matched-density controls** (`jax_scout/gravity_A1_matched_density.py`, `MATCHED_DENSITY_CONTROLS_PASS`):
- **B1 source-blindness** — the *same* pointwise density (max|Δρ|=0, and `Ω²(ρ)` identical, diff=0) is realized by an
  interacting pair (`I_int=2.33`) and a single coherent blob (`I_int=0`). A density-sourced `Ω²(ρ)` assigns both the
  identical geometry despite different interaction → **density cannot be the interaction source.**
- **B2 local multivaluedness** — for a genuine interacting pair, `I_int(x)` is *not* a single-valued function of
  `ρ(x)`: the median within-ρ-bin coefficient of variation of `I_int` is **3.29** (even the densest bin, 2.49). Same
  density → a *range* of `I_int`. This is the physical (fixed soliton-profile) decomposition, **not** by construction.

**Causal-ordering ablation** (`jax_scout/gravity_A1_causal_ablation.py`) — **DOWNGRADED to a short-time consistency
check.** The divergence `div(t)=ψ_full−[ψ_L^iso+ψ_R^iso]` aligns 1.0000 with `ΔF_int(0)` at small t — but this is a
**Taylor identity** (`div(t)=t·ΔF_int(0)+O(t²)` by construction), so it validates the solver/implementation, **not**
independent causality. "Divergence before COM motion" is likewise automatic (COM is an integrated macroscopic
quantity). Retained only as `SHORT_TIME_COUNTERFACTUAL_CONSISTENCY_PASS`.

**Decomposition-invariance — explicit two-subsystem coupling-ablation** (`jax_scout/gravity_A1_mirror_ablation.py`,
`MIRROR_ABLATION_DECOMPOSITION_INVARIANT_PASS`): replacing the analyst-chosen single-field split with explicit
subsystems A (target) + B (environment) and a coupling switch λ. Interaction togglable without redefining the field
(T1: λ=0→0, λ=1→0.71); **same target density, interaction varies only with the environment** (T2: ρ_A fixed, env
moved → I_int 10.0→3e-11); **exact invariance** under global phase (0.0), translation (2e-12), relabel A↔B (2e-16),
grid refine (2e-12); tracks environment amplitude at fixed ρ_A (T4). This resolves the decomposition-dependence
critique — and is the natural model for a probe-in-a-load (a physically-defined decomposition).

**Predictive validity beyond the identity** (`jax_scout/gravity_A1_predictive.py`, `PREDICTIVE_VALIDITY_PARTIAL`):
evolving the coupled two-field model, `I_int^A(0)` predicts the **independent** later observable `|P_A(T)|`
(momentum imparted to A) with **log-log correlation 0.967**; because ρ_A(0) is identical across configs, every
target-only predictor (density, gradient energy, nonlinear energy) has zero variance and predicts nothing. `PARTIAL`
because `|P_A(T)|` peaks at intermediate distance, not at max `I_int` — revealing that the **force ∝ ∇I_int** (a
gradient), while `I_int` is the scalar intensity/load. This is the reviewer's intensity-vs-force distinction shown
empirically, and it sharpens Audit B (force/clock effects use the *gradient* of the rate).

## Consequences for Audit B / C (guardrails)

- **B must determine `G` (and hence `N`) independently of the clock result.** Minimal admissible form:
  `N = 1 + α·(rate) + O(²)`, background `N=1`, `α`'s **sign fixed by Jake's intended mechanism** (not fitted
  per-run), bounded/non-clipped, with an explicit falsification condition.
- **C must not be tautological.** A clock `Θ=∫ω·N·dt` trivially differs wherever `N` differs — that validates
  wiring. The oscillator must couple through the *same dynamical law* `G`, and the decisive test is
  **equal-pointwise-ρ / unequal-`I_int`** giving different clocks.

## Bottom line

A genuine advance: a clean, reproducible, decomposition-robust measure of nonlinear relational coupling that
vanishes without a partner and responds to separation and phase. It is a strong candidate `I_int`. It is **not yet**
the completed-resolution rate; that requires the matched-density + causal-ordering controls above and, for the
clock, an independently-specified `G`.

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `25020bf` (2026-07-22) — *dual substrate*
**Revised since:** 6 commit(s), most recently `f2b0527` (2026-08-27)

**Harness code changed since it was written:** 5 commit(s) to `jax_scout/`.
  - `06edae5` 2026-08-27 — Fix run provenance: 30% -> 86% of runs now dated
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine
  - `25020bf` 2026-07-22 — dual substrate

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
