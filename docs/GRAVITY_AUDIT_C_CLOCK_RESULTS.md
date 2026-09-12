# IRER Gravity Audit C — First Clock Experiment (throttling-lapse consistency)

**Verdict:** `C_CLOCK_FIRST_PASS` — a first, **static** (profile-weighted effective-rate) consistency check of the
H-G− throttling lapse `N=1/(1+β·Î_int)`. It is an internal-consistency + source-organization check, **not** a
time-dilation proof: H-G− remains a postulate under test, and coupling a clock through `N` makes it slow *by
construction*. Standalone mirror; production gravity ladder CLOSED; no matter/gravity/time-dilation claim.

Reviewer guards built in: **global preregistered scale `I_*`** (fixed reference config, not per-run); **weak-field
β ladder** (target `N_min ∈ {1, 0.95, 0.8, 0.5}`); **matched-strength overlap control** (same `N_min`, so any
difference is spatial *organization*); objective-vs-relational fork measured, not assumed.

## Results

| N_min | β | relational N/W | objective N/W | I_int vs matched-overlap (narrow) |
|---|---|---|---|---|
| 1.00 | 0 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 (null) |
| 0.95 | 0.042 | 0.989 / 0.994 | 0.982 / 0.993 | 0.989 / 0.986 |
| 0.80 | 0.202 | 0.951 / 0.973 | 0.922 / 0.968 | 0.951 / 0.938 |
| 0.50 | 0.807 | 0.846 / 0.916 | 0.766 / 0.901 | 0.846 / **0.810** |

## What passed

- **β=0 null exact; slowing monotone with β.** The lapse couples consistently and reduces to no-effect at β=0.
- **`I_int` is distinguishable from a *matched-strength* overlap** (the decisive source test): at N_min=0.5, both
  fields share the same minimum, yet the I_int-sourced clock (0.846) differs from the overlap-sourced (0.810) —
  gap ~4%, growing with β. Interpretable: `I_int ∝ ρ_B²` concentrates the throttling more tightly than overlap
  `∝ ρ_B`. So it is **not "overlap in disguise"** at the clock level — though the modest margin is consistent with
  its being an overlap-*family* quantity.

## What did NOT work (honest limitation, flagged)

- **The objective-vs-relational "universality" test is confounded by clock EXTENT.** Under the *objective*
  (env-only) lapse — a single shared field — the narrow and wide clocks still read different rates (0.766 vs 0.901,
  gap 0.135) simply because an extended clock spatially averages a varying `N` differently than a compact one. So
  clock "universality" (`ω₁'/ω₁ ≈ ω₂'/ω₂`) is only meaningful for clocks **small compared to the lapse's variation
  scale**; the Gaussian clocks here are not, and the test cannot separate objective from relational as built.

## Open decisions / next refinements

1. **Lapse type (theory call for Jake): objective (B-O) vs relational (B-R).** `I_int ∝ ρ_A·ρ_B²` contains the probe
   density, so under a relational source each clock has its own `N` (probe-specific chronology — fits the observer-loop
   framing and predicts *non-universal* clock rates, a distinctive IRER signature vs GR). Under an objective source,
   the environment defines one common `N` (GR-like universality). The experiment must fix which before "universality"
   is even the right criterion.
2. **Refine the universality test** with point-like (sub-lapse-scale) clocks or center-referenced rates, so extent
   no longer confounds it.
3. **Dynamical clocks** (proper-time-coupled `D_τ=(1/N)∂_t`), replacing the static effective-rate proxy, for a
   genuine (still consistency-level) test.

## Audit C.1 — B-R relational clocks + B-O emergence probe (frozen stage)

Theory frame (Jake, decided): **B-R fundamental** (`N_{A|E}=1/(1+β·Î_{A|E})`, `dτ_A=N_{A|E}dt`), **B-O an emergent
weak-probe limit** — tested, not assumed. (`jax_scout/gravity_C1_relational_clocks.py`, `C1_FROZEN_PASS`.)

- **Clock-response consistency (NOT source-layer emergence) — `SHARED_FROZEN_LAPSE_CLOCK_CONSISTENCY`:** two
  structurally different mechanisms driven by the **same** frozen lapse slow by identical fractions (gap ~1e-5). This
  shows the time operator is implemented consistently across 1st/2nd-order dynamics — **necessary but expected**,
  since both were *given* the same N. It does **not** show an objective lapse *emerged*: that requires each clock to
  generate its **own** `N_{A|E}` from its **own** interaction and those to converge (the source-layer test — C.2).
- **`I_int` ≈ overlap in THIS configuration (config-specific degeneracy, not proven general):** with the
  **integrated lapse deficit** `∫(1−N)dV` matched, the I_int-vs-overlap clock gap collapses to **~0.2%** (vs ~4%
  under the weaker N_min-only match). Correct framing: *for this source profile, clock envelope, and single position*
  the two sources are **observationally degenerate** — a compact clock reads a weighted average, so spatially
  different fields can read alike. This does **not** prove the fields are equivalent (C.2 maps where they differ).
  Retracted: "env-only source is more honest because it differs ~5%" — a larger clock shift only means *easier to
  distinguish*, not truer; source choice is judged by conceptual correspondence/invariance/prediction, not shift size.

**Retracted (premature):** "B-R predicts non-universal clock rates, a departure from GR." Probe dependence can come
from extent/normalization/coupling/overlap/backreaction; only a difference persisting after matching all of those
would support non-universality. Not claimed.

**Next stage:** backreaction (clock evolution changes its own `I_{A|E}` → its own lapse — the recursive
observer–environment loop), and whether the near-equivalence with overlap changes under backreaction or in the KG
substrate. Plus: whether an env-only (objective-source) construction is the more honest lapse source given `I_int`'s
near-overlap-equivalence.

## Audit C.2 — source-discrimination map + real (source-layer) emergence test

`jax_scout/gravity_C2_source_map.py`:
- **Source map: GENERAL degeneracy.** `Δr(I_int, overlap) ≤ 0.0017` across *both* environments (soliton, blob), all
  positions (0.6–3.0), both clock widths (0.4, 0.8), at matched integrated deficit. So `I_int ≈ overlap` for the
  clock is **general, not config-specific** — overlap is an adequate reduced source; `I_int` adds **no clock-level
  structure** over plain overlap anywhere tested.
- **B-O does NOT cleanly emerge (and exposes a deeper issue).** Each clock generating its *own* `N_{A|E}`: as
  normalized clocks shrink (width 1.6→0.3), own-N *diverges* (0.976→0.248) — because `I_{A|E} ∝ ρ_A·ρ_E²` contains
  the probe, so self-concentration drives self-throttling; conversely a *weak* test probe (ρ_A→0) feels **no lapse
  at all**, the opposite of GR-like universal time dilation. Caveat: this is entangled with the ∫ρ=1 normalization,
  so it is a *caution* about the mutual-`I_int` source, not a clean non-universality claim.

## Audit C.3 — gradual backreaction (recursive observer–environment loop)

`jax_scout/gravity_C3_backreaction.py`, `C3_BACKREACTION_STABLE`: the recursive loop (clock amplitude → own load →
own lapse → clock) is **well-behaved** — returns to frozen at λ_br=0, energy-conserving (drift ~1e-11), bounded (no
runaway), dt-convergent. As backreaction turns on (λ 0→1) the clock frequency *rises* (0.800→0.889): the loop
**partially self-relieves** the throttling (the clock spends time at low load where N≈1). A real, stable recursive
effect — but nothing `I_int`-specific.

## Where the gravity thread stands (honest overall)

The re-entry investigation was methodologically sound and self-correcting, but the **specific `I_int`-based B-R
gravity mechanism is coming up weak**:
- `I_int` is **generally clock-degenerate with plain overlap** (C.2) — the elaborate interaction-load observable
  adds no gravity-relevant structure over a simple overlap for the lapse.
- The **mutual-`I_int` relational lapse does not reproduce GR-like universal time dilation** — the effect scales with
  the probe's own strength, so weak test clocks feel nothing (C.2).
- Backreaction is stable but not distinctive (C.3).

What still stands: the **source-characterization method** (A/A.1/A.2 — a lawful, decomposition-invariant,
predictive load observable), the **naturally-bounded throttling lapse** (no cliff, unlike production `Ω²(ρ)`), the
**determinism framing**, and the **discipline**. What is now doubtful: that `I_int`-sourced geometry is a better
gravity mechanism than plain density/overlap, and that a mutual-relational lapse gives gravitational time dilation.
Options for a future pass: (a) an **environment-only objective source** `S_B(x)` (the gravity-natural choice — the
load creates geometry, probes navigate it), tested for universal dilation; (b) move the causal-cone / lightcone
question to the **KG substrate**; or (c) accept a characterized negative for the `I_int`/B-R path. Production gravity
ladder remains CLOSED throughout.

## Verdict discipline (binding)

A future clean C pass would support at most:
`THROTTLING_LAPSE_IMPLEMENTATION_INTERNALLY_CONSISTENT` and
`CLOCK_RESPONSE_DISTINGUISHES_INTERACTION_LOAD_FROM_MATCHED_OVERLAP`. It would **not** establish emergent gravity,
physical time dilation, or the correctness of H-G− in nature.

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

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
