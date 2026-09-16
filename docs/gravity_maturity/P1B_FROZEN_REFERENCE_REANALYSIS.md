---
tags: [result, gravity]
date: 2026-09-17
branch: Branch - Gravity - Index
status: partial
verdict: P1B_NORMALISATION_REMOVES_60PCT_OF_THE_LOADS_MASS_DEPENDENCE__MASS_AXIS_SHOWS_NO_THRESHOLD
---

# P1-b (reanalysis) — What the Normalisation Removes, Measured

Author: Claude (primary), 2026-09-17. Plan item **P1-b**, promoted above **D1** by
[[S1_STATIC_TG_GREENS_FUNCTION|S1]].
Derived from the existing [[../runs/TG_B2_CHARGE_AUDIT_20260719_005326]] rows — **no new simulation**,
in the same style as [[TG_P1A_ENERGY_OBSERVABLE|P1-a]].

Observation-only. **No gravity / UFF / IRER claim.**

> [!abstract] Verdict
> `P1B_NORMALISATION_REMOVES_60PCT_OF_THE_LOADS_MASS_DEPENDENCE__MASS_AXIS_SHOWS_NO_THRESHOLD`

> [!warning] This is a reanalysis, not the run
> P1-b proper is *"one mass sweep with frozen `e_ref`/`q_ref` beside the per-`w` recomputation"*. What
> follows **predicts** its result from linearity rather than measuring it, and the prediction is only
> as good as that linearity. It is worth having because it is free and because it says whether the
> run is worth its cost — it does not replace it.

---

## 1. The chain, measured end to end

Across a **2.74× range in M**, log-log exponents against M:

| quantity | exponent | range | what it is |
|---|---:|---:|---|
| `M` | +1.0000 | 2.74× | the mass |
| raw load (= `∫S × e_ref`) | **+0.945** | — | the **physical** load, before normalisation |
| `e_ref` | +0.3605 | 1.54× | the normaliser — *it grows with M* |
| `∫S_state` | +0.5847 | 1.91× | the load **as the geometry sees it** |
| \|`G_min`\| | **+0.3908** | 1.47× | the mediator's response |
| \|`F_R`\| | **−0.0673** | 1.22× | the force |

**The physical load grows as M⁰·⁹⁵. The geometry's response grows as M⁰·³⁹.** About **60% of the
load's mass dependence is removed before it reaches the geometry at all**, in two stages:

- **≈0.36 by the peak normalisation.** `e_ref`/`q_ref` are `max|energy|` and `max(charge)` of *that
  configuration's own initial state* ([[TG_P1_EVIDENCE_RECONCILIATION|P1 §2]]). A heavier node is
  divided by its own larger peak. This is P1's finding, confirmed here by the exact bookkeeping:
  0.5847 + 0.3605 = **0.945**, the raw exponent.
- **≈0.19 more between `∫S` and `G_min`.** The source spreads as it grows (`source_rms_width` 2.96 →
  3.55), so a larger integral produces a proportionally smaller peak.

## 2. Two checks on S1's linear-response picture

**A responds to G exactly linearly.** `(A_well_min − 1)/G_min` across all five rows:

```
0.059998  0.059998  0.059999  0.059999  0.059999
```

against ε_G = **0.06**. Five-digit agreement — the linearisation
[[S2_V7_ACOUSTIC_METRIC_RESULTS|S2]] and S1 both rest on is exact at these amplitudes.

**`G_min` is not proportional to `∫S`** — the ratio varies 28%. That is *not* a failure of linearity,
and the reason matters: S1 gives the mediator ranges as **0.875 and 0.484**, while the source is
**~3 wide**. The mediator is far shorter-ranged than its source, so G tracks S **locally** rather than
integrating it, and the peak of G follows the peak of S, not its integral. A source-width model
using `source_rms_width` does not account for the spread either (R² = 0.03), which says `rms width`
is too crude a shape descriptor at this ratio of scales — not that the map is nonlinear.

## 3. What P1-b should find

Since the T/G map is linear and A is linear in G, `F_R ∝ S_state`. Freezing the normalisers is then a
per-row rescaling, and the result can be predicted:

| | per-row normalisers (as run) | **frozen `e_ref`** |
|---|---:|---:|
| `∫S_state` exponent vs M | +0.585 | **+0.945** |
| \|`F_R`\| exponent vs M | **−0.067** | **+0.293** |

> [!important] The qualitative reading of the mass axis changes
> With per-row normalisers the force is **flat in mass** (−0.067) — the reading the record carries.
> With frozen references it **grows** with mass (+0.293). Those are different physical claims, and
> the difference is a normalisation choice rather than a measurement.
>
> Neither is M¹. A body force proportional to mass is what a gravity-like reading would want, and
> **+0.29 is not that either** — so freezing the reference does not rescue the mass scaling, it just
> stops the flatness being an artefact of the normaliser.

The estimate assumes `e_ref` and `q_ref` rescale together, which holds here: their exponents are
**0.3605** and **0.3694**.

## 4. The consequence for the saturation cliff and for D1

S1 established structurally that the load→geometry map is **exactly linear**, so it can contain no
threshold. This dataset adds the empirical half: **across a 2.74× mass range the response is a smooth
power law with no threshold anywhere** — every quantity above is a clean log-log line.

So on the mass axis there is **no cliff to measure**. If the cliff is real it lives on a different
axis (the load-capacity/yield sweeps), and by S1's argument it still cannot be in the T→G chain.

> [!danger] D1 should not run yet
> **D1** proposes the cliff's critical exponent as the primary discriminating quantity. Before that is
> worth compute, the cliff has to be shown to be physics rather than the peak normalisation
> saturating by construction. The mass axis — the one axis with a clean five-row dataset — shows no
> threshold at all, which is consistent with the cliff being a feature of how load is *defined*
> rather than how geometry *responds*.
>
> **The real P1-b run is now the cheapest way to settle this**, and it decides whether D1 measures
> physics or a normalisation choice.

---

## Issues raised

- The suppression between `∫S` and `G_min` (≈0.19 in the exponent) is attributed here to the source
  spreading, but `source_rms_width` does not fit it. A proper shape diagnostic — the source's overlap
  with the mediator's Green's function — would settle it and is cheap.
- `S0_AUTHORITATIVE = 135.686` is a frozen global scale applied on top of the per-configuration
  normalisers. It does not affect exponents, but it means `S_state` is not O(1) and the phrase
  "state load" does not carry an obvious physical unit.

## Previous experiments

- [[TG_P1_EVIDENCE_RECONCILIATION]] — §2 found the per-`w` peak normalisers
- [[TG_P1A_ENERGY_OBSERVABLE]] — the same no-new-simulation method
- [[S1_STATIC_TG_GREENS_FUNCTION]] — the linearity this reanalysis depends on

## Associated docs

- [[../IRER_MASTER_HYPOTHESIS_CATALOG]] · [[../INTEGRATED_PLAN_2026-09]]
- [[../runs/TG_B2_CHARGE_AUDIT_20260719_005326]]

## Branches

- [[../Branch - Gravity - Index]]
