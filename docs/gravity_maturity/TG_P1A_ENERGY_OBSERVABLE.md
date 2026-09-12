---
tags: [result, gravity]
date: 2026-08-25
branch: Branch - Gravity - Index
status: complete
verdict: TG_P1A_FORCE_COUPLES_TO_4PCT_OF_NODE_ENERGY__COUPLING_FRACTION_VARIES_29PCT_ACROSS_MASS_AXIS
---

# TG-P1a — Per-Half-Space Energy Observable

Author: Claude (primary), 2026-08-25. Campaign item **P1-a**, the last blocker on the P1 checklist.
Implemented in `jax_scout/gravity_TG_B2_midplane_stress_flux.py` (`stress_diag` + `energetics`);
mass-axis analysis derived from the existing [[../runs/TG_B2_CHARGE_AUDIT_20260719_005326]] rows.

Mirror-only, observation-only. **No gravity / UFF / IRER claim.**

> [!abstract] Verdict
> `TG_P1A_FORCE_COUPLES_TO_4PCT_OF_NODE_ENERGY__COUPLING_FRACTION_VARIES_29PCT_ACROSS_MASS_AXIS`

---

## 1. What was missing

[[TG_P1_EVIDENCE_RECONCILIATION|P1]] found that `F_R` weights the **gradient energy only** — one of
four terms in the KG energy density — while the only available "mass" observable is the charge-like
`∫ρ dV`. So `F/M_p` is not an acceleration, and **the harness recorded no energy observable at all**.
P1-a adds one.

## 2. Implementation

The KG energy density for this Lagrangian is

$$e = |\pi|^2 + c^2 A|\nabla\phi|^2 + m^2\rho - U(\rho)$$

(the `A=1` case matches `phase_d_c3_wave.py:80`). The observation module now reports, per half-space:
`E_R`, `E_L`, and the split `E_kin_R`, `E_grad_R`, `E_mass_R`, `E_pot_R`, plus `M_R = ∫ρ dV`, and
derives `F/E_R`, `F/M_R` and the **gradient fraction** `E_grad/E`.

## 3. The force couples to 4.5% of the node's energy

Settled-window decomposition (N=48, L=12, sep 3.0, well arm):

| term | value | share of `E_R` |
|---|---:|---:|
| `E_kin_R` (`\|π\|²`) | 78.15 | 48.6% |
| `E_mass_R` (`m²ρ`) | 87.71 | 54.6% |
| `E_pot_R` (`−U`) | −12.36 | −7.7% |
| **`E_grad_R`** (`c²A\|∇φ\|²`) | **7.28** | **4.5%** |
| **`E_R`** total | **160.78** | 100% |

**`F_R` can couple to 4.5% of the half-space energy. The other 95.5% is invisible to it.**
P1 said "one of four terms"; the term is one twenty-second of the total.

Two incidental clarifications:
- `M_R ≡ E_mass_R` exactly (since `m = 1`), so the historical `F/M_p` was force divided by the
  **mass-energy term alone** — 54.6% of the total, not the total.
- `F/E_R` is now available: −5.502e-07 against `F/M_R` = −1.011e-06, differing by the factor
  `E_R/M_R = 1.838`.

## 4. The coupling fraction varies across the mass axis — P1's diagnosis, quantified

Computed from the existing charge-audit rows (`E_grad = c²∫|∇φ|²`, both already recorded), so this
needed **no new simulation**:

| `w` | M | `E_tot` | `E_grad` | **grad fraction** |
|---|---:|---:|---:|---:|
| 0.945 | 105.93 | 195.10 | 8.794 | **0.04507** |
| 0.955 | 69.92 | 131.81 | 6.362 | **0.04827** |
| 0.964 | 52.06 | 100.00 | 4.801 | **0.04800** |
| 0.972 | 42.83 | 83.45 | 3.690 | **0.04423** |
| 0.980 | 38.68 | 76.17 | 2.696 | **0.03540** |

**Range 0.0354 → 0.0483: a 29.1% variation, and NON-MONOTONIC in M** (it peaks near M ≈ 70).

This is the direct quantitative confirmation of P1's diagnosis. The fraction of a node's energy that
the force estimator can couple to is **not a constant** across the mass sweep — it varies by nearly a
third, and not even monotonically. A "force versus mass" curve measured this way was never measuring
a clean mass dependence, independent of every other confound P1 identified.

## 5. What `F/E` does and does not buy

| quantity | exponent vs M | R² |
|---|---:|---:|
| `F` (raw) | −0.067 | 0.111 |
| **`F/E_tot`** | **−1.002** | **0.965** |
| `F/M` | −1.067 | 0.969 |
| `E_tot` | +0.935 | **1.0000** |
| `1/E_tot` | −0.935 | 1.0000 |
| grad fraction | +0.164 | 0.278 |

> [!warning] The R² jump from 0.111 to 0.965 is NOT independent evidence
> `E_tot` is a near-perfect power law in M (R² = 1.0000) and `F` is approximately flat. Therefore
> `F/E ≈ const/E` inherits `E`'s smoothness by construction. Compare R²(`F/E`) = 0.965 against
> R²(`1/E`) = 1.0000: the fit quality comes from the denominator, not from `F/E` being a more
> fundamental quantity. **Do not report `F/E ∝ M^-1.00, R²=0.97` as a discovered scaling law.**
>
> The honest content remains P1's: `F` is flat to ±10% across a 2.74× mass range, on an axis
> confounded with morphology.

`F/E_R` is nevertheless worth having: it is the only **dimensionally coherent** acceleration-like
ratio in the sector, and it is now recorded rather than reconstructed after the fact.

## 6. Status of the P1 checklist

| P1 item | status |
|---|---|
| audit fixed-vs-recomputed `e_ref`/`q_ref` | DONE (P1 §2) |
| classify what `F_R_well` actually is | DONE (P1 §1) |
| expose per-row source integral, mediator amplitude/gradient, raw force | DONE (P1 §2–5) |
| **per-row probe susceptibility, `F/M_p`, energy normalization** | **DONE — this document** |
| per-row uncertainty | **PARTIAL** — the dynamic module now reports sem and σ per arm ([[TG_P2_MIDPLANE_STRESS_FLUX_RESULTS|P2]]); the static mass rows still carry no error bars |

**P1 is closed except for error bars on the static mass rows.**

## 7. Next

- The gradient fraction should be reported alongside every future force row, as a standing covariate.
- **P1-b** (frozen-reference mass sweep) is now better motivated: with the coupling fraction varying
  29%, separating the normalization offset from morphology is the only way to get an interpretable
  mass exponent.
- **GAP-4** remains the sharpest open problem: the force's sign is still an input.

---

## Previous experiments

- [[TG_P1_EVIDENCE_RECONCILIATION]] — the audit that specified this item
- [[TG_P2_MIDPLANE_STRESS_FLUX_RESULTS]] — the independent estimator, same module

## Associated docs

- [[../IRER_MASTER_HYPOTHESIS_CATALOG]] · [[../RUN_QUEUE]]
- [[../runs/TG_B2_CHARGE_AUDIT_20260719_005326]] · [[../runs/TG_B2_P1A_SMOKE]]

## Next experiment

- [[TG_P1B_FROZEN_REF_MASS_SWEEP]]

## Branches

- [[../Branch - Gravity - Index]]

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `b4e0613` (2026-08-25) — *P1-a: per-half-space energy observable closes the P1 checklist*
**Revised since:** 11 commit(s), most recently `bd93089` (2026-09-12)

**Harness code changed since it was written:** 5 commit(s) to `jax_scout/`.
  - `caf61af` 2026-09-10 — Build HUD items 1+2: snapshot writer and universal field renderer
  - `06edae5` 2026-08-27 — Fix run provenance: 30% -> 86% of runs now dated
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Also referenced by (same date or earlier):** [[ACTION_PLAN_2026-08]], [[IRER_MASTER_HYPOTHESIS_CATALOG]], [[RUN_QUEUE]], [[SESSION_SYNTHESIS_2026-08]], [[gravity_maturity/TG_P2_MIDPLANE_STRESS_FLUX_RESULTS]]

**The master catalog references this document** — the catalog is the authority on whether its verdict is still live:

> | **TG P1a — energy observable** | **CLOSED (2026-08-25)** | `docs/gravity_maturity/TG_P1A_ENERGY_OBSERVABLE.md`. Per-half-space KG energy decomposition added to the observation module. **(a)** `F_R` couples to **4.5% of the node's half-space energy** (E_grad 4.5%, E_kin 48.6%, E_mass 54.6%, E_pot −…

<!-- LINEAGE:END -->
