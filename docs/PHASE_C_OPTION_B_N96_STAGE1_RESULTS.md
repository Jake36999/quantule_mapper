# Phase C Option B — N96 / T6000 Stage 1 Validation Results

**Date:** 2026-06-24
**Plan:** [PHASE_C_OPTION_B_N96_VALIDATION_PLAN.md](PHASE_C_OPTION_B_N96_VALIDATION_PLAN.md)
**Physics gate:** [MATHS_VERIFICATION.md](MATHS_VERIFICATION.md) — `PHYSICS_STATE_FROZEN` at commit `e8d6a78ea`.
**Output root (not committed):** `sweep_runs/PHASE_C_OPTION_B_N96_STAGE1_20260624_223147/`
**Data:** [phase_c_option_b_n96_stage1_results.csv](phase_c_option_b_n96_stage1_results.csv) · per-case `summary.json` + `probe_data.npz` + `stage1_manifest.json`.

**Run config:** N=96, T=6000, **FP64 (complex128)**, verdict-first (no trace capture), one seed per
exact shortlisted row, sequential. Total wallclock **0.80 h** for 8 runs (≈6 min/run). All 8 completed;
no early stop. Every cross-resolution replay carried an explicit `--target-initial-mass-override`
(raw×8) and auto-stamped `mass_scaling_mode=resolution_scaled_raw_target`; feb used `--ref` with
`exact_saved_raw_target`. `replay_resolution_N=96` and `metadata_ok` on all 8.

> Same classifier (`core_saturation_search.classify`), solver (`jax_scout.physics`), and geometry as
> N48 — only N and T changed. This is a resolution check of the N48 discovery verdicts, not a new search.

---

## Results — N48 → N96

| # | candidate | K | N96 raw (=raw₄₈×8) | N48 class | **N96 class** | n_fin | late slope | er_fin | er_max | survived? | label |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | K6 high-mass | 6 | 16402.35 | TRUE | **SPIN_DOWN_REJECT** | 4 | −9.19e‑5 | 0.438 | 0.992 | **NO** | `K6_HIGH_MASS_RESOLUTION_WEAKENS` |
| 2 | K6 mid-mass | 6 | 8000 | TRUE | **TRUE_SATURATED** | 6 | +2.62e‑5 | 0.849 | 0.992 | YES | `K6_MID_MASS_N96_SUPPORTED` |
| 3 | K4 intermediate | 4 | 9600 | TRUE | **TRUE_SATURATED** | 5 | −3.28e‑5 | 0.825 | 0.988 | YES | `K4_INTERMEDIATE_N96_SUPPORTED` |
| 4 | feb56dc7 control | 6 | — | TRUE | **TRUE_SATURATED** | 4 | −2.32e‑6 | 1.578 | 1.596 | YES | `FEB56DC7_CONTROL_REPRODUCED` |
| 5 | K2 intermediate | 2 | 16402.35 | TRUE | **TRUE_SATURATED** | 2 | −1.03e‑5 | 0.664 | 0.972 | YES | `K2_COMPACT_BRANCH_N96_SUPPORTED` |
| 6 | K1 low-mass | 1 | 8000 | TRUE | **LATE_BLOWUP_REJECT** | 0 | nan | nan | nan | **NO** | `K1_LOW_MASS_RESOLUTION_WEAKENS` |
| 7 | K1 high-mass failure | 1 | 16402.35 | BLOWUP | **LATE_BLOWUP_REJECT** | 0 | nan | nan | nan | YES (failure) | `K1_FAILURE_WALL_N96_SUPPORTED` |
| 8 | K6 near-threshold | 6 | 8000 | NEAR | **TRUE_SATURATED** | 5 | +1.23e‑4 | 1.876 | 1.876 | INCONCLUSIVE | `K6_NEAR_THRESHOLD_N96_INCONCLUSIVE` |

Morphology metrics (compactness / core radius / high-k fraction) are **n/a** at N96 — this was a
verdict-first pass (no trace capture), per the agreed runtime policy. Re-runs with `--trace-snaps` can
recover them for any case worth a closer look.

---

## Controls — both passed (results are trustworthy)

- **feb56dc7 reproduced** as `TRUE_SATURATED` (4 nodes, slope −2.3e‑6, er_fin 1.58) → the N96 classifier
  recognizes a known good bound state. `FEB56DC7_CONTROL_REPRODUCED`.
- **K1 high-mass failure reproduced** as `LATE_BLOWUP_REJECT` (delocalized growth) → the classifier still
  rejects the known failure at N96. `K1_FAILURE_WALL_N96_SUPPORTED`.

Both controls behaving correctly means the survivals and the flips below are signal, not classifier drift.

---

## Headline: the N48 morphology prediction did NOT cleanly hold

The N48 read (distributed = lower-risk, compact = higher-resolution-risk) predicted distributed branches
would survive and compact branches were the ones at risk. **Stage 1 contradicts that mapping in two of
the four decisive cases:**

- **K6 high-mass weakened** — the flagship *distributed* branch (low high-k ≈0.009), expected to be the
  most robust, lost energy at N96 (er_fin 0.62 → **0.44**, below the 0.5 TRUE floor) and reclassified as
  spin-down. The morphology proxy flagged it LOW risk; it did not hold.
- **K2 survived** — the *compact* branch (high high-k ≈0.022, compactness ≈111), flagged HIGH
  resolution-risk, reproduced a stable 2-node `TRUE_SATURATED` at N96. The proxy flagged it HIGH risk; it
  held anyway.
- Only **K1 low-mass** behaved as the proxy predicted: highest high-k (≈0.039) → **blew up** at N96.
- K6 mid-mass and K4 (both distributed, lower mass) held, consistent with the proxy.

So the high-k / compactness "resolution-risk" axis is **only partially predictive**. It correctly flagged
K1 low-mass and was consistent for the two lower-mass distributed cases, but it mis-ranked both K6
high-mass (false LOW) and K2 (false HIGH).

### Tentative alternative reading (observation, not a law)

The two failures sit at opposite extremes and fail in opposite directions:
- **K6 high-mass spun *down*** — 6 nodes sharing raw≈16402 (≈2700/node).
- **K1 low-mass blew *up*** — all of raw≈8000 in a single node.
- **K2 at the *same* highest mass as K6-high survived** — 2 nodes (≈8200/node), saturated cleanly.

This is consistent with a **per-node mass window** rather than a global morphology class: at fixed total
mass, more nodes ⇒ less mass per node ⇒ the high-mass K6 falls *below* the per-node saturation band and
bleeds energy, while K2 (few nodes, more mass each) stays inside it; too much mass in one node (K1)
overshoots into blowup. This is a hypothesis raised by 8 single-seed points — **not** established, and
explicitly **not** any topological/transition/proof/ground-state/universal claim. It would need its own
pre-registered test (e.g. vary node count at fixed total mass, or per-node mass at fixed K).

---

## Branch-family verdicts

- **Distributed branch family — PARTIALLY supported, not uniformly.** K6 mid-mass and K4 survive; K6
  high-mass does not. `OPTION_B_N96_STAGE1_DISTRIBUTED_BRANCHES_SUPPORTED` is **not** awarded (it would
  overstate a 2-of-3 result with the flagship case failing).
- **Compact branch family — does NOT uniformly weaken.** K2 survived; K1 low-mass blew up.
  `OPTION_B_N96_STAGE1_COMPACT_BRANCHES_WEAKEN` is **not** awarded.
- **Top-line: `OPTION_B_N96_STAGE1_INCONCLUSIVE`** — controls are valid, but branch survival does not
  follow the N48 morphology axis; the surviving set (K6-mid, K4, K2, feb) and the failing set (K6-high
  spin-down, K1-low blowup) cut across the distributed/compact split.

What *is* solid after Stage 1: **K6 mid-mass, K4, and K2 are resolution-supported single-seed bound
states at N96**, the feb anchor reproduced, and the K1 high-mass failure wall reproduced.

---

## Caveats

- **One seed per case.** These are single points; survival/failure could be seed-sensitive. Seed
  robustness is exactly what Stage 2 is for — do not read any single flip as definitive.
- **Verdict-first.** No N96 morphology metrics (compactness/core radius/high-k), so the resolution-risk
  *mechanism* is inferred from the N48 metrics, not measured at N96.
- **Borderline cases are near classifier thresholds.** K6 high-mass er_fin 0.44 sits just under the 0.5
  TRUE floor; K6 near-threshold is `TRUE` but with a **positive** late slope and `er_max == er_fin`
  (still rising at T=6000), i.e. not cleanly saturated — hence `INCONCLUSIVE`, not a clean upgrade.
- **No physics/classifier/threshold changes** were made to obtain these — the only code change this
  session was a one-line replay-harness serialization bug fix (see provenance note below); the PDE,
  solver, geometry, classifier, and search logic are untouched and frozen.

---

## Proposed next step (no launch without approval)

Stage 2 (seed expansion, 2 extra seeds within `{20260619,20260620,20260621}`) is justified for the cases
where a single seed is not enough to trust the flip:

1. **K6 high-mass** — is the spin-down seed-robust, or did this seed land just under the floor? (decisive
   for the per-node-mass reading)
2. **K6 near-threshold** — resolve the still-rising `INCONCLUSIVE`.
3. **K1 low-mass** — confirm the blowup is seed-robust (one seed already strongly suggests it).
4. Optionally re-run **K6 high-mass, K2, K6 mid-mass** with `--trace-snaps` to *measure* N96 morphology
   and test the per-node-mass hypothesis directly.

Confirm before launching Stage 2.
