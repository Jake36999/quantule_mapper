# Phase C Option B — Analysis

Disciplined read of the structured Option B discovery layer (`N=48`, `T=4000`, total-mass-controlled
ICs). This is **discovery evidence, not validation**. Source of truth:
`docs/phase_c_structured_discovery_B_summary.csv` (720 rows),
`runtime_logs/phase_c_structured_discovery_B_shortlist_metrics.json` (8 shortlist rows),
`docs/PHASE_C_STRUCTURED_DISCOVERY_OPTION_B.md`.

**Accepted framing:** the Option B structured discovery layer supports a structured mass/K-dependent
branch landscape at N=48/T=4000. K6 high-mass distributed branches remain the strongest robust branch
family. K4 appears to be a meaningful distributed/intermediate branch. K2 appears to be a real but
compact, higher-resolution-risk intermediate branch. K1 remains useful as both a fragile low-mass
survivor pocket and a robust high-mass failure control. No proof / topology / matter / ground-state /
black-hole / phase-transition language is warranted.

---

## Evidence

**Population (720 configs):** `94 TRUE`, `53 NEAR`, `338 BLOWUP`, `195 SPIN_DOWN`, `40 GROWER`.

**TRUE rate by K and raw target mass** (each cell `TRUE / 24` = 3 seeds × 8 rows):

| K | 500 | 800 | 1000 | 1200 | 1600 | 2050.293702 |
|---|---|---|---|---|---|---|
| 1 | 3/24 (12%) | 1/24 | 1/24 | 0 | 0 | 0 |
| 2 | 6/24 (25%) | 6/24 | 6/24 | 5/24 | 4/24 | 3/24 |
| 3 | 3/24 | 3/24 | 2/24 | 0 | 0 | 2/24 |
| 4 | 0 | 3/24 | 6/24 | 8/24 (33%) | 8/24 (33%) | 5/24 |
| 6 | 0 | 0 | 5/24 | 3/24 | 3/24 | 8/24 (33%) |

- **K1 low-mass survivor pocket:** TRUE only at low mass (3 at 500, 1 at 800, 1 at 1000), gone at
  1200+. Class mix is blowup-dominated even where TRUE exists (e.g. 500: 15 BLOWUP / 4 SPIN / 3 TRUE).
- **K1 high-mass failure wall:** 1200: 21 BLOWUP / 3 SPIN; 1600: 22/2; 2050: 23/1. Shortlist control
  `idx0` is `LATE_BLOWUP_REJECT` / `DELOCALIZED_GROWTH`, fails by `t≈100`.
- **K6 distributed branch:** no TRUE at 500/800; emerges at 1000 (5 TRUE, nodes {4,5,6×3}) and peaks
  at 2050 (8 TRUE, nodes {4×2,5×2,6×4}). Low-mass K6 cells are spin-down/near-dominated.
- **K2 and K4 intermediate branches:** K2 TRUE at every mass (6,6,6,5,4,3), final nodes mostly 1–2.
  K4 rises with mass (0,3,6,8,8,5), final nodes mostly 4 (some 2–3).
- **K3:** weaker / more scattered (3,3,2,0,0,2), more NEAR than TRUE at mid/high mass.

**Shortlist diagnostic morphology split** (metrics exist only for these 8 rows):

| case | K / raw mass | n_fin | high‑k frac | compactness | core_r | late slope | diagnostic |
|---|---|---|---|---|---|---|---|
| feb56dc7 (N96 anchor) | 6 / — | 4 | 0.0079 | 63.0 | 27.4 | −2.1e‑6 | SATURATED_BOUND_STATE |
| K6 high‑mass idx32 | 6 / 2050 | 5 | 0.0090 | 45.3 | 9.1 | −9.1e‑5 | SATURATED_BOUND_STATE |
| K6 mid‑mass idx34 | 6 / 1000 | 6 | 0.0083 | 31.8 | 18.6 | +3.5e‑5 | SATURATED_BOUND_STATE |
| K4 idx25 | 4 / 1200 | 4 | 0.0135 | 44.4 | 22.6 | −2.1e‑5 | SATURATED_BOUND_STATE |
| K2 idx10 | 2 / 2050 | 2 | **0.0221** | **111.4** | 6.1 | −1.2e‑5 | SATURATED_BOUND_STATE |
| K1 low‑mass idx4 | 1 / 1000 | 1 | **0.0386** | 100.0 | 4.5 | +8.7e‑5 | SATURATED_BOUND_STATE |
| K1 high‑mass idx0 | 1 / 2050 | 0 | NaN | NaN | 4.0 | NaN | DELOCALIZED_GROWTH |
| K6 NEAR idx33 | 6 / 1000 | 5→4 | 0.0090 | 30.2 | 14.4 | +1.5e‑4 | INCONCLUSIVE_FAILURE_TRACE |

Two morphological families emerge:
- **distributed, low-high-k family:** `feb56dc7`, K6 high-mass, K6 mid-mass, K4 (high_k ≈ 0.008–0.013,
  larger core radius, lower compactness).
- **compact, high-high-k family:** K2 (high_k 0.022, compactness 111), K1 low-mass (high_k 0.039,
  curvature_proxy ≈ 1580, ω²_min ≈ 0.042 — near the conformal-geometry edge).

---

## Inference (cautious labels)

- `K6_DISTRIBUTED_BRANCH_ROBUST` — supported; clearly mass-dependent, strongest at high mass.
- `K4_INTERMEDIATE_DISTRIBUTED_BRANCH_SUPPORTED` — dense TRUE band at 1200–1600, 4-node, low high-k;
  behaves like the distributed family at lower K/mass.
- `K2_COMPACT_INTERMEDIATE_BRANCH_SUPPORTED_BUT_RESOLUTION_RISK` — real (TRUE at all masses) but a
  *distinct compact* branch (2-node, high compactness/high-k, low diffusion), not a smaller-K version
  of the distributed family.
- `K1_LOW_MASS_BRANCH_FRAGILE` — real but narrow, seed-fragile, and morphologically near the numerical
  edge (highest high-k of any candidate); the least likely to survive resolution.
- `K1_FAILURE_BOUNDARY_ROBUST` — high-mass K1 reliably fails (delocalized growth, not collapse).
- `K3_WEAK_OR_NEAR_THRESHOLD_IN_THIS_SAMPLE` — not a priority branch in this sample.

Cross-cutting inference (testable, not proof): at fixed total mass, higher K means lower mass per
blob, so each K has a mass window where per-blob mass suffices to saturate but not run away — hence
the diagonal shift of the TRUE band with K. The distributed-vs-compact morphology split is the
scientifically important axis, and it maps onto resolution-risk (compact = higher high-k = more
likely to change at N96).

---

## Caveats

- This is `N=48`, `T=4000` **discovery** evidence; it is **not** N96 validation, and the absolute
  rates are not population laws.
- Diagnostic morphology metrics (compactness, core radius, high-k fraction) exist **only for the 8
  shortlist rows**, not all 720 — so the morphology split is shortlist-level, not population-proven.
- `feb56dc7` is an **external** anchor control (the multi-K loop starts at K=1, so the script does not
  inject it per block).
- Compact branches (K2, K1 low-mass) have higher high-k fraction → **higher resolution-risk**; their
  N=48 "TRUE" status is the most likely to change under N96.
- The high-mass K1 failure was `DELOCALIZED_GROWTH`, **not** a collapse signature — attach no
  collapse/singularity language.
- The search writes fresh timestamped directories and does not resume partial CSVs; two incomplete
  runs were excluded.

---

## Proposed action

`visual inspection v2 -> branch-diverse N96/T6000 validation -> then decide whether a longer search is justified.`

1. Inspect the v2 morphology map + matched branch-comparison panel (distributed vs compact made
   visible) and the N96 inspection sheet.
2. Run the branch-diverse N96/T6000 validation (`docs/PHASE_C_OPTION_B_N96_VALIDATION_PLAN.md`),
   **distributed candidates first** (K6 high-mass, K6 mid-mass, K4), then the compact resolution-risk
   contrasts (K2, K1 low-mass), keeping the K1 high-mass failure + feb56dc7 controls and the K6
   near-threshold probe. Every cross-resolution replay must carry an explicit resolution-scaled raw
   target override and stamped mass-resolution metadata.
3. Decide on a longer structured search only after seeing which branches survive resolution.
