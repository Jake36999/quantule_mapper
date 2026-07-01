# Phase C Structured Discovery Option B

## Purpose

Map resolution-fair long-time saturated branches under total-mass-controlled initial conditions while
keeping three controls visible:

- fragile low-mass `K=1` survivor pocket,
- robust high-mass `K=1` failure boundary,
- distributed high-mass `K=6` saturation branch.

This is still an `N=48`, `T=4000` discovery layer. It is not a population law or final validation.

## Method

- Search script: `jax_scout/core_saturation_search.py`
- Commit: `bb6989afc` (`Add Phase C ic-seed control`)
- Grid / horizon: `N=48`, `T=4000`
- IC normalization: `total_mass_fixed`
- IC blob counts: `K in {1,2,3,4,6}`
- Raw target masses: `500, 800, 1000, 1200, 1600, 2050.293702`
- IC seeds: `20260619, 20260620, 20260621`
- Batch contract per block: `batch=8`, `max_batches=5`
- Total planned matrix: `6 masses x 3 seeds x 5 K x 8 = 720` configs

Mass convention:

- Search input uses raw `sum(|psi|^2)` targets.
- The physically comparable documented quantity is the `dx^3`-weighted target mass.
- At `N=48`, `dx = 10 / 48`, so `dx^3 = 0.009042245370370372`.

Corresponding documented target masses:

- `500 -> 4.521122685`
- `800 -> 7.233796296`
- `1000 -> 9.042245370`
- `1200 -> 10.850694444`
- `1600 -> 14.467592593`
- `2050.293702 -> 18.539258736`

## Run Directories

Complete 40-row blocks used in this report:

- Seed `20260619`:
  `CORE_SAT_HUNT_20260624_094432`,
  `CORE_SAT_HUNT_20260624_100311`,
  `CORE_SAT_HUNT_20260624_102149`,
  `CORE_SAT_HUNT_20260624_104037`,
  `CORE_SAT_HUNT_20260624_111044`,
  `CORE_SAT_HUNT_20260624_112918`
- Seed `20260620`:
  `CORE_SAT_HUNT_20260624_114744`,
  `CORE_SAT_HUNT_20260624_120642`,
  `CORE_SAT_HUNT_20260624_122538`,
  `CORE_SAT_HUNT_20260624_124444`,
  `CORE_SAT_HUNT_20260624_130405`,
  `CORE_SAT_HUNT_20260624_132302`
- Seed `20260621`:
  `CORE_SAT_HUNT_20260624_134206`,
  `CORE_SAT_HUNT_20260624_140205`,
  `CORE_SAT_HUNT_20260624_142152`,
  `CORE_SAT_HUNT_20260624_144135`,
  `CORE_SAT_HUNT_20260624_150101`,
  `CORE_SAT_HUNT_20260624_152029`

Operational exclusions:

- `CORE_SAT_HUNT_20260624_093948`: seed smoke only, `n_eval=8`
- `CORE_SAT_HUNT_20260624_105916`: interrupted partial, `n_eval=16`

Also note: the current multi-K search script only injects `ref_feb56dc7` when batch 1 is `K=6`. This
Option B loop starts at `K=1`, so `feb56dc7` is treated here as an external anchored control, not an
in-run row.

## Operational Integrity

Evidence:

- Completed structured runs: `18`
- Completed configs analyzed: `720`
- Excluded incomplete runs: `2`
- No nonzero exit codes in the resumed master loop

Overall class counts:

- `TRUE_SATURATED_BOUND_STATE`: `94`
- `NEAR_SATURATED_BOUND_STATE`: `53`
- `LATE_BLOWUP_REJECT`: `338`
- `SPIN_DOWN_REJECT`: `195`
- `TRANSIENT_GROWER_REJECT`: `40`

## Evidence

### TRUE Counts By K And Mass

Each cell is `TRUE count / 24` configs (`3 seeds x 8 rows`).

| K | 500 | 800 | 1000 | 1200 | 1600 | 2050.293702 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 3/24 | 1/24 | 1/24 | 0/24 | 0/24 | 0/24 |
| 2 | 6/24 | 6/24 | 6/24 | 5/24 | 4/24 | 3/24 |
| 3 | 3/24 | 3/24 | 2/24 | 0/24 | 0/24 | 2/24 |
| 4 | 0/24 | 3/24 | 6/24 | 8/24 | 8/24 | 5/24 |
| 6 | 0/24 | 0/24 | 5/24 | 3/24 | 3/24 | 8/24 |

### K1 Control Branches

Evidence:

- `K=1` low-mass TRUEs exist:
  - `500`: `3 TRUE`, `1 NEAR`
  - `800`: `1 TRUE`
  - `1000`: `1 TRUE`
- `K=1` high-mass TRUEs vanish:
  - `1200`: `0 TRUE`
  - `1600`: `0 TRUE`
  - `2050.293702`: `0 TRUE`
- `K=1` class mix by mass:
  - `500`: `15 BLOWUP`, `4 SPIN`, `1 GROW`, `3 TRUE`, `1 NEAR`
  - `800`: `18 BLOWUP`, `4 SPIN`, `1 GROW`, `1 TRUE`
  - `1000`: `19 BLOWUP`, `3 SPIN`, `1 GROW`, `1 TRUE`
  - `1200`: `21 BLOWUP`, `3 SPIN`
  - `1600`: `22 BLOWUP`, `2 SPIN`
  - `2050.293702`: `23 BLOWUP`, `1 SPIN`

Inference:

- The `K=1` branch is real but narrow.
- The practical break occurs between `1000` and `1200` in this structured sample.
- The high-mass `K=1` failure boundary remains robust.

### K6 Distributed Branch

Evidence:

- `K=6` produces no TRUE rows at `500` or `800`.
- `K=6` begins producing TRUE rows at `1000` and remains active through the upper masses:
  - `1000`: `5 TRUE`, node counts `{4:1, 5:1, 6:3}`
  - `1200`: `3 TRUE`, node counts `{4:1, 6:2}`
  - `1600`: `3 TRUE`, node counts `{4:1, 6:2}`
  - `2050.293702`: `8 TRUE`, node counts `{4:2, 5:2, 6:4}`
- `K=6` low-mass cells are mostly spin-down / near:
  - `500`: `21 SPIN`, `3 NEAR`
  - `800`: `18 SPIN`, `3 NEAR`, `3 BLOWUP`

Inference:

- The distributed `K=6` branch is robust and clearly mass-dependent.
- In this discovery layer it is not a low-mass branch.
- Its TRUE density rises sharply by the top mass block, and it supports both `4-5` node feb-adjacent
  outcomes and a `6`-node distributed family.

### Intermediate Branches

Evidence:

- `K=2` is stronger than expected and produces TRUE rows at every tested mass:
  - TRUE counts: `6, 6, 6, 5, 4, 3`
  - final nodes mostly `2`, with some `1` and occasional `3`
- `K=4` is also strong:
  - `500`: `0 TRUE`
  - `800`: `3 TRUE`
  - `1000`: `6 TRUE`
  - `1200`: `8 TRUE`
  - `1600`: `8 TRUE`
  - `2050.293702`: `5 TRUE`
  - final nodes mostly `4`, with some `2` and `3`
- `K=3` is weaker:
  - TRUE counts: `3, 3, 2, 0, 0, 2`
  - many more `NEAR` than TRUE at `1000-2050`

Inference:

- The discovery layer did not reduce to a simple `K=1` vs `K=6` story.
- `K=2` and `K=4` both show persistent intermediate branches that should be preserved in the next
  shortlist.
- `K=3` looks more like a near-threshold / weaker branch in this sample.

## Candidate Shortlist

Shortlist metrics were computed with targeted post-run diagnostics using the existing replay path. Only
these selected rows received compactness / core-radius / high-k annotations.

### Best K6 Distributed TRUE At High Mass

- Source: `CORE_SAT_HUNT_20260624_112918`, `idx=32`
- `K=6`, raw target mass `2050.293702`, documented mass `18.539258736`, `ic_seed=20260619`
- Class: `TRUE_SATURATED_BOUND_STATE`
- Diagnostic label: `SATURATED_BOUND_STATE`
- Final nodes: `5`
- Late slope: `-9.13e-05`
- Compactness max: `45.258`
- Core radius min: `9.107`
- High-k fraction max: `0.0090`
- Params:
  `D=3.6201, eta=0.0597, rho_vac=1.3871, omega0=0.9716, a_coupling=1.9495, s=-0.3250, f=0.1255, a=0.3158`
- Reason:
  high-mass stable `5`-node branch, feb-adjacent, with `n_mid == n_fin`

### Best K6 Distributed TRUE At Mid Mass

- Source: `CORE_SAT_HUNT_20260624_142152`, `idx=34`
- `K=6`, raw target mass `1000`, documented mass `9.042245370`, `ic_seed=20260621`
- Class: `TRUE_SATURATED_BOUND_STATE`
- Diagnostic label: `SATURATED_BOUND_STATE`
- Final nodes: `6`
- Late slope: `3.45e-05`
- Compactness max: `31.772`
- Core radius min: `18.589`
- High-k fraction max: `0.0083`
- Params:
  `D=2.4793, eta=0.0948, rho_vac=1.6947, omega0=1.5954, a_coupling=2.3827, s=-0.1985, f=-0.0183, a=0.4605`
- Reason:
  lowest-slope stable distributed `K=6` TRUE in the mid-mass band

### Best K4 Intermediate Distributed TRUE

- Source: `CORE_SAT_HUNT_20260624_124444`, `idx=25`
- `K=4`, raw target mass `1200`, documented mass `10.850694444`, `ic_seed=20260620`
- Class: `TRUE_SATURATED_BOUND_STATE`
- Diagnostic label: `SATURATED_BOUND_STATE`
- Final nodes: `4`
- Late slope: `-2.11e-05`
- Compactness max: `44.437`
- Core radius min: `22.568`
- High-k fraction max: `0.0135`
- Params:
  `D=2.8861, eta=0.0663, rho_vac=1.5523, omega0=0.6541, a_coupling=3.9815, s=-0.6166, f=0.1484, a=0.4728`
- Reason:
  clean non-`K=6` distributed `4`-node branch at mid mass

### Best K2 Intermediate Branch

- Source: `CORE_SAT_HUNT_20260624_152029`, `idx=10`
- `K=2`, raw target mass `2050.293702`, documented mass `18.539258736`, `ic_seed=20260621`
- Class: `TRUE_SATURATED_BOUND_STATE`
- Diagnostic label: `SATURATED_BOUND_STATE`
- Final nodes: `2`
- Late slope: `-1.21e-05`
- Compactness max: `111.388`
- Core radius min: `6.142`
- High-k fraction max: `0.0221`
- Params:
  `D=0.1227, eta=0.0944, rho_vac=1.0499, omega0=0.3503, a_coupling=1.3435, s=-0.0994, f=-0.2822, a=0.3811`
- Reason:
  stable `2`-node branch that persists all the way to the top mass block

### Best K1 Low-Mass TRUE

- Source: `CORE_SAT_HUNT_20260624_142152`, `idx=4`
- `K=1`, raw target mass `1000`, documented mass `9.042245370`, `ic_seed=20260621`
- Class: `TRUE_SATURATED_BOUND_STATE`
- Diagnostic label: `SATURATED_BOUND_STATE`
- Final nodes: `1`
- Late slope: `8.67e-05`
- Compactness max: `100.046`
- Core radius min: `4.525`
- High-k fraction max: `0.0386`
- Params:
  `D=3.3827, eta=-0.0031, rho_vac=0.3385, omega0=1.7092, a_coupling=1.6545, s=-0.6062, f=-0.0102, a=0.3739`
- Reason:
  single-node near-threshold `K=1` survivor; best carry-forward fragile control

### Best K1 High-Mass Failure Control

- Source: `CORE_SAT_HUNT_20260624_132302`, `idx=0`
- `K=1`, raw target mass `2050.293702`, documented mass `18.539258736`, `ic_seed=20260620`
- Class: `LATE_BLOWUP_REJECT`
- Diagnostic label: `DELOCALIZED_GROWTH`
- Final nodes: `0`
- Time to failure: `100`
- Params:
  `D=4.6502, eta=0.0984, rho_vac=0.2906, omega0=0.8025, a_coupling=3.9369, s=-0.4355, f=0.4486, a=0.2799`
- Reason:
  clean top-mass failure control from an `8/8` blowup batch; diagnostic replay did not support a
  collapse-like label

### Best K6 Near-Threshold NEAR Candidate

- Source: `CORE_SAT_HUNT_20260624_102149`, `idx=33`
- `K=6`, raw target mass `1000`, documented mass `9.042245370`, `ic_seed=20260619`
- Class: `NEAR_SATURATED_BOUND_STATE`
- Diagnostic label: `INCONCLUSIVE_FAILURE_TRACE`
- Final nodes: `4`
- Late slope: `1.51e-04`
- Compactness max: `30.192`
- Core radius min: `14.437`
- High-k fraction max: `0.0090`
- Params:
  `D=0.5405, eta=0.0250, rho_vac=1.8819, omega0=0.4033, a_coupling=3.1001, s=-0.5226, f=-0.4230, a=0.4041`
- Reason:
  closest K=6 near-threshold candidate to the TRUE band without over-upgrading it

### feb56dc7 Control

- Source: external reference replay, `N=96`, `T=6000`
- Class: `TRUE_SATURATED_BOUND_STATE`
- Diagnostic label: `SATURATED_BOUND_STATE`
- Final nodes: `4`
- Late slope: `-2.09e-06`
- Params:
  `D=2.7329, eta=0.0704, rho_vac=1.1866, omega0=0.0, a_coupling=2.3098, s=0.0129, f=-0.4861, a=0.4802`
- Reason:
  anchor control for the known `LONG_TIME_STABLE_4_NODE_ATTRACTOR`

## Inference

- `K6_DISTRIBUTED_BRANCH_ROBUST` remains supported.
- `K1_FAILURE_BOUNDARY_ROBUST` remains supported.
- `K1_LOW_MASS_BRANCH_FRAGILE` remains supported, but the structured discovery layer widened the
  pocket more than the earlier local robustness pass suggested: a single-node `K=1` TRUE survived up
  through `1000` for one seed.
- The discovery layer also found stronger-than-expected intermediate branches:
  - `K=2` supports stable `1-2` node TRUE rows across the full mass ladder.
  - `K=4` supports stable distributed `3-4` node TRUE rows from `800` upward and is especially dense
    at `1200-1600`.
- The high-mass `K=1` failure control did not acquire a collapse-like signature in the targeted
  diagnostic replay; it remained `DELOCALIZED_GROWTH`.

## Caveats

- This is still `N=48`, `T=4000` discovery evidence.
- `feb56dc7` was not injected inside each structured multi-K block; it remains an external control.
- Diagnostic labels, compactness, core radius, and high-k metrics were computed only for the selected
  shortlist, not for all `720` discovery rows.
- The search script still writes timestamped fresh directories; it does not resume partial CSVs.

## Proposed Action

Carry this exact branch-diverse shortlist into the next `N=96`, `T=6000` validation layer with
explicit resolution-scaled raw target overrides:

- best `K=6` high-mass distributed TRUE
- best `K=6` mid-mass distributed TRUE
- best `K=4` intermediate distributed TRUE
- best `K=2` intermediate branch
- best `K=1` low-mass TRUE
- best `K=1` high-mass failure control
- best `K=6` near-threshold NEAR candidate
- `feb56dc7` control

Use `docs/phase_c_structured_discovery_B_summary.csv` as the row-level source of truth for candidate
provenance; each cross-resolution replay must carry an explicit mass override and stamped
mass-resolution metadata.
