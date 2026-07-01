# Phase C Visual Analysis Pack

## Purpose

Produce a standardized visual inspection layer for the completed Phase C structured discovery Option B results
without launching any new search.

This pack is for analysis and inspection only. It does not change the PDE, solver, `gravity/unified_omega.py`,
or classifier thresholds.

## Output Root

Generated figure bundle:

`sweep_runs/PHASE_C_VISUAL_ANALYSIS_20260624_161650`

Structure:

- `population/`
- `shortlist/`
- `cases/`

## Sources

Primary structured-discovery sources:

- `docs/PHASE_C_STRUCTURED_DISCOVERY_OPTION_B.md`
- `docs/phase_c_structured_discovery_B_summary.csv`
- `runtime_logs/phase_c_structured_discovery_B_shortlist_metrics.json`
- `docs/RUNBOOK_PHASE_C_AND_VISUALS.md`

Shortlisted representative rows:

- `CORE_SAT_HUNT_20260624_112918`, `idx=32` (`K=6` high-mass TRUE)
- `CORE_SAT_HUNT_20260624_142152`, `idx=34` (`K=6` mid-mass TRUE)
- `CORE_SAT_HUNT_20260624_124444`, `idx=25` (`K=4` intermediate TRUE)
- `CORE_SAT_HUNT_20260624_152029`, `idx=10` (`K=2` intermediate TRUE)
- `CORE_SAT_HUNT_20260624_142152`, `idx=4` (`K=1` low-mass TRUE)
- `CORE_SAT_HUNT_20260624_132302`, `idx=0` (`K=1` high-mass failure control)
- `CORE_SAT_HUNT_20260624_102149`, `idx=33` (`K=6` near-threshold NEAR)
- external `feb56dc7` control

## Representative Replay Prep

The population figures were rendered directly from the structured-discovery summary CSV and shortlist metrics JSON.

The case packs needed saved `frames.npz` and `diagnostic_summary.json` bundles for the exact shortlisted rows.
Those were produced with the existing replay helper only for the eight representative cases above.

Representative replay manifest:

- `sweep_runs/PHASE_C_VISUAL_ANALYSIS_20260624_161650/shortlist_replay_manifest.json`

No new hunt was launched.

## Rendered Figures

### Population

- `population/phase_c_true_rate_heatmap.png`
- `population/phase_c_true_rate_by_seed.png`
- `population/phase_c_class_composition.png`
- `population/phase_c_true_nodecount_distribution.png`
- `population/phase_c_branch_landscape_summary.png`

Purpose:

- inspect where TRUE support concentrates across `K` and raw target mass,
- separate seed effects from pooled rates,
- inspect whether TRUE support carries different final node families,
- make the intermediate `K=2` and `K=4` branches visually explicit instead of collapsing the story into only
  `K=1` versus `K=6`.

### Shortlist

- `shortlist/phase_c_shortlist_table.csv`
- `shortlist/phase_c_shortlist_overview.png`
- `shortlist/phase_c_shortlist_diagnostic_summary.png`

Purpose:

- keep the candidate selection criteria legible,
- show which shortlisted rows are TRUE, NEAR, or reject-side controls,
- preserve why each row was selected for follow-on validation.

### Case Packs

For each of:

- `cases/k6_high_mass_true/`
- `cases/k6_mid_mass_true/`
- `cases/k4_intermediate_true/`
- `cases/k2_intermediate_true/`
- `cases/k1_low_mass_true/`
- `cases/k1_high_mass_failure/`
- `cases/k6_near_threshold_near/`
- `cases/feb56dc7_control/`

generated:

- `timeline_panel.png`
- `density_slices.png`
- `vector_map.png`
- `case_summary.png`

Purpose:

- inspect scalar trace behavior over time,
- inspect late density organization,
- inspect whether a coherent vector/flow picture exists in the saved terminal frame,
- keep class, diagnostic label, node count, and branch role visible in one place.

## Quantule Viz Updates

The visual pack was rendered through the centralized read-only visualization layer.

Added/extended:

- `quantule_viz` CLI subcommand for structured Phase C discovery rendering
- structured population heatmaps / branch-summary figures
- shortlist overview and diagnostic summary rendering
- case-pack rendering from saved `diagnostic_summary.json` and `frames.npz`
- safer vector-preview fallback for fail-side cases with no coherent finite terminal vector field

`quantule_viz` remained read-only with respect to saved artifacts. The representative replays were prepared outside
the package via the existing replay helper.

## Disciplined Read

### Evidence

- The pooled TRUE-rate map supports a robust high-mass `K=6` distributed branch.
- The pooled TRUE-rate map also supports a narrow low-mass `K=1` survivor pocket that thins out above the low/mid
  masses.
- The branch-summary and TRUE node-count figures show that `K=2` and `K=4` are real enough to retain in the next
  shortlist.
- The shortlist diagnostic panel keeps the reject-side `K=1` high-mass control separate from the TRUE branches.
- The case packs show visibly different late-state organization across:
  - `K=1` low-mass fragile survivor,
  - `K=2` intermediate two-node branch,
  - `K=4` intermediate distributed branch,
  - `K=6` distributed TRUE branches,
  - `K=6` near-threshold NEAR candidate,
  - `K=1` high-mass failure control,
  - `feb56dc7` external control.

### Inference

- `K6_DISTRIBUTED_BRANCH_ROBUST` remains supported.
- `K1_FAILURE_BOUNDARY_ROBUST` remains supported.
- `K1_LOW_MASS_BRANCH_FRAGILE` remains supported.
- The structured discovery layer is richer than a binary `K=1` versus `K=6` story because `K=2` and `K=4`
  contribute meaningful intermediate branches.

### Caveat

- This visual pack is built on the `N=48`, `T=4000` structured discovery layer plus representative replay/diagnostic
  bundles for shortlisted rows.
- These visuals are inspection support, not a new claim of proof, a universal `K`-tracking law, or a universal
  `4-5` node law.

### Proposed Use

- Use the population figures to preserve branch diversity in the next N96 shortlist.
- Use the case packs to compare:
  - fragile low-mass `K=1` structure,
  - reject-side high-mass `K=1` control behavior,
  - intermediate `K=2` and `K=4` branches,
  - robust distributed `K=6` branches,
  - and the external `feb56dc7` anchor control.
