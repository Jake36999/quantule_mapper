# Evidence Inventory (Baseline Review — Stage 0)

**Purpose:** catalogue the load-bearing results that the closed Phase C claims rest on, with hashes + paths, so
every audit assertion is traceable to data. Stage 0 of `BASELINE_REVIEW_PLAN.md`. This is an inventory of *what
exists* — no interpretation or redesign.

**Reproducibility chain.** Scripts are in git (frozen geometry `e8d6a78ea`, jax_scout FP64 mirror); **results
data is on-disk only** (`sweep_runs/` was purged from git in the 2026-07-02 cleanup to keep the repo ~31 MB).
So: *code = versioned; evidence = local disk, manifested here.* `sweep_runs/` = **~26 GB across 120 run dirs**;
the load-bearing subset below is ~22 runs. Hashes are sha256, first 16 hex chars; sizes in bytes/MB as noted.

## Ledgers / databases
| artifact | size | status |
|---|---|---|
| `queue_runtime.db` | 2.8 MB | orchestrator queue — transient runtime state |
| `Simulation_ledgers/*.csv` (`simulation_ledger (5)/(6)`, `extracted log.txt`) | ~140 KB | **legacy** (Feb 2026), pre-Phase-C |
| `simulation_ledger.db` (aste_hunter production DB) | — | **NOT present on this box** — the NSGA-II hunter's accumulated ledger lives on the CuPy production path, not here |

---

## A. Stability arc (basin → a\*) — load-bearing runs
All under `sweep_runs/<run>/`; metadata = `*.csv` + `*_summary.json` (claim-bearing numbers); raw = `psi_fin`
`.npz` fields. "meta-sha" = sha256 of the concatenated metadata files.

| run (date) | script (in git) | supports claim | meta-sha | raw |
|---|---|---|---|---|
| `PHASE_C_OPTION_B_N96_STAGE1_20260624_223147` | core_saturation_replay | morphology axis did NOT predict survival (INCONCLUSIVE) | `fbdb4e1e0bc3cb62` | — |
| `PHASE_C_N96_OVERNIGHT_20260625_014313` | core_saturation_replay | T6000 verdicts = window artifacts; capacity hypothesis falsified | `1449b797799a4bcf` | — |
| `PHASE_C_N96_LONGT_CONTROL_20260625_083731` | core_saturation_replay | feb stable @T24000; K4/K2 decay | `a9773b40ae1a852e` | — |
| `PHASE_C_STABILITY_GATE_CALIB_20260625_121731` | — | v2/v3 gate calibration (see doc) | *(md/log-based)* | — |
| `FEB_BASIN_20260625_122824` | feb_basin_search | feb param-regime = robust basin (K3–8 stable) | `abd70c176a8c0740` | 8 npz / 313 MB |
| `FEB_BASIN_CONFIRM_20260625_154503` | feb_basin_confirm | seed-robust; v2 over-rejects breathing @T24000 | `fd0fa0b73413c371` | 12 npz / 470 MB |
| `FEB_PARAM_BASIN_20260626_004039` | feb_param_basin | v3 gate; param_a critical (±10%), structured basin | `9d22ef5ce454a432` | 52 npz / 684 MB |
| `FEB_PARAM_EDGE_CONFIRM_20260626_124432` | feb_param_edge_confirm | basin interior seed-robust, upper edges seed-sensitive | `fcbf0f22eac7bab6` | — |
| `FEB_BASIN_POSTHOC_VALIDATION_20260626_122835` | quantulemapper_real+tda | prime-SSE 0/60, TDA ~0 = NULL (non-discriminating) | `32afa703e885cd74` | — |
| `FEB_CENTER_RESOLUTION_N128_20260626_195449` | feb_center_resolution | basin not an N96 grid artifact (N128 TRUE) | `c0abe2c2b099bb5b` | 2 npz / 62 MB |
| `FEB_OBSERVABLE_EXTRACTION_20260626_220710` | (observable extract) | stable/failed don't separate on snapshot/spatial spectra | `0cafaba56c2ea6f6` | — |
| `FEB_JOINT_BASIN_20260626_224056` | feb_joint_basin | boundary = gain/loss balance surface (coupling) | `e4c61fec22974ee9` | 45 npz / 592 MB |
| `FEB_JOINT_STAGE2_20260627_123235` | feb_joint_stage2 | core robust, T24000 core narrower; matched controls | `29db65099a4f20c5` | 19 npz / 250 MB |
| `FEB_CORE_DELINEATION_T24000_20260627_175050` | feb_core_delineation | eta×1.0 plane 12/15 TRUE @T24000 | `da6c37e53ee54159` | 15 npz / 199 MB |

## B. Mobility + a\* arc — load-bearing runs

| run (date) | script (in git) | supports claim | metadata (sha, bytes) | raw |
|---|---|---|---|---|
| `FEB_BREATHING_LONGT_T72000_20260628_003032` | feb_breathing_longt | falsifier: feb-center + ×1.05 DECAY @T72000 (T24000=window) | csv `fb41bd3c29253014`/472 · json `bb65a5b151d7f7de`/1853 | 2 npz / 26.6 MB |
| `FEB_GAIN_LADDER_LONGT_T72000_20260701_175708` | feb_gain_ladder_longt | **a\* found**: late-slope→0 at ×1.15 | csv `5feb3b8e0f8ab24e`/1282 · json `ef2fde98a8f76200`/3624 | 4 npz / 52.9 MB |
| `FEB_ASTAR_CONFIRM_20260702_003055` | feb_astar_confirm | a\* confirmed: T144k flat + seeds + bracket ×1.16 grows | csv `1335ede40b03db35`/2093 · json `33d0a81c2bb33047`/6191 | 7 npz / 92.6 MB |
| `FEB_KICK_INERTIA_20260702_122013` | feb_kick_inertia | Galilean-kick inertial null (μ≈0, coherent) | csv `1ec803a3b68a0c00`/892 · json `e1a23e4fddaf0d6d`/2499 | 1 npz (traj) |
| `FEB_ADIABATIC_DRAG_static_20260702_142009` | feb_adiabatic_drag | static well NO_COUPLING (weak V0), baseline bit-identical | csv `7d6311a953771e51`/635 · json `84faf96ec7ed639a`/2086 | 4 npz (traj) |
| `FEB_ADIABATIC_DRAG_V0LADDER_20260702_154008` | feb_adiabatic_drag | V0 ladder 0.075→0.4 = ACCRETION_ONLY (4-node) | csv `ac4f9dcb5e89a75b`/1212 · json `d119c0ca8c06485e`/4348 | 8 npz (traj) |
| `FEB_ADIABATIC_DRAG_V0LADDER_seed620_20260702` | feb_adiabatic_drag | 6-node morphology: NO_COUPLING | csv `ce2cb399287ff9a4`/1115 · json `36bd67c13150c3d3`/4234 | 8 npz (traj) |
| `FEB_ADIABATIC_DRAG_V0LADDER_seed621_20260702` | feb_adiabatic_drag | 6-node morphology: ACCRETION_THEN_NUCLEATION | csv `f385136d445c45d7`/1185 · json `bff96290aeb081b0`/4318 | 8 npz (traj) |

*Note: kick/drag `.npz` hold small COM-trajectory arrays (not full fields); the a\*-arc a×1.15 settled state
used by the kick/drag runs is `FEB_GAIN_LADDER_.../a1.15_ladder_T72000_probe.npz` + the `FEB_ASTAR_CONFIRM`
seed620/621 states.*

## C. Exploratory / superseded base (NOT manifested — inventory-level only)
~98 earlier discovery/exploratory run dirs remain on disk: `CORE_SAT_HUNT_*` (×many, 2026-06-23/24 discovery
sweeps), `CORE_SAT_MASS_THRESHOLD_*`, `CORE_SAT_TRACE_COMPARE_*`, `PHASE_C_OPTION_B_N96_TRACE/CLOSURE/CURRENT_*`,
`PHASE_C_VISUAL_ANALYSIS_*`, `FEB_BASIN_TOPOLOGY_*`, `CORE_BASIN_*`, `BRIDGE_HUNT_*`, `AF_*` (A-field/routing,
Stage-B era). These are **superseded** by the load-bearing subset and by the final dossier; retained on disk for
provenance but not claim-bearing. (Stage-B / A-field / BSSN lines are `DEAD/LEGACY` per the architecture audit.)

## Preservation note (→ Stage 2 parking lot, not actioned here)
The load-bearing evidence is **on-disk only**. This manifest is the traceability record; the *scripts* that
regenerate the metadata are in git at frozen geometry `e8d6a78ea`. Recommended (Stage 2 hardening, not now):
archive the ~22 load-bearing runs' metadata + the key `psi_fin` states (a\*-arc: gain-ladder + astar-confirm;
basin exemplars) off-box alongside this manifest, so the closed claims remain reproducible-from-data even if the
26 GB `sweep_runs/` is pruned.
