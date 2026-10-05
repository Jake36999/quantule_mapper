---
tags: [index, runs, triage]
---

# Triage - runs with no `summary.json`

76 run directories under `sweep_runs/` carry no `summary.json` and are therefore **not**
catalogued as notes.

> [!success] The 77 **substantive** no-summary runs have been promoted
> Codex-lane runs use a different convention - `RUN_COMPLETE.json` with a `status` field, a `TECHNICAL_HANDOFF.md`, preregistered matrices and model specs - rather than a `summary.json`. They now have derived notes in [[_INDEX|the catalogue]], marked `source: derived`.
> What remains below is genuinely thin: smoke tests, aborted launches, single-file dirs, and empty directories.

| class | n | meaning |
|---|---:|---|
| `minor` | 68 | smoke tests, aborted launches, single-file dirs |
| `empty` | 8 | no files at all; safe to delete |

To promote any of these, give the run a `summary.json` (or write the note by hand) and rebuild.

| class | run | date | files | csv | contents |
|---|---|---|---:|---:|---|
| `minor` | `TG_B2_CONVERGENCE_N80_L20_20260822_205815` | 2026-08-22 | 3 | 1 | ROW_sep3.00_STARTED.json, config.json, scalars_sep3.00_off.csv |
| `empty` | `TG_B2_STATIC_FORCE_20260715_235113` | 2026-07-15 | 0 | 0 |  |
| `minor` | `TG_B1S_D4_D5_CLOSURE_GPU_20260715_002840` | 2026-07-15 | 11 | 1 | PID, TMUX_SESSION, baseline_reproduction.csv, environment_versions.json, git_state_before.txt, gpu_preflight.json ... |
| `minor` | `TG_B1S_D4_D5_CLOSURE_GPU_20260715_002649` | 2026-07-15 | 3 | 0 | PID, launch.sh, overnight_stdout_stderr.log |
| `minor` | `TG_B1S_D4_D5_CLOSURE_GPU_20260715_002555` | 2026-07-15 | 2 | 0 | PID, overnight_stdout_stderr.log |
| `empty` | `TG_A_CONTRACTS_20260714_113802` | 2026-07-14 | 0 | 0 |  |
| `minor` | `GRAVITY_D_ROBUSTNESS_GPU_20260713_194643` | 2026-07-13 | 13 | 0 | R0_flat_null_N96_T4.json, R0_flat_null_N96_T4_trajectory.npz, R0_reference_rho2_N96_T4.json, R0_reference_rho2_N96_T4_trajectory.npz, R1_source_rho.json, R1_source_rho2.json ... |
| `minor` | `GRAVITY_D_ROBUSTNESS_GPU_20260713_194053` | 2026-07-13 | 5 | 0 | R0_reference_rho2_N96_T4.json, R0_reference_rho2_N96_T4_trajectory.npz, gpu_preflight.json, preregistered_matrix.json, source_matching.json |
| `empty` | `GRAVITY_D_EXISTING_REPRO_CODEX_20260713` | 2026-07-13 | 0 | 0 |  |
| `empty` | `GRAVITY_D_EFFECTIVE_MEDIUM_CODEX_SMOKE_20260713` | 2026-07-13 | 0 | 0 |  |
| `minor` | `GRAVITY_A_RINT_20260712_214941` | 2026-07-12 | 1 | 0 | rint_local_sep4.npz |
| `empty` | `GRAVITY_A1_MATCHED_20260712_220933` | 2026-07-12 | 0 | 0 |  |
| `minor` | `GRAVITY_DESAT_PILOT` | 2026-07-10 | 1 | 0 | desat_pilot.json |
| `minor` | `C27_QUICK_V2` | 2026-07-10 | 2 | 0 | r0_cfl.json, r2_feb.json |
| `minor` | `C27_QUICK_TEST` | 2026-07-10 | 2 | 0 | r0_cfl.json, r2_feb.json |
| `minor` | `C27_REDERIVE` | 2026-07-09 | 4 | 0 | r0_cfl.json, r2_feb.json, r3_n96.json, r3_soliton_n96.npy |
| `minor` | `C25_P2B_hit1b` | 2026-07-09 | 1 | 0 | p2b.json |
| `minor` | `C25_P2B_hit1` | 2026-07-09 | 1 | 0 | p2b.json |
| `empty` | `C23_N96` | 2026-07-08 | 0 | 0 |  |
| `minor` | `live_smoke` | 2026-07-07 | 20 | 0 | artifacts/14136fd9bc48.h5, artifacts/14136fd9bc48.params.json, artifacts/1648992c82e5.h5, artifacts/1648992c82e5.params.json, artifacts/306c3182c978.h5, artifacts/306c3182c978.params.json ... |
| `minor` | `c1_smoke` | 2026-07-04 | 1 | 1 | c1_transport_results.csv |
| `minor` | `PHASE_D_STRESS_TENSOR_20260704` | 2026-07-04 | 1 | 0 | stress_tensor_results.json |
| `minor` | `PHASE_D_C2_CONFIRM_20260704_201928` | 2026-07-04 | 1 | 0 | confirm_boost.json |
| `minor` | `PHASE_D_C1_TRANSPORT_20260704_105309` | 2026-07-04 | 2 | 1 | c1_transport_results.csv, c1_transport_summary.json |
| `empty` | `PHASE_D_C1_TRANSPORT_20260704_104359` | 2026-07-04 | 0 | 0 |  |
| `minor` | `PHASE_D6_REDUCED_MODEL` | 2026-07-04 | 1 | 0 | reduced_model_validation.json |
| `minor` | `PHASE_C_NODE_LIBRARY_20260704` | 2026-07-04 | 2 | 1 | PHASE_C_NODE_LIBRARY.csv, PHASE_C_NODE_LIBRARY.json |
| `minor` | `step2_smoke` | 2026-07-03 | 5 | 0 | min_params.json, min_test.h5, probe96.h5, probe96.json, provenance_e5dfda922abd1b64886db4807c6b3ce6f6f3b881e0bd555c34f02ef90f9dcc39_seed20260619_20260703.json |
| `minor` | `LIVE_HUNTER_20260703_230055` | 2026-07-03 | 38 | 0 | artifacts/3f053d8025d6.h5, artifacts/3f053d8025d6.params.json, artifacts/49ec58c33002.h5, artifacts/49ec58c33002.params.json, artifacts/5d97af0062b5.h5, artifacts/5d97af0062b5.params.json ... |
| `minor` | `HUNTER_REAIM_REDISCOVERY_STAGEC_20260703_110904` | 2026-07-03 | 2 | 1 | rediscovery_results.csv, rediscovery_summary.json |
| `minor` | `HUNTER_REAIM_REDISCOVERY_20260703_100206` | 2026-07-03 | 2 | 1 | rediscovery_results.csv, rediscovery_summary.json |
| `minor` | `FEB_KICK_INERTIA_20260702_122013` | 2026-07-02 | 3 | 1 | feb_kick_inertia_results.csv, feb_kick_inertia_summary.json, kick_trajectories.npz |
| `minor` | `FEB_ADIABATIC_DRAG_static_20260702_142009` | 2026-07-02 | 6 | 1 | baseline_None_traj.npz, feb_adiabatic_drag_results.csv, feb_adiabatic_drag_summary.json, well_offset_V0.025_traj.npz, well_offset_V0.05_traj.npz, well_on_centre_traj.npz |
| `minor` | `FEB_ADIABATIC_DRAG_V0LADDER_seed621_20260702` | 2026-07-02 | 10 | 1 | baseline_None_traj.npz, feb_adiabatic_drag_results.csv, feb_adiabatic_drag_summary.json, well_offset_V0.075_traj.npz, well_offset_V0.15_traj.npz, well_offset_V0.1_traj.npz ... |
| `minor` | `FEB_ADIABATIC_DRAG_V0LADDER_seed620_20260702` | 2026-07-02 | 10 | 1 | baseline_None_traj.npz, feb_adiabatic_drag_results.csv, feb_adiabatic_drag_summary.json, well_offset_V0.075_traj.npz, well_offset_V0.15_traj.npz, well_offset_V0.1_traj.npz ... |
| `minor` | `FEB_ADIABATIC_DRAG_V0LADDER_20260702_154008` | 2026-07-02 | 10 | 1 | baseline_None_traj.npz, feb_adiabatic_drag_results.csv, feb_adiabatic_drag_summary.json, well_offset_V0.075_traj.npz, well_offset_V0.15_traj.npz, well_offset_V0.1_traj.npz ... |
| `minor` | `FEB_PARAM_EDGE_CONFIRM_20260626_124432` | 2026-06-26 | 2 | 1 | feb_param_edge_confirm_results.csv, feb_param_edge_confirm_summary.json |
| `empty` | `FEB_BASIN_TOPOLOGY_20260625_154544` | 2026-06-25 | 0 | 0 |  |
| `minor` | `CORE_SAT_THRESHOLD_DIAG_20260623_180519` | 2026-06-23 | 98 | 1 | replays/CORE_SAT_HUNT_20260623_170944_idx_0/diagnostic_summary.json, replays/CORE_SAT_HUNT_20260623_170944_idx_1/diagnostic_summary.json, replays/CORE_SAT_HUNT_20260623_170944_idx_10/diagnostic_summary.json, replays/CORE_SAT_HUNT_20260623_170944_idx_11/diagnostic_summary.json, replays/CORE_SAT_HUNT_20260623_170944_idx_12/diagnostic_summary.json, replays/CORE_SAT_HUNT_20260623_170944_idx_13/diagnostic_summary.json ... |
| `minor` | `CORE_SAT_PILOT_20260623_112559` | 2026-06-23 | 1 | 1 | all_evals.csv |
| `minor` | `CORE_SAT_HUNT_20260623_112941` | 2026-06-23 | 1 | 1 | all_evals.csv |
| `minor` | `CORE_SAT_HUNT_20260623_004423` | 2026-06-23 | 1 | 1 | all_evals.csv |
| `minor` | `CORE_SAT_HUNT_20260622_211628` | 2026-06-22 | 1 | 1 | all_evals.csv |
| `minor` | `SUBSTRATE_CALIB_20260621_161248` | 2026-06-21 | 3 | 1 | all_evals.csv, frozen_substrates.json, status.json |
| `minor` | `AF_BRIDGE_HUNT_20260621_060714` | 2026-06-21 | 15 | 1 | ANISOTROPIC_PROXY_BRIDGE_SELECTIVITY_MANIFEST.json, ROUTING_NULL_RESULT_MANIFEST.json, afield_aniso_deconfound.json, afield_aniso_strongbridge.json, afield_anisotropic.json, afield_routing_hunt.json ... |
| `minor` | `VALIDATION_20260620_213438` | 2026-06-20 | 1 | 0 | validation_report.json |
| `minor` | `TRANSFER_DIAG_20260620_111531` | 2026-06-20 | 2 | 1 | transfer_diag.json, transfer_evals.csv |
| `minor` | `TRANSFER_DIAG_20260620_111119` | 2026-06-20 | 2 | 1 | transfer_diag.json, transfer_evals.csv |
| `minor` | `TRANSFER_DIAG_20260620_110932` | 2026-06-20 | 2 | 1 | transfer_diag.json, transfer_evals.csv |
| `minor` | `TRANSFER_DIAG_20260620_110434` | 2026-06-20 | 2 | 1 | transfer_diag.json, transfer_evals.csv |
| `minor` | `GEN15_DEEPDIVE_20260620_113052` | 2026-06-20 | 1 | 0 | gen15_deepdive.json |
| `minor` | `BRIDGE_HUNT_20260620_180938` | 2026-06-20 | 11 | 1 | afield_current_coupled.json, afield_current_tune.json, afield_scalar_rescue.json, all_evals.csv, corridor_longwindow.json, corridor_phasetest.json ... |
| `minor` | `BRIDGE_HUNT_20260620_174057` | 2026-06-20 | 2 | 1 | all_evals.csv, status.json |
| `minor` | `BRIDGE_HUNT_20260620_173938` | 2026-06-20 | 1 | 1 | all_evals.csv |
| `minor` | `BRIDGE_HUNT_20260620_172716` | 2026-06-20 | 2 | 1 | all_evals.csv, status.json |
| `minor` | `BRIDGE_HUNT_20260620_161613` | 2026-06-20 | 2 | 1 | all_evals.csv, status.json |
| `minor` | `BRIDGE_HUNT_20260620_155828` | 2026-06-20 | 2 | 1 | all_evals.csv, status.json |
| `minor` | `BRIDGE_HUNT_20260620_121115` | 2026-06-20 | 2 | 1 | all_evals.csv, status.json |
| `minor` | `BRIDGE_CALIB_20260620_115935` | 2026-06-20 | 2 | 1 | all_evals.csv, status.json |
| `minor` | `BRIDGE_CALIB_20260620_114904` | 2026-06-20 | 2 | 1 | all_evals.csv, status.json |
| `minor` | `BRIDGE_CALIB_20260620_113635` | 2026-06-20 | 2 | 1 | all_evals.csv, status.json |
| `minor` | `AF_BRIDGE_CALIB_20260620_231010` | 2026-06-20 | 2 | 1 | all_evals.csv, status.json |
| `minor` | `ADAPTIVE_HUNT_20260620_082624` | 2026-06-20 | 2 | 1 | all_evals.csv, status.json |
| `minor` | `ADAPTIVE_HUNT_20260620_060311` | 2026-06-20 | 2 | 1 | all_evals.csv, status.json |
| `minor` | `ADAPTIVE_HUNT_20260620_012644` | 2026-06-20 | 2 | 1 | all_evals.csv, status.json |
| `minor` | `ADAPTIVE_HUNT_20260620_012257` | 2026-06-20 | 2 | 1 | all_evals.csv, status.json |
| `minor` | `STABLE_COLLAPSE_noise_20260619_101114` | 2026-06-19 | 2 | 1 | meta.json, stable_collapse_results.csv |
| `minor` | `STABLE_COLLAPSE_noise_20260619_100923` | 2026-06-19 | 2 | 1 | meta.json, stable_collapse_results.csv |
| `minor` | `STABLE_COLLAPSE_noise_20260619_100746` | 2026-06-19 | 2 | 1 | meta.json, stable_collapse_results.csv |
| `minor` | `STABLE_COLLAPSE_noise_20260619_100715` | 2026-06-19 | 2 | 1 | meta.json, stable_collapse_results.csv |
| `minor` | `STABLE_COLLAPSE_20260619_093409` | 2026-06-19 | 2 | 1 | meta.json, stable_collapse_results.csv |
| `minor` | `STABLE_COLLAPSE_20260619_093350` | 2026-06-19 | 2 | 1 | meta.json, stable_collapse_results.csv |
| `minor` | `FRESH_HUNT_20260619_192434` | 2026-06-19 | 2 | 1 | fresh_hunt_results.csv, meta.json |
| `minor` | `FRESH_HUNT_20260619_192353` | 2026-06-19 | 2 | 1 | fresh_hunt_results.csv, meta.json |
| `minor` | `ADAPTIVE_HUNT_20260619_203526` | 2026-06-19 | 2 | 1 | all_evals.csv, status.json |
| `minor` | `ADAPTIVE_HUNT_20260619_203127` | 2026-06-19 | 2 | 1 | all_evals.csv, status.json |
