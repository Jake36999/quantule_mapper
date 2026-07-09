# Candidate 2-Node / 3-Node Geometry Prior Index

Read-only search over existing Quantule Mapper artifacts. These candidates are geometry priors and diagnostic seed/layout candidates only; they do not prove stable 3-node configurations in the moving/conservative substrate.

## Search Scope
- Phase C node library: `F:\quantule_mapper\sweep_runs\PHASE_C_NODE_LIBRARY_20260704\PHASE_C_NODE_LIBRARY.json`
- Sweep artifacts: `sweep_runs/**/{diagnostic_summary.json,summary.json,frames.npz,probe_data.npz,*.h5}`
- Ledgers/provenance: `Simulation_ledgers/*`, `queue_runtime.db`, `live_hunt.db`, `provenance*.json`
- Ledger/provenance note: Searched Simulation_ledgers CSV/text, queue_runtime.db, live_hunt.db, and provenance*.json. These sources provided config/provenance context but no additional explicit final node_count==2/3 candidates beyond sweep/node-library diagnostics.

## Counts
- total indexed candidates: 53
- final 2-node candidates: 42
- final 3-node candidates: 11
- stable candidates: 18
- near-stable candidates: 3
- frame-backed candidates: 18

## Ranking Method
Score favors stable/near-stable labels, final node_count exactly 2 or 3, nearest-neighbour spacing inside the measured hold/coupling band (~0.20-0.62 box units), lower node-mass/peak imbalance, frame history, config hash availability, and longer T.

## Top Ranked Candidates
| score | final nodes | bucket | candidate | nn min | mass/peak CV | T | frames |
|---:|---:|---|---|---:|---:|---:|---|
| 93.931 | 3 | stable | `library::FEB_BASIN_CONFIRM_20260625_154503/K3_s20260621_T12000` | 0.32896802176515144 | 0.0707102213573354 | 12000 | False |
| 93.537 | 3 | stable | `library::FEB_BASIN_CONFIRM_20260625_154503/K3_s20260620_T12000` | 0.5090987364089534 | 0.3400557632651051 | 12000 | False |
| 92.743 | 2 | stable | `library::FEB_PARAM_BASIN_20260626_004039/revalid_K4_T24000` | 0.48493846009343433 | 0.762976190021199 | 24000 | False |
| 92.689 | 2 | stable | `library::FEB_PARAM_BASIN_20260626_004039/revalid_K3_T24000` | 0.5920012429074945 | 0.6059001663978397 | 24000 | False |
| 92.173 | 3 | stable | `library::FEB_PARAM_BASIN_20260626_004039/revalid_K5_T24000` | 0.4847786656186122 | 0.810193642617172 | 24000 | False |
| 85.245 | 2 | stable | `library::FEB_BASIN_20260625_122824/K4_perblob` | 0.4890328076245858 | 0.7280023328899051 |  | False |
| 85.122 | 2 | stable | `library::FEB_BASIN_20260625_122824/K3_perblob` | 0.5942650101859744 | 0.56606043431662 |  | False |
| 84.875 | 2 | stable | `diagnostic::sweep_runs/PHASE_C_VISUAL_ANALYSIS_20260624_161650/cases/k2_intermediate_true/diagnostic_summary.json` | 0.625 | 0.28816203535165263 | 4000.0 | True |
| 84.764 | 3 | stable | `library::FEB_BASIN_20260625_122824/K5_perblob` | 0.48904616196165535 | 0.7681100335230974 |  | False |
| 77.0 | 2 | stable | `diagnostic::sweep_runs/CORE_SAT_THRESHOLD_BRANCH_ROBUSTNESS_20260624_001934/k1_low_mass_branch_idx_4_m_4000_seed_20260619_base/diagnostic_summary.json` |  |  | 6000 | True |
| 77.0 | 2 | stable | `diagnostic::sweep_runs/CORE_SAT_THRESHOLD_BRANCH_ROBUSTNESS_20260624_001934/k1_low_mass_branch_idx_4_m_4000_seed_20260619_jitter_01/diagnostic_summary.json` |  |  | 6000 | True |
| 76.333 | 2 | stable | `diagnostic::sweep_runs/CORE_SAT_COLLAPSE_DIAG_20260623_160000/CORE_SAT_HUNT_20260623_113318_idx_4/diagnostic_summary.json` |  |  | 4000.0 | True |
| 76.333 | 2 | stable | `diagnostic::sweep_runs/CORE_SAT_COLLAPSE_DIAG_20260623_160000/CORE_SAT_HUNT_20260623_120758_idx_4/diagnostic_summary.json` |  |  | 4000.0 | True |
| 76.333 | 2 | stable | `diagnostic::sweep_runs/CORE_SAT_TRACE_COMPARE_20260623_185302/CORE_SAT_HUNT_20260623_171758_idx_4/diagnostic_summary.json` |  |  | 4000.0 | True |
| 76.333 | 2 | stable | `diagnostic::sweep_runs/CORE_SAT_TRACE_COMPARE_20260623_185450/CORE_SAT_HUNT_20260623_171758_idx_4/diagnostic_summary.json` |  |  | 4000.0 | True |

## Stable 2-Node / 3-Node Highlights

- `library::FEB_BASIN_CONFIRM_20260625_154503/K3_s20260621_T12000`: n=3, nn_min=0.32896802176515144, mass_cv=0.0707102213573354, T=12000, frames=False, artifact=`F:\quantule_mapper\sweep_runs\FEB_BASIN_CONFIRM_20260625_154503\K3_s20260621_T12000_probe.npz`
- `library::FEB_BASIN_CONFIRM_20260625_154503/K3_s20260620_T12000`: n=3, nn_min=0.5090987364089534, mass_cv=0.3400557632651051, T=12000, frames=False, artifact=`F:\quantule_mapper\sweep_runs\FEB_BASIN_CONFIRM_20260625_154503\K3_s20260620_T12000_probe.npz`
- `library::FEB_PARAM_BASIN_20260626_004039/revalid_K4_T24000`: n=2, nn_min=0.48493846009343433, mass_cv=0.762976190021199, T=24000, frames=False, artifact=`F:\quantule_mapper\sweep_runs\FEB_PARAM_BASIN_20260626_004039\revalid_K4_T24000_probe.npz`
- `library::FEB_PARAM_BASIN_20260626_004039/revalid_K3_T24000`: n=2, nn_min=0.5920012429074945, mass_cv=0.6059001663978397, T=24000, frames=False, artifact=`F:\quantule_mapper\sweep_runs\FEB_PARAM_BASIN_20260626_004039\revalid_K3_T24000_probe.npz`
- `library::FEB_PARAM_BASIN_20260626_004039/revalid_K5_T24000`: n=3, nn_min=0.4847786656186122, mass_cv=0.810193642617172, T=24000, frames=False, artifact=`F:\quantule_mapper\sweep_runs\FEB_PARAM_BASIN_20260626_004039\revalid_K5_T24000_probe.npz`
- `library::FEB_BASIN_20260625_122824/K4_perblob`: n=2, nn_min=0.4890328076245858, mass_cv=0.7280023328899051, T=, frames=False, artifact=`F:\quantule_mapper\sweep_runs\FEB_BASIN_20260625_122824\K4_perblob_probe.npz`
- `library::FEB_BASIN_20260625_122824/K3_perblob`: n=2, nn_min=0.5942650101859744, mass_cv=0.56606043431662, T=, frames=False, artifact=`F:\quantule_mapper\sweep_runs\FEB_BASIN_20260625_122824\K3_perblob_probe.npz`
- `diagnostic::sweep_runs/PHASE_C_VISUAL_ANALYSIS_20260624_161650/cases/k2_intermediate_true/diagnostic_summary.json`: n=2, nn_min=0.625, mass_cv=0.28816203535165263, T=4000.0, frames=True, artifact=`F:\quantule_mapper\sweep_runs\PHASE_C_VISUAL_ANALYSIS_20260624_161650\cases\k2_intermediate_true\frames.npz`
- `library::FEB_BASIN_20260625_122824/K5_perblob`: n=3, nn_min=0.48904616196165535, mass_cv=0.7681100335230974, T=, frames=False, artifact=`F:\quantule_mapper\sweep_runs\FEB_BASIN_20260625_122824\K5_perblob_probe.npz`
- `diagnostic::sweep_runs/CORE_SAT_THRESHOLD_BRANCH_ROBUSTNESS_20260624_001934/k1_low_mass_branch_idx_4_m_4000_seed_20260619_base/diagnostic_summary.json`: n=2, nn_min=, mass_cv=0.0, T=6000, frames=True, artifact=`F:\quantule_mapper\sweep_runs\CORE_SAT_THRESHOLD_BRANCH_ROBUSTNESS_20260624_001934\k1_low_mass_branch_idx_4_m_4000_seed_20260619_base\frames.npz`

## Near-Stable / Long-Lived Controls

- `diagnostic::sweep_runs/CORE_SAT_COLLAPSE_DIAG_20260623_160000/CORE_SAT_HUNT_20260623_113318_idx_3/diagnostic_summary.json`: n=2, bucket=near-stable, verdict=NEAR_SATURATED_BOUND_STATE, frames=True, caveat=Diagnostic replay/frame analysis; detector spacing derived from rendered/final field when not in node library. Phase C dissipative prior only. 
- `library::FEB_BASIN_CONFIRM_20260625_154503/K3_s20260619_T24000`: n=2, bucket=spin-down / long-lived control, verdict=SPIN_DOWN_REJECT, frames=False, caveat=Phase C dissipative geometry prior only; not evidence for conservative/moving substrate. No frame history found; final/mid/final probe only. 
- `diagnostic::sweep_runs/PHASE_C_N96_LONGT_CONTROL_20260625_083731/k4_T24000/diagnostic_summary.json`: n=3, bucket=spin-down / long-lived control, verdict=SPIN_DOWN_REJECT, frames=True, caveat=Diagnostic replay/frame analysis; detector spacing derived from rendered/final field when not in node library. Phase C dissipative prior only. 
- `diagnostic::sweep_runs/CORE_SAT_COLLAPSE_DIAG_20260623_160000/CORE_SAT_HUNT_20260623_113318_idx_2/diagnostic_summary.json`: n=2, bucket=reject / caution, verdict=LATE_BLOWUP_REJECT, frames=True, caveat=Diagnostic replay/frame analysis; detector spacing derived from rendered/final field when not in node library. Phase C dissipative prior only. 
- `diagnostic::sweep_runs/CORE_SAT_COLLAPSE_DIAG_20260623_160000/CORE_SAT_HUNT_20260623_123527_idx_2/diagnostic_summary.json`: n=2, bucket=reject / caution, verdict=LATE_BLOWUP_REJECT, frames=True, caveat=Diagnostic replay/frame analysis; detector spacing derived from rendered/final field when not in node library. Phase C dissipative prior only. 
- `diagnostic::sweep_runs/CORE_SAT_THRESHOLD_DIAG_20260623_180519/replays/CORE_SAT_HUNT_20260623_170944_idx_3/diagnostic_summary.json`: n=2, bucket=near-stable, verdict=NEAR_SATURATED_BOUND_STATE, frames=False, caveat=Diagnostic replay/frame analysis; detector spacing derived from rendered/final field when not in node library. Phase C dissipative prior only. No frame history. 
- `diagnostic::sweep_runs/CORE_SAT_THRESHOLD_DIAG_20260623_180519/replays/CORE_SAT_HUNT_20260623_171758_idx_3/diagnostic_summary.json`: n=2, bucket=near-stable, verdict=NEAR_SATURATED_BOUND_STATE, frames=False, caveat=Diagnostic replay/frame analysis; detector spacing derived from rendered/final field when not in node library. Phase C dissipative prior only. No frame history. 
- `diagnostic::sweep_runs/CORE_SAT_TRACE_COMPARE_20260623_185450/CORE_SAT_HUNT_20260623_173417_idx_2/diagnostic_summary.json`: n=2, bucket=reject / caution, verdict=LATE_BLOWUP_REJECT, frames=True, caveat=Diagnostic replay/frame analysis; detector spacing derived from rendered/final field when not in node library. Phase C dissipative prior only. 
- `diagnostic::sweep_runs/CORE_SAT_COLLAPSE_DIAG_20260623_160000/CORE_SAT_HUNT_20260623_113318_idx_6/diagnostic_summary.json`: n=2, bucket=spin-down / long-lived control, verdict=SPIN_DOWN_REJECT, frames=True, caveat=Diagnostic replay/frame analysis; detector spacing derived from rendered/final field when not in node library. Phase C dissipative prior only. 
- `diagnostic::sweep_runs/CORE_SAT_COLLAPSE_DIAG_20260623_160000/CORE_SAT_HUNT_20260623_113318_idx_7/diagnostic_summary.json`: n=2, bucket=spin-down / long-lived control, verdict=SPIN_DOWN_REJECT, frames=True, caveat=Diagnostic replay/frame analysis; detector spacing derived from rendered/final field when not in node library. Phase C dissipative prior only. 

## Rendered Frame-Backed Examples

- `diagnostic::sweep_runs/PHASE_C_VISUAL_ANALYSIS_20260624_161650/cases/k2_intermediate_true/diagnostic_summary.json`
  - rho GIF: `F:\quantule_mapper\quantule_viz\outputs\candidate_2node_3node_search\renders\top1_diagnostic_sweep_runs_PHASE_C_VISUAL_ANALYSIS_20260624_161650_cases_k2_intermediate_true_d\rho.gif`
  - rho montage: `F:\quantule_mapper\quantule_viz\outputs\candidate_2node_3node_search\renders\top1_diagnostic_sweep_runs_PHASE_C_VISUAL_ANALYSIS_20260624_161650_cases_k2_intermediate_true_d\rho_montage.png`
  - rho-masked phase montage: `F:\quantule_mapper\quantule_viz\outputs\candidate_2node_3node_search\renders\top1_diagnostic_sweep_runs_PHASE_C_VISUAL_ANALYSIS_20260624_161650_cases_k2_intermediate_true_d\phase_masked_montage.png`
  - omega/geometry: skipped: params unavailable for actual unified_omega logic
- `diagnostic::sweep_runs/CORE_SAT_THRESHOLD_BRANCH_ROBUSTNESS_20260624_001934/k1_low_mass_branch_idx_4_m_4000_seed_20260619_base/diagnostic_summary.json`
  - rho GIF: `F:\quantule_mapper\quantule_viz\outputs\candidate_2node_3node_search\renders\top2_diagnostic_sweep_runs_CORE_SAT_THRESHOLD_BRANCH_ROBUSTNESS_20260624_001934_k1_low_mass_bra\rho.gif`
  - rho montage: `F:\quantule_mapper\quantule_viz\outputs\candidate_2node_3node_search\renders\top2_diagnostic_sweep_runs_CORE_SAT_THRESHOLD_BRANCH_ROBUSTNESS_20260624_001934_k1_low_mass_bra\rho_montage.png`
  - rho-masked phase montage: `F:\quantule_mapper\quantule_viz\outputs\candidate_2node_3node_search\renders\top2_diagnostic_sweep_runs_CORE_SAT_THRESHOLD_BRANCH_ROBUSTNESS_20260624_001934_k1_low_mass_bra\phase_masked_montage.png`
  - omega/geometry: rendered via gravity.unified_omega.derive_stable_conformal_factor with param_skip_topology_cap=True -> `F:\quantule_mapper\quantule_viz\outputs\candidate_2node_3node_search\renders\top2_diagnostic_sweep_runs_CORE_SAT_THRESHOLD_BRANCH_ROBUSTNESS_20260624_001934_k1_low_mass_bra\omega_unified_omega_montage.png`
- `diagnostic::sweep_runs/PHASE_C_N96_LONGT_CONTROL_20260625_083731/k4_T24000/diagnostic_summary.json`
  - rho GIF: `F:\quantule_mapper\quantule_viz\outputs\candidate_2node_3node_search\renders\top3_diagnostic_sweep_runs_PHASE_C_N96_LONGT_CONTROL_20260625_083731_k4_T24000_diagnostic_summa\rho.gif`
  - rho montage: `F:\quantule_mapper\quantule_viz\outputs\candidate_2node_3node_search\renders\top3_diagnostic_sweep_runs_PHASE_C_N96_LONGT_CONTROL_20260625_083731_k4_T24000_diagnostic_summa\rho_montage.png`
  - rho-masked phase montage: `F:\quantule_mapper\quantule_viz\outputs\candidate_2node_3node_search\renders\top3_diagnostic_sweep_runs_PHASE_C_N96_LONGT_CONTROL_20260625_083731_k4_T24000_diagnostic_summa\phase_masked_montage.png`
  - omega/geometry: rendered via gravity.unified_omega.derive_stable_conformal_factor with param_skip_topology_cap=True -> `F:\quantule_mapper\quantule_viz\outputs\candidate_2node_3node_search\renders\top3_diagnostic_sweep_runs_PHASE_C_N96_LONGT_CONTROL_20260625_083731_k4_T24000_diagnostic_summa\omega_unified_omega_montage.png`

## Caveats
- Node-library spacing/mass metrics are preferred where available; diagnostic-only spacing/peak balance was derived from saved fields with a simple high-density component detector.
- Several high-ranked stable 3-node candidates are probe-only (`psi0/psi_mid/psi_fin` or final-only) and therefore lack full frame history.
- Frame-backed 3-node examples found in this search are spin-down/long-lived controls, not stable 3-node proof cases.
- All candidates are from Phase C/dissipative saved artifacts unless explicitly labeled otherwise; they are geometry priors only for future diagnostics.
- Omega render is emitted only when saved params were available and `gravity.unified_omega.derive_stable_conformal_factor` could be called; otherwise it is skipped rather than approximated.

## Short Recommendation
- Best 3-node geometry prior: `library::FEB_BASIN_CONFIRM_20260625_154503/K3_s20260621_T12000` (stable label, balanced masses/peaks, clean three-node count; use as a seed/layout target even if frame history is absent).
- Best 2-node geometry prior: `library::FEB_PARAM_BASIN_20260626_004039/revalid_K4_T24000` (stable/long-lived two-node final state with spacing in the measured hold/coupling range).
- Best frame-backed diagnostic render: `diagnostic::sweep_runs/PHASE_C_VISUAL_ANALYSIS_20260624_161650/cases/k2_intermediate_true/diagnostic_summary.json` (stable 2-node case with full frame history, good for immediate visual/diagnostic pipeline testing).
- Best 3-node frame-backed control: `diagnostic::sweep_runs/PHASE_C_N96_LONGT_CONTROL_20260625_083731/k4_T24000/diagnostic_summary.json` (only frame-backed 3-node case found; useful as a caution/control, not as a stable 3-node claim).
