# Harness Depth Audit

Status: provisional until Claude review.

This audit separates formalism alignment from behavioural comparison. IRER formulations remain primary; external models are comparison instruments.

| Metric | Status | Validation depth | Behavioural result | Files read | Claude-reviewable? | Numeric fields used |
|---|---|---|---|---:|---|---|
| `v1` | `RAN` | `INTERNAL_DATA_ANALYTIC_FIT` | `PASS` | 13 | True | acceleration_window, c2_fit_quality, c3_fit_quality, crossover_error_from_pi_over_2, crossover_phase_estimate, fit_C, fit_lambda, force_sign_proxy, n_files_used, n_non_neutral, n_phase_points, n_rows, normalized_rms_residual, phase_sign_accuracy, phase_sign_accuracy_qddot_legacy, pi_over_2_neutral_rows, q_convention, rms_residual |
| `v2` | `RAN` | `INTERNAL_DATA_BEHAVIOUR_CHECK` | `PASS` | 1 | True | D_used, expected_slope_2D, intercept, mass_retention_min, mean_mass_retention, n_boosts, n_points, r_squared, slope, slope_percent_error, source_of_D |
| `v3` | `RAN` | `INTERNAL_DATA_BEHAVIOUR_CHECK` | `PASS` | 1 | True | Q_max, Q_min, dQ_domega_fit, monotone_decreasing, monotonic_decreasing, n_points, omega_max, omega_min, sign, sign_pass |
| `v5` | `RAN` | `INTERNAL_DATA_BEHAVIOUR_CHECK` | `PASS` | 5 | True | anti_phase_transmission_cells, max_charge_drift, max_dE_rel, max_dQ_rel, max_energy_drift, missing_phase_speed_cells, n_cells, n_missing_cells, non_antiphase_capture_fraction, outcome_counts, outcomes, phase_values, rows, speed_values |
| `v6` | `RAN` | `INTERNAL_DATA_BEHAVIOUR_CHECK` | `PASS` | 3 | True | clean_overlay_available, corrected_interpretation_status, corrected_mean_v_frac, corrected_v_over_2Dk, expected_bug_ratio_1_over_151, old_D_eff_ratio, old_interpretation_status |
| `v7` | `RAN` | `INTERNAL_DATA_BEHAVIOUR_CHECK` | `PARTIAL` | 3 | True | bec_reference_a, bec_reference_gamma, cap_fraction, cliff_detected, design_constraint_only, effective_gamma, effective_steeper_than_nominal, fitted_effective_exponent, floor_fraction, graded_fraction, implied_gamma, n_radial_points, nominal_a, nominal_gamma, production_a, rows, same_sign_as_bec_analogue, unsaturated_points |

## Summary

- Behavioural/internal-data metrics: 6
- External-data correlations: 0
- Formalism-only or weak metrics: 0

No metric is treated as physical correspondence. Metrics marked weak should not be used as scientific results.
