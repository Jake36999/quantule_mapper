# IRER Gravity Ladder Geometry Characterization / De-Saturation Audit

Scope: read-only characterization of existing Rung A+D telemetry. No probe/path-bending run was performed, and no production solver, Hunter, validation, Phase C path, or default config was modified.

## Inputs

- Source runs: `GRAVITY_AD_bg0, GRAVITY_AD_bgvac`
- Git commit: `584ad260c9369915cb598f6de3c7f4854adf1d65`
- Python: `F:\quantule_mapper\.venv\Scripts\python.exe`
- CuPy/GPU metadata: `{'available': True, 'version': '14.0.1', 'device_count': 1, 'gpu_name': 'NVIDIA GeForce GTX 1080'}`
- Protected-file diff: `empty`

## Production Geometry Law

- rho_vac: `1.1866`
- a_coupling: `2.3098`
- softclip beta: `3.0`
- Omega^2 window: `[1e-09, 1000000.0]`
- Curve evaluation range: rho `7.886e-10` to `2.000e+00`
- Region counts across sampled curve: `{'cap_saturated': 364, 'flat_softclip': 5, 'graded': 143}`

The implemented Omega^2(rho) curve is dominated by the log-space soft clip over the saved load density range. The bg0 radial profile retains the previously observed pattern: a low, nearly flat dense-core response followed by a jump toward the high-Omega^2 ambient/cap region near the load boundary. This is a geometry-law characterization result only, not a gravity interpretation.

## Saved Rung A+D Cross-Check

- bg0 verdict: `A_D_GEOMETRY_RESPONSE_PERSISTENT_AND_DISTINCT`; n_nodes `1`, omega_center `534.7776299919985`, omega_far `928815.7416030684`, tensor_shear `1.076747844617899`.
- bgvac verdict: `A_D_LOAD_NOT_PERSISTENT`; n_nodes `0`, omega_center `727.1180585888434`, omega_far `727.1180632154775`.

## Diagnostic Soft-Clip Scan

The scan below is diagnostic-only. It does not change defaults and should be read as a map of possible non-saturated parameter regimes for future geometry-contract review.

| rank | beta | omega_min | omega_max | score | graded_fraction | cap_fraction | flat_fraction | dyn_range_log10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.75 | 1e-06 | 1e+08 | 1.300 | 1.000 | 0.000 | 0.000 | 6.151 |
| 2 | 0.75 | 1e-03 | 1e+08 | 1.300 | 1.000 | 0.000 | 0.000 | 6.197 |
| 3 | 0.75 | 1e-02 | 1e+08 | 1.300 | 1.000 | 0.000 | 0.000 | 6.146 |
| 4 | 0.75 | 1e-01 | 1e+08 | 1.300 | 1.000 | 0.000 | 0.000 | 6.033 |
| 5 | 1.0 | 1e-12 | 1e+07 | 1.300 | 1.000 | 0.000 | 0.000 | 6.120 |

## C2 Prime Cross-Reference

`docs/PHASE_D_C2PRIME_CANONICAL_GEOMETRY_RFC.md` is the relevant principled alternative to the current soft-clip-dominated geometry contract. This audit does not select or alter that contract; it provides the geometry-law data needed for that decision.

## Outputs

- `geometry_characterization_metadata.json`
- `source_run_summary.csv`
- `geometry_law_curve.csv` and `geometry_law_curve.png`
- `geometry_law_log_slope.png`
- `softclip_parameter_scan.csv` and `softclip_scan_top_candidates.png`
- `radial_profiles.csv` and `radial_omega_profiles.png`

## Reproducible Command

```powershell
F:\quantule_mapper\.venv\Scripts\python.exe tools\analyze_gravity_geometry_characterization.py --out F:\quantule_mapper\quantule_viz\outputs\gravity_geometry_characterization
```

## Attestation

- No production solver change.
- No Hunter, validation, default config, or Phase C path change.
- No probe/path-bending run.
- No gravity claim.
- Existing saved outputs were used for telemetry.
