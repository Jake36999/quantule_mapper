# Quantule Emergence Visualisation Metadata

This is a read-only visualisation pass over saved artifacts. It does not modify solver, physics, Hunter, validation gates, or configs. Visuals alone are not treated as scientific evidence.

## Source
- source artifact: `F:\quantule_mapper\sweep_runs\PHASE_C_OPTION_B_N96_TRACE_20260625_003926\k6_mid_mass_true\frames.npz`
- source SHA256: `d38b1d4e39018ee632b19b40a0ffdbba0cb8329b89c75ea67d3e9d9a28f0a23f`
- summary metadata: `F:\quantule_mapper\sweep_runs\PHASE_C_OPTION_B_N96_TRACE_20260625_003926\k6_mid_mass_true\summary.json`
- config hash / proxy: `f27dda3c1d915ac2b63f97567aa47227b6e606aaab789ced184011c6bf36f529`
- git commit: `8d8e97100fb33535a1285493d0c8ac4da6e79572`
- psi dataset key: `psi`
- time dataset key: `times`
- frame shape: `(41, 96, 96, 96)`
- timestep range rendered: `0.0` to `6000.0`

## Render Settings
- rho_formula: `abs(psi)^2`
- rho_projection: `max over z axis`
- rho_color_scale: `0 to p99.7=0.550465`
- phase_render: `angle(psi) on fixed final-peak slice axis=0, index=24`
- selected_frame_indices: `[0, 5, 10, 20, 30, 40]`
- gif_fps: `6`
- spatial_stride: `1`
- time_downsampling: `none`
- omega / geometry render: rendered as derived local proxy `(param_rho_vac / rho)^param_a_coupling`

## Outputs
- `F:\quantule_mapper\quantule_viz\outputs\k6_mid_mass_true_emergence\rho_emergence.gif`
- `F:\quantule_mapper\quantule_viz\outputs\k6_mid_mass_true_emergence\rho_still_000.png`
- `F:\quantule_mapper\quantule_viz\outputs\k6_mid_mass_true_emergence\rho_still_005.png`
- `F:\quantule_mapper\quantule_viz\outputs\k6_mid_mass_true_emergence\rho_still_010.png`
- `F:\quantule_mapper\quantule_viz\outputs\k6_mid_mass_true_emergence\rho_still_020.png`
- `F:\quantule_mapper\quantule_viz\outputs\k6_mid_mass_true_emergence\rho_still_030.png`
- `F:\quantule_mapper\quantule_viz\outputs\k6_mid_mass_true_emergence\rho_still_040.png`
- `F:\quantule_mapper\quantule_viz\outputs\k6_mid_mass_true_emergence\rho_stills_montage.png`
- `F:\quantule_mapper\quantule_viz\outputs\k6_mid_mass_true_emergence\phase_emergence.gif`
- `F:\quantule_mapper\quantule_viz\outputs\k6_mid_mass_true_emergence\phase_still_000.png`
- `F:\quantule_mapper\quantule_viz\outputs\k6_mid_mass_true_emergence\phase_still_005.png`
- `F:\quantule_mapper\quantule_viz\outputs\k6_mid_mass_true_emergence\phase_still_010.png`
- `F:\quantule_mapper\quantule_viz\outputs\k6_mid_mass_true_emergence\phase_still_020.png`
- `F:\quantule_mapper\quantule_viz\outputs\k6_mid_mass_true_emergence\phase_still_030.png`
- `F:\quantule_mapper\quantule_viz\outputs\k6_mid_mass_true_emergence\phase_still_040.png`
- `F:\quantule_mapper\quantule_viz\outputs\k6_mid_mass_true_emergence\phase_stills_montage.png`
- `F:\quantule_mapper\quantule_viz\outputs\k6_mid_mass_true_emergence\omega_proxy_emergence.gif`
- `F:\quantule_mapper\quantule_viz\outputs\k6_mid_mass_true_emergence\omega_proxy_still_000.png`
- `F:\quantule_mapper\quantule_viz\outputs\k6_mid_mass_true_emergence\omega_proxy_still_005.png`
- `F:\quantule_mapper\quantule_viz\outputs\k6_mid_mass_true_emergence\omega_proxy_still_010.png`
- `F:\quantule_mapper\quantule_viz\outputs\k6_mid_mass_true_emergence\omega_proxy_still_020.png`
- `F:\quantule_mapper\quantule_viz\outputs\k6_mid_mass_true_emergence\omega_proxy_still_030.png`
- `F:\quantule_mapper\quantule_viz\outputs\k6_mid_mass_true_emergence\omega_proxy_still_040.png`
- `F:\quantule_mapper\quantule_viz\outputs\k6_mid_mass_true_emergence\omega_proxy_stills_montage.png`

## Limitations
- Density is rendered as `rho = abs(psi)^2` from saved frames only.
- Density GIF uses a max-intensity projection for visibility; selected stills use the same projection.
- Phase render is a fixed final-peak slice and is masked only by the visual colormap, not by an analytic phase-quality gate.
- Derived omega is a local proxy from saved params and rho, not a stored solver omega field unless explicitly stated otherwise.
- No scientific claims should be made from these visuals alone.
