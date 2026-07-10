# 06 · Gravity Ladder (PAUSED)

Sector status: the geometry-density channel is *live* (a coherent load produces a geometry-dependent Ω²/T_info
response), but the production Ω²(ρ) is a saturation cliff, not a graded potential; de-saturation makes it worse.
**Paused** at a documented gate. No probe / path-bending / rung-B evidence exists (rung B is blocked) — and none
should be added until the re-entry condition is met.

## Evidence cards (manifest `EV-GR-AD`, `EV-GR-DESAT`)
| claim | verdict | run (local) | figures | note |
|---|---|---|---|---|
| coherent load → geometry-dependent Ω²/T_info response | CONFIRMED (entangled) | `sweep_runs/GRAVITY_AD_bg0/` (load_*.npy) | radial profile (below) | geometry-off → load disperses (shear 1.35→1e-33) |
| the response is a graded gravity-like potential | FALSIFIED | `GRAVITY_AD_bg0/summary.json` | radial Ω² (cliff) | Ω²~750 core → ~9e5 at r>1.35 |
| bg=ρ_vac gives a clean unsaturated well | FALSIFIED | `GRAVITY_AD_bgvac/` | — | load dissolves on filled bg |
| de-saturating the soft-clip fixes the cliff | FALSIFIED | `sweep_runs/GRAVITY_DESAT_PILOT/` | — | cliff 250× steeper (vacuum-ρ divergence, not the cap) |
| C2′ unblocks the ladder | FALSIFIED | (contract review) | — | same Ω²(ρ) factor |

## Figures (local)
`sweep_runs/PHASE_D_CODEX_REPRODUCTION_20260709_233433/gravity_geometry/`:
`geometry_law_curve.png`, `radial_omega_profiles.png`, `softclip_scan_top_candidates.png`, `geometry_law_log_slope.png`
— the Ω²(ρ) characterization + radial-cliff evidence. Candidate for a small promoted figure in `docs/` (gap list).

## Re-entry condition (documented)
A validated **stable-overdense-load-on-ρ_vac-background** regime (vacuum at ρ≈ρ_vac + load core ρ>ρ_vac + persistent)
so Ω²(ρ) becomes graded — a research sub-project — before any neutral-probe rung.

## Framing (binding)
"geometry-density interaction", "candidate mechanism", "not yet an emergent-gravity claim". Never "mass attracts
mass". Docs: `docs/IRER_GRAVITY_RUNG_A_D_RESULTS.md`, `GRAVITY_LADDER_GEOMETRY_DECISION.md`,
`IRER_GEOMETRY_DENSITY_GRAVITY_RFC.md`, `GRAVITY_LADDER_CODEX_HANDOFF.md`.
