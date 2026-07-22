# 05 · Phase D — C3 Klein–Gordon / Q-ball

Sector verdict: a genuinely-conservative second-order (KG) substrate hosting a VK-stable Q-ball that transports
inertially, shares the NLS π/2 two-body law (cross-substrate universality), and whose collisions form a mapped
phase×speed diagram (capture generic; transmission a narrow anti-phase node feature).

## Evidence cards (manifest `EV-C3-01`, `EV-C3-2QB`, `EV-C3-PHASE`)
| claim | verdict | run (local) | data | gate |
|---|---|---|---|---|
| Strang stepper exact vs analytic rotation | CONFIRMED (max\|Δ\|=0) | `sweep_runs/C3_WAVE*` | — | G1 linear parity |
| Q-ball exists at the C2.7-mapped point | CONFIRMED | `sweep_runs/C3_WAVE_MAPPED/` (qball.npy) | — | Petviashvili residual 1e-9 |
| E + U(1) charge conserved ~1e-13 | CONFIRMED | `C3_WAVE_BOOSTFIX2/` | summary.json | G2 |
| inertial transport (density co-moves) | CONFIRMED | `C3_WAVE_BOOSTFIX2/` | — | G4 (corrected boost IC) |
| contraction is NOT the fidelity limiter | FALSIFIED | `C3_EXACT_VK2/` | — | v_frac unchanged |
| VK-stable (dQ/dω=−849<0) | CONFIRMED | `C3_EXACT_VK2/` | Q(ω) points | G6 |
| KG two-Q-ball = same π/2 law as NLS | CONFIRMED | `C3_TWOQBALL_FULL/` (pair_*.npz) | separation | dE_rel~1e-9 |
| in-phase collisions capture to 0.75c | CONFIRMED | `C3_COLLISION_LADDER_FULL/` (collide_*.npz) | — | dE_rel≤5.6e-6 |
| anti-phase transmits below ~0.5c | CONFIRMED (narrow) | `C3_ANTIPHASE_LADDER/` | — | elasticity, rad |
| off-phase (π/2,3π/4,7π/8) all capture | CONFIRMED | `C3_PHASE_{pi2,3pi4,7pi8}/` | — | classifier |
| static force law governs collisions | FALSIFIED | (above) | — | 3π/4 repels statically, captures dynamically |
| collision phase×speed structure is cross-substrate (NLS analog holds) | CONFIRMED (RUN-2) | `sweep_runs/C2_10_*` (see folder 04) | — | NLS anti-phase pass-through→capture mirrors this diagram |

## Docs
`docs/PHASE_D_C3_WAVE_KINETIC_RFC.md`, `PHASE_D_C3_WAVE_KINETIC_RESULTS.md`, `PHASE_D_C3_TWOQBALL_RESULTS.md`,
`PHASE_D_C3_COLLISION_LADDER_RESULTS.md`. Boost-IC bug: folder 07.
