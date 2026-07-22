# 04 · Phase D — C2 Transport & Two-Body (corrected substrate)

Sector verdict: on the fixed true-flat substrate, conservative NLS solitons transport cleanly (v=2Dk) and interact
via a π/2 phase-force; collisions are capture-dominated EXCEPT at exact anti-phase, where the destructive-
interference node yields a non-merging channel that transmits (pass-through) at intermediate speed and captures
above a sharp threshold — the SAME collision phase×speed structure as the C3/KG substrate (cross-substrate
universality of the collision diagram, RUN-2). (feb/a\* itself is structureless there — the movers need s<0
saturation + box-compatible D.) All post-C2.6-fix; see folder 03 for the correction.

## Evidence cards (manifest `EV-C27-01`, `EV-C29-01`, `EV-C210-ANTI`)
| claim | verdict | run (local) | data/figures | gate |
|---|---|---|---|---|
| true soliton translates at v=2Dk | CONFIRMED | `sweep_runs/C27_REDERIVE/` (r3_n96.json, r3_soliton_n96.npy) | — | Galilean identity |
| feb/a\* pure-NLS is structureless | FALSIFIED (has solitons) | `C27_REDERIVE/r2_feb.json` | — | g_max<box floor |
| moving families exist (s<0 + box-fit D) | CONFIRMED | `sweep_runs/C25_SCOUT_T1B_FIXED/` | family_scout.csv | 2 GALILEAN |
| static force crossover at Δφ=π/2 | CONFIRMED | `sweep_runs/C29_ROBUST/` (track_*.npz) | separation trajectories | momentum-density obs |
| in-phase collisions capture (non-integrable binding) | CONFIRMED | `C29_ROBUST/` | track_asym_*.npz | mass/P conserved |
| **anti-phase collision = node channel (pass-through → capture); RUN-2** | CONFIRMED | `sweep_runs/C2_10_ANTIPHASE{,_MID}/`, `C2_10_INPHASE/`, `C2_10_SMOKE/` (collide_*.npz) | separation/rad trajectories | mass drift ≤1.5e-3; v=2Dk boost 0.02% |
| **collision phase×speed structure universal (NLS≡KG); RUN-2** | CONFIRMED | (above vs `C3_*_LADDER`) | — | capture-generic + anti-phase pass-through |
| CFL re-baseline (all (N,dt) stable) | CONFIRMED | `C27_REDERIVE/r0_cfl.json` | — | full dispersion |

## Quick-smoke reproducibility
`C27_QUICK_V2/`, `C29_QUICK_V2/` — the bounded `--quick` smoke outputs (R0+R2 for C2.7; V0+one static for C2.9),
added so replication does not need N=96 long holds.

## Docs
`docs/PHASE_D_C2_7_REDERIVATION_RESULTS.md`, `PHASE_D_C2_9_TWONODE_ROBUST_RESULTS.md`,
`PHASE_D_C2_8_TWONODE_RESULTS.md` (+ C2.8b inconclusive note), `PHASE_D_C2_5_FAMILY_SCOUT_PLAN.md`,
`PHASE_D_C2_10_ANTIPHASE_COLLISION_RESULTS.md` (RUN-2: anti-phase collision map — bounce ≲0.42 → pass-through
0.47–0.57 → capture ≳0.61; harness `jax_scout/phase_d_c2_10_antiphase_collision.py`).
