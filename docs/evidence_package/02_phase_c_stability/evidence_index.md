# 02 · Phase C — Stability Sector (CLOSED)

Sector verdict: a\*≈×1.15 is a real long-time gain/loss-balanced attractor, site-pinned; its mobility and every
structural predictor (prime-SSE, TDA, tensor-routing, Payan, vortex) are null. Sector **closed**.

## Evidence cards (see manifest `EV-PHC-01`, `EV-PHC-02`)
| claim | verdict | run (local) | figures | doc/gate |
|---|---|---|---|---|
| a\* is a long-time stable attractor | CONFIRMED | `sweep_runs/PHASE_C_OPTION_B_N96_TRACE_20260625_003926/` (frames.npz) | `sweep_runs/PHASE_C_VISUAL_ANALYSIS_V2_20260624_185426/cases/*` (PNG panels) | er-slope≤1.5e-4; catalog C-1 |
| stable nodes near-current-free | CONFIRMED | `sweep_runs/PHASE_C_N96_CURRENT_CLOSURE_20260625_001621/` (panels) | `case_metric_panels/*` | catalog D3 |
| nodes are rotational cores, not topological vortices | FALSIFIED | `sweep_runs/SUBSTRATE_HUNT_20260621_161557/hifi_*` | `hifi_*` panels | hi-fi continuation |
| prime-SSE / TDA / routing / Payan predict stability | NULL | Phase C harnesses | — | catalog C-4..C-9 |
| a\* mobility (kick / static well) | FALSIFIED | Phase C mobility probes | — | catalog C-2/C-3 |
| stability objective re-discovers a\* (H7) | CONFIRMED | H7 rediscovery run | — | catalog C-11 |

## Figures worth promoting (candidate, Codex enrichment)
The `PHASE_C_VISUAL_ANALYSIS_V2` case panels and the `k6_mid_mass_true_emergence` gifs
(`quantule_viz/outputs/k6_mid_mass_true_emergence/`) are the most illustrative Phase C visuals. Currently local
(gitignored). Candidate for a small promoted contact-sheet in `docs/` — see `../EVIDENCE_GAPS.md`.

## Docs
`docs/PHASE_C_*` (per catalog rows C-1..C-11); Hunter re-aim in the `phase-c-...morphology-null` memory lineage.
