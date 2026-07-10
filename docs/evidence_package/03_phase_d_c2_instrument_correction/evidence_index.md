# 03 · C2 Instrument Correction (old vs corrected)

The pivot folder. It documents the C2 conservative "false-negative" period, the C2.6 geometry-off bug that caused it,
and the corrected transport that superseded it. This is *why older results changed* — presented as maturity evidence.

## The old-vs-corrected comparison (the core of this folder)
| aspect | OLD (pre-C2.6, RETRACTED) | CORRECTED (post-fix, CANONICAL) |
|---|---|---|
| geometry-off switch | `param_a_coupling=0` → Ω²≈151 → **D_eff = D/151** (secretly ~99.3% of dispersion cancelled) | `param_geom_off=True` → geom_fac=0 → **true flat**; default byte-identical to baseline |
| single-soliton transport | "pinned / flow-through", drag μ≈0.036 | **v = 2Dk exactly** (v/2Dk=0.9999, mass 0.9999, N=96) |
| the "drag" number | μ≈0.036 read as a physical pinning coefficient | μ≈0.036 = **2·D_eff** exactly — the Galilean velocity of the *broken* substrate |
| C2.3 "no soliton" | quasi-soliton, no stationary state | existence machinery correct; the *evolution* had run D/151 (mismatched equations) |
| verdict | "conservative transport arc closed negative" | conservative transport **positive** for s<0 + box-fit families |
| runs | `sweep_runs/C23_N96/`, `C24_LOCAL_N96/`, `PHASE_D_C2_2_LOSS_20260704_231230/` | `sweep_runs/C27_REDERIVE/`, `C25_SCOUT_T1B_FIXED/` |

**How the bug was caught:** a *linear* wave packet must translate at v=2Dk (a Galilean identity, no theory
ambiguity). On the old path it moved at ~1/151 of that → a contradiction with a known identity → instrument fault.
See proof ledger Report 3 §2, and `docs/PHASE_D_C2_6_GEOMETRY_OFF_BUG_REPORT.md`.

## Retracted verdicts (first-class, preserved)
- **C2.1b** "native solitons transport lossily/ballistically" — the μ≈0.04 was 2·D_eff.
- **C2.2b** "loss is geometry-Galilean-breaking / purely dt-numerical" — measured on the bugged substrate.
- **C2.3b** "velocity anomaly = ring-winding drag" — artifact.
- **C2.4** "STILL_PINNED_PHYSICAL / flow-through mechanism" — the pinning never existed.
- Aggregate "conservative transport arc closed negative" — **overturned**.
All are retained in `docs/PHASE_D_C2_*_RESULTS.md` with retraction notes, and in the catalog with `RETRACTED` tags.

## Independent audit
Codex independently re-audited the fix (`sweep_runs/PHASE_D_C2_6_CODEX_AUDIT_20260709/`,
`PHASE_D_CODEX_REPRODUCTION_20260709_233433/c2_6_*`): default parity max|Δ|=0.0, `param_geom_off` truly flat,
old-bug D_eff ratio **0.006511 ≈ 1/151**, geometry-off flux ~1e-16, and true-soliton transport agreeing between
ETDRK4 and an independent RK4 integrator (mass 0.99992 vs 1.0000000). Two codebases, same conclusion.

## Manifest entries
`EV-C2-OLD` (retracted), `EV-C26-BUG` (canonical), `EV-C27-01` (canonical) — see
`../09_machine_readable/evidence_manifest.json`.

## Docs
`docs/PHASE_D_C2_6_GEOMETRY_OFF_BUG_REPORT.md`, `PHASE_D_C2_6_CODEX_AUDIT_HANDOVER.md`,
`PHASE_D_C2_7_REDERIVATION_RESULTS.md`, `PHASE_D_C2_CONTRACT_REVIEW.md`; proof ledger Report 3 §2/§7.
