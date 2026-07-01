# Phase C N96 Validation Plan

## 1. Purpose

Validate the most informative K-varied Phase C candidates at higher fidelity:

- `N = 96`
- `T = 6000`

This is a replay-and-validate step, not a new random search. Each validation should reuse the saved
parameter vector, saved `ic_blobs`, fixed IC family, and the same Phase C saturation classifier.

## 2. Validation Path

Replay helper:

```powershell
python jax_scout\core_saturation_replay.py --csv <all_evals.csv> --idx <row_idx> --N 96 --T 6000 --out <outdir>
```

Reference control:

```powershell
python jax_scout\core_saturation_replay.py --ref feb56dc7 --N 96 --T 6000 --out <outdir>
```

The helper saves:

- `summary.json`
- `probe_data.npz`

`summary.json` records:

- candidate source
- class
- `er_fin`, `er_max`, `late_slope`
- `n_mid`, `n_fin`, `core_fin`
- `N`, `T`, `L`, `dt`
- IC seed
- classifier spec
- current git commit
- exact replay command

## 3. Controls

### Control A — `ref_feb56dc7`

Expected:

- `TRUE_SATURATED_BOUND_STATE`
- `n_fin = 4`
- flat late slope
- bounded energy saturation
- localized late state

Observed:

`FEB_CONTROL_REPRODUCED`

Current replay result:

- class: `TRUE_SATURATED_BOUND_STATE`
- `n_mid = 4`
- `n_fin = 4`
- `late_slope = -2.316e-06`
- `er_fin = 1.5781`

### Control B — `idx 623`

Purpose:

- prove the replay helper works for a normal CSV-backed hunt row, not only the built-in reference

Observed:

- class: `TRUE_SATURATED_BOUND_STATE`
- `n_mid = 4`
- `n_fin = 4`
- `late_slope = -2.951e-05`
- `er_fin = 0.8313`

## 4. Shortlist

Primary shortlist:

| idx | K | N=48/T=4000 class | N=48 final nodes | Why keep it |
| --- | ---: | --- | ---: | --- |
| `663` | `1` | `TRUE_SATURATED_BOUND_STATE` | `2` | best `K=1` TRUE by low `|late_slope|` |
| `743` | `2` | `TRUE_SATURATED_BOUND_STATE` | `1` | best `K=2` TRUE and best low-node TRUE |
| `154` | `3` | `TRUE_SATURATED_BOUND_STATE` | `2` | best `K=3` TRUE |
| `570` | `4` | `TRUE_SATURATED_BOUND_STATE` | `3` | best `K=4` TRUE and strongest localization |
| `623` | `6` | `TRUE_SATURATED_BOUND_STATE` | `4` | best `K=6` TRUE; closest feb-like hunt row |
| `744` | `2` | `NEAR_SATURATED_BOUND_STATE` | `2` | low-K fallback near-miss |
| `ref_feb56dc7` | `6` | reference control | `4` | known long-time 4-node attractor |

## 5. Reporting Template

For each validation replay, record:

- source row / reference name
- `K`
- `N=48/T=4000` class
- `N=96/T=6000` class
- `n_mid`
- `n_fin`
- `late_slope`
- `er_fin`
- `er_max`
- `core_fin`
- whether node count stayed stable
- whether the late state stayed localized
- whether it regrew / split / merged / spun down / blew up

## 6. Interpretation Guardrail

Until the shortlist completes, keep the provisional hunt verdict at:

`K_DEPENDENCE_UNCLEAR`

Why:

- low-K TRUEs exist at `N=48/T=4000`
- high-K feb-like TRUEs exist
- IC mass/energy scales with `K`
- only the higher-fidelity row-targeted replays can tell us which low-K cases survive the stricter test

## 7. Next Action

Status update:

- controls completed:
  - `ref_feb56dc7` -> TRUE, `n_fin=4`
  - `623` -> TRUE, `n_fin=4`
- shortlist completed:
  - `663` -> TRUE, `n_fin=2`
  - `743` -> TRUE, `n_fin=1`
  - `154` -> TRUE, `n_fin=4` after `n_mid=2`
  - `570` -> TRUE, `n_fin=3`
  - `744` -> NEAR, `n_fin=1`

Interpretation remains:

`K_DEPENDENCE_UNCLEAR`

Next action:

Update:

- `docs/PHASE_C_METHOD_PARITY_AUDIT.md`
- `docs/PHASE_C_K_VARIED_SATURATION_HUNT.md`

with side-by-side `N=48` versus `N=96/T=6000` outcomes and the revised shortlist recommendation.
