# Digitized Dataset Schema

The comparator (`tools/external_validation/empirical_comparison/run_empirical_comparison.py`) and the digitization
task are contract-matched by this schema. Extract each figure into a CSV with exactly these columns so the
comparator can consume it without reinterpretation.

## V1 — force law (`--metric v1`)

CSV columns (one row per digitized point):

| column | units | meaning |
|---|---|---|
| `relative_phase` | radians | Δφ between the two solitons (0 = in-phase, π = anti-phase) |
| `q_half_separation` | dimensionless (soliton widths) | half the pair separation; normalize to the soliton width reported in the paper |
| `interaction_measure` | signed, arbitrary monotone force proxy | **sign encodes direction** (>0 repulsive / separation grows, <0 attractive / separation shrinks); **magnitude scales with force** (initial acceleration, inverse-squared interaction/oscillation period, etc.) |

Notes:
- The comparator checks the **sign law** (attract when cos Δφ > 0, repel when cos Δφ < 0) and fits the **decay
  rate** `|measure / cos Δφ| = C·exp(−λ q)` to recover an experimental `λ` for comparison to our internal C2
  `λ ≈ 1.98` (canonical `2A`). It does not need the proxy to be in physical units — only sign-correct and
  monotone in force.
- Record in the provenance file exactly what `interaction_measure` was extracted from and its units before
  normalization.
- Minimum 3 rows; at least 2 distinct `q_half_separation` values for a decay-rate fit.

## V5 — collision phase diagram (scaffolded; comparator support pending)

Planned columns: `relative_phase` (rad), `impact_velocity` (fraction of characteristic speed), `outcome`
(`capture` / `transmit` / `bounce`). Comparison will be a categorical boundary overlay vs our C3 phase×speed grid.
Do not extract until the V5 comparator path is implemented.

## General

- One CSV per figure; name it `<EMP-id>_<figure>.csv` (e.g. `EMP-V1-MM_fig3.csv`).
- Raw figure images / papers stay off-git on E: (`datasets/README.md`); only the small derived CSV + provenance
  record are committed.
- Every dataset needs a completed `provenance_<EMP-id>.md` (copy of the template) or the comparator refuses it.
