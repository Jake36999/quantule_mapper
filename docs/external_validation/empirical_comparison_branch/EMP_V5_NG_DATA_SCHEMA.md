# EMP-V5-NG Digitized Dataset Schema (collision phase×outcome)

Contract between the digitization (Codex) and the V5 comparator
(`tools/external_validation/empirical_comparison/run_empirical_comparison.py --metric v5`). The V5 comparison is
**qualitative**: does the experiment show phase-dependent collision outcomes in the *same direction* as our C3
phase×speed grid (transmission concentrated at anti-phase; in-phase and off-phase stick)? It is **not** a numeric
match and **not** a speed-boundary match.

## CSV columns

| column | required | values / units | meaning |
|---|---|---|---|
| `source_figure` | yes | string | figure/table the point came from |
| `relative_phase_rad` | yes | radians | Δφ between the colliding solitons (0 = in-phase, π = anti-phase) |
| `phase_class` | optional | `in_phase` / `anti_phase` / `off_phase` | explicit class; if blank it is derived from `relative_phase_rad` |
| `speed_value` | optional | float | relative collision speed if reported |
| `speed_units` | optional | string | units/definition of `speed_value` |
| `observed_outcome` | yes | see set below | what the experiment reports for this point |
| `include_in_comparison` | optional | `true`/`false` (default `true`) | set `false` to exclude a point (e.g. off-regime), with a reason |
| `exclusion_reason` | optional | string | required if `include_in_comparison=false` |
| `extraction_uncertainty` | yes | string/float | per-point digitization uncertainty |
| `digitizer_notes` | optional | string | anything the reviewer should know |

## `observed_outcome` allowed values

```
PASS_THROUGH      solitons separate and survive the collision
TRAJECTORY_JUMP   survive with a position/phase shift (still separated)
NO_COLLAPSE       collision without destruction (survived)
BOUNCE            reflect / repel without merging
CAPTURE           bind into a bound state
COLLAPSE          implode / destroyed / atom loss at the collision
MERGE             coalesce into one
INCONCLUSIVE      reported but ambiguous  -> excluded from the direction test
UNKNOWN           not determinable        -> excluded from the direction test
```

## How the comparator reduces outcomes

- **TRANSMITTED (survived as two):** `PASS_THROUGH`, `TRAJECTORY_JUMP`, `NO_COLLAPSE`, `BOUNCE`
- **NOT-TRANSMITTED (bound/merged/destroyed):** `CAPTURE`, `COLLAPSE`, `MERGE`
- **AMBIGUOUS (dropped):** `INCONCLUSIVE`, `UNKNOWN`

Caveat recorded automatically: `COLLAPSE` (destruction) is grouped with not-transmitted, but its *mechanism*
(density-spike implosion in the attractive BEC) differs from our C3 `CAPTURE` (binding). The comparison is a
**qualitative direction** test only — `CLOSE ANALOGUE` ceiling — and this mechanism difference is a stated limit.

## Phase-class derivation (when `phase_class` blank)

`in_phase` if `|Δφ| ≤ 0.4 rad`; `anti_phase` if `|Δφ − π| ≤ 0.4 rad` (mod 2π); else `off_phase`.

## The question the comparator answers

> Is transmission/survival **concentrated at anti-phase** (higher survived-fraction at anti-phase than at
> in-phase), matching the direction of the C3 grid — yes/no, with the per-phase-class fractions shown?

Not: do the speeds, boundaries, or rates match. One CSV per figure; raw off-git on E: (`datasets/README.md`).
