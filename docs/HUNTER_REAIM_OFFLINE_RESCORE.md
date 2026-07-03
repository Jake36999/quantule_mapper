# Hunter Re-Aim — Offline Re-Score (H7.1 + H7.2, no runs, no hunt)

**Status:** `HUNTER_GAIN_LOSS_REAIM_DESIGNED` + offline scorer implemented and validated on existing data.
Still `HUNTER_REAIM_NOT_IMPLEMENTED` at the hunter level — **not wired into `aste_hunter`, no GPU hunt, no
solver/gate/physics change.** Governing rule holds: Hunter proposes, `css.classify` v3 certifies.

## H7.1 — scorer module
`tools/stability_objective.py` — standalone, pure-Python (no jax/cupy). Scores a run's recorded metrics toward
the validated target (gain/loss-balanced standing attractor): **late-slope→0 (primary), long-window
boundedness, v3 bounded-breathing, energy-band safety, and a long-window-length gate**; disqualifiers mirror the
gate (grower/blow-up, spin-down). **`log_prime_sse` is retired** (`PRIME_SSE_RETIRED = True`) — exploratory
diagnostic only, never steering. **Indivisibility** is an explicit run-based hook (`None` here) — a
perturbation-response measurement (design spec §3), not summary-derivable.

## H7.2 — offline re-score over the load-bearing evidence
The scorer was applied to the real `.csv` summaries (dev box, no runs). It must rank the validated a\* case above
known decayers/growers and discount short windows — and it does:

**`FEB_ASTAR_CONFIRM` (ranked):**
| score | cell | a×feb / seed / T |
|--:|---|---|
| **+0.898** | `a1.15_longT` | ×1.15 / 619 / **T=144000** |
| +0.841 | `a1.15_seed621` | ×1.15 / 621 / 72000 |
| +0.815 | `a1.16` | ×1.16 / 619 / 72000 |
| +0.766 | `a1.15_seed620` | ×1.15 / 620 / 72000 |
| +0.634 | `a1.125_longT` | ×1.125 / 619 / 144000 |
| +0.632 | `a1.175` | ×1.175 / 619 / 72000 |
| +0.284 | `a1.20` | ×1.20 (grower) / 619 / 72000 |

**`FEB_GAIN_LADDER` (ranked):** `a1.15` +0.868 > `a1.125` +0.626 > `a1.10` +0.518 > `a1.075` +0.466.

**Result:** the confirmed **a\* ≈ ×1.15 tops both sets**; the slow-decayers (×1.075–1.125) and the grower (×1.20,
lowest) rank below; the T=144000 a\* outranks the T=72000 seeds. Feb-center (×1.0) and ×1.05 — which decay at
T=72000 — are disqualified (`SPIN_DOWN`) by the scorer, consistent with the gate.

**Window-artifact check:** the long-window gate (`T ≥ 24000`) discounts short-window results — a T=6000 copy of
the a\* metrics scores *below* the same metrics at long T and is marked `certifiable=False`, so short-window
"saturation" cannot be promoted. This is the codified window-artifact lesson, now enforced in the objective.

## Tests
`tests/test_stability_objective.py` — 6 pass (dev box): stationary > decayer > grower; blow-up + spin-down
rejected; long-window gate; indivisibility remains a pending hook; seed-robustness (weakest seed governs, any
reject disqualifies).

## What this establishes (and what it does NOT)
- **Establishes:** the re-aimed objective **discriminates the validated stability result on real data** — it
  independently re-ranks the a\* basin to the top and deprioritizes the failures, *without* prime-SSE. This is the
  offline evidence that the re-aim is sound before touching the hunter.
- **Does NOT yet:** measure indivisibility (needs perturbation runs), wire into `aste_hunter`, run any hunt, or
  claim re-discovery via search. Those are H7.1-wiring / H7.3 / H7.4, held.

## Held (gated)
- **H7.1 hunter wiring** — add the objective behind an explicit flag in `aste_hunter`; keep prime-SSE available,
  not default. (A hunter code change — deferred to explicit go.)
- **Indivisibility scoring** — implement the division-perturbation battery (GPU/WSL runs).
- **H7.3 re-validation harness** — the full acid test (re-find a\* basin *via search*, matched-control recovery).
- **H7.4 small controlled hunt** — gated on H7.3 + the H4 CuPy parity + explicit approval.
