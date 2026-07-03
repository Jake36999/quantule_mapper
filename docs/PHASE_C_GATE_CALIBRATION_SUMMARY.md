# Phase C — Gate Calibration Summary (H5)

One-page consolidation of the **promotion gate** — the only criterion the closed Phase C claims passed through.
Consolidates `PHASE_C_STABILITY_GATE_CALIBRATION.md` + `PHASE_C_GATE_V3_BREATHING_BOUND_STATE.md` and adds the
a\*-arc late-slope criterion + the honest calibration caveat. Gate = `core_saturation_search.classify`.

## Version history
| gate | rule | why it changed |
|---|---|---|
| slope-only | `|late er slope|` per step below a threshold | missed slow growers/decayers that sit *in* the energy band |
| **v2** | + normalized late-half **drift** (`late_drift`, linear fit over [T/2, T]); `LATE_DRIFT_MAX=0.15` | catches in-band drift; but **over-rejects the downswing of a bounded breather** at long T |
| **v3** | + **breathing exception**: accept `|drift|>0.15` iff `er_max≤3`, `floor_ratio=er_min/er0 ≥ 0.85`, `er_fin ≤ 0.95·er_max` | rescues feb's breathing at T24000; still rejects decay (floor fails) + growers (peak-margin fails); window-agnostic, preserves all T12000 verdicts |
| **a\*-arc refinement** | **late-window slope → 0** as the long-time stationarity criterion (`late_slope_10/50pct_per1k`) | v3 "TRUE" can still be a slow decayer caught before it crosses `er0`; slope→0 is the true fixed-point test |

## The window-artifact lesson (the core calibration fact)
Short validation windows **over-report** bound states — the failure mode that drove every recalibration:
- **T=6000** "TRUE_SATURATED" for K6 configs → transients (`PHASE_C_N96_OVERNIGHT`);
- **T=12000 / T=24000** — v2 over-rejected feb's breathing (fixed by v3);
- even **T=24000** "TRUE" for feb-center → it **slowly decays by T=72000** (`FEB_BREATHING_LONGT`);
- resolved at **T=72000 / T=144000** by the **slope→0** criterion.

**Standing rule:** a bound-state claim requires a **long-window slope→0** (or v3 breathing-bounded) verdict.
Short-window "saturation" is *not* stability.

## a\* confirmation (what the calibrated gate certified)
Under v3 + slope→0: `a* ≈ ×1.15` (param_a ≈ 0.552) is a genuine **T=144000-stationary** bound state
(slope −0.0004), bracketed ×1.15–1.16 (below decays, above grows), stable across seeds 619/620/621, reproduced at
N=128. This is the load-bearing supported claim.

## Single-exemplar caveat (honest limitation)
The v3 thresholds (`LATE_DRIFT_MAX=0.15`, `floor_ratio_min=0.85`, `peak_margin=0.05`) were calibrated on **one**
stable trace (feb) plus its nearest failures. They are *provisional*: discriminating and reproducible on the
tested set, but a broader multi-exemplar calibration has **not** been done. (→ Stage 2 backlog item.)

## What the gate CAN and CANNOT certify
**CAN:** long-time energy-**stability** of a localized state (bounded / stationary / bounded-breathing vs
decay / growth / blow-up / fragmentation), given a sufficiently long window; the a\* gain/loss balance.
**CANNOT:** matter-likeness, mobility, topology, prime-harmonic structure, or transport — the gate is an
**energy-stability** classifier only. Prime-SSE and TDA are **not** part of it and are non-discriminating
(`BASELINE_AUDIT_VALIDATION.md`). It certifies a *dissipative attractor*, nothing about a matter/transport sector.
