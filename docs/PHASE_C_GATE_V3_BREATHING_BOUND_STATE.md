# Phase C — Gate v3: breathing-aware bound-state classifier

**Date:** 2026-06-25
**Change:** `jax_scout/core_saturation_search.py` classifier v2 → **v3** (classifier/doc/tests only;
no PDE/solver/geometry/search/parameter change). Built on the v2 long-time drift gate
([PHASE_C_STABILITY_GATE_CALIBRATION.md](PHASE_C_STABILITY_GATE_CALIBRATION.md)).

## Why v3 is needed (the v2 false rejection)

The T=24000 confirmation showed v2 rejects a genuine bounded breathing bound state. The counterexample is
**K6_s20260619 = feb's exact IC** at T=24000:
- gained energy/mass (held/init 1.20), `er0 = er_min = 0.99` (**never** fell below its start),
  `er_max = 1.596`, `er_fin = 1.197` (bounded, in-band) — the known stable breathing bound state;
- v2 rejected it `SPIN_DOWN` only because the late-half drift over `[12000, 24000]` (−0.177) crossed the
  0.15 threshold during a **breathing downswing** from its er_max peak.

So v2 is a valid *T=12000* gate but conflates a bounded oscillation's downswing with real decay at longer
windows.

## The v3 rule (smallest safe change)

In `classify()`, when `|late_drift| > LATE_DRIFT_MAX (0.15)`, **accept instead of reject iff the state is a
bounded breather:**

```
er0 = er[0];  er_min = min(er);  floor_ratio = er_min / er0
bounded_breathing =
    er_max <= 3.0                                  # bounded above (anti-runaway)
    and 0.5 <= er_fin <= 2.5                       # final energy in band
    and floor_ratio >= 0.85                        # never fell to a floor  (anti-decay)
    and er_fin <= 0.95 * er_max                    # came back down from a peak (anti-monotonic-grower)

if abs(late_drift) > 0.15:
    if bounded_breathing:  -> TRUE_SATURATED_BOUND_STATE   # drift is oscillation, not a trend
    else:                  -> TRANSIENT_GROWER_REJECT (drift>0) / SPIN_DOWN_REJECT (drift<0)
```

Two conditions do the work:
- **`floor_ratio >= 0.85`** rejects floorward decay (a decayer's energy drops well below its start;
  k6_high → floor 0.00, k4 → 0.13).
- **`er_fin <= 0.95*er_max`** rejects a monotonic in-band grower that ends at its own peak
  (k6_mid → er_fin = er_max, fails this) — i.e. it requires the state to have *returned* from a peak.

**Deviation from the suggested design (intentional):** a `return_ratio = er_fin/er0 >= 0.8` rule (as
proposed externally) would *false-accept* a monotonic in-band grower like k6_mid (return_ratio 1.7, but it
blows up). The peak-margin condition (`er_fin < er_max`) is what safely excludes growers, so v3 uses that
instead of return_ratio. v3 also needs **no explicit T≥24000 gate**: the conditions are window-agnostic and
were verified not to change any T=12000 verdict (below), so the gate stays simplest.

New recorded metrics: `er0`, `er_min`, `floor_ratio`, `bounded_breathing`. `classifier_spec()` bumped to
`PHASE_C_SATURATION_CLASSIFIER_v3` with the breathing constants.

## Real-trace reclassification (verification)

| case | v3 class | late_drift | floor_ratio | er_fin | breathing | expected |
|---|---|---|---|---|---|---|
| feb K6_s619 @ T=24000 | **TRUE_SATURATED** | −0.177 | 1.00 | 1.197 | True | TRUE (breathing) ✅ |
| K3_s619 @ T=24000 | **TRUE_SATURATED** | −0.200 | 1.00 | 1.099 | True | TRUE (breathing) ✅ |
| k6_high @ T=24000 | SPIN_DOWN_REJECT | −1.290 | 0.00 | 0.000 | False | reject (decay→0) ✅ |
| k4 @ T=24000 | SPIN_DOWN_REJECT | −0.792 | 0.13 | 0.125 | False | reject (decay) ✅ |
| k6_mid @ T=12000 | TRANSIENT_GROWER_REJECT | +1.138 | 0.74 | 1.708 | False | reject (grower) ✅ |
| k6_near @ T=12000 | TRANSIENT_GROWER_REJECT | +0.384 | 1.00 | 2.596 | False | reject (grower) ✅ |
| k6_high @ T=12000 | SPIN_DOWN_REJECT | −0.702 | 0.15 | 0.145 | False | reject (decay) ✅ |

(k6_mid/k6_near at T=24000 are already caught by the existing `er_max > 3` LATE_BLOWUP rule.)

## Tests

`tests/test_core_saturation_search.py` — 13/13 pass (WSL unittest). Covers: flat→TRUE; slow grower &
decayer with tiny slope still gated out (v2); **bounded breathing downswing → TRUE**; floorward decay →
SPIN_DOWN (not breathing); monotonic grower ending at peak → GROWER (not breathing); breathing metrics
recorded; spec version v3.

## What v3 does and does not claim

- **Does:** correctly accept a bounded breathing bound state (feb at long T) while still rejecting genuine
  slow decay, floorward decay, and monotonic/runaway growth. Window-agnostic; preserves all T=12000
  verdicts.
- **Does NOT:** assert anything physical beyond "bounded oscillation in the accepted energy band." No
  charge / topology / proof / ground-state / black-hole / universal-law claim. Thresholds (0.85, 0.95) are
  calibrated on a small real-trace set and remain **provisional** — re-check as more long-T bound states
  and breathers are observed. v3 is conservative on the *growth* side (an upswing-phase breather caught at
  its peak is not rescued — re-run longer to resolve).

## Recommended use

- **T=12000 drift-gated validation remains the main practical gate** (unchanged by v3).
- **T≥24000 now uses the breathing-aware interpretation** — long-window bound-state confirmation is safe.
- Discovery stays T=4000 for shortlisting; promotion to "bound state" still requires a long-T drift-gated
  (now breathing-aware) pass.
