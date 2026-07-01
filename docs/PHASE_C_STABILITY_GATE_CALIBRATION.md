# Phase C — Long-time stability gate calibration

**Date:** 2026-06-25
**Purpose:** turn the negative N96 result into a calibrated instrument. Using the trajectories we already
collected (zero new compute), derive a *late-half mass-drift* gate that would correctly separate the one
validated bound state (feb56dc7) from the transients the T=6000 classifier over-reported, and find the
minimum validation window.
**Data:** five T=24000 trace bundles with dense mass(t) — feb56dc7 + k4 (control batch), k6_high/k6_mid/
k6_near (overnight batch). Figure: `sweep_runs/PHASE_C_STABILITY_GATE_CALIB_20260625_121731/stability_gate_calibration.png`.
**Read-only / analysis only.**

---

## The metric

For a run truncated at validation window `Tv`, fit a line to `mass(t)` over the **late half**
`[Tv/2, Tv]` and take the fractional change of the fit across that window:

```
drift(Tv) = ( m_fit(Tv) − m_fit(Tv/2) ) / |m_fit(Tv/2)|
```

The linear fit makes it robust to feb's breathing (its mass oscillates ±30% around ~20k); raw endpoints
would be fooled by the oscillation phase. A genuine bound state has `drift ≈ 0`; a slow decayer is
negative; a slow grower is positive; a blowup → ∞.

## Calibration result (|drift| by validation window)

| config | fate | Tv=6000 | 8000 | 10000 | 12000 | 16000 | 24000 |
|---|---|---|---|---|---|---|---|
| **feb56dc7** | **stable** | **0.003** | 0.036 | 0.060 | **0.080** | 0.114 | 0.177 |
| k4 | decay | 0.106 | 0.179 | 0.255 | 0.330 | 0.486 | 0.792 |
| k6_high | decay | 0.387 | 0.534 | 0.638 | 0.702 | 0.876 | 1.273 |
| k6_near | blowup | 0.245 | 0.276 | 0.324 | 0.384 | 0.521 | 0.816 |
| **k6_mid** | blowup | **0.103** | 0.163 | 0.464 | **1.139** | 6.700 | ∞ |

**feb has the smallest |drift| at every window**, and the failures all rise with `Tv`. The binding case
is **k6_mid** — the slow grower the T=6000 er-slope classifier missed: its drift is only 0.10 at T=6000
but **1.14 by T=12000** (mass more than doubling), so a modest window extension makes it unmistakable.

## Why the original T=6000 classifier failed (and the cheap fix)

The classifier gated on the **er late-slope**, which for k6_mid was ~2.6e-5 at T=6000 → scored TRUE. But
k6_mid's **mass** was already drifting +10% over the same late half. The normalized mass-drift is the more
sensitive early-warning: even at T=6000 it separates feb (0.003) from every failure (≥0.10). So part of
the fix is just a **better metric**, independent of window length.

## Recommended gate

Promote a config to bound-state **only if all hold** at the validation window:

```
1. validation window  Tv = 12000      (2x the discovery T6000; flags even k6_mid by a wide margin)
2. |late-half mass drift(Tv)| <= 0.15  (feb 0.080  <  0.15  <  0.330 = nearest failure -> margin both sides)
3. er in band  (0.5 <= er_fin <= 2.5)   AND  bounded  (er_max <= 3)        [existing checks]
4. node_count stable over the late half  AND  rho_peak not monotonically falling  [existing diagnostics]
```

- **T=12000, not T=24000.** The control used T=24000, but the calibration shows T=12000 already separates
  the slowest failure (k6_mid drift 1.14) from feb (0.08) with a wide margin — so the validation pass is
  ~2x discovery cost, not 4x. A cheaper **two-tier** option also works: gate at T=6000 with the tighter
  `|drift| <= 0.05` (feb 0.003 vs fails ≥0.10) as a fast screen, then confirm survivors at T=12000.
- **Use the fit-based drift, not raw endpoints** (feb's breathing would otherwise trip a raw gate).

## Caveats (do not over-fit)

- Calibrated on **one** stable exemplar (feb) + four failures. The `0.15` threshold is bounded by feb's
  breathing amplitude; it must be re-checked against more stable exemplars as they are found. The
  *ordering* (feb lowest at every Tv) is robust; the absolute threshold is provisional.
- The drift gate is necessary, not proven sufficient — a config could be flat over `[Tv/2,Tv]` yet fail
  later (none did here, but the sample is small). Keep the bounded-er and node/ρ_peak checks alongside it.
- Mass-drift is norm-agnostic (relative), which is important because feb is per-blob-fixed-norm while the
  search cases are total-mass-fixed — so the gate transfers across IC-norm families.

## How to apply going forward

1. **Bake the gate into the saturation classifier spec** (discovery still at T=4000 for *shortlisting*;
   no bound-state label without the T=12000 drift-gated confirmation). Re-stamp `classifier_spec()`.
2. **Re-screen prior "bound state" claims** (incl. the GL-basin "SUSTAIN" classifier) for the same
   short-window artifact.
3. **Refocus search on feb's basin** rather than broad hunting: vary node-count / IC-norm / a tight param
   neighborhood around feb, each T=12000 drift-gated — map what else is genuinely stable. The feb-vs-failed
   IC-norm difference (per-blob-fixed vs total-mass-fixed) is the first lead.

## Update (2026-06-25) — the gate over-rejects breathing at T=24000 (confirmed limitation)

The T=24000 confirmation runs exposed the provisional-threshold caveat in practice: feb's own IC
(K6, seed 20260619) at T=24000 is `SPIN_DOWN_REJECT`-ed by the gate (drift −0.177 > 0.15) even though it
is the known stable bound state — it **gained mass** (held/init 1.20) and its energy **never dropped below
the start** (`er_min = er[0]`, er_max 1.596, er_fin 1.197). The gate's late-half drift catches the
**breathing downswing** from feb's er_max peak over the wide `[12000,24000]` window.

**Conclusion:** the 0.15 threshold is **Tv=12000-specific** and must not be extended to longer windows
as-is. A breathing-robust refinement is needed before T≥24000 use — e.g. only flag SPIN_DOWN when the
energy is *also* falling toward a floor (`er_fin ≈ er_min` with `er_min` well below `er[0]`), so a bounded
oscillation (`er_min ≈ er[0]`, er_fin still a large fraction of er_max) is **not** rejected. Until then,
apply the gate at the calibrated **T=12000** window only.

No charge / topological / proof / ground-state / black-hole / universal-law claim is made.
