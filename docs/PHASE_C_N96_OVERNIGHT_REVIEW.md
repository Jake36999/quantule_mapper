# Phase C N96 — Overnight longer-T / seed / capacity review

**Date:** 2026-06-25 (batch ran 01:43–08:24, 6.68 h, 25/25 runs)
**Batch:** `sweep_runs/PHASE_C_N96_OVERNIGHT_20260625_014313/` (auto-summary `OVERNIGHT_SUMMARY.md`,
`overnight_results.csv`, per-run `summary.json`/`probe_data.npz`, `closure_dynamics/`).
**Scope:** analysis/replay only — exact shortlisted rows, longer T / extra seeds / mass overrides;
no PDE/solver/classifier/geometry/search change. Geometry frozen at `e8d6a78ea`.

> **Headline (overturns the Stage-1 reading): the N96/T6000 "TRUE_SATURATED_BOUND_STATE" verdicts for
> the K6 multi-node Option-B cases are short-window artifacts.** At longer integration these
> configurations are **long-time unstable** — they either decay to zero or run away to blowup. The T=6000
> classifier window is too short to separate a genuine long-time attractor from a slow transient passing
> through the in-band energy range. Two hypotheses I floated are now **falsified**: the K6-high
> "settling-overshoot" idea and the "stable mass-holding capacity" idea.

---

## P1 — K6 high-mass longer-T: it DECAYS TO ZERO (genuine failure, not a settling overshoot)

| T | class | er_fin | held_mass_raw | held/input |
|---|---|---|---|---|
| 6000 (Stage 1) | SPIN_DOWN | 0.438 | 7180 | 0.438 |
| 9000 | SPIN_DOWN | 0.247 | 4050 | 0.247 |
| 12000 | SPIN_DOWN | 0.145 | 2372 | 0.145 |
| 18000 | SPIN_DOWN | 0.016 | 269 | 0.016 |
| 24000 | SPIN_DOWN | 0.000 | 4.1 | 0.000 |
| 36000 / 48000 | SPIN_DOWN | 0.000 | 0.0 | 0.000 |

Monotonic decay to **zero mass**. My earlier "the T=6000 mass (7180) is a plateau / it's settling toward
the survivor's capacity" reading was **wrong** — 7180 was simply mid-decay. `K6_HIGH_MASS_LONG_T_DECAYS_TO_ZERO`
(a real, complete failure, confirmed and seed-robust — see P4).

## P2 — K6 near-threshold: runs away to BLOWUP

| T | class | er_fin | held_mass_raw |
|---|---|---|---|
| 12000 | TRANSIENT_GROWER | 2.60 | 20764 |
| 24000 | LATE_BLOWUP | 4.66 | 37253 |
| 36000 | LATE_BLOWUP | 7.76 | 62091 |

The "still rising" T=6000 TRUE was the early part of a runaway. `K6_NEAR_THRESHOLD_LONG_T_BLOWUP` —
resolved: a slow grower, not a bound state.

## P3 — K6 mid-mass (the Stage-1 "robust survivor"): BLOWS UP by T=24000

| T | class | er_fin | held_mass_raw |
|---|---|---|---|
| 6000 (Stage 1) | TRUE_SATURATED | 0.85 | 6794 |
| 12000 | TRUE_SATURATED | 1.71 | 13662 |
| 24000 | **LATE_BLOWUP** | 29.99 | 239958 |

This is the most important correction: the case I called `K6_MID_MASS_N96_SUPPORTED` is **long-time
unstable**. At T=6000 it sat in-band (the "dip then recover" we saw in the trace was the start of a slow
growth phase, not stabilization); by T=12000 its mass had doubled; by T=24000 it ran away.
`K6_MID_MASS_TRUE_IS_T6000_WINDOW_ARTIFACT`.

## P4 — Seed expansion: the T=6000 verdicts are seed-robust (but the window itself is the problem)

| case | extra seeds → class | vs Stage-1 |
|---|---|---|
| k6_high | SPIN_DOWN, SPIN_DOWN (er 0.32, 0.30) | consistent (decays) |
| k6_mid  | NEAR, NEAR (er 1.34, 1.64) | original was TRUE → seed-sensitive TRUE/NEAR, all in-band |
| k6_near | TRUE, TRUE (er 1.80, 1.84) | consistent at T=6000 |
| k1_low  | BLOWUP, BLOWUP | consistent (immediate blowup) |

The broad class is seed-robust at T=6000, so the failure modes (decay / blowup / immediate-blowup) are not
seed flukes. But P1–P3 show the **T=6000 classification itself** is unreliable for the growers.

## P5 — Mass-capacity ladder: NO stable capacity — a decay-vs-blowup separatrix (FALSIFIES the hypothesis)

k6_mid IC, vary input mass, T=6000:

| input_mass | class | held_mass_raw | held/input |
|---|---|---|---|
| 4000 | SPIN_DOWN | 12 | 0.003 |
| 6000 | SPIN_DOWN | 31 | 0.005 |
| 8000 (=k6_mid) | TRUE@T6000→BLOWUP@T24000 | 6794→runaway | — |
| 10000 | LATE_BLOWUP | 32141 | 3.21 |
| 12000 | LATE_BLOWUP | 53650 | 4.47 |
| 14000 | LATE_BLOWUP | 71656 | 5.12 |
| 20000 | LATE_BLOWUP | 115618 | 5.78 |

Below ~8000 the field **decays toward zero**; above ~8000–10000 it **blows up**. There is **no stable
held-mass attractor** — only an unstable separatrix near the input mass where the T=6000 window catches a
transient in-band. `PER_NODE_MASS_WINDOW_HYPOTHESIS_FALSIFIED` (as a *stable capacity*; the mass threshold
that separates decay from blowup is real, but it is a separatrix, not an attractor, and is IC-dependent —
note k6_high at 16402 decays while the k6_mid-IC ladder blows up at ≥10000, so the decay/blowup boundary
depends on the specific row params, not mass alone).

---

## What this means (disciplined)

- **`PHASE_C_T6000_SATURATION_WINDOW_TOO_SHORT`.** The saturation classifier's T=6000 window over-reports
  bound states: slow growers and slow decayers both pass through the in-band energy range and get labelled
  TRUE/NEAR. The Stage-1 Option-B "supported branches" reading must be downgraded accordingly.
- At least **K6 mid-mass is refuted** as a long-time bound state; K6 high-mass is a genuine decay; K6
  near-threshold and K1 are growers/blowups. None of the tested K6 multi-node cases is a long-time
  attractor.
- This is consistent with the project's "dissipative solitons, not missing topology" reading
  ([[gl-rotational-core-basin]]): dissipative multi-node structures need a fine energy balance the bare
  S-NCGL (`γ_A=0`) apparently does not hold for these configs over long times.

## Gaps and caveats (do NOT overclaim the negative)

- **K4, K2, and feb56dc7 were NOT longer-T tested in this batch.** They were the other Stage-1
  "supported"/control cases. Given K6-mid (also "supported") failed at long T, K4 and K2 are now
  **suspect but untested**. feb56dc7 is the reference bound state and the **critical control**.
- **Numerical caveat:** T=24000–48000 is 4.8–9.6M FP64 steps; the equivalence proof was at 40 steps.
  Very-long-T numerical accumulation is unquantified. The decays/runaways are monotonic and physical-looking
  (gain/loss imbalance), but the feb long-T control is what tells us whether long-T is *numerically
  trustworthy*: if feb stays a stable bound state to T=24000, the K6 instabilities are real; if feb also
  blows up, suspect numerics.
- No charge / topological / proof / ground-state / black-hole / universal-law claim is made.

## Recommended next step (the decisive control — confirm before broad conclusions)

Run at **T=24000** (one seed each, ~24 min/run, ~72 min total), exact rows:

```bash
# feb56dc7 — the critical long-T control (is ANY of these a real bound state? is long-T trustworthy?)
python jax_scout/core_saturation_replay.py --ref feb56dc7 --N 96 --T 24000 --trace-snaps 40 \
  --out sweep_runs/PHASE_C_N96_LONGT_CONTROL_<ts>/feb56dc7_T24000
# K4 — Stage-1 "supported", now suspect
python jax_scout/core_saturation_replay.py --csv sweep_runs/CORE_SAT_HUNT_20260624_124444/all_evals.csv --idx 25 \
  --N 96 --T 24000 --ic-seed-override 20260620 --target-initial-mass-override 9600.0 --trace-snaps 40 \
  --out sweep_runs/PHASE_C_N96_LONGT_CONTROL_<ts>/k4_T24000
# K2 — Stage-1 "supported" (compact), now suspect
python jax_scout/core_saturation_replay.py --csv sweep_runs/CORE_SAT_HUNT_20260624_152029/all_evals.csv --idx 10 \
  --N 96 --T 24000 --ic-seed-override 20260621 --target-initial-mass-override 16402.349616 --trace-snaps 40 \
  --out sweep_runs/PHASE_C_N96_LONGT_CONTROL_<ts>/k2_T24000
```

Outcome logic:
- **feb stable to T=24000 + K4/K2 blow up/decay** → the K6/K4/K2 Option-B branches are genuinely not
  long-time bound states; feb remains the only validated bound state; long-T is trustworthy. Phase C
  bound-state claims should be restricted to feb-like configs and the classifier window revisited.
- **feb also unstable at T=24000** → suspect long-T numerics (or the bare S-NCGL has no long-time bound
  states at all at N96); investigate before any further bound-state claims.
- **K4 and/or K2 stable to T=24000** → those specific branches survive; only the K6 multi-node family is
  long-time unstable.

Not launched — for review first.

---

# Longer-T control result — 2026-06-25 (CONCLUSIVE)

Ran the control batch (`sweep_runs/PHASE_C_N96_LONGT_CONTROL_20260625_083731/`, T=24000, trace; K2's
trace capture was cut short when the host process stopped mid-run, but its verdict run completed —
`summary.json` + `probe_data.npz` written, so all three verdicts are in hand).

| case | T=24000 class | er_fin | er_max | held_mass (t0 → end) | verdict |
|---|---|---|---|---|---|
| **feb56dc7** | **TRUE_SATURATED** | 1.20 | 1.60 | 16534 → 19787 (bounded, breathes ~16k–26k) | **STABLE bound state** |
| K4 | SPIN_DOWN | 0.125 | 0.99 | 9600 → 7917 → 5309 → **1197** | decays toward zero |
| K2 | SPIN_DOWN | 0.48 | 0.97 | 16402 → **7900** | decays |

feb stays in-band with **steady 4 nodes and stable ρ_peak (~1.0)** across the whole T=24000 (mass
fluctuates ±30% around ~20k — a breathing bound state — late-window slope is negligible vs the mass).
K4 and K2 shed mass monotonically (K4's node_count 4→3, ρ_peak falling) — the same decay mode as K6-high.

## Conclusion (resolves the Phase C Option-B arc)

1. **Long-T integration is numerically trustworthy.** The reference bound state feb56dc7 remains a stable,
   bounded, in-band 4-node state to T=24000 (5M FP64 steps). So the decays/blowups of the K-cases are
   **real dynamics, not numerical accumulation.** The numerical caveat is retired.
2. **None of the discovered Option-B branches is a genuine long-time bound state.** Every one tested —
   K6 high/mid/near, K4, K2 — decays to ~zero or blows up by T=24000. Only **feb56dc7** (the pre-existing
   reference, not a search discovery) survives.
3. **`PHASE_C_T6000_SATURATION_WINDOW_TOO_SHORT` is confirmed and decisive.** The saturation classifier's
   T=6000 window systematically over-reported bound states: slow growers and slow decayers sit in the
   in-band energy range at T=6000 and are labelled TRUE/NEAR. The Option-B "branch landscape" is an
   artifact of that window, **not** a set of physical bound states. The Stage-1/visual "supported branches"
   conclusions are **withdrawn**.

## Implications / recommended direction

- Treat **feb56dc7 as the one validated long-time bound state** at N96; do not cite the K6/K4/K2 Option-B
  branches as bound states.
- Before any further saturation-search claims, **lengthen the classifier's validation window** (e.g.
  require stability to T≈24000, or add a late-window growth/decay gate) so transients are not scored as
  saturated. The discovery search at T=4000 is fine for *shortlisting*, but promotion to "bound state"
  needs a long-T confirmation pass.
- Open question worth one careful look: *why is feb special?* (it uses per-blob-fixed IC norm and is a
  4-node config). A small, pre-registered comparison of feb-like vs the failed configs — at long T — would
  be the principled next experiment, not another broad search.

No charge / topological / proof / ground-state / black-hole / universal-law claim is made.
