# Phase C N96 — Current-Closure / Signed-Vorticity Static Analysis

**Date:** 2026-06-25
**Scope:** static (single-field) analysis only — no new hunt, no PDE/solver/classifier/geometry/search change.
**Inputs:** Stage 1 saved N96 final fields `PHASE_C_OPTION_B_N96_STAGE1_20260624_223147/<case>/probe_data.npz`
(`psi_fin`, 96³ complex128). No GPU used.
**Renderer:** `quantule_viz/renderers/phase_c_current_closure.py` (`python -m quantule_viz phase-c-current-closure`).
**Output root (not committed):** `sweep_runs/PHASE_C_N96_CURRENT_CLOSURE_20260625_001621/`
**Terminology:** *signed vorticity, current closure, signed circulation, polarity balance* — geometric
handedness of the current field. **Not** electric-charge polarity; no charge / topology / proof /
ground-state / black-hole / universal-law claim.

> Headline: `CURRENT_CLOSURE_ANALYSIS_COMPLETE`, but the static single-snapshot metrics are
> `SIGNED_VORTICITY_CLOSURE_INCONCLUSIVE` and `PER_NODE_MASS_WINDOW_HYPOTHESIS_INCONCLUSIVE`. The snapshot
> currents are too small and too *instantaneous* to test the closure hypothesis. The analysis does pin
> down two things cleanly (K1 fails by blowup before any structure forms; feb is a near-zero-current pure
> standing state) and it converts the closure test into a well-posed **time-averaged** measurement that
> now justifies targeted trace replays.

---

## Metrics used (as proposed, computed in voxel-index space about the dominant density node)

- `J = Im(conj(ψ)·∇ψ)` (3-vector, `np.gradient`); ρ = |ψ|²; node center = argmax ρ; core radius `r0` =
  spherically-averaged half-density radius (voxels, clipped [3, N/4]).
- **net signed radial flux** `net_flux = Σ J_r / Σ|J_r|` on a 300-pt Fibonacci sphere at `r0`
  (`>0` outflow/leak, `<0` inflow/feed, `≈0`/sign-flipping = balanced/no coherent flux). Checked at `r0`,
  `1.5 r0`, `2 r0` for radius-robustness.
- **rad/tan ratio** = ⟨|J_r|⟩/⟨|J_t|⟩; **tangential fraction** = ⟨|J_t|⟩/(⟨|J_t|⟩+⟨|J_r|⟩).
- **ring signed-vorticity profile** ω(θ) at `r0` in the strongest coordinate plane; **lobe balance**
  `(P−N)/(P+N)`; **dipole / quadrupole alternation** = normalized m=1 / m=2 Fourier amplitudes of ω(θ).
- **node→current-center offset** = |ω|-weighted circulation centroid vs density node (voxels & box units).
- **axis polarity** = signed J_r at ±r0 along each index axis.
- **current-closure score** = tang_fraction · (1−|lobe balance|) · exp(−offset/r0) ∈ [0,1].

`signed_vorticity` for K1 is **undefined**: both `psi_mid` and `psi_fin` are non-finite (the field blew up
before T=3000), so there is no formed field to analyze.

---

## Results (N96 final fields)

| case | N96 class | \|J\|/ρ_peak @r0 | net_flux (r0/1.5/2) | tang_frac | lobe_bal | dipole | quad | offset(vox) | closure |
|---|---|---|---|---|---|---|---|---|---|
| K6 high-mass | **SPIN_DOWN** | 0.027 | **+0.85/+0.84/+0.71** | 0.435 | +0.04 | 1.16 | 0.76 | 3.05 | 0.268 |
| K6 mid-mass | TRUE | 0.047 | +1.00/+1.00/+1.00 | 0.029 | −0.00 | 0.70 | 0.05 | 0.07 | 0.028 |
| K4 intermediate | TRUE | 0.014 | +1.00/+1.00/+1.00 | 0.147 | −0.00 | 0.41 | 0.07 | 1.89 | 0.116 |
| feb56dc7 | TRUE | **0.0003** | −0.05/+0.32/−0.28 | 0.549 | −0.07 | 0.05 | 0.21 | 1.59 | 0.441 |
| K2 compact | TRUE | 0.007 | **−0.29/−0.32/−0.25** | 0.612 | +0.06 | 0.11 | 0.09 | 6.75 | 0.272 |
| K1 low-mass | BLOWUP | — | — non-finite (blowup) — | — | — | — | — | — | — |
| K1 high-mass | BLOWUP | — | — non-finite (blowup) — | — | — | — | — | — | — |
| K6 near-threshold | TRUE(rising) | 0.051 | **+1.00/+0.62/−0.95** | 0.046 | +0.01 | 0.06 | 0.17 | 2.18 | 0.037 |

Full data: [phase_c_option_b_n96_stage1... → current_closure_score_table.csv] in the output root.
Figures: `ring_vorticity_profiles.png`, `current_closure_comparison.png`,
`density_current_vorticity_slices.png`, `node_center_offset_summary.png`, `axis_polarity_profiles.png`,
`case_metric_panels/`.

**The dominant fact:** every snapshot current is *small* (|J|/ρ_peak between 3e‑4 and 5e‑2). These are
near-standing fields at the capture instant. For an oscillating saturated soliton the instantaneous
current is a poor proxy for *persistent* circulation/transport, and `net_flux` for feb and K6
near-threshold flips sign across radii — i.e. there is no coherent steady flux to read from one frame.

---

## The four requested comparisons

- **K6 high-mass spin-down vs K6 mid-mass survivor.** Both show net radial *outflow* at the snapshot
  (k6-high +0.84, k6-mid +1.0). The closure/flux metrics do **not** cleanly separate them; the only
  difference is k6-high carries a higher tangential fraction (0.44 vs 0.03) and a much stronger
  dipole/quadrupole vorticity ring (1.16/0.76 vs 0.70/0.05). Suggestive but not decisive from one frame.
- **K2 compact survivor vs K1 compact failure.** K2 is a genuine small, tangential-dominated (0.61),
  mildly *inflowing* circulation (net_flux robustly −0.25…−0.32). K1 has **no field to compare** — it blew
  up before mid-time. So the contrast is "K2 forms a coherent, mildly-closed core; K1 never forms one."
- **K4 survivor vs feb56dc7.** feb is the cleanest *standing* state in the set (|J|/ρ ≈ 3e‑4, essentially
  zero current, incoherent flux) — it just sits. K4 has a small radial-outflow snapshot current and two
  cores. Both are TRUE; their snapshot current signatures differ, again pointing to instantaneous phase.
- **K6 near-threshold rising vs clean TRUE.** k6-near has the lowest ρ_peak (0.53), a low tangential
  fraction, and a radius-*incoherent* net_flux (+1.0 → −0.95) — consistent with an unsettled, still-rising
  state rather than a converged standing one. Matches its positive late energy slope.

---

## Answers to the seven interpretation questions

1. **Did compactness alone fail as a predictor?** **Yes** (already from Stage 1; not rescued here). Static
   morphology — ρ_peak, core radius, node count — does not separate survivors from the spin-down case.
2. **Does current-closure / signed-vorticity balance better separate survivor from failure?** **No, not
   from a single snapshot.** `SIGNED_VORTICITY_CLOSURE_INCONCLUSIVE`. The closure/flux/lobe metrics do not
   rank survivors above K6 high-mass; the currents are too small/instantaneous. The only clean separations
   are trivial (K1 = no field; feb = ~zero-current standing).
3. **Does K2 compact TRUE show coherent closure?** **Weakly yes** — highest tangential fraction (0.61) of
   the multi-node cases and a robust, balanced, slightly-*inflowing* circulation (no outward leak). It
   looks "closed/feeding," not "leaking," consistent with survival — but the absolute current is small.
4. **Does K1 low-mass failure lack closure, or show closure but still fail?** **Neither is measurable —
   K1 blows up before the midpoint** (both `psi_mid` and `psi_fin` non-finite). It never forms a sustained
   core to have closure; the failure mode is rapid delocalized growth, not a formed-then-lost closed
   structure. Seeing its pre-blowup morphology requires an early-frame trace replay.
5. **Does K6 high-mass spin-down show leakage / radial flux / poor source support?** **Suggestively, but
   not decisively.** It shows robust net radial *outflow* across radii (+0.71…+0.85) and the strongest
   dipole/quadrupole vorticity ring — qualitatively a leakier/less-closed signature. But K6 mid-mass (a
   survivor) also shows net outflow, so snapshot radial flux alone does not isolate the spin-down.
6. **Does K6 mid-mass or K4 show stronger closure?** **No** — by the composite score they are *lower*
   (0.028, 0.116) than feb (0.44) and even K6 high-mass (0.27). The composite score, as defined, is
   contaminated by the instantaneous snapshot phase and does **not** behave as a clean survivor metric.
   (Reported honestly rather than presented as a tidy separator.)
7. **Does the per-node mass window hypothesis look more plausible after closure metrics?**
   **Inconclusive** — `PER_NODE_MASS_WINDOW_HYPOTHESIS_INCONCLUSIVE`. Static closure neither confirms nor
   refutes it. The Stage 1 per-node-mass reading (K6-high 6 nodes starved → spin-down; K2 2 nodes in-band
   → survive; K1 1 node → blowup) remains the better-supported observation, but gained no independent
   support from single-frame closure.

---

## What this analysis does and does not establish

**Establishes (static, solid):**
- K1 (both masses) fails by **blowup before any sustained structure** — non-finite by mid-time.
- feb56dc7 is a **near-zero-current standing** bound state (the cleanest saturated reference).
- Snapshot current magnitudes are uniformly small ⇒ a single final frame cannot test "persistent
  current/vorticity closure"; the test must be **time-averaged**.

**Does not establish:** that closure/signed-vorticity separates survivor from spin-down (it does not, from
one frame), nor the per-node-mass-window hypothesis (inconclusive).

---

## Recommended next step — targeted trace replays now justified

The user's gate ("run trace replays only if static results clearly require dynamics to interpret a
specific case") is **met**: the closure test is well-posed only with time-averaged current, and K1's
failure morphology is invisible in the saved fields. Recommend `--trace-snaps 40` N96 replays for a
**focused 4** (≈12 min each, ~48 min, ~1.2 GB), not all 6:

1. **K6 high-mass** (spin-down) and 2. **K6 mid-mass** (survivor) — time-averaged current to test
   leak-vs-closed on the decisive contrast.
3. **K1 low-mass** — early frames to capture pre-blowup morphology (does a core form then delocalize?).
4. **K6 near-threshold** — confirm "still rising" vs settling.

(feb already has a 96³ 12-frame series at `SUBSTRATE_HUNT_20260621_161557/feb56dc7_bound_state/frames.npz`
for the standing-state reference; K4 and K2 static reads are adequate for now.)

Targeted seed expansion (K6 high-mass, K6 near-threshold, K1 low-mass ×2 seeds) remains the separate,
later step. Nothing launched without approval.

---

# Time-resolved (trace) addendum — 2026-06-25

The focused-4 `--trace-snaps 40` N96/T6000 replays ran (batch
`PHASE_C_OPTION_B_N96_TRACE_20260625_003926`, 0.84 h, verdicts reproduced Stage 1). Time-resolved
analysis: `quantule_viz/renderers/phase_c_current_closure_dynamics.py`
(`python -m quantule_viz phase-c-current-closure-dynamics`); output root
`sweep_runs/PHASE_C_N96_CLOSURE_DYNAMICS_20260625_013018/`
(`time_averaged_closure_table.csv`, `closure_dynamics_timeseries.png`, `morphology_over_time.png`,
`spindown_vs_survivor_comparison.png`, `density_slice_timelines.png`).

## Finding 1 — closure is NOT the discriminator (time-averaging confirms the static read)

Time-averaging the current field over the settled half of the trajectory does **not** separate
spin-down from survivor:

| case | per-frame net_flux (mean±std) | persistent ⟨J⟩ net flux | persistent ⟨J⟩ \|J\|/ρ_peak |
|---|---|---|---|
| K6 high-mass (spin-down) | 0.95 ± 0.17 | **+1.00** | 6.3e‑4 |
| K6 mid-mass (survivor)   | 0.96 ± 0.19 | **+1.00** | 6.6e‑4 |
| K6 near-threshold (rising) | 0.73 ± 0.33 | +1.00 | 1.1e‑3 |

Both the decisive cases are near-current-free standing states with the same persistent radial-flux
signature. `SIGNED_VORTICITY_CLOSURE_INCONCLUSIVE` — and, with the proper time-averaged test now done,
the closure hypothesis is effectively **not supported** as a survivor/failure discriminator.

## Finding 2 — the discriminator is the mass/energy trajectory (a holding capacity)

| case | mass t0 → end (×t0) | late mass slope | ρ_peak t0→end | reading |
|---|---|---|---|---|
| K6 high-mass (spin-down) | 16402 → 7180 (**×0.44**) | −1.5 (still shedding) | 1.06→1.16 | overshoot, sheds excess |
| K6 mid-mass (survivor) | 8000 → 6794 (×0.85) | +0.21 (held) | 0.50→1.59 | near capacity, condenses & holds |
| K6 near-threshold (rising) | 8000 → 15007 (**×1.88**) | +0.98 (still rising) | 0.51→0.53 | below, still accumulating |
| K1 low-mass (blowup) | finite only at t=0; **non-finite by t=150** | — | — | immediate delocalized blowup |

The 6-node K6 configuration appears to have a **mass-holding capacity ≈ 6.8–7.2k raw at N96**: k6-mid
(start 8000) dips then holds 6794; k6-high (start 16402) sheds *toward the same level* (~7180, still
declining); k6-near (start 8000) is below and still growing. This **supports the per-node / per-config
mass-window reading** over the morphology and the closure framings. `PER_NODE_MASS_WINDOW_HYPOTHESIS_SUPPORTED`
(3 single-seed points — confirm with seed expansion).

## Finding 3 — K6 high-mass "spin-down" may be a T=6000 window artifact

K6 high-mass mass is **still declining at T=6000** and converging toward the survivor's holding level; its
cores persist (ρ_peak 1.16, compactness rising 47→70). The classifier flags it `SPIN_DOWN` because
`er_fin = energy_fin / energy_IC` falls below the 0.5 floor — but that ratio is taken against a *large*
(16402) initial mass, so an overshoot that sheds excess scores low even while settling into a valid bound
state. So `K6_HIGH_MASS_SPIN_DOWN_CONFIRMED` (as classified, and reproduced) **with a caveat**: it may be a
slow-equilibrating overshoot, not a distinct failure. A longer-T probe would tell whether its `er`
recovers above 0.5 once shedding completes.

## Finding 4 — K1 low-mass: immediate blowup

Only the t=0 frame is finite (1/41); the field is non-finite by t=150. K1 low-mass does **not** form a
transient core that later delocalizes — it blows up almost immediately at N96. `K1_LOW_MASS_RESOLUTION_WEAKENS`
is firmly the blowup variant; no closure structure ever exists to measure.

## Finding 5 — K6 near-threshold: not saturated

Mass nearly doubled (×1.88) and is still rising at T=6000 with low ρ_peak/compactness — an accumulating,
unsettled state, not a converged bound state. `K6_NEAR_THRESHOLD_REMAINS_INCONCLUSIVE`.

## Net labels after dynamics

`CURRENT_CLOSURE_ANALYSIS_COMPLETE` · `SIGNED_VORTICITY_CLOSURE_INCONCLUSIVE` (leans negative) ·
`PER_NODE_MASS_WINDOW_HYPOTHESIS_SUPPORTED` · `K6_HIGH_MASS_SPIN_DOWN_CONFIRMED` (with window-artifact
caveat) · `K6_NEAR_THRESHOLD_REMAINS_INCONCLUSIVE`.

## Recommended next steps

1. **Targeted seed expansion** (×2 seeds each, verdict-first, ~36 min): K6 high-mass, K6 near-threshold,
   K1 low-mass — confirm the mass-trajectory pattern is seed-robust.
2. **Longer-T probe of K6 high-mass** (e.g. T=12000, one seed) to test the window-artifact hypothesis:
   does it stop shedding and recover `er` above the floor, or keep decaying? This is the scientifically
   decisive follow-up and reframes whether "K6 high-mass" is a failure at all.

No charge / topological / proof / ground-state / black-hole / universal-law claim is made.
