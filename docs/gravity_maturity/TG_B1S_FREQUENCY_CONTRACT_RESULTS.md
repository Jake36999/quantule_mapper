# TG-B1S Frequency-Contract Check — Results (FC-1)

Author: Claude, 2026-07-15. Script: `jax_scout/gravity_TG_B1S_frequency_contract.py` (CPU-only; the live D4 GPU
campaign was untouched). Run: `sweep_runs/TG_B1S_FREQ_CONTRACT_20260715_181825`.

**Verdict (preregistered thresholds):** `TG_B1S_FREQUENCY_CONTRACT_SIGN_MATCH_ONLY` — with the honest reading that
the measured shift is **bracketed** by the two analytic limits (0.25×–2.2×), and the audit's "sign puzzle" is
**resolved as a reporting-convention artifact**, not a physics discrepancy.

## The convention resolution (supersedes audit finding 3's sign concern)

The D3/D4 observable `delta_omega_infty` is the slope of `Δθ = θ_full − θ_off`, and the modal phase runs as
`θ ≈ −ωt`. Therefore the measured `delta_omega_infty = −2.1509e-6` means the **physical** modal frequency shift is

```
ω_full − ω_off = −delta_omega_infty = +2.1509e-06     (the full-loop node oscillates FASTER)
```

Both analytic channels predict **positive** physical shift — same sign as measured. The equation-audit's tension
("naive estimate positive vs measured negative") was an artifact of comparing conventions; there is **no sign
anomaly in the model**.

## The two analytic legs (quasi-static, first order in A−1)

Static screened T/G response solved exactly in Fourier space from the frozen Q-ball's `S_state`
(κ² < ω_T²ω_G² holds, margin 3.7×, so the static solve is well-posed):

- `T_node=+4.29e-4`, `G_node=−2.87e-4` → **sign chain confirmed quantitatively** (G<0 → A>1: A−1 up to `4.8e-5`
  in the core — the A-hill of audit finding 2, now with numbers).
- Analytic steady `T_peak=1.44e-3` vs the short-run measured `2.25e-3`: consistent — an underdamped driven field
  (Q-factor ~15) overshoots its steady value by ~1.5–2× during ring-in.

| leg | meaning | predicted physical Δω | ratio to measured (+2.1509e-6) |
|---|---|---|---|
| 1 — fixed profile (Rayleigh) | direct stiffening, no relaxation | `+5.398e-07` | 0.25 |
| 2 — fixed Q (envelope theorem over the Petviashvili family) | fully relaxed adiabatic limit | `+4.690e-06` | 2.18 |

Bonus independent re-check: `dQ/dω = −618 < 0` — the D3 Q-ball family is VK-stable (fresh confirmation on this
exact branch).

## Interpretation (bounded)

The measured shift sits **between** the unrelaxed and fully-relaxed analytic limits. The natural mechanism reading:

> The D3/D4 frequency shift is the **quasi-static geometric stiffening response with partial profile relaxation** —
> right sign, right order of magnitude, bracketed by the two analytic limits.

What would sharpen it from `SIGN_MATCH_ONLY` toward a full magnitude match: (a) using the *time-averaged dynamical*
`A(x,t)` from a run instead of the analytic steady field; (b) second-order/relaxation-dynamics treatment; (c) the
FC-2 static-profile contract (compare the analytic T/G fields against the D3 100P steady fields directly). None of
these block D4/D5 — the contract already upgrades the shift from "numerically converged" to "mechanistically
understood in sign and scale".

## Boundary

Analytic-mechanism check of the frozen phenomenological model only. No gravity, time-dilation, objective-chronology,
or IRER-validation claim. No model, label, or production change; formal promotion still rests on D4+D5.
