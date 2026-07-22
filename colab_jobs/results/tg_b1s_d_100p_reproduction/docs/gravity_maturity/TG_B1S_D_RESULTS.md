# TG-B1S-D Drift Decomposition Results

Timestamp: 2026-07-14.
Run directory: `/content/qm_job/results/tg_b1s_d_100p_reproduction/simulation`.
Status: `TG_STATE_LOAD_LONG_TIME_DRIFT_UNRESOLVED`.

## Labels

- `TG_STATE_LOAD_LONG_TIME_DRIFT_UNRESOLVED`

## Preserved Prior Labels

- `TG_STATE_LOAD_BACKREACTION_ROBUST`
- `TG_STATE_LOAD_BACKREACTION_ROBUST_BUT_WEAK`
- `TG_STATE_LOAD_CONTINUOUS_SLOW_DRIFT`

## Key Metrics

- Target run: `D3_100P_lam1`.
- Final phase-aligned orbital distance: `4.9140079664587915e-05`.
- Final phase drift: `-0.0013329967322306402`.
- Asymptotic frequency slope: `-2.150936882856951e-06`.
- Structural channels: ``.

## Interpretation

The audit separates phase drift from profile drift, but the available gates do not yet distinguish stable frequency shift, structural drift and numerical accumulation robustly enough for promotion.

No damping, coupling, source normalization or field map was changed. No bounded-feedback, gravity, photon, objective-time, geodesic, universal-free-fall, production or IRER-validation claim is made unless explicitly listed above.