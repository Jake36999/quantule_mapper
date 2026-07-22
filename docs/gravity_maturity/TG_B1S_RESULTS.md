# TG-B1S State-Load Feedback Results

Timestamp: 2026-07-14.
Run directory: `/mnt/f/quantule_mapper/sweep_runs/TG_B1S_STATE_LOAD_GPU_20260714_140157`.
Status: `TG_STATE_LOAD_GEOMETRIC_BACKREACTION_DETECTED`.

## Summary

This run tests exactly `STATE_LOAD_FEEDBACK` using the TG-S-supported `S_state` source. It does not enable `R_relax`, `L_lock`, or `P_threshold`.

The hardened run is:

`/mnt/f/quantule_mapper/sweep_runs/TG_B1S_STATE_LOAD_GPU_20260714_140157`

GPU preflight showed JAX backend `gpu`, device `cuda:0`, JAX/JAXLIB `0.10.2`, and x64 enabled.

## Baseline Gate

- Q-ball residual: `1.1854848438084138e-09`.
- Energy drift: `5.470133127026032e-14`.
- Charge drift: `6.768139891839495e-14`.
- Profile overlap: `0.999999999999997`.
- Shell-flux proxy: `2.148959071815996e-10`.
- Status: `TG_SOURCE_NODE_BASELINE_CLOSED`.

The source normalization was fixed globally from the TG-S stationary reference `S0 = 135.6862187684289`; no case-wise source normalization was used.

## Key Metrics

- Feed-forward T peak: `0.00225097903078486`.
- Feed-forward G peak: `0.0005703573852635597`.
- Full-loop minus feedback-off final core energy: `-1.5745503789688087e-09`.
- Full-loop minus feedback-off charge: `1.4210854715202004e-14`.
- Full-loop minus feedback-off node-frequency shift: `4.778801965255042e-08`.

## Controls

- Source-off removed T and G.
- Temporal-off removed T and downstream G.
- Geometric-off retained T and removed G.
- Global phase rotation reproduced the full-loop response.
- Translated node produced a translated nonzero response.

## Validation

The validation table records T/G amplitude ratios plus full-loop minus feedback-off deltas for `dt/2`, grid-refined, larger-box and absorber-width checks. All listed validation rows passed the preregistered feed-forward amplitude tolerance, and `dt/2` retained a nonzero backreaction delta.

## Caveat

The detected backreaction is small and should be treated as a short-run geometric backreaction detection, not as bounded long-term feedback. A later longer-duration stability run is required before using `TG_STATE_LOAD_BOUNDED_FEEDBACK_SUPPORTED`.

## Bounded Labels

- `TG_STATE_LOAD_FEEDFORWARD_CHAIN_SUPPORTED`
- `TG_STATE_LOAD_GEOMETRIC_BACKREACTION_DETECTED`

No gravity, photon, objective-time, geodesic, universal-free-fall, or IRER-validation claim is made.
