---
name: stage-b-tensor-geometry-no-support
description: Stage B faithful anisotropic metric (g_ij=Omega^2(delta+lam Q)) returned TENSOR_GEOMETRY_NO_SUPPORT — proxy bridge-routing was a flat-form artifact
metadata: 
  node_type: memory
  type: project
  originSessionId: a478a4fb-4a23-4687-9590-1997693e8c84
---

Built and tested the faithful anisotropic spatial metric `g_ij = Ω²(δ + λQ)` (Stage B of the
anisotropy arc) as a conservative/dispersive Laplace–Beltrami *deviation* added to the scalar path
in `jax_scout/physics.py` (`n_op` `g_aniso` arg + `_aniso_metric_deviation`; guarded 3×3 inverse,
det floor `ANISO_DET_FLOOR=1e-3`, λ-clamp `ANISO_LAMBDA_MAX=0.5`). Contract
`IRER-SNCGL-ANISO-METRIC-LB-ETDRK4-v1`. Drivers: `afield_aniso_metric.py` (+ `_validate`,
`_lamscan`, `_specificity`); test `tests/test_aniso_metric.py` (7/7 PASS, λ=0 bit-exact 0.00e+00).

**Verdict: `TENSOR_GEOMETRY_NO_SUPPORT`.** The faithful metric does NOT reproduce the proxy's
bridge-selective routing (`CONSERVATIVE_ANISOTROPY_CREATES_BRIDGE_SELECTIVE_ROUTING_PROXY`):
- gen6 NULL at all bounded λ (bridge/void 0.20→0.58, <1) → the proxy's gen6 9.21 was a **flat-space
  proxy-form artifact**; proper conformal weighting suppresses anisotropy by ~1/Ω² in the corridor.
- gen29's high-λ "signal" (bridge/void 7.83→205 at λ≥0.3) is a **confound**: generic (weak-bridge
  control gen20 explodes to 74× too), structural distortion (node fragmentation 4→5→8), seed-fragile
  (205/59/103 across 3 ICs), and γ_A=0 barely moves with λ.

**Why:** the de-confounding discipline this project runs on — the proxy's de-confound never ran
weak/no-bridge λ-on controls, so it missed that the bridge/void phase-kick ratio goes unreliable
when response peaks are tiny (void-denominator collapse: a NO-bridge config reads 33× at λ=0).

**How to apply:** do NOT build Stage C (tensor branch + Hunter) — it was riding a confounded lead.
If anisotropy is revisited, first fix the routing metric (bound the void denominator / require a
minimum absolute response) and ALWAYS include weak + no-bridge λ-on controls. Payan-state /
phase-alignment coupling is the alternative RFC direction. Full data:
`sweep_runs/AF_BRIDGE_HUNT_20260621_060714/STAGE_B_RESULT_MANIFEST.json` +
`docs/STAGE_B_TENSOR_GEOMETRY_HANDOFF.md` (⮞ STAGE B OUTCOME section). Work on git branch
`phase-b-tensor-geometry` (proxy baseline committed `1ef9cd8a4`; Stage B code uncommitted).
See [[no-cupy-dev-box]] for how the scout runs.
