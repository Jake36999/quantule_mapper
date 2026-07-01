---
name: hifi-continuation-vortex-real
description: "Resolution/time validation (N=48/96/128, T=1600) — node dynamics are REAL (converged); nodes are rotational cores; stability = vortex sustain-vs-decay (energy balance), not missing spin"
metadata: 
  node_type: memory
  type: project
  originSessionId: 5cf36465-e51a-450c-b166-5b68adb5100c
---

After the passive Payan nulls (see [[payan-diagnostic-defined]]), the user read the chiral slices as
node-merger / secondary-approach / centripetal-rotation, and proposed a higher-N longer-T run. Ran a
resolution/time ladder (`jax_scout/payan_hifi_continuation.py`, render `plugins/visualizers/payan_hifi_render.py`):
configs feb56dc7 (stable) + b31c0396 (unstable), N=48/96/128, L=10, T=1600, FP64, no solver mod, no
OOM at N=128. Result: `HIFI_CONTINUATION_STRUCTURES_REAL_VORTEX_SUSTAIN_VS_DECAY`
(`sweep_runs/SUBSTRATE_HUNT_20260621_161557/HIFI_CONTINUATION_RESULT.json`).

**The structures are REAL (resolution-converged by N=96):** er(t) curves essentially identical across
N (stable →1.48, unstable →0.29); final node counts converge (stable 4, unstable 5). NOT low-grid
artifacts.

**Corrected the visual readings (honest):**
- "runaway gravity well" = FALSIFIED: the unstable config is a smooth DISSIPATIVE decay (er 1.0→0.29,
  core density 0.82→0.21), stays finite — energy leaks away, not piles up.
- "node merger / secondary approach / decreasing separation" = NOT confirmed at high N: node tracks
  are ~stationary; that impression was a low-N/short-window artifact.
- "rotation around the core" = VALIDATED (qualified): dominant nodes show sustained tangential
  phase-current circulation (v_t large, swirl ratio ~1, v_r≈0).
- "centripetal inflow → rotation" = NOT supported: v_r≈0 (no net radial inflow); pure swirl, not a
  spiral-sink. And core density stays HIGH (no ρ→0 singularity) → these are smooth rotational cores,
  NOT topological/quantized vortices.

**Refined picture (why/how to apply):** stability is an ENERGY gain/dissipation BALANCE of the
rotational core — stable sustains circulation + densifies (bounded growth), unstable spins down +
evaporates — set by the GL params (a,s,f,eta), NOT by a missing topological-spin DOF. Spin is PRESENT
(nodes rotate) and there are NO topological vortex cores, so there is no absent angular-deficit DOF
for Payan coupling to supply → consistent with the alignment/balance nulls; **Payan coupling still
NOT justified.** If pursued further, the live question is the GL parameter basin (sustain vs decay),
a different investigation from Payan. Visuals: `hifi_convergence_overlay.png`, `hifi_N128_*/vortex_*.png`.
