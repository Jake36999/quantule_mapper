---
name: feb56dc7-bound-state
description: "feb56dc7 = T=6000-confirmed stable non-rotating 4-node attractor (repulsively-stabilized, count-regenerating); NOT a bonded molecule, NOT infinite-stable; rotation is a transient"
metadata: 
  node_type: memory
  type: project
  originSessionId: 5cf36465-e51a-450c-b166-5b68adb5100c
---

Characterized feb56dc7 (the one genuinely long-time-steady config; see [[gl-rotational-core-basin]])
as the current primary stable object. Bare S-NCGL (gamma_A=0; eta=+0.07 LOSS-side, a=+0.48 strong
cubic gain -> nonlinear gain/loss balance, NOT eta-driven). N=96, T=6000. Code
`jax_scout/feb_bound_state.py` + `feb_bond_test.py`, render `plugins/visualizers/feb_bound_state_render.py`,
doc `docs/FEB56DC7_BOUND_STATE_CHARACTERIZATION.md`, data
`sweep_runs/SUBSTRATE_HUNT_20260621_161557/feb56dc7_bound_state/`.

**Result `LONG_TIME_STABLE_4_NODE_BOUND_STATE` / `FOUR_NODE_ATTRACTOR`:** er saturates 1.00->1.58
(flat by T~3000), 4 stable nodes, core density ~0.93. Rotation is a TRANSIENT (v_t 0.10->~0 by
T~1500) -> final state NON-rotating. Nodes near-maximally separated (~56 vox in 96-box), ZERO inter-
node density corridors. **Corrected SPATIAL bond test (the weak local-kick Phase-A test was a
RESTORING artifact):** displace a node inward -> REPELLED back to preferred spacing; remove a node ->
4th REGROWS (others unmoved) -> the 4-node config is a genuine dynamical ATTRACTOR, interacting +
self-organizing, but via REPULSION + count-regeneration, NOT attractive bonding.

**Why / how to apply:** honest reframe — this is a repulsively-stabilized non-rotating multi-node
lattice-like steady state, NOT a "molecule"/"ground state"/"infinite" (only tested to T=6000). The
rotation was the relaxation channel, not a persistent feature. Disciplined naming enforced (resisted
Gemini's "stable molecule" framing). Next = Phase C: true long-time SATURATION search (T~6000, not
T=1600 sustain) around feb56dc7's regime (eta>0 + strong cubic gain); open Q: do single/2-/3-node
saturated states exist or is multi-node+repulsive-spacing generic.
