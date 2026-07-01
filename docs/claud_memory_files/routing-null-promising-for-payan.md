---
name: routing-null-promising-for-payan
description: Corrected denominator-safe routing hunt = window-robust NULL; strong geometric bridges do not route; bridge/void ratio is invalid; next direction is Payan/phase-alignment
metadata: 
  node_type: memory
  type: project
  originSessionId: a478a4fb-4a23-4687-9590-1997693e8c84
---

After Stage B (`TENSOR_GEOMETRY_NO_SUPPORT`, see [[stage-b-tensor-geometry-no-support]]) returned to
the hunt on a NEW branch `hunt-corrected-substrate` (off proxy baseline `1ef9cd8a4`; main untouched;
Stage B preserved on `phase-b`). Built a **denominator-safe routing gate**
(`jax_scout/afield_routing_gate.py`) + corrected hunt (`afield_routing_hunt.py`).

**Result: `PROMISING_FOR_PAYAN_PHASE_ALIGNMENT` (window-robust NULL).** Strong geometric bridges do
NOT produce selective inter-node routing:
- Absolute phase-kick responses are ~1e-4 noise at short windows; the responses that grow with a
  longer window are boundary-pinned (unresolved drift), NOT bridge-specific (gen29 cont=2000:
  node 0.289 > bridge 0.229), and the substrates go nonfinite by cont=4000.
- The bridge/void RATIO is intrinsically invalid: a NO-bridge config reaches **b/v=949** by
  denominator collapse. No absolute-bridge floor passes a real bridge without leaking the no-bridge
  control. So the prior "web→wires" signals (proxy gen6 0.20→9.21, Stage B gen29 →205) were
  **denominator-collapse artifacts** in under-resolved windows.
- Confirmation hunt: `0/4` bounded strong-bridge substrates routed; controls clean.

**Why / how to apply:** the mechanisms tried (scalar Ω², current-coupled A, faithful anisotropic
metric) couple geometry to density/current/stress-DIRECTION — amplitude/flow quantities — and none
route. The indicated next DOF is **Payan-state / phase-alignment / angular-deficit** (a coherence/
topology quantity, currently emergent-only in the framework). RFC: `docs/PAYAN_PHASE_ALIGNMENT_RFC.md`
(scope only — do NOT build coupling yet; first ground the Payan field/coupling in the framework math
and optionally broaden the substrate pool beyond the current N=4).

**N-EXPANDED (2026-06-21, caveat resolved):** a fresh 4h evolutionary substrate hunt
(`afield_substrate_hunt.py`, 24 gens / 576 evals; substrate-quality objective, no routing in the
loop) + denominator-safe validation (`afield_routing_validate_pool.py`) →
`STRONG_BRIDGE_ROUTING_NO_SUPPORT_CORRECTED_GATE_N_EXPANDED`. TWO independent nulls: (1)
routing-horizon-stable STRONG (≥0.3) bridges essentially DON'T EXIST — Pareto wall, ceiling
cond_settle=0.272, strong bridges destabilize (energy_drift 380/576); (2) the 15 strongest stable
bridges (0.21–0.27) produce NO above-noise routing (15/15 WEAK_ABS_BRIDGE, controls clean). The
GEOMETRY-ONLY ROUTING ARC IS CLOSED. Manifest:
`sweep_runs/SUBSTRATE_HUNT_20260621_161557/N_EXPANDED_ROUTING_NULL_MANIFEST.json`. The Payan RFC's
broaden-evidence gate is satisfied; next blocking step = DERIVE the Payan field/coupling from the
framework math (RFC §3), still do NOT build.

**Permanent metric lesson:** never reward a bridge/void RATIO alone. Require absolute bridge response
above a floor, void denominator above a floor, RESOLVED (non-boundary-pinned) response,
bridge>node (bridge-specific), node count preserved, energy/curvature bounded, seed robustness, and
weak + no-bridge controls that stay clean. Codified in `afield_routing_gate.py`.
Data: `sweep_runs/AF_BRIDGE_HUNT_20260621_060714/ROUTING_NULL_RESULT_MANIFEST.json`. See [[no-cupy-dev-box]].
