# feb56dc7 — Long-Time Stable 4-Node State: Characterization

**Result id:** `LONG_TIME_STABLE_4_NODE_BOUND_STATE` / corrected bond verdict `FOUR_NODE_ATTRACTOR`
(repulsively-stabilized, count-regenerating, non-rotating). **Date:** 2026-06-22. JAX scout, bare
S-NCGL (gamma_A=0), N=96, FP64. **Disciplined naming:** a *T=6000-confirmed bounded multi-node steady
state under tested conditions* — NOT "infinite stability", NOT "thermodynamic ground state", NOT
"molecule" (no attractive bonding; see below). Data:
`sweep_runs/SUBSTRATE_HUNT_20260621_161557/feb56dc7_bound_state/` (`feb_bound_state.json`,
`feb_bond_test.json`, figures); code `jax_scout/feb_bound_state.py`, `jax_scout/feb_bond_test.py`,
render `plugins/visualizers/feb_bound_state_render.py`.

## Config
feb56dc7 (bare S-NCGL, gamma_A=0): D=2.733, **eta=+0.0704 (net linear LOSS)**, rho_vac=1.187,
omega0=0.0, a_coupling=2.310, **s=0.013, f=-0.486, a=+0.480 (strong cubic gain)**. The stability is a
gain/loss balance of the *nonlinear* terms (strong cubic gain a pumped, quintic/septic + diffusion
dissipate), NOT the linear eta (which is loss-side) — consistent with the eta-band refinement finding
that growth/stability is background-driven, not eta-driven.

## What was tested
Relaxation to T=6000; node geometry; per-node radial profiles; inter-node corridors; residual
circulation; and a CORRECTED spatial bond test (displace a node; remove a node) after the weak
Phase-A local-kick test proved insufficient.

## What was observed
- **Long-time saturation:** er 1.00 -> 1.58, flat by ~T=3000 (late slope ~ -8e-7). Stable 4 nodes
  (brief 5-node transient at T~200-700). Core density saturates ~0.93.
- **Rotation is a TRANSIENT:** dominant-core tangential v_t decays 0.10 -> ~0.001 by ~T=1500; v_r ~0.
  The final state is **non-rotating** (the "rotating soliton" framing does NOT survive to long time).
- **Geometry:** 4 nodes, pairwise distances 47-58 vox (mean 56, std 4) in a 96-box -> nodes are
  **near-maximally separated** (>half the box apart).
- **Inter-node corridors:** all 6 conductances = 0.0 -> **no density bridges** between nodes.
- **Corrected bond test (decisive):**
  - DISPLACE node toward the group (31->19 vox) -> returns to 29.6 ~ original 31 -> **REPELLED**
    (preferred separation is restored).
  - REMOVE a node (4->3) -> a **4th node REGROWS** (others did not move) -> the 4-node count is a
    genuine attractor of the dynamics.
  -> `FOUR_NODE_ATTRACTOR`.

## What is supported
- A genuine **long-time-stable (to T=6000), non-rotating, 4-node steady state** exists in bare S-NCGL.
- It is **interacting and self-organizing**, NOT passive coexistence: it actively **restores its
  preferred node spacing (repulsion)** and **regenerates its node count**. It is a real dynamical
  attractor of the field.
- The rotation seen at T<=1600 is a **transient** that vents the initial energy imbalance before the
  state locks in.

## What is NOT supported / unknown
- **NOT a bonded "molecule":** the interaction is REPULSIVE (nodes repel to a preferred spacing) with
  ZERO density corridors -> no attractive bonding. Better described as a repulsively-stabilized
  multi-node lattice-like state.
- **NOT proven beyond T=6000** (no claim of infinite stability / ground state).
- The **repulsion mechanism** (field-mediated; long-range vs Omega^2-geometry-mediated) is not isolated.
- The **node-regrowth** test (4th regrew, others_drift=0) is consistent with a 4-node attractor but the
  mechanism (residual-energy reformation vs true regeneration) was not separated.
- **Single isolated soliton** not established; this is inherently a multi-node configuration.
- N=96 only (resolution-converged for the earlier hi-fi, not re-verified at N=128 for this long run).

## Why this replaces the earlier "rotating soliton" framing
The hi-fi (T<=1600) showed strong circulation (v_t~0.5) and looked like a rotating core. At T=6000 the
rotation decays to ~0 and the energy locks into a steady NON-rotating 4-node attractor. So the object
is a long-time-stable, repulsively-stabilized multi-node state — the rotation was the *relaxation
channel*, not a persistent feature. The earlier eta-band "sustainers" were transient growers; feb56dc7
(cubic-gain-balanced, eta loss-side) is the genuinely-steady one.

## Next (Phase C, deferred until this is committed)
True long-time saturation search around the feb56dc7 regime (eta>0 + strong cubic gain), classifying
by T~6000 SATURATION (not T=1600 sustain): `TRUE_SATURATED_BOUND_STATE` vs `TRANSIENT_GROWER_REJECT` /
`LATE_BLOWUP` / `FRAGMENTATION` / `DELOCALIZED_HALO` / `SPIN_DOWN` / `WINDOW_ARTIFACT`. Open question:
do single-node (or 2-/3-node) saturated states exist, or is multi-node + repulsive spacing generic?
