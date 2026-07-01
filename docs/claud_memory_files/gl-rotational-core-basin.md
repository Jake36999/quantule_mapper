---
name: gl-rotational-core-basin
description: Bare S-NCGL has a real eta-dominated parameter basin of self-sustaining rotational cores (dissipative solitons); robust band eta~-0.05..0; BASIN_PARTIAL at N=96 (strong-gain edge is a resolution artifact)
metadata: 
  node_type: memory
  type: project
  originSessionId: 5cf36465-e51a-450c-b166-5b68adb5100c
---

Pivoted from the closed routing/Payan arc (see [[hifi-continuation-vortex-real]], [[payan-diagnostic-defined]])
to the live object of study: self-sustaining ROTATIONAL CORES. Ran a 6h GL parameter basin sweep
(`jax_scout/core_basin_sweep.py`, engine `physics.sweep_probe` — vmap, BARE S-NCGL, no A-field, no
coupling, no solver mod) on the GTX1080. Result `GL_ROTATIONAL_CORE_BASIN_ETA_DOMINATED_PARTIAL_N96`
(`sweep_runs/CORE_BASIN_20260622_023759/CORE_BASIN_RESULT.json`).

**Basin structure (1920 configs, N=48, T=1600; 210 SUSTAIN=11%):** the dominant control is **param_eta
(linear gain/loss)**. SUSTAIN peaks sharply at **eta ~ -0.05** (small net gain balancing higher-order
dissipation); eta > ~0.1 → COLLAPSE (dissipative spin-down), eta < ~-0.15 → BLOWUP (gain runaway). The
nonlinear (a,s,f) and geometry (a_coupling, D) terms are broadly PERMISSIVE (modulate, don't gate).
Bare S-NCGL reproduced both refs (feb56dc7→SUSTAIN er1.51, b31c0396→COLLAPSE er0.29) — so the A-field
was never needed for the rotation; the field self-spins.

**N=96 multiseed validation = `BASIN_PARTIAL` (3/6 robust):** the basin is REAL at converged resolution
but NARROWER than the N=48 map — robust band **eta ~ -0.05 to 0**. The strong-gain edge (eta < ~-0.08)
is a RESOLUTION ARTIFACT (eta=-0.108 sustained at N=48 but BLOWS UP at all N=96 seeds); near-edge
sustain is seed-probabilistic. The disciplined N=96/multiseed check mattered.

**Why / how to apply:** confirms stability = energy gain/dissipation balance of a rotational core
(dissipative soliton), NOT missing topology. Does NOT revive FMIA wires / Payan. Code:
core_basin_sweep / core_basin_validate / render `plugins/visualizers/core_basin_render.py`.

**ETA-BAND REFINEMENT (done; `ETA_BAND_REFINE_ETA_NECESSARY_NOT_SUFFICIENT`,
`sweep_runs/CORE_BASIN_REFINE_20260622_095451/CORE_BASIN_REFINE_RESULT.json`):** P(SUSTAIN|eta) over
eta∈[-0.08,0.02], N=96, 24 samples/eta (3 seeds × 8 permissive backgrounds). HONEST TEMPERING of the
"eta is the master knob" framing: eta optimum is REAL but MODEST — peak P(SUSTAIN)=0.38 at eta≈-0.035;
eta tunes SUSTAIN↔SPIN_DOWN, but BLOWUP is ~FLAT 0.38 across the band (BACKGROUND-driven a/s/f/D, not
eta), spiking only at strong gain (-0.08→0.62). So eta is NECESSARY tuning, NOT sufficient: sustain
needs good eta AND a tame nonlinear background. The N=48 "sharp eta peak" was P(eta|SUSTAIN) (a
Bayesian artifact); P(SUSTAIN|eta) tops at ~0.38 over random backgrounds. Code:
`jax_scout/core_basin_refine.py`, render `plugins/visualizers/core_basin_refine_render.py`
(refine_stability_curve.png). Robust sustaining configs: idx 1194 (eta=-0.042), 1125 (eta=-0.005), 488 (eta=-0.046).

**LONG-TIME CHARACTERIZATION (T=6000) REFUTES steady solitons**
(`BARE_SNCGL_SUSTAINERS_NOT_STEADY_SOLITONS_LONGTIME`, `sweep_runs/CORE_BASIN_20260622_023759/core_characterize/`):
all 3 bare-S-NCGL sustainers are STILL_GROWING at T=6000 (er→7.3/3.8/9.5) and lose coherence (1194
core-density runaway + despin to v_t~0.005; 1125 fragments 4→7 nodes; 488 delocalizes nodes→0). The
T=1600 SUSTAIN was a WINDOW ARTIFACT. They are robust ATTRACTORS (kicked v_t relaxes back) but onto
NON-STEADY growing trajectories — NOT steady dissipative solitons. CRITICAL: bare-S-NCGL circulation
is WEAK (v_t~0.005-0.02) vs the hi-fi feb56dc7 (v_t~0.4-0.6, run WITH the A-field, only to T=1600).
So either the A-field is essential for strong sustained rotation, or feb56dc7 also grows past 1600 →
no steady soliton exists. Two-core interaction NOT well-posed until a steady building block is found.
Code: `jax_scout/core_characterize.py`, render `core_characterize_render.py`.

**DECISIVE feb56dc7 long-time test (`jax_scout/feb_longtime.py`, `feb56dc7_longtime.json`):**
feb56dc7 is BARE S-NCGL (gamma_A=0! — never A-coupled; eta=+0.07 loss-side but a=0.48 strong cubic
gain). At T=6000 it **SATURATES**: er 1.00→1.58 (slope ~ -8e-7), 4 stable nodes (4→4), stable core
density (0.93) — a TRUE long-time-steady structure. So genuinely-stable bounded structures DO exist
in bare S-NCGL. TWO corrections: (1) the eta-basin "sustainers" (T=1600) were the WRONG objects —
they GROW (er 3.7-9.5); feb56dc7's different regime (cubic-gain-balanced) saturates. T=1600 SUSTAIN
was systematically misleading. (2) feb56dc7's ROTATION DECAYS to ~0 (v_t 0.10→0.0) — it's a steady
NON-ROTATING 4-node configuration, not a rotating soliton; the v_t~0.5 rotation was a long transient.
Net: the genuine stable object is a long-time-steady multi-node BOUND configuration (a multi-core
coexistence), NOT a single rotating dissipative soliton. Honest reframe of the "stable matter" arc.
True-soliton search would need a LONG-TIME (T~6000) saturation criterion, not T=1600.
