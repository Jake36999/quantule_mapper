# Phase C — Adiabatic Drag: Coupling Form & Static-Well Control Design

**Status:** DESIGN — for review before the moving-well run. Script `jax_scout/feb_adiabatic_drag.py`
(default-off variant). Target = confirmed a\* state (param_a ×1.15, seed 619, 4-node). Segregated from Phase C
basin results. **No inertial-matter claims** — this measures **relational mobility / adiabatic tracking**.

## Why this test (one line)

For a dissipative attractor the mobility question is *"can it track a slowly-moving energetic preference?"*,
not *"can it coast after an impulse?"* (the latter is structurally absent — see
`docs/PHASE_C_KICK_INERTIA_AND_OPERATOR_FINDING.md`).

## Proposed coupling form (for your approval)

A weak, localized **real** preference field is added to the real-space RHS only:

```
N(ψ)  →  N(ψ)  +  V0 · G(x − x_c(t)) · ψ ,      G(r) = exp( −|r|² / (2 w²) )
```

- **Real coefficient V0** (a localized modulation of the local gain/loss balance) — this is the key choice: it
  keeps the substrate **dissipative**, so we test gradient-following, *not* an injected conservative/inertial
  trap. (A Schrödinger-style `i·V·ψ` would add dispersive dynamics and defeat the purpose — deliberately NOT used.)
- **V0 > 0** = a localized low-loss / net-gain "comfort well" the attractor should prefer to occupy.
- **G** = isotropic Gaussian, width `w` (≈ the soliton core scale), centred at `x_c(t)`.
- **Default off: V0 = 0 ⇒ drag_field = None ⇒ exact baseline** (the `n_op` term is skipped entirely). Implemented
  as the optional `drag_field` argument in `physics.n_op` / `physics.step`, mirroring the existing default-off
  `a_vec` / `q_tensor` fields; baseline is byte-identical by construction and is asserted at run start
  (`BASELINE_REPRODUCED`).
- `x_c(t)` = constant (static control) or `x0 + v_well·t` (moving).

## Static-well control (runs FIRST; gates the moving test)

Battery on the a×1.15 state, each cell evolved T≈6000 steps, COM/mass/peak/nodes tracked:

1. **Baseline** (`drag_field=None`) — confirm COM stays put and mass/peak/nodes unchanged ⇒ `BASELINE_REPRODUCED`.
2. **Well ON-CENTRE** (x_c = soliton COM) — sanity: a centred well should at most tighten it, not move it.
3. **Well OFFSET** (x_c = COM + `offset·x̂`, a few offsets and/or V0 levels) — **the key test**: does the
   soliton COM bias *toward* the well?
   - Metric: Δ = (COM_fin − COM_0)·(offset direction), normalized by `offset`.
   - Coherence guard: mass / peak / node-count retained (V0 small enough not to disrupt — start weak, escalate).

**Gate:**
- COM biases toward the offset well, coherent ⇒ `STATIC_WELL_BIAS_SUPPORTED` → proceed to moving well.
- COM ignores the well ⇒ `STATIC_WELL_NO_COUPLING` → **stop**; report `RELATIONAL_MOBILITY_NOT_SUPPORTED`
  (via this drive) and reconsider the coupling before any moving run.

Proposed starting knobs (for review): `w ≈ 1.0` (L=10, N=96), offset ≈ 1.5–2.0, V0 ∈ {0.02, 0.05, 0.1}
(relative to η≈0.07 uniform loss — a local loss-cancellation of comparable scale). These are guesses to be
sanity-checked in the static battery; I'll tune V0 to the weakest value that produces a clean bias without
disrupting the 4-node structure.

## Moving-well test (HELD — not launched until static coupling is shown)

Only if `STATIC_WELL_BIAS_SUPPORTED`: start the well at the soliton COM, move at `v_well` along x, measure
- COM tracking of the well; **lag** = x_well − x_COM;
- deformation (er / peak / node-count);
- mass / energy retention;
- **slip threshold** — scan v_well upward until the COM detaches from the well.
Labels: `ADIABATIC_DRAG_TRACKING_SUPPORTED`, `ADIABATIC_DRAG_SLIP_THRESHOLD_FOUND`, or
`RELATIONAL_MOBILITY_NOT_SUPPORTED`.

## Contract / safety

- Default-off physics variant only; **no other PDE/solver changes**; baseline path unchanged & asserted.
- All output under a dedicated `sweep_runs/FEB_ADIABATIC_DRAG_*` root — **not** mixed with Phase C basin runs.
- Verdicts labelled relational-mobility / adiabatic-tracking; no inertial-matter / no isotope / no QCD language.
- Geometry frozen `e8d6a78ea`.
