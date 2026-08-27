# IRER Geometry-Density Gravity Ladder — Rung A + D: Results

**A coherent load does create a spatially-structured, geometry-dependent response in the shared geometric field
(Ω²) and the informational stress tensor (T_info) — a necessary precursor to the candidate mechanism. BUT the
response in the production geometry is a *saturation cliff* at the core boundary, not a smooth distance-graded
potential, because the conformal factor is dominated by the C2.6 soft-clip (Ω² saturates at the 1e6 cap in vacuum and
maps Ω²(ρ_vac)→~720, not 1). So the first rung's real result is: the geometry response exists and is geometry-on/off
distinct, but a clean gravity-like interpretation is GATED on characterizing/fixing the geometry (the C2.6 contract).**
Cautious framing per `docs/IRER_GEOMETRY_DENSITY_GRAVITY_RFC.md`: **not yet an emergent-gravity claim.**
Verdict: `RUNG_A_D_RESPONSE_EXISTS_BUT_GEOMETRY_UNCHARACTERIZED`.

## Setup
Dissipative production-geometry substrate (a_coupling=2.31, the real IRER conformal law), N=64, L=10, load = a
relaxed multi-blob standing cluster. Read-only production telemetry: `gravity.unified_omega` Ω²(ρ),
`metrics.tensor_validation` T_info (symmetry/shear/anisotropy), `metrics.collapse_dynamics` correlation length. Two
ambients: bg=0 (low-background sim regime) and bg=ρ_vac=1.19 (attempt at unsaturated ambient). Rung D = same load
evolved geometry-ON vs geometry-OFF (`param_geom_off`).

## Rung A — the load's geometry/tensor response (bg=0)
- **Ω² response is a saturation cliff, not a potential.** Radial profile about the load core:
  | r (box) | 0.10 | 0.52 | 0.94 | **1.35** | 1.77 | 2.60 | 4.69 |
  |---|---|---|---|---|---|---|---|
  | Ω² | 712 | 740 | 820 | **375767** | 893304 | 907745 | 929631 |
  Ω² ≈ 750 inside the dense core (r<1), then **jumps to ~9×10⁵ at r≈1.35** and stays at the vacuum saturation level.
  This is a step at the core boundary (where ρ falls below the soft-clip saturation threshold), **not** a smooth
  1/r-like falloff. The apparent "falloff length ~0.99 box-units" is an exp-fit on a step, not a physical gradient.
- **Tensor response is real:** T_info shear = 1.08, anisotropy 0.02, symmetry error 0.00 (exactly Noether-symmetric).
  Correlation length ξ = 8.8 (the merged load is extended).
- **Geometry-law characterization (the C2.6 prerequisite, now quantified):** implemented Ω² ∈ [535, 1e6] vs nominal
  (ρ_vac/ρ)^a ∈ [1.25, 2.6×10²⁷]; median implemented/nominal = 0.00. The production Ω²(ρ) is dominated by the
  log-space soft-clip (β=3 over the [1e-9,1e6] window), which amplifies Ω²(ρ_vac)=1→~720 and saturates vacuum at the
  cap. **The geometry the load perturbs is the soft-clip's shape, not the conformal law.**

## Rung D — geometry-null control (bg=0)
| observable | geometry-ON (end) | geometry-OFF (end) |
|---|---|---|
| load mass | 14501 (from 13029) | **32** (dispersed) |
| T_info shear | 1.35 | **9.8×10⁻³³** (~0) |
| Ω² deviation (ambient-ref) | 122941 | **0.0** |
**The response is strongly geometry-dependent** — geometry-off, the load disperses entirely and every geometry/tensor
response vanishes. This passes the rung-D "geometry-on ≠ geometry-off" criterion. **Honest caveat:** the difference
is *entangled with load stability* — the dissipative a\* load is geometry-stabilized, so geometry-off it disperses;
the observed on/off gap therefore reflects "geometry stabilizes the load" (Phase C physics) as much as "geometry
sources a response field." Disentangling these needs a load stable in *both* branches, or a fixed-density-layout
comparison.

## The bg=ρ_vac attempt (unsaturated ambient) — did not yield a clean well
Filling the box with ρ≈ρ_vac to get ambient Ω²≈1 failed twice over: (i) the overdensity blobs **dissolved into the
background** (n_nodes=0, no localized load — the a\* attractor lives on a *low* background, not a filled box), so
there was nothing to source a well; (ii) even so, the soft-clip maps the uniform Ω²(ρ_vac) to **~727, not 1**
(median impl/nominal = 500×). So neither a clean load nor a clean Ω²≈1 ambient was obtained. Result:
`A_D_LOAD_NOT_PERSISTENT`.

## Interpretation (cautious)
- **What is supported:** a large coherent load produces a *persistent, spatially-structured, geometry-dependent*
  response in Ω² and T_info (rung A + D positive in the low-background regime). This is the necessary precursor the
  ladder was built to check — the geometry-density channel is *live*, not inert.
- **What is NOT supported (and why):** the response is **not yet a gravity-like distance-graded potential** — it is a
  saturation cliff set by the soft-clip, and its on/off distinctness is entangled with load stabilization. There is
  **no acceleration measurement here by design** (interaction-density/tensor first).
- **The gating prerequisite is now concrete:** the RFC listed "a characterized geometry (know the actual Ω²(ρ),
  geometry-on vs true-null)" as a prerequisite. Rung A+D *demonstrates* that prerequisite is unmet: the production
  geometry is soft-clip-dominated and saturated, so a probe (rung B) placed in this field would respond to the
  saturation-cliff structure, not to a conformal gravity potential. **The C2.6 soft-clip contract must be resolved
  (characterized, or replaced by the C2′ canonical-geometry branch) before rungs B–E can be read as gravity-like.**

## Next (design owned by Claude; execution can be delegated)
1. **Geometry characterization / de-saturation** (the gate): map the effective Ω²(ρ) law over the load's ρ-range;
   find a parameter regime (a_coupling, ρ_vac, softclip window/β, or the C2′ divergence-form geometry) where Ω²(ρ)
   is a *graded* function over the load, not a saturation switch. Only then does a distance-graded response become
   measurable.
2. **A genuinely stable, balanced load** (not gain-drifting a\*×1.15) — a certified standing Phase C attractor.
3. **Load-free baseline subtraction:** Ω²(with load) − Ω²(background only) to isolate the perturbation.
4. **Disentangle stability from response:** a fixed-density-layout on/off comparison, or a conservative load stable
   in both branches.
Only after (1)–(4) should rung B (neutral probe) run — otherwise a "path bend" would be a soft-clip artifact.

## Guardrails / language
Read-only production telemetry; mirror-only evolution; no production solver/Hunter/validation changes. Language per
the RFC contract: "geometry-density interaction," "candidate mechanism," "not yet an emergent-gravity claim." No
"gravity proven," no "mass attracts mass." Provenance: `jax_scout/gravity_ladder_A_D.py`;
runs `sweep_runs/GRAVITY_AD_bg0`, `GRAVITY_AD_bgvac`.

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `584ad26` (2026-07-09) — *IRER gravity ladder rung A+D results + Codex follow-up handoff*
**Revised since:** 3 commit(s), most recently `9517d9f` (2026-08-26)

**Harness code changed since it was written:** 19 commit(s) to `jax_scout/`.
  - `06edae5` 2026-08-27 — Fix run provenance: 30% -> 86% of runs now dated
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine
  - `25020bf` 2026-07-22 — dual substrate
  - *…and 14 more.*

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Also referenced by (same date or earlier):** [[GRAVITY_LADDER_CODEX_HANDOFF]]

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
