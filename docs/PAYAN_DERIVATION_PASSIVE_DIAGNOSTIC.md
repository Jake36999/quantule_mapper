# Payan-State Derivation — Stage 0 (framework grounding + passive diagnostic spec)

**Status:** RFC derivation stage for `docs/PAYAN_PHASE_ALIGNMENT_RFC.md` §3. **No solver coupling.**
**No code yet** — this defines, from the framework math, *what* a Payan state is and a *passive*
diagnostic computable from the current `ψ=√ρ e^{iφ}` field. **Date:** 2026-06-21.

**Earned claim (disciplined):** *Geometry-only routing failed under corrected, N-expanded tests
(`STRONG_BRIDGE_ROUTING_NO_SUPPORT_CORRECTED_GATE_N_EXPANDED`), making Payan-state / phase-alignment
the next justified hypothesis to formalize.* NOT: "Payan is proven", "spin is born", "FMIA wires
solved", or "IRER validated."

## Sources (framework, not invented)
- `docs/_Declaration of Intellectual Provenance v9.txt`: Payan States = **"spin along axis"**; tied to
  **Angular Deficits**, **quantule chiral pairs**, "missing angles".
- `Downloads/chiral too emergent forces..txt`: **Angular Deficit Δθ** = a Quantule's "topological
  failure to close", a *source of geometric tension*; the residual angular imbalance resolves as a
  "continuous, compensatory rotational action — a quantized internal resonance mode" = **Payan State
  = intrinsic spin** (§3.2). **Alignment / phase-locking of Payan states "function as resonance
  gates for all interactions"** (§4.3). Chirality = handedness of the spin alignment. ρ=|ψ|²,
  φ=angle(ψ), ∂φ = information current (§ field defs).
- `docs/IRER_MATH_REFERENCE.md:344`, `IRER_MATH_SANITY_CHECK.md:103`: angular-deficit / Payan = the
  3D Quantule topological structure, currently **emergent-only**, not an explicit variable.

## The 10 derivation questions

1. **What is a Payan state?** A node's intrinsic spin arising from an *angular deficit* — a local
   failure of the OIW phase to close smoothly ("missing angle"). It is the compensatory rotation
   that resolves that mismatch. Operationally: a **topological charge of the phase field around the
   node, oriented along an axis** ("spin along axis").
2. **Represented as which?** Most faithfully as **phase winding / circulation (topological charge)**
   of `∇φ` around the node, with an **orientation axis** (the spin axis) and a **sign** (chirality).
   "Vorticity", "winding", "angular deficit", "topological charge", "spin-like orientation" in the
   question list are the *same object* viewed differently: vorticity `∇×∇φ` is the density of that
   topological charge (nonzero only at ψ-zeros / vortex lines); circulation `∮∇φ·dl` is its
   integral. **Not** merely a derived coherence scalar — it is a signed, axial, topological quantity.
3. **What is computable from the current ψ without changing the solver?** Everything needed:
   `ρ=|ψ|²`, `φ=angle(ψ)`, `v≡∇φ`, the current `J=ρ∇φ=Im(ψ*∇ψ)` (already in
   `transfer_diag.geometry_fields`), and the **circulation/vorticity of v**. Phase singularities are
   the ψ-zeros (`ρ→0`); winding is the integral of `∇φ` on a loop around a node.
4. **Can we define a passive Payan diagnostic first?** Yes — see "Passive diagnostic" below. It is a
   pure post-processing of existing snapshots; no solver change, default-off by construction.
5. **What counts as Payan alignment between two nodes?** Their spins being **co-oriented and
   co-handed along the shared (bridge) axis**: same-sign axial circulation `s_i·s_j > 0` and small
   angle between spin axes. Framework reading: aligned Payan states phase-lock → the corridor's
   "resonance gate" is open.
6. **What counts as Payan mismatch?** Opposite handedness (`s_i·s_j < 0`) or large spin-axis angle →
   topological frustration; the corridor cannot phase-lock (gate closed).
7. **How would alignment reduce bridge tension / preserve stability?** Hypothesis (framework §5.2:
   "chiral torsional resonances around topological ring structures"): a scalar density bridge has *no
   internal channel* to vent accumulating gradient tension → it blows up (exactly our `energy_drift`
   wall, 380/576). If the two nodes' Payan spins are aligned, the corridor supports a **shared
   torsional/rotational mode** that converts density-gradient tension into rotational phase-energy,
   relieving it → the bridge stays bounded. Mismatch → no shared mode → tension accumulates → drift.
8. **How does it stay default-off and falsifiable?** The diagnostic is passive (reads ψ, changes
   nothing). Any *future* coupling parameter must reduce to baseline bit-exactly at 0 (the Stage B
   λ=0 gate). Falsifiable via the controls below and a sharp prediction (next section).
9. **Null controls it must beat?** See "Mandatory controls" — chiefly phase-scramble and
   density-preserved phase-randomization: if a *randomized* phase reproduces the same "alignment"
   signal, the signal is not topological and the hypothesis fails.
10. **What would prove this direction wrong?** If, across the existing 576-eval population, **Payan
    alignment shows no correlation with bridge persistence / energy-drift** (stable webs and
    blown-up strong bridges have the same alignment distribution), OR a phase-scramble reproduces the
    signal → `PASSIVE_PAYAN_ALIGNMENT_NO_SIGNAL` → `PAYAN_COUPLING_NOT_JUSTIFIED`.

## Passive Payan diagnostic (concrete, computable now; NOT yet implemented)

For a settled field ψ on the N³ grid, for each detected node k (from `detect_nodes`) and the bridge
axis `â_ij` between a node pair (i,j):
- **Axial Payan spin** `s_k = (1/2π) ∮_{C_k} ∇φ · dl`, where `C_k` is a small loop on the node's
  bounding sphere in the plane ⊥ `â_ij` (discrete: sum phase increments, branch-cut–safe via
  `angle(ψ)` differences). This is literally "spin along the [bridge] axis". Sign = chirality.
- **Vorticity field** `ω = ∇×∇φ` (compute via `∇×(J/ρ)` with a ρ-floor) → concentrated on vortex
  lines; net flux through the corridor cross-section = the corridor's topological charge.
- **Payan alignment** `A_ij = ŝ_i·ŝ_j` (axial sign/handedness agreement), optionally weighted by a
  corridor **phase-coherence** order parameter `R_ij = |⟨e^{iφ}⟩|` along the bridge tube (Kuramoto-
  style; framework's `cos(θ_j−θ_i)` coupling, §collapse).
- **Angular-deficit proxy** `Δθ_k` = degree of non-closure of φ around the node (residual of the
  loop sum beyond the nearest 2πn).

Then test the **falsifiable prediction** on data ALREADY IN HAND (the 576-eval substrate population +
the prior strong-bridge configs): **does Payan alignment `A_ij` (and corridor coherence `R_ij`)
correlate with bridge persistence / anti-correlate with `energy_drift`?** Specifically: do the
stable weak-bridge webs have *partial alignment*, and do the destabilizing strong-bridge configs
*lack alignment* (frustrated)? This needs **no new solver and no new long run** — only post-
processing of stored snapshots (and cheap re-settles where snapshots weren't kept).

## Mandatory controls (any Payan signal must beat all)
phase-scrambled; density-preserved phase-randomized (same ρ, random φ); same-density/different-phase;
same-node-count no-bridge; weak-bridge; seed perturbation; parameter perturbation; longer-window
stability. (The phase-randomized control is decisive: Payan is a *phase-topology* claim — if random
phase reproduces the alignment↔stability link, it is not Payan.)

## Classifications for this and the next stage
- **`PAYAN_DIAGNOSTIC_DEFINED`** ← *this document's outcome*: a concrete, framework-grounded,
  passively-computable diagnostic exists (above).
- `PAYAN_DIAGNOSTIC_NOT_DEFINED` (not the case here).
- Next stage (running the passive diagnostic): `PASSIVE_PAYAN_ALIGNMENT_CORRELATES_WITH_STABILITY`
  (alignment↔persistence holds and beats controls) → `PAYAN_COUPLING_RFC_READY`; or
  `PASSIVE_PAYAN_ALIGNMENT_NO_SIGNAL` → `PAYAN_COUPLING_NOT_JUSTIFIED`.

## What this stage concludes
`PAYAN_DIAGNOSTIC_DEFINED`. The next step is to **implement and run the passive diagnostic** on the
existing substrate population (no solver coupling), with the controls above, to decide
`PASSIVE_PAYAN_ALIGNMENT_CORRELATES_WITH_STABILITY` vs `..._NO_SIGNAL`. Only a positive there →
`PAYAN_COUPLING_RFC_READY`, and only then is a default-off coupling proxy (Stage 2) justified.
**Do not build coupling before the passive correlation is shown.**
