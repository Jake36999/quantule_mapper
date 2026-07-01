# RFC — Payan-state / Phase-Alignment Coupling (scope, staged)

**Status:** DRAFT scope contract. NOT implemented. Motivated by the routing NULL below. JAX scout
only; production untouched. **Date:** 2026-06-21. **Do NOT build the coupling yet** — this RFC
records the decision that Payan-state coupling is the *indicated* next direction and specifies the
staged investigation + validation gate, per the agreed sequence.

## 1. Evidence base (why this RFC exists)

Three mechanisms have now been tried and none produce selective inter-node routing:
1. **Scalar Ω²(ρ) geometry** → globally-coupled web (`global_mode ≈ 0.89`), no selectivity.
2. **Current-coupled A-field** → bridge-localizing but no robust web→wires (1/3).
3. **Faithful anisotropic metric** `g_ij=Ω²(δ+λQ)` (Stage B) → `TENSOR_GEOMETRY_NO_SUPPORT`
   (`STAGE_B_RESULT_MANIFEST.json`); the proxy's selectivity was a flat-form artifact.

The **corrected, denominator-safe routing hunt** (this session) then established a **window-robust
NULL** (`ROUTING_NULL_RESULT_MANIFEST.json`):
- No bounded strong-bridge substrate shows a *resolved, bridge-specific, above-noise* response.
- The bridge/void phase-kick **ratio is intrinsically invalid** — a NO-bridge config reaches
  `b/v=949` by denominator collapse; there is no absolute-bridge floor that passes a real bridge
  without leaking the no-bridge control.
- `0/4` strong-bridge substrates routed; controls clean → `PROMISING_FOR_PAYAN_PHASE_ALIGNMENT`.

**Conclusion:** mechanisms that couple geometry to **density / current / stress-direction** do not
create routing. Strong *geometric* (Ω²/density) bridges are not dynamical routing channels.

> **Caveat (do not overstate):** the confirmation rests on **N=4** unique bounded strong-bridge
> substrates (the existing pool) + the window scan + the metric-invalidity proof — not a broad
> fresh search. Broadening N (a fresh evolutionary substrate hunt) would harden the "repeatedly"
> claim before this RFC is acted on. See §6.

## 2. Hypothesis

Selective routing may require coupling geometry/evolution to the **phase-alignment / angular-deficit
("Payan") degree of freedom**, which the framework treats as the 3D Quantule topological structure
and informational-collapse threshold state — currently **emergent-only**, not an explicit dynamical
variable (`docs/IRER_MATH_REFERENCE.md:344`, `docs/IRER_MATH_SANITY_CHECK.md:103`). The intuition:
a corridor becomes a *wire* not because density bridges two nodes, but because their **phases are
aligned / phase-locked** along the corridor (a low-angular-deficit channel), so a perturbation
propagates coherently between them rather than dissipating into the web.

This is distinct from everything tried: density/current/stress are **amplitude/flow** quantities;
Payan/phase-alignment is a **coherence/topology** quantity.

> **Update 2026-06-21:** the framework grounding (§3) is DONE — see
> `docs/PAYAN_DERIVATION_PASSIVE_DIAGNOSTIC.md` (`PAYAN_DIAGNOSTIC_DEFINED`). Payan = "spin along
> axis" = axial circulation/winding of ∇φ around a node (angular deficit Δθ); alignment =
> co-handed spins along the bridge axis ("resonance gate"). A passive, solver-free diagnostic is
> specified there. The blocking next step is now to RUN that passive diagnostic (§6.2 below).

## 3. What must be derived from the framework FIRST (not invented)

Before any code, pin down from the IRER math reference / provenance:
- the precise definition of the **Payan state / angular deficit** as a field (e.g. a local
  phase-coherence or solid-angle-deficit functional of ψ), and
- the framework-sanctioned **coupling form** (how it should enter the geometry or N_op), analogous
  to how Ω²(ρ) enters now. **Do not fabricate the coupling equation** — cite its provenance.

A plausible scout starting point (to be validated against the framework, NOT assumed): a
phase-alignment field `Φ_align = |∇φ aligned between node pair| / angular-deficit proxy`, entering
as a *coherence-weighted* coupling along corridors — but the exact form is the RFC's open question.

## 4. Staged plan (mirror the anisotropy arc's discipline)

0. **Derivation/grounding** (§3) — the Payan field + coupling form, from the framework.
1. **Passive diagnostic** — measure the candidate Payan/phase-alignment field on existing nodes:
   does it localize on corridors and *distinguish* the (rare) coherent channel from generic bridges,
   on the same substrates? If a passive Payan field already separates routing-capable corridors,
   that justifies a coupling proxy; if not, the field/definition is wrong.
2. **Minimal proxy** — add the coupling default-off (parameter → 0 reproduces baseline
   **bit-exactly**, like the Stage B λ=0 gate), conservative form, guarded/bounded.
3. **Validation** — the **corrected denominator-safe routing gate** (`afield_routing_gate.py`):
   resolved + bridge-specific (bridge>node) + valid void denominator + weak/no-bridge controls
   clean + seed-robust. Same gate that produced this NULL — so a positive is meaningful.

## 5. Hard requirements (carried from Stage B)

- coupling default-off param → exact baseline (bit-identical equivalence test);
- conservative/dispersive, bounded, guarded; finite checks; contract-stamped, new distinct key;
- segregated from scalar/A/tensor leaderboards; no Hunter until the tiny panel passes;
- production (`gravity/`, `solver/`, `orchestrator/`) untouched; JAX scout only.

## 6. Decision gate (before building anything)

1. **✅ DONE — broaden the evidence (2026-06-21):** a fresh 4h evolutionary substrate hunt
   (`afield_substrate_hunt.py`, 24 gens / 576 evals) + denominator-safe routing validation
   (`afield_routing_validate_pool.py`) returned `STRONG_BRIDGE_ROUTING_NO_SUPPORT_CORRECTED_GATE_N_EXPANDED`
   (`sweep_runs/SUBSTRATE_HUNT_20260621_161557/N_EXPANDED_ROUTING_NULL_MANIFEST.json`). Two
   independent nulls: routing-horizon-stable STRONG bridges do not exist (ceiling 0.272), and the
   15 strongest stable bridges produce no above-noise routing (controls clean). **This gate is
   SATISFIED — the geometry-only routing arc is closed and the Payan direction is justified.**
2. **✅ DONE — ground the Payan FIELD (§3):** `PAYAN_DIAGNOSTIC_DEFINED`
   (`docs/PAYAN_DERIVATION_PASSIVE_DIAGNOSTIC.md`). Payan field = axial phase-winding/circulation
   of ∇φ (computable from the current ψ; emergent-only made explicit-measurable).
3. **✅ DONE (2026-06-21) — passive diagnostic = `PASSIVE_PAYAN_ALIGNMENT_NO_SIGNAL`
   → `PAYAN_COUPLING_NOT_JUSTIFIED`** (`afield_payan_diagnostic.py`,
   `sweep_runs/SUBSTRATE_HUNT_20260621_161557/payan_passive_diagnostic.json`). On 25 stable vs 25
   unstable substrates: Payan-spin alignment does NOT discriminate stability (aligned fraction
   stable 0.04 vs unstable 0.12; gap −0.08, within the phase-random control's ±0.04). The
   precondition fails too — bridges are predominantly *anti*-aligned (~90% counter-rotating vs 50%
   control), and stable bridges are *more* anti-aligned, not less. Real vortical structure exists
   (mean|s|≈5e2, beats control) but is orthogonal to stability.
4. **✅ DONE — one re-derived pre-registered test (chiral-pair BALANCE) = `BALANCE_NO_SIGNAL`**
   (`afield_payan_balance_test.py`, `payan_balance_test.json`). From "quantule chiral pairs": a real
   pair = opposite-sign, matched-magnitude spin (charge cancels). Balance B=1−|s_i+s_j|/(|s_i|+|s_j|)
   does NOT gate stability (mean B stable 0.006 vs unstable 0.005, dB +0.001, AUC 0.535). Data: spins
   are systematically ONE-SIDED — one node ~0, the other large — so **no balanced chiral pairs exist
   at all**, in either class. Visual analysis (`chiral_viz/chiral_slices_*.png`) corroborates: no
   coherent counter-rotating cores; the ~90% "anti-alignment" is noisy/weak (partly definitional),
   not genuine chiral-pair anti-handedness.
5. **Therefore the coupling is NOT justified by passive evidence (two pre-registered tests +
   visual inspection, all null).** Do NOT build the coupling. The geometry-only + passive-Payan arc
   closes as a scoped negative; the routing NULL + the alignment NULL + the balance NULL stand as the
   terminal result of this arc.

## 7. Validation metric correction (permanent lesson)

Never reward a bridge/void *ratio* alone. Always require: absolute bridge response above a floor,
void denominator above a floor, **resolved** (non-boundary-pinned) response, **bridge > node**
(bridge-specific), node count preserved, energy/curvature bounded, seed robustness, and weak +
no-bridge **λ-on controls that stay clean**. Codified in `jax_scout/afield_routing_gate.py`.
