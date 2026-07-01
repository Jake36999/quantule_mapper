---
name: payan-diagnostic-defined
description: "Payan = axial phase-winding of grad-phi (spin along axis); passive diagnostic RAN -> NO_SIGNAL (alignment doesn't predict stability) -> PAYAN_COUPLING_NOT_JUSTIFIED; arc closes negative"
metadata: 
  node_type: memory
  type: project
  originSessionId: 5cf36465-e51a-450c-b166-5b68adb5100c
---

After the geometry-only routing arc closed (see [[routing-null-promising-for-payan]]), did the RFC
derivation stage for Payan-state coupling (user-directed: derive from framework math, do NOT build
coupling yet). Grounded in `docs/_Declaration of Intellectual Provenance v9.txt` (Payan = "spin
along axis"), `Downloads/chiral too emergent forces..txt` (Angular Deficit Δθ = "topological failure
to close" = geometric tension; Payan = compensatory rotation = intrinsic spin; aligned/phase-locked
Payan states = "resonance gates for interactions"), and `docs/IRER_MATH_REFERENCE.md`.

**Outcome `PAYAN_DIAGNOSTIC_DEFINED`** (`docs/PAYAN_DERIVATION_PASSIVE_DIAGNOSTIC.md`): Payan state =
the **axial circulation / phase-winding of ∇φ around a node** (`s_k=(1/2π)∮∇φ·dl` on a loop ⊥ the
bridge axis; sign=chirality) — i.e. "spin along axis", an angular-deficit topological charge. Payan
ALIGNMENT between nodes = co-handed spins along the shared axis (+ corridor phase-coherence
`R_ij=|⟨e^{iφ}⟩|`). All **passively computable from the current ψ=√ρ e^{iφ}** (codebase already has
`J=ρ∇φ=Im(ψ*∇ψ)` in `transfer_diag.geometry_fields`) — NO solver change.

**Hypothesis (why/how):** a scalar density bridge has no internal channel to vent gradient tension →
blows up (the observed energy_drift Pareto wall). Aligned Payan spins let the corridor carry a shared
torsional mode that converts density tension into rotational phase-energy → stable bridge; mismatch =
frustration → drift.

**RESULT (2026-06-21): `PASSIVE_PAYAN_ALIGNMENT_NO_SIGNAL` → `PAYAN_COUPLING_NOT_JUSTIFIED`.**
`afield_payan_diagnostic.py` on 25 stable vs 25 unstable substrates: Payan-spin alignment does NOT
predict bridge stability (aligned fraction stable 0.04 vs unstable 0.12; gap −0.08, within the
density-preserved phase-random control's ±0.04). Precondition fails too — bridges are ~90%
*anti*-aligned (counter-rotating, vs 50% control), and stable are *more* anti-aligned not less. Real
vortical structure exists (mean|s|≈5e2, beats control = genuine "chiral pair"-like counter-rotation)
but is ORTHOGONAL to stability. Data: `sweep_runs/SUBSTRATE_HUNT_20260621_161557/payan_passive_diagnostic.json`.
**Do NOT build Payan coupling** — passive evidence doesn't support the alignment→stability mechanism.

**Follow-up (Option B, 2026-06-21): visual analysis + ONE re-derived pre-registered test, both null.**
Chiral-pair slice renders (`chiral_viz/chiral_slices_*.png`; capture `payan_chiral_capture.py`, render
`plugins/visualizers/payan_chiral_slices.py`) show NO coherent counter-rotating cores — the ~90%
"anti-alignment" is noisy/weak (partly definitional), not genuine chiral pairs. The re-derived
chiral-pair BALANCE test (`afield_payan_balance_test.py` → `BALANCE_NO_SIGNAL`, `payan_balance_test.json`):
B=1−|s_i+s_j|/(|s_i|+|s_j|) does NOT gate stability (dB +0.001, AUC 0.535) — spins are systematically
ONE-SIDED (one node ~0, other large), so no balanced pairs exist at all. **Two pre-registered passive
tests (alignment, balance) + visuals all null → Payan coupling NOT justified; geometry-only +
passive-Payan routing arc closed as a scoped NEGATIVE.** Building coupling now would be speculative
(no passive hook). Next move, if any, is theoretical (re-examine whether this S-NCGL substrate can
host FMIA wires at all), not more scout compute.
Disciplined wording: earned claim is only "geometry-only routing failed → Payan is the next justified
hypothesis to formalize" — NOT "Payan proven / spin born / wires solved / IRER validated".
