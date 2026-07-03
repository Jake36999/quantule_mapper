# Provenance Kinetic Audit — Formalism Gap Review

**Task type:** reading/audit only. **No runs, no code changes, no new PDE terms in the body** (capability ideas
→ Parking-Lot appendix). Source: `docs/_Declaration of Intellectual Provenance v9.txt` (3647 lines) + concept
appendices, read against the Stage-1 baseline audit.

**Question:** does the theory imply **(A)** a dissipative stability-sector substrate, **(B)** a
conservative/dispersive transport-sector substrate, **(C)** both as separate sectors, or **(D)** are the
kinetics/time-dynamics under-specified?

## Method note — the operator vocabulary is largely absent
A whole-document term census: `Hamiltonian`, `wave equation`, `Schrödinger`, `momentum`, `advect`,
`complex kinetic`, `gradient descent` = **0 occurrences**. `second-order`/`first-order` = 1 each (incidental,
not a pinned field time-derivative). By contrast the free-energy / phase-field functional `F[·]` appears **23×**.
⇒ The theory does **not** specify a conservative/dispersive kinetic operator; its dominant *mathematical* framing
is the **free-energy gradient-flow** (`F[φ]`, `∇F`, `∇S`) — an inherently **relaxational/dissipative** scaffold.

## Findings by question (classified + cited)

**Q1 — first- vs second-order time dynamics.** The theory frames time as an *"Emergent Chronology of Action /
Resolution"* — an ordered *sequence* of resolutions (Concept: Time). No wave-equation / second-order field
dynamics is specified. → `AMBIGUOUS_OR_UNDER_SPECIFIED` (leans first-order/sequential, dissipative-compatible).

**Q2 — real diffusion vs complex/dispersive kinetics.** No Schrödinger/complex-kinetic/dispersive *operator* is
stated; the pervasive `F[φ]` gradient-flow scaffold is real/dissipative. → `AMBIGUOUS_OR_UNDER_SPECIFIED`; the
theory's own scaffold **supports the current dissipative baseline**.

**Q3 — "Gradient-Derived Informational Forces": relaxation or transport?** Concept 20 (`line 490-491`): laws
*"governing forces, motion, conservation principles… emerge… Forces… are Gradient-Derived… arising from
informational gradients."* The **intent** includes motion + conservation (transport-leaning), but the mechanism
is gradient-derived (in a gradient-flow = relaxation) and is an *emergent* "protocol," not an operator. →
`SUPPORTS_TRANSPORT_EXTENSION` in intent, but `ASPIRATIONAL_NOT_FORMALIZED`.

**Q4 — FMIA / Informational Parallels: do they require propagation?** `line 521`: *"FMIA defines the default
**stabilization channel**, minimizing 'Entropy as Resonance Fragmentation'… may act as informational **low-pass
filters** suppressing novelty."* The *defined role* is stabilization/smoothing, not transport. →
`SUPPORTS_CURRENT_DISSIPATIVE_BASELINE` (FMIA = stabilization; "channels/parallels" are nominal, not a
propagation requirement).

**Q5 — Payan / spin: conserved circulation?** Payan = *"quantized 'spin-along-axis'… chirality"* with
*"**topological phase locking**"* (`line 469-474`), *"coherent **phonon-like propagation** / spin-like waves…
mediating long-range resonance"* (`line 481`), and *"analogy to gauge symmetry or topological charge
(**exploratory**)"* (`line 480`). These imply conserved circulation + propagation →
`SUPPORTS_TRANSPORT_EXTENSION` — **but** every instance is hedged (*"may exhibit," "exploratory," "ongoing
exploration," "analogous"*), Payan is not an explicit simulation variable (`IRER_MATH_REFERENCE.md:344`) → also
`ASPIRATIONAL_NOT_FORMALIZED`. **Split concept** (see alignment note).

**Q6 — non-local "splash" / Field of Affect: conceptual, mathematical, or implemented?** Non-locality is
*conceptual* (informational interference across the substrate); *"locality could be relative to entropic load…
a dynamic property"* (`line 619-621`). Mathematically it is *scaffolded* (the Field of Affect is **computed**)
but **not coupled** into the baseline evolution (`IRER_MATH_REFERENCE.md:342`; architecture audit). →
`AMBIGUOUS_OR_UNDER_SPECIFIED` (conceptual expectation, partially scaffolded, not an implemented coupling).

**Q7 — phase-field / reaction-diffusion: final formalism or provisional scaffold?** The provenance frames the
phase-field formalism as an **AI-suggested "formal translation"** of the author's qualitative concepts
(Sections 3.2, 4.1; "phase-field discussion", Appendix C); `reaction-diffusion` (3×) describes the *refactored
simulator*, and `scaffold` appears once. The formalism is a **chosen analogy/scaffold**, not a theory-mandated
operator. → `AMBIGUOUS_OR_UNDER_SPECIFIED` (provisional scaffold).

## Key alignment note (a positive theory↔result match)
The theory's **"resonant inertia" = *"resisting perturbations until a critical threshold"*** and **"measurable
informational half-life"** (`line 474`), plus its **thermodynamic / Boltzmann / "informational temperature"**
framing (`line 477`) and *"desire to be conservative of energy"* balanced against entropic load (`line 621`), are
**dissipative/relaxational** concepts — and they **match the Phase C findings directly**: site-pinning
(resistance to relocation), slow-decay dissipative solitons (half-life), and gain/loss balance. So the current
baseline is not merely a convenient scaffold; it **faithfully implements, and Phase C empirically confirmed, the
theory's stability-sector concepts.** The theory's *"resonant inertia"* is precisely resistance-to-perturbation,
**not** Newtonian coasting — consistent with the mobility null.

## Classification summary
| theory element | tag | cite |
|---|---|---|
| FMIA = stabilization channel / low-pass | `SUPPORTS_CURRENT_DISSIPATIVE_BASELINE` | 521 |
| "resonant inertia" = perturbation-resistance; "informational half-life" | `SUPPORTS_CURRENT_DISSIPATIVE_BASELINE` | 474 |
| thermodynamic / Boltzmann / minimal-action / "conservative of energy vs entropic load" | `SUPPORTS_CURRENT_DISSIPATIVE_BASELINE` | 477, 621 |
| time = chronology of resolution (sequential) | `AMBIGUOUS_OR_UNDER_SPECIFIED` (→ dissipative-compatible) | Concept: Time |
| kinetic operator / time-order (no Hamiltonian/wave/Schrödinger; `F[φ]` scaffold) | `AMBIGUOUS_OR_UNDER_SPECIFIED` | census; Q1/Q2/Q7 |
| non-local splash / Field of Affect (computed, uncoupled) | `AMBIGUOUS_OR_UNDER_SPECIFIED` | 342, 619-621 |
| Gradient-Derived Forces → motion + conservation | `SUPPORTS_TRANSPORT_EXTENSION` / `ASPIRATIONAL_NOT_FORMALIZED` | 490-491 |
| Payan quantized spin / topological phase-locking / spin-wave propagation | `SUPPORTS_TRANSPORT_EXTENSION` / `ASPIRATIONAL_NOT_FORMALIZED` | 469-483 |
| gauge/topological-charge analogy; Quantule structure-formation | `ASPIRATIONAL_NOT_FORMALIZED` (exploratory) | 480, 483 |

## A/B/C/D determination → **C + D**
The theory conceives **both** sectors: a **stability sector** (dissipative-aligned: FMIA-stabilization, resonant
inertia, half-life, thermodynamic/minimal-action) that the current baseline **implements and Phase C validated**;
**and** a **matter/transport sector** (conservative-leaning: gradient-forces→motion, Payan spin/topological
charge, spin-wave propagation) that is **entirely aspirational and un-formalized**. The **kinetic operator and
time-order are under-specified** — no conservative/dispersive/wave operator is stated; the phase-field GL is a
provisional scaffold.

## Final conclusion → **2 + 3**
> **(2)** The current dissipative baseline is a **scoped stability-sector implementation** of IRER — and, beyond
> a mere scaffold, it is **faithful to the theory's explicit stability concepts** (resonant inertia = resistance,
> informational half-life = decay, FMIA = stabilization), which Phase C empirically confirmed. **(3)** IRER's
> **matter/transport sector remains under-specified and unimplemented**: the transport-implying concepts are
> aspirational, and the theory does not pin a kinetic formalism. Any Phase D transport work would therefore be a
> **deliberate formalism *choice and justification*, not the recovery of an already-specified operator.**

This is **not** "evidence is mixed" (option 4) — the evidence is cleanly **split by sector**, not contradictory.

## Parking-Lot (Stage-3 RFC pointers only — NOT proposed here)
Concepts that a *future* capability RFC would have to formalize a kinetic choice for (recorded, not designed):
transport/propagation (Q3, Q5-propagation), conserved circulation / protected spin (Q5-topological), true
non-local coupling (Q6), and the time-order decision (Q1). Each is a Stage-3 `CAPABILITY_EXPANSION_RFC.md` item,
gated on Stages 1–2, and must state its risk to the validated dissipative baseline. **No operator is proposed in
this audit.**
