# Phase D — Transport / Coupling Sector Plan (the scope pivot, made explicit)

**This is not a patch to Phase C. It is the planned transition of the *question*:**
> Phase C: *Can IRER-derived geometry-feedback PDEs generate stable field configurations?* → **YES** (closed).
> Phase D: *Can stable IRER-derived configurations **communicate, couple, move, and aggregate** into larger
> self-sustaining structures — with purely IRER-translated maths?*

## What Phase C established (the stability sector — CLOSED as baseline)
- Stable **dissipative** node configurations exist in the IRER geometry-feedback / self-referential PDE regime
  (a\*≈×1.15, gain/loss-balanced, seed/N128/T144k-confirmed); the geometric feedback loop makes stable nodes
  **from the vacuum**, as the theory predicts.
- The Hunter **re-finds** the a\* stability basin by objective **and** adaptive search (cross-IC, production path).
- We have explored **~4000** configs of a combinatorially vast space — Phase C proved the *mechanism*, not the
  *catalogue*.
- These are **dispersive standing nodes** that behave as the stability-sector theory predicts. Phase C is the
  **evidence base** Phase D builds on; its frozen operator (`e8d6a78ea`) is **not** to be altered.

## The governing principle for Phase D (learned from C1)
The C1 test (add a dispersive kinetic channel to a\*) → `C1_NO_STABLE_TRANSPORT`: it produced transient mobility but
**destabilised the a\* attractor** before stable transport emerged. The lesson is **not** "Phase D failed" — it is:
> **The Phase C attractor is not itself the transport object.** Phase D must *use* Phase C knowledge (data,
> dynamics, metrics) but is **not constrained to preserve a\* unchanged.** Do not keep trying to force a\* to move
> (except as one clearly-scoped diagnostic). The next question is not *"how do we make the node move?"* but
> **"what additional mathematical structure lets stable IRER nodes exchange momentum / phase / current / coupling /
> constraint information while remaining self-sustaining?"**

Success in Phase D = **measured coupling / motion / aggregation**, not preserving a\* unchanged. **No matter-like
claims**; motion/coupling is asserted only from reproducible metrics, never visuals.

## The next ideal simulation capability (defined, so we don't miss steps)
> A **transport-capable IRER branch** + a **node-interaction measurement system**, where stable/metastable nodes can
> exchange phase/current/coupling information, respond to gradients, and form larger bounded structures, with
> motion/coupling quantified by reproducible metrics.

The capability chain — this is the **"zoom out"** that lets us reach macro behaviour without simulating every macro
system at full vacuum-field resolution:
```
micro field dynamics  →  node descriptors  →  interaction / coupling metrics  →  reduced meso model  →  macro behaviour
```
You cannot scale to macro while every test is a full-resolution vacuum-field sim. First **learn what the nodes are**:
spacing, coupling radii, current structure, phase relations, failure modes, attraction/repulsion, and whether
multi-node arrangements have stable interaction *laws*.

## Roadmap (bounded steps; each names its guardrails)
| step | goal | output | status |
|---|---|---|---|
| **D.1** | kinetic-term transport (C1) | `PHASE_D_C1_RESULTS.md` | **done → `C1_NO_STABLE_TRANSPORT`** |
| **D.2** | **informational stress-tensor bridge** — measure how nodes strain/couple/communicate (READ-ONLY diagnostic) | `PHASE_D_STRESS_TENSOR_BRIDGE_PLAN.md` + `..._RESULTS.md` | **next** |
| **D.3** | **Phase C node-library expansion** — diverse stable node *families* (spacing, count, current/vorticity, seed-robustness, perturbation response) via targeted Hunter searches | `PHASE_C_NODE_LIBRARY` | planned |
| **D.4** | **coupling-law extraction** — infer node↔node interaction laws from stress / flux / corridor metrics | coupling-law report | planned |
| **D.5** | **transport branch** — C2 conservative/NLS (or a better IRER-translated transport regime): mobile coherent structures; two-node interaction/collision tests | transport-branch results | gated on D.2 signal |
| **D.6** | **reduced macro simulator ("zoom out")** — node = object (mass/phase/current/stability-class/coupling-radius), edges = measured interaction laws, macro sim at lower resolution | meso/macro simulator | the scale bridge |

**Why D.2 (stress tensor) before D.5 (C2):** C1 showed the gap is not "kinetic → motion" but the *measurement* of
directional interaction. A scalar ρ says *where* nodes are, not *how* they strain, shear, or couple. The stress
tensor is the candidate **coarse-graining variable** between "stable field nodes" and "larger communicating
structures" — the missing bridge that D.4/D.6 need. It is built **read-only first** (a measurement, not a force
term), so it cannot corrupt the solver or the validated baseline.

## Guardrails (Phase-D-wide, non-negotiable)
Hard stops (halt + report): a change would alter the **frozen Phase C default**; provenance/gate breaks; artificial
clipping is introduced to force stability; visual motion is promoted without metrics; claims exceed evidence. Every
Phase D branch is **default-off and separate** from Phase C, records its `kinetic_mode`/new coefficients in
provenance, and leaves Phase C reproducible as the baseline (as C1 did: `D_imag=0` = Phase C byte-for-byte).

## On the theory sources
Phase D formalises **Jake-authored IRER concepts** — Gradient-Derived Informational Forces, FMIA paths, Payan-state
alignment, Informational Manifold topology/angles, coupling equations, and the **Informational Stress-Energy
Tensor**. External blueprint documents (e.g. the "Causal Field of Affect / V13" doc) are **design ancestors**: we
extract candidate *forms* (the stress-tensor expression, the retarded / Field-of-Affect coupling concept) but **do
not** import their older/over-strong claims (prime-log & SPDC validation targets, BSSN/Hamiltonian framing,
production-rollout language, mandatory Ω clamping, active stress-force coupling, immediate `worker_cupy` refactor) —
those conflict with the hardened conclusions (prime-SSE retired as a certifier; stability metrics certify; no
clipping to force stability).
