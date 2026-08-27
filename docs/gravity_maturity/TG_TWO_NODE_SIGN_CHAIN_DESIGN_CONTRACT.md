# TG Two-Node Force — Sign-Chain Design Contract (SIGN-1)

Author: Claude (primary), 2026-07-15. Design-only; no runs, no change to the frozen TG-B1S model, production closed.
Resolves the standing gate `TG_TWO_NODE_SIGN_CHAIN_DECISION_REQUIRED` into a single decision for Jake, then defines
the experiment that decision unblocks. Cross-ref: `TG_B1S_EQUATION_AUDIT_ADDENDUM_20260715.md` (finding 2),
`TG_B1S_FREQUENCY_CONTRACT_RESULTS.md`, `GRAVITY_D_EFFECTIVE_MEDIUM_DYNAMICS_CHARACTERIZATION.md` (the force law).

## 0. The question this unblocks

Does the temporal-geometric feedback loop produce an **inter-node force** — and is it **attraction** (the gravity-like
direction) or **repulsion**? This is the first genuinely gravity-relevant test of the loop; the D3/D4 line only tests
a single node's self-frequency shift. The force's *direction* is fixed by one sign convention in the chain, which must
be decided **before** running, and decided **by the theory**, not tuned to a desired outcome.

## 1. The force law (rigorous, already verified)

Second node = a wavepacket in the KG substrate `π_t = c²∇·(A∇φ) − m²φ + g(ρ)φ`, `A = exp(−ε_G·G)`. The Gravity-D
work established and force-contract-verified (residual 3.6e-17) the momentum law:

```
d⟨P⟩/dt = −c² ∫ (∇A) |∇φ|² dV        (gradient-energy-weighted)
```

**Direction: a node is pushed toward *smaller* A.** So:

- **A-well** at a node (A < 1 at the node, → 1 far) → a neighbor is pulled *in* → **ATTRACTION**.
- **A-hill** (A > 1 at the node) → neighbor pushed *out* → **REPULSION**.

(In the two-node case both nodes source and respond; if each sits in the A-well the other creates, they attract.)

## 2. What the current frozen chain does → A-hill → repulsion

Trace the frozen signs (defaults α_T=0.35, κ=0.55, ε_G=0.06, all positive):

```
S_state ≥ 0  →(+α_T)  T > 0 near node  →(−κ coupling)  G ≈ −κT/ω_G² < 0  →  A = exp(−ε_G G) > 1  →  A-HILL
```

Quantified (FC-1 / box-dependence): `G_node ≈ −2.9e-4`, `A−1 ≈ +4.8e-5` at the core. **The frozen model predicts
inter-node REPULSION.** Consistently, its self-frequency shift is *positive* (+2.15e-6): the loop makes the node's
modal clock run **faster**.

## 3. The theory tension Jake must rule on (the crux)

IRER's own throttling logic (H-G−, the C-series): **dense interaction → higher resolution cost → slower
completed-resolution rate → the node ticks *slower* → lower local lapse.** The spatial analogue of "lower lapse /
slower propagation at the node" is an **A-well** (A < 1 at the node) → **attraction** — the GR-consistent direction,
and the direction of your original intuition (denser load → geometric well → attraction).

**But the frozen TG-B1S implements the opposite: an A-hill, node ticks faster, repulsion.**

So the SIGN-1 decision is not a free parameter sweep — it is a question about whether the frozen implementation's
sign matches the theory:

> **Decision for Jake:** In IRER, does a dense node make the surrounding medium propagate **slower** (A-well, lower
> lapse near the load, → attraction — matching the throttling logic and GR) or **faster** (A-hill, → repulsion, as
> currently frozen)? State the theory-mandated sign and the reasoning.

My read of the project's own throttling model says **A-well / attraction**, which means the two-node experiment
should use a chain that produces `G_node > 0` (A-well) — i.e. **one sign flip** relative to the frozen model (either
the `T→G` coupling `κ → −κ`, or the `G→A` exponent `exp(−ε_G G) → exp(+ε_G G)`). But this is a **theory ruling, not
my call**; if IRER genuinely predicts the A-hill, we test repulsion and report it. The frozen model was validated for
its bounded-shift *property* (D3/D4), not for having the throttling-consistent sign — those are separate.

## 4. The sign knobs (for whichever direction the theory picks)

| knob | frozen | flip effect on A at node | two-node consequence |
|---|---|---|---|
| `α_T` (S→T) sign | + | flips T sign → flips G → flips A | reverses force |
| `κ` (T↔G) sign | + (−κ in eqns) | flips G sign → flips A | reverses force |
| `G→A` exponent sign | `exp(−ε_G G)` | flips A directly | reverses force |

An **odd** number of flips relative to frozen → A-well (attraction); **even** → A-hill (repulsion). The theory picks
the *target* (well or hill); we then choose the *minimal, documented* sign change that realizes it (recommended: the
`G→A` exponent, since it is the most direct and leaves the validated T/G sector untouched). The chosen convention is
frozen and recorded before any run.

## 5. The experiment (once the sign is set)

- **Setup:** two Q-balls (reuse the C3 two-Q-ball machinery), half-separation `q`, relative phase `Δφ`, in a box that
  contains both screened clouds *and* the separation.
- **Isolation:** evolve **full-loop** and **feedback-off** at the *same* geometry; the loop-induced force is
  `F_TG = F_full − F_off`. This subtracts the **bare KS/Gordon phase force** the nodes already have (the π/2-crossover
  force from C2.9/C3), which is the dominant confound.
- **Observable:** inter-node force via the validated **momentum-density observable**; sign = inward (attract) / outward
  (repel). Measure `F_TG(q)` at fixed `Δφ` — ideally at `Δφ = π/2` where the bare force ≈ 0 — and also as
  `F_full − F_off` at other phases.
- **Range:** report `F_TG(q)` vs separation; expect **short-range** (Yukawa, screening length ~0.8–1.7 from SCREEN-1),
  *not* 1/r².

## 6. Preregistered gates & controls

- **ATTRACTION** supported iff: `F_TG` points inward at all tested `q`; magnitude decays with `q`; **vanishes at
  feedback-off**; **reverses** under the `G→A` sign control; survives dt/grid/box refinement.
- **REPULSION**: `F_TG` outward, same controls.
- **NULL**: `|F_TG|` below the `F_full − F_off` subtraction noise floor.
- **Kill the confounds:** bare phase force (off-subtraction + Δφ=π/2), capture/merger (stay at separations where nodes
  remain distinct — the C3 bounce/capture map bounds this), box-comparability (fixed adequate geometry; the D4
  larger-box lesson: hold geometry fixed and contain the clouds), single-node control (no inter-node force),
  phase-scramble control.

## 7. Boundaries (binding)

- A positive result = "the TG loop produces a **short-range inter-node force** of sign X in the mirror" — **not**
  gravity, not 1/r², not universal free fall, not IRER-validated. Long-range (ω_G→0) is a separate fork (LR-1).
- Mirror-only; production geometry/Hunter/solvers closed; no master-verdict change.
- We test whichever sign the theory dictates and **report the measured direction**, including repulsion or null. No
  tuning the sign to manufacture attraction (the retracted "flip a<0" over-claim is the anti-pattern to avoid).

## 8. The blocking decision (single item for Jake)

**State the theory-mandated sign of the state-load → geometry → propagation coupling: A-well (dense node → slower
propagation → attraction, matching IRER throttling + GR) or A-hill (as currently frozen → repulsion), with IRER
reasoning.** That one ruling sets the experiment; everything in §5–6 follows mechanically.

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `25020bf` (2026-07-22) — *dual substrate*
**Revised since:** 3 commit(s), most recently `9517d9f` (2026-08-26)

**Harness code changed since it was written:** 4 commit(s) to `jax_scout/`.
  - `06edae5` 2026-08-27 — Fix run provenance: 30% -> 86% of runs now dated
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
