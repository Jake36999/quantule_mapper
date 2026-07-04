# Phase D.5 — Two-Node Dynamics: the measured node-node interaction law

**Classification: `TWO_NODE_MERGE_OR_HOLD`** — a measured dynamical interaction law: pairs closer than a **merge
threshold ≈ 0.3 box** coalesce; pairs at 0.3–0.5 box **hold their separation exactly** (site-pinned, no drift); no
repulsion, no gradual attraction. `jax_scout/two_node_dynamics.py`, **pure Phase C physics** (`D_imag=0` = frozen
baseline); this is an IC + measurement experiment, no solver/gate/physics change.

## Setup
Two a\*-profile Gaussian nodes (width L/12, per-blob-fixed) at controlled initial separation, evolved at the
validated a\* params (feb×1.15) for T=12000, tracking node count, separation d(t), phase-difference, and total mass.

| initial sep (box) | outcome | n_fin | sep 0→fin | mass ratio |
|---|---|---|---|---|
| 0.15 | **MERGE** | 1 | 0→0 | 1.65 |
| 0.25 | **MERGE** | 1 | 0→0 | 2.15 |
| 0.30* | **HOLD** | 2 | 0.30→0.30 | 2.04 |
| 0.35 | **HOLD** | 2 | 0.349→0.348 | 2.03 |
| 0.50 | **HOLD** | 2 | 0.500→0.500 | 2.03 |

*the `d0=0.70` run wrapped to a 0.30 minimal-image separation (periodic box caps separation at 0.5 box).

## The law
- **Merge threshold ≈ 0.3 box.** Pairs closer than ~0.3 coalesce into a single node (their L/12 profiles overlap
  enough to merge). Pairs at 0.3–0.5 stay two nodes.
- **HOLD = site-pinned, no drift.** In the non-merging regime the separation is **fixed to ~0.001 box** over T=12000
  — the nodes neither drift together nor apart. There is **no gradual attraction/repulsion, no orbiting**.
- **Phase-locked throughout** (Δφ≈0), consistent with the global phase-coherence of stable configs (D.4).

## Combined picture (D.4 static + D.5 dynamic)
```
sep < ~0.3 box      : MERGE (coalesce to one node)
0.3 – ~0.5 box      : coupled (density bridge, D.4) but HOLD (pinned, no motion)
> ~0.5 box          : isolated (no static coupling)
all regimes         : no free advective drift
```

## Interpretation
This **reinforces the transport null at the pair level.** Even *coupled* nodes do not *move* toward each other — the
substrate has no advective channel (the same structural reason a\* is site-pinned, C1). Node "interaction" is
therefore **merge-or-hold**, set by profile overlap, **not** a relational drift/attraction force. So IRER stable
nodes *couple* (static density bridge + a merge radius) but do **not** exhibit matter-like relational *motion* —
the non-advective substrate is fundamental at both the single-node (C1) and pair (D.5) level.

**Caveats.** The two-blob configs grew ~2× in mass (two isolated nodes are not exactly at the a\* balance, which was
tuned for the 4–6-node config — itself weak evidence that a\* stability is partly *cooperative*); the merge-vs-hold
law is overlap-driven and robust to this, but a cleaner amplitude/force law would tune param_a for the 2-node
balance and vary the phase relation. No repulsion regime was found in the tested range.

## For D.6 (zoom-out / reduced model)
Concrete node-interaction rules: a node is a **pinned point** with a **merge radius (~0.3)** and a **connectivity
radius (~0.5)**; within-connectivity coupling is a static density bridge; **no advective drift term**. A macro model
is therefore a **pinned/merging network**, not a mobile many-body gas — the relational-transport degree of freedom is
absent (a finding, not a modelling choice). No matter-like claims; read-only physics, frozen baseline intact.
