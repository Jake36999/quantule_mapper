# Current Status Notice: Superseded By GPU Characterization

This report is retained as the historical first positive mirror result. Its original language about "universal", "geodesic-consistent", or "gravity-like" attraction is now superseded by:

- `docs/GRAVITY_D_EFFECTIVE_MEDIUM_CODEX_REPLICATION.md`
- `docs/GRAVITY_D_EFFECTIVE_MEDIUM_ROBUSTNESS_FIRST_PASS.md`
- `docs/GRAVITY_D_EFFECTIVE_MEDIUM_ROBUSTNESS_CLOSURE.md`
- `docs/GRAVITY_D_EFFECTIVE_MEDIUM_DYNAMICS_CHARACTERIZATION.md`

Current bounded interpretation:

```text
i d_t psi = -D div(N(x) grad psi)
d<P>/dt = -D integral grad(N) |grad psi|^2 dV
F_cg ~= -D K_grad grad N(R)
```

The mechanism is a robust spatial effective-medium wave force, not Newtonian gravity, not relativistic geodesic validation, not universal free fall, not temporal-lapse validation, and not IRER gravity-source confirmation. The production gravity ladder remains closed.

---

ARCHIVED HISTORICAL TEXT BELOW - NOT CURRENT INTERPRETATION.

# IRER Gravity Audit D (Option 1) â€” Neutral-Probe Force under an Env-Only Bounded Lapse

**Verdict:** `D_PROBE_ATTRACTION_PROMISING` â€” an **environment-only** objective bounded lapse
`N_B(x)=1/(1+Î²Â·Åœ_B(x))` (here `Åœ_B = Ï_BÂ²`, minimal at the load, â†’1 far) produces **universal, accelerating
attraction** of a neutral probe toward the load. This is the gravity-like signature the mutual-`I_int`/B-R path
could not produce (there, weak probes felt nothing). **Not** an emergent-gravity claim: it is a mechanism test in the
standalone mirror; the production `Î©Â²(Ï)` ladder stays CLOSED (this uses the NEW bounded env-only lapse).

## Setup

Load = compact Gaussian at origin; env-only source `Åœ_B=Ï_BÂ²`; lapse `N_B=1/(1+Î²Â·Åœ_B) âˆˆ (0,1]` (smallest at the
load = time runs slower there). Neutral probe = Gaussian wavepacket at rest at x=4, evolved under the
norm-conserving covariant Laplacian `iâˆ‚_tÏˆ = âˆ’Dâˆ‡Â·(N_Bâˆ‡Ïˆ)`; measure COM drift. `jax_scout/gravity_D_neutral_probe.py`.

## Results (N=64, L=30, Î²=1)

| run | drift_x | reading |
|---|---|---|
| **main (Î²=+1)** | **âˆ’1.25e-2** | **toward the load** |
| null (Î²=0) | âˆ’1.7e-4 | clean ~0 (free wavepacket conserves COM in the big box) |
| neg Î² (âˆ’0.5) | +9.7e-3 | away â€” **drift reverses sign with Î²** |
| wide probe (Î²=+1) | âˆ’6.3e-3 | toward the load â€” **universal across probe width** |

Baseline-subtracted (main âˆ’ null) COM trajectory is **accelerating inward** â€” interval steps grow monotonically
(âˆ’0.0004 â†’ âˆ’0.0040), a *sustained force*, not a settling transient. The measured attraction **agrees with the
timelike-geodesic prediction** `a=âˆ’âˆ‡ln N_B` (which pointed toward the load, âˆ’1.37e-3). Mass conserved (1.0000).

## What this resolves

- **The sign question that started the thread.** My original claim ("`Î©Â²(Ï)` is reversed, flip a<0") was wrong, and
  my D_eff heuristic (predicting repulsion) was the wrong argument â€” the **geodesic argument was right**: a bounded
  lapse with `N` *smaller at the load* attracts (clocks slow in the well, probes fall in â€” the GR-consistent
  direction). The *intuition* (denser load â†’ geometric well â†’ attraction) was directionally right; the specific
  mechanism (mutual `I_int`) and the power-law conformal form were not.
- **The cliff.** The production `Î©Â²=(Ï_vac/Ï)^a` diverges at the vacuum; the bounded throttling form
  `1/(1+Î²Åœ)` cannot cliff â€” and it gives graded attraction. So the geometry-source problem, *in the mirror*, is
  addressed by (a) an **env-only objective source** (universality) and (b) a **bounded lapse** (no cliff).

## Honest caveats (sizing the claim to the evidence)

- **Mirror mechanism test, not gravity.** One coupling form (`âˆ’Dâˆ‡Â·(Nâˆ‡)`, an acoustic-metric / variable-index
  reading), one candidate env-only source (`Ï_BÂ²`), modest resolution (N=64). "Promising," not "confirmed."
- **Not yet convergence-checked** (grid/dt refinement) â€” required before the "accelerating" and magnitude claims harden.
- **Coupling-form dependence:** whether `âˆ’Dâˆ‡Â·(Nâˆ‡)` is the *correct* IRER lapse coupling (vs a genuine spacetime
  lapse, or another operator) is a modeling assumption, not established.
- **Source-form dependence:** `Ï_BÂ²` is a candidate; other env-only interaction-rate fields should give the same
  qualitative attraction if the effect is robust.

## What would upgrade it (next)

1. **Convergence** under N and dt refinement (does the accelerating attraction survive?).
2. **Source robustness:** repeat with other env-only sources (`Ï_B`, the load's self-interaction energy density) â€”
   the sign/attraction should be source-form-independent.
3. **Distance law:** vary the probe's initial distance â†’ is the acceleration graded with distance (a falloff)?
4. **Coupling-form audit:** compare `âˆ’Dâˆ‡Â·(Nâˆ‡)` to a lapse-in-time-component implementation; do both attract?
5. Only much later, and separately, the production re-entry gate (bounded source replacing the cliffing `Î©Â²(Ï)`).

## Verdict discipline (binding)

`D_PROBE_ATTRACTION_PROMISING` supports: *an env-only bounded lapse yields universal, Î²-responsive, geodesic-
consistent, accelerating probe attraction in the mirror.* It does **not** establish emergent gravity, physical
time dilation, or that this is IRER's mechanism in nature. Production gravity ladder remains CLOSED.

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `25020bf` (2026-07-22) — *dual substrate*
**Revised since:** 8 commit(s), most recently `bc5b54c` (2026-08-31)

**Harness code changed since it was written:** 5 commit(s) to `jax_scout/`.
  - `06edae5` 2026-08-27 — Fix run provenance: 30% -> 86% of runs now dated
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine
  - `25020bf` 2026-07-22 — dual substrate

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
