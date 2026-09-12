# IRER Gravity Audit B — Metric / Lapse Design (H-G− throttling)

**Status:** DESIGN-ONLY. Standalone mirror; **production gravity ladder stays firmly closed** (the `Ω²(ρ)`
saturation-cliff / de-saturation / flat-background re-entry gate is a *separate*, still-open condition — Audit B does
**not** reopen it). Scope: **A-P closed** (probe–environment interaction-load validated); **A-S open** (single-field
self-sourcing not established). No matter/gravity claim.

**Purpose.** Specify the lapse `N` from the decided **H-G− throttling** mechanism — *independently of any clock
result* — with a β-sweep, an overlap-sourced control, and explicit falsification, so that Audit C (the clock
experiment) is non-tautological. Audit B is a **design + a small algebra check**; the clock effect is measured in C.

---

## 1. Inputs (from the closed source audit A / A.1 / A.2 / A.2b)

- `I_int` — the validated interaction-**load** observable (deterministic, decomposition-invariant within an explicit
  target–environment model, density-independent, predictive; overlap-family, edges plain overlap).
- Local field `I_int(x)` (the integrand), which a lapse needs (not a global norm).
- Force law (validated, Ehrenfest, A.2b): `dP/dt = −κ∫ρ_A∇ρ_B` — uses **`∇ρ_B`** (coupling-potential gradient),
  **not `∇I_int`**.
- Mechanism (decided): **H-G−**, `R_res = K/(1+β·I_int)`, `β>0` (finite resolution capacity).

## 2. The lapse construction

```
Î_int(x) = I_int(x) / I_0            # dimensionless load (I_0 = fixed reference scale, not per-run-fitted)
N(x)     = R_res/R_ref = 1 / (1 + β·Î_int(x))        # H-G− throttling
```
- **Background flat:** where `I_int→0` (isolated / vacuum), `N→1`. No soft-clip needed — `1/(1+βÎ)` is naturally
  bounded in `(0,1]` for `Î≥0`, so **no saturation cliff** by construction (contrast the production `Ω²(ρ)`).
- **Sign fixed, strength swept:** only `dN/dÎ < 0` is set by the mechanism. `β` is a **controlled sweep**
  `β ∈ {0, β₁, β₂, β₃}`; **`β=0` is the essential null** (`N≡1`, no effect). `β` is NOT fitted to the clock.
- **Metric slot:** this is a **lapse** `N` (temporal). A pure-conformal `N=A` leaves null geodesics invariant (no
  lensing), so a *clock* (time-dilation) effect requires `N` specifically — which is what we build. Spatial `A` is
  deferred; the NLS mirror tests the **lapse algebra** (does a probe whose local time is `N·dt` tick slower in
  high-`I_int` regions?), not spatial curvature.

## 3. The overlap control (decisive — because `I_int` is overlap-family)

Run the identical lapse construction with the source replaced by a **plain relational overlap**:
```
N_overlap(x) = 1 / (1 + β·Ô(x)),    Ô = overlap field (e.g. local ρ_A·ρ_B / O_0)
```
- If the throttling/clock effect is **reproduced** by `N_overlap` → `I_int` is *not* earning its keep (it's a fancy
  overlap); the source claim collapses to "density/overlap-sourced," which is the very thing the audit set out to
  move beyond.
- If `I_int` yields a **distinct** effect (different magnitude/spatial pattern that overlap cannot match) → the load
  observable is doing real work. This control is **mandatory**, reported alongside every B/C result.

## 4. Scaling-law variants (open hypotheses — do not pre-select)

The "resolution cost" law is a hypothesis to test, not settled:
```
C_res ∝ 1 + β·I           (linear;   N = 1/(1+βI))
C_res ∝ e^{β·I}           (exponential; N = e^{−βI})
C_res ∝ 1 + β·V_int       (interaction-VOLUME driven)
```
- The **"exponential lightcone growth"** intuition (Jake) is a *scaling* hypothesis about how relational constraints
  grow across the causally-available region. A genuine finite causal cone lives on the **KG substrate** — the NLS
  mirror can test the throttling *algebra* only, **not** relativistic causal-cone behaviour. Record accordingly.

## 5. Independence & falsification

- **Non-tautology:** `N` is fixed here from the mechanism + a β-sweep, *before* and *independent of* the clock (C).
  The clock oscillator in C couples **through** `N` (this dynamical law), not a post-hoc phase multiplier.
- **Falsified / negative (any one):** `β=0` and `β>0` are indistinguishable in C; the clock effect is **reproduced
  by the overlap control**; the effect is non-convergent under `N`/`dt` refinement; `N` is degenerate; or the "effect"
  is just the density-coupling potential re-expressed. A characterized negative is a valid outcome.

## 6. What Audit B delivers vs defers

- **Delivers:** the lapse spec above, the overlap-control spec, the β-sweep grid (incl. `β=0`), the falsification
  set, and a small **algebra check** — DONE (`jax_scout/gravity_B_lapse_precheck.py`, `B_LAPSE_PRECHECK_PASS`):
  β=0 → `N≡1` exactly; `N∈(0,1]` **naturally bounded, no cliff** (a structural advantage over production `Ω²(ρ)`);
  flat at background (`|N−1|≤2e-6` where load→0); and **distinguishable from the overlap control** — N(I_int) vs
  N(overlap) spatial correlation only ~0.66 (I_int weights `ρ_B²`), so the lapse is not degenerate with plain
  overlap at the design level (measurability in a clock is C's job).
- **Defers to C:** the actual clock experiment — identical deterministic probes in **equal-pointwise-ρ /
  unequal-`I_int`** regions, coupled through `N`, with the **clock-universality** check (two structurally different
  clocks must share the same fractional rate change `≈N`), plus the β-sweep and overlap control run dynamically.
- **Does not touch:** production solver / Hunter / config; the production gravity ladder (gate closed); any
  self-gravity or lightcone claim.

## 7. Next

Implement the static algebra pre-check (`N`, `N_overlap`, β-sweep, boundedness, flat-background), then Audit C
(clock experiment with universality + overlap control). Only after a reproducible, overlap-distinct, convergent
clock difference does a spatial-force rung (D) — and only then, far later, the production re-entry gate — come into
scope.

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `25020bf` (2026-07-22) — *dual substrate*
**Revised since:** 11 commit(s), most recently `bd93089` (2026-09-12)

**Harness code changed since it was written:** 6 commit(s) to `jax_scout/`.
  - `caf61af` 2026-09-10 — Build HUD items 1+2: snapshot writer and universal field renderer
  - `06edae5` 2026-08-27 — Fix run provenance: 30% -> 86% of runs now dated
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine
  - *…and 1 more.*

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
