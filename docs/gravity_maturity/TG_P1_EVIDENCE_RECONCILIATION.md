# TG P1 — Evidence Reconciliation & `F_R` Classification

Author: Claude (primary), 2026-08-25. Campaign item **P1** of *TG dual-substrate — factor apart &
close contracts* ([[TG_RECOVERED_CONCEPTS_INTEGRATION_AND_REPRIORITIZATION]], catalog §11).

Desk audit — **no new simulation was run.** Sources: `jax_scout/gravity_TG_B2_definitive_force.py`,
`gravity_TG_B2_two_node_awell.py`, `gravity_TG_B1S_state_load_feedback_gpu.py`,
`gravity_TG_B2_charge_audit.py`, `phase_d_c3_wave.py`, and the run rows in
`sweep_runs/TG_B2_CHARGE_AUDIT_20260719_005326/` (catalogued at [[../runs/TG_B2_CHARGE_AUDIT_20260719_005326]]).

Mirror-only. **No gravity / UFF / IRER claim.** This document *reduces* the strength of several earlier
framings; that is what P1 was for.

---

## Verdict

```text
TG_P1_MASS_AXIS_CONFOUNDED__FORCE_EXPONENTS_NOT_INTERPRETABLE_AS_MASS_SCALING
```

Three independent problems make the existing mass-scaling evidence uninterpretable *as mass scaling*.
None of them is a bug — each is a defensible design choice whose consequence was never propagated into
how the results were read.

---

## 1. What `F_R` actually is (the open P1 question — now answered)

From `gravity_TG_B2_definitive_force.py:60`:

```python
F_R = -c*c * jnp.sum(jnp.where(X > 0, dxA * grad2, 0.0)) * dV
```

i.e.

$$F_R = -c^2 \int_{x>0} (\partial_x A)\, |\nabla\phi|^2 \, dV , \qquad A = e^{\,a_{\text{sign}}\,\varepsilon_G G}$$

| question | answer |
|---|---|
| total force, density integral, average, or acceleration? | **A total body force** in code units. Not an acceleration, not mass-normalized, not a per-unit-volume density. `<F_R>` is a *time-average of* that force over the settled window. |
| force on *what*? | **Everything in the half-space `x>0`** — not the right node's support. Tails, radiation and the absorber region are all inside the integration domain. |
| weighted by which density? | **The gradient energy density only**, `c²∣∇φ∣²`. |

**That last row is the load-bearing one.** The full KG energy density in this codebase
(`phase_d_c3_wave.py:80`) is

```python
E = float(np.sum(np.abs(pi)**2 + c**2 * grad2 + m**2 * rho - G) * dV)
```

so `F_R` weights **one of four terms**. It is not the force on the node's energy; it is the force on the
node's *gradient* content. The gradient fraction of a Q-ball's energy depends on its profile, so this
choice couples the observable to node morphology.

**Consequence for `F/M_p`.** The available mass observable (`gravity_TG_B2_two_node_awell.py:105`) is

```python
mp = jnp.sum(jnp.where(X > 0, rho, 0.0)) * dV        # rho = |phi|^2
```

a **charge-like density integral**, not an energy. So `F/M_p` divides a gradient-energy-weighted force by
a `∫|φ|²` integral. **That ratio is not an acceleration**, and no UFF-style statement can rest on it.
The definitive-force harness records `F_R, P_R, charge, sep, amp` — **no energy observable at all**.

> The charge audit's own classification note already said "a TOTAL (half-space) body FORCE, NOT an
> acceleration and NOT mass-normalized". That was correct and is confirmed here. What it did not record
> is the gradient-only weighting and the energy/charge mismatch in the denominator.

---

## 2. `e_ref` / `q_ref` audit (fixed vs recomputed)

The source is built in `gravity_TG_B1S_state_load_feedback_gpu.py:252-260`:

```python
e_ref, q_ref, S0 = refs
raw = 0.5 * jnp.maximum(energy, 0.0) / (e_ref + 1e-12)
raw = raw + 0.5 * charge / (q_ref + 1e-12)
```

with, per `gravity_TG_B2_charge_audit.py:60`:

```python
e_ref = float(np.max(np.abs(E_dens)));  q_ref = float(np.max(charge_dens))
```

**Finding: `e_ref` and `q_ref` are peak (max) normalizers, recomputed per-`w` — i.e. per mass row.**
`S0` alone is frozen. So each mass row is normalized by *its own peak*, and mass dependence is
partially divided out by construction:

| quantity | exponent vs M | reading |
|---|---:|---|
| `E_tot` (raw extensive load) | **0.935** | raw load is ~linear in M, as expected |
| `Q` (charge) | 0.966 | ~linear |
| `e_ref` | **0.361** | the normalizer itself grows with M |
| `q_ref` | **0.369** | ditto |
| `∫S_state` (normalized source) | **0.585** | ≈ 0.935 − 0.36 — the peak normalization removes ~0.36 |
| `∫∣∇φ∣²` (receiver) | 1.099 | receiver scales slightly superlinearly |

**~0.36 of the source's mass exponent is removed by the normalization choice, not by physics.**
This is design-as-intended (the source is meant to be a shape-normalized state load, per the frozen
`S0` contract) — but it means `∫S_state` is *not* an extensive load, and any force exponent measured
against `M` inherits that offset.

---

## 3. The mass axis is not a clean mass axis

Mass is varied by sweeping the Q-ball frequency `w` (0.945 → 0.980, giving M = 105.9 → 38.7, a 2.74×
range). Changing `w` changes the node's **shape** as well as its mass:

| quantity | ordered by ascending M | monotonic in M? |
|---|---|---|
| `source_rms_width` | 3.551, 3.081, 2.892, 2.850, 2.961 | **NO** — 25% variation, minimum in the interior |
| `∫S_state` | 0.389 … 0.743 | **NO** |
| `well_integral` | 0.00617, 0.00544, 0.00582, 0.00718, 0.01039 | **NO** — dips then rises |
| `E_tot`, `Q`, `e_ref`, `q_ref`, `∫∣∇φ∣²`, `amp` | — | yes |

**The lightest node has the widest source.** Source width and mass move partly *against* each other, and
three of the source-side quantities turn over inside the sampled range. A power law fitted across these
rows is fitting mass **and** morphology simultaneously; it does not isolate a mass dependence.

---

## 4. The reported force exponent is not a power law

`F_measured ~ M^-0.067` has been carried as the near-field mass-independence result. Refitting the five
rows in log-log:

```text
slope = -0.0673      R^2 = 0.111
```

**R² = 0.11 over five points.** The fit explains 11% of the variance — this is flat scatter, not a power
law with a small exponent. `|F_R|` spans 5.00e-05 … 6.11e-05, a 1.22× ratio across a 2.74× mass range.

**The defensible statement is:** *`F_R` is constant to within about ±10% across a 2.74× mass range, on a
mass axis that also varies source width by 25%.* Not "F ∝ M^0", and not "F ∝ M^-0.067".

The same caution applies to the far-field result (`F ~ M^+1.42` at sep 8,
[[TG_B2_FARFIELD_KERNEL_RESULTS]]): it is computed on **the same confounded mass axis**. That result
already carried a caveat (all sampled separations have negative surface gap, so it is not yet a true far
field); this is a second, independent limitation on it. **The near-field/far-field exponent contrast
stands as a real separation-dependence of the kernel, but neither number should be quoted as a mass
exponent.**

---

## 5. Scale check — the effect lives in the fifth decimal of `A`

| quantity | value (across all mass rows) |
|---|---|
| `G_min` | −1.11e-03 … −7.55e-04 |
| `ε_G` | 0.06 |
| **`A_well_min`** | **0.999934 … 0.999955** |
| `F_R` | ≈ −5e-05 |

The A-well is **7 parts in 100,000** deep. `ε_G·G ~ 6.6e-5`, so `A = exp(εG) = 1 + εG + O(ε²G²)` with the
quadratic term at ~4e-9 absolute — the system is **strictly in linear response**.

**This requires me to qualify something I said earlier in this session.** I reported the N=80/L=20
convergence antisymmetry (`|well+hill|` = 2.13e-05 relative) as showing "the response is linear in the
A-coupling to five digits, so it is genuinely the coupling and not noise." The arithmetic is right and it
remains a clean instrument check — but near-perfect antisymmetry is *what linear response guarantees* at
`εG ~ 7e-5`; the even-order residual should be O(εG) relative, ≈ 6.6e-5, and we measured 2.1e-5. The
antisymmetry confirms the estimator is arithmetically sound. **It carries essentially no information
about the physics**, and I over-read it as evidence that the coupling was doing something special.

The convergence result itself is unaffected: the force is still not a discretization or box artifact.

---

## 6. What this does and does not change

**Unchanged:**
- The A-well attraction is real within the model and converged (3.15% over N=64→80, L=16→20).
- The `off` (A=1) null is exact.
- The sign control (well/hill reversal) works.
- The TG-B1S feed-forward chain and the "robust but weak" backreaction characterization.

**Downgraded:**
- Any statement of the form "force is mass-independent" → *"force is flat to ±10% on a confounded axis."*
- Any use of `F/M_p` as an acceleration → **not dimensionally coherent**; do not use for UFF reasoning.
- The far-field `M^1.42` as a *mass* exponent → it is a separation-contrast, on the same confounded axis.
- The convergence antisymmetry as evidence of anything beyond estimator soundness.

**Newly explained:**
- `SOURCE_SCALES_BUT_FORCE_FLAT__RECEIVER_OR_OBSERVABLE_NORMALIZATION` (the July charge-audit verdict)
  resolves toward **observable normalization**: peak-normalized source (−0.36 in exponent) plus
  gradient-only receiver weighting plus a non-monotonic morphology axis are jointly sufficient to
  produce a flat force without invoking any receiver physics.

---

## 7. Recommended P1 closures (design changes, no new physics)

| id | change | why |
|---|---|---|
| **P1-a** | Add a per-half-space **total energy** observable to the B2 diag; report `F/E_R` alongside `F/M_R`. | The only way to form a quantity with acceleration-like dimensions. Currently no energy observable exists in the definitive-force harness. |
| **P1-b** | Run one mass sweep with **frozen** `e_ref`/`q_ref` (fixed at a single reference node) beside the per-`w` recomputation. | Separates the −0.36 normalization offset from any physical source scaling. This is the single highest-value cheap run. |
| **P1-c** | Report `source_rms_width` as a covariate; fit `F` against `(∫S_state, width)` jointly, not against `M`. | The mass axis is confounded; a two-covariate fit is honest where a power law is not. |
| **P1-d** | Add a **node-support-limited** force integral beside the half-space one. | Half-space includes tails, radiation and absorber; a support-limited integral is what "force on the node" means. |
| **P1-e** | Retire single-exponent power-law quotes on 5-point sweeps. Quote ratio + range + R², or widen the sweep. | R² = 0.11 was being reported as an exponent. |
| **P1-f** | Consider weighting `F_R` by the **full** energy density, as a second estimator. | Tests whether gradient-only weighting is driving the flatness. Cheap: post-processing on stored fields. |

P1-b and P1-f are the two that could change a verdict; P1-a, P1-c, P1-d, P1-e are reporting hygiene.

---

## 8. Status against the P1 checklist

| P1 item | status |
|---|---|
| audit fixed-vs-recomputed `e_ref`/`q_ref` | **DONE** (§2) — recomputed per-`w`, peak normalizers, ~0.36 exponent offset |
| classify what `F_R_well` actually is | **DONE** (§1) — half-space total body force on gradient energy; not an acceleration |
| expose per-row source integral, mediator amplitude/gradient, raw force | **DONE from existing rows** (§2–§5) — the charge-audit run already carries these |
| per-row probe susceptibility, `F/M_p`, breathing amplitude, uncertainty | **BLOCKED** — no energy observable and no per-row uncertainty is recorded; needs P1-a |

**P1 is substantially closed on the audit questions and blocked on the instrumentation questions.**
The blocking item is small: P1-a adds one observable.

---

## Next

P1-b (frozen-reference mass sweep) is the highest-value follow-up and is a short run. **P2's midplane
stress-flux force remains the strongest outstanding instrument check** and is now more clearly motivated:
it would be an independent estimator that does *not* inherit the gradient-only weighting analysed in §1.

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `292fc96` (2026-08-25) — *Obsidian vault: run catalogue, experiment tracker, documentation methodology*
**Revised since:** 2 commit(s), most recently `3eb93af` (2026-08-26)

**Harness code changed since it was written:** 3 commit(s) to `jax_scout/`.
  - `b4e0613` 2026-08-25 — P1-a: per-half-space energy observable closes the P1 checklist
  - `5a19e60` 2026-08-25 — P2 CONFIRMED: midplane stress-flux estimator validates the body force 
  - `eeb8f6d` 2026-08-25 — P2: midplane stress-flux estimator, stability review checklist, refine

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Also referenced by (same date or earlier):** [[IRER_MASTER_HYPOTHESIS_CATALOG]], [[Main branch]], [[RUN_QUEUE]], [[SYSTEM_PRESSURE_TEST_AND_METHOD_REVIEW]], [[gravity_maturity/TG_P1A_ENERGY_OBSERVABLE]], [[gravity_maturity/TG_P2_MIDPLANE_STRESS_FLUX_RESULTS]]

**The master catalog references this document** — the catalog is the authority on whether its verdict is still live:

> | **TG P1 — evidence reconciliation** | **CLOSED on the audit questions (2026-08-25); BLOCKED on instrumentation** | `docs/gravity_maturity/TG_P1_EVIDENCE_RECONCILIATION.md`. Verdict `TG_P1_MASS_AXIS_CONFOUNDED__FORCE_EXPONENTS_NOT_INTERPRETABLE_AS_MASS_SCALING`. **(a)** `F_R` classified: a **total …

<!-- LINEAGE:END -->
