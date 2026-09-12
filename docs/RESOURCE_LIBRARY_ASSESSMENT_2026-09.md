---
tags: [record, meta, plan]
date: 2026-09-11
status: complete
---

# Resource Library Assessment — what to take, what to decline, and in what order

Author: Claude, 2026-09-11. Review of the nine Quantule-Mapper application records in
`D:\Resource-Library\09-Applications\`, mapped onto this project's **actual open items** rather than
onto the library's own categories.

**Nothing here has been integrated.** This is an assessment and a plan.

---

## 1. Verdict on the research itself

**The quality is high, and unusually so in one specific respect: it records rejections with reasons.**

Most tool research returns a list of things that matched. These records name what was checked and
discarded — `SymPT` as the wrong branch (quantum many-body, not continuum soliton asymptotics),
`pyMultiobjective` as unlicensed, `openwave` as a precedent rather than a vetted donor, Aim and
Sacred as *found and explicitly not recommended* because DuckDB already covered the gap. That is the
same discipline this project applies to its own nulls, and it makes the records trustworthy in a way
a hit-list would not be.

I verified the load-bearing claims rather than taking them on trust:

| claim | verified? |
|---|---|
| Backend parity needs no new dependency | **YES** — `cupy.testing.assert_array_almost_equal_nulp` present (cupy 14.0.1); `jnp.allclose/isclose` present |
| PyVista renders raw numpy volumes, graphics not compute | **YES** — and **already installed** (0.47.1) |
| scikit-image gives a standard centroid finder | **YES** — **already installed** (0.26.0) |
| `sympy` "already catalogued" | **CAVEAT** — catalogued in the library, **not installed in this project** |

> [!warning] Catalogued is not installed
> The records use "already catalogued" to mean present in the resource library. For planning effort,
> what matters is presence in `.venv` or the WSL JAX env. Of the twelve tools checked, **two are
> already installed** and ten would need adding.

---

## 2. The one finding that is worth more than any tool

> [!important] The perturbation-theory negative result
> The symbolic-geometry record searched Hirota's bilinear method, reductive perturbation theory, and
> adiabatic soliton-parameter perturbation theory, and concluded **no usable open-source package
> exists** — with the near-misses named and individually rejected.
>
> That is the single most valuable item in the whole sprint, because **Phase 1a of the action plan is
> exactly that derivation**. Without this record I would have opened that search myself and spent a
> day confirming the same emptiness. The record converts an unknown into a closed question:
> **the hand derivation is the correct plan, not a fallback.**

A negative result that prevents a search is worth more than a tool that saves an afternoon. The
library should keep doing this.

The same record also supplies the method reference — Jakobsen 2013 on multiple scales — and names
`diffrax` as the numerical destination for whatever reduced ODE the derivation produces. That is a
complete, honest answer to a question the project had open.

---

## 3. Mapping onto the project's actual open items

Ordered by the [[ACTION_PLAN_2026-08|action plan]], not by resource category.

### Tier 1 — serves Phase 1, the sign problem (the #1 scientific priority)

| resource | open item | verdict |
|---|---|---|
| **perturbation-theory negative** | Phase 1a | **Take.** Closes the search. Derivation proceeds by hand. |
| **Jakobsen 2013 (multiple scales)** | Phase 1a | **Take.** The method reference for the derivation. |
| `sympy` | Phase 1a | **Install.** General-purpose CAS is the realistic scaffold. Not currently present. |
| `diffrax` | Phase 1a output | **Defer until there is an ODE to integrate.** JAX-native, shares the substrate. |
| `cadabra2` | Phase 1b (GAP-4) | **Preferred over EinsteinPy** — purpose-built for Lagrangians and equations of motion, which is the action-principle framing GAP-4 actually needs. |
| `EinsteinPy` | Phase 1b (GAP-4) | Second reading; better if approached from equations of motion. |

### Tier 2 — serves Phase 2, the discriminating quantity

| resource | open item | verdict |
|---|---|---|
| **`SALib`** | **Threat T2** | **Take — best single fit in the sprint.** See §4. |
| `PySR` + `SISSO` | Phase 1a corroboration | **Take later.** Two algorithmically distinct methods; agreement between them on a discovered force law would be a blind cross-check of the hand derivation. |
| `chaospy` | T2, smooth regions | Complementary to SALib; not yet. |
| `dynesty` / `UltraNest` | model comparison | **Premature.** Bayesian evidence needs a likelihood, and the project has no external data to form one against. Revisit after Phase 2 names a measurable. |

### Tier 3 — verification residuals (cheap, and two are already-paid-for)

| resource | open item | verdict |
|---|---|---|
| **`cupy.testing` + `jnp.allclose`** | open backend-parity residual | **Take now. Zero new dependencies.** Highest value-per-effort in the entire sprint. |
| **`mutmut`** | does the identity CI actually work? | **Take.** See §5 — this tests a claim I made. |
| `Hypothesis` | identity suite | **Take after mutmut.** Turns 11 hand-picked assertions into properties over generated configs. |
| MMS via `sympy` | dt-convergence residual | **Take the technique, not MASA.** sympy alone suffices. |
| `exponax` / `svirl` / `py-pde` | independent solver cross-check | **Strong but not now.** `svirl` is the sharpest — it solves the complex Ginzburg–Landau equation itself. |

### Tier 4 — the HUD items I left unbuilt

| resource | RFC item | verdict |
|---|---|---|
| **`scikit-image` `peak_local_max`** | **6.3 centroid overlay** | **Take now — already installed.** Replaces the bespoke peak-tracker whose class caused C2.8b. |
| **`PyVista`** | **6.4 live/interactive viewer** | **Take when a 3-D question demands it — already installed.** |
| `cplot` / `complexplorer` | *not in the RFC* | **Consider.** Domain colouring shows magnitude and phase as one image. The RFC never framed this as a question. |
| `pyqtgraph` | 6.4 live monitor | Lighter than Panel; desktop-native, no server. Right shape for the reader-only rule. |

### Tier 5 — infrastructure: mostly decline

| resource | verdict |
|---|---|
| `Orbax` | **Consider.** Addresses a real, observed failure (three WSL launch deaths). But the keep-alive + detached launch already fixed that, so the pain is historic. |
| `pixi` | **Consider later.** Directly addresses the numpy-ABI breakage and the two-venv footgun, lighter than the deferred Dockerfile. |
| `Hydra` / `dynaconf` | **Decline for now.** Targets a config-drift bug caught once. Real, but not a bottleneck. |
| `Aim` / `Sacred` | **Decline** — as the record itself recommends. |
| `psutil` + `Supervisor` | **Decline.** The record honestly concludes no single tool crosses the Windows↔WSL boundary. |

---

## 4. Why SALib is the best fit in the sprint

[[SYSTEM_PRESSURE_TEST_AND_METHOD_REVIEW|The pressure test]] named **T2 — 16 free parameters against
one external comparison** — as a *critical, unmitigated* threat. The mitigation proposed in the
infrastructure assessment (A3) was only to **record** degrees of freedom per run: a count.

SALib converts that count into a **measured ranking**, and it does so through a decoupled black-box
workflow that needs no gradients and no model internals — `sample()`, run the simulator exactly as
the project already does for a basin sweep, `analyze()`.

> [!important] The detail that makes it a direct hit
> Sobol indices capture **parameter interactions**, not just main effects. P1-a found that the mass
> axis varies node morphology by 29% non-monotonically — **that is an interaction**, and it is
> precisely what a one-parameter-at-a-time sweep cannot see. The project's entire current sweep
> methodology is one-at-a-time.

And it serves Phase 2 directly: knowing *which* of the 16 parameters actually drive an outcome tells
you where a discriminating quantity could possibly live, and which parameters a prediction would have
to be insensitive to in order to count as parameter-free.

**One caution the record raises and I would enforce:** SALib's sampling is less reliable near sharp
thresholds — and this project has one, the saturation cliff. Run it inside the confirmed a\* basin
first, where the response is smooth, and treat cliff-adjacent results as suspect.

---

## 5. The recommendation that tests my own work

Last session I built `tests/test_physics_identities.py` and claimed it "encodes the identities that
caught all three bugs in the integrity ledger," turning the project's best habit into a property of
the system.

**That claim is untested.** `mutmut` or `cosmic-ray` would measure it: deliberately mutate the
operators in a way equivalent to C2.6 (`D_eff = D/151`), C2.8b (peak-tracking), or the C3 boost IC,
and check whether the suite actually fails.

If a C2.6-equivalent mutation **survives** the suite, then the CI does not do what I said it does,
and I would want to know that before anyone relies on it. This is the highest-value item in Tier 3
after the free parity check, precisely because it can embarrass a claim already in the record.

---

## 6. The honest caution

> [!danger] Most of this is tooling, and tooling is not the bottleneck
> The pressure test's central finding stands: **verification is excellent (A), validation is absent
> (F)** — 16 free parameters against one external comparison yielding a binary sign law. Almost
> everything in this sprint is *verification or infrastructure* tooling.
>
> A better solver cross-check, a mutation-tested suite, a nicer viewer: all genuinely good, none of
> them moves the evidential position. The action plan's own warning applies unchanged — **more
> instrument work would now be avoidance.**
>
> The items that actually serve the science are narrow: the **perturbation-theory negative** (Phase
> 1a can proceed), **cadabra2** (Phase 1b if needed), and **SALib** (T2 → measured, which feeds Phase
> 2). Everything else is maintenance, however well-researched.

This is not a criticism of the sprint. The research is better than the project deserves. It is a
statement about what the project should *do* with it.

---

## 7. The plan

### Immediate (this week, ~half a day, zero or near-zero install)

| # | action | install | serves |
|---|---|---|---|
| **R1** | **Close the backend-parity residual.** Save one state array from the CuPy engine and the JAX mirror on a matched short run; compare with `cupy.testing.assert_array_almost_equal_nulp` and `jnp.allclose`. Record as a result note. | **none** | an open numerical residual, open since the baseline audit |
| **R2** | **Build HUD item 6.3** — centroid overlay using `skimage.feature.peak_local_max` on the density field, drawn into `tools/render_fields.py`'s overlay layer. | **none** (installed) | the RFC item most likely to have caught C2.8b |
| **R3** | **Run `mutmut` against `tests/test_physics_identities.py`** and record whether C2.6/C2.8b/C3-class mutations survive. | `mutmut` | §5 — tests a claim already in the record |

### Next (Phase 1, the scientific priority)

| # | action | install | serves |
|---|---|---|---|
| **R4** | **Start the Phase 1a hand derivation**, using Jakobsen 2013 as the method reference. The library has confirmed no shortcut exists. | `sympy` | **the #1 priority** |
| **R5** | If 1a is inconclusive → GAP-4 via **`cadabra2`** (action-principle framing), not EinsteinPy. | `cadabra2` | Phase 1b |
| **R6** | Once 1a produces a reduced ODE, integrate it with **`diffrax`** batched over separations/phases. | `diffrax` | Phase 1a validation |

### Then (Phase 2, and only inside the smooth basin)

| # | action | install | serves |
|---|---|---|---|
| **R7** | **SALib Sobol analysis** of the 16 parameters inside the confirmed a\* basin. Produces a measured importance ranking with interactions. | `SALib` | **T2**, and narrows where a discriminating quantity can live |
| **R8** | **PySR + SISSO** on the extracted force data as a blind cross-check of the R4 derivation. Agreement between two algorithmically distinct methods *and* the hand derivation would be strong. | `pysr` | Phase 1a corroboration |

### Deliberately deferred

`svirl`/`exponax`/`py-pde` (solver cross-check — strong, but verification), `Hypothesis`,
MMS/dt-convergence, `PyVista` 6.4, `cplot`, `Orbax`, `pixi`, `dynesty`. Each has a named trigger
in §3 rather than a date.

### Declined outright

`Aim`, `Sacred` (the record itself recommends against), `Hydra`/`dynaconf`, `psutil`+`Supervisor`,
`pyMultiobjective` (no license), `MASA` (the technique suffices), `openwave` (precedent only).

---

## 8. What the library should learn from this pass

Reciprocating the records' own "what the catalogue should learn" discipline:

1. **Record installed-vs-catalogued.** "Already catalogued" (sympy) read as "available" and it is
   not. An application record aimed at a specific project could cheaply check the project's own
   environment and say which finds are zero-install — that changes the effort estimate more than
   anything else in the record.
2. **Negative results are the highest-value output.** The perturbation-theory finding saved more
   than any tool in the sprint. Brief for them explicitly.
3. **A tool that tests the project's own claims is worth surfacing as its own category.** `mutmut`
   is not "a testing tool" here — it is a way to check whether an assertion already written into the
   project's record is true. That framing is what made it jump the queue.

---

## What changed as a result

- **Code / model changes:** none yet.
- **Verdicts changed:** none.
- **What was done next, and why:** pending Jake's decision on R1–R3.

## Issues raised

| # | issue | status | resolved by |
|---|---|---|---|
| 1 | Backend-parity residual open since the baseline audit; primitives were available all along | OPEN | R1 |
| 2 | Identity-CI bug-catching power is claimed but unmeasured | OPEN | R3 |
| 3 | T2 (16 free parameters) still a count, not a measurement | OPEN | R7 |
| 4 | HUD 6.3 centroid overlay unbuilt; standard tool already installed | OPEN | R2 |

## Associated docs

- [[ACTION_PLAN_2026-08]] · [[SYSTEM_PRESSURE_TEST_AND_METHOD_REVIEW]] · [[INFRASTRUCTURE_AND_ORCHESTRATION_ASSESSMENT]]
- [[VISUAL_HUD_SCOPE_RFC]] · [[SESSION_SYNTHESIS_2026-08]] · [[Main branch]]

## Branches

- [[Main branch]]

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `e42b5bb` (2026-09-11) — *Assess the external resource sprint: take three, decline most*
**Revised since:** 1 commit(s), most recently `bd93089` (2026-09-12)

**Later documents that cite this one** — the downstream consequences:

- [[INTEGRATED_PLAN_2026-09]] &middot; `2026-09-12`
- [[gravity_maturity/DERRICK_SCALING_AND_TARGET_TRIAGE]] &middot; `2026-09-12`

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
