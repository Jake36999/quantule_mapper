---
tags: [result, gravity, theory]
date: 2026-09-12
branch: Branch - Gravity - Index
status: complete
verdict: C2PRIME_GETS_NO_DERRICK_EVASION_FROM_OMEGA_RHO__TOWNES_TARGET_DISQUALIFIED_BY_PARAM_S_SIGN
---

# Derrick scaling for C2′, and triage of the Townes target

Author: Claude, 2026-09-12. Two checks the
research findings (`D:\Resource-Library\RESEARCH FINDINGS - Quantule Mapper Discriminating Evidence.md`)
identified as "project-side analytic or numerical work, not a literature question," performed
immediately because both are minutes rather than days.

Desk work. **No new simulation.** No gravity / UFF / IRER claim.

> [!abstract] Verdict
> `C2PRIME_GETS_NO_DERRICK_EVASION_FROM_OMEGA_RHO__TOWNES_TARGET_DISQUALIFIED_BY_PARAM_S_SIGN`

---

## 1. Derrick's theorem applies to C2′ unchanged

### The question

The findings established that **Derrick's theorem (1964)** governs any Lorentz-invariant scalar theory
with second-order derivatives in d > 2, that GAP-4's C2′ reformulation is **conservative by
construction** (its whole purpose), and therefore that C2′ needs one of the two documented evasions —
a time-periodic-phase Q-ball ansatz, or genuine dissipation, which C2′ explicitly is not.

The open question they named: **does C2′'s density-dependent kinetic weight `Ω(ρ)` change the scaling
argument's outcome?** If it did, the geometry itself would supply a third evasion route.

### The argument

C2′'s functional is `H[ψ] = ∫ [ D·Ω(ρ)|∇ψ|² + V(ρ) ] d³x`. Apply the standard Derrick dilation,
`ψ_λ(x) = ψ(x/λ)`:

- `ρ_λ(x) = |ψ(x/λ)|² = ρ(x/λ)` — **the density dilates, but its values are unchanged**
- therefore `Ω(ρ_λ(x)) = Ω(ρ(x/λ))` is the same function, merely stretched
- `|∇ψ_λ|² = λ⁻²|∇ψ|²(x/λ)`
- substituting `u = x/λ`, `d³x = λ³d³u`:

$$\int \Omega(\rho_\lambda)|\nabla\psi_\lambda|^2 d^3x \;=\; \lambda^{3}\cdot\lambda^{-2}\int \Omega(\rho)|\nabla\psi|^2 d^3u \;=\; \lambda\,E_\text{grad}$$

**`Ω(ρ)` is carried through as a passive weight and contributes no factor of λ.** The exponents are
the standard ones: `E_grad ~ λ^(d−2) = λ`, `E_pot ~ λ^d = λ³`.

### Numerical confirmation

Computed on a 96³ grid, L=40, with the blob well inside the box at every λ, for three different
weights including the exponential form the project actually uses (`A = exp(ε_G·G)`):

| `Ω(ρ)` | fitted `E_grad` exponent | fitted `E_pot` exponent |
|---|---:|---:|
| `1` (standard Derrick control) | **λ^1.00000** | λ^3.00000 |
| `1 + 2.5ρ − 0.8ρ²` | **λ^1.00000** | λ^3.00000 |
| `exp(0.9ρ)` | **λ^1.00001** | λ^3.00000 |

> [!warning] A first attempt gave 0.943 and was wrong
> On a 64³/L=12 grid the fitted gradient exponent came out at 0.9429. Inspecting the per-λ table
> showed `E_grad/λ` constant to five digits for λ ≤ 1.25 and degrading above it — the dilated blob was
> reaching the periodic boundary and corrupting the spectral gradient. **The deviation was a box
> artefact, not a modified exponent.** Recorded because it is exactly the class of error the project's
> own integrity ledger exists to catch, and because a 6% "deviation" would have been an interesting
> and completely false result.

### What this means

**C2′ buys no Derrick evasion from its geometry.** The density-dependent kinetic coefficient — the
feature that makes the reformulation "geometric" — leaves the scaling argument untouched.

| sector | Derrick status |
|---|---|
| **C3 Klein–Gordon Q-balls** | Evades correctly via time-periodic phase + conserved charge. A *known, correctly-applied* exemption, by construction. |
| **Phase C dissipative substrate** | Outside Derrick entirely — its localized states are attractors of a dissipative flow, not extrema of a conserved energy. A structurally different mathematical object. |
| **GAP-4 / C2′** | **Inside the constraint, with no evasion from `Ω(ρ)`.** Conservative by construction, static ansatz, standard exponents. |

> [!important] Consequence for Phase 1b
> **A variational reformulation of the geometry will not, on its own, produce stable static localized
> solutions.** If GAP-4 is pursued, it must be pursued with a Q-ball (time-periodic-phase) ansatz
> rather than a static one — the same evasion the C3 sector already uses.
>
> This does not weaken GAP-4's actual purpose. GAP-4 exists to make the geometry-on case exactly
> conservative and to fix the force sign by construction; neither of those goals requires static
> solutions. But a version of GAP-4 that expected static solitons would have been pursuing something
> Derrick rules out, and that is now checked rather than assumed.

### Scope

This is the **amplitude-preserving** dilation, which is the standard Derrick variation and the one the
theorem is stated for. Fixed-charge and fixed-norm variations behave differently — that difference is
precisely the Q-ball evasion, and is not in scope here.

---

## 2. The Townes critical-power target is disqualified for this project's actual runs

### The question

Request 4 of the findings proposed two dimensionless targets. The leading one was the **critical power
for 2-D self-focusing collapse** (Townes soliton), `p_cr ≈ 1.86`, proved sharply by Weinstein (1983).
The findings flagged one dependency: the cubic-quintic literature shows that a **defocusing** quintic
term does not merely shift that number, it **changes the bifurcation structure entirely** — arrested
collapse yields a family of *stable* solitons rather than a marginal unstable threshold.

Their stated action item: *check `param_s`'s sign in the runs the discriminator would be measured on.*

### The answer, from the results index

```sql
SELECT p.num, COUNT(*) FROM run_params p JOIN runs r USING(run_id)
 WHERE p.key LIKE '%param_s' OR p.key='s' GROUP BY p.num;
```

| `s` | runs | regime |
|---|---:|---|
| **−0.5** | **33 / 33** | **self-defocusing (stabilizing)** |

**100% of catalogued runs carrying this parameter sit at `s = −0.5`** — entirely inside the
cubic-quintic *stabilizing* regime, across both the KG-conservative and TG-dual-substrate sectors.

### What this means

> [!danger] Townes is the wrong target for this project
> The project does not sit near a marginal collapse threshold. It sits on the **stable-soliton branch**
> that the defocusing quintic term creates. Comparing against `p_cr ≈ 1.86` would be comparing against
> a bifurcation structure the model's own parameters exclude.
>
> This also retires the convention-matching step the findings flagged as "the one piece of arithmetic
> standing between this number and an actual comparison" — the arithmetic is no longer worth doing.

**A clean negative, arrived at before any effort was spent on it.** It leaves Request 4's *second*
candidate — the BEC critical-atom-number ratio — as the surviving target, and opens a new and sharper
question the findings could not have asked: **does the cubic-quintic stable-soliton branch itself
supply a dimensionless target?** Its existence and stability boundaries are structural properties of
the equation class. Whether they are *parameter-free* is the open part, since they plausibly depend on
the cubic:quintic coefficient ratio — which would be a free parameter and would fail criterion 2.

---

## 3. What these two checks change

| | before | after |
|---|---|---|
| GAP-4 / Phase 1b | "does `Ω(ρ)` change Derrick's argument?" — open | **Settled: no.** Q-ball ansatz required; static ruled out. |
| Request 4 target | Townes `p_cr ≈ 1.86`, pending a convention check | **Disqualified** by `s = −0.5` across all runs. |
| Effort saved | — | the convention-matching arithmetic, and any Phase-1b work premised on static solutions |

Neither result changes a verdict. Both close an open question at a cost of minutes, and both are the
kind of cheap analytic check [[../ACTION_PLAN_2026-08|the action plan]] argues should precede compute.

---

## What changed as a result

- **Code / model changes:** none. Both checks are analytic/desk.
- **Verdicts changed:** none. Two open questions closed.
- **What was done next, and why:** see [[../RESOURCE_LIBRARY_ASSESSMENT_2026-09]] for the remaining
  items from the same findings.

## Issues raised

| # | issue | status | resolved by |
|---|---|---|---|
| 1 | Does `Ω(ρ)` evade Derrick? | **RESOLVED — no** | §1 |
| 2 | Is Townes the right target given `param_s`? | **RESOLVED — no** | §2 |
| 3 | Does the cubic-quintic stable branch supply a parameter-free target? | OPEN | |
| 4 | Does GAP-4's stated scope assume static solutions anywhere? | OPEN | needs an RFC re-read |

## Previous experiments

- [[TG_P1_EVIDENCE_RECONCILIATION]] · [[TG_P2_MIDPLANE_STRESS_FLUX_RESULTS]]

## Associated docs

- [[../ACTION_PLAN_2026-08]] · [[CONCEPT_IMPLEMENTATION_AND_GAP_ANALYSIS]] — GAP-4
- [[../RESOURCE_LIBRARY_ASSESSMENT_2026-09]]

## Branches

- [[../Branch - Gravity - Index]]

---

<!-- LINEAGE:BEGIN (generated by tools/build_doc_lineage.py - do not edit inside) -->

## Lineage — what changed after this

> [!info] Generated by `tools/build_doc_lineage.py` — regenerated on each build.
> It reports *that* later work exists, not *why* it happened. The reasoning belongs in
> the **What changed as a result** and **Issues raised** sections above, written by hand.

**Version written against:** `bd93089` (2026-09-12) — *Derrick scaling for C2-prime and Townes-target triage: two open items closed*
**Revised since:** 1 commit(s), most recently `59b7ec3` (2026-09-12)

**Later documents that cite this one:** none. *Either this line of work stopped here, or the consequence was never written down — both are worth knowing when reviewing it.*

**Also referenced by (same date or earlier):** [[ACTION_PLAN_2026-08]], [[INTEGRATED_PLAN_2026-09]]

**Master catalog:** neither this document nor any verdict it reports appears in the catalog. Its status is **not tracked centrally** — treat anything inside as historical until confirmed.

<!-- LINEAGE:END -->
