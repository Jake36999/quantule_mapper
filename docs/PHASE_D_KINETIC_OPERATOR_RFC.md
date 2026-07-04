# Phase D — Kinetic Operator RFC

> **UPDATE (2026-07-04): C1 has been PROTOTYPED + TESTED on the jax_scout mirror → `C1_NO_STABLE_TRANSPORT`**
> (`PHASE_D_C1_RESULTS.md`). Adding the dispersive channel does **not** mobilise a\*: it confers only a transient,
> negligible drift and destabilises the attractor (mass runaway/fragmentation) at any dispersion strong enough to
> matter. a\* is a **fundamentally stationary dissipative attractor** — coherence and transport-conferring dispersion
> are incompatible. The frozen Phase C operator is untouched (`D_imag=0` = baseline byte-for-byte). **Remaining
> route: C2 (conservative NLS) — a *different object* (norm-conserving soliton, abandons the gain/loss balance),
> not an extension of a\*.** The sections below are the original design comparison.

**Status:** design/decision document. **No solver, physics, geometry, or gate change is made or proposed for
execution here.** This RFC compares candidate kinetic operators for the IRER **transport/matter sector**, states
what each would test, and recommends a first step. Any implementation is a *separate* go-ahead and happens **first
on the jax_scout mirror / a Phase-D branch**, never on the frozen Phase C production operator, until validated.

## 1. Why this exists (what Phase C settled, and the gap)
- **Stability sector — DONE.** The validated Phase C engine is a **dissipative** cubic-quintic-septic Ginzburg–
  Landau with density-sourced conformal geometry. Its linear operator is `L_k = -D·k² - η + i·ω₀` with **ω₀=0 at
  feb** ([`jax_scout/physics.py:297`](../jax_scout/physics.py), `solver/core.py:97`) — purely **real diffusion**.
  The re-aimed Hunter re-finds the gain/loss-balanced attractor a\*≈×1.15 (objective + adaptive search, cross-IC).
- **Mobility null is STRUCTURAL, not empirical.** Kick (Galilean phase boost) and adiabatic-drag tests found no
  inertial or advective transport — because a *real* `-D·k²` has **no advective/dispersive channel by
  construction**: every mode just damps (`exp(-Dk²dt)`); nothing propagates ([[hifi-continuation-vortex-real]],
  Phase C kick/drag findings). The attractor is a **site-pinned dissipative soliton**.
- **The formalism gap review** (`PHASE_D_FORMALISM_GAP_REVIEW.md`) determined **C+D**: the theory conceives *both*
  a dissipative stability sector (implemented, faithful) *and* a matter/transport sector (Gradient-Derived
  forces→motion, Payan spin, spin-wave propagation) that is **aspirational and un-formalized**, with the kinetic
  operator/time-order **under-specified**. ⇒ **Phase D transport is a deliberate formalism *choice*, not the
  recovery of a specified operator.** This RFC makes that choice explicit and testable.

## 2. What has already been tried (so we don't repeat nulls)
The solver carries **default-off** optional coupling fields, several already explored in the Phase-C/geometry era:
| hook | form | intent | Phase-C-era result |
|---|---|---|---|
| `a_vec` | minimal coupling `∇→∇ − i·γ_A·A` (`physics.py:226`) | gauge-like current coupling / routing | routing **NULL** ([[routing-null-promising-for-payan]]) |
| `q_tensor` | `+ i·D·div(Q∇ψ)` anisotropic **dispersive** (`physics.py:238`, `ANISOTROPIC_METRIC_TENSOR_RFC.md`) | metric-like flow redirection | Stage-B **NO_SUPPORT** ([[stage-b-tensor-geometry-no-support]]) |
| `drag_field` | `+ V0·G(x−x_c(t))·ψ`, **real** (`physics.py:253`) | relational (gradient-following) mobility | accretion/nucleation, **no relocation** |
| Payan (phase-winding) | passive diagnostic | protected circulation | `PAYAN_COUPLING_NOT_JUSTIFIED` ([[payan-diagnostic-defined]]) |

**Lesson:** these were added as *perturbations on top of the still-dominant real-diffusive substrate* (which keeps
dissipating), or as passive diagnostics — **not** as a change to the kinetic operator that gives the field a
genuine transport channel. Phase D's real question is at the **operator** level: *does the kinetic operator itself
carry an advective/dispersive channel?*

## 3. Candidate kinetic operators
Notation: field ψ (complex), density ρ=|ψ|². Current baseline nonlinearity `a·ψρ + s·ψρ² + f·ψρ³` and the
conformal geometry Ω²(ρ) are **held fixed** in all candidates unless stated — we vary only the *kinetic* term.

### C0 — Real-diffusive (BASELINE, validated) — *reference, not a proposal*
`∂ψ/∂t = D∇²ψ − ηψ + N(ψ,ρ)`, `L_k = −D·k² − η`. Dissipative reaction–diffusion. **No transport** (mobility null).
This is Phase C; it stays frozen. Every candidate below must **reduce to C0** in the appropriate limit so C0's
results are preserved.

### C1 — Mixed dissipative+dispersive (complex diffusion) — **RECOMMENDED first step**
`∂ψ/∂t = (D_r + i·D_i)∇²ψ − ηψ + N`, i.e. `L_k = −(D_r)·k² − η + i·(−D_i·k²)`. A genuine **complex Ginzburg–Landau**:
keeps the validated gain/loss balance (D_r, η, N unchanged) and adds a **Schrödinger-like dispersive channel**
(i·D_i∇²) that is |ψ|²-*redirecting* rather than dissipating. `D_i=0` ⇒ **exactly C0** (baseline preserved).
- **IRER concept tested:** does the validated **dissipative attractor acquire a transport DOF** when a dispersive
  channel is opened? (matter = *moving* dissipative soliton; "resonant inertia" gains a genuine velocity response).
- **Distinguishing observable:** re-run the Phase-C **kick** and **adiabatic-drag** tests. C0 gave v≈0 (pinned);
  C1 should give a **finite mobility** μ=v/k>0 and/or a soliton that **translates** to follow a moving well —
  while a\*'s amplitude/breathing survive (a moving dissipative soliton, not a decaying transient).
- **Success:** `D_i=0` reproduces a\* bit-for-bit; a small `D_i>0` yields measurable, `D_i`-scaling inertial/
  advective transport with the coherent structure preserved. **Failure:** no transport at any stable `D_i`, or the
  dispersive term destabilizes a\* for all `D_i>0` (transport and stability incompatible in this form).
- **Numerical risk:** LOW–MED. Fits the existing **ETDRK4** scheme directly (`L_k` just gains an imaginary `k²`
  part; the exponential integrator already handles complex `L_k`). Dispersive `i·k²` needs adequate dealiasing/
  resolution (Schrödinger CFL); the KT contour is unaffected. **Phase D** (changes what is tested), not C.

### C2 — Conservative nonlinear Schrödinger (Hamiltonian) — *larger reformulation*
`i·∂ψ/∂t = −D∇²ψ + N` (η=0, gain/loss OFF). Fully **conservative**, |ψ|²-conserving; supports **moving bright/dark
solitons** with genuine momentum.
- **IRER concept tested:** matter as a **conservative** coherent particle (momentum/energy conserved), not a
  dissipative attractor. Directly probes "Gradient-Derived forces → motion + conservation" (Q3).
- **Distinguishing observable:** Galilean boost gives ballistic translation at constant v with conserved norm;
  two-soliton collisions are elastic/near-elastic.
- **Success/Failure:** stable moving solitons exist / don't. **But this ABANDONS the gain/loss balance that *is*
  the Phase C result** — a\* (a dissipative attractor) is not a state of C2. So C2 is a *different physics regime*,
  not an extension of the validated one. **Numerical risk:** MED (needs a norm-conserving integrator; ETDRK4 with
  η=0 works but the physics is different). **Phase D**, and a scope change worth flagging.

### C3 — Second-order-in-time (damped wave / FMIA-propagation)
`∂²ψ/∂t² + γ·∂ψ/∂t = D∇²ψ − ηψ + N`. Adds **inertia in time** → ballistic/wave propagation with a group velocity;
`∂²ₜ→0` (overdamped) recovers C0.
- **IRER concept tested:** the theory's "FMIA channels" / spin-wave propagation as *genuine* wave dynamics; a
  literal reading of "resonant inertia" as second-order inertia.
- **Distinguishing observable:** finite propagation speed, dispersion relation ω(k), wave echoes/reflections.
- **Numerical risk:** HIGH. **Requires a new time integrator** (2nd-order in time; the ETDRK4/`L_k` exponential
  scheme is first-order-in-time by construction) — a solver-architecture change, not a term swap. **Phase D, heavy.**
  Also weakly supported by the provenance (time = "chronology of resolution", leans first-order) — see the gap review Q1.

### C4 — Genuine non-local coupling (Field-of-Affect coupled)
Couple the already-*computed*-but-*uncoupled* Field of Affect (γ_A≠0) as a real non-local kernel `∫K(x−x')ρ(x')`.
- **IRER concept tested:** "splash" non-locality (Q6). Transport via non-local interaction, not local dispersion.
- **Note:** `a_vec`/`γ_A` routing was **NULL** in the geometry era; this would be a *different* coupling of the
  same field. **Numerical risk:** MED. **Phase D**, but lower priority given the prior null.

## 4. Comparison
| candidate | preserves a\*/Phase C? | opens transport? | integrator change? | IRER concept | risk | priority |
|---|---|---|---|---|---|---|
| C0 real-diffusive | — (is Phase C) | no (structural) | none | stability sector (validated) | — | frozen ref |
| **C1 mixed complex-cGL** | **yes (`D_i=0`→C0)** | **yes (dispersive)** | **none (ETDRK4)** | moving dissipative soliton | **low–med** | **1st** |
| C2 conservative NLS | no (drops gain/loss) | yes (ballistic) | norm-conserving | conservative matter | med | alt (scope change) |
| C3 2nd-order wave | limit only | yes (wave) | **yes (new scheme)** | FMIA/spin-wave inertia | high | later |
| C4 non-local coupled | yes (γ_A add-on) | maybe (non-local) | none | splash non-locality | med | later (prior null) |

## 5. Recommendation
**Start Phase D with C1 (mixed dissipative+dispersive complex-cGL).** Rationale: it is the **minimal** change that
(a) **preserves the entire validated stability sector** (`D_i=0` is exactly C0, so a\* and all Phase C claims
survive as the `D_i→0` limit), (b) opens a **genuine transport channel** at the operator level (the thing the
Phase-C-era add-on proxies never did), (c) needs **no new integrator** (drops into ETDRK4 via a complex `L_k`), and
(d) directly answers the sharpest open question — *"can the validated dissipative attractor MOVE once given a
dispersive channel, or is site-pinning fundamental?"* C2 (conservative) and C3 (2nd-order) are larger, more
speculative reformulations to hold until C1 answers that. This mirrors the physics.py comment already in the code:
a real `D∇²` "just smooths/dissipates", whereas an imaginary `i·D∇²` is "metric-like (Schrödinger), |ψ|²-conserving,
redirects flow" (`physics.py:232-235`).

## 6. Proposed validation protocol for C1 (when/if approved — design only)
1. **Baseline-preservation (parity):** implement complex-`L_k` behind a **default-off `param_D_imag` (=0)** on the
   **jax_scout mirror**; confirm `D_i=0` reproduces a\* bit-for-bit (rel-L2 ~1e-12) — the C0 result is untouched.
2. **Transport probe:** at small `D_i>0`, re-run the **kick** and **adiabatic-drag** batteries that were NULL in C0;
   measure mobility μ=v/k and well-following displacement vs `D_i`. Certify structure survival with `css.classify`.
3. **Stability×transport map:** small grid over (`D_i`, param_a) — where does a moving *and* certifiable soliton
   exist? Pre-register PASS = "a `D_i>0` regime with finite mobility AND a\*-like certifiable coherence."
4. **Only on PASS** consider a production-path (CuPy) mirror of the term, with its own parity + re-validation.

## 7. Guardrails / non-goals (unchanged discipline)
- **Nothing is implemented by this RFC.** The frozen Phase C operator (`e8d6a78ea`) and all Phase C claims stand;
  C1 is an **additive, default-off** extension validated on the mirror first.
- Adding a dispersive term **changes what the simulator tests** (transport sector) — Phase C's dissipative-sector
  results remain valid *for the dissipative baseline* and are the `D_i→0` limit.
- Not proposed: replacing the nonlinearity, the conformal geometry, or the gate; a new time-integrator (unless C3
  is later chosen); any matter/mobility *claim* before the C1 protocol passes.
- **Decision required before any code:** which candidate direction (default: C1), and whether to prototype C1 on
  the mirror behind a default-off flag.
