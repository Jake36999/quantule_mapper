# Phase C — research strategy, end goal, and falsification plan

**Date:** 2026-06-26
**Purpose:** define a defensible near-term end goal, the falsifiers, what to look for beyond stability,
and a disciplined external-comparison protocol — so the work can go to a wider technical discussion as a
**numerical/mathematical result**, not a physical proof. Anchors:
[CONSOLIDATED_ANALYSIS](PHASE_C_FEB_PARAMETER_BASIN_CONSOLIDATED_ANALYSIS.md),
[VALIDATION_STACK_AUDIT](PHASE_C_VALIDATION_STACK_AUDIT.md).

## 1. Defensible end goal (not "prove IRER")

> Demonstrate that this IRER-derived numerical model contains a **reproducible, parameter-controlled,
> bounded-attractor family** — with explicit stability criteria, systematic failure modes, seed /
> long-time / resolution robustness, and a **falsifiable** basin boundary — and characterise whether its
> dimensionless dynamical signatures resemble recorded dissipative-nonlinear-optical phenomena more than
> null controls do.

Framed as a numerical result this is discussion-ready well before 100% basin mapping. It is explicitly
**not** a claim of physical proof, of the log-prime hypothesis, of a topological invariant, or of matter /
ground states / molecules / black holes.

## 2. What we are looking for, specifically

A **predictable, falsifiable structure–function relationship**: does the parameter regime *control* the
attractor coherently and reproducibly, and do failures occur *systematically* (typed: spin-down / grower /
blowup) rather than randomly? The currency of evidence is **matched success/failure pairs** — same grid,
same pipeline, one parameter perturbation flips the outcome. Existence of a stable state is the start;
**controlled, typed, reproducible structure** is the result.

## 3. Claims & falsifiers (the discussion package skeleton)

| # | Claim | Supporting runs | Counter / open | Falsifier |
|---|---|---|---|---|
| A | feb-center survives K / seed / mass-norm variation | FEB_BASIN, FEB_BASIN_CONFIRM (10/10) | — | feb fails to reproduce from saved config/seed |
| B | `param_a` (cubic gain) is the tightest knob (≈±10%) | OAT map; edge confirm (×0.9/×1.1 robust) | upper edge seed-moves | `param_a` window not reproducible / not tightest jointly |
| C | `param_eta` & `param_rho_vac` are the sensitive trade-off axes | OAT map | coupling untested (joint grid) | joint grid shows no coherent param→outcome relation |
| D | basin interior seed-robust; **upper boundary seed-sensitive** | FEB_PARAM_EDGE_CONFIRM (10/14) | quantify boundary spread | boundary behaviour random, not structured |
| E | log-prime-SSE & TDA/Betti are **null** for this family | POSTHOC (0/60 peaks, ~0 topo) | — | a structured state shows real prime/TDA signal we missed |
| F | stable states are **bounded breathers**, not static objects | T=24000 family (drift, breathing) | — | "stable" states are static / or actually slow transients |
| **G ✓** | feb-center is **resolution-robust** (N=128) | FEB_CENTER_RESOLUTION_N128 (TRUE @ T12000 & T24000, 4 nodes, er_max≈1.58 like N96, bounded breathing) | breathing deeper at N128 (drift −0.26 @T24000) — same state, larger amplitude | feb-center fails at higher N → grid artifact |

Global falsifiers for the headline claim ("a bounded attractor basin controlled by gain/loss balance via
`param_a`, `param_eta`, `param_rho_vac`"): (1) feb-center fails under resolution or long-time replay;
(2) the basin disappears across seeds; (3) boundary behaviour is random; (4) v3 accepts known
transient/grower controls; (5) stable states don't reproduce from saved configs; (6) post-hoc diagnostics
show **no** difference between stable states and matched failed controls on **any** axis; (7) the joint
grid shows no coherent parameter→outcome relationship.

## 4. What to watch beyond "stable nodes" (priority = how decisively it could kill the result)

1. **Resolution robustness (N=96 → N=128).** Highest-leverage falsifier; the whole feb-basin family is
   N=96. **DONE — PASSED:** feb-center is TRUE at N=128 at T12000 & T24000, same 4-node morphology and
   er_max envelope (≈1.58) as N=96, bounded breathing. The basin is not an N=96 grid artifact. (Side
   finding: the breathing amplitude grows with resolution — drift −0.26 @T24000/N128 vs −0.18 @N96 — so
   v3's breathing exception is *more* necessary at higher N; v2 would false-reject it.)
2. **Matched negative controls** under the identical gate + post-hoc diagnostics (the biggest method
   upgrade).
3. **Failure-mode structure** — is the boundary a taxonomy (predictable) or noise?
4. **Breathing dynamics as observables** — period/amplitude/spectra of `er(t)`, `rho_peak(t)`; the bridge
   to external data (extractable from existing runs now).
5. **Coupled basin geometry** (joint grid) — a falsification test of the OAT story under coupling.
6. **Current/vorticity closure** as a structural descriptor (does not discriminate stability, but is a
   signature for external comparison).

## 5. External comparison — post-run analogy/search, with strict framing

The method (extract dimensionless observables → compare to external optical data via
SSE / spectral-correlation / peak-ratio / KL / DTW, **against null controls**) is sound and reusable. The
legitimate question is *"do the simulated stable states' signatures resemble recorded nonlinear-optical
phenomena more than off-basin controls do?"* — analogy and search, never proof. Two load-bearing rules:

- **Target dissipative / cavity-soliton & complex-Ginzburg-Landau optics, NOT SPDC.** The S-NCGL is a
  Ginzburg-Landau-family dissipative wave equation, so dissipative solitons, breathing solitons, cavity
  solitons, and mode-locked-laser sidebands are the *mathematically natural* comparison class. SPDC is a
  quantum χ² parametric process (signal–idler photon pairs, phase matching) with no analogue in this
  classical model — a direct SPDC comparison is a category error a reviewer will flag. SPDC is at most an
  explicitly-labelled spectral-envelope analogy, demoted below the CGL/soliton optics targets.
- **A CGL match is a consistency check, not IRER evidence.** Because the governing equation *is* in the
  CGL family, resembling CGL optics is near-tautological — it shows "credible dissipative-nonlinear
  system," which earns the right to ask the "seen in real life?" question but does **not** support the
  IRER-*specific* content (emergent geometry, prime-harmonic target, A-field). The diagnostics that would
  test that specific content (prime-SSE, TDA) are null (Claim E). This must be stated plainly.

**Observable vector to extract (free, read-only on existing data first):** temporal spectra of
`er(t)`/`rho_peak(t)` (breathing period/amplitude, sidebands); spatial power spectra of `rho` and current;
dominant peak/sideband spacing & harmonic ratios; bifurcation curves of these across the `a`/`eta`/`rho_vac`
boundary; failure signatures (stable interior vs spin-down/grower/blowup controls). **Prerequisite check:**
do stable vs failed states even *differ* in these observables? If not, external comparison is moot.

> **Prerequisite result (2026-06-26, read-only on 60 states, `FEB_OBSERVABLE_EXTRACTION_20260626_220710`):**
> stable and failed states do **not** separate strongly in *snapshot* observables — all separation metrics
> weak (d < 0.3). Spatial spectra are envelope-dominated (`k_peak = 1` for **every** state → spectrally
> featureless smooth solitons, consistent with the prime-SSE/TDA null). The discriminator that *does* work
> is **dynamical** — the `er(t)` trajectory shape (bounded breathing vs monotonic decay/grow), i.e. what v3
> already uses. **Implication:** external comparison via *spatial spectra* is not meaningful (stable/failed
> look alike); the meaningful axis is the **breathing dynamics** (period, amplitude, attractor shape) vs
> optical breathing/cavity solitons — but the breathing period is long (~ the current T window), so it is
> **under-resolved** and needs **longer-T runs (multiple breathing cycles)** to characterise before any
> external comparison. External comparison is therefore **premature on current data**; the path to making
> it meaningful is the breathing-dynamics route, not snapshot spectra.

## 6. Phased path

**Phase 1 — close internal falsifiers (decisiveness order):** (1a) resolution check feb-center + 1–2
interior cells at N=128; (1b) joint `(a,eta,rho_vac)` grid Stage 1 breadth + Stage 2 boundary
seed-expansion; (1c) matched off-basin controls through the identical gate + post-hoc diagnostics.
**Phase 2 — validation dossier:** the §3 table filled with runs/counters/tests/falsifiers.
**Phase 3 — external-comparison pilot:** *(prerequisite done — see §5 result: snapshot observables do not
separate stable/failed; the breathing period is under-resolved.)* Revised: (i) run a small **long-T
breathing-characterisation** set (feb-center + 1–2 interior cells at T≈48000–72000 with dense `er`/snapshot
sampling) to resolve the breathing period, amplitude, and any sideband structure across multiple cycles;
(ii) confirm those *dynamical* observables separate stable from failed; (iii) only then compare to
CGL/dissipative-soliton **breathing-soliton** optics data vs null controls. Do not pursue a spatial-spectrum
comparison — these states are spectrally featureless.

**Discussion-ready threshold (no 100% basin needed):** joint Stage 1 done · boundary cells seed-repeated ·
1–2 interior cells confirmed at T=24000 *(node family done)* · **one resolution check** · external pilot
with null controls · all claims written with falsifiers.

No charge / topology-proof / log-prime-proof / matter / ground-state / molecule / black-hole language.
