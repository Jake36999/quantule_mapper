# Phase C — feb attractor dossier (discussion-ready end state)

**Date:** 2026-06-27 · **Geometry frozen at** `e8d6a78ea` · **Classifier:** `PHASE_C_SATURATION_CLASSIFIER_v3`
This is the single authoritative summary of the feb-attractor result: the defensible claim, the evidence,
the honest negatives, reproducibility, and the forward roadmap. It indexes the detailed docs rather than
re-deriving them. **This is framed as a numerical/mathematical result, not a physical proof.**

---

## Headline claim (defensible)

> The current IRER-derived S-NCGL + emergent-conformal-geometry solver supports a **reproducible,
> resolution-checked, long-time-stable bounded attractor core** (the "feb" regime). Its existence is
> governed by a **gain/loss balance** across `param_a` (cubic gain), `param_eta` (linear loss), and
> `param_rho_vac` (conformal reference density). The basin boundary is **structured and partly
> seed-sensitive**; marginal `T=12000`-TRUE states can decay or grow by `T=24000`. The attractor is a
> **bounded breather**, and near the boundary stable vs failed states differ **only dynamically** — not in
> node count, spatial spectrum, or topology. **Prime-harmonic and topological diagnostics are null** for
> this family.

Plain language: there is a stable, repeatable "object" in this model (a 4-node breathing core). We know the
parameter knobs that make it exist, what makes it fail, that it survives resolution/seed/long-time at its
core, and that the older prime/topology validation tools say nothing distinctive about it.

---

## Claims table

Each claim: **supporting runs** · **caveat / counterexample** · **falsifier** · **next test**.

### A — feb-center is stable and resolution-robust
- **Support:** `FEB_BASIN` + `FEB_BASIN_CONFIRM` (TRUE across K/seed/mass) → `b0d393889`, `ac9ca380f`;
  **`FEB_CENTER_RESOLUTION_N128`** TRUE at N=128 for T=12000 **and** T=24000, same 4-node morphology,
  er_max≈1.58 like N=96, bounded breathing → `41e05816d`.
- **Caveat:** breathing amplitude grows with resolution (drift −0.26 @ N128/T24000 vs −0.18 @ N96).
- **Falsifier:** feb-center fails / changes class at higher N or from saved config.
- **Next test:** one N=256 spot-check if GPU allows (optional).

### B — the basin is parameter-controlled, not visual/morphological
- **Support:** OAT map (`ba1acb3f0`) + joint grid Stage 1 (`772903083`); matched controls show **n_fin=4 on
  both sides of every stable↔failed flip** (morphology does not change at the boundary) → `8c82e3f0d`.
- **Caveat:** node count itself *does* vary with IC blob placement (seed) — but not with stability.
- **Falsifier:** outcome uncorrelated with parameters / driven by morphology.
- **Next test:** none needed; established.

### C — gain (`param_a`) and reference density (`param_rho_vac`) rescue high-loss cells (coupling)
- **Support:** joint Stage 1 + 2 — at high loss (`eta×1.2`), raising `param_a` (×1.05/1.1) or `param_rho_vac`
  (×1.25) flips SPIN→TRUE; lowering `param_a` flips a low-loss/high-drive GROWER→TRUE → `772903083`,
  `8c82e3f0d`.
- **Caveat:** the rho-rescue at the *lowest* gain is seed-fragile (Claim D).
- **Falsifier:** no compensation relationship between the three axes.
- **Next test:** finer joint resolution near the diagonal boundary (optional).

### D — the basin boundary is structured and partly seed-sensitive (quantified)
- **Support:** edge-confirm (OAT edges 10/14, upper edges seed-move) `0a3940e98`; joint Stage 2 boundary
  seed-repeat **14/16 match** — gain-rescue seed-robust; one seed-fragile corner
  (`a×0.9, eta×1.2, rho×1.25`: TRUE@619 → SPIN@620/621) → `8c82e3f0d`.
- **Caveat:** boundary location moves ~one cell with IC seed at the marginal high-loss corner.
- **Falsifier:** boundary behaviour random rather than structured; or interior also seed-flips.
- **Next test:** seed-distribution at the fragile corner (3–4 seeds) if precision needed.

### E — the `T=24000` core is narrower than the `T=12000` basin
- **Support:** joint Stage 2 interior — core `a1.05,e1.0,r1.0` TRUE/breathing at T24000, but low-drive
  corner `a0.9,e1.0,r0.85` **decays SPIN by T24000** (TRUE@T12000), high-drive corner `a1.1,e1.0,r1.25`
  marginal (er_fin 2.31) → `8c82e3f0d`.
- **Caveat:** the eta-width of the core and its upper-drive limit are not yet mapped (only the eta×1.0
  plane was delineated).
- **Falsifier:** feb-center itself fails at T24000 (it does **not** — confirmed, er_fin 1.20, interior).
- **Resolved:** `FEB_CORE_DELINEATION_T24000` → `PHASE_C_T24000_CORE_DELINEATION.md`. The T24000 core is a
  clean diagonal band, **12/15 of the eta×1.0 plane**; only the low-drive corner (low gain + low rho)
  decays. `er_fin` rises monotonically with gain and reference density. feb-center and `a×1.05,r×1.0` are
  well-interior — the object definition for downstream tests.

### F — stable vs failed differ dynamically, not topologically/morphologically
- **Support:** matched controls — one-parameter step flips outcome under the identical gate; for **every**
  pair, **n_fin=4 both sides, prime peaks=0 and persistent topology=0 both sides**; the difference is
  bounded-breathing (`breath=True`) vs decay/grow (`breath=False`) → `8c82e3f0d`. Observable-extraction
  (`b88de01c6`): snapshot/spatial observables do not separate stable/failed (all d<0.3); `k_peak=1` for all.
- **Caveat:** the dynamical discriminator is essentially what the v3 gate uses (not independent of it).
- **Falsifier:** a morphological/spectral observable found that separates stable/failed.
- **Next test:** long-T breathing dynamics as the genuine independent observable (Claim H route).

### G — prime-harmonic / TDA / log-prime diagnostics are null for this family
- **Support:** validation-stack audit + post-hoc on 60 + 45 states — **0 prime peaks, ~0 persistent
  topology** everywhere; `log_prime_sse=999` for all → `1bfd59bd1`. The tools are alive and run on a numpy
  `rho` with no adapter; they simply do not light up.
- **Caveat:** profiler thresholds were tuned for the CuPy production fields; "no peaks" = no prime structure
  at those settings — but uniform across 105 varied states, so genuine absence.
- **Falsifier:** a structured state in this family shows a real prime/TDA signal.
- **Next test:** none for promotion; keep as exploratory diagnostics only.

### H — (pending) the next object-level test is matter-likeness (mobility/inertia), not more basin widening
- **Support:** the result now defines a clean object class — *bounded 4-node breathing attractors in a
  gain/loss-balanced basin* — stable enough at its T24000 core to perturb.
- **Caveat:** must use the **T24000-confirmed core** (feb-center, `a1.05,e1.0,r1.0`), **not** the marginal
  corners.
- **Falsifier:** the core does not move coherently / has no reproducible effective inertia under a kick.
- **Next test:** the **kick/inertia test** (see roadmap).

---

## Honest negatives & explicit non-claims

- **No prime-harmonic structure** (Claim G). **No topological invariant / phase transition** (G).
- **No external-data validation yet** — snapshot spectra do not separate stable/failed, so spatial-spectrum
  comparison is a dead end; the only viable external route is long-T **breathing dynamics** vs CGL /
  dissipative-/cavity-soliton optics, treated as analogy/consistency-check, **not** as IRER proof. A CGL
  match would be near-tautological (the equation *is* in the CGL family).
- **No matter / ground-state / molecule / black-hole / "quantum" claims.** The object is a classical
  dissipative breathing soliton in a parameter-controlled basin.
- **Not a physical proof of IRER.** This is a numerical/mathematical characterisation of the solver's
  attractor structure.

---

## Methodology & reproducibility

- **Solver/geometry frozen** at commit `e8d6a78ea` (comment-only diff; verified equivalence-proven FP64 JAX
  mirror — `docs/MATHS_VERIFICATION.md`). **No PDE/solver/geometry change in any of this work.**
- **Promotion gate = `classify` v3** (`docs/PHASE_C_GATE_V3_BREATHING_BOUND_STATE.md`): in-band energy,
  bounded `er_max`, node count, **normalized late-half drift with a bounded-breathing exception**. Calibrated
  + tested (13/13). Discovery T=4000 shortlists; promotion requires a **T≥12000** drift-gated pass; long-time
  promotion at the margins requires **T=24000**.
- **Standard config:** N=96 (N=128 resolution-checked), K=6 / per-blob-fixed norm, seed 20260619 (+ 620/621
  for robustness), `param_omega0=0`, the four insensitive params at feb.
- **Scripts (read-only w.r.t. physics; resumable, deadline-aware):** `jax_scout/feb_basin_search.py`,
  `feb_basin_confirm.py`, `feb_param_basin.py`, `feb_param_edge_confirm.py`, `feb_center_resolution.py`,
  `feb_joint_basin.py`, `feb_joint_stage2.py`. Diagnostics: `quantulemapper_real.prime_log_sse`,
  `tda_profiler.extract_and_classify_topology` (numpy, no adapter), `quantule_viz` renderers.
- **Detailed records:** strategy/falsification `PHASE_C_RESEARCH_STRATEGY_AND_FALSIFICATION.md`;
  consolidated `PHASE_C_FEB_PARAMETER_BASIN_CONSOLIDATED_ANALYSIS.md`; joint Stage 1/2
  `PHASE_C_JOINT_PARAM_BASIN_RESULTS.md`, `..._STAGE2_RESULTS.md`; validation audit
  `PHASE_C_VALIDATION_STACK_AUDIT.md`; gate calibration `PHASE_C_STABILITY_GATE_CALIBRATION.md`.

---

## Forward roadmap (the remaining work to a solid stopping point)

1. **Delineate the T=24000 core precisely** *(next run, cheap):* T=24000 on a handful of interior cells
   around feb to draw the long-time-stable region's boundary (separate the genuine core from the
   T12000-marginal shell). ~1–2 h.
2. **Long-T breathing characterisation** *(next):* feb-center + 1–2 core cells at **T≈48000–72000** with
   dense `er`/snapshot sampling, to resolve the breathing **period, amplitude, and sidebands** over multiple
   cycles — the genuine dynamical observable (and the only viable external-comparison handle). ~few hours.
3. **Robustness tests:** confirm the core under remaining axes (more seeds at the core, the N=256 spot-check,
   per-blob vs total-mass at the core long-T).
4. **Matter-likeness Phase 1 — kick/inertia test** *(the next object-level gate, only on the T24000 core —
   feb-center, `a1.05,e1.0,r1.0`):* apply phase-gradient kicks; measure centre-of-density motion; derive
   velocity vs kick (effective inertia); check shape/breathing preservation and energy stability; repeat at
   N96/N128. **Coherent motion + reproducible effective inertia = the first real step toward a matter-like
   excitation.** If it does not move coherently, that is a clean, informative stop.

A solid project end point = roadmap items 1–3 done (core delineated, breathing characterised, robustness
confirmed) with the dossier above; item 4 is the optional extension that, if it holds, defines the next
chapter; if it fails, it bounds the claim cleanly. Either way the result is defensible and finite.

No charge / topology-proof / log-prime-proof / matter / ground-state / molecule / black-hole language is
asserted; "matter-likeness" above names a *test*, not a claim.
