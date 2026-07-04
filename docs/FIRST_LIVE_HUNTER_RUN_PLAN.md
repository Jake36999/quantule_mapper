# First Live Hunter Run — Plan (tiny, controlled)

> **EXECUTED 2026-07-04 → `LIVE_HUNTER_REDISCOVERY_PASS`.** See `FIRST_LIVE_HUNTER_RUN_RESULTS.md`. Ran in the minimal
> stability-selection mode below (H7.1b delivered the search-operator alignment). This plan is retained for provenance.

**Gate:** this plan is unlocked because **A5 production re-validation PASSED** (`sweep_runs/A5_PROD_20260703_192713`:
a\*≈×1.15 re-found from the production single-Gaussian IC, top-ranked + certified at 0.884; decayer below; grower
`GROWER_BLOWUP`; short-window not promoted) **and H4/A1 parity PASSED** (rel-L2 1.7e-12). What is *validated*: the
re-aimed **objective** re-finds a\* on the production CuPy path by **replay**. What is **not** yet shown: that the
Hunter re-finds a\* by **adaptive search**. This run is the smallest experiment that closes that last gap.

**This document does not authorize a run.** Executing it is a separate, explicit go-ahead. No adaptive Hunter hunt
has been started.

## Purpose (narrow, pre-registered)
Confirm the re-aimed Hunter's **search loop** (propose → evaluate → select → converge) with `objective="stability"`
walks *toward* the known a\* basin and ranks it top — i.e. **re-finds a known result**, not discovers new physics.

## Configuration (deliberately tiny)
| knob | value | why |
|---|---|---|
| `objective` | **`"stability"`** | fitness from `tools.stability_objective`; prime-SSE recorded but non-steering |
| population | **6** | smallest that still has selection pressure |
| generations | **≤ 3** | enough to show convergence, not a hunt |
| search box | **narrow, around the validated basin** (feb-relative): `param_a ∈ [0.4802, 0.6003]` (feb×1.00–1.25, brackets a\*=0.5522 with the decayer/grower edges), `param_eta ∈ [0.0598, 0.0810]` (×0.85–1.15), `param_rho_vac ∈ [1.068, 1.365]` (×0.90–1.15); **all other params fixed at feb** | "no broad search" — a tight box on the 3 validated-sensitive axes only |
| Gen-0 seeding | include feb + a couple of draws near a\* | this is a **re-find**, not a blind hunt |
| resolution / window | **N=96**, staged eval: **screen T=8000** (rank only — uncertifiable by design), **certify top-2/gen at T=36000** (≥ MIN_STABILITY_T) | screen is cheap (~10 min/cell); only finalists pay the ~44 min long-T cost |
| certifier | **`css.classify` / the stability gate** (unchanged) | Hunter proposes; the gate certifies. No promotion of T<24000 screens. |

**Cost:** ~6×3 = 18 screens (~3 h) + ~6 long-T certifications (~4.4 h) ≈ **7 h** on the local GTX 1080 (`.venv`).
It is a long *tiny* run because N=96 long-T is intrinsically slow; reduce to gens=2 / certify top-1 to halve it.

## Prerequisite / honest scope limit (must be settled before running)
The H7.1 wiring makes **fitness** come from stability, but the Hunter's **search operators** (SGN gradient-nav, ASMT
manifold, NSGA fronts) are still tuned on the *spectral* objectives and steer `param_a_coupling`/`splash`, **not**
`param_a`/`eta`/`rho_vac` (this is the open **H7.1b**). So for this first run, use the **minimal selection mode**:
- **stability-fitness tournament** (`TOURNAMENT_SIZE=3`) + **bounded Gaussian mutation** confined to the narrow box;
- **SGN / ASMT / SBD disabled** (they would steer on the wrong, spectral-tuned axes);
- this isolates the test to *"does stability-fitness selection converge to a\*?"* without the un-redirected operators.

A fuller adaptive stability hunt needs **H7.1b** (redirect SGN/ASMT + a real stability-NSGA front over
late-slope / boundedness / breathing / seed-robustness). That is **out of scope** for this first re-find run.

## Success (all must hold) — PASS = re-found the known basin
- the top-ranked **certifiable** individual after ≤3 gens lands in the a\* region: `param_a ∈ ~[0.52, 0.57]`, `eta`
  and `rho_vac` near feb, **late-slope ≈ 0**, gate-certified;
- the population mean `param_a` moves **toward ~0.55** across generations (convergence, not drift);
- **no** short-window (T<24000) individual is promoted; **no** prime-SSE steering;
- `css.classify` certifies the winner.

## Fail / STOP-and-classify triggers
- the top individual drifts **out** of the a\* box, or the population diverges;
- a short-window artifact is promoted, or the gate rejects the "winner";
- selection ranks a decayer/grower above a\*;
- the run needs prime-SSE to converge.
→ **stop, classify** the failure (search-operator vs objective vs plumbing) **before** any broader hunt.

## Guardrails (unchanged, non-negotiable)
No broad search; no Phase D claims; `objective="stability"`; `css.classify`/stability gate remains the certifier; no
solver / physics / kinetic-term / geometry / gate / IC change; success = **re-finding the known basin**, not new
physics. Runs locally in `.venv` (cupy 14.0.1). **Not executed by this plan.**
