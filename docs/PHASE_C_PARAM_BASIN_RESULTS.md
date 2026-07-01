# Phase C — parameter-basin battery: results

**Date:** 2026-06-26
**Run:** `sweep_runs/FEB_PARAM_BASIN_20260626_004039/` (classifier **v3**, N=96, seed 20260619). 52/52
configs completed in 11.49 h, no interruption. Plan: [PHASE_C_PARAM_BASIN_BATTERY_PLAN.md](PHASE_C_PARAM_BASIN_BATTERY_PLAN.md).
Map figure: `…/basin_map/param_basin_map.png`. Analysis/replay only; geometry frozen at `e8d6a78ea`.

> **Headline:** feb's bound-state basin is real and **structured**. The node-count family is long-time
> stable to T=24000 under the breathing-aware v3 gate. In parameter space the basin is **wide in 4 of 8
> params and narrow in the gain/balance directions** — `param_a` (cubic gain) is the critical knob
> (only ±10% tolerance), with `param_eta`, `param_rho_vac`, and `param_omega0` the other sensitive ones.

## P1 — node-family long-T revalidation (T=24000, v3) — CONFIRMED

| K | class | n_fin | drift | breathing | er_fin |
|---|---|---|---|---|---|
| 3 | TRUE_SATURATED | 2 | −0.200 | True | 1.10 |
| 4 | TRUE_SATURATED | 2 | −0.165 | True | 1.23 |
| 5 | TRUE_SATURATED | 3 | −0.171 | True | 1.22 |
| 6 | TRUE_SATURATED | 4 | −0.177 | True | 1.20 |
| 8 | TRUE_SATURATED | 4 | −0.155 | True | 1.30 |

All five are long-time stable at T=24000, correctly accepted as bounded breathers by v3 (each would have
been false-rejected by v2). This **closes the loop**: the discovered node-count family is a genuine
long-time bound-state family, and the v3 gate handles it.

## P2 — parameter one-at-a-time basin map (T=12000)

`center_feb` = TRUE (reference). Per-parameter result across ×{0.5, 0.75, 0.9, 1.1, 1.25, 1.5}:

| param | feb | ×0.5 | ×0.75 | ×0.9 | ×1.1 | ×1.25 | ×1.5 | stable window |
|---|---|---|---|---|---|---|---|---|
| param_D | 2.7329 | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | **all** (insensitive) |
| param_a_coupling | 2.3098 | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | **all** (insensitive) |
| param_s | 0.0129 | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | **all** (insensitive) |
| param_f | −0.4861 | TRUE | TRUE | TRUE | TRUE | TRUE | TRUE | **all** (insensitive) |
| param_rho_vac | 1.1866 | SPIN | TRUE | TRUE | TRUE | TRUE | TRUE | ≥0.75× (one-sided) |
| param_eta | 0.0704 | BLOW | TRUE | TRUE | TRUE | TRUE | SPIN | ≈[0.75×, 1.25×] (two-sided) |
| **param_a** | 0.4802 | SPIN | SPIN | TRUE | TRUE | GROW | BLOW | **≈[0.9×, 1.1×] — narrowest** |
| param_omega0 | 0.0 | — | — | set0.05: TRUE · set0.1: TRUE · set0.2: GROW · set0.4: BLOW | | | | ≤≈0.1 (weak only) |

### Structure of the basin

- **Insensitive directions (basin wide): `param_D`, `param_a_coupling`, `param_s`, `param_f`.** Diffusion,
  the conformal exponent, and the quintic/septic nonlinearities can move ±50% and the bound state persists.
  (s and f are small at feb, so this partly reflects their minor role.)
- **Sensitive directions (clear stability windows):**
  - **`param_a` (cubic gain) — the critical knob.** Stable only ≈±10%: below → spin-down (insufficient
    gain → decay), above → grower → blowup (excess gain → runaway).
  - **`param_eta` (linear gain/loss)** — two-sided ≈[0.75×,1.25×]: too low → blowup, too high → spin-down.
  - **`param_rho_vac` (conformal reference density)** — one-sided, needs ≥0.75×; lowering it → spin-down.
  - **`param_omega0` (vacuum oscillator, off at feb)** — tolerates a weak oscillator (≤≈0.1) but a stronger
    one drives growth → blowup.

### Disciplined reading

The bound state is a **gain/loss balance** on the emergent geometry: its existence is governed mainly by
the gain terms (`param_a` cubic, `param_eta` linear) against the conformal reference `param_rho_vac`, and is
nearly indifferent to diffusion, the conformal exponent, and the higher nonlinearities. `param_a` is the
tightest constraint (±10%). This is consistent with the project's dissipative-soliton reading — a fine
energy balance, here pinned by the cubic gain. No charge / topology / proof / ground-state / black-hole /
universal-law claim is made.

## Caveats

- **OAT (one-at-a-time).** These are 1-D slices through feb; the *joint* basin is generally smaller than
  the product of the 1-D windows (sensitive params likely couple, e.g. a–eta–rho_vac trade off). A joint
  map of the sensitive trio is the natural follow-up.
- **Single seed (20260619), T=12000 for P2** (P1 confirms the family to T=24000). Window boundaries are
  coarse (6 factors); the true edges lie between sampled points.
- Promotion to "bound state" used the v3 gate throughout.

## Implication for future search

The effective search space collapses from 8 parameters to **~3 sensitive dimensions**
(`param_a`, `param_eta`, `param_rho_vac`; optionally `param_omega0`), holding the four insensitive params
near feb. A **joint (a, eta, rho_vac) grid** at finer resolution around feb — v3-gated, T=12000 with T=24000
confirmation of survivors — would map the actual bound-state region efficiently. This is the principled
successor; it is **not** a broad random hunt.
