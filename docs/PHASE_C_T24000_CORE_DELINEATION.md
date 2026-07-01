# Phase C — T=24000 core delineation (eta×1.0 plane)

**Date:** 2026-06-27
**Run:** `sweep_runs/FEB_CORE_DELINEATION_T24000_20260627_175050/` (classifier v3, N=96, K=6, per-blob,
seed 20260619, **T=24000**, `param_eta` at feb ×1.0; `param_a` × `param_rho_vac` grid). 15/15 complete
(ran across several sessions; resumable). Map: `…/core_delineation_map.png`. Analysis/replay only; geometry
frozen at `e8d6a78ea`. Resolves Claim E of [PHASE_C_DOSSIER.md](PHASE_C_DOSSIER.md).

> **Result:** the long-time-stable **core is 12/15 of the eta×1.0 plane** — a clean diagonal band. Only the
> **low-drive corner** (low gain + low/mid reference density) decays by T=24000. `er_fin` rises
> monotonically with both `param_a` and `param_rho_vac`, mapping the gain/loss balance directly;
> feb-center sits comfortably interior.

## The plane (CORE = TRUE@T24000; shell = decays; `er_fin` in parens)

```
a \ rho_vac    r×0.85         r×1.0          r×1.25
a×0.9     shell-SPIN(0.52)  shell-SPIN(0.74)  CORE(1.06)
a×0.95    shell-SPIN(0.69)  CORE(0.96)        CORE(1.34)
a×1.0     CORE(0.88)        CORE(1.20)←feb    CORE(1.64)
a×1.05    CORE(1.10)        CORE(1.46)        CORE(1.96)
a×1.1     CORE(1.33)        CORE(1.76)        CORE(2.31)
```

12 CORE · 3 shell (all SPIN_DOWN). The shell is the low-drive corner: `(a×0.9, rho×0.85)`,
`(a×0.9, rho×1.0)`, `(a×0.95, rho×0.85)`.

## Interpretation

- **The core boundary is a diagonal gain/loss-balance contour.** `er_fin` increases smoothly with both gain
  (`param_a`) and reference density (`param_rho_vac`): bottom-left `er_fin 0.52` (decays) → top-right
  `er_fin 2.31` (near grower edge). Long-time survival requires sufficient **combined drive** to hold the
  energy above the spin-down floor:
  - at low gain (`a×0.9`) you need high rho (`×1.25`);
  - at `a×0.95` you need rho ≥ ×1.0;
  - at `a ≥ ×1.0` the state survives across the whole tested rho range (×0.85–1.25).
- **feb-center (`a×1.0, rho×1.0`, er_fin 1.20) is comfortably interior** — not near either edge.
- **Upper-drive caution:** the high-drive corner (`a×1.1, rho×1.25`, er_fin 2.31) is still CORE but sits high
  in-band, approaching the grower threshold — the core also has an (untested here) upper-drive limit.
- This is the same window-artifact lesson made precise: the T=12000 eta×1.0 plane was 15/15 TRUE; at T=24000
  three low-drive cells reveal as slow decayers. **Quote the basin at this T24000 core**, not the T12000
  extent.

## Claim E — resolved

> The T=24000-validated core is the diagonal band `a ≥ ×1.0` (any tested rho) ∪ `a×0.95, rho ≥ ×1.0` ∪
> `a×0.9, rho×1.25` of the eta×1.0 plane (12/15). The low-drive corner (low gain + low rho) is a
> T=12000-window-marginal shell that decays by T=24000. feb-center is interior. **Falsifier** (feb-center
> fails at T24000) did not trigger — feb-center is CORE (er_fin 1.20, confirmed). Remaining: the eta-width
> of the core (only eta×1.0 mapped here) and the upper-drive limit — optional.

## Object definition for downstream tests

The matter-likeness / kick tests must use the **interior of this core** — feb-center (`a×1.0, e×1.0, r×1.0`)
and `a×1.05, e×1.0, r×1.0` (er_fin 1.46, drift −0.11, breathing) are the safest well-interior cells, far
from both the decay shell and the grower edge.

No charge / topology-proof / log-prime-proof / matter / ground-state / molecule / black-hole language.
