# Phase C — feb parameter-basin: consolidated analysis

**Date:** 2026-06-26
**Status labels:** `GATE_V3_BREATHING_AWARE_CONFIRMED` · `FEB_PARAMETER_BASIN_SUPPORTED` ·
`FEB_NODE_FAMILY_LONG_TIME_CONFIRMED` · `PARAM_A_CRITICAL_GAIN_KNOB` ·
`ETA_RHO_VAC_SENSITIVE_BASIN_AXES` · `OAT_PARAM_BASIN_MAP_COMPLETE` · `JOINT_PARAM_BASIN_NOT_YET_MAPPED` ·
`BASIN_INTERIOR_SEED_ROBUST` · `BASIN_UPPER_EDGES_SEED_SENSITIVE`
**Geometry frozen at** `e8d6a78ea`. Analysis/replay only.

## Accepted interpretation

> The feb parameter regime supports a robust bound-state basin with multiple stable node-count
> morphologies. The OAT basin map indicates that stability is most sensitive to `param_a`, `param_eta`,
> and `param_rho_vac`, with `param_omega0` as a secondary check, while `param_D`, `param_a_coupling`,
> `param_s`, and `param_f` appear comparatively insensitive across the tested one-axis ranges.

> **Caution:** this is an OAT (one-axis-at-a-time) parameter-sensitivity map, **not yet a full coupled
> basin**. The joint basin may be smaller than the product of the 1-D windows.

---

## 1. Gate v3 — rationale and verification (`GATE_V3_BREATHING_AWARE_CONFIRMED`)

Detail: [PHASE_C_GATE_V3_BREATHING_BOUND_STATE.md](PHASE_C_GATE_V3_BREATHING_BOUND_STATE.md) (commit
`f6055c861`). v2's late-half drift gate over-rejected feb's **breathing downswing** at T=24000 (feb gained
mass, `er_min=er0`, bounded — yet v2 said SPIN_DOWN). v3 adds a bounded-breathing exception: when
`|late_drift|>0.15`, accept iff `er_max≤3`, `0.5≤er_fin≤2.5`, `floor_ratio=er_min/er0≥0.85` (anti-decay),
and `er_fin≤0.95·er_max` (anti-monotonic-grower). Verified on real traces: feb & K3 @T24000 → TRUE;
k6_high/k4 (decay) and k6_mid/k6_near (growers) still reject. 13/13 tests pass. Window-agnostic; preserves
all T=12000 verdicts.

## 2. Feb basin search results (`FEB_PARAMETER_BASIN_SUPPORTED`)

Detail: [PHASE_C_FEB_BASIN_RESULTS.md](PHASE_C_FEB_BASIN_RESULTS.md) (commit `b0d393889`). At feb's params,
all 8 IC variants (node count K∈{3,4,5,6,8} per-blob; total-mass-fixed at 0.5/1/2×) reached gate-validated
bound states — overturning the earlier per-node-mass-window and IC-norm hypotheses. **Bound-state
formation is a parameter-regime property**, robust to IC details; the final node count is set by IC blob
placement.

## 3. Seed robustness + T=24000 breathing confirmation (`FEB_NODE_FAMILY_LONG_TIME_CONFIRMED`)

Detail: confirmation §, same doc. 10/10 TRUE for 2 extra seeds × K{3,4,5,6,8} @T12000 (stability
seed-robust; node count seed-dependent). T=24000: K3–8 all TRUE under v3 (bounded breathing) — the
node-count family is **long-time stable**.

## 4. 52-config OAT parameter-basin battery (`OAT_PARAM_BASIN_MAP_COMPLETE`)

Detail: [PHASE_C_PARAM_BASIN_RESULTS.md](PHASE_C_PARAM_BASIN_RESULTS.md) (commit `ba1acb3f0`). 52/52 in
11.5 h. P1 re-confirmed the node family at T=24000 under v3. P2 mapped each of feb's 8 params one-at-a-time
(×0.5–1.5; omega0 set 0.05–0.4), K6/per-blob/T12000/v3.

## 5. Sensitivity ranking of parameters

| rank | parameter | role | 1-D stable window | label |
|---|---|---|---|---|
| 1 (tightest) | **param_a** | cubic gain | ≈ **±10%** (×0.9–1.1); below→spin-down, above→grower/blowup | `PARAM_A_CRITICAL_GAIN_KNOB` |
| 2 | **param_eta** | linear gain/loss | ≈[0.75×, 1.25×], two-sided | `ETA_RHO_VAC_SENSITIVE_BASIN_AXES` |
| 3 | **param_rho_vac** | conformal reference density | ≥0.75× (one-sided) | `ETA_RHO_VAC_SENSITIVE_BASIN_AXES` |
| 4 (secondary) | param_omega0 | vacuum oscillator (feb=0) | ≤≈0.1 (weak only) | secondary check |
| insensitive | param_D, param_a_coupling, param_s, param_f | diffusion / conformal exponent / quintic / septic | TRUE across ×0.5–1.5 | — |

The bound state is a **gain/loss balance on the emergent geometry**, pinned tightest by the cubic gain
(`param_a`).

## 6. What claims ARE supported

- The feb parameter regime supports a **robust, structured bound-state basin** (`FEB_PARAMETER_BASIN_SUPPORTED`).
- The node-count family (2–4 nodes from K3–8) is **long-time stable to T=24000** under the breathing-aware
  gate (`FEB_NODE_FAMILY_LONG_TIME_CONFIRMED`).
- v3 correctly separates bounded breathing from decay/growth (`GATE_V3_BREATHING_AWARE_CONFIRMED`).
- Stability is **most sensitive to `param_a`, then `param_eta`, `param_rho_vac`** (`param_omega0` secondary);
  four params are comparatively insensitive (`OAT_PARAM_BASIN_MAP_COMPLETE`, `PARAM_A_CRITICAL_GAIN_KNOB`,
  `ETA_RHO_VAC_SENSITIVE_BASIN_AXES`).

## 7. What claims are NOT supported

- **`JOINT_PARAM_BASIN_NOT_YET_MAPPED`** — only 1-D OAT slices exist; the coupled basin may be smaller than
  the product of the 1-D windows (sensitive params likely trade off).
- **No prime-harmonic structure.** Post-hoc `prime_log_sse` is 999 (no peaks) for all 60 states — **no
  support for the log-prime hypothesis** on this state family (see
  [PHASE_C_VALIDATION_STACK_AUDIT.md](PHASE_C_VALIDATION_STACK_AUDIT.md)).
- **No topological structure.** Post-hoc TDA/Betti is ~null for all states — **no support for any
  topological-invariant / phase-transition reading.**
- No matter / ground-state / molecule / black-hole claim. These are **smooth dissipative solitons** whose
  existence is an energy-balance property; the stability classifier is the only discriminating metric.
- Window/edge precision: OAT windows are coarse (6 factors), single-seed for P2.

## 8. Recommended next validation steps (before the joint grid)

1. **(done) Validation-stack audit + read-only post-hoc diagnostics** — confirmed prime-SSE/TDA are
   runnable but null for these states; the energy classifier is the gate.
2. **(done) Sensitive-axis edge seed-confirmation** (commit `425123267`,
   `sweep_runs/FEB_PARAM_EDGE_CONFIRM_20260626_124432/`): 7 edge cells × 2 new seeds (20260620/20260621),
   K6/T12000/v3. **10/14 match seed-619.** Finding (`BASIN_INTERIOR_SEED_ROBUST`,
   `BASIN_UPPER_EDGES_SEED_SENSITIVE`):
   - **Seed-robust:** the interior and *lower* boundaries — `param_a` ×0.75 (SPIN), ×0.9 (TRUE), ×1.1
     (TRUE); `param_eta` ×0.75 (TRUE); `param_rho_vac` ×0.75 (TRUE) all reproduce at both new seeds.
   - **Seed-sensitive (both mismatches consistent across both new seeds):** the *upper* edges —
     `param_a` ×1.25 was GROW at seed-619 but **TRUE** at 620/621 (basin wider, grower onset > ×1.25);
     `param_eta` ×1.25 was TRUE at seed-619 but **SPIN_DOWN** at 620/621 (basin narrower, upper edge < ×1.25).
   - **Read:** the bound-state basin's *core* is seed-robust, but the high-gain (`param_a`) and high-loss
     (`param_eta`) **upper boundaries move ~one grid step with the IC seed**. Boundary location is
     IC-dependent; basin existence is not.
3. Keep prime-SSE/TDA as **exploratory post-hoc diagnostics**, not promotion gates, unless explicitly
   calibrated.

---

## Drafted next experiment — joint `(param_a, param_eta, param_rho_vac)` grid (NOT launched)

Framed as **mapping the coupled basin boundary around feb**, not a hunt for proof.

- **Axes (the 3 sensitive params), centered on feb, ranges from the OAT windows:**
  - `param_a` ∈ {×0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15} (tight — straddle the ±10% edge)
  - `param_eta` ∈ {×0.7, 0.85, 1.0, 1.15, 1.3}
  - `param_rho_vac` ∈ {×0.7, 0.85, 1.0, 1.25, 1.5}
- Hold the four insensitive params + `param_omega0=0` at feb; K=6 / per-blob.
- **Validation:** N=96, **T=12000**, v3 gate. **Multi-seed at the edges is now required** — the rigor step
  showed the upper boundaries move ~one grid step with seed, so single-seed boundary cells are unreliable.
- **Revised two-stage plan (fits 10–12 h windows):**
  - **Stage 1 (breadth, single seed 20260619, ~9.5 h):** `param_a` {0.9,0.95,1.0,1.05,1.1} ×
    `param_eta` {0.8,1.0,1.2} × `param_rho_vac` {0.85,1.0,1.25} = 45 configs @ T12000, v3-gated, resumable +
    deadline-aware (reuse the `feb_param_basin.py` pattern). Note `param_a ≤ ×1.1` is seed-robust (safe);
    the `param_eta ×1.2` plane sits near the seed-sensitive upper edge — expect seed-dependent results there.
  - **Stage 2 (boundary seed-expansion):** for every Stage-1 cell on a TRUE↔reject transition, re-run at
    seeds 20260620 + 20260621 (and T=24000 for interior survivors). This characterizes the coupled boundary
    as a **seed-distribution**, not a single verdict — the rigor step proved that is necessary on the upper
    edges.
- **Output:** per-config class + drift + bounded_breathing + n_fin + held_mass → a 3-D basin volume
  (sliceable), to see whether the sensitive axes trade off (coupled boundary) vs. act independently.
- **Reads:** is the coupled basin appreciably smaller than the OAT product? Is `param_a` still the tightest
  axis jointly? Does any (a, eta, rho_vac) combo widen the basin?

**Do not launch until approved.** Build it as a guarded, resumable script (like `feb_param_basin.py`) and
queue overnight (daytime WSL teardowns persist).

No charge / topology / proof / log-prime-proof / matter / ground-state / molecule / black-hole language.
