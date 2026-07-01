# Negative result — legacy-derived candidate panel (corrected solver + A-field rescue)

**Scope (read first).** This closes the **legacy-candidate** investigation only. It is
**NOT** a global negative for IRER, not "stable collapse is impossible", and not a
model-class negative for the full corrected parameter space. A stronger model-level
negative would require a *fresh* corrected-physics sweep (see §6 / FRESH_HUNT).

> **Result, precisely stated:** *Legacy-derived candidates — the best old low-SSE and
> multiseed panels — do not produce mutual-support stable collapse under the corrected
> ETDRK4 SNCGL solver. Both reserved A-field rescue topologies (vacuum-reference and
> additive-potential) also fail to create mutual support on that panel. Falsified for the
> tested legacy-derived candidate panel and the tested A-field topologies.*

## 1. Instruments (validated)
- JAX scout FP64-equivalent to the CuPy backbone (rel_L2 ≈ 5.8e-16), re-verified after every refactor.
- `prime_log_sse` crash fixed (`_get_shell_map` in-place broadcast → silent 999/0 fallback). NOTE: the
  full prime-SSE pipeline still does not reproduce historical peak counts (k²-weighting + detect_peaks)
  and is treated as unreliable — prime-SSE is auxiliary, never headline.
- Stable-collapse observable (`stable_collapse.py`) + SDG/emergent-geometry diagnostic
  (`geometry_diag.py`, `IRER-SDG-DIAG-v1`, passive); both falsifiable (validated coherent-vs-noise).

## 2. Old low-SSE ledger = historical-only
The old low-SSE "evidence" was a compound artifact of (a) the old solver's instability/saturation
(max_amp 1e4–1e5, clamp≈1) and (b) the now-fixed scorer crash. Re-running the SSE<6 panel through
the corrected solver: structure does **not** reproduce. Old ledger gives **no validated leads**.

## 3. Corrected solver, γ_A = 0 (local geometry)
- Dissipative regime → smooth decay; broadband noise → incoherent speckle; gain regime → bounded
  growth into condensates; high gain → blow-up.
- Multiseed deep-dive (k=8, intact/ablation/isolated): **independent self-focusing condensates** —
  isolated node persists ≈ as well as the cluster, ablation merely removes one node, geometry
  decoupled/runaway. mutual_support 1/8 (a blow-up artifact), PROMOTE 0/8.

## 4. A-field rescue (γ_A > 0) — both reserved topologies
Paired micro-sweeps (3 candidates × γ_A∈{0,0.2,1,5} × {intact, isolated, ablation, phase_scrambled},
800 steps). Safety: γ_A=0 reproduces baseline to 4.1e-16; ρ_vac_eff floored; A-on contract-keyed;
production `unified_omega` untouched.

| topology | coupling | result |
|---|---|---|
| vacuum_reference | ρ_vac_eff = max(ρ_vac + γ_A·Ã, ε) | **PROMOTE 0/12** — independent condensates at strong bounded mod-depth (1.53); high γ_A → runaway |
| additive_potential | Ω²_eff = Ω²·exp(γ_A·Ã) | **PROMOTE 0/12** — same; mod-depth to 2.14, nodes unchanged; high γ_A → runaway |

Across all four falsification conditions: γ_A>0 still gives independent condensates; isolated nodes
survive as well as clusters; ablation does not disrupt the rest; geometry runaway / structure only via
instability. Notably `A_node_corr ≈ 0.6–0.73` — **A reaches and tracks the nodes, but modulating the
geometry by it creates no stabilizing inter-node dependency.**

## 5. No CuPy promotion from this branch
Nothing on the legacy panel cleared the promotion gate; A-on runs are not rank-compatible with γ_A=0
and were never merged into CuPy validation.

## 6. What is NOT closed
The **full current corrected-physics parameter space** has not been searched for mutual-support stable
collapse. The legacy panel is a tiny, biased (old-artifact-seeded) slice. A fresh corrected-physics
hunt — scored directly on stable-collapse / mutual-support, not prime-SSE, not seeded by old low-SSE —
is the next phase (see `FRESH_HUNT.md`). Only if that fresh sweep *also* yields only dissipation /
independent condensates / incoherent speckle / runaway should a broader corrected-model negative be written.
