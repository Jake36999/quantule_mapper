# Phase D.4 (first pass) — Node-Node Coupling on the Phase C Library

**Classification: `COUPLING_SHORT_RANGE_DENSITY_GEOMETRIC`** — stable nodes couple through a *short-range
density/geometry channel* (not current, not distance-dependent phase). Read-only static analysis
(`jax_scout/node_coupling.py`) over **1202 node-pairs (881 stable, 321 failing)** from the harvested library. No
solver/gate/physics change; no active source term.

## Method
For every node pair (all stable configs) compute NON-current coupling proxies (the D.3 finding is that the current
`J=ρ∇φ` is ≈0 in stable nodes) and test SPACING-dependence + a **density-only null** (partial correlation controlling
for node density `rho_ref`):
- `conductance` = density-bridge (min inter-node ρ / node ρ) — connectivity;
- `phase_diff` = |Δφ| between nodes — phase coupling;
- `dstress_axial` = density-strain stress `∂√ρ ∂√ρ` projected on the node-pair axis — axial tension;
- `divT_axial` = `(∇·T_dens)·û` on the segment — effective directional force-density.

## Results
**Stable node-pairs: proxy vs spacing (raw | partial controlling for density):**
| proxy | raw r | partial r (|ρ) | reading |
|---|---|---|---|
| conductance | **−0.327** | **−0.326** | short-range; **survives density-null** |
| dstress_axial | **−0.376** | **−0.379** | short-range tension; survives density-null |
| phase_diff | −0.000 | −0.001 | **no** distance dependence |
| divT_axial | +0.034 | +0.034 | large only at short range, but no clean law |

**Binned by spacing quartile (stable):** conductance = **0.022** in the tightest bin `[0.21,0.495]` → **0.000** in all
wider bins. i.e. a density bridge exists **only** for pairs closer than ~0.5 box; beyond that, isolated. Same
short-range pattern for `divT_axial` (−0.37 tightest → ~0 wider).

**Stable vs failing:** phase_diff **0.0009 (stable) vs 0.016 (failing)** and conductance 0.006 vs 0.022 — failing
configs are more phase-dispersed and more bridged; **phase coherence tracks stability.**

## Interpretation (honest, static)
1. **There IS a node-node coupling signal, and it is SHORT-RANGE and density/geometry-mediated.** Density-bridge
   conductance and axial density-strain stress both fall with spacing and **survive the density-only null**, defining
   an effective **coupling radius ≈ 0.5 box** beyond which stable nodes are effectively isolated. This is a real
   coarse-graining variable — not a density proxy, not the current-routing null.
2. **Phase is a stability signature, not a coupling channel.** Stable nodes are *globally* phase-locked (Δφ≈0 at all
   spacings); the phase-difference carries no distance information but cleanly separates stable (coherent) from
   failing (dispersed). (Consistent with the earlier "all nodes phase-locked" observation.)
3. **The current channel is null** (D.3) — reaffirmed; coupling is not current-mediated.
4. **No clean force-law yet.** `divT_axial` (effective force) is short-range in magnitude but not monotone in
   spacing — a *static* snapshot cannot resolve a dynamical interaction force. A measured node-node force law needs
   **two-node time-evolution** (D.5 territory), which this analysis motivates but does not perform.

## What this gives the roadmap (D.6 zoom-out)
Concrete first coarse-graining ingredients for a reduced node model:
- **coupling radius** ≈ 0.5 box (connectivity threshold);
- **within-radius** coupling = density-bridge conductance + axial density-strain stress (both short-range);
- **phase** = a global coherence / stability class (not a pairwise distance law);
- **current/vorticity** = negligible (drop from the node descriptor for stable structures).

## Honesty / scope
Static single-snapshot analysis on saved final fields; correlations are modest (|r|~0.3–0.4) but survive the
density-null and show a clear short-range structure. This is a **coupling *descriptor*** result, not a measured
dynamical interaction law — that is the next step (D.5 two-node dynamics) and is **not** claimed here. Read-only; no
matter-like claims; the stress tensor remains a diagnostic, never an active source term on this evidence.
